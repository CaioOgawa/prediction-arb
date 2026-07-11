"""
run_cycle.py
Ciclo completo do sistema Polymarket Quant.

Dois modos de execução:

  --light  (padrão, a cada 30min via launchd)
    1. Atualiza mercados (Gamma API)
    2. Gera sinais deribit frescos (sem quota)
    3. Paper trader: resolve posições + abre novas
    4. Backtest: atualiza métricas

  --full   (a cada 4h via launchd)
    Igual ao light + gera sinais odds (consome The Odds API quota: ~500 req/mês)

Quota estimada com esse esquema:
    odds: 6 vezes/dia × 30 dias = 180 req/mês  (free tier: 500)

Uso manual:
    uv run python run_cycle.py              # light
    uv run python run_cycle.py --full       # com odds
    uv run python run_cycle.py --dry-run    # não salva posições
"""

import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import click
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent))
from notify import alert, write_heartbeat

LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)


def check_degradation(full: bool) -> list[str]:
    """
    Verifica sinais de degradação silenciosa olhando os ARTEFATOS do ciclo
    (parquets, CSVs, banco), não os exit codes dos subprocessos.

    Retorna lista de mensagens de alerta (vazia = saudável).
    """
    import sqlite3

    problems: list[str] = []
    now_ts = datetime.now(timezone.utc).timestamp()

    # 1. Universo de mercados anormalmente pequeno (bug de paginação de 2026-05)
    try:
        import pandas as pd
        parquets = sorted(Path("data/raw/markets").glob("markets_all_*.parquet"))
        if parquets:
            n_markets = len(pd.read_parquet(parquets[-1], columns=["conditionId"]))
            if n_markets < 300:
                problems.append(f"universo com só {n_markets} mercados (esperado >= 300)")
        else:
            problems.append("nenhum parquet de mercados encontrado")
    except Exception as e:
        problems.append(f"falha ao checar universo: {e}")

    # 2. Sinais deribit sem atualização há mais de 24h (gerados a cada ciclo light)
    try:
        sigs = sorted(Path("outputs/reports").glob("signals_deribit_*.csv"),
                      key=lambda p: p.stat().st_mtime)
        if not sigs:
            problems.append("nenhum CSV de sinais deribit já gerado")
        elif (now_ts - sigs[-1].stat().st_mtime) > 24 * 3600:
            age_h = (now_ts - sigs[-1].stat().st_mtime) / 3600
            problems.append(f"sinais deribit sem atualização há {age_h:.0f}h")
    except Exception as e:
        problems.append(f"falha ao checar sinais: {e}")

    # 3. Exits no floor 0.001 nas últimas 24h — assinatura do bug de book vazio
    #    (incidente 2026-05: 51 posições executadas a 0.001)
    try:
        conn = sqlite3.connect("data/db/paper_trading.db")
        n_floor = conn.execute("""
            SELECT COUNT(*) FROM positions
            WHERE status='closed' AND exit_price <= 0.0015
              AND closed_at >= datetime('now', '-1 day')
        """).fetchone()[0]
        conn.close()
        if n_floor >= 3:
            problems.append(f"{n_floor} exits a preço-floor (0.001) em 24h — book/feed suspeito")
    except Exception:
        pass  # banco pode não existir ainda

    return problems


def run(cmd: list[str], label: str) -> bool:
    """Executa um subcomando e retorna True se bem-sucedido."""
    start = datetime.now(timezone.utc)
    logger.info(f"[{label}] iniciando...")

    result = subprocess.run(
        cmd,
        capture_output=False,   # deixa o output aparecer no log
        text=True,
    )

    elapsed = (datetime.now(timezone.utc) - start).total_seconds()
    if result.returncode == 0:
        logger.success(f"[{label}] OK ({elapsed:.1f}s)")
        return True
    else:
        logger.error(f"[{label}] FALHOU com código {result.returncode} ({elapsed:.1f}s)")
        return False


@click.command()
@click.option("--full",        is_flag=True, default=False,
              help="Ciclo completo: inclui sinais odds (consome API quota). Default: light (só deribit).")
@click.option("--dry-run",     is_flag=True, default=False,
              help="Passa --dry-run para o paper trader (não salva posições).")
@click.option("--html-report", is_flag=True, default=False,
              help="Gera relatório HTML do backtest ao final.")
def main(full: bool, dry_run: bool, html_report: bool) -> None:
    """Pipeline: fetch → sinais → paper trading → backtest.
    Light (default): deribit + resolve + backtest.
    Full (--full):   + odds fresh (usa quota da API).
    """
    mode_label = "FULL" if full else "LIGHT"
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    logger.info(f"=== Ciclo {mode_label} iniciado: {now} ===")

    uv_bin = shutil.which("uv") or "uv"
    uv = [uv_bin, "run", "python"]
    failures = []

    # ── 1. Atualiza mercados (sempre) ─────────────────────
    ok = run(uv + ["pipeline/fetch_markets.py"], "fetch_markets")
    if not ok:
        failures.append("fetch_markets")
        logger.warning("Continuando sem atualização de mercados...")

    # ── 2. Sinais odds — só no ciclo full ─────────────────
    if full:
        ok = run(
            uv + ["signals/run_signals.py", "--mode", "odds", "--fetch-fresh"],
            "signals_odds",
        )
        if not ok:
            failures.append("signals_odds")

    # ── 3. Sinais deribit — sempre (sem quota) ────────────
    # --min-hours-left 8: mercados com >= 8h até expiração
    # Evita gamma ruidoso das últimas horas; MIN_HOLD_HOURS=2 protege contra spikes.
    ok = run(
        uv + ["signals/run_signals.py", "--mode", "deribit", "--fetch-fresh",
              "--min-hours-left", "8"],
        "signals_deribit",
    )
    if not ok:
        failures.append("signals_deribit")

    # ── 3.5 Arb estrutural — scanner intra-Polymarket ────
    # CSV + alerta Telegram (garantidas); o paper trader (etapa 4) executa os
    # baskets garantidos atomicamente via open_basket().
    # NegRisk limitado a 40 eventos/ciclo (~40 requests à Gamma, sem quota).
    ok = run(uv + ["signals/structural_arb.py"], "structural_arb")
    if not ok:
        failures.append("structural_arb")

    # ── 4. Paper trader ───────────────────────────────────
    # Sempre modo "all": combina odds + deribit quando disponíveis.
    # Em ciclos light, o odds CSV estará stale (> MAX_SIGNAL_AGE_MINUTES) e
    # será ignorado automaticamente pelo paper_trader — sem quota desperdiçada.
    trader_cmd  = uv + ["execution/run_paper_trader.py", "--mode", "all"]
    if dry_run:
        trader_cmd.append("--dry-run")
    ok = run(trader_cmd, "paper_trader")
    if not ok:
        failures.append("paper_trader")

    # ── 5. Snapshot de preços das posições abertas ───────
    ok = run(uv + ["pipeline/price_snapshot.py"], "price_snapshot")
    if not ok:
        failures.append("price_snapshot")

    # ── 6. Backtest / tracker de performance ─────────────
    bt_cmd = uv + ["backtest/run_backtest.py"]
    if html_report:
        bt_cmd.append("--html")
    ok = run(bt_cmd, "backtest")
    if not ok:
        failures.append("backtest")

    # ── 7. Retenção do price_history (barato; VACUUM só manual) ──
    ok = run(uv + ["pipeline/db_maintenance.py", "--days", "14"], "db_maintenance")
    if not ok:
        failures.append("db_maintenance")

    # ── 8. Health checks — degradação silenciosa ─────────
    # O incidente 2026-05→07 (universo em 100 mercados, 0 sinais por 2 meses,
    # exits no floor 0.001) passou despercebido porque só falha de subprocesso
    # gerava alerta. Estes checks olham o RESULTADO, não o exit code.
    degradations = check_degradation(full=full)
    for msg in degradations:
        logger.warning(f"[health] {msg}")

    # ── Resumo ────────────────────────────────────────────
    end = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    write_heartbeat(failures + [f"health: {m}" for m in degradations])
    if degradations:
        alert("Degradação detectada:\n• " + "\n• ".join(degradations), cycle=mode_label)
    if failures:
        logger.warning(f"=== Ciclo {mode_label} concluído com erros: {failures} | {end} ===")
        alert(f"Etapas com falha: {', '.join(failures)}", cycle=mode_label)
        sys.exit(1)
    else:
        logger.success(f"=== Ciclo {mode_label} concluído com sucesso | {end} ===")


if __name__ == "__main__":
    main()
