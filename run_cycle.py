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
from notify import alert, write_heartbeat, daily_summary, is_configured as telegram_configured

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

    # 1b. Snapshot que o paper trader efetivamente carregaria (P0-2: seleção
    #     por nome escolhia markets_incremental_* velho em vez do mais recente)
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        from execution.paper_trader import load_current_markets, RAW_MKT_DIR

        selected = sorted(
            list(RAW_MKT_DIR.glob("markets_all_*.parquet")) +
            list(RAW_MKT_DIR.glob("markets_incremental_*.parquet")),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not selected:
            problems.append("load_current_markets(): nenhum candidato de snapshot encontrado")
        else:
            newest = selected[0]
            age_min = (now_ts - newest.stat().st_mtime) / 60
            current = load_current_markets()
            if current.empty:
                problems.append(
                    f"load_current_markets() vazio — {newest.name} tem {age_min:.0f}min "
                    "(stale ou sem linhas)"
                )
            else:
                logger.debug(
                    f"snapshot efetivo do paper trader: {newest.name} "
                    f"({age_min:.0f}min, {len(current)} mercados)"
                )
    except Exception as e:
        problems.append(f"falha ao checar snapshot do paper trader: {e}")

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

    # P0-7: sem isso, TODO o trabalho de check_degradation/alert() deste
    # arquivo não produz nenhuma notificação — silenciosamente, desde sempre
    # (era logger.debug em notify.py). Aviso ruidoso, mas não aborta o ciclo:
    # o token ausente é responsabilidade do operador (BotFather), não um bug
    # de código, e um cycle sem trading é pior que um cycle sem alerta.
    if not telegram_configured():
        msg = (
            "TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID não configurados — "
            "NENHUM alerta será enviado neste ciclo (ver .env.example)"
        )
        print(f"🚨 {msg}", file=sys.stderr)
        logger.warning(msg)

    uv_bin = shutil.which("uv") or "uv"
    uv = [uv_bin, "run", "python"]
    failures: list[str] = []
    degradations: list[str] = []

    # P0-8: heartbeat sempre escreve, mesmo se algo abaixo lançar uma exceção
    # não tratada — o modo de falha que o heartbeat existe para detectar é
    # exatamente o que impedia sua escrita (heartbeat só rodava perto do fim).
    try:
        # ── 1. Atualiza mercados (sempre) ─────────────────────
        ok = run(uv + ["pipeline/fetch_markets.py"], "fetch_markets")
        if not ok:
            failures.append("fetch_markets")
            # P0-6 já fechou o buraco na origem: save_snapshot() não grava mais
            # snapshot vazio sob o nome que os loaders veem (EmptySnapshotError),
            # e o gate de frescor do P0-2 já rejeita dado velho-mas-válido. Não
            # há mais um "modo perigoso" para os passos abaixo evitarem — só
            # registra a falha alto o bastante pra virar alerta no resumo.
            logger.error("fetch_markets falhou — seguindo com o último snapshot válido (se ainda fresco)")

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
        # Seguro mesmo com fetch_markets falho: o gate de frescor (P0-2) devolve
        # DataFrame vazio se o snapshot mais recente passou de MAX_MARKET_SNAPSHOT_AGE_MIN.
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
    finally:
        write_heartbeat(failures + [f"health: {m}" for m in degradations])

    # ── Resumo diário (P0-8: daily_summary nunca era chamado) ────
    # Só LÊ o portfólio existente — nunca cria um (get_or_create_portfolio
    # criaria uma linha nova como efeito colateral de um resumo, o que não
    # é papel dele). Sem portfólio ainda, não há o que resumir.
    # daily_summary() já se auto-limita a 1x/dia (hora < 1h UTC).
    try:
        sys.path.insert(0, str(Path(__file__).parent / "risk"))
        sys.path.insert(0, str(Path(__file__).parent / "execution"))
        import sqlite3 as _sqlite3
        from risk_manager import portfolio_risk_summary
        from paper_trader import DB_PATH, get_open_positions

        _conn = _sqlite3.connect(DB_PATH)
        _row  = _conn.execute(
            "SELECT id, initial_capital, current_cash FROM portfolio ORDER BY id DESC LIMIT 1"
        ).fetchone()
        _conn.close()
        if _row is not None:
            portfolio = {"id": _row[0], "initial_capital": _row[1], "current_cash": _row[2]}
            open_pos  = get_open_positions()
            summary   = portfolio_risk_summary(portfolio, open_pos, DB_PATH)
            initial   = summary["initial_capital"]
            pnl       = summary["realized_pnl"]
            daily_summary(
                pnl_usdc=pnl,
                pnl_pct=(pnl / initial * 100) if initial else 0.0,
                open_positions=len(open_pos),
                cash=summary["current_cash"],
            )
    except Exception as e:
        logger.debug(f"daily_summary: falha ao montar resumo — {e}")

    # ── Resumo ────────────────────────────────────────────
    end = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
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
