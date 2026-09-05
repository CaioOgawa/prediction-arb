"""
run_execution.py
Execution daemon — roda a cada 5 minutos via launchd/cron.

Responsabilidade separada do run_cycle.py (signal generation, 30min):
  1. Resolve posições fechadas / expiradas
  2. Verifica early exits (profit target, edge flip)
  3. Verifica stop loss por drawdown (bloqueia rebalance/abertura em halt)
  4. Abre novas posições com sinais disponíveis (CSV mais recente)
  5. Mark-to-market das posições abertas

Por que separar?
  O ciclo de geração de sinais demora 2-5 minutos (chamadas Deribit, Gamma API).
  Durante esse tempo, posições abertas não são monitoradas. Um profit target de 1.5×
  pode ser atingido e perdido entre dois ciclos de 30min.

  Com o execution daemon rodando a cada 5min:
  - Early exits são capturados em < 5min após o preço cruzar o target
  - Novas posições são abertas assim que sinais frescos chegam (gerados pelo run_cycle)
  - O run_cycle continua focado em geração de sinais e não precisa rodar mais rápido

Configuração launchd (~/Library/LaunchAgents/com.polymarket.execution.plist):
  <key>StartInterval</key><integer>300</integer>   <!-- 5 minutos -->

Uso manual:
  uv run python run_execution.py
  uv run python run_execution.py --dry-run
"""

import atexit
import fcntl
import sys
from datetime import datetime, timezone
from pathlib import Path

import click
import pandas as pd
from loguru import logger

LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)

logger.add(
    LOGS_DIR / "execution_{time:YYYY-MM-DD}.log",
    rotation="1 day",
    retention="7 days",
    level="INFO",
    format="{time:HH:mm:ss} | {level} | {message}",
)

sys.path.insert(0, str(Path(__file__).parent / "execution"))
sys.path.insert(0, str(Path(__file__).parent / "risk"))
sys.path.insert(0, str(Path(__file__).parent))

# R2: run_cycle.py já tinha esse lock (P1-20) — este daemon roda 6x mais
# rápido (StartInterval de 5min) e não tinha proteção nenhuma contra
# execuções sobrepostas. Uma chamada de rede pendurada (open_position,
# load_current_markets) empilharia o próximo tick mutando as mesmas
# posições ao mesmo tempo. LOCK_NB: se já tem outro rodando, pula este
# ciclo em vez de esperar — o próximo tick de 5min já resolve.
LOCK_PATH = Path("data/run_execution.lock")


@click.command()
@click.option("--dry-run", is_flag=True, default=False,
              help="Não salva posições, só simula.")
@click.option("--mode", default="all",
              type=click.Choice(["odds", "deribit", "all"]),
              help="Fonte de sinais a considerar (default: all).")
def main(dry_run: bool, mode: str) -> None:
    """
    Execution loop: resolve, early exits, abre posições, mark-to-market.
    Projetado para rodar a cada 5min enquanto run_cycle.py roda a cada 30min.
    """
    from paper_trader import (
        DB_PATH,
        init_db,
        get_or_create_portfolio,
        get_open_positions,
        get_traded_condition_ids,
        load_current_markets,
        load_signals,
        mark_to_market,
        open_position,
        rebalance_positions,
        print_portfolio,
    )
    from risk_manager import (
        resolve_positions, early_exit_positions, reclassify_orphan_arb_legs,
        check_drawdown_stop, MAX_OPEN_POSITIONS, MIN_SIGNAL_LIQUIDITY, MAX_SIGNALS_PER_CYCLE,
    )

    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    lock_fh = open(LOCK_PATH, "w")
    try:
        fcntl.flock(lock_fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        logger.warning(f"Outro run_execution.py já em execução (lock {LOCK_PATH}) — ciclo pulado")
        lock_fh.close()
        return
    atexit.register(lambda: (fcntl.flock(lock_fh, fcntl.LOCK_UN), lock_fh.close()))

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    logger.info(f"=== Execution daemon iniciado: {now} ===")

    init_db()
    portfolio     = get_or_create_portfolio()
    current_mkts  = load_current_markets()
    open_pos      = get_open_positions()

    # ── 1. Resolve posições fechadas ─────────────────────────
    if not open_pos.empty and not current_mkts.empty:
        resolved = resolve_positions(
            open_positions=open_pos,
            current_markets=current_mkts,
            db_path=DB_PATH,
            dry_run=dry_run,
        )
        if resolved:
            logger.info(f"Resolvidas: {len(resolved)} posições")
            for r in resolved:
                sign = "+" if r["pnl_usdc"] >= 0 else ""
                logger.info(
                    f"  [{r['status'].upper()}] {r['question'][:50]} "
                    f"→ {sign}${r['pnl_usdc']:.2f}"
                )
        # Recarrega após resolução
        open_pos  = get_open_positions()
        portfolio = get_or_create_portfolio()

    # ── 1b. Pernas de arb órfãs (P0-5) ───────────────────────
    orphans = reclassify_orphan_arb_legs(DB_PATH, dry_run=dry_run)
    if orphans:
        groups = sorted({o["arb_group"] for o in orphans})
        logger.warning(
            f"Pernas de arb órfãs reclassificadas para 'value': {len(orphans)} "
            f"({', '.join(groups)})"
        )
        try:
            from notify import alert
            alert(
                f"{len(orphans)} perna(s) de arb órfã(s) reclassificada(s) para 'value' "
                f"(basket parcialmente resolvido): {', '.join(groups)}",
                cycle="run_execution",
            )
        except Exception:
            logger.exception("Falha ao notificar reclassificação de pernas órfãs")
        open_pos  = get_open_positions()
        portfolio = get_or_create_portfolio()

    # ── 2. Early exits ───────────────────────────────────────
    if not open_pos.empty and not current_mkts.empty:
        exits = early_exit_positions(
            open_positions=open_pos,
            current_markets=current_mkts,
            db_path=DB_PATH,
            dry_run=dry_run,
        )
        if exits:
            logger.info(f"Early exits: {len(exits)} posições fechadas")
            for e in exits:
                sign = "+" if e["pnl_usdc"] >= 0 else ""
                logger.info(
                    f"  [EARLY_EXIT/{e.get('trigger','?')}] {e['question'][:50]} "
                    f"→ {sign}${e['pnl_usdc']:.2f}"
                )
        # Recarrega após exits
        open_pos  = get_open_positions()
        portfolio = get_or_create_portfolio()

    # ── 3. Verifica stop loss (valor total, antes de rebalance/aberturas) ──
    # P1-13/P1-12: este daemon roda 6x mais rápido que o run_paper_trading
    # (5min vs 30min) e negociava sem NENHUMA das guardas de risco de lá —
    # nem drawdown, nem piso de liquidez, nem budget de slots, nem cap de
    # sinais por ciclo. O stop de drawdown era decorativo: o processo que
    # mais negocia era o único que o ignorava por completo.
    open_pos = get_open_positions()
    mtm_for_stop = mark_to_market(open_pos, current_mkts) if not open_pos.empty and not current_mkts.empty else open_pos
    pos_value_for_stop = float(mtm_for_stop["current_value"].sum()) if "current_value" in mtm_for_stop.columns else float(open_pos.get("cost_usdc", pd.Series([0])).sum()) if not open_pos.empty else 0.0
    total_value_for_stop = float(portfolio["current_cash"]) + pos_value_for_stop

    stop, reason = check_drawdown_stop(portfolio, DB_PATH, total_value=total_value_for_stop, record=True)
    if stop:
        logger.warning(f"STOP LOSS ATIVADO: {reason} — rebalance e abertura de posição suspensos")
        print_portfolio(portfolio, mtm_for_stop if not mtm_for_stop.empty else open_pos)
        end = datetime.now(timezone.utc).strftime("%H:%M UTC")
        logger.info(f"=== Execution daemon concluído: {end} | halt de drawdown ===")
        return

    # ── 3b. Rebalanceamento de posições existentes ──────────
    open_pos = get_open_positions()
    if not open_pos.empty and not current_mkts.empty:
        open_pos_mtm = mark_to_market(open_pos, current_mkts)
        rebalanced = rebalance_positions(open_pos_mtm, portfolio, dry_run=dry_run)
        if rebalanced:
            logger.info(f"Rebalanceamentos: {len(rebalanced)}")
            for r in rebalanced:
                logger.info(f"  +${r['add_usdc']:.2f} {r['question'][:50]} edge={r['new_edge']:.1%}")
            portfolio = get_or_create_portfolio()

    # ── 4. Abre novas posições ───────────────────────────────
    # Mesmos filtros de run_paper_trading (liquidez, budget de slots, cap de
    # sinais por ciclo) — antes deste fix, este loop iterava TODOS os sinais
    # sem piso de liquidez nenhum, incluindo a população ilíquida que
    # dispara o exit no piso de 0.001 (P1-14).
    signals = load_signals(mode=mode)
    if "liquidity" in signals.columns:
        signals = signals[
            pd.to_numeric(signals["liquidity"], errors="coerce").fillna(0) >= MIN_SIGNAL_LIQUIDITY
        ]
    signals_top = signals.head(MAX_SIGNALS_PER_CYCLE)

    # Pernas de arb não ocupam slots direcionais (mesma regra do check_exposure);
    # nem posições travadas esperando revisão manual (P1-16) — mercado já
    # fechou. Mesmo filtro de paper_trader.run_paper_trading — antes deste
    # fix, uma posição travada inflava n_open aqui e podia fazer este loop
    # parar de escanear sinais mais cedo do que deveria.
    open_pos_for_slots = open_pos
    if not open_pos.empty and "needs_manual_resolution" in open_pos.columns:
        open_pos_for_slots = open_pos[open_pos["needs_manual_resolution"].fillna(0) != 1]
    if not open_pos_for_slots.empty and "trade_type" in open_pos_for_slots.columns:
        n_open = int((open_pos_for_slots["trade_type"].fillna("value") != "arb").sum())
    else:
        n_open = len(open_pos_for_slots) if not open_pos_for_slots.empty else 0
    slots = max(0, MAX_OPEN_POSITIONS - n_open)

    # P1-19: içados pro chamador uma vez por ciclo, em vez de open_position
    # reler o parquet e reconsultar o banco a cada sinal candidato — este
    # daemon roda sem cap de sinais nenhum, então era ilimitado. Atualizados
    # a cada abertura pra não perder o efeito de uma posição aberta 2
    # candidatos atrás no mesmo ciclo (caps de diversificação, duplicata).
    # A linha apendada em open_pos não tem "id" (só existe depois do
    # INSERT) — serve só pra leitura de caps/duplicata neste loop; o passo
    # 5 abaixo recarrega open_pos do banco antes do mark_to_market.
    traded_ids = get_traded_condition_ids()

    n_opened = 0
    if not signals_top.empty:
        logger.info(f"Sinais disponíveis: {len(signals_top)} (de {len(signals)} pós-liquidez) | slots: {slots}")
        for _, sig in signals_top.iterrows():
            if slots <= 0 or float(portfolio["current_cash"]) < 5:
                break
            result = open_position(
                portfolio, sig.to_dict(), dry_run=dry_run,
                current_markets=current_mkts, open_positions=open_pos, traded_ids=traded_ids,
            )
            if result:
                n_opened += 1
                slots -= 1
                logger.info(
                    f"  ABERTA: {result['question'][:50]} "
                    f"{result['direction']} ${result['cost_usdc']:.2f} "
                    f"@ {result['entry_price']:.3f}"
                )
                if not dry_run:
                    traded_ids.add(str(result["condition_id"]))
                    open_pos = pd.concat(
                        [open_pos, pd.DataFrame([{**result, "status": "open", "needs_manual_resolution": 0}])],
                        ignore_index=True,
                    )
                # Atualiza portfolio para próxima iteração (cash decrementado)
                portfolio = get_or_create_portfolio()
        if n_opened == 0:
            logger.info("Nenhuma posição aberta (limites/edge/duplicatas)")
    else:
        logger.info("Sem sinais disponíveis (aguarde ciclo de geração, ou liquidez abaixo do piso)")

    # ── 5. Mark-to-market ────────────────────────────────────
    open_pos = get_open_positions()
    if not open_pos.empty and not current_mkts.empty:
        open_pos_mtm = mark_to_market(open_pos, current_mkts)
        print_portfolio(portfolio, open_pos_mtm)
    else:
        logger.info(f"Portfólio: ${portfolio['current_cash']:.2f} disponíveis | sem posições abertas")

    end = datetime.now(timezone.utc).strftime("%H:%M UTC")
    logger.info(f"=== Execution daemon concluído: {end} | abriu {n_opened} posições ===")


if __name__ == "__main__":
    main()
