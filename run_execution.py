"""
run_execution.py
Execution daemon — roda a cada 5 minutos via launchd/cron.

Responsabilidade separada do run_cycle.py (signal generation, 30min):
  1. Resolve posições fechadas / expiradas
  2. Verifica early exits (profit target, edge flip)
  3. Abre novas posições com sinais disponíveis (CSV mais recente)
  4. Mark-to-market das posições abertas

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

import sys
from datetime import datetime, timezone
from pathlib import Path

import click
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
        load_current_markets,
        load_signals,
        mark_to_market,
        open_position,
        rebalance_positions,
        print_portfolio,
    )
    from risk_manager import resolve_positions, early_exit_positions

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

    # ── 3. Rebalanceamento de posições existentes ───────────
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
    signals = load_signals(mode=mode)
    n_opened = 0
    if not signals.empty:
        logger.info(f"Sinais disponíveis: {len(signals)}")
        for _, sig in signals.iterrows():
            result = open_position(portfolio, sig.to_dict(), dry_run=dry_run)
            if result:
                n_opened += 1
                logger.info(
                    f"  ABERTA: {result['question'][:50]} "
                    f"{result['direction']} ${result['cost_usdc']:.2f} "
                    f"@ {result['entry_price']:.3f}"
                )
                # Atualiza portfolio para próxima iteração (cash decrementado)
                portfolio = get_or_create_portfolio()
        if n_opened == 0 and not signals.empty:
            logger.info("Nenhuma posição aberta (limites/edge/duplicatas)")
    else:
        logger.info("Sem sinais disponíveis (aguarde ciclo de geração)")

    # ── 4. Mark-to-market ────────────────────────────────────
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
