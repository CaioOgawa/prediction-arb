"""
run_execution.py
Execution daemon — roda a cada 5 minutos via launchd/cron.

Responsabilidade separada do run_cycle.py (signal generation, 30min):
  1. Resolve posições fechadas / expiradas
  2. Verifica early exits (profit target, edge flip)
  3. Verifica stop loss por drawdown (bloqueia rebalance/abertura em halt)
  3c. Abre baskets de arb estrutural pendentes (CSV mais recente, retry a
      cada 5min de baskets que falharam abrir no ciclo anterior)
  4. Abre novas posições direcionais com sinais disponíveis (CSV mais recente)
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
    from execution.paper_trader import execute_cycle, print_portfolio

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

    # R2: núcleo compartilhado com paper_trader.run_paper_trading() (30min).
    # alert_manual_resolution=False — esse alerta já sai no ciclo de 30min;
    # repetir aqui a cada 5min spammaria Telegram pela mesma posição travada.
    result = execute_cycle(mode=mode, dry_run=dry_run, alert_manual_resolution=False)

    if result["resolved"]:
        logger.info(f"Resolvidas: {len(result['resolved'])} posições")
        for r in result["resolved"]:
            sign = "+" if r["pnl_usdc"] >= 0 else ""
            logger.info(
                f"  [{r['status'].upper()}] {r['question'][:50]} "
                f"→ {sign}${r['pnl_usdc']:.2f}"
            )

    if result["orphans"]:
        groups = sorted({o["arb_group"] for o in result["orphans"]})
        logger.warning(
            f"Pernas de arb órfãs reclassificadas para 'value': {len(result['orphans'])} "
            f"({', '.join(groups)})"
        )

    if result["early_exits"]:
        logger.info(f"Early exits: {len(result['early_exits'])} posições fechadas")
        for e in result["early_exits"]:
            sign = "+" if e["pnl_usdc"] >= 0 else ""
            logger.info(
                f"  [EARLY_EXIT/{e.get('trigger','?')}] {e['question'][:50]} "
                f"→ {sign}${e['pnl_usdc']:.2f}"
            )

    if result["stopped"]:
        logger.warning(f"STOP LOSS ATIVADO: {result['stop_reason']} — rebalance e abertura de posição suspensos")
        print_portfolio(result["portfolio"], result["positions_mtm"])
        end = datetime.now(timezone.utc).strftime("%H:%M UTC")
        logger.info(f"=== Execution daemon concluído: {end} | halt de drawdown ===")
        return

    if result["rebalanced"]:
        logger.info(f"Rebalanceamentos: {len(result['rebalanced'])}")
        for r in result["rebalanced"]:
            logger.info(f"  +${r['add_usdc']:.2f} {r['question'][:50]} edge={r['new_edge']:.1%}")

    if result["baskets"]:
        logger.info(f"Baskets estruturais abertos: {len(result['baskets'])}")
        for b in result["baskets"]:
            logger.info(
                f"  BASKET {b['arb_group']} | {b['n_legs']} pernas × {b['shares']:.1f} sh "
                f"| custo ${b['total_cost']:.2f} → lucro garantido ${b['guaranteed_profit']:.2f}"
            )

    if result["signals_evaluated"] == 0:
        logger.info("Sem sinais disponíveis (aguarde ciclo de geração, ou liquidez abaixo do piso)")
    else:
        logger.info(f"Sinais avaliados: {result['signals_evaluated']}")
        for pos in result["opened_positions"]:
            logger.info(
                f"  ABERTA: {pos['question'][:50]} "
                f"{pos['direction']} ${pos['cost_usdc']:.2f} "
                f"@ {pos['entry_price']:.3f}"
            )
        if not result["opened_positions"]:
            logger.info("Nenhuma posição aberta (limites/edge/duplicatas)")

    portfolio, positions_mtm = result["portfolio"], result["positions_mtm"]
    if not positions_mtm.empty:
        print_portfolio(portfolio, positions_mtm)
    else:
        logger.info(f"Portfólio: ${portfolio['current_cash']:.2f} disponíveis | sem posições abertas")

    end = datetime.now(timezone.utc).strftime("%H:%M UTC")
    logger.info(f"=== Execution daemon concluído: {end} | abriu {len(result['opened_positions'])} posições ===")


if __name__ == "__main__":
    main()
