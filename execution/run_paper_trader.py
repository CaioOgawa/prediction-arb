"""
run_paper_trader.py
Ponto de entrada do paper trader.

Uso:
    uv run python execution/run_paper_trader.py
    uv run python execution/run_paper_trader.py --mode deribit
    uv run python execution/run_paper_trader.py --mode all --dry-run
    uv run python execution/run_paper_trader.py --status

Gerar sinais antes de operar:
    uv run python signals/run_signals.py --mode odds   --fetch-fresh
    uv run python signals/run_signals.py --mode deribit --fetch-fresh

Ciclo completo (recomendado):
    uv run python pipeline/fetch_markets.py
    uv run python signals/run_signals.py --mode odds --fetch-fresh
    uv run python signals/run_signals.py --mode deribit --fetch-fresh
    uv run python execution/run_paper_trader.py --mode all
"""

import sys
from pathlib import Path

import click
import pandas as pd
from loguru import logger
from rich.console import Console

sys.path.insert(0, str(Path(__file__).parent.parent / "pipeline"))
sys.path.insert(0, str(Path(__file__).parent.parent / "risk"))

console = Console()


@click.command()
@click.option("--mode",           default="odds",  type=click.Choice(["odds", "deribit", "all"]), show_default=True,
              help="Fonte de sinais: 'odds' (esportes), 'deribit' (crypto) ou 'all' (ambos).")
@click.option("--capital",        default=1_000.0, type=float, show_default=True,
              help="Capital inicial em USDC (usado só ao criar o portfólio pela primeira vez).")
@click.option("--max-positions",  default=None,    type=int,
              help="Máximo de posições abertas simultaneamente. Default: MAX_OPEN_POSITIONS do risk_manager.")
@click.option("--edge-threshold", default=None,    type=float,
              help="Pré-filtro de edge mínimo. Default: MIN_EDGE_ABS do risk_manager "
                   "(o gate por fonte é aplicado no sizing).")
@click.option("--min-liquidity",  default=5_000.0, type=float, show_default=True,
              help="Liquidez mínima do mercado em USDC.")
@click.option("--top-signals",    default=30,      type=int,   show_default=True,
              help="Quantos sinais (top por edge × confidence) avaliar por ciclo.")
@click.option("--dry-run",        is_flag=True, default=False,
              help="Simula sem salvar posições no banco.")
@click.option("--status",         is_flag=True, default=False,
              help="Apenas exibe o portfólio atual, sem abrir novas posições.")
def main(
    mode: str,
    capital: float,
    max_positions: int | None,
    edge_threshold: float | None,
    min_liquidity: float,
    top_signals: int,
    dry_run: bool,
    status: bool,
) -> None:
    """Paper trader com sinais reais (odds + deribit) e Kelly × confidence sizing."""
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="<green>{time:HH:mm:ss}</green> | {message}")

    console.print("\n[bold]Polymarket Quant — Paper Trader[/bold]")

    from paper_trader import (
        init_db, get_or_create_portfolio, get_open_positions,
        load_current_markets, mark_to_market, print_portfolio,
        run_paper_trading,
    )

    if status:
        init_db()
        portfolio = get_or_create_portfolio(capital)
        open_pos  = get_open_positions()
        markets   = load_current_markets()
        mtm = mark_to_market(open_pos, markets) if not open_pos.empty and not markets.empty else open_pos
        print_portfolio(portfolio, mtm if not mtm.empty else pd.DataFrame())
        return

    run_paper_trading(
        initial_capital=capital,
        max_positions=max_positions,
        edge_threshold=edge_threshold,
        min_liquidity=min_liquidity,
        dry_run=dry_run,
        top_signals=top_signals,
        mode=mode,
    )

    console.print("\n[dim]Para atualizar: uv run python execution/run_paper_trader.py[/dim]")
    console.print("[dim]Para status:    uv run python execution/run_paper_trader.py --status[/dim]")


if __name__ == "__main__":
    main()
