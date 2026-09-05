"""
run_backtest.py
CLI para o forward-test performance tracker.

Uso:
    uv run python -m backtest.run_backtest
    uv run python -m backtest.run_backtest --html
    uv run python -m backtest.run_backtest --html --open-browser

O tracker lê o SQLite do paper trader e computa métricas de performance
das posições já fechadas. Quanto mais posições fecharem, mais ricas as métricas.
"""

import sys
import webbrowser
from pathlib import Path

import click
from loguru import logger
from rich.console import Console

console = Console()


@click.command()
@click.option("--html",          is_flag=True, default=False,
              help="Gera relatório HTML com gráficos Plotly.")
@click.option("--open-browser",  is_flag=True, default=False,
              help="Abre o relatório HTML no browser após gerar (implica --html).")
@click.option("--db",            default="data/db/paper_trading.db",
              type=click.Path(), show_default=True,
              help="Caminho para o banco SQLite do paper trader.")
@click.option("--output",        default=None, type=click.Path(),
              help="Caminho de saída para o HTML (default: backtest/results/backtest_report_<ts>.html).")
def main(html: bool, open_browser: bool, db: str, output: str | None) -> None:
    """Forward-test tracker: métricas de performance do paper trader."""
    logger.remove()
    logger.add(sys.stderr, level="WARNING",
               format="<green>{time:HH:mm:ss}</green> | {message}")

    console.print("\n[bold]Polymarket Quant — Forward Test Tracker[/bold]")

    from backtest.backtest import (
        load_positions, load_portfolio,
        print_backtest_summary, generate_html_report,
        DB_PATH,
    )

    db_path = Path(db)
    if not db_path.exists():
        console.print(f"[red]Banco não encontrado: {db_path}[/red]")
        console.print("[dim]Execute primeiro: uv run python -m execution.run_paper_trader[/dim]")
        sys.exit(1)

    closed, open_ = load_positions(db_path)
    portfolio      = load_portfolio(db_path)

    print_backtest_summary(closed, open_, portfolio)

    if html or open_browser:
        out = Path(output) if output else None
        report_path = generate_html_report(closed, open_, portfolio, out)
        console.print(f"\n[green]Relatório salvo:[/green] {report_path}")
        if open_browser:
            webbrowser.open(report_path.resolve().as_uri())

    if closed.empty:
        console.print(
            "\n[yellow]Dica:[/yellow] nenhuma posição fechada ainda — "
            "as métricas crescem conforme os mercados são resolvidos.\n"
            "[dim]Para resolver posições agora:[/dim]\n"
            "[dim]  uv run python -m execution.run_paper_trader --mode all[/dim]"
        )


if __name__ == "__main__":
    main()
