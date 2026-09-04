"""
fetch_markets.py
Script principal da Fase 1: coleta mercados ativos via Gamma API,
salva em SQLite + Parquet e gera relatório EDA inicial.

Uso:
    uv run python pipeline/fetch_markets.py
    uv run python pipeline/fetch_markets.py --min-volume 10000
    uv run python pipeline/fetch_markets.py --min-volume 0 --no-report
"""

import sys
import click
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timezone

from loguru import logger
from rich.console import Console
from rich.table import Table
from rich import box

from gamma_collector import run as collect_markets, EmptySnapshotError

console = Console()

REPORTS_DIR = Path("outputs/reports")
PLOTS_DIR   = Path("outputs/plots")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def _eda_report(df: pd.DataFrame) -> None:
    """
    Gera relatório EDA no terminal e salva CSV/HTML com os principais insights.
    Analisa: distribuição por categoria, volume, liquidez, calibração de preços.
    """
    console.print("\n[bold cyan]═══════════════════════════════════════════[/bold cyan]")
    console.print("[bold cyan]       RELATÓRIO EDA — POLYMARKET MARKETS  [/bold cyan]")
    console.print(f"[bold cyan]  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}[/bold cyan]")
    console.print("[bold cyan]═══════════════════════════════════════════[/bold cyan]\n")

    # --- Resumo geral ---
    console.print(f"[bold]Total de mercados coletados:[/bold] {len(df):,}")
    console.print(f"[bold]Volume total:[/bold] ${df['volume'].sum():,.0f} USDC")
    console.print(f"[bold]Liquidez total:[/bold] ${df['liquidity'].sum():,.0f} USDC")
    console.print(f"[bold]Com preço YES disponível:[/bold] {df['yes_price'].notna().sum():,}\n")

    # --- Distribuição por categoria ---
    cat_stats = (
        df.groupby("category")
          .agg(
              n_markets  = ("conditionId", "count"),
              volume_sum = ("volume", "sum"),
              vol_median = ("volume", "median"),
              liq_sum    = ("liquidity", "sum"),
          )
          .sort_values("volume_sum", ascending=False)
          .reset_index()
    )

    table = Table(title="Distribuição por Categoria", box=box.ROUNDED, show_lines=True)
    table.add_column("Categoria",    style="cyan",  no_wrap=True)
    table.add_column("Mercados",     justify="right")
    table.add_column("Volume Total", justify="right", style="green")
    table.add_column("Vol. Mediana", justify="right")
    table.add_column("Liquidez",     justify="right", style="yellow")

    for _, row in cat_stats.iterrows():
        table.add_row(
            str(row["category"] or "N/A"),
            str(row["n_markets"]),
            f"${row['volume_sum']:,.0f}",
            f"${row['vol_median']:,.0f}",
            f"${row['liq_sum']:,.0f}",
        )
    console.print(table)

    # --- Top 20 mercados por volume 24h (mais relevante para trading ativo) ---
    vol_col = "volume24hr" if "volume24hr" in df.columns else "volume"
    top20 = df.nlargest(20, vol_col)[
        ["question", "category", "volume24hr", "volume", "liquidity", "yes_price", "spread"]
    ].reset_index(drop=True)

    table2 = Table(title=f"\nTop 20 Mercados — Volume 24h", box=box.SIMPLE, show_lines=False)
    table2.add_column("#",         width=3,  justify="right")
    table2.add_column("Mercado",   style="white", max_width=50)
    table2.add_column("Categoria", style="cyan",  width=14)
    table2.add_column("Vol 24h",   justify="right", style="bold green")
    table2.add_column("Liquidez",  justify="right", style="yellow")
    table2.add_column("YES",       justify="right")
    table2.add_column("Spread",    justify="right")

    for i, row in top20.iterrows():
        yes_str    = f"{row['yes_price']:.3f}" if pd.notna(row.get("yes_price")) else "—"
        spread_str = f"{row['spread']:.3f}"    if pd.notna(row.get("spread"))    else "—"
        vol24_str  = f"${row.get('volume24hr', 0):,.0f}" if pd.notna(row.get("volume24hr")) else "—"
        table2.add_row(
            str(i + 1),
            str(row["question"])[:50],
            str(row["category"] or "—"),
            vol24_str,
            f"${row['liquidity']:,.0f}",
            yes_str,
            spread_str,
        )
    console.print(table2)

    # --- Top 20 por liquidez disponível (melhor spread para execução) ---
    top20_liq = df[df["liquidity"] > 0].nlargest(20, "liquidity")[
        ["question", "category", "liquidity", "volume24hr", "yes_price", "spread"]
    ].reset_index(drop=True)

    if len(top20_liq) > 0:
        table3 = Table(title="\nTop 20 Mercados — Liquidez (melhor para execução)", box=box.SIMPLE)
        table3.add_column("#",         width=3,  justify="right")
        table3.add_column("Mercado",   style="white", max_width=50)
        table3.add_column("Categoria", style="cyan",  width=14)
        table3.add_column("Liquidez",  justify="right", style="bold yellow")
        table3.add_column("Vol 24h",   justify="right", style="green")
        table3.add_column("YES",       justify="right")
        table3.add_column("Spread",    justify="right")

        for i, row in top20_liq.iterrows():
            yes_str    = f"{row['yes_price']:.3f}" if pd.notna(row.get("yes_price")) else "—"
            spread_str = f"{row['spread']:.3f}"    if pd.notna(row.get("spread"))    else "—"
            vol24_str  = f"${row.get('volume24hr', 0):,.0f}" if pd.notna(row.get("volume24hr")) else "—"
            table3.add_row(
                str(i + 1),
                str(row["question"])[:50],
                str(row["category"] or "—"),
                f"${row['liquidity']:,.0f}",
                vol24_str,
                yes_str,
                spread_str,
            )
        console.print(table3)

    # --- Análise de calibração (distribuição de yes_price) ---
    prices = df["yes_price"].dropna()
    if len(prices) > 0:
        console.print("\n[bold]Distribuição de Preços YES (proxy de calibração):[/bold]")

        bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        labels = [f"{int(b*100)}-{int(bins[i+1]*100)}%" for i, b in enumerate(bins[:-1])]
        price_dist = pd.cut(prices, bins=bins, labels=labels).value_counts().sort_index()

        for label, count in price_dist.items():
            bar = "█" * int(count / max(price_dist) * 30)
            console.print(f"  {label:10s} {bar} {count:4d}")

        console.print(f"\n  Mediana: {prices.median():.3f} | Média: {prices.mean():.3f}")
        console.print(f"  Mercados perto de resolução (>85% ou <15%): "
                      f"{((prices > 0.85) | (prices < 0.15)).sum():,}")

    # --- Distribuição de dias até resolução ---
    if "endDate" in df.columns:
        end_dates = pd.to_datetime(df["endDate"], errors="coerce", utc=True)
        now = pd.Timestamp.now(tz="UTC")
        days_left = (end_dates - now).dt.days.dropna()
        days_left = days_left[days_left >= 0]

        if len(days_left) > 0:
            console.print(f"\n[bold]Dias até resolução:[/bold]")
            console.print(f"  <= 7 dias: {(days_left <= 7).sum():,} mercados")
            console.print(f"  8-30 dias: {((days_left > 7) & (days_left <= 30)).sum():,} mercados")
            console.print(f"  31-90 dias: {((days_left > 30) & (days_left <= 90)).sum():,} mercados")
            console.print(f"  > 90 dias:  {(days_left > 90).sum():,} mercados")

    # --- Salva relatório CSV ---
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = REPORTS_DIR / f"eda_markets_{ts}.csv"
    df.to_csv(report_path, index=False)

    top20_path = REPORTS_DIR / f"top_markets_{ts}.csv"
    top20.to_csv(top20_path, index=False)

    console.print(f"\n[green]Relatório salvo:[/green] {report_path}")
    console.print(f"[green]Top 20 salvo:[/green]   {top20_path}")

    # --- Próximos passos ---
    console.print("\n[bold yellow]Próximos passos:[/bold yellow]")
    console.print("  1. Configure credenciais no .env (POLY_API_KEY etc.)")
    console.print("  2. Execute: uv run python pipeline/clob_collector.py")
    console.print("     → coleta histórico de preços dos top mercados")
    console.print("  3. Execute: uv run python features/feature_engineering.py")
    console.print("     → constrói features para os modelos de ML\n")


@click.command()
@click.option("--min-volume", default=5_000, type=float, show_default=True,
              help="Volume mínimo em USDC para incluir o mercado.")
@click.option("--report/--no-report", default=True, show_default=True,
              help="Gerar relatório EDA após coleta.")
def main(min_volume: float, report: bool) -> None:
    """Coleta mercados ativos do Polymarket e gera análise exploratória."""
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="<green>{time:HH:mm:ss}</green> | {message}")

    console.print(f"\n[bold]Polymarket Quant — Fase 1: Pipeline de Dados[/bold]")
    console.print(f"Coletando mercados com volume >= ${min_volume:,.0f} USDC...\n")

    try:
        df = collect_markets(min_volume=min_volume)
    except EmptySnapshotError as e:
        console.print(f"[red]{e}[/red]")
        console.print("[red]Verifique a conexão com a Gamma API — nenhum snapshot novo foi publicado "
                       "(o último snapshot válido continua sendo o mais recente para o resto do sistema).[/red]")
        sys.exit(1)

    if report:
        _eda_report(df)
    else:
        console.print(f"[green]{len(df):,} mercados coletados e salvos.[/green]")


if __name__ == "__main__":
    main()
