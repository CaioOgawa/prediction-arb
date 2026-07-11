"""
run_signals.py
Ponto de entrada da Fase 4 — Geração de Sinais.

MODOS (apenas stat arb — ADR-008):
  --mode odds     (padrão) — edge real contra odds de bookmakers (Pinnacle).
                  Requer ODDS_API_KEY no .env e dados frescos do odds_collector.
                  Use para mercados esportivos.

  --mode deribit  — edge contra Black-Scholes com IV do Deribit (crypto BTC/ETH).

O modo ml foi APOSENTADO em 2026-07 (ADR-008): inferência incompatível com o
bundle v2 + data leakage. O código segue congelado em ml_lab / signal_generator.

Uso:
    uv run python signals/run_signals.py
    uv run python signals/run_signals.py --mode odds --edge-threshold 0.05
    uv run python signals/run_signals.py --mode deribit --fetch-fresh
"""

import sys
from pathlib import Path

import click
from loguru import logger
from rich.console import Console

sys.path.insert(0, str(Path(__file__).parent.parent / "pipeline"))

console = Console()


@click.command()
@click.option("--mode",            default="odds",  type=click.Choice(["odds", "deribit"]), show_default=True,
              help="Fonte de edge: 'odds' (esportes) ou 'deribit' (crypto B-S).")
@click.option("--edge-threshold",  default=None,    type=float,
              help="Edge mínimo |poly - fair_prob|. Default por fonte: MIN_EDGE_TO_TRADE do risk_manager "
                   "(odds 0.08, deribit 0.05).")
@click.option("--min-liquidity",   default=5_000.0, type=float, show_default=True,
              help="Liquidez mínima do mercado em USDC.")
@click.option("--min-volume24h",   default=500.0,   type=float, show_default=True,
              help="Volume mínimo nas últimas 24h em USDC.")
@click.option("--min-days-left",   default=0,       type=int,   show_default=True,
              help="Dias mínimos até resolução (0 = aceita jogos de hoje que ainda não começaram).")
@click.option("--min-hours-left",  default=None,    type=float,
              help="[Modo deribit] Horas mínimas até expiração. Sobrescreve --min-days-left×24 quando fornecido.")
@click.option("--max-hours-left",  default=8_760.0, type=float, show_default=True,
              help="[Modo deribit] Horas máximas até expiração (padrão: 8760h = 1 ano).")
@click.option("--min-match-score", default=0.25,    type=float, show_default=True,
              help="[Modo odds] Score mínimo de matching Poly ↔ bookmaker (0–1).")
@click.option("--max-delta-days",  default=30,      type=int,   show_default=True,
              help="[Modo deribit] Janela máxima (dias) entre expiração Deribit e Polymarket.")
@click.option("--fetch-fresh",     is_flag=True,    default=False,
              help="Re-executa o coletor externo antes de gerar sinais (odds ou deribit).")
@click.option("--top-n",           default=20,      type=int,   show_default=True,
              help="Número de sinais a exibir na tabela terminal.")
@click.option("--no-save",         is_flag=True,    default=False,
              help="Não salva CSV de resultados.")
def main(
    mode: str,
    edge_threshold: float | None,
    min_liquidity: float,
    min_volume24h: float,
    min_days_left: int,
    min_hours_left: float | None,
    max_hours_left: float,
    min_match_score: float,
    fetch_fresh: bool,
    max_delta_days: int,
    top_n: int,
    no_save: bool,
) -> None:
    """Fase 4: gera sinais de trading (modo odds ou deribit)."""
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="<green>{time:HH:mm:ss}</green> | {message}")

    # Threshold default por fonte vem do risk_manager (fonte única de constantes).
    # --edge-threshold explícito sempre vence (antes, 0.08 explícito virava 0.05).
    sys.path.insert(0, str(Path(__file__).parent.parent / "risk"))
    from risk_manager import MIN_EDGE_TO_TRADE
    _edge = edge_threshold if edge_threshold is not None else MIN_EDGE_TO_TRADE.get(mode, 0.08)

    console.print(f"\n[bold]Polymarket Quant — Geração de Sinais[/bold]")
    console.print(f"Modo: [cyan]{mode.upper()}[/cyan]  |  edge ≥ {_edge:.0%}  |  liquidez ≥ ${min_liquidity:,.0f}\n")

    from signal_generator import generate_odds_signals, generate_deribit_signals

    if mode == "odds":
        signals = generate_odds_signals(
            min_divergence=_edge,
            min_liquidity=min_liquidity,
            min_volume24h=min_volume24h,
            min_days_left=min_days_left,
            min_match_score=min_match_score,
            top_n=top_n,
            save=not no_save,
            fetch_fresh=fetch_fresh,
        )
    else:  # deribit
        # min_hours_left: explicit flag wins; else default 8h
        # (min_days_left é ignorado no modo deribit — controle é em horas)
        _hours = min_hours_left if min_hours_left is not None else 8.0
        signals = generate_deribit_signals(
            min_divergence=_edge,
            min_liquidity=min_liquidity,
            min_hours_left=_hours,
            max_hours_left=max_hours_left,
            max_delta_days=max_delta_days,
            top_n=top_n,
            save=not no_save,
            fetch_fresh=fetch_fresh,
        )

    if signals.empty:
        console.print("[yellow]Nenhum sinal gerado. Verifique os filtros ou atualize os dados.[/yellow]")
        if mode == "odds":
            console.print("[dim]  → uv run python pipeline/fetch_markets.py   # atualiza mercados[/dim]")
            console.print("[dim]  → uv run python signals/run_signals.py --mode odds --fetch-fresh[/dim]")
        sys.exit(0)

    dir_col   = signals["direction"].astype(str)
    n_buy_yes = (dir_col == "BUY_YES").sum()
    n_buy_no  = (dir_col == "BUY_NO").sum()
    console.print(f"[bold green]BUY_YES:[/bold green] {n_buy_yes}  |  [bold magenta]BUY_NO:[/bold magenta] {n_buy_no}")
    console.print(f"\n[dim]Próximo passo: uv run python execution/run_paper_trader.py[/dim]")


if __name__ == "__main__":
    main()
