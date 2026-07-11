"""
eda.py
Análise exploratória do dataset de features do Polymarket.
Gera um relatório HTML interativo (Plotly) em outputs/plots/eda_report.html
e um resumo estatístico no terminal.

Uso:
    uv run python features/eda.py
    uv run python features/eda.py --open   # abre o browser automaticamente
"""

import sys
import webbrowser
import click
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from rich.console import Console
from rich.table import Table
from rich import box

sys.path.insert(0, str(Path(__file__).parent.parent / "pipeline"))

PLOTS_DIR = Path("outputs/plots")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

console = Console()

NUMERIC_FEATURES = [
    "yes_price", "days_to_resolution", "log_volume", "log_liquidity",
    "log_volume24hr", "log_volume1wk", "volume_recency_ratio",
    "spread", "price_1d_change", "price_1w_change", "price_1m_change",
]

BINARY_FEATURES = [
    "is_near_resolution", "is_long_horizon", "is_extreme_price", "has_resolution_source",
]


# ===========================================================================
# Carregamento
# ===========================================================================

def load_latest_features() -> pd.DataFrame:
    """Carrega o Parquet de features mais recente."""
    files = sorted(Path("data/raw/features").glob("*.parquet"), reverse=True)
    if not files:
        raise FileNotFoundError(
            "Nenhum arquivo de features encontrado. "
            "Execute: uv run python features/feature_engineering.py"
        )
    df = pd.read_parquet(files[0])
    console.print(f"[cyan]Features carregadas:[/cyan] {files[0].name} — {df.shape[0]:,} mercados × {df.shape[1]} features")
    return df


# ===========================================================================
# Relatório no terminal
# ===========================================================================

def print_summary(df: pd.DataFrame) -> None:
    """Imprime resumo estatístico completo no terminal."""

    # --- Visão geral ---
    console.print("\n[bold cyan]══════════════════════════════════════[/bold cyan]")
    console.print("[bold cyan]   EDA — POLYMARKET FEATURE DATASET   [/bold cyan]")
    console.print(f"[bold cyan]   {datetime.now().strftime('%Y-%m-%d %H:%M')}[/bold cyan]")
    console.print("[bold cyan]══════════════════════════════════════[/bold cyan]\n")

    console.print(f"[bold]Mercados no dataset:[/bold]  {len(df):,}")
    console.print(f"[bold]Features disponíveis:[/bold] {df.shape[1]}")

    # --- Missing values ---
    nan_pct = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
    nan_pct = nan_pct[nan_pct > 0]

    if len(nan_pct) > 0:
        console.print("\n[bold yellow]Valores faltantes:[/bold yellow]")
        nan_table = Table(box=box.SIMPLE)
        nan_table.add_column("Feature",  style="yellow")
        nan_table.add_column("NaN %",    justify="right")
        nan_table.add_column("NaN n",    justify="right")
        nan_table.add_column("Nota")
        for col, pct in nan_pct.items():
            n = int(df[col].isnull().sum())
            nota = "sem histórico CLOB (normal sem credenciais)" if pct > 40 else ""
            nan_table.add_row(col, f"{pct:.1f}%", str(n), nota)
        console.print(nan_table)

    # --- Estatísticas das features numéricas ---
    available = [c for c in NUMERIC_FEATURES if c in df.columns]
    stats = df[available].describe().T[["mean", "std", "min", "50%", "max"]]
    stats.columns = ["Média", "Desvio", "Min", "Mediana", "Máx"]

    console.print("\n[bold]Estatísticas das features numéricas:[/bold]")
    stat_table = Table(box=box.ROUNDED, show_lines=True)
    stat_table.add_column("Feature",  style="cyan", width=22)
    stat_table.add_column("Média",    justify="right")
    stat_table.add_column("Desvio",   justify="right")
    stat_table.add_column("Min",      justify="right")
    stat_table.add_column("Mediana",  justify="right")
    stat_table.add_column("Máx",      justify="right")

    for col, row in stats.iterrows():
        stat_table.add_row(
            col,
            f"{row['Média']:.3f}",
            f"{row['Desvio']:.3f}",
            f"{row['Min']:.3f}",
            f"{row['Mediana']:.3f}",
            f"{row['Máx']:.3f}",
        )
    console.print(stat_table)

    # --- Features binárias ---
    console.print("\n[bold]Features binárias (% = True):[/bold]")
    for col in BINARY_FEATURES:
        if col in df.columns:
            pct = df[col].mean() * 100
            bar = "█" * int(pct / 5)
            console.print(f"  {col:28s} {bar:20s} {pct:5.1f}%")

    # --- Distribuição de yes_price (calibração) ---
    prices = df["yes_price"].dropna()
    console.print(f"\n[bold]Distribuição de yes_price[/bold] ({len(prices):,} mercados com preço):")
    bins = [0, 0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.80, 0.90, 0.95, 1.0]
    labels = [f"{bins[i]:.2f}–{bins[i+1]:.2f}" for i in range(len(bins)-1)]
    counts = pd.cut(prices, bins=bins, labels=labels).value_counts().sort_index()
    for label, count in counts.items():
        bar = "█" * int(count / max(counts) * 35)
        console.print(f"  {label:12s} {bar:35s} {count:5,}")

    # --- Correlação com yes_price ---
    corr_cols = [c for c in NUMERIC_FEATURES if c != "yes_price" and c in df.columns]
    corrs = df[corr_cols].corrwith(df["yes_price"]).sort_values(key=abs, ascending=False)
    console.print("\n[bold]Correlação das features com yes_price:[/bold]")
    for col, val in corrs.head(10).items():
        color = "green" if val > 0 else "red"
        bar_len = int(abs(val) * 30)
        sign = "+" if val >= 0 else "-"
        console.print(f"  {col:28s} [{color}]{sign}{'█'*bar_len}[/{color}]  {val:+.3f}")

    # --- Segmentos de mercado ---
    console.print("\n[bold]Segmentos de mercado:[/bold]")
    segs = {
        "Próximos de resolução (≤7d)": df["days_to_resolution"].between(0, 7).sum(),
        "Curto prazo (8–30d)":          df["days_to_resolution"].between(8, 30).sum(),
        "Médio prazo (31–90d)":         df["days_to_resolution"].between(31, 90).sum(),
        "Longo prazo (>90d)":           (df["days_to_resolution"] > 90).sum(),
        "Sem data definida (-1)":       (df["days_to_resolution"] == -1).sum(),
        "Preço extremo (>85% ou <10%)": df["is_extreme_price"].sum(),
        "Com source de resolução":      df["has_resolution_source"].sum(),
    }
    seg_table = Table(box=box.SIMPLE)
    seg_table.add_column("Segmento",  style="cyan")
    seg_table.add_column("Mercados", justify="right")
    seg_table.add_column("%",        justify="right")
    for seg, count in segs.items():
        seg_table.add_row(seg, f"{count:,}", f"{count/len(df)*100:.1f}%")
    console.print(seg_table)


# ===========================================================================
# Relatório HTML com Plotly
# ===========================================================================

def build_html_report(df: pd.DataFrame) -> Path:
    """
    Gera relatório HTML interativo com ~8 gráficos.
    Salva em outputs/plots/eda_report.html.
    """
    figs = []

    # 1. Distribuição de yes_price (histograma + KDE visual)
    prices = df["yes_price"].dropna()
    fig1 = px.histogram(
        prices, nbins=50,
        title="Distribuição de yes_price — Calibração do Mercado",
        labels={"value": "Probabilidade YES", "count": "Número de mercados"},
        color_discrete_sequence=["#00b4d8"],
    )
    fig1.add_vline(x=0.5, line_dash="dash", line_color="gray",
                   annotation_text="50% (linha de equilíbrio)")
    fig1.update_layout(showlegend=False)
    figs.append(("1. Distribuição de yes_price", fig1))

    # 2. Missing values (barras horizontais)
    nan_pct = (df.isnull().sum() / len(df) * 100).sort_values()
    nan_pct = nan_pct[nan_pct > 0]
    if len(nan_pct) > 0:
        fig2 = px.bar(
            x=nan_pct.values, y=nan_pct.index, orientation="h",
            title="Valores Faltantes por Feature (%)",
            labels={"x": "% NaN", "y": "Feature"},
            color=nan_pct.values,
            color_continuous_scale="Reds",
        )
        fig2.update_layout(coloraxis_showscale=False)
        figs.append(("2. Missing Values", fig2))

    # 3. yes_price vs log_volume (scatter)
    sample = df.dropna(subset=["yes_price", "log_volume"]).sample(min(3000, len(df)), random_state=42)
    fig3 = px.scatter(
        sample, x="log_volume", y="yes_price",
        color="is_extreme_price",
        title="yes_price vs Volume (log) — detectando mercados extremos",
        labels={"log_volume": "log(Volume USDC)", "yes_price": "Probabilidade YES"},
        opacity=0.5,
        color_continuous_scale=["#0077b6", "#e63946"],
    )
    figs.append(("3. yes_price vs Volume", fig3))

    # 4. Dias até resolução (histograma, excluindo -1)
    days = df[df["days_to_resolution"] >= 0]["days_to_resolution"]
    fig4 = px.histogram(
        days.clip(upper=365), nbins=60,
        title="Distribuição de Dias até Resolução (cap 365d)",
        labels={"value": "Dias restantes", "count": "Mercados"},
        color_discrete_sequence=["#06d6a0"],
    )
    figs.append(("4. Dias até Resolução", fig4))

    # 5. Matriz de correlação
    corr_cols = [c for c in NUMERIC_FEATURES if c in df.columns]
    corr_matrix = df[corr_cols].corr()
    fig5 = px.imshow(
        corr_matrix,
        title="Matriz de Correlação — Features Numéricas",
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        text_auto=".2f",
    )
    fig5.update_layout(height=600)
    figs.append(("5. Correlação", fig5))

    # 6. Spread vs Liquidez
    sample2 = df.dropna(subset=["spread", "log_liquidity"]).sample(min(3000, len(df)), random_state=42)
    sample2 = sample2[sample2["spread"] < sample2["spread"].quantile(0.99)]  # remove outliers extremos
    fig6 = px.scatter(
        sample2, x="log_liquidity", y="spread",
        color="yes_price",
        title="Spread vs Liquidez — Custo de Execução",
        labels={"log_liquidity": "log(Liquidez)", "spread": "Spread bid-ask"},
        opacity=0.5,
        color_continuous_scale="Viridis",
    )
    figs.append(("6. Spread vs Liquidez", fig6))

    # 7. volume_recency_ratio (mercados ativos vs adormecidos)
    ratio = df["volume_recency_ratio"].dropna().clip(upper=10)
    fig7 = px.histogram(
        ratio, nbins=50,
        title="Volume Recency Ratio — Mercado Aquecendo ou Esfriando?",
        labels={"value": "Vol 24h / (Vol total / 30)", "count": "Mercados"},
        color_discrete_sequence=["#f77f00"],
    )
    fig7.add_vline(x=1.0, line_dash="dash", line_color="gray",
                   annotation_text="Ritmo normal (=1)")
    figs.append(("7. Volume Recency Ratio", fig7))

    # 8. Box plots das variações de preço
    change_cols = [c for c in ["price_1d_change", "price_1w_change", "price_1m_change"] if c in df.columns]
    if change_cols:
        melted = df[change_cols].melt(var_name="Período", value_name="Variação")
        melted = melted.dropna()
        melted = melted[melted["Variação"].between(
            melted["Variação"].quantile(0.01),
            melted["Variação"].quantile(0.99)
        )]
        fig8 = px.box(
            melted, x="Período", y="Variação",
            title="Distribuição de Variações de Preço por Período",
            color="Período",
            color_discrete_sequence=["#4361ee", "#7209b7", "#f72585"],
        )
        figs.append(("8. Variações de Preço", fig8))

    # Monta HTML final
    html_parts = ["""
    <html>
    <head>
      <meta charset="utf-8">
      <title>Polymarket Quant — EDA Features</title>
      <style>
        body { font-family: 'Segoe UI', sans-serif; background: #0d1117; color: #e6edf3; margin: 0; padding: 20px; }
        h1   { color: #58a6ff; border-bottom: 1px solid #30363d; padding-bottom: 12px; }
        h2   { color: #79c0ff; margin-top: 40px; font-size: 1.1em; }
        .plot-container { background: #161b22; border-radius: 8px; padding: 10px;
                          margin: 20px 0; border: 1px solid #30363d; }
        .meta  { color: #8b949e; font-size: 0.9em; }
        .badge { display: inline-block; background: #21262d; border: 1px solid #30363d;
                 border-radius: 20px; padding: 3px 10px; margin: 4px;
                 font-size: 0.85em; color: #8b949e; }
      </style>
    </head>
    <body>
    <h1>Polymarket Quant — EDA de Features</h1>
    """]

    html_parts.append(f"""
    <p class="meta">
      Gerado em {datetime.now().strftime('%Y-%m-%d %H:%M')} &nbsp;|&nbsp;
      <span class="badge">{len(df):,} mercados</span>
      <span class="badge">{df.shape[1]} features</span>
      <span class="badge">Fase 2 — Feature Engineering</span>
    </p>
    <p class="meta" style="color:#f0883e">
      ⚠️ Features de preço (price_Xd_change) e orderbook têm NaN elevado —
      isso é esperado sem credenciais CLOB. As features de contexto (volume, liquidez,
      spread, dias para resolução) estão completas e suficientes para um modelo baseline.
    </p>
    """)

    for title, fig in figs:
        fig.update_layout(
            paper_bgcolor="#161b22",
            plot_bgcolor="#0d1117",
            font=dict(color="#e6edf3"),
            title_font=dict(color="#58a6ff"),
        )
        html_parts.append(f'<div class="plot-container">')
        html_parts.append(f'<h2>{title}</h2>')
        html_parts.append(fig.to_html(full_html=False, include_plotlyjs="cdn" if title == figs[0][0] else False))
        html_parts.append("</div>")

    html_parts.append("</body></html>")

    path = PLOTS_DIR / "eda_report.html"
    path.write_text("\n".join(html_parts), encoding="utf-8")
    return path


# ===========================================================================
# Entrypoint
# ===========================================================================

@click.command()
@click.option("--open", "open_browser", is_flag=True, default=False,
              help="Abre o relatório HTML no browser automaticamente.")
def main(open_browser: bool) -> None:
    """EDA interativo do dataset de features do Polymarket."""
    df = load_latest_features()

    print_summary(df)

    console.print("\n[bold]Gerando relatório HTML interativo...[/bold]")
    path = build_html_report(df)
    console.print(f"\n[bold green]Relatório salvo:[/bold green] {path.resolve()}")
    console.print(f"[dim]Abra no browser: open {path.resolve()}[/dim]")

    if open_browser:
        webbrowser.open(f"file://{path.resolve()}")


if __name__ == "__main__":
    main()
