"""
backtest.py
Forward-test performance tracker para o paper trader Polymarket.

Como funciona:
  Não há dados históricos free — este é um tracker de performance *forward*:
  acumula posições conforme elas resolvem no SQLite e computa métricas reais.

Métricas calculadas:
  - P&L curve (cumulativo ao longo do tempo)
  - Win rate total e por fonte de sinal (odds / deribit / ml)
  - Edge calibration: edge previsto vs. retorno realizado
  - Confidence accuracy: score de conf vs. taxa de acerto real
  - Sharpe ratio anualizado (baseado em retornos diários de posições fechadas)
  - Max drawdown
  - Exposição atual (posições abertas por categoria e fonte)

Tudo é computado a partir do SQLite — basta rodar periodicamente conforme
novas posições fecham. Com 0 posições fechadas o relatório mostra apenas
o estado atual do portfólio (posições abertas, capital, etc.).
"""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

DB_PATH     = Path("data/db/paper_trading.db")
RESULTS_DIR = Path("backtest/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ──────────────────────────────────────────────────────────
# Leitura do banco
# ──────────────────────────────────────────────────────────

def load_positions(db_path: Path = DB_PATH) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Retorna (closed_df, open_df) a partir do SQLite.
    Adiciona colunas `signal_source` e `confidence` se a tabela antiga não tiver.

    Escopo: apenas posições abertas a partir do portfólio ATUAL (última linha da
    tabela portfolio). Um reset de portfólio zera as métricas — sem isso, os 47
    trades corrompidos do incidente 2026-05 poluiriam o win rate para sempre.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    portfolio_start = conn.execute(
        "SELECT created_at FROM portfolio ORDER BY id DESC LIMIT 1"
    ).fetchone()
    since = portfolio_start["created_at"] if portfolio_start else "1970-01-01"

    pragma = {r["name"] for r in conn.execute("PRAGMA table_info(positions)")}

    select_cols = [
        "id", "opened_at", "closed_at", "condition_id", "question",
        "category", "direction", "entry_price", "exit_price", "shares",
        "cost_usdc", "pnl_usdc", "status", "edge_at_entry",
        "prob_at_entry", "end_date",
    ]
    if "signal_source" in pragma:
        select_cols.append("signal_source")
    if "confidence" in pragma:
        select_cols.append("confidence")
    if "trade_type" in pragma:
        select_cols.append("trade_type")

    cols_sql = ", ".join(select_cols)
    df = pd.read_sql_query(
        f"SELECT {cols_sql} FROM positions WHERE opened_at >= ?",
        conn, params=(since,),
    )
    conn.close()

    if "signal_source" not in df.columns:
        df["signal_source"] = "unknown"
    if "confidence" not in df.columns:
        df["confidence"] = float("nan")
    if "trade_type" not in df.columns:
        df["trade_type"] = "unknown"

    for col in ["opened_at", "closed_at", "end_date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)

    closed = df[df["status"].isin(["closed", "expired"])].copy()
    open_  = df[df["status"] == "open"].copy()
    return closed, open_


def load_portfolio(db_path: Path = DB_PATH) -> dict:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT * FROM portfolio ORDER BY id DESC LIMIT 1"
    ).fetchone()
    conn.close()
    if row is None:
        return {"initial_capital": 1000.0, "current_cash": 1000.0}
    return dict(row)


# ──────────────────────────────────────────────────────────
# Métricas de performance
# ──────────────────────────────────────────────────────────

def pnl_curve(closed: pd.DataFrame) -> pd.DataFrame:
    """
    Retorna DataFrame com P&L cumulativo ao longo do tempo (por data de fechamento).
    """
    if closed.empty:
        return pd.DataFrame(columns=["date", "pnl_usdc", "cum_pnl"])

    df = closed[["closed_at", "pnl_usdc"]].copy()
    df["date"] = df["closed_at"].dt.date
    daily = df.groupby("date")["pnl_usdc"].sum().reset_index()
    daily["cum_pnl"] = daily["pnl_usdc"].cumsum()
    return daily


def win_rate_by_source(closed: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula win rate, P&L médio e total por fonte de sinal.
    """
    if closed.empty:
        return pd.DataFrame(columns=["signal_source", "n_trades", "n_wins", "win_rate", "avg_pnl", "total_pnl"])

    closed = closed.copy()
    closed["won"] = closed["pnl_usdc"] > 0

    stats = (
        closed.groupby("signal_source")
        .agg(
            n_trades  = ("pnl_usdc", "count"),
            n_wins    = ("won", "sum"),
            avg_pnl   = ("pnl_usdc", "mean"),
            total_pnl = ("pnl_usdc", "sum"),
        )
        .reset_index()
    )
    stats["win_rate"] = stats["n_wins"] / stats["n_trades"]
    return stats


def win_rate_by_trade_type(closed: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula win rate, P&L médio e total por tipo de trade (value / momentum).
    """
    if closed.empty or "trade_type" not in closed.columns:
        return pd.DataFrame(columns=["trade_type", "n_trades", "n_wins", "win_rate", "avg_pnl", "total_pnl"])

    closed = closed.copy()
    closed["won"] = closed["pnl_usdc"] > 0

    stats = (
        closed.groupby("trade_type")
        .agg(
            n_trades  = ("pnl_usdc", "count"),
            n_wins    = ("won", "sum"),
            avg_pnl   = ("pnl_usdc", "mean"),
            total_pnl = ("pnl_usdc", "sum"),
        )
        .reset_index()
    )
    stats["win_rate"] = stats["n_wins"] / stats["n_trades"]
    return stats


def edge_calibration(closed: pd.DataFrame, n_bins: int = 5) -> pd.DataFrame:
    """
    Compara edge previsto (edge_at_entry) vs. retorno realizado por posição.

    Retorno realizado = pnl_usdc / cost_usdc (ROI da posição).
    Agrupa em bins de edge para ver se o modelo é calibrado.
    """
    if closed.empty or "edge_at_entry" not in closed.columns:
        return pd.DataFrame()

    df = closed[closed["edge_at_entry"].notna() & (closed["cost_usdc"] > 0)].copy()
    if df.empty:
        return pd.DataFrame()

    df["realized_return"] = df["pnl_usdc"] / df["cost_usdc"]
    df["edge_bin"] = pd.cut(df["edge_at_entry"], bins=n_bins)

    cal = (
        df.groupby("edge_bin", observed=False)
        .agg(
            n         = ("realized_return", "count"),
            avg_edge  = ("edge_at_entry", "mean"),
            avg_return= ("realized_return", "mean"),
        )
        .reset_index()
    )
    cal["edge_bin"] = cal["edge_bin"].astype(str)
    return cal


def confidence_accuracy(closed: pd.DataFrame, n_bins: int = 4) -> pd.DataFrame:
    """
    Verifica se confidence score prevê corretamente a taxa de acerto.
    Bins de confidence → win rate realizado dentro de cada bin.
    """
    if closed.empty:
        return pd.DataFrame()

    df = closed[closed["confidence"].notna()].copy()
    if df.empty:
        return pd.DataFrame()

    df["won"] = (df["pnl_usdc"] > 0).astype(int)
    df["conf_bin"] = pd.cut(df["confidence"], bins=n_bins)

    acc = (
        df.groupby("conf_bin", observed=False)
        .agg(
            n        = ("won", "count"),
            avg_conf = ("confidence", "mean"),
            win_rate = ("won", "mean"),
        )
        .reset_index()
    )
    acc["conf_bin"] = acc["conf_bin"].astype(str)
    return acc


def sharpe_ratio(closed: pd.DataFrame, initial_capital: float = 1000.0) -> float:
    """
    Sharpe ratio anualizado usando retornos diários de P&L realizado.
    Assume risk-free = 0 (crypto context).
    """
    if closed.empty:
        return float("nan")

    df = closed[["closed_at", "pnl_usdc"]].copy()
    df["date"] = df["closed_at"].dt.date
    daily_pnl = df.groupby("date")["pnl_usdc"].sum()

    daily_ret = daily_pnl / initial_capital
    if len(daily_ret) < 2 or daily_ret.std() == 0:
        return float("nan")

    return float(daily_ret.mean() / daily_ret.std() * np.sqrt(252))


def max_drawdown(closed: pd.DataFrame) -> float:
    """
    Máximo drawdown em USDC a partir da curva de P&L cumulativo.
    """
    curve = pnl_curve(closed)
    if curve.empty:
        return 0.0

    cum = curve["cum_pnl"].values
    running_max = np.maximum.accumulate(cum)
    drawdowns = cum - running_max
    return float(drawdowns.min())


def _mtm_open_value(open_: pd.DataFrame) -> float | None:
    """
    Valor mark-to-market das posições abertas usando o parquet de mercados
    mais recente. Retorna None se não houver parquet (caller usa cost basis).
    Sem isso, o valor do portfólio ignorava a variação não realizada.
    """
    if open_.empty:
        return 0.0
    parquets = sorted(Path("data/raw/markets").glob("markets_all_*.parquet"))
    if not parquets:
        return None
    try:
        mkts = pd.read_parquet(parquets[-1], columns=["conditionId", "yes_price", "spread"])
    except Exception:
        return None
    prices = (
        mkts.drop_duplicates("conditionId", keep="last")
        .set_index("conditionId")[["yes_price", "spread"]]
        .to_dict(orient="index")
    )
    total = 0.0
    for _, pos in open_.iterrows():
        info   = prices.get(pos["condition_id"], {})
        yes    = float(info.get("yes_price") or pos["entry_price"])
        spread = max(float(info.get("spread") or 0), 0.005)
        if pos["direction"] == "BUY_YES":
            px = max(yes - spread / 2, 0.001)
        else:
            px = max((1.0 - yes) - spread / 2, 0.001)
        total += px * float(pos["shares"])
    return round(total, 2)


def summary_stats(
    closed: pd.DataFrame,
    open_: pd.DataFrame,
    portfolio: dict,
) -> dict[str, Any]:
    """Compila todas as métricas em um dicionário."""
    initial = float(portfolio.get("initial_capital", 1000))
    cash    = float(portfolio.get("current_cash", initial))

    total_cost_open   = float(open_["cost_usdc"].sum()) if not open_.empty else 0.0
    mtm_value         = _mtm_open_value(open_)
    open_value        = mtm_value if mtm_value is not None else total_cost_open
    total_pnl_closed  = float(closed["pnl_usdc"].sum()) if not closed.empty else 0.0
    portfolio_value   = cash + open_value
    total_return_pct  = (portfolio_value - initial) / initial * 100

    n_closed = len(closed)
    n_wins   = int((closed["pnl_usdc"] > 0).sum()) if not closed.empty else 0

    return {
        "initial_capital":   initial,
        "current_cash":      cash,
        "open_positions":    len(open_),
        "closed_positions":  n_closed,
        "total_cost_open":   round(total_cost_open, 2),
        "unrealized_pnl":    round(open_value - total_cost_open, 2) if mtm_value is not None else None,
        "portfolio_value":   round(portfolio_value, 2),
        "total_pnl_closed":  round(total_pnl_closed, 2),
        "total_return_pct":  round(total_return_pct, 2),
        "win_rate":          round(n_wins / n_closed, 3) if n_closed > 0 else None,
        "n_wins":            n_wins,
        "sharpe":            round(sharpe_ratio(closed, initial), 3) if n_closed >= 5 else None,
        "max_drawdown_usdc": round(max_drawdown(closed), 2),
    }


# ──────────────────────────────────────────────────────────
# Relatório HTML
# ──────────────────────────────────────────────────────────

def _df_to_html_table(df: pd.DataFrame, fmt: dict | None = None) -> str:
    """Converte DataFrame para HTML table com classes de estilo."""
    if df.empty:
        return "<p><em>Sem dados suficientes ainda.</em></p>"

    fmt = fmt or {}
    rows = ""
    for _, row in df.iterrows():
        cells = ""
        for col in df.columns:
            val = row[col]
            if col in fmt:
                try:
                    val = fmt[col](val)
                except Exception:
                    val = str(val)
            else:
                if isinstance(val, float):
                    val = f"{val:.4f}"
            cells += f"<td>{val}</td>"
        rows += f"<tr>{cells}</tr>"

    headers = "".join(f"<th>{c}</th>" for c in df.columns)
    return f"""
    <table>
      <thead><tr>{headers}</tr></thead>
      <tbody>{rows}</tbody>
    </table>"""


def generate_html_report(
    closed: pd.DataFrame,
    open_:  pd.DataFrame,
    portfolio: dict,
    output_path: Path | None = None,
) -> Path:
    """Gera relatório HTML com Plotly charts e tabelas de métricas."""
    try:
        import plotly.graph_objects as go
        import plotly.io as pio
        HAS_PLOTLY = True
    except ImportError:
        HAS_PLOTLY = False
        logger.warning("plotly não instalado — relatório sem gráficos (pip install plotly)")

    stats    = summary_stats(closed, open_, portfolio)
    wr_src   = win_rate_by_source(closed)
    wr_tt    = win_rate_by_trade_type(closed)
    edge_cal = edge_calibration(closed)
    conf_acc = confidence_accuracy(closed)
    curve    = pnl_curve(closed)

    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # ── Plotly charts ──────────────────────────────────────
    # Separados por seção — o hack antigo de split("</div>") duplicava o
    # gráfico de P&L no relatório
    pnl_chart_html = ""
    charts_html    = ""

    if HAS_PLOTLY and not curve.empty:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=curve["date"].astype(str),
            y=curve["cum_pnl"],
            mode="lines+markers",
            name="P&L Cumulativo",
            line=dict(color="#00d4aa", width=2),
            fill="tozeroy",
            fillcolor="rgba(0,212,170,0.15)",
        ))
        fig.update_layout(
            title="P&L Cumulativo (posições fechadas)",
            xaxis_title="Data", yaxis_title="P&L (USDC)",
            template="plotly_dark", height=350,
            margin=dict(l=40, r=20, t=50, b=40),
        )
        pnl_chart_html = pio.to_html(fig, full_html=False, include_plotlyjs=False)

    if HAS_PLOTLY and not wr_src.empty:
        fig2 = go.Figure(go.Bar(
            x=wr_src["signal_source"],
            y=(wr_src["win_rate"] * 100).round(1),
            text=(wr_src["win_rate"] * 100).round(1).astype(str) + "%",
            textposition="outside",
            marker_color=["#00d4aa", "#ff6b6b", "#ffd93d"][:len(wr_src)],
        ))
        fig2.update_layout(
            title="Win Rate por Fonte de Sinal",
            xaxis_title="Fonte", yaxis_title="Win Rate (%)",
            template="plotly_dark", height=300,
            margin=dict(l=40, r=20, t=50, b=40),
        )
        charts_html += pio.to_html(fig2, full_html=False, include_plotlyjs=False)

    if HAS_PLOTLY and not edge_cal.empty and not edge_cal["avg_edge"].isna().all():
        fig3 = go.Figure()
        fig3.add_trace(go.Bar(
            name="Retorno Realizado",
            x=edge_cal["edge_bin"],
            y=edge_cal["avg_return"],
            marker_color="#4dabf7",
        ))
        fig3.add_trace(go.Scatter(
            name="Edge Previsto",
            x=edge_cal["edge_bin"],
            y=edge_cal["avg_edge"],
            mode="markers+lines",
            marker=dict(color="#ffd93d", size=10),
        ))
        fig3.update_layout(
            title="Calibração: Edge Previsto vs. Retorno Realizado",
            xaxis_title="Bin de Edge", yaxis_title="Valor",
            template="plotly_dark", height=350,
            margin=dict(l=40, r=20, t=50, b=40),
        )
        charts_html += pio.to_html(fig3, full_html=False, include_plotlyjs=False)

    # ── Open positions table ───────────────────────────────
    if not open_.empty:
        open_disp = open_[[
            "question", "direction", "entry_price", "cost_usdc",
            "signal_source", "confidence", "end_date",
        ]].copy()
        open_disp["question"] = open_disp["question"].str[:50]
        open_disp["end_date"] = open_disp["end_date"].dt.strftime("%Y-%m-%d").fillna("-")
        open_table = _df_to_html_table(open_disp, fmt={
            "entry_price": lambda v: f"{v:.3f}",
            "cost_usdc":   lambda v: f"${v:.2f}",
            "confidence":  lambda v: f"{v:.2f}" if v == v else "-",
        })
    else:
        open_table = "<p><em>Nenhuma posição aberta.</em></p>"

    # ── Closed positions table ─────────────────────────────
    if not closed.empty:
        cl_disp = closed[[
            "question", "direction", "entry_price", "exit_price",
            "pnl_usdc", "signal_source",
        ]].copy()
        cl_disp["question"] = cl_disp["question"].str[:50]
        closed_table = _df_to_html_table(cl_disp, fmt={
            "entry_price": lambda v: f"{v:.3f}",
            "exit_price":  lambda v: f"{v:.3f}" if v == v else "-",
            "pnl_usdc":    lambda v: f'<span style="color:{"#00d4aa" if v >= 0 else "#ff6b6b"}">${v:+.2f}</span>',
        })
    else:
        closed_table = "<p><em>Sem posições fechadas ainda.</em></p>"

    # ── Win rate table ─────────────────────────────────────
    wr_fmt = {
        "win_rate":  lambda v: f"{v:.1%}",
        "avg_pnl":   lambda v: f"${v:+.2f}",
        "total_pnl": lambda v: f'<span style="color:{"#00d4aa" if v >= 0 else "#ff6b6b"}">${v:+.2f}</span>',
    }
    wr_table = _df_to_html_table(wr_src, fmt=wr_fmt)

    # ── Stats cards ────────────────────────────────────────
    def card(label: str, value: str, color: str = "#e9ecef") -> str:
        return f"""
        <div class="stat-card">
          <div class="stat-value" style="color:{color}">{value}</div>
          <div class="stat-label">{label}</div>
        </div>"""

    pnl_color = "#00d4aa" if stats["total_pnl_closed"] >= 0 else "#ff6b6b"
    ret_color  = "#00d4aa" if stats["total_return_pct"] >= 0 else "#ff6b6b"

    cards = "".join([
        card("Capital Inicial",  f"${stats['initial_capital']:,.0f}"),
        card("Caixa Atual",      f"${stats['current_cash']:,.2f}"),
        card("Valor do Portfólio", f"${stats['portfolio_value']:,.2f}"),
        card("P&L Realizado",    f"${stats['total_pnl_closed']:+.2f}", pnl_color),
        card("Retorno Total",    f"{stats['total_return_pct']:+.2f}%", ret_color),
        card("Posições Abertas", str(stats["open_positions"])),
        card("Posições Fechadas", str(stats["closed_positions"])),
        card("Win Rate",         f"{stats['win_rate']:.1%}" if stats["win_rate"] is not None else "—"),
        card("Sharpe (anual.)",  f"{stats['sharpe']:.2f}" if stats["sharpe"] is not None else "—"),
        card("Max Drawdown",     f"${stats['max_drawdown_usdc']:.2f}", "#ff6b6b" if stats["max_drawdown_usdc"] < 0 else "#e9ecef"),
    ])

    plotly_cdn = '<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>' if HAS_PLOTLY else ""

    html = f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
  <meta charset="UTF-8">
  <title>Polymarket Quant — Backtest Report</title>
  {plotly_cdn}
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ font-family: 'Segoe UI', sans-serif; background: #0d1117; color: #e9ecef; padding: 24px; }}
    h1 {{ font-size: 1.6rem; margin-bottom: 4px; color: #f8f9fa; }}
    h2 {{ font-size: 1.1rem; margin: 24px 0 10px; color: #adb5bd; border-bottom: 1px solid #2d3748; padding-bottom: 6px; }}
    .subtitle {{ color: #6c757d; font-size: 0.85rem; margin-bottom: 24px; }}
    .stats-grid {{ display: flex; flex-wrap: wrap; gap: 12px; margin-bottom: 24px; }}
    .stat-card {{ background: #161b22; border: 1px solid #2d3748; border-radius: 8px; padding: 14px 18px; min-width: 130px; }}
    .stat-value {{ font-size: 1.4rem; font-weight: 700; }}
    .stat-label {{ font-size: 0.75rem; color: #6c757d; margin-top: 4px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 0.82rem; margin-bottom: 8px; }}
    th {{ background: #21262d; padding: 8px 12px; text-align: left; color: #adb5bd; font-weight: 600; }}
    td {{ padding: 7px 12px; border-top: 1px solid #21262d; }}
    tr:hover td {{ background: #161b22; }}
    .chart-section {{ margin-bottom: 24px; }}
    .no-data {{ color: #6c757d; font-style: italic; padding: 12px 0; }}
  </style>
</head>
<body>
  <h1>Polymarket Quant — Forward Test Report</h1>
  <p class="subtitle">Gerado em {now_str} &nbsp;|&nbsp; {stats['closed_positions']} posições fechadas &nbsp;|&nbsp; {stats['open_positions']} abertas</p>

  <h2>Métricas Gerais</h2>
  <div class="stats-grid">{cards}</div>

  <h2>P&L Cumulativo</h2>
  <div class="chart-section">
    {pnl_chart_html if pnl_chart_html else "<p class='no-data'>Aguardando posições fechadas para plotar a curva.</p>"}
  </div>

  <h2>Win Rate por Fonte</h2>
  {wr_table}

  <h2>Win Rate por Trade Type</h2>
  {_df_to_html_table(wr_tt, fmt={
      "win_rate":  lambda v: f"{v:.1%}",
      "avg_pnl":   lambda v: f"${v:+.2f}",
      "total_pnl": lambda v: f'<span style="color:{"#00d4aa" if v >= 0 else "#ff6b6b"}">${v:+.2f}</span>',
  })}

  <div class="chart-section">{charts_html}</div>

  <h2>Posições Abertas ({stats['open_positions']})</h2>
  {open_table}

  <h2>Posições Fechadas ({stats['closed_positions']})</h2>
  {closed_table}

</body>
</html>"""

    if output_path is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = RESULTS_DIR / f"backtest_report_{ts}.html"

    output_path.write_text(html, encoding="utf-8")
    return output_path


# ──────────────────────────────────────────────────────────
# Terminal display
# ──────────────────────────────────────────────────────────

def print_backtest_summary(
    closed: pd.DataFrame,
    open_:  pd.DataFrame,
    portfolio: dict,
) -> None:
    """Exibe resumo das métricas no terminal via Rich."""
    from rich.console import Console
    from rich.table import Table
    from rich import box

    console = Console()
    stats = summary_stats(closed, open_, portfolio)

    console.print("\n[bold]── Performance Summary ──────────────────────────[/bold]")
    console.print(f"  Capital inicial:   ${stats['initial_capital']:>10,.2f}")
    console.print(f"  Caixa atual:       ${stats['current_cash']:>10,.2f}")
    console.print(f"  Valor portfólio:   ${stats['portfolio_value']:>10,.2f}")

    pnl = stats["total_pnl_closed"]
    pnl_color = "green" if pnl >= 0 else "red"
    console.print(f"  P&L realizado:    [{pnl_color}]${pnl:>+10.2f}[/{pnl_color}]")

    ret = stats["total_return_pct"]
    ret_color = "green" if ret >= 0 else "red"
    console.print(f"  Retorno total:    [{ret_color}]{ret:>+10.2f}%[/{ret_color}]")
    console.print(f"  Posições abertas:  {stats['open_positions']:>10}")
    console.print(f"  Posições fechadas: {stats['closed_positions']:>10}")

    if stats["win_rate"] is not None:
        wr_color = "green" if stats["win_rate"] >= 0.5 else "yellow"
        console.print(f"  Win rate:         [{wr_color}]{stats['win_rate']:>10.1%}[/{wr_color}]  ({stats['n_wins']}/{stats['closed_positions']})")
    else:
        console.print("  Win rate:              —  (aguardando posições fechadas)")

    if stats["sharpe"] is not None:
        sh_color = "green" if stats["sharpe"] >= 1 else "yellow" if stats["sharpe"] >= 0 else "red"
        console.print(f"  Sharpe anualiz.:  [{sh_color}]{stats['sharpe']:>10.2f}[/{sh_color}]")

    if stats["max_drawdown_usdc"] < 0:
        console.print(f"  Max drawdown:     [red]${stats['max_drawdown_usdc']:>+.2f}[/red]")

    # Win rate by source
    wr_src = win_rate_by_source(closed)
    if not wr_src.empty:
        console.print("\n[bold]── Win Rate por Fonte ───────────────────────────[/bold]")
        tbl = Table(box=box.SIMPLE, show_header=True, header_style="bold cyan")
        tbl.add_column("Fonte",        style="cyan")
        tbl.add_column("Trades",       justify="right")
        tbl.add_column("Wins",         justify="right")
        tbl.add_column("Win Rate",     justify="right")
        tbl.add_column("Avg P&L",      justify="right")
        tbl.add_column("Total P&L",    justify="right")
        for _, row in wr_src.iterrows():
            wr_str    = f"{row['win_rate']:.1%}"
            avg_color = "green" if row["avg_pnl"] >= 0 else "red"
            tot_color = "green" if row["total_pnl"] >= 0 else "red"
            tbl.add_row(
                str(row["signal_source"]),
                str(int(row["n_trades"])),
                str(int(row["n_wins"])),
                wr_str,
                f"[{avg_color}]${row['avg_pnl']:+.2f}[/{avg_color}]",
                f"[{tot_color}]${row['total_pnl']:+.2f}[/{tot_color}]",
            )
        console.print(tbl)

    # Win rate by trade_type
    wr_tt = win_rate_by_trade_type(closed)
    if not wr_tt.empty:
        console.print("\n[bold]── Win Rate por Trade Type ──────────────────────[/bold]")
        tbl_tt = Table(box=box.SIMPLE, show_header=True, header_style="bold cyan")
        tbl_tt.add_column("Trade Type",   style="cyan")
        tbl_tt.add_column("Trades",       justify="right")
        tbl_tt.add_column("Wins",         justify="right")
        tbl_tt.add_column("Win Rate",     justify="right")
        tbl_tt.add_column("Avg P&L",      justify="right")
        tbl_tt.add_column("Total P&L",    justify="right")
        for _, row in wr_tt.iterrows():
            wr_str    = f"{row['win_rate']:.1%}"
            avg_color = "green" if row["avg_pnl"] >= 0 else "red"
            tot_color = "green" if row["total_pnl"] >= 0 else "red"
            tbl_tt.add_row(
                str(row["trade_type"]),
                str(int(row["n_trades"])),
                str(int(row["n_wins"])),
                wr_str,
                f"[{avg_color}]${row['avg_pnl']:+.2f}[/{avg_color}]",
                f"[{tot_color}]${row['total_pnl']:+.2f}[/{tot_color}]",
            )
        console.print(tbl_tt)

    # Edge calibration (only if enough data)
    cal = edge_calibration(closed)
    if not cal.empty and cal["n"].sum() >= 5:
        console.print("[bold]── Calibração de Edge ───────────────────────────[/bold]")
        tbl2 = Table(box=box.SIMPLE, show_header=True, header_style="bold cyan")
        tbl2.add_column("Bin de Edge",      style="dim")
        tbl2.add_column("N",                justify="right")
        tbl2.add_column("Edge Previsto",    justify="right")
        tbl2.add_column("Retorno Realizado",justify="right")
        for _, row in cal.iterrows():
            if row["n"] == 0:
                continue
            diff = row["avg_return"] - row["avg_edge"]
            diff_color = "green" if diff >= 0 else "red"
            tbl2.add_row(
                str(row["edge_bin"]),
                str(int(row["n"])),
                f"{row['avg_edge']:.3f}",
                f"[{diff_color}]{row['avg_return']:.3f}[/{diff_color}]",
            )
        console.print(tbl2)
    elif closed.empty:
        console.print("\n[dim]Aguardando posições fechadas para métricas detalhadas.[/dim]")
