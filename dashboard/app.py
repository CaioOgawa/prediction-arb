"""
dashboard/app.py
Dashboard de monitoramento do Polymarket Quant.

Execução:
    uv run streamlit run dashboard/app.py
    uv run streamlit run dashboard/app.py --server.port 8502
"""

import glob
import pickle
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "ml_lab"))
sys.path.insert(0, str(ROOT / "risk"))

DB_PATH     = ROOT / "data/db/paper_trading.db"
MODELS_DIR  = ROOT / "outputs/models"
REPORTS_DIR = ROOT / "outputs/reports"
LOGS_DIR    = ROOT / "logs"

# ── Constantes de risco — importadas do risk_manager ───────
from risk_manager import (  # noqa: E402
    MAX_POSITION_PCT,
    MAX_CATEGORY_PCT,
    MAX_SOURCE_PCT,
    MAX_OPEN_POSITIONS,
    MAX_MOMENTUM_POS,
    MAX_VALUE_POS,
    WEEKLY_STOP_PCT,
    DAILY_STOP_PCT,
    EARLY_EXIT,
)

st.set_page_config(
    page_title="Polymarket Quant",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Carregamento de dados ───────────────────────────────────

@st.cache_data(ttl=60)
def load_portfolio() -> tuple[pd.DataFrame, dict]:
    """
    Lê posições e dados do portfólio do SQLite.
    Filtra posições pelo portfólio atual: apenas posições abertas APÓS a
    criação do portfólio mais recente (garante que resets não poluam o P&L).
    """
    conn = sqlite3.connect(DB_PATH)
    portfolio = pd.read_sql("SELECT * FROM portfolio ORDER BY id DESC LIMIT 1", conn).iloc[0].to_dict()
    portfolio_start = portfolio.get("created_at", "1970-01-01")
    positions = pd.read_sql(
        "SELECT * FROM positions WHERE opened_at >= ? ORDER BY opened_at DESC",
        conn, params=(portfolio_start,),
    )
    conn.close()
    for col in ["opened_at", "closed_at", "end_date"]:
        if col in positions.columns:
            positions[col] = pd.to_datetime(positions[col], errors="coerce")
    return positions, portfolio


@st.cache_data(ttl=60)
def load_trades_log() -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    portfolio_start = pd.read_sql(
        "SELECT created_at FROM portfolio ORDER BY id DESC LIMIT 1", conn
    ).iloc[0]["created_at"]
    df = pd.read_sql(
        "SELECT * FROM trades_log WHERE timestamp >= ? ORDER BY timestamp DESC",
        conn, params=(portfolio_start,),
    )
    conn.close()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df


@st.cache_data(ttl=10)
def load_price_history(asset_id: str | None = None, last_n: int = 500) -> pd.DataFrame:
    """Lê histórico de ticks do WebSocket feed."""
    conn = sqlite3.connect(DB_PATH)
    try:
        if asset_id:
            df = pd.read_sql(
                "SELECT * FROM price_history WHERE asset_id=? ORDER BY ts DESC LIMIT ?",
                conn, params=(asset_id, last_n),
            )
        else:
            df = pd.read_sql(
                "SELECT * FROM price_history ORDER BY ts DESC LIMIT ?",
                conn, params=(last_n,),
            )
    except Exception:
        df = pd.DataFrame()
    conn.close()
    if not df.empty:
        df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
    return df


@st.cache_data(ttl=10)
def load_early_exits() -> pd.DataFrame:
    """Retorna posições fechadas por early exit (action=EARLY_EXIT no trades_log)."""
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql(
            "SELECT tl.*, p.question, p.entry_price as pos_entry, p.cost_usdc "
            "FROM trades_log tl "
            "LEFT JOIN positions p ON tl.condition_id = p.condition_id "
            "WHERE tl.action = 'EARLY_EXIT' ORDER BY tl.timestamp DESC",
            conn,
        )
    except Exception:
        df = pd.DataFrame()
    conn.close()
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df


@st.cache_data(ttl=5)
def ws_feed_status() -> dict:
    """Métricas detalhadas do ws_feed: ticks/s, latência estimada, status."""
    conn = sqlite3.connect(DB_PATH)
    try:
        row = conn.execute(
            "SELECT COUNT(*) as total, MAX(ts) as last_tick FROM price_history"
        ).fetchone()
        total, last_tick = row if row else (0, None)

        # Ticks no último minuto (proxy de throughput)
        row2 = conn.execute(
            "SELECT COUNT(*) FROM price_history "
            "WHERE ts >= datetime('now', '-60 seconds')"
        ).fetchone()
        ticks_last_min = row2[0] if row2 else 0

        # Latência estimada: diferença entre ts do tick e now
        # (proxy — não é latência de rede, mas staleness)
        row3 = conn.execute(
            "SELECT ts FROM price_history ORDER BY id DESC LIMIT 1"
        ).fetchone()
        last_ts_raw = row3[0] if row3 else None

    except Exception:
        total, last_tick, ticks_last_min, last_ts_raw = 0, None, 0, None
    conn.close()

    alive     = False
    staleness = None
    if last_ts_raw:
        try:
            last_dt = datetime.fromisoformat(last_ts_raw.replace(" ", "T"))
            if last_dt.tzinfo is None:
                last_dt = last_dt.replace(tzinfo=timezone.utc)
            staleness = (datetime.now(timezone.utc) - last_dt).total_seconds()
            alive     = staleness < 30
        except Exception:
            pass

    tps = ticks_last_min / 60 if ticks_last_min else 0.0
    return {
        "total_ticks":    total,
        "last_tick":      last_tick,
        "alive":          alive,
        "ticks_per_sec":  tps,
        "staleness_s":    staleness,
        "ticks_last_min": ticks_last_min,
    }


@st.cache_data(ttl=5)
def load_position_price_detail(condition_id: str) -> dict:
    """Para uma posição aberta, retorna último bid/ask/mid do price_history."""
    conn = sqlite3.connect(DB_PATH)
    try:
        row = conn.execute(
            "SELECT best_bid, best_ask, mid, spread, ts FROM price_history "
            "WHERE condition_id=? ORDER BY id DESC LIMIT 1",
            (condition_id,),
        ).fetchone()
    except Exception:
        row = None
    conn.close()
    if row:
        return {"best_bid": row[0], "best_ask": row[1], "mid": row[2],
                "spread": row[3], "ts": row[4]}
    return {}


@st.cache_data(ttl=300)
def load_latest_signals() -> pd.DataFrame:
    """Lê o CSV de sinais mais recente."""
    files = sorted(glob.glob(str(REPORTS_DIR / "signals_*.csv")))
    if not files:
        return pd.DataFrame()
    return pd.read_csv(files[-1])


@st.cache_data(ttl=300)
def load_latest_markets() -> pd.DataFrame:
    files = sorted(glob.glob(str(REPORTS_DIR / "top_markets_*.csv")))
    if not files:
        return pd.DataFrame()
    return pd.read_csv(files[-1])


@st.cache_data(ttl=600)
def load_model_bundle() -> dict | None:
    latest = MODELS_DIR / "best_model_latest.pkl"
    if not latest.exists():
        return None
    with open(latest, "rb") as f:
        return pickle.load(f)


@st.cache_data(ttl=600)
def load_model_comparison() -> tuple[pd.DataFrame, str]:
    """Carrega o resultado mais recente: walk-forward CV (wf_cv_*.csv) ou random split."""
    wf_files  = sorted(glob.glob(str(ROOT / "ml_lab/results/wf_cv_*.csv")))
    rnd_files = sorted(glob.glob(str(ROOT / "ml_lab/results/model_comparison_*.csv")))

    if wf_files:
        return pd.read_csv(wf_files[-1]), "walk_forward"
    if rnd_files:
        return pd.read_csv(rnd_files[-1]), "random_split"
    return pd.DataFrame(), "none"


@st.cache_data(ttl=60)
def load_log_tail(log_file: str, n: int = 100) -> str:
    path = LOGS_DIR / log_file
    if not path.exists():
        return f"[arquivo não encontrado: {path}]"
    lines = path.read_text(errors="replace").splitlines()
    return "\n".join(lines[-n:])


def compute_pnl_curve(positions: pd.DataFrame, initial_capital: float) -> pd.DataFrame:
    """Reconstrói curva de P&L a partir das posições fechadas."""
    closed = positions[positions["status"] == "closed"].copy()
    if closed.empty:
        return pd.DataFrame(columns=["date", "cumulative_pnl", "capital"])
    closed = closed.sort_values("closed_at")
    closed["cumulative_pnl"] = closed["pnl_usdc"].cumsum()
    closed["capital"] = initial_capital + closed["cumulative_pnl"]
    return closed[["closed_at", "cumulative_pnl", "capital"]].rename(columns={"closed_at": "date"})


def compute_exposure(positions: pd.DataFrame, current_cash: float) -> dict:
    """Calcula exposição atual por categoria e fonte."""
    open_pos = positions[positions["status"] == "open"]
    total_capital = current_cash + open_pos["cost_usdc"].sum()
    by_category = open_pos.groupby("category")["cost_usdc"].sum()
    by_direction = open_pos.groupby("direction")["cost_usdc"].sum()
    return {
        "total_capital": total_capital,
        "total_exposure": open_pos["cost_usdc"].sum(),
        "by_category": by_category,
        "by_direction": by_direction,
        "open_count": len(open_pos),
    }


# ── ML ao vivo ─────────────────────────────────────────────

def _extract_features(m: dict, category: str = "other") -> dict:
    """Extrai features agnósticas ao outcome de um mercado ativo (sem resolver)."""
    volume    = float(m.get("volume", 0) or 0)
    liquidity = float(m.get("liquidity", 0) or 0)
    v24h      = float(m.get("volume24hr", 0) or 0)
    v1wk      = float(m.get("volume1wk", 0) or 0)

    start = pd.to_datetime(m.get("createdAt") or m.get("startDate"), errors="coerce", utc=True)
    end   = pd.to_datetime(m.get("endDate"), errors="coerce", utc=True)
    days_ran = int((end - start).days) if pd.notna(start) and pd.notna(end) else 30

    avg_daily = volume / max(days_ran, 1)
    recency_ratio = float(np.clip(v24h / (avg_daily + 1e-9), 0, 100))
    spread = float(m.get("spread", 1.0) or 1.0)

    KNOWN_CATEGORIES = [
        "sports", "politics", "crypto", "economics", "pop-culture",
        "science", "technology", "weather", "entertainment", "other",
    ]
    norm_cat = category.lower().strip() if category else "other"
    if norm_cat not in KNOWN_CATEGORIES:
        norm_cat = "other"

    row = {
        "log_volume":            float(np.log1p(volume)),
        "log_liquidity":         float(np.log1p(liquidity)),
        "log_volume24hr":        float(np.log1p(v24h)),
        "log_volume1wk":         float(np.log1p(v1wk)),
        "volume_recency_ratio":  recency_ratio,
        "spread":                spread,
        "days_ran":              max(days_ran, 0),
        "has_resolution_source": int(bool(m.get("resolutionSource"))),
    }
    for cat in KNOWN_CATEGORIES:
        row[f"category_{cat}"] = int(norm_cat == cat)
    return row


def _parse_yes_price(raw) -> float:
    """Extrai yes_price de outcomePrices (string JSON ou lista)."""
    import json as _json
    if isinstance(raw, str):
        try:
            raw = _json.loads(raw)
        except Exception:
            return 0.5
    if hasattr(raw, "__len__") and len(raw) > 0:
        return float(raw[0])
    return 0.5


@st.cache_data(ttl=300)
def run_live_predictions(n_markets: int = 50) -> pd.DataFrame:
    """Roda predições do modelo nos mercados ativos mais líquidos."""
    bundle = load_model_bundle()
    if bundle is None:
        return pd.DataFrame()

    files = sorted(glob.glob(str(ROOT / "data/raw/markets/markets_all_*.parquet")))
    if not files:
        return pd.DataFrame()

    markets = pd.read_parquet(files[-1])
    active = markets[markets["active"] == True].copy()  # noqa: E712
    active = active.nlargest(n_markets, "liquidity").reset_index(drop=True)

    feature_names = bundle.get("feature_names", [])
    rows = [_extract_features(r.to_dict(), r.get("category", "other"))
            for _, r in active.iterrows()]
    feat_df = pd.DataFrame(rows)

    for col in feature_names:
        if col not in feat_df.columns:
            feat_df[col] = 0.0
    X = feat_df[feature_names].fillna(0.0)

    scaler = bundle.get("scaler")
    if scaler is not None:
        X = pd.DataFrame(scaler.transform(X), columns=feature_names)

    probs = bundle["model"].predict_proba(X)[:, 1]

    result = active[["question", "category", "liquidity"]].copy()
    result["yes_price"] = active["outcomePrices"].apply(_parse_yes_price)
    result["model_prob"] = probs
    result["edge"] = result["model_prob"] - result["yes_price"]
    result["abs_edge"] = result["edge"].abs()
    return result.sort_values("abs_edge", ascending=False).head(20)


# ── Sidebar ─────────────────────────────────────────────────

with st.sidebar:
    st.title("Polymarket Quant")
    st.caption(f"Atualizado: {datetime.now(timezone.utc).strftime('%H:%M UTC')}")

    if st.button("Forçar atualização", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.divider()
    ws = ws_feed_status()
    ws_color = "🟢" if ws["alive"] else "🔴"
    st.markdown(f"**WebSocket Feed** {ws_color}")
    tps = ws.get("ticks_per_sec", 0)
    stale = ws.get("staleness_s")
    stale_str = f"{stale:.1f}s atrás" if stale is not None else "—"
    st.caption(
        f"{ws['total_ticks']:,} ticks totais\n"
        f"{tps:.1f} ticks/s | último: {stale_str}"
    )
    if not ws["alive"]:
        st.warning("Feed inativo.\n`launchctl start com.caio.polymarket-ws-feed`")
    st.divider()
    auto_refresh = st.toggle("Auto-refresh (60s)", value=False)
    st.divider()
    page = st.radio(
        "Navegação",
        ["Visão Geral", "Portfolio", "Feed ao Vivo", "Risco", "Sinais", "Mercados", "ML / Modelo", "Logs"],
        label_visibility="collapsed",
    )

# ── Carrega dados ───────────────────────────────────────────
positions, portfolio_meta = load_portfolio()

initial_capital = float(portfolio_meta.get("initial_capital", 1000.0))
current_cash    = float(portfolio_meta.get("current_cash", initial_capital))
open_pos        = positions[positions["status"] == "open"]
closed_pos      = positions[positions["status"] == "closed"]

total_value   = current_cash + open_pos["current_value"].sum() if "current_value" in open_pos.columns else current_cash + open_pos["cost_usdc"].sum()
pnl_realized  = closed_pos["pnl_usdc"].sum()
pnl_unrealized = open_pos["unrealized_pnl"].sum() if "unrealized_pnl" in open_pos.columns else 0.0
pnl_total     = pnl_realized + pnl_unrealized
pnl_pct       = pnl_total / initial_capital * 100

win_rate = (closed_pos["pnl_usdc"] > 0).mean() if len(closed_pos) > 0 else float("nan")
exposure  = compute_exposure(positions, current_cash)


# ════════════════════════════════════════════════════════════
# VISÃO GERAL
# ════════════════════════════════════════════════════════════
if page == "Visão Geral":
    st.header("Visão Geral")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Capital Total", f"${total_value:,.2f}", f"${pnl_total:+.2f}")
    c2.metric("P&L Total", f"${pnl_total:+.2f}", f"{pnl_pct:+.2f}%")
    c3.metric("P&L Realizado", f"${pnl_realized:+.2f}")
    c4.metric("Posições Abertas", f"{len(open_pos)}", f"/ {MAX_OPEN_POSITIONS} máx")
    c5.metric("Win Rate", f"{win_rate:.1%}" if not np.isnan(win_rate) else "—", f"{len(closed_pos)} trades")

    st.divider()

    col_chart, col_alloc = st.columns([2, 1])

    with col_chart:
        st.subheader("Curva de P&L")
        curve = compute_pnl_curve(positions, initial_capital)
        if not curve.empty:
            fig = px.line(
                curve, x="date", y="capital",
                labels={"capital": "Capital (USDC)", "date": ""},
                template="plotly_dark",
            )
            fig.add_hline(y=initial_capital, line_dash="dot", line_color="gray",
                          annotation_text="Capital inicial")
            fig.update_layout(margin=dict(l=0, r=0, t=20, b=0), height=280)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Nenhuma posição fechada ainda — a curva aparecerá após os primeiros outcomes.")

    with col_alloc:
        st.subheader("Alocação atual")
        if not open_pos.empty and "cost_usdc" in open_pos.columns:
            by_cat = open_pos.groupby("category")["cost_usdc"].sum().reset_index()
            by_cat.columns = ["Categoria", "USDC"]
            fig2 = px.pie(by_cat, values="USDC", names="Categoria",
                          template="plotly_dark", hole=0.4)
            fig2.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=280,
                               showlegend=True, legend=dict(font_size=11))
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Sem posições abertas.")

    # Posições abertas com múltiplos de exit em tempo real
    st.subheader("Posições Abertas")
    if not open_pos.empty:
        rows_enriched = []
        for _, pos in open_pos.iterrows():
            live = load_position_price_detail(str(pos["condition_id"]))
            row  = {
                "Questão":      str(pos.get("question", ""))[:45],
                "Dir.":         pos.get("direction", ""),
                "Entrada":      f"{float(pos['entry_price']):.3f}",
                "Bid Live":     f"{live['best_bid']:.3f}"  if live.get("best_bid") else "—",
                "Ask Live":     f"{live['best_ask']:.3f}"  if live.get("best_ask") else "—",
                "Spread":       f"{live['spread']:.3f}"    if live.get("spread")   else "—",
                "Custo $":      f"${float(pos['cost_usdc']):.2f}",
                "Múltiplo":     None,
                "P&L Atual":    None,
                "Edge Entry":   f"{float(pos.get('edge_at_entry', 0)):.3f}",
                "Expira":       str(pos.get("end_date", ""))[:10],
            }
            if live.get("best_bid") and live.get("best_ask"):
                direction  = str(pos.get("direction", ""))
                spread_val = live["best_ask"] - live["best_bid"]
                exit_px    = max(live["best_bid"] - spread_val / 2, 0.001)
                shares     = float(pos.get("shares", 0))
                cost       = float(pos.get("cost_usdc", 1))
                cur_val    = exit_px * shares
                mult       = cur_val / cost if cost > 0 else 0
                pnl        = cur_val - cost
                row["Múltiplo"] = f"{mult:.2f}×"
                row["P&L Atual"] = f"${pnl:+.2f}"
            rows_enriched.append(row)

        df_live = pd.DataFrame(rows_enriched)

        # Coloração inline via markdown nas colunas chave
        st.dataframe(df_live, use_container_width=True, height=280)

        # Gauge de proximidade dos gatilhos de exit
        st.caption("Proximidade dos gatilhos de early exit (baseado em preço ao vivo)")
        for _, pos in open_pos.iterrows():
            live = load_position_price_detail(str(pos["condition_id"]))
            if not live.get("best_bid"):
                continue
            cost    = float(pos["cost_usdc"])
            shares  = float(pos["shares"])
            bid     = live["best_bid"]
            ask     = live["best_ask"]
            spread_v = ask - bid
            exit_px  = max(bid - spread_v / 2, 0.001)
            cur_val  = exit_px * shares
            mult     = cur_val / cost if cost > 0 else 0

            profit_pct = min(mult / 2.0, 1.0)   # 2× = 100%

            q_short = str(pos.get("question", ""))[:35]
            col1, col2 = st.columns([2, 1])
            col1.caption(q_short)
            col2.progress(profit_pct, text=f"Profit {mult:.1f}×/2×")
    else:
        st.info("Nenhuma posição aberta.")


# ════════════════════════════════════════════════════════════
# PORTFOLIO
# ════════════════════════════════════════════════════════════
elif page == "Portfolio":
    st.header("Portfolio")

    tab_open, tab_closed, tab_exits, tab_prices, tab_log = st.tabs(
        ["Abertas", "Fechadas", "Early Exits", "Preços ao Vivo", "Trade Log"]
    )

    with tab_open:
        if not open_pos.empty:
            st.dataframe(open_pos.reset_index(drop=True), use_container_width=True)
        else:
            st.info("Nenhuma posição aberta.")

    with tab_closed:
        if not closed_pos.empty:
            df = closed_pos.copy()
            df["pnl_pct"] = df["pnl_usdc"] / df["cost_usdc"] * 100
            fig = px.bar(
                df.sort_values("closed_at"),
                x="question", y="pnl_usdc",
                color="pnl_usdc", color_continuous_scale="RdYlGn",
                labels={"pnl_usdc": "P&L (USDC)", "question": ""},
                template="plotly_dark",
            )
            fig.update_layout(height=300, xaxis_tickangle=-30)
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(df.reset_index(drop=True), use_container_width=True)
        else:
            st.info("Nenhuma posição fechada ainda.")

    with tab_exits:
        early_exits_df = load_early_exits()
        if early_exits_df.empty:
            st.info("Nenhum early exit registrado ainda.")
        else:
            # Extrai trigger type do campo note
            early_exits_df["trigger_type"] = early_exits_df["note"].str.extract(
                r"^(profit_target|stop_loss|edge_flip_\w+)"
            ).fillna("outro")

            col_a, col_b, col_c = st.columns(3)
            total_exits = len(early_exits_df)
            pnl_exits   = early_exits_df["usdc_amount"].sum() - early_exits_df.get("cost_usdc", pd.Series(0)).sum()
            wins        = (early_exits_df["usdc_amount"] > early_exits_df.get("cost_usdc", 0)).sum()
            col_a.metric("Total Early Exits", total_exits)
            col_b.metric("P&L Early Exits", f"${pnl_exits:+.2f}")
            col_c.metric("Win Rate", f"{wins/total_exits:.0%}" if total_exits else "—")

            col_pie, col_bar = st.columns(2)
            with col_pie:
                trigger_counts = early_exits_df["trigger_type"].value_counts().reset_index()
                trigger_counts.columns = ["Trigger", "N"]
                fig_t = px.pie(trigger_counts, values="N", names="Trigger",
                               title="Exits por Gatilho", template="plotly_dark", hole=0.4)
                fig_t.update_layout(height=280, margin=dict(l=0, r=0, t=40, b=0))
                st.plotly_chart(fig_t, use_container_width=True)

            with col_bar:
                eb = early_exits_df.copy()
                eb["pnl"] = eb["usdc_amount"] - eb.get("cost_usdc", pd.Series(0))
                fig_b = px.bar(
                    eb.sort_values("timestamp"),
                    x="timestamp", y="pnl",
                    color="trigger_type",
                    labels={"pnl": "P&L (USDC)", "timestamp": ""},
                    title="P&L por Early Exit",
                    template="plotly_dark",
                )
                fig_b.update_layout(height=280, margin=dict(l=0, r=0, t=40, b=0))
                st.plotly_chart(fig_b, use_container_width=True)

            st.dataframe(
                early_exits_df[["timestamp", "question", "direction", "price", "note"]]
                .rename(columns={"price": "exit_price", "note": "trigger"}),
                use_container_width=True,
            )

    with tab_prices:
        ws_status = ws_feed_status()
        if not ws_status["alive"]:
            st.warning(
                "WebSocket feed não está ativo. Inicie o daemon para ver preços em tempo real:\n\n"
                "```\nlaunchctl start com.caio.polymarket-ws-feed\n```"
            )

        # Lista de mercados com dados no price_history
        conn = sqlite3.connect(DB_PATH)
        try:
            tracked = pd.read_sql(
                "SELECT DISTINCT asset_id, condition_id, COUNT(*) as n_ticks, "
                "MAX(ts) as last_tick, MAX(mid) as max_mid, MIN(mid) as min_mid "
                "FROM price_history GROUP BY asset_id ORDER BY last_tick DESC LIMIT 30",
                conn,
            )
        except Exception:
            tracked = pd.DataFrame()
        conn.close()

        if tracked.empty:
            st.info("Nenhum dado de preço coletado ainda. O feed WebSocket precisa estar rodando.")
        else:
            st.caption(f"{ws_status['total_ticks']:,} ticks coletados | último: {ws_status['last_tick']}")
            sel_asset = st.selectbox(
                "Selecionar mercado para ver histórico",
                tracked["asset_id"].tolist(),
                format_func=lambda a: tracked[tracked["asset_id"]==a]["condition_id"].iloc[0][:30]
                    + f" ({tracked[tracked['asset_id']==a]['n_ticks'].iloc[0]} ticks)",
            )

            if sel_asset:
                history = load_price_history(sel_asset, last_n=1000)
                if not history.empty:
                    history = history.sort_values("ts")
                    fig_p = go.Figure()
                    if "best_bid" in history.columns:
                        fig_p.add_trace(go.Scatter(
                            x=history["ts"], y=history["best_bid"],
                            name="Bid", line=dict(color="#2ecc71", width=1),
                        ))
                    if "best_ask" in history.columns:
                        fig_p.add_trace(go.Scatter(
                            x=history["ts"], y=history["best_ask"],
                            name="Ask", line=dict(color="#e74c3c", width=1),
                        ))
                    if "mid" in history.columns:
                        fig_p.add_trace(go.Scatter(
                            x=history["ts"], y=history["mid"],
                            name="Mid", line=dict(color="#f39c12", width=2),
                        ))
                    fig_p.update_layout(
                        template="plotly_dark", height=380,
                        margin=dict(l=0, r=0, t=20, b=0),
                        yaxis=dict(title="Preço", tickformat=".3f"),
                        xaxis=dict(title=""),
                        legend=dict(orientation="h"),
                    )
                    st.plotly_chart(fig_p, use_container_width=True)

                    # Spread ao longo do tempo
                    if "spread" in history.columns:
                        fig_s = px.line(
                            history, x="ts", y="spread",
                            labels={"spread": "Spread", "ts": ""},
                            template="plotly_dark",
                            title="Spread bid-ask",
                        )
                        fig_s.update_layout(height=200, margin=dict(l=0, r=0, t=30, b=0))
                        st.plotly_chart(fig_s, use_container_width=True)

    with tab_log:
        trades = load_trades_log()
        if not trades.empty:
            action_filter = st.multiselect(
                "Filtrar por ação",
                trades["action"].unique().tolist(),
                default=trades["action"].unique().tolist(),
            )
            st.dataframe(
                trades[trades["action"].isin(action_filter)],
                use_container_width=True,
            )
        else:
            st.info("Log vazio.")


# ════════════════════════════════════════════════════════════
# FEED AO VIVO
# ════════════════════════════════════════════════════════════
elif page == "Feed ao Vivo":
    st.header("Feed ao Vivo")

    ws = ws_feed_status()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Status", "🟢 Ativo" if ws["alive"] else "🔴 Inativo")
    c2.metric("Throughput", f"{ws.get('ticks_per_sec', 0):.1f} ticks/s")
    c3.metric("Total Ticks", f"{ws.get('total_ticks', 0):,}")
    stale = ws.get("staleness_s")
    c4.metric("Staleness", f"{stale:.2f}s" if stale is not None else "—",
              delta="ok" if (stale or 99) < 5 else "alto", delta_color="normal" if (stale or 99) < 5 else "inverse")

    st.divider()

    # Últimos ticks em tempo real
    st.subheader("Últimos Ticks")
    recent = load_price_history(last_n=50)
    if not recent.empty:
        recent_display = recent.sort_values("ts", ascending=False).head(20)
        # Calcula spread em bps
        if "spread" in recent_display.columns and "mid" in recent_display.columns:
            recent_display = recent_display.copy()
            recent_display["spread_bps"] = (recent_display["spread"] / recent_display["mid"].replace(0, float("nan")) * 10000).round(0)
        show = [c for c in ["ts", "condition_id", "best_bid", "best_ask", "mid", "spread_bps", "side", "size"] if c in recent_display.columns]
        st.dataframe(recent_display[show].reset_index(drop=True), use_container_width=True, height=350)
    else:
        st.info("Nenhum tick coletado ainda.")

    st.divider()

    # Posições com monitoramento em tempo real e distância dos gatilhos
    st.subheader("Posições — Distância dos Gatilhos")
    if not open_pos.empty:
        trigger_data = []
        for _, pos in open_pos.iterrows():
            live = load_position_price_detail(str(pos["condition_id"]))
            cost    = float(pos["cost_usdc"])
            shares  = float(pos["shares"])
            entry   = float(pos["entry_price"])
            direction = str(pos.get("direction", ""))

            if live.get("best_bid") and live.get("best_ask"):
                bid, ask = live["best_bid"], live["best_ask"]
                spread_v = ask - bid
                exit_px  = max(bid - spread_v / 2, 0.001)
                cur_val  = exit_px * shares
                mult     = cur_val / cost if cost > 0 else 0
                pnl      = cur_val - cost

                yes_impl = (1.0 - bid) if direction == "BUY_NO" else bid

                # Distância percentual de cada gatilho
                dist_profit = (2.0 - mult) / 2.0     # 0 = no gatilho

                # Edge flip relativo ao entry — delta vem do EARLY_EXIT por trade_type
                # (fix 2026-07: EDGE_FLIP_DELTA solto não existia → NameError nesta aba)
                # BUY_NO:  entry_yes = 1 - entry_price; gatilho em entry_yes + delta
                # BUY_YES: entry_yes = entry_price;     gatilho em entry_yes - delta
                entry_yes = (1.0 - entry) if direction == "BUY_NO" else entry
                _tt = str(pos.get("trade_type") or "value")
                flip_delta = float(EARLY_EXIT.get(_tt, {}).get("edge_flip_delta", 0.20))
                if direction == "BUY_NO":
                    flip_trigger = entry_yes + flip_delta
                    dist_edge = (flip_trigger - yes_impl) / flip_delta
                else:
                    flip_trigger = entry_yes - flip_delta
                    dist_edge = (yes_impl - flip_trigger) / flip_delta

                trigger_data.append({
                    "Questão":        str(pos.get("question", ""))[:40],
                    "Dir.":           direction,
                    "Múltiplo":       mult,
                    "P&L $":          pnl,
                    "YES impl.":      yes_impl,
                    "Spread (bps)":   round(spread_v / ((bid + ask) / 2) * 10000) if (bid + ask) > 0 else None,
                    "% p/ Profit":    max(dist_profit, 0),
                    "% p/ EdgeFlip":  max(1 - dist_edge, 0) if dist_edge is not None else 0,
                    "Último tick":    str(live.get("ts", ""))[:19],
                })
            else:
                trigger_data.append({
                    "Questão": str(pos.get("question", ""))[:40],
                    "Dir.": direction,
                    "Múltiplo": None, "P&L $": None, "YES impl.": None,
                    "Spread (bps)": None, "% p/ Profit": None,
                    "% p/ EdgeFlip": None,
                    "Último tick": "sem dados",
                })

        df_trig = pd.DataFrame(trigger_data)

        # Gráfico de múltiplos
        valid = df_trig[df_trig["Múltiplo"].notna()].copy()
        if not valid.empty:
            fig = go.Figure()
            fig.add_bar(
                x=valid["Questão"], y=valid["Múltiplo"],
                name="Múltiplo atual",
                marker_color=["#2ecc71" if m >= 1 else "#e74c3c" for m in valid["Múltiplo"]],
            )
            fig.add_hline(y=2.0, line_dash="dash", line_color="gold",
                          annotation_text="Profit target 2×")
            fig.add_hline(y=1.0, line_dash="dot", line_color="gray",
                          annotation_text="Break-even")
            fig.update_layout(
                template="plotly_dark", height=350,
                margin=dict(l=0, r=0, t=20, b=80),
                xaxis_tickangle=-20,
                yaxis_title="Múltiplo (valor atual / custo)",
            )
            st.plotly_chart(fig, use_container_width=True)

        st.dataframe(df_trig, use_container_width=True)
    else:
        st.info("Nenhuma posição aberta.")

    # Throughput histórico
    st.divider()
    st.subheader("Throughput do Feed (últimas 500 entradas)")
    hist = load_price_history(last_n=500)
    if not hist.empty and "ts" in hist.columns:
        hist = hist.sort_values("ts")
        hist["minuto"] = hist["ts"].dt.floor("min")
        by_min = hist.groupby("minuto").size().reset_index(name="ticks")
        fig2 = px.bar(by_min, x="minuto", y="ticks",
                      labels={"minuto": "", "ticks": "Ticks/min"},
                      template="plotly_dark", title="Ticks por minuto")
        fig2.update_layout(height=220, margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig2, use_container_width=True)


# ════════════════════════════════════════════════════════════
# RISCO
# ════════════════════════════════════════════════════════════
elif page == "Risco":
    st.header("Gestão de Risco")

    total_cap = exposure["total_capital"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Capital em risco", f"${exposure['total_exposure']:,.2f}",
              f"{exposure['total_exposure']/total_cap:.1%} do portfólio")
    c2.metric("Posições abertas", f"{exposure['open_count']}", f"/ {MAX_OPEN_POSITIONS} máx")
    c3.metric("Drawdown realizado", f"{abs(pnl_realized/initial_capital):.2%}",
              delta=f"limite semanal: {WEEKLY_STOP_PCT:.0%}", delta_color="off")
    c4.metric("Cash disponível", f"${current_cash:,.2f}",
              f"{current_cash/total_cap:.1%} do portfólio")

    st.divider()

    col_cat, col_src = st.columns(2)

    with col_cat:
        st.subheader("Exposição por Categoria")
        if not open_pos.empty:
            by_cat = open_pos.groupby("category")["cost_usdc"].sum().reset_index()
            by_cat["pct"] = by_cat["cost_usdc"] / total_cap
            by_cat["limite"] = MAX_CATEGORY_PCT
            by_cat["status"] = by_cat["pct"].apply(
                lambda x: "⚠️ Acima" if x > MAX_CATEGORY_PCT else "✅ OK"
            )
            fig = go.Figure()
            fig.add_bar(x=by_cat["category"], y=by_cat["pct"],
                        name="Atual", marker_color="#4c9be8")
            fig.add_hline(y=MAX_CATEGORY_PCT, line_dash="dash", line_color="red",
                          annotation_text=f"Limite {MAX_CATEGORY_PCT:.0%}")
            fig.update_layout(
                yaxis_tickformat=".0%", template="plotly_dark",
                height=300, margin=dict(l=0, r=0, t=20, b=0),
                yaxis_title="% do capital",
            )
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(by_cat[["category", "cost_usdc", "pct", "status"]]
                         .rename(columns={"cost_usdc": "USDC", "pct": "% capital", "category": "Categoria"}),
                         use_container_width=True)
        else:
            st.info("Sem posições abertas.")

    with col_src:
        st.subheader("Exposição por Direção")
        if not open_pos.empty:
            by_dir = open_pos.groupby("direction")["cost_usdc"].sum().reset_index()
            by_dir["pct"] = by_dir["cost_usdc"] / total_cap
            fig2 = px.bar(by_dir, x="direction", y="pct",
                          labels={"pct": "% do capital", "direction": ""},
                          template="plotly_dark", color="direction",
                          color_discrete_map={"BUY_YES": "#2ecc71", "BUY_NO": "#e74c3c"})
            fig2.update_layout(height=300, margin=dict(l=0, r=0, t=20, b=0),
                               yaxis_tickformat=".0%", showlegend=False)
            fig2.add_hline(y=MAX_SOURCE_PCT, line_dash="dash", line_color="red",
                           annotation_text=f"Limite {MAX_SOURCE_PCT:.0%}")
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Sem posições abertas.")

    st.subheader("Parâmetros de Risco Ativos")
    params = {
        "Kelly máximo":                  "25% (quarter-Kelly)",
        "Max por posição — momentum":    f"{MAX_POSITION_PCT['momentum']:.1%} do capital",
        "Max por posição — value":       f"{MAX_POSITION_PCT['value']:.1%} do capital",
        "Max por categoria":             f"{MAX_CATEGORY_PCT:.0%} do capital",
        "Max por fonte de sinal":        f"{MAX_SOURCE_PCT:.0%} do capital",
        "Max posições simultâneas":      f"{MAX_OPEN_POSITIONS} (momentum ≤ {MAX_MOMENTUM_POS}, value ≤ {MAX_VALUE_POS})",
        "Stop semanal":                  f"{WEEKLY_STOP_PCT:.0%} drawdown",
        "Stop diário":                   f"{DAILY_STOP_PCT:.0%} drawdown",
        "Edge mínimo":                   "8%",
        "Profit target — momentum":      f"{EARLY_EXIT['momentum']['profit_target_mult']}× custo",
        "Profit target — value":         f"{EARLY_EXIT['value']['profit_target_mult']}× custo",
        "Edge flip — momentum":          f"{EARLY_EXIT['momentum']['edge_flip_delta']:.0%} desde entry",
        "Edge flip — value":             f"{EARLY_EXIT['value']['edge_flip_delta']:.0%} desde entry",
        "Hold mínimo — momentum":        f"{EARLY_EXIT['momentum']['min_hold_hours']:.0f}h",
        "Hold mínimo — value":           f"{EARLY_EXIT['value']['min_hold_hours']:.0f}h",
        "Deribit moneyness máx":         "1.5 × σ√T (B-S válido apenas near-ATM)",
        "Deribit BUY_NO veto":           "bloqueado quando YES > 70% no mercado",
    }
    st.table(pd.DataFrame(list(params.items()), columns=["Parâmetro", "Valor"]))


# ════════════════════════════════════════════════════════════
# SINAIS
# ════════════════════════════════════════════════════════════
elif page == "Sinais":
    st.header("Sinais")

    signals = load_latest_signals()

    if signals.empty:
        st.info("Nenhum arquivo de sinais encontrado em outputs/reports/.")
    else:
        files = sorted(glob.glob(str(REPORTS_DIR / "signals_*.csv")))
        st.caption(f"Fonte: `{Path(files[-1]).name}` — {len(signals)} sinais")

        col_f1, col_f2, col_f3, col_f4 = st.columns(4)
        with col_f1:
            dirs = ["Todos"] + sorted(signals["direction"].dropna().unique().tolist()) if "direction" in signals.columns else ["Todos"]
            sel_dir = st.selectbox("Direção", dirs)
        with col_f2:
            cats = ["Todas"] + sorted(signals["category"].dropna().unique().tolist()) if "category" in signals.columns else ["Todas"]
            sel_cat = st.selectbox("Categoria", cats)
        with col_f3:
            sources = ["Todas"] + sorted(signals["signal_source"].dropna().unique().tolist()) if "signal_source" in signals.columns else ["Todas"]
            sel_src = st.selectbox("Fonte", sources)
        with col_f4:
            mktypes = ["Todos"] + sorted(signals["market_type"].dropna().unique().tolist()) if "market_type" in signals.columns else ["Todos"]
            sel_mkt = st.selectbox("Tipo Mercado", mktypes)

        df = signals.copy()
        if "direction" in df.columns and sel_dir != "Todos":
            df = df[df["direction"] == sel_dir]
        if "category" in df.columns and sel_cat != "Todas":
            df = df[df["category"] == sel_cat]
        if "signal_source" in df.columns and sel_src != "Todas":
            df = df[df["signal_source"] == sel_src]
        if "market_type" in df.columns and sel_mkt != "Todos":
            df = df[df["market_type"] == sel_mkt]

        if "edge" in df.columns:
            fig = px.histogram(df, x="edge", nbins=30, template="plotly_dark",
                               labels={"edge": "Edge", "count": "N"},
                               title="Distribuição de Edge")
            fig.add_vline(x=0.08, line_dash="dash", line_color="yellow",
                          annotation_text="Mín. 8%")
            fig.update_layout(height=250, margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig, use_container_width=True)

        st.dataframe(df.reset_index(drop=True), use_container_width=True)


# ════════════════════════════════════════════════════════════
# MERCADOS
# ════════════════════════════════════════════════════════════
elif page == "Mercados":
    st.header("Top Mercados")

    markets = load_latest_markets()
    files = sorted(glob.glob(str(REPORTS_DIR / "top_markets_*.csv")))
    if markets.empty:
        st.info("Nenhum arquivo de mercados encontrado.")
    else:
        st.caption(f"Fonte: `{Path(files[-1]).name}` — {len(markets)} mercados")

        col_search, col_cat = st.columns(2)
        with col_search:
            query = st.text_input("Filtrar por nome", "")
        with col_cat:
            if "category" in markets.columns:
                cats = ["Todas"] + sorted(markets["category"].dropna().unique().tolist())
                sel_cat = st.selectbox("Categoria", cats, key="mkt_cat")

        df = markets.copy()
        if query:
            df = df[df["question"].str.contains(query, case=False, na=False)]
        if "category" in df.columns and sel_cat != "Todas":
            df = df[df["category"] == sel_cat]

        if "yes_price" in df.columns and "spread" in df.columns:
            fig = px.scatter(
                df, x="yes_price", y="spread",
                size="volume" if "volume" in df.columns else None,
                color="category" if "category" in df.columns else None,
                hover_data=["question"] if "question" in df.columns else None,
                labels={"yes_price": "Preço YES", "spread": "Spread"},
                template="plotly_dark",
                title="Preço YES vs Spread",
            )
            fig.update_layout(height=320, margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig, use_container_width=True)

        st.dataframe(df.reset_index(drop=True), use_container_width=True)


# ════════════════════════════════════════════════════════════
# ML / MODELO
# ════════════════════════════════════════════════════════════
elif page == "ML / Modelo":
    st.header("ML / Modelo")

    bundle = load_model_bundle()
    comparison, cmp_type = load_model_comparison()

    if bundle is None:
        st.warning("Nenhum modelo treinado encontrado. Execute: `uv run python ml_lab/run_lab.py`")
    else:
        # Métricas do modelo ativo
        st.subheader("Modelo Ativo")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Modelo", bundle.get("model_name", "—"))
        c2.metric("AUC-ROC", f"{bundle.get('auc_roc', 0):.4f}")
        c3.metric("Brier Score", f"{bundle.get('brier_score', 0):.4f}")
        c4.metric("Treinado em", str(bundle.get("trained_at", "—"))[:19])

        st.caption(f"Features: `{', '.join(bundle.get('feature_names', []))}`")

    st.divider()

    # Comparação de modelos — suporta walk-forward CV (média ± std) e random split
    if not comparison.empty:
        is_wf = cmp_type == "walk_forward"
        label = "Walk-Forward CV (média ± std)" if is_wf else "Random Split"
        st.subheader(f"Comparação de Modelos — {label}")

        if is_wf:
            # Colunas com sufixo _mean/_std
            auc_col = "auc_roc_mean" if "auc_roc_mean" in comparison.columns else None
            if auc_col:
                fig = go.Figure()
                fig.add_bar(
                    x=comparison["model"],
                    y=comparison["auc_roc_mean"],
                    error_y=dict(type="data", array=comparison.get("auc_roc_std", [0]*len(comparison)).tolist()),
                    name="AUC-ROC",
                    marker_color="#4c9be8",
                )
                fig.add_bar(
                    x=comparison["model"],
                    y=comparison.get("accuracy_mean", []),
                    error_y=dict(type="data", array=comparison.get("accuracy_std", [0]*len(comparison)).tolist()),
                    name="Accuracy",
                    marker_color="#2ecc71",
                )
                fig.update_layout(
                    barmode="group", template="plotly_dark", height=350,
                    margin=dict(l=0, r=0, t=20, b=0),
                    yaxis=dict(range=[0.4, 0.85], title="Score"),
                )
                st.plotly_chart(fig, use_container_width=True)
            st.caption("Barras de erro = desvio padrão entre folds. Modelo com menor desvio = mais estável.")
        else:
            fig = go.Figure()
            metrics = [c for c in ["auc_roc", "accuracy", "f1_weighted"] if c in comparison.columns]
            for m in metrics:
                fig.add_bar(x=comparison["model"], y=comparison[m], name=m)
            fig.update_layout(
                barmode="group", template="plotly_dark", height=350,
                margin=dict(l=0, r=0, t=20, b=0),
                yaxis=dict(range=[0.4, 0.8]),
            )
            st.plotly_chart(fig, use_container_width=True)

        st.dataframe(comparison.set_index("model"), use_container_width=True)

    st.divider()

    # Predições ao vivo
    st.subheader("Predições ao Vivo (top 20 mercados por liquidez)")
    with st.spinner("Rodando modelo nos mercados ativos..."):
        preds = run_live_predictions()

    if preds.empty:
        st.info("Não foi possível gerar predições (verifique se há dados de mercado recentes).")
    elif "error" in preds.columns:
        st.warning(f"Erro nas predições: {preds['error'].iloc[0]}")
    else:
        fig3 = px.scatter(
            preds,
            x="yes_price", y="model_prob",
            color="edge",
            color_continuous_scale="RdYlGn",
            hover_data=["question", "category"] if "question" in preds.columns else None,
            labels={"yes_price": "Preço de Mercado", "model_prob": "P(YES) Modelo"},
            template="plotly_dark",
            title="Modelo vs Mercado — pontos acima da diagonal = modelo mais otimista",
        )
        # linha y=x (mercado bem calibrado)
        mn, mx = preds["yes_price"].min(), preds["yes_price"].max()
        fig3.add_trace(go.Scatter(x=[mn, mx], y=[mn, mx], mode="lines",
                                  line=dict(color="gray", dash="dot"),
                                  name="Sem edge", showlegend=True))
        fig3.add_hline(y=0.5, line_dash="dot", line_color="gray", opacity=0.3)
        fig3.update_layout(height=400, margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig3, use_container_width=True)

        st.dataframe(
            preds[["question", "category", "yes_price", "model_prob", "edge", "liquidity"]]
            .rename(columns={
                "yes_price": "Preço Mercado", "model_prob": "P(YES) Modelo",
                "edge": "Edge", "liquidity": "Liquidez",
            })
            .style.background_gradient(subset=["Edge"], cmap="RdYlGn", vmin=-0.2, vmax=0.2),
            use_container_width=True,
        )

    # Calibração histórica (posições fechadas)
    st.divider()
    st.subheader("Calibração de Edge (posições fechadas)")
    if not closed_pos.empty and "edge_at_entry" in closed_pos.columns:
        df_cal = closed_pos.copy()
        df_cal["ganhou"] = (df_cal["pnl_usdc"] > 0).astype(int)
        df_cal["edge_bin"] = pd.cut(df_cal["edge_at_entry"], bins=5)
        cal = df_cal.groupby("edge_bin").agg(
            n=("ganhou", "count"),
            edge_medio=("edge_at_entry", "mean"),
            win_rate=("ganhou", "mean"),
        ).reset_index()
        fig4 = go.Figure()
        fig4.add_bar(x=cal["edge_bin"].astype(str), y=cal["win_rate"],
                     name="Win Rate Real", marker_color="#4c9be8")
        fig4.add_bar(x=cal["edge_bin"].astype(str),
                     y=cal["edge_medio"] + 0.5,
                     name="Edge Previsto + 0.5", marker_color="#f39c12", opacity=0.6)
        fig4.update_layout(
            barmode="group", template="plotly_dark", height=300,
            yaxis_tickformat=".0%", margin=dict(l=0, r=0, t=20, b=0),
        )
        st.plotly_chart(fig4, use_container_width=True)
        st.dataframe(cal, use_container_width=True)
    else:
        st.info("Dados insuficientes de posições fechadas para calibração.")


# ════════════════════════════════════════════════════════════
# LOGS
# ════════════════════════════════════════════════════════════
elif page == "Logs":
    st.header("Logs do Sistema")

    tab_light, tab_full, tab_ws, tab_err = st.tabs(["Ciclo Light", "Ciclo Full", "WS Feed", "Erros"])

    with tab_light:
        n_lines = st.slider("Linhas", 20, 200, 80, key="sl_light")
        content = load_log_tail("cycle_light.log", n_lines)
        st.code(content, language="text")

    with tab_full:
        n_lines2 = st.slider("Linhas", 20, 200, 80, key="sl_full")
        content2 = load_log_tail("cycle_full.log", n_lines2)
        st.code(content2, language="text")

    with tab_ws:
        n_lines3 = st.slider("Linhas", 20, 200, 80, key="sl_ws")
        st.code(load_log_tail("ws_feed.log", n_lines3), language="text")

    with tab_err:
        col_e1, col_e2, col_e3 = st.columns(3)
        with col_e1:
            st.caption("cycle_light_error.log")
            st.code(load_log_tail("cycle_light_error.log", 50), language="text")
        with col_e2:
            st.caption("cycle_full_error.log")
            st.code(load_log_tail("cycle_full_error.log", 50), language="text")
        with col_e3:
            st.caption("ws_feed_error.log")
            st.code(load_log_tail("ws_feed_error.log", 50), language="text")


# ── Auto-refresh ────────────────────────────────────────────
if auto_refresh:
    import time
    time.sleep(60)
    st.rerun()
