"""
feature_engineering.py
Constrói o dataset de features para os modelos de ML a partir dos dados coletados.

Três fontes de features:
  1. Contexto do mercado   → metadados da Gamma API (market_registry)
  2. Dinâmica de preços    → séries temporais do CLOB (prices/*.parquet)
  3. Microestrutura        → orderbook snapshots do SQLite

Uso:
    uv run python features/feature_engineering.py
    uv run python features/feature_engineering.py --min-liquidity 5000
"""

import sys
import click
import sqlite3
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent / "pipeline"))
from db import get_connection

FEATURES_DIR = Path("data/raw/features")
PRICES_DIR   = Path("data/raw/prices")
FEATURES_DIR.mkdir(parents=True, exist_ok=True)


# ===========================================================================
# 1. Features de contexto — derivadas dos metadados do mercado
# ===========================================================================

# Mapeamento manual das categorias mais comuns para IDs numéricos estáveis
CATEGORY_MAP = {
    "crypto":    0, "bitcoin":   0, "ethereum":  0,
    "politics":  1, "democratic":1, "republican":1, "presidential":1, "trump":1,
    "sports":    2, "nba":       2, "nfl":       2, "mlb":       2, "nhl":  2,
    "economics": 3, "fed":       3,
    "science":   4, "space":     4, "spacex":    4,
    "entertainment": 5, "eurovision": 5,
    "geopolitics":   6, "iran": 6, "russia": 6, "ukraine": 6, "israel": 6,
}


def _encode_category(category) -> int:
    """Converte string de categoria para inteiro estável. 99 = desconhecido."""
    if not category or not isinstance(category, str):
        return 99
    return CATEGORY_MAP.get(category.lower().strip(), 99)


def build_context_features(markets_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extrai features contextuais de cada mercado a partir dos metadados.
    Captura 'tipo' e 'maturidade' do mercado sem depender de histórico de preços.

    Colunas geradas:
        days_to_resolution, is_near_resolution_date, log_volume, log_liquidity,
        log_volume24hr, volume_recency_ratio, category_id, has_resolution_source,
        yes_price, spread, bid_ask_mid
    """
    now = pd.Timestamp.now(tz="UTC")
    rows = []

    for _, m in markets_df.iterrows():
        end_date = pd.to_datetime(m.get("endDate"), errors="coerce", utc=True)
        days_left = (end_date - now).days if pd.notna(end_date) else -1

        volume     = float(m.get("volume", 0) or 0)
        liquidity  = float(m.get("liquidity", 0) or 0)
        volume24hr = float(m.get("volume24hr", 0) or 0)
        volume1wk  = float(m.get("volume1wk", 0) or 0)

        # Ratio volume recente / volume total — mercado "esquentando" ou "esfriando"
        recency_ratio = volume24hr / (volume / 30 + 1e-9)  # normaliza por média diária estimada

        best_bid = float(m.get("bestBid", 0) or 0)
        best_ask = float(m.get("bestAsk", 0) or 0)
        mid      = (best_bid + best_ask) / 2 if best_bid > 0 and best_ask > 0 else None

        rows.append({
            "condition_id":            m.get("conditionId"),
            "token_yes":               m.get("token_yes"),
            "token_no":                m.get("token_no"),
            "yes_price":               float(m.get("yes_price") or 0),
            # Tempo
            "days_to_resolution":      max(days_left, -1),
            "is_near_resolution":      int(0 <= days_left <= 7),
            "is_long_horizon":         int(days_left > 90),
            # Volume e liquidez (log-transformados para reduzir skewness)
            "log_volume":              np.log1p(volume),
            "log_liquidity":           np.log1p(liquidity),
            "log_volume24hr":          np.log1p(volume24hr),
            "log_volume1wk":           np.log1p(volume1wk),
            "volume_recency_ratio":    min(recency_ratio, 100.0),  # cap para outliers
            # Preço e spread
            "spread":                  float(m.get("spread", 0) or 0),
            "bid_ask_mid":             mid if mid else float(m.get("yes_price") or 0),
            "price_1d_change":         float(m.get("oneDayPriceChange", 0) or 0),
            "price_1w_change":         float(m.get("oneWeekPriceChange", 0) or 0),
            "price_1m_change":         float(m.get("oneMonthPriceChange", 0) or 0),
            # Categorical
            "category_id":             _encode_category(m.get("category")),
            "has_resolution_source":   int(bool(m.get("resolutionSource"))),
            # Calibração — distância dos bounds (0 = NO certo, 1 = YES certo)
            "dist_to_zero":            float(m.get("yes_price") or 0),
            "dist_to_one":             1.0 - float(m.get("yes_price") or 0),
            "is_extreme_price":        int(
                float(m.get("yes_price") or 0) > 0.85
                or float(m.get("yes_price") or 0) < 0.10
            ),
        })

    return pd.DataFrame(rows)


# ===========================================================================
# 2. Features de preço — momentum e volatilidade de séries temporais
# ===========================================================================

def _compute_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """RSI — detecta mercados sobrecomprados/sobrevendidos."""
    delta = prices.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    return 100 - (100 / (1 + gain / (loss + 1e-9)))


def build_price_features(prices_df: pd.DataFrame, token_id: str) -> pd.DataFrame:
    """
    Calcula features de momentum e volatilidade para um token específico.
    Assume série temporal horária com colunas: timestamp, price, token_id.

    Retorna DataFrame com uma linha por timestep, pronto para join com context features.
    """
    d = prices_df[prices_df["token_id"] == token_id].copy()
    if d.empty:
        return pd.DataFrame()

    d = d.sort_values("timestamp").set_index("timestamp")

    # Retornos em múltiplas janelas
    d["return_1h"]  = d["price"].pct_change(1)
    d["return_6h"]  = d["price"].pct_change(6)
    d["return_24h"] = d["price"].pct_change(24)
    d["return_7d"]  = d["price"].pct_change(168)

    # Médias móveis
    d["ma_6h"]  = d["price"].rolling(6).mean()
    d["ma_24h"] = d["price"].rolling(24).mean()
    d["ma_7d"]  = d["price"].rolling(168).mean()

    # Desvio da média — detecta sobre/sub-reação do mercado
    d["dev_ma6h"]  = (d["price"] - d["ma_6h"])  / (d["ma_6h"]  + 1e-9)
    d["dev_ma24h"] = (d["price"] - d["ma_24h"]) / (d["ma_24h"] + 1e-9)

    # Aceleração (segunda derivada do preço) — momentum crescendo ou diminuindo?
    d["momentum_velocity"] = d["return_1h"].diff()

    # Volatilidade realizada
    d["vol_24h"] = d["return_1h"].rolling(24).std()
    d["vol_7d"]  = d["return_1h"].rolling(168).std()

    # RSI — sobrecomprado (>70) ou sobrevendido (<30)?
    d["rsi_14"] = _compute_rsi(d["price"], period=14)

    # Distância dos bounds
    d["dist_to_zero"]      = d["price"]
    d["dist_to_one"]       = 1.0 - d["price"]
    d["is_near_resolution"] = ((d["price"] > 0.85) | (d["price"] < 0.15)).astype(int)

    d["token_id"] = token_id
    return d.reset_index()


def build_price_features_summary(prices_df: pd.DataFrame, token_id: str) -> dict:
    """
    Versão resumida: retorna UMA linha com as features mais recentes do token.
    Útil para join com context features no dataset de treino.
    """
    feat_df = build_price_features(prices_df, token_id)
    if feat_df.empty:
        return {"token_id": token_id}

    last = feat_df.iloc[-1]
    return {
        "token_id":           token_id,
        "price_current":      last.get("price"),
        "return_1h":          last.get("return_1h"),
        "return_24h":         last.get("return_24h"),
        "return_7d":          last.get("return_7d"),
        "vol_24h":            last.get("vol_24h"),
        "vol_7d":             last.get("vol_7d"),
        "rsi_14":             last.get("rsi_14"),
        "dev_ma6h":           last.get("dev_ma6h"),
        "dev_ma24h":          last.get("dev_ma24h"),
        "momentum_velocity":  last.get("momentum_velocity"),
        "is_near_resolution": last.get("is_near_resolution"),
    }


# ===========================================================================
# 3. Features de microestrutura — orderbook (do SQLite)
# ===========================================================================

def build_orderbook_features(token_id: str, n_snapshots: int = 20) -> dict:
    """
    Lê os últimos N snapshots do orderbook para um token e calcula features
    de microestrutura: spread regime, imbalance trend, liquidez.
    Retorna dict com uma linha por token, pronto para join.
    """
    conn = get_connection()
    rows = conn.execute(
        """SELECT best_bid, best_ask, mid_price, spread, spread_pct,
                  bid_volume_10, ask_volume_10, order_imbalance, collected_at
           FROM orderbook_snapshots
           WHERE token_id = ?
           ORDER BY collected_at DESC
           LIMIT ?""",
        (token_id, n_snapshots),
    ).fetchall()
    conn.close()

    if not rows:
        return {"token_id": token_id}

    df = pd.DataFrame([dict(r) for r in rows]).sort_values("collected_at")

    spread    = df["spread"].dropna()
    imbalance = df["order_imbalance"].dropna()
    liquidity = (df["bid_volume_10"] + df["ask_volume_10"]).dropna()

    spread_ma  = spread.mean()
    spread_std = spread.std()

    return {
        "token_id":            token_id,
        "spread_current":      float(df["spread"].iloc[-1]) if not df["spread"].empty else None,
        "spread_ma":           float(spread_ma) if not pd.isna(spread_ma) else None,
        "spread_regime":       int(float(df["spread"].iloc[-1]) > spread_ma) if len(spread) > 1 else 0,
        "spread_z":            float((float(df["spread"].iloc[-1]) - spread_ma) / (spread_std + 1e-9)) if len(spread) > 1 else 0,
        "imbalance_current":   float(imbalance.iloc[-1]) if not imbalance.empty else None,
        "imbalance_ma":        float(imbalance.mean()) if not imbalance.empty else None,
        "imbalance_trend":     float(imbalance.iloc[-1] - imbalance.mean()) if len(imbalance) > 1 else 0,
        "total_liquidity":     float(liquidity.iloc[-1]) if not liquidity.empty else None,
        "liquidity_change":    float(liquidity.pct_change().iloc[-1]) if len(liquidity) > 1 else 0,
        "n_snapshots":         len(df),
    }


# ===========================================================================
# 4. Pipeline completo — junta as três fontes em um dataset final
# ===========================================================================

def build_full_feature_set(
    min_liquidity: float = 1_000,
    min_volume24hr: float = 0,
) -> pd.DataFrame:
    """
    Pipeline principal: carrega dados do DB e Parquet, constrói todas as features,
    junta em um único DataFrame e salva em data/raw/features/.

    Filtros:
        min_liquidity:  exclui mercados sem liquidez suficiente para operar
        min_volume24hr: exclui mercados sem atividade recente
    """
    logger.info("Carregando market_registry do SQLite...")
    conn = get_connection()
    markets_raw = pd.read_sql(
        "SELECT * FROM market_registry WHERE is_resolved = 0",
        conn,
    )
    conn.close()
    logger.info(f"  {len(markets_raw):,} mercados não-resolvidos no DB")

    # Carrega o Parquet mais recente dos metadados enriquecidos (com campos extras)
    parquet_files = sorted(Path("data/raw/markets").glob("*.parquet"), reverse=True)
    if parquet_files:
        markets_rich = pd.read_parquet(parquet_files[0])
        # Faz merge para adicionar campos que não estão no market_registry (ex: volume24hr)
        # Exclui a chave do merge e colunas já presentes no market_registry
        extra_cols = [
            c for c in markets_rich.columns
            if c not in markets_raw.columns and c != "conditionId"
        ]
        if extra_cols:
            # Deduplica pelo conditionId mais recente (Parquet pode ter múltiplos snapshots)
            markets_rich_dedup = markets_rich.drop_duplicates("conditionId", keep="last")
            merge_cols = ["conditionId"] + extra_cols
            markets_df = markets_raw.merge(
                markets_rich_dedup[merge_cols],
                left_on="condition_id",
                right_on="conditionId",
                how="left",
            )
        else:
            markets_df = markets_raw.rename(columns={"condition_id": "conditionId"})
    else:
        logger.warning("Nenhum Parquet de mercados encontrado — usando apenas SQLite")
        markets_df = markets_raw.rename(columns={"condition_id": "conditionId"})

    # Aplica filtros de qualidade
    before = len(markets_df)
    if "liquidity" in markets_df.columns:
        markets_df = markets_df[markets_df["liquidity"].fillna(0) >= min_liquidity]
    if "volume24hr" in markets_df.columns and min_volume24hr > 0:
        markets_df = markets_df[markets_df["volume24hr"].fillna(0) >= min_volume24hr]
    logger.info(f"  Após filtros: {len(markets_df):,} mercados (removidos {before - len(markets_df):,})")

    # 1. Features de contexto
    logger.info("Construindo features de contexto...")
    context_df = build_context_features(markets_df)

    # 2. Features de preço (se existir Parquet de preços)
    price_files = sorted(PRICES_DIR.glob("*.parquet"), reverse=True)
    price_summaries = []
    if price_files:
        logger.info(f"Carregando preços de {price_files[0].name}...")
        prices_df = pd.read_parquet(price_files[0])
        token_ids = context_df["token_yes"].dropna().unique().tolist()
        logger.info(f"Calculando features de preço para {len(token_ids)} tokens...")
        for tid in token_ids:
            summary = build_price_features_summary(prices_df, tid)
            price_summaries.append(summary)
    else:
        logger.info("  Nenhum arquivo de preços encontrado — pulando features de preço")

    # 3. Features de orderbook (se existir dados no SQLite)
    conn = get_connection()
    n_ob = conn.execute("SELECT COUNT(*) FROM orderbook_snapshots").fetchone()[0]
    conn.close()
    ob_features = []
    if n_ob > 0:
        logger.info(f"Construindo features de orderbook ({n_ob} snapshots)...")
        token_ids = context_df["token_yes"].dropna().unique().tolist()
        for tid in token_ids:
            ob_features.append(build_orderbook_features(tid))
    else:
        logger.info("  Nenhum snapshot de orderbook encontrado — pulando microestrutura")

    # Junta tudo
    final_df = context_df.copy()

    if price_summaries:
        price_df = pd.DataFrame(price_summaries)
        final_df = final_df.merge(price_df, on="token_yes", how="left", suffixes=("", "_price"))

    if ob_features:
        ob_df = pd.DataFrame(ob_features).rename(columns={"token_id": "token_yes"})
        final_df = final_df.merge(ob_df, on="token_yes", how="left", suffixes=("", "_ob"))

    # Salva
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = FEATURES_DIR / f"features_{ts}.parquet"
    final_df.to_parquet(path, index=False, compression="snappy")

    logger.info(f"Dataset salvo: {path}")
    logger.info(f"  Shape: {final_df.shape[0]:,} mercados × {final_df.shape[1]} features")
    logger.info(f"  Features: {list(final_df.columns)}")

    return final_df


@click.command()
@click.option("--min-liquidity",  default=1_000, type=float, show_default=True,
              help="Liquidez mínima em USDC para incluir o mercado.")
@click.option("--min-volume24hr", default=0,     type=float, show_default=True,
              help="Volume mínimo nas últimas 24h em USDC.")
def main(min_liquidity: float, min_volume24hr: float) -> None:
    """Constrói o dataset de features para os modelos de ML."""
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="<green>{time:HH:mm:ss}</green> | {message}")

    df = build_full_feature_set(
        min_liquidity=min_liquidity,
        min_volume24hr=min_volume24hr,
    )

    print(f"\nDataset pronto: {df.shape[0]:,} mercados × {df.shape[1]} features")
    print(f"\nAmostra (top 5 por liquidez):")
    cols = ["condition_id", "yes_price", "days_to_resolution", "log_volume",
            "log_liquidity", "spread", "category_id"]
    available = [c for c in cols if c in df.columns]
    print(
        df.nlargest(5, "log_liquidity")[available].to_string(index=False)
    )
    print(f"\nNaN por coluna (top 10):")
    nan_counts = df.isnull().sum().sort_values(ascending=False).head(10)
    print(nan_counts[nan_counts > 0].to_string())


if __name__ == "__main__":
    main()
