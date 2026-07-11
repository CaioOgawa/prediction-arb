"""
dataset_builder.py
Constrói o dataset de treino/teste a partir de mercados já resolvidos.

VERSÃO CORRIGIDA — sem data leakage.

Leakage confirmado na v1:
  - `yes_price` / `lastTradePrice`: para mercados resolvidos, a API retorna o
    preço de settlement (0.0 ou 1.0) → o modelo aprende a ler o outcome, não
    a prever. Acurácia de 92% era artificial.
  - `is_extreme_price`, `price_1d_change`, `price_1w_change`: derivados do
    preço pós-resolução → mesma contaminação.

Features seguras (agnósticas ao outcome):
  - log_volume, log_liquidity, log_volume24hr, log_volume1wk — tamanho/atividade
  - volume_recency_ratio — se o trading intensificou perto do fechamento
  - spread — qualidade do book de ordens
  - days_ran — duração do mercado
  - has_resolution_source — critério de resolução claro
  - category_* — one-hot da categoria do evento pai (esportes, política, etc.)

O modelo aprende: "dado o perfil estrutural deste mercado, qual é a taxa base de
resolução YES?" Na inferência, comparamos P(YES)_modelo vs yes_price_atual → edge.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from loguru import logger
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent / "pipeline"))

GAMMA_BASE = "https://gamma-api.polymarket.com"
RAW_DIR    = Path("data/raw/markets")
RAW_DIR.mkdir(parents=True, exist_ok=True)

# Features base (sem one-hot de categoria — adicionadas dinamicamente)
BASE_FEATURE_COLS = [
    "log_volume",
    "log_liquidity",
    "log_volume24hr",
    "log_volume1wk",
    "volume_recency_ratio",
    "spread",
    "days_ran",
    "has_resolution_source",
]

# Categorias conhecidas — usadas para garantir colunas consistentes entre treino e inferência
KNOWN_CATEGORIES = [
    "sports", "politics", "crypto", "economics", "pop-culture",
    "science", "technology", "weather", "entertainment", "other",
]


def _resolve_outcome(market: dict, threshold: float = 0.95) -> int | None:
    """
    Extrai o resultado do mercado a partir de outcomePrices.
    Retorna 1 (YES ganhou), 0 (NO ganhou), ou None (ambíguo/não resolvido).
    """
    import json as _json
    prices = market.get("outcomePrices", [])
    if isinstance(prices, str):
        try:
            prices = _json.loads(prices)
        except Exception:
            return None
    if not isinstance(prices, list) or len(prices) < 2:
        return None
    try:
        p_yes = float(prices[0])
        p_no  = float(prices[1])
        if p_yes >= threshold:
            return 1
        if p_no >= threshold:
            return 0
        return None
    except (ValueError, TypeError):
        return None


def _normalize_category(raw: str) -> str:
    """Normaliza categoria para um conjunto fixo de labels."""
    if not raw:
        return "other"
    raw = raw.lower().strip()
    mapping = {
        "sports": "sports",
        "sport":  "sports",
        "nba": "sports", "nfl": "sports", "mlb": "sports", "nhl": "sports",
        "soccer": "sports", "football": "sports",
        "politics": "politics",
        "political": "politics",
        "elections": "politics",
        "election": "politics",
        "us-current-affairs": "politics",
        "crypto": "crypto",
        "cryptocurrency": "crypto",
        "defi": "crypto",
        "economics": "economics",
        "economy": "economics",
        "finance": "economics",
        "pop-culture": "pop-culture",
        "pop culture": "pop-culture",
        "entertainment": "entertainment",
        "science": "science",
        "technology": "technology",
        "tech": "technology",
        "weather": "weather",
    }
    for key, val in mapping.items():
        if key in raw:
            return val
    return "other"


def _parse_resolved_market(m: dict, category: str = "other") -> dict | None:
    """
    Extrai features de um mercado resolvido SEM usar preço como feature.
    Retorna None se o mercado não tiver outcome definido.
    """
    outcome = _resolve_outcome(m)
    if outcome is None:
        return None

    volume    = float(m.get("volume", 0) or 0)
    liquidity = float(m.get("liquidity", 0) or 0)
    v24h      = float(m.get("volume24hr", 0) or 0)
    v1wk      = float(m.get("volume1wk", 0) or 0)

    start = pd.to_datetime(m.get("createdAt") or m.get("startDate"), errors="coerce", utc=True)
    end   = pd.to_datetime(m.get("endDate"), errors="coerce", utc=True)
    days_ran = int((end - start).days) if pd.notna(start) and pd.notna(end) else -1

    # Volume recente vs. histórico (atividade cresceu perto do close?)
    avg_daily = volume / max(days_ran, 1)
    recency_ratio = float(np.clip(v24h / (avg_daily + 1e-9), 0, 100))

    spread = float(m.get("spread", 1.0) or 1.0)

    return {
        "condition_id":         m.get("conditionId"),
        "question":             str(m.get("question", ""))[:80],
        "category":             _normalize_category(category),
        "outcome":              outcome,
        "end_date":             end.isoformat() if pd.notna(end) else None,
        # Features agnósticas ao outcome
        "log_volume":           float(np.log1p(volume)),
        "log_liquidity":        float(np.log1p(liquidity)),
        "log_volume24hr":       float(np.log1p(v24h)),
        "log_volume1wk":        float(np.log1p(v1wk)),
        "volume_recency_ratio": recency_ratio,
        "spread":               spread,
        "days_ran":             max(days_ran, 0),
        "has_resolution_source": int(bool(m.get("resolutionSource"))),
    }


def fetch_resolved_markets(
    min_volume: float = 1_000,
    max_events: int = 2_000,
    limit: int = 100,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Busca mercados resolvidos via /events — categoria extraída do evento pai.

    Args:
        min_volume:    volume mínimo em USDC por mercado
        max_events:    máximo de eventos a percorrer
        limit:         eventos por página
        force_refresh: ignora cache e baixa novamente

    Returns:
        DataFrame com features sem leakage + coluna 'outcome' (target)
    """
    logger.info(f"Buscando mercados resolvidos (min_volume=${min_volume:,.0f})...")

    if not force_refresh:
        # Cache da versão sem leakage (prefixo v2_)
        cached = sorted(RAW_DIR.glob("resolved_markets_v2_*.parquet"), reverse=True)
        if cached:
            logger.info(f"Cache encontrado: {cached[0].name} — carregando...")
            df = pd.read_parquet(cached[0])
            logger.info(f"  {len(df):,} mercados resolvidos carregados do cache")
            return df

    rows     = []
    n_events = 0
    offset   = 0

    while n_events < max_events:
        params = {"closed": "true", "limit": limit, "offset": offset}
        try:
            resp = requests.get(f"{GAMMA_BASE}/events", params=params, timeout=20)
            resp.raise_for_status()
            events = resp.json()
        except requests.RequestException as e:
            logger.error(f"Erro na API de eventos (offset={offset}): {e}")
            break

        if not events:
            break

        for event in events:
            raw_cat  = event.get("category", "") or ""
            category = _normalize_category(raw_cat)

            for m in event.get("markets", []):
                vol = float(m.get("volume", 0) or 0)
                if vol < min_volume:
                    continue
                parsed = _parse_resolved_market(m, category=category)
                if parsed is not None:
                    rows.append(parsed)

        n_events += len(events)
        logger.debug(f"  eventos={n_events} | com_outcome={len(rows)}")
        offset += limit
        time.sleep(0.5)

        if len(events) < limit:
            break

    df = pd.DataFrame(rows) if rows else pd.DataFrame()
    if df.empty:
        logger.warning("Nenhum mercado resolvido encontrado.")
        return df

    logger.info(f"Total resolvidos: {len(df):,} | YES rate: {df['outcome'].mean():.1%}")
    logger.info(f"Categorias: {df['category'].value_counts().to_dict()}")

    ts   = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    path = RAW_DIR / f"resolved_markets_v2_{ts}.parquet"
    df.to_parquet(path, index=False, compression="snappy")
    logger.info(f"Salvo em {path}")

    return df


def add_category_dummies(
    df: pd.DataFrame,
    known_categories: list[str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Adiciona colunas one-hot para a categoria. Garante colunas consistentes
    entre treino e inferência usando `known_categories`.

    Returns:
        (df_com_dummies, lista_de_colunas_categoria)
    """
    if known_categories is None:
        known_categories = KNOWN_CATEGORIES

    for cat in known_categories:
        col = f"cat_{cat}"
        df[col] = (df["category"] == cat).astype(int)

    cat_cols = [f"cat_{c}" for c in known_categories]
    return df, cat_cols


def build_train_test_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    scale: bool = True,
) -> tuple:
    """
    Prepara X, y e faz split treino/teste com features sem leakage.

    Returns:
        X_train, X_test, y_train, y_test, feature_names, scaler
    """
    df, cat_cols = add_category_dummies(df)
    feature_cols = BASE_FEATURE_COLS + cat_cols

    available = [c for c in feature_cols if c in df.columns]
    missing   = [c for c in feature_cols if c not in df.columns]
    if missing:
        logger.warning(f"Features ausentes (ignoradas): {missing}")

    X = df[available].copy()
    y = df["outcome"].astype(int)

    for col in X.columns:
        if X[col].isnull().any():
            X[col] = X[col].fillna(X[col].median())

    logger.info(f"Dataset: {len(X):,} amostras | {y.mean():.1%} YES | {len(available)} features")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y,
    )

    scaler = None
    if scale:
        scaler = StandardScaler()
        X_train = pd.DataFrame(
            scaler.fit_transform(X_train),
            columns=available, index=X_train.index,
        )
        X_test = pd.DataFrame(
            scaler.transform(X_test),
            columns=available, index=X_test.index,
        )

    logger.info(f"Treino: {len(X_train):,} | Teste: {len(X_test):,}")
    return X_train, X_test, y_train, y_test, available, scaler


def build_features_matrix(
    df: pd.DataFrame,
    scale: bool = False,
) -> tuple[pd.DataFrame, pd.Series, list[str], "StandardScaler | None"]:
    """
    Constrói X e y sem fazer split, ordenando por end_date (mais antigo primeiro).
    Usar scale=False para walk-forward CV (a escala é feita dentro de cada fold).

    Returns:
        (X, y, feature_names, scaler)
    """
    # Ordenação temporal — garante que walk-forward não olhe o futuro
    if "end_date" in df.columns:
        df = df.copy()
        df["end_date"] = pd.to_datetime(df["end_date"], errors="coerce", utc=True)
        df = df.sort_values("end_date", na_position="last").reset_index(drop=True)

    df, cat_cols = add_category_dummies(df.copy())
    feature_cols = BASE_FEATURE_COLS + cat_cols
    available    = [c for c in feature_cols if c in df.columns]

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        logger.warning(f"Features ausentes (ignoradas): {missing}")

    X = df[available].copy()
    y = df["outcome"].astype(int)

    for col in X.columns:
        if X[col].isnull().any():
            X[col] = X[col].fillna(X[col].median())

    logger.info(f"Dataset: {len(X):,} amostras | {y.mean():.1%} YES | {len(available)} features")

    scaler = None
    if scale:
        scaler = StandardScaler()
        X = pd.DataFrame(scaler.fit_transform(X), columns=available, index=X.index)

    return X, y, available, scaler


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="<green>{time:HH:mm:ss}</green> | {message}")

    df = fetch_resolved_markets(min_volume=1_000, force_refresh=True)
    if not df.empty:
        print(f"\nShape: {df.shape}")
        print(f"YES rate: {df['outcome'].mean():.1%}")
        print(f"\nCategoria × outcome:")
        print(df.groupby("category")["outcome"].agg(["mean", "count"]).round(3).sort_values("count", ascending=False))
        print(f"\nAmostra:")
        print(df[["question", "category", "outcome", "log_volume", "spread", "days_ran"]].head(5).to_string(index=False))
