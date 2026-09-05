"""
clob_collector.py
Coleta histórico de preços e snapshots de orderbook via CLOB API.
Endpoints públicos funcionam sem auth; histórico requer credenciais.
"""

import os
import time
from pathlib import Path
from datetime import datetime, timedelta, timezone

import requests
import pandas as pd
from loguru import logger
from dotenv import load_dotenv

from pipeline.db import init_db, get_connection

load_dotenv()

CLOB_BASE = "https://clob.polymarket.com"
PRICES_DIR = Path("data/raw/prices")
ORDERBOOK_DIR = Path("data/raw/orderbook")
PRICES_DIR.mkdir(parents=True, exist_ok=True)
ORDERBOOK_DIR.mkdir(parents=True, exist_ok=True)

# Mapeamento de intervalo legível para fidelidade em minutos (parâmetro da API)
INTERVAL_MAP = {"1m": 1, "5m": 5, "15m": 15, "1h": 60, "6h": 360, "1d": 1440}


def fetch_price_history(
    token_id: str,
    days_back: int = 30,
    interval: str = "1h",
) -> pd.DataFrame:
    """
    Busca histórico de preços (série temporal) de um token via CLOB API.
    Endpoint público — não requer autenticação.

    Args:
        token_id: ID do token YES ou NO no CLOB
        days_back: quantos dias de histórico buscar
        interval: granularidade dos pontos ("1m", "5m", "1h", "1d")

    Returns:
        DataFrame com colunas: timestamp, token_id, price
    """
    fidelity = INTERVAL_MAP.get(interval, 60)
    start_ts = int((datetime.now() - timedelta(days=days_back)).timestamp())
    end_ts   = int(datetime.now().timestamp())

    params = {
        "market":    token_id,
        "startTs":   start_ts,
        "endTs":     end_ts,
        "fidelity":  fidelity,
    }

    try:
        resp = requests.get(f"{CLOB_BASE}/prices-history", params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as e:
        logger.warning(f"Erro ao buscar preços para {token_id[:20]}: {e}")
        return pd.DataFrame()

    history = data.get("history", [])
    if not history:
        return pd.DataFrame()

    df = pd.DataFrame(history)
    df["timestamp"] = pd.to_datetime(df["t"], unit="s", utc=True)
    df["price"]     = df["p"].astype(float)
    df["token_id"]  = token_id
    return df[["timestamp", "token_id", "price"]].sort_values("timestamp").reset_index(drop=True)


def fetch_orderbook_snapshot(token_id: str) -> dict | None:
    """
    Captura snapshot atual do orderbook de um token.
    Calcula spread, imbalance e liquidez dos 10 melhores níveis.
    Endpoint público — não requer autenticação.
    """
    try:
        resp = requests.get(f"{CLOB_BASE}/book", params={"token_id": token_id}, timeout=10)
        resp.raise_for_status()
        book = resp.json()
    except requests.RequestException as e:
        logger.warning(f"Erro ao buscar orderbook para {token_id[:20]}: {e}")
        return None

    bids = [(float(b["price"]), float(b["size"])) for b in book.get("bids", [])]
    asks = [(float(a["price"]), float(a["size"])) for a in book.get("asks", [])]

    if not bids or not asks:
        return None

    best_bid = max(bids, key=lambda x: x[0])
    best_ask = min(asks, key=lambda x: x[0])
    mid_price = (best_bid[0] + best_ask[0]) / 2
    spread    = best_ask[0] - best_bid[0]

    # Liquidez agregada nos 10 melhores níveis de cada lado
    bid_vol = sum(s for _, s in sorted(bids, key=lambda x: -x[0])[:10])
    ask_vol = sum(s for _, s in sorted(asks, key=lambda x:  x[0])[:10])
    imbalance = (bid_vol - ask_vol) / (bid_vol + ask_vol + 1e-9)

    return {
        "token_id":        token_id,
        "collected_at":    datetime.now(timezone.utc).isoformat(),
        "best_bid":        best_bid[0],
        "best_ask":        best_ask[0],
        "mid_price":       mid_price,
        "spread":          spread,
        "spread_pct":      spread / mid_price if mid_price > 0 else None,
        "bid_volume_10":   bid_vol,
        "ask_volume_10":   ask_vol,
        "order_imbalance": imbalance,
    }


def save_orderbook_snapshot(snapshot: dict) -> None:
    """Persiste snapshot do orderbook no SQLite."""
    conn = get_connection()
    conn.execute("""
        INSERT INTO orderbook_snapshots
            (token_id, collected_at, best_bid, best_ask, mid_price,
             spread, spread_pct, bid_volume_10, ask_volume_10, order_imbalance)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        snapshot["token_id"],
        snapshot["collected_at"],
        snapshot.get("best_bid"),
        snapshot.get("best_ask"),
        snapshot.get("mid_price"),
        snapshot.get("spread"),
        snapshot.get("spread_pct"),
        snapshot.get("bid_volume_10"),
        snapshot.get("ask_volume_10"),
        snapshot.get("order_imbalance"),
    ))
    conn.commit()
    conn.close()


def collect_prices_batch(
    token_ids: list[str],
    days_back: int = 30,
    interval: str = "1h",
    save: bool = True,
) -> pd.DataFrame:
    """
    Coleta histórico de preços para uma lista de tokens em batch.
    Salva resultado combinado em Parquet.

    Args:
        token_ids: lista de token IDs (YES/NO) para coletar
        days_back: janela histórica em dias
        interval: granularidade dos candles
        save: se True, persiste em Parquet

    Returns:
        DataFrame combinado com preços de todos os tokens
    """
    all_dfs = []
    n = len(token_ids)

    for i, tid in enumerate(token_ids):
        logger.debug(f"  [{i+1}/{n}] Preços: {tid[:24]}...")
        df = fetch_price_history(tid, days_back=days_back, interval=interval)
        if not df.empty:
            all_dfs.append(df)
        time.sleep(0.35)  # respeita rate limit

    if not all_dfs:
        logger.warning("Nenhum dado de preços coletado.")
        return pd.DataFrame()

    combined = pd.concat(all_dfs, ignore_index=True)

    if save:
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = PRICES_DIR / f"prices_{interval}_{ts}.parquet"
        combined.to_parquet(path, index=False, compression="snappy")
        logger.info(f"Preços salvos: {path} ({len(combined):,} linhas, {combined['token_id'].nunique()} tokens)")

    return combined


def collect_orderbook_batch(token_ids: list[str], save_db: bool = True) -> list[dict]:
    """
    Captura snapshots do orderbook para múltiplos tokens.
    Persiste cada snapshot no SQLite.
    """
    snapshots = []
    n = len(token_ids)

    for i, tid in enumerate(token_ids):
        logger.debug(f"  [{i+1}/{n}] Orderbook: {tid[:24]}...")
        snap = fetch_orderbook_snapshot(tid)
        if snap:
            snapshots.append(snap)
            if save_db:
                save_orderbook_snapshot(snap)
        time.sleep(0.2)

    logger.info(f"Orderbook: {len(snapshots)}/{n} tokens coletados")
    return snapshots


if __name__ == "__main__":
    # Teste rápido com um token público (Bitcoin acima de $X no Polymarket)
    # Para rodar com tokens reais, use token_ids do market_registry
    print("Testando CLOB API (endpoint publico)...")
    resp = requests.get(f"{CLOB_BASE}/markets?limit=3", timeout=10)
    if resp.ok:
        markets = resp.json().get("data", [])
        if markets:
            sample_token = markets[0].get("tokens", [{}])[0].get("token_id", "")
            if sample_token:
                snap = fetch_orderbook_snapshot(sample_token)
                print(f"Orderbook snapshot: {snap}")
    print("CLOB API OK")
