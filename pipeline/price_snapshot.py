"""
price_snapshot.py
Salva snapshots horários dos preços de todos os mercados monitorados (posições abertas).

Objetivo: acumular histórico de preços para backtesting real (item 6.1 do guia).
  - Lê posições abertas do SQLite
  - Busca preço atual de cada mercado no Parquet mais recente
  - Appenda ao arquivo data/snapshots/price_history.parquet

Execução:
  uv run python -m pipeline.price_snapshot
  uv run python -m pipeline.price_snapshot --db data/db/paper_trading.db

Integrado ao run_cycle.py para execução a cada ciclo (30min → dados horários acumulados).
"""

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from loguru import logger

DB_PATH       = Path("data/db/paper_trading.db")
RAW_DIR       = Path("data/raw/markets")   # fix 2026-07: apontava para data/raw e nunca achava os parquets
SNAPSHOT_DIR  = Path("data/snapshots")
SNAPSHOT_FILE = SNAPSHOT_DIR / "price_history.parquet"


def _latest_markets_parquet() -> pd.DataFrame | None:
    """Retorna o Parquet de mercados mais recente (all ou incremental)."""
    candidates = sorted(
        list(RAW_DIR.glob("markets_all_*.parquet")) +
        list(RAW_DIR.glob("markets_incremental_*.parquet")),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        logger.warning("Nenhum Parquet de mercados encontrado em data/raw/. Skipping snapshot.")
        return None
    df = pd.read_parquet(candidates[0])
    logger.debug(f"Parquet carregado: {candidates[0].name} ({len(df):,} mercados)")
    return df


def _load_open_condition_ids(db_path: Path) -> list[str]:
    """Retorna lista de conditionIds das posições abertas."""
    if not db_path.exists():
        return []
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        "SELECT DISTINCT condition_id FROM positions WHERE status = 'open'"
    ).fetchall()
    conn.close()
    return [r[0] for r in rows]


def take_snapshot(db_path: Path = DB_PATH) -> int:
    """
    Captura preços atuais das posições abertas e appenda ao histórico.

    Returns:
        Número de mercados incluídos no snapshot (0 se nenhum).
    """
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)

    open_ids = _load_open_condition_ids(db_path)
    if not open_ids:
        logger.info("Nenhuma posição aberta — snapshot vazio, nada a salvar.")
        return 0

    markets = _latest_markets_parquet()
    if markets is None:
        return 0

    # Normaliza coluna de id
    id_col = "conditionId" if "conditionId" in markets.columns else "condition_id"
    if id_col not in markets.columns:
        logger.warning(f"Coluna de conditionId não encontrada no Parquet. Colunas: {list(markets.columns[:10])}")
        return 0

    subset = markets[markets[id_col].isin(open_ids)].copy()
    if subset.empty:
        logger.warning(f"Nenhum dos {len(open_ids)} conditionIds abertos encontrado no Parquet.")
        return 0

    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    cols_keep = [id_col]
    for c in ["yes_price", "spread", "liquidity", "volume24hr", "question", "category", "endDate"]:
        if c in subset.columns:
            cols_keep.append(c)

    snap = subset[cols_keep].copy()
    snap.rename(columns={id_col: "condition_id"}, inplace=True)
    snap.insert(0, "snapshot_at", now_utc)

    # Appenda ao Parquet existente (ou cria novo)
    if SNAPSHOT_FILE.exists():
        existing = pd.read_parquet(SNAPSHOT_FILE)
        combined = pd.concat([existing, snap], ignore_index=True)
    else:
        combined = snap

    combined.to_parquet(SNAPSHOT_FILE, index=False)
    logger.info(
        f"Snapshot salvo: {len(snap)} mercados @ {now_utc} "
        f"(total acumulado: {len(combined):,} linhas em {SNAPSHOT_FILE})"
    )
    return len(snap)


def snapshot_summary() -> None:
    """Exibe resumo do histórico acumulado."""
    if not SNAPSHOT_FILE.exists():
        logger.info("Nenhum histórico de snapshots ainda.")
        return

    df = pd.read_parquet(SNAPSHOT_FILE)
    n_unique   = df["condition_id"].nunique()
    n_snaps    = df["snapshot_at"].nunique()
    oldest     = df["snapshot_at"].min()
    newest     = df["snapshot_at"].max()

    logger.info(
        f"Histórico: {len(df):,} registros | "
        f"{n_unique} mercados únicos | "
        f"{n_snaps} snapshots | "
        f"{oldest} → {newest}"
    )


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Salva snapshot de preços das posições abertas")
    p.add_argument("--db",      default=str(DB_PATH), help="Caminho do SQLite")
    p.add_argument("--summary", action="store_true",  help="Mostra resumo do histórico acumulado")
    args = p.parse_args()

    if args.summary:
        snapshot_summary()
    else:
        n = take_snapshot(Path(args.db))
        if n == 0:
            logger.info("Nada a fazer.")
