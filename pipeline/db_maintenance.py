"""
db_maintenance.py
Manutenção do banco do paper trader — retenção do price_history.

Por que existe:
  O ws_feed grava ticks de preço continuamente. Sem retenção, o price_history
  chegou a 95M linhas / 34 GB (2026-07). Este script mantém apenas os últimos
  N dias e remove linhas corrompidas de book snapshot (bid/ask invertidos do
  incidente 2026-05: spread ≈ 0.99).

Uso:
    uv run python pipeline/db_maintenance.py                # retenção 14d
    uv run python pipeline/db_maintenance.py --days 30
    uv run python pipeline/db_maintenance.py --vacuum       # + VACUUM (lento, exclusivo)

Integrado ao run_cycle (1×/dia): retenção sem --vacuum é barata.
"""

import sqlite3
import sys
from pathlib import Path

import click
from loguru import logger

DB_PATH = Path("data/db/paper_trading.db")


def prune_price_history(days: int = 14, vacuum: bool = False, db_path: Path = DB_PATH) -> dict:
    """
    Remove do price_history:
      1. Linhas mais antigas que `days` dias
      2. Linhas de book corrompidas (spread > 0.5 — só existiram pelo bug bids[0]/asks[0])

    Returns:
        dict com contagens antes/depois e bytes recuperados (se vacuum).
    """
    if not db_path.exists():
        logger.warning(f"Banco não encontrado: {db_path} — nada a fazer.")
        return {}

    size_before = db_path.stat().st_size
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")

    try:
        # Tabela pode não existir se o ws_feed nunca rodou neste banco
        exists = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='price_history'"
        ).fetchone()
        if not exists:
            logger.info("Tabela price_history não existe — nada a fazer.")
            return {}

        n_before = conn.execute("SELECT COUNT(*) FROM price_history").fetchone()[0]

        cur = conn.execute(
            "DELETE FROM price_history WHERE ts < datetime('now', ?)",
            (f"-{int(days)} days",),
        )
        n_old = cur.rowcount

        # Book snapshots corrompidos (bid=0.001/ask=0.999 do bug de 2026-05)
        cur = conn.execute(
            "DELETE FROM price_history WHERE best_bid IS NOT NULL AND best_ask IS NOT NULL "
            "AND (best_ask - best_bid) > 0.5"
        )
        n_corrupt = cur.rowcount
        conn.commit()

        n_after = conn.execute("SELECT COUNT(*) FROM price_history").fetchone()[0]
        logger.info(
            f"price_history: {n_before:,} → {n_after:,} linhas "
            f"(removidas: {n_old:,} antigas, {n_corrupt:,} corrompidas)"
        )

        if vacuum:
            logger.info("VACUUM em andamento (requer acesso exclusivo — pause os daemons)...")
            conn.isolation_level = None
            conn.execute("VACUUM")
            logger.info("VACUUM concluído.")

    finally:
        conn.close()

    size_after = db_path.stat().st_size
    if vacuum:
        logger.info(
            f"Tamanho do banco: {size_before/1e9:.2f} GB → {size_after/1e9:.2f} GB "
            f"({(size_before-size_after)/1e9:.2f} GB recuperados)"
        )

    return {
        "rows_before": n_before,
        "rows_after": n_after,
        "removed_old": n_old,
        "removed_corrupt": n_corrupt,
        "bytes_recovered": size_before - size_after,
    }


@click.command()
@click.option("--days",   default=14,   type=int, show_default=True,
              help="Reter apenas os últimos N dias de price_history.")
@click.option("--vacuum", is_flag=True, default=False,
              help="Roda VACUUM após a limpeza (lento; requer daemons pausados).")
@click.option("--db",     default=str(DB_PATH), type=click.Path(), show_default=True,
              help="Caminho do banco SQLite.")
def main(days: int, vacuum: bool, db: str) -> None:
    """Retenção do price_history do paper trader."""
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="<green>{time:HH:mm:ss}</green> | {message}")
    prune_price_history(days=days, vacuum=vacuum, db_path=Path(db))


if __name__ == "__main__":
    main()
