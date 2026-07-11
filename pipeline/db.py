"""
db.py
Inicializa e gerencia o banco SQLite do projeto.
Schema central para controle de pipeline, registro de mercados e runs.
"""

import sqlite3
from pathlib import Path

DB_PATH = Path("data/db/polymarket.db")


def get_connection() -> sqlite3.Connection:
    """Retorna conexão com o banco, criando o arquivo se necessário."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Cria todas as tabelas do schema se ainda não existirem."""
    conn = get_connection()
    conn.executescript("""
        -- Registro de execuções do pipeline (controle de runs)
        CREATE TABLE IF NOT EXISTS pipeline_runs (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            run_type     TEXT    NOT NULL,
            started_at   TEXT    NOT NULL,
            finished_at  TEXT,
            n_records    INTEGER DEFAULT 0,
            status       TEXT    DEFAULT 'running',
            notes        TEXT
        );

        -- Registro central de mercados conhecidos
        CREATE TABLE IF NOT EXISTS market_registry (
            condition_id    TEXT PRIMARY KEY,
            question        TEXT,
            category        TEXT,
            token_yes       TEXT,
            token_no        TEXT,
            volume          REAL    DEFAULT 0,
            liquidity       REAL    DEFAULT 0,
            yes_price       REAL,
            end_date        TEXT,
            is_resolved     INTEGER DEFAULT 0,
            resolution      TEXT,
            first_seen      TEXT    NOT NULL,
            last_seen       TEXT    NOT NULL
        );

        -- Snapshots de preços do orderbook coletados
        CREATE TABLE IF NOT EXISTS orderbook_snapshots (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            token_id        TEXT    NOT NULL,
            collected_at    TEXT    NOT NULL,
            best_bid        REAL,
            best_ask        REAL,
            mid_price       REAL,
            spread          REAL,
            spread_pct      REAL,
            bid_volume_10   REAL,
            ask_volume_10   REAL,
            order_imbalance REAL
        );
    """)
    conn.commit()
    conn.close()


if __name__ == "__main__":
    init_db()
    print(f"Banco inicializado em {DB_PATH.resolve()}")
