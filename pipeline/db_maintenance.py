"""
db_maintenance.py
Manutenção do banco do paper trader — retenção do price_history e dos
arquivos de report/parquet que cada ciclo gera.

Por que existe:
  O ws_feed grava ticks de preço continuamente. Sem retenção, o price_history
  chegou a 95M linhas / 34 GB (2026-07). Este script mantém apenas os últimos
  N dias e remove linhas corrompidas de book snapshot (bid/ask invertidos do
  incidente 2026-05: spread ≈ 0.99).

  P2-37: outputs/reports e data/raw/markets nunca tinham retenção nenhuma —
  load_current_markets()/load_signals() só leem o arquivo mais recente de
  cada padrão (por mtime), então tudo mais velho que alguns dias é só
  histórico morto. Chegou a 15 GB / 11.556 arquivos em outputs/reports.

Uso:
    uv run python pipeline/db_maintenance.py                # retenção 14d
    uv run python pipeline/db_maintenance.py --days 30
    uv run python pipeline/db_maintenance.py --vacuum       # + VACUUM (lento, exclusivo)
    uv run python pipeline/db_maintenance.py --dry-run      # só reporta o que apagaria

Integrado ao run_cycle (1×/dia): retenção sem --vacuum é barata.

IMPORTANTE sobre --vacuum: DELETE não encolhe o arquivo .db, só marca páginas
como livres — paper_trading.db fica em ~3,4 GB com ~385 MB de páginas vivas
até alguém rodar --vacuum manualmente. Não é automático no ciclo de propósito:
VACUUM precisa de lock exclusivo, e com o ws_feed escrevendo continuamente
(history_writer commita a cada 100ms) rodar isso sem pausar os daemons trava
ou falha. Pause ws_feed antes de rodar --vacuum.
"""

import sqlite3
import sys
import time
from pathlib import Path

import click
from loguru import logger

DB_PATH      = Path("data/db/paper_trading.db")
REPORTS_DIR  = Path("outputs/reports")
RAW_MKT_DIR  = Path("data/raw/markets")

# P2-37: só padrões que sabemos ser artefato rotativo de ciclo (nome com
# timestamp, um arquivo novo a cada execução) — nunca um glob genérico do
# diretório inteiro. resolved_markets_*.parquet, por exemplo, é dataset de
# referência do ml_lab (congelado pelo ADR-008), não lixo de ciclo, e não
# está nesta lista de propósito.
REPORT_PATTERNS = [
    "eda_markets_*.csv", "top_markets_*.csv", "portfolio_*.csv",
    "signals_odds_*.csv", "signals_deribit_*.csv", "signals_structural_*.csv",
]
RAW_MARKET_PATTERNS = [
    "markets_all_*.parquet", "markets_incremental_*.parquet", "markets_partial_all_*.parquet",
]


def prune_price_history(days: int = 14, vacuum: bool = False, db_path: Path = DB_PATH,
                         dry_run: bool = False) -> dict:
    """
    Remove do price_history:
      1. Linhas mais antigas que `days` dias
      2. Linhas de book corrompidas (spread > 0.5 — só existiram pelo bug bids[0]/asks[0])

    dry_run=True conta em vez de apagar (SELECT COUNT(*) com o mesmo WHERE,
    sem commit, sem VACUUM) — pra inspecionar antes de confiar na retenção.

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

        old_where = "ts < datetime('now', ?)"
        corrupt_where = (
            "best_bid IS NOT NULL AND best_ask IS NOT NULL AND (best_ask - best_bid) > 0.5"
        )

        if dry_run:
            n_old = conn.execute(
                f"SELECT COUNT(*) FROM price_history WHERE {old_where}", (f"-{int(days)} days",)
            ).fetchone()[0]
            n_corrupt = conn.execute(
                f"SELECT COUNT(*) FROM price_history WHERE {corrupt_where}"
            ).fetchone()[0]
            n_after = n_before
            logger.info(
                f"[dry-run] price_history: {n_before:,} linhas — apagaria {n_old:,} antigas "
                f"(> {days}d) + {n_corrupt:,} corrompidas (nada foi removido)"
            )
        else:
            cur = conn.execute(f"DELETE FROM price_history WHERE {old_where}", (f"-{int(days)} days",))
            n_old = cur.rowcount

            cur = conn.execute(f"DELETE FROM price_history WHERE {corrupt_where}")
            n_corrupt = cur.rowcount
            conn.commit()

            n_after = conn.execute("SELECT COUNT(*) FROM price_history").fetchone()[0]
            logger.info(
                f"price_history: {n_before:,} → {n_after:,} linhas "
                f"(removidas: {n_old:,} antigas, {n_corrupt:,} corrompidas)"
            )

        if vacuum and not dry_run:
            logger.info("VACUUM em andamento (requer acesso exclusivo — pause os daemons)...")
            conn.isolation_level = None
            conn.execute("VACUUM")
            logger.info("VACUUM concluído.")

    finally:
        conn.close()

    size_after = db_path.stat().st_size
    if vacuum and not dry_run:
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


def prune_report_files(reports_dir: Path = REPORTS_DIR, raw_markets_dir: Path = RAW_MKT_DIR,
                        days: int = 14, dry_run: bool = False) -> dict:
    """
    Remove arquivos de padrões conhecidos de artefato rotativo de ciclo
    (REPORT_PATTERNS/RAW_MARKET_PATTERNS) mais velhos que `days` dias, por
    mtime. dry_run=True só lista/soma, não apaga.

    Returns:
        dict com contagem e bytes por diretório, e a lista de arquivos
        (nome only, não path completo) — útil pro --dry-run inspecionar.
    """
    cutoff = time.time() - days * 86400
    result = {"files_removed": 0, "bytes_removed": 0, "removed": []}

    for directory, patterns in (
        (reports_dir, REPORT_PATTERNS),
        (raw_markets_dir, RAW_MARKET_PATTERNS),
    ):
        if not directory.exists():
            continue
        for pattern in patterns:
            for path in directory.glob(pattern):
                try:
                    st = path.stat()
                except FileNotFoundError:
                    continue  # apagado por outro processo entre o glob e o stat
                if st.st_mtime >= cutoff:
                    continue
                result["files_removed"] += 1
                result["bytes_removed"] += st.st_size
                result["removed"].append(path.name)
                if not dry_run:
                    path.unlink(missing_ok=True)

    prefix = "[dry-run] " if dry_run else ""
    if result["files_removed"]:
        logger.info(
            f"{prefix}outputs/reports + data/raw/markets: "
            f"{result['files_removed']:,} arquivo(s) ({result['bytes_removed']/1e6:.0f} MB) "
            f"{'apagaria' if dry_run else 'removidos'} (> {days}d, padrões conhecidos)"
        )
    else:
        logger.info(f"{prefix}Nenhum arquivo de report/parquet além de {days}d encontrado.")

    return result


@click.command()
@click.option("--days",   default=14,   type=int, show_default=True,
              help="Reter apenas os últimos N dias (price_history e arquivos de report/parquet).")
@click.option("--vacuum", is_flag=True, default=False,
              help="Roda VACUUM após a limpeza (lento; requer daemons pausados).")
@click.option("--dry-run", is_flag=True, default=False,
              help="Só reporta o que seria removido — não apaga nada, não roda VACUUM.")
@click.option("--db",     default=str(DB_PATH), type=click.Path(), show_default=True,
              help="Caminho do banco SQLite.")
def main(days: int, vacuum: bool, dry_run: bool, db: str) -> None:
    """Retenção do price_history e dos arquivos de report/parquet por ciclo."""
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="<green>{time:HH:mm:ss}</green> | {message}")
    prune_price_history(days=days, vacuum=vacuum, db_path=Path(db), dry_run=dry_run)
    prune_report_files(days=days, dry_run=dry_run)


if __name__ == "__main__":
    main()
