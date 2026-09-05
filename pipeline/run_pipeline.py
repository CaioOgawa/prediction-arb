"""
run_pipeline.py
Orquestra o pipeline incremental de coleta de dados do Polymarket.
Detecta mercados novos vs. atualizados, registra cada run no SQLite
e evita re-download de dados já coletados.

Uso:
    uv run python -m pipeline.run_pipeline
    uv run python -m pipeline.run_pipeline --min-volume 10000
    uv run python -m pipeline.run_pipeline --dry-run   (não salva nada)
"""

import sys
import click
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger
from rich.console import Console
from rich.table import Table
from rich import box

import pandas as pd

from pipeline.db import init_db, get_connection
from pipeline.gamma_collector import fetch_markets, save_snapshot, upsert_to_db

console = Console()


def _load_known_markets() -> dict[str, dict]:
    """
    Carrega os mercados já conhecidos do SQLite.
    Retorna dict condition_id → {volume, liquidity, yes_price}.
    """
    conn = get_connection()
    rows = conn.execute(
        "SELECT condition_id, volume, liquidity, yes_price FROM market_registry"
    ).fetchall()
    conn.close()
    return {
        r["condition_id"]: {
            "volume":    r["volume"],
            "liquidity": r["liquidity"],
            "yes_price": r["yes_price"],
        }
        for r in rows
    }


def _diff_markets(
    fetched: pd.DataFrame,
    known: dict[str, dict],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Compara mercados recém-buscados com o estado atual do DB.

    Returns:
        new_df:     mercados que nunca foram vistos antes
        updated_df: mercados já conhecidos com volume/liquidez/preço alterados
        unchanged:  mercados sem nenhuma mudança detectável
    """
    new_rows, updated_rows, unchanged_rows = [], [], []

    for _, row in fetched.iterrows():
        cid = row.get("conditionId")
        if not cid:
            continue

        if cid not in known:
            new_rows.append(row)
        else:
            prev = known[cid]
            vol_changed  = abs(float(row.get("volume", 0) or 0) - float(prev["volume"] or 0)) > 1.0
            liq_changed  = abs(float(row.get("liquidity", 0) or 0) - float(prev["liquidity"] or 0)) > 0.01
            price_changed = (
                row.get("yes_price") is not None
                and prev["yes_price"] is not None
                and abs(float(row.get("yes_price", 0) or 0) - float(prev["yes_price"] or 0)) > 0.001
            )
            if vol_changed or liq_changed or price_changed:
                updated_rows.append(row)
            else:
                unchanged_rows.append(row)

    new_df       = pd.DataFrame(new_rows)       if new_rows       else pd.DataFrame()
    updated_df   = pd.DataFrame(updated_rows)   if updated_rows   else pd.DataFrame()
    unchanged_df = pd.DataFrame(unchanged_rows) if unchanged_rows else pd.DataFrame()
    return new_df, updated_df, unchanged_df


def _log_run_start(run_type: str) -> int:
    """Registra início de um run no SQLite. Retorna o run_id."""
    conn = get_connection()
    cur = conn.execute(
        "INSERT INTO pipeline_runs (run_type, started_at, status) VALUES (?, ?, ?)",
        (run_type, datetime.now(timezone.utc).isoformat(), "running"),
    )
    run_id = cur.lastrowid
    conn.commit()
    conn.close()
    return run_id


def _log_run_end(run_id: int, n_records: int, status: str = "ok", notes: str = "") -> None:
    """Atualiza o run com resultado final."""
    conn = get_connection()
    conn.execute(
        """UPDATE pipeline_runs
           SET finished_at = ?, n_records = ?, status = ?, notes = ?
           WHERE id = ?""",
        (datetime.now(timezone.utc).isoformat(), n_records, status, notes, run_id),
    )
    conn.commit()
    conn.close()


def run_incremental(
    min_volume: float = 5_000,
    dry_run: bool = False,
) -> dict:
    """
    Executa uma rodada do pipeline incremental.

    Fluxo:
        1. Carrega estado atual do DB
        2. Busca mercados ativos na Gamma API
        3. Compara: novo / atualizado / inalterado
        4. Salva apenas se não for dry_run
        5. Registra run no SQLite

    Returns:
        dict com métricas da execução: n_new, n_updated, n_unchanged, n_total
    """
    init_db()
    run_id = _log_run_start("incremental_markets")
    started_at = datetime.now(timezone.utc)

    try:
        # 1. Estado atual do DB
        known = _load_known_markets()
        logger.info(f"Mercados já no DB: {len(known):,}")

        # 2. Busca API
        logger.info("Buscando mercados na Gamma API...")
        fetched = fetch_markets(active=True, min_volume=min_volume)

        if fetched.empty:
            raise RuntimeError("Gamma API retornou zero mercados.")

        # 3. Diff
        new_df, updated_df, unchanged_df = _diff_markets(fetched, known)
        n_new       = len(new_df)
        n_updated   = len(updated_df)
        n_unchanged = len(unchanged_df)
        n_total     = len(fetched)

        elapsed = (datetime.now(timezone.utc) - started_at).total_seconds()
        logger.info(
            f"Diff: {n_new} novos | {n_updated} atualizados | "
            f"{n_unchanged} inalterados | total={n_total} ({elapsed:.1f}s)"
        )

        # 4. Persiste
        if not dry_run:
            save_snapshot(fetched, tag="incremental")
            upserted = upsert_to_db(fetched)
            _log_run_end(
                run_id,
                n_records=upserted,
                status="ok",
                notes=f"new={n_new} updated={n_updated} unchanged={n_unchanged}",
            )
        else:
            logger.info("dry_run=True — nenhum dado salvo.")
            _log_run_end(run_id, n_records=0, status="dry_run")

        return {
            "n_new": n_new,
            "n_updated": n_updated,
            "n_unchanged": n_unchanged,
            "n_total": n_total,
            "elapsed_s": elapsed,
            "run_id": run_id,
        }

    except Exception as exc:
        _log_run_end(run_id, n_records=0, status="error", notes=str(exc))
        raise


def _print_summary(metrics: dict) -> None:
    """Exibe resumo visual do run incremental."""
    console.print()
    table = Table(title="Pipeline Incremental — Resultado", box=box.ROUNDED)
    table.add_column("Métrica",    style="cyan")
    table.add_column("Valor",      justify="right")

    table.add_row("Mercados novos",       f"[green]{metrics['n_new']:,}[/green]")
    table.add_row("Mercados atualizados", f"[yellow]{metrics['n_updated']:,}[/yellow]")
    table.add_row("Sem mudança",          str(metrics["n_unchanged"]))
    table.add_row("Total coletados",      str(metrics["n_total"]))
    table.add_row("Tempo (s)",            f"{metrics['elapsed_s']:.1f}")
    table.add_row("Run ID",               str(metrics["run_id"]))
    console.print(table)

    # Histórico de runs do DB
    conn = get_connection()
    runs = conn.execute(
        "SELECT id, started_at, status, n_records, notes FROM pipeline_runs ORDER BY id DESC LIMIT 5"
    ).fetchall()
    conn.close()

    if runs:
        console.print()
        hist = Table(title="Últimos 5 Runs", box=box.SIMPLE)
        hist.add_column("ID",        width=4,  justify="right")
        hist.add_column("Iniciado",  width=20)
        hist.add_column("Status",    width=10)
        hist.add_column("Registros", justify="right")
        hist.add_column("Notas")
        for r in runs:
            color = "green" if r["status"] == "ok" else "red" if r["status"] == "error" else "yellow"
            hist.add_row(
                str(r["id"]),
                str(r["started_at"])[:19],
                f"[{color}]{r['status']}[/{color}]",
                str(r["n_records"]),
                str(r["notes"] or ""),
            )
        console.print(hist)


@click.command()
@click.option("--min-volume", default=5_000, type=float, show_default=True)
@click.option("--dry-run", is_flag=True, default=False, help="Simula sem salvar nada.")
def main(min_volume: float, dry_run: bool) -> None:
    """Pipeline incremental: detecta e persiste novos mercados e atualizações."""
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="<green>{time:HH:mm:ss}</green> | {message}")

    if dry_run:
        console.print("[yellow]Modo dry-run: nenhum dado será salvo.[/yellow]\n")

    metrics = run_incremental(min_volume=min_volume, dry_run=dry_run)
    _print_summary(metrics)


if __name__ == "__main__":
    main()
