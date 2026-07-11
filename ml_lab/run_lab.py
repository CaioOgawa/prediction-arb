"""
run_lab.py

⚠️ CONGELADO (ADR-008, 2026-07): o projeto atua apenas com stat arb.
O modo ml foi removido dos CLIs de sinais; este módulo permanece no repo
para eventual retomada, mas está fora do fluxo de produção.

Ponto de entrada do ML Lab — executa o pipeline completo:
  1. Busca mercados resolvidos (dataset de treino)
  2. Constrói features + labels
  3. Walk-forward CV para comparação honesta de modelos (default)
  4. Treina modelo final no split temporal (80% antigo / 20% recente)
  5. Gera relatório HTML e salva o melhor modelo

Walk-forward CV vs. random split:
  - Random split: pode treinar em mercados "futuros" → AUC inflado ~0.05-0.10
  - Walk-forward: treina no passado, valida no futuro → métricas confiáveis
  Use --no-walk-forward apenas para debugging rápido.

Uso:
    uv run python ml_lab/run_lab.py
    uv run python ml_lab/run_lab.py --min-volume 5000
    uv run python ml_lab/run_lab.py --no-walk-forward  # split aleatório (debug)
    uv run python ml_lab/run_lab.py --open             # abre relatório no browser
"""

import sys
import webbrowser
import click
from pathlib import Path

from loguru import logger
from rich.console import Console

sys.path.insert(0, str(Path(__file__).parent.parent / "pipeline"))

console = Console()


@click.command()
@click.option("--min-volume",       default=1_000,  type=float, show_default=True,
              help="Volume mínimo dos mercados de treino (USDC).")
@click.option("--max-markets",      default=20_000, type=int,   show_default=True,
              help="Máximo de mercados a buscar da API.")
@click.option("--test-size",        default=0.2,    type=float, show_default=True,
              help="Fração do dataset para holdout final (0.2 = últimos 20% por data).")
@click.option("--wf-splits",        default=5,      type=int,   show_default=True,
              help="Número de folds no walk-forward CV.")
@click.option("--walk-forward/--no-walk-forward", default=True, show_default=True,
              help="Usa walk-forward CV (recomendado). --no-walk-forward para split aleatório.")
@click.option("--open", "open_browser", is_flag=True, default=False,
              help="Abre relatório HTML no browser ao finalizar.")
@click.option("--force-refresh",    is_flag=True, default=False,
              help="Ignora cache e rebaixa dados da API.")
def main(
    min_volume: float,
    max_markets: int,
    test_size: float,
    wf_splits: int,
    walk_forward: bool,
    open_browser: bool,
    force_refresh: bool,
) -> None:
    """ML Lab: treina e compara modelos para prediction markets do Polymarket."""
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="<green>{time:HH:mm:ss}</green> | {message}")

    console.print("\n[bold]Polymarket Quant — ML Lab[/bold]")
    mode_label = "Walk-Forward CV" if walk_forward else "Random Split (debug)"
    console.print(f"Fase 3: treino e comparação de modelos  |  modo: [cyan]{mode_label}[/cyan]\n")

    # ── 1. Dataset ────────────────────────────────────────────
    from dataset_builder import (
        fetch_resolved_markets,
        build_train_test_split,
        build_features_matrix,
    )
    console.print("[bold]Etapa 1/4[/bold] — Coletando mercados resolvidos...")
    df = fetch_resolved_markets(
        min_volume=min_volume,
        max_events=max_markets,
        force_refresh=force_refresh,
    )

    if len(df) < 100:
        console.print(
            f"[red]Dataset muito pequeno ({len(df)} amostras). "
            f"Tente reduzir --min-volume.[/red]"
        )
        sys.exit(1)

    console.print(f"  {len(df):,} mercados resolvidos | YES rate: {df['outcome'].mean():.1%}\n")

    # ── 2. Features ───────────────────────────────────────────
    console.print("[bold]Etapa 2/4[/bold] — Preparando features...")

    from trainer import (
        compare_all_models,
        walk_forward_cv,
        print_results_table,
        print_wf_results_table,
        save_best_model,
        build_html_report,
    )

    if walk_forward:
        # Matriz sem escala e ordenada por data — walk_forward_cv escala dentro de cada fold
        X, y, feature_names, _ = build_features_matrix(df, scale=False)
        console.print(f"  Features ({len(feature_names)}): {feature_names}")
        console.print(f"  Período: {df['end_date'].min()} → {df['end_date'].max()}\n")

        # ── 3. Walk-forward CV ────────────────────────────────
        console.print("[bold]Etapa 3/4[/bold] — Walk-Forward Cross-Validation...")
        wf_results = walk_forward_cv(X, y, n_splits=wf_splits)
        print_wf_results_table(wf_results)

        # Salva CSV de resultados WF
        ts       = __import__("datetime").datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = Path("ml_lab/results") / f"wf_cv_{ts}.csv"
        wf_results.to_csv(csv_path, index=False)

        # ── 4. Modelo final: split temporal (train antigo / test recente) ──
        console.print("[bold]Etapa 4/4[/bold] — Treinando modelo final (split temporal)...")
        # Reutiliza X e y já ordenados; split no índice
        split_idx = int(len(X) * (1 - test_size))
        X_train_raw, X_test_raw = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train,     y_test     = y.iloc[:split_idx], y.iloc[split_idx:]

        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        import pandas as pd
        X_train = pd.DataFrame(
            scaler.fit_transform(X_train_raw), columns=feature_names
        )
        X_test = pd.DataFrame(
            scaler.transform(X_test_raw), columns=feature_names
        )
        console.print(
            f"  Treino: {len(X_train):,} mercados mais antigos  |  "
            f"Teste: {len(X_test):,} mais recentes\n"
        )

        results_df, trained_models = compare_all_models(X_train, X_test, y_train, y_test)
        print_results_table(results_df)

        # Melhor modelo do WF (mais confiável) → seleciona pelo nome
        best_wf_name = wf_results.iloc[0]["model"]
        if best_wf_name in trained_models:
            # Reordena results_df para colocar o campeão do WF no topo
            results_df = results_df.set_index("model")
            if best_wf_name in results_df.index:
                idx_order = [best_wf_name] + [m for m in results_df.index if m != best_wf_name]
                results_df = results_df.loc[idx_order].reset_index()
            else:
                results_df = results_df.reset_index()
        else:
            pass  # fallback: usa o melhor do split final

    else:
        # Modo legado: split aleatório
        X_train, X_test, y_train, y_test, feature_names, scaler = build_train_test_split(
            df, test_size=test_size, scale=True,
        )
        trained_models = {}
        console.print(f"  Features: {feature_names}\n")

        console.print("[bold]Etapa 3/4[/bold] — Treinando modelos (split aleatório)...")
        results_df, trained_models = compare_all_models(X_train, X_test, y_train, y_test)

        ts       = __import__("datetime").datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = Path("ml_lab/results") / f"model_comparison_{ts}.csv"
        results_df.to_csv(csv_path, index=False)
        print_results_table(results_df)

        console.print("[bold]Etapa 4/4[/bold] — Salvando resultados...")

    # ── Salva melhor modelo e gera relatório ──────────────────
    console.print("\n[bold]Salvando modelo e relatório...[/bold]")
    model_path  = save_best_model(trained_models, results_df, scaler=scaler, feature_names=feature_names)
    report_path = build_html_report(results_df, trained_models, X_test, y_test)

    console.print(f"\n[bold green]Concluído![/bold green]")
    console.print(f"  Melhor modelo:  {model_path}")
    console.print(f"  Relatório HTML: {report_path}")
    console.print(f"  Resultados CSV: {csv_path}")
    console.print(f"\n[dim]Abra no browser: open {report_path.resolve()}[/dim]")

    if open_browser:
        webbrowser.open(f"file://{report_path.resolve()}")


if __name__ == "__main__":
    main()
