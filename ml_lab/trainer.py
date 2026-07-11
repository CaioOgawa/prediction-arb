"""
trainer.py
Treina, avalia e compara todos os modelos do zoo automaticamente.
Gera relatório HTML interativo e salva o melhor modelo em outputs/models/.
"""

import sys
import time
import pickle
from pathlib import Path
from datetime import datetime

from collections import defaultdict

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    brier_score_loss, log_loss, roc_curve,
)
from sklearn.calibration import calibration_curve
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from loguru import logger
from rich.console import Console
from rich.table import Table
from rich import box

RESULTS_DIR = Path("ml_lab/results")
MODELS_DIR  = Path("outputs/models")
PLOTS_DIR   = Path("outputs/plots")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

console = Console()


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """
    Avalia um modelo com conjunto completo de métricas.
    Brier Score e Log Loss são as mais importantes para prediction markets
    (medem qualidade de probabilidades, não só acurácia).
    """
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    return {
        "accuracy":     round(accuracy_score(y_test, y_pred), 4),
        "f1_weighted":  round(f1_score(y_test, y_pred, average="weighted"), 4),
        "auc_roc":      round(roc_auc_score(y_test, y_prob), 4),
        "brier_score":  round(brier_score_loss(y_test, y_prob), 4),  # Menor = melhor
        "log_loss":     round(log_loss(y_test, y_prob), 4),          # Menor = melhor
    }


def walk_forward_cv(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
) -> pd.DataFrame:
    """
    Walk-forward cross-validation com expanding window temporal.
    X deve estar ordenado por data de resolução (mais antigo primeiro).

    A escala StandardScaler é feita dentro de cada fold — fit apenas no treino —
    para evitar data leakage entre folds.

    Args:
        X:        Features ordenadas temporalmente (sem escala)
        y:        Target (outcome)
        n_splits: Número de folds do TimeSeriesSplit

    Returns:
        DataFrame com AUC-ROC, Brier, LogLoss médio ± desvio padrão por modelo,
        rankeado por auc_roc_mean decrescente.
    """
    from model_zoo import get_all_models

    tscv         = TimeSeriesSplit(n_splits=n_splits)
    fold_results: dict[str, list[dict]] = defaultdict(list)

    console.print(f"\n[bold]Walk-Forward CV — {n_splits} folds (expanding window)[/bold]")
    console.print("[dim]Treina no passado, valida no futuro — sem data leakage temporal[/dim]\n")

    for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train_raw = X.iloc[train_idx].copy()
        X_test_raw  = X.iloc[test_idx].copy()
        y_train     = y.iloc[train_idx]
        y_test      = y.iloc[test_idx]

        # Escala dentro do fold: fit apenas no treino
        scaler = StandardScaler()
        X_tr = pd.DataFrame(
            scaler.fit_transform(X_train_raw),
            columns=X.columns,
        )
        X_te = pd.DataFrame(
            scaler.transform(X_test_raw),
            columns=X.columns,
        )

        console.print(
            f"  Fold {fold_idx + 1}/{n_splits}: "
            f"treino={len(X_tr):,} | teste={len(X_te):,}",
            end="  ",
        )

        models     = get_all_models()
        fold_aucs  = []
        for name, model in models.items():
            try:
                model.fit(X_tr, y_train)
                m        = evaluate_model(model, X_te, y_test)
                m["fold"] = fold_idx
                fold_results[name].append(m)
                fold_aucs.append(f"{name.split('_')[0][:6]}={m['auc_roc']:.3f}")
            except Exception as e:
                logger.warning(f"Fold {fold_idx} | {name}: {e}")

        console.print("  ".join(fold_aucs))

    # Agrega resultados por modelo
    rows = []
    for name, metrics_list in fold_results.items():
        fdf = pd.DataFrame(metrics_list)
        row: dict = {"model": name}
        for metric in ["auc_roc", "brier_score", "log_loss", "accuracy", "f1_weighted"]:
            if metric in fdf.columns:
                row[f"{metric}_mean"] = round(float(fdf[metric].mean()), 4)
                row[f"{metric}_std"]  = round(float(fdf[metric].std()),  4)
        rows.append(row)

    return (
        pd.DataFrame(rows)
        .sort_values("auc_roc_mean", ascending=False)
        .reset_index(drop=True)
    )


def print_wf_results_table(wf_df: pd.DataFrame) -> None:
    """Exibe tabela de resultados walk-forward com média ± desvio padrão."""
    console.print("\n[bold cyan]══════════════════════════════════════════════════════[/bold cyan]")
    console.print("[bold cyan]   WALK-FORWARD CV — POLYMARKET ML (média ± std)      [/bold cyan]")
    console.print("[bold cyan]══════════════════════════════════════════════════════[/bold cyan]\n")

    table = Table(box=box.ROUNDED, show_lines=True)
    table.add_column("#",         width=3,  justify="right")
    table.add_column("Modelo",    style="cyan", width=28)
    table.add_column("AUC-ROC",   justify="right", style="bold")
    table.add_column("Brier ↓",   justify="right")
    table.add_column("LogLoss ↓", justify="right")
    table.add_column("Accuracy",  justify="right")

    for i, row in wf_df.iterrows():
        color = "green" if i == 0 else "white"
        table.add_row(
            str(i + 1),
            row["model"],
            f"[{color}]{row['auc_roc_mean']:.4f} ±{row['auc_roc_std']:.4f}[/{color}]",
            f"{row['brier_score_mean']:.4f} ±{row['brier_score_std']:.4f}",
            f"{row['log_loss_mean']:.4f} ±{row['log_loss_std']:.4f}",
            f"{row['accuracy_mean']:.4f} ±{row['accuracy_std']:.4f}",
        )

    console.print(table)

    best = wf_df.iloc[0]
    console.print(f"\n[bold green]Melhor modelo (WF CV):[/bold green] {best['model']}")
    console.print(f"  AUC-ROC: {best['auc_roc_mean']:.4f} ± {best['auc_roc_std']:.4f}")
    console.print(f"  Brier:   {best['brier_score_mean']:.4f} ± {best['brier_score_std']:.4f}")
    console.print(
        "\n[dim]Walk-forward: cada fold treina em mercados mais antigos e valida "
        "nos mais recentes. Sem vazamento temporal.[/dim]\n"
    )


def compare_all_models(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> tuple[pd.DataFrame, dict]:
    """
    Treina e compara todos os modelos do zoo.
    Retorna DataFrame de resultados (rankeado por AUC-ROC) e dict de modelos treinados.
    """
    from model_zoo import get_all_models
    models = get_all_models()

    results = []
    trained = {}

    console.print(f"\n[bold]Treinando {len(models)} modelos...[/bold]\n")

    for name, model in models.items():
        console.print(f"  [cyan]→[/cyan] {name}...", end=" ")
        t0 = time.time()

        try:
            model.fit(X_train, y_train)
            elapsed = time.time() - t0
            metrics = evaluate_model(model, X_test, y_test)
            metrics["model"]      = name
            metrics["train_time"] = round(elapsed, 1)
            results.append(metrics)
            trained[name] = model
            console.print(
                f"[green]AUC={metrics['auc_roc']:.4f}[/green]  "
                f"Brier={metrics['brier_score']:.4f}  "
                f"({elapsed:.1f}s)"
            )
        except Exception as e:
            console.print(f"[red]ERRO: {e}[/red]")

    results_df = (
        pd.DataFrame(results)
        .sort_values("auc_roc", ascending=False)
        .reset_index(drop=True)
    )
    return results_df, trained


def print_results_table(results_df: pd.DataFrame) -> None:
    """Exibe tabela de comparação no terminal."""
    console.print("\n[bold cyan]══════════════════════════════════════════[/bold cyan]")
    console.print("[bold cyan]   COMPARAÇÃO DE MODELOS — POLYMARKET ML   [/bold cyan]")
    console.print("[bold cyan]══════════════════════════════════════════[/bold cyan]\n")

    table = Table(box=box.ROUNDED, show_lines=True)
    table.add_column("#",           width=3,  justify="right")
    table.add_column("Modelo",      style="cyan",  width=26)
    table.add_column("AUC-ROC",     justify="right", style="bold")
    table.add_column("Brier↓",      justify="right")
    table.add_column("Log Loss↓",   justify="right")
    table.add_column("Accuracy",    justify="right")
    table.add_column("F1",          justify="right")
    table.add_column("Tempo(s)",    justify="right")

    for i, row in results_df.iterrows():
        auc_color = "green" if i == 0 else "white"
        table.add_row(
            str(i + 1),
            row["model"],
            f"[{auc_color}]{row['auc_roc']:.4f}[/{auc_color}]",
            str(row["brier_score"]),
            str(row["log_loss"]),
            str(row["accuracy"]),
            str(row["f1_weighted"]),
            str(row["train_time"]),
        )
    console.print(table)

    best = results_df.iloc[0]
    console.print(f"\n[bold green]Melhor modelo:[/bold green] {best['model']}")
    console.print(f"  AUC-ROC:     {best['auc_roc']:.4f}  (1.0 = perfeito, 0.5 = aleatório)")
    console.print(f"  Brier Score: {best['brier_score']:.4f}  (0.0 = perfeito, 0.25 = aleatório)")


def save_best_model(
    trained: dict,
    results_df: pd.DataFrame,
    scaler=None,
    feature_names: list[str] | None = None,
) -> Path:
    """
    Persiste o melhor modelo em outputs/models/ como bundle pickle.
    O bundle inclui: model, scaler, feature_names — necessários para inferência
    no signal_generator sem ter que reconstruir o pipeline de features.
    """
    best_name  = results_df.iloc[0]["model"]
    best_model = trained[best_name]
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = MODELS_DIR / f"best_model_{best_name}_{ts}.pkl"

    bundle = {
        "model":         best_model,
        "scaler":        scaler,
        "feature_names": feature_names or [],
        "model_name":    best_name,
        "trained_at":    ts,
        "auc_roc":       float(results_df.iloc[0]["auc_roc"]),
        "brier_score":   float(results_df.iloc[0]["brier_score"]),
    }
    with open(path, "wb") as f:
        pickle.dump(bundle, f)

    # Symlink latest para facilitar carregamento no signal_generator
    latest = MODELS_DIR / "best_model_latest.pkl"
    if latest.exists() or latest.is_symlink():
        latest.unlink()
    latest.symlink_to(path.name)

    logger.info(f"Melhor modelo salvo: {path}")
    logger.info(f"Symlink atualizado: {latest}")
    return path


def build_html_report(
    results_df: pd.DataFrame,
    trained: dict,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Path:
    """
    Gera relatório HTML com 4 gráficos:
      1. Comparação de métricas (barras)
      2. Curvas ROC de todos os modelos
      3. Curva de calibração do melhor modelo
      4. Feature importances (se disponível)
    """
    figs = []

    # 1. Comparação de métricas
    fig1 = make_subplots(rows=1, cols=3, subplot_titles=["AUC-ROC ↑", "Brier Score ↓", "Log Loss ↓"])
    colors = ["#2ecc71" if i == 0 else "#3498db" for i in range(len(results_df))]

    fig1.add_trace(go.Bar(
        x=results_df["auc_roc"], y=results_df["model"],
        orientation="h", marker_color=colors, name="AUC-ROC",
    ), row=1, col=1)
    fig1.add_trace(go.Bar(
        x=results_df["brier_score"],
        y=results_df.sort_values("brier_score")["model"],
        orientation="h", marker_color=list(reversed(colors)), name="Brier",
    ), row=1, col=2)
    fig1.add_trace(go.Bar(
        x=results_df["log_loss"],
        y=results_df.sort_values("log_loss")["model"],
        orientation="h", marker_color=list(reversed(colors)), name="Log Loss",
    ), row=1, col=3)
    fig1.update_layout(height=400, title="Comparação de Métricas — Todos os Modelos", showlegend=False)
    figs.append(("1. Comparação de Métricas", fig1))

    # 2. Curvas ROC
    fig2 = go.Figure()
    fig2.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                   line=dict(dash="dash", color="gray"))
    palette = px.colors.qualitative.Plotly
    for i, (name, model) in enumerate(trained.items()):
        try:
            y_prob = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            auc = roc_auc_score(y_test, y_prob)
            fig2.add_trace(go.Scatter(
                x=fpr, y=tpr,
                name=f"{name} (AUC={auc:.3f})",
                line=dict(color=palette[i % len(palette)]),
            ))
        except Exception:
            pass
    fig2.update_layout(
        title="Curvas ROC — Todos os Modelos",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        height=500,
    )
    figs.append(("2. Curvas ROC", fig2))

    # 3. Curva de calibração do melhor modelo
    best_name  = results_df.iloc[0]["model"]
    best_model = trained.get(best_name)
    if best_model:
        try:
            y_prob_best = best_model.predict_proba(X_test)[:, 1]
            fraction_pos, mean_pred = calibration_curve(y_test, y_prob_best, n_bins=10)

            fig3 = go.Figure()
            fig3.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                           line=dict(dash="dash", color="gray"),
                           name="Calibração perfeita")
            fig3.add_trace(go.Scatter(
                x=mean_pred, y=fraction_pos,
                mode="lines+markers",
                name=f"{best_name}",
                line=dict(color="#2ecc71"),
                marker=dict(size=8),
            ))
            fig3.update_layout(
                title=f"Curva de Calibração — {best_name}<br><sub>Quanto mais próxima da diagonal, mais calibrado</sub>",
                xaxis_title="Probabilidade predita (modelo)",
                yaxis_title="Frequência real de YES",
                height=450,
            )
            figs.append(("3. Calibração do Melhor Modelo", fig3))
        except Exception:
            pass

    # 4. Feature importances
    best_for_fi = None
    for name in ["xgboost", "lightgbm", "xgboost_calibrated", "lightgbm_calibrated"]:
        m = trained.get(name)
        if m is None:
            continue
        # CalibratedClassifierCV wraps o estimador base
        base = getattr(m, "estimator", m)
        if hasattr(base, "feature_importances_"):
            best_for_fi = (name, base)
            break

    if best_for_fi:
        name_fi, model_fi = best_for_fi
        importances = pd.Series(
            model_fi.feature_importances_,
            index=X_test.columns,
        ).sort_values(ascending=True)

        fig4 = px.bar(
            x=importances.values, y=importances.index,
            orientation="h",
            title=f"Feature Importances — {name_fi}",
            labels={"x": "Importância", "y": "Feature"},
            color=importances.values,
            color_continuous_scale="Viridis",
        )
        fig4.update_layout(height=400, coloraxis_showscale=False)
        figs.append(("4. Feature Importances", fig4))

    # Monta HTML
    html_parts = [f"""
    <html><head>
      <meta charset="utf-8">
      <title>Polymarket Quant — ML Lab</title>
      <style>
        body {{ font-family:'Segoe UI',sans-serif; background:#0d1117; color:#e6edf3; margin:0; padding:20px; }}
        h1   {{ color:#58a6ff; border-bottom:1px solid #30363d; padding-bottom:12px; }}
        h2   {{ color:#79c0ff; margin-top:40px; font-size:1.1em; }}
        .plot-container {{ background:#161b22; border-radius:8px; padding:10px;
                           margin:20px 0; border:1px solid #30363d; }}
        .meta {{ color:#8b949e; font-size:0.9em; }}
        .badge {{ display:inline-block; background:#21262d; border:1px solid #30363d;
                  border-radius:20px; padding:3px 10px; margin:4px; font-size:0.85em; color:#8b949e; }}
      </style>
    </head><body>
    <h1>Polymarket Quant — ML Lab</h1>
    <p class="meta">
      Gerado em {datetime.now().strftime('%Y-%m-%d %H:%M')} &nbsp;|&nbsp;
      <span class="badge">{len(results_df)} modelos comparados</span>
      <span class="badge">Melhor: {results_df.iloc[0]['model']}</span>
      <span class="badge">AUC={results_df.iloc[0]['auc_roc']}</span>
    </p>
    """]

    for title, fig in figs:
        fig.update_layout(
            paper_bgcolor="#161b22", plot_bgcolor="#0d1117",
            font=dict(color="#e6edf3"), title_font=dict(color="#58a6ff"),
        )
        html_parts.append(f'<div class="plot-container"><h2>{title}</h2>')
        html_parts.append(fig.to_html(full_html=False,
                                      include_plotlyjs="cdn" if title == figs[0][0] else False))
        html_parts.append("</div>")

    html_parts.append("</body></html>")
    path = PLOTS_DIR / "ml_lab_report.html"
    path.write_text("\n".join(html_parts), encoding="utf-8")
    return path
