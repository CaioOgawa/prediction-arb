"""
signal_generator.py
Gera sinais de trading com duas estratégias:

  MODO ODDS (primário, esportes):
    Edge = divergência entre preço Polymarket e probabilidade implícita de bookmakers.
    Requer ODDS_API_KEY no .env e dados do odds_collector.py.
    É o modo com edge real e calibrado.

  MODO ML (auxiliar, outros mercados):
    Edge = P(modelo) - yes_price. Útil para mercados sem odds externas disponíveis.
    ATENÇÃO: o modelo atual tem data leakage — use como indicativo, não como verdade.

Fluxo (modo odds):
  1. Carrega último Parquet de data/raw/odds/ (gerado pelo odds_collector.py)
  2. Filtra por divergência mínima, liquidez e tempo até resolução
  3. Exporta sinais rankeados em CSV + exibe no terminal

Fluxo (modo ml):
  1. Carrega o modelo mais recente de outputs/models/
  2. Carrega mercados ativos (último Parquet de data/raw/markets/)
  3. Reconstrói as features usadas no treino
  4. Filtra por edge mínimo, liquidez, volume e tempo até resolução
  5. Exporta sinais rankeados em CSV + exibe no terminal
"""

import pickle
import sys
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from loguru import logger
from rich.console import Console
from rich.table import Table
from rich import box

RAW_DIR      = Path("data/raw/markets")
RAW_ODDS_DIR = Path("data/raw/odds")
MODELS_DIR   = Path("outputs/models")
REPORTS_DIR  = Path("outputs/reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

console = Console()

# Features base sem leakage — lidas do bundle do modelo em runtime
# (mantido aqui só como fallback se o bundle for antigo)
FEATURE_COLS_LEGACY = [
    "yes_price", "log_volume", "log_liquidity", "log_volume24hr", "log_volume1wk",
    "volume_recency_ratio", "spread", "days_ran", "has_resolution_source",
    "is_extreme_price", "price_1d_change", "price_1w_change",
]


def load_best_model(model_path: Path | None = None) -> tuple[dict, Path]:
    """
    Carrega o bundle do melhor modelo de outputs/models/.
    Prefere o symlink 'best_model_latest.pkl'; fallback para o arquivo mais recente.

    Returns:
        (bundle_dict, model_path)
        bundle_dict tem: 'model', 'scaler', 'feature_names', 'auc_roc', 'brier_score'
    """
    if model_path is None:
        latest = MODELS_DIR / "best_model_latest.pkl"
        if latest.exists():
            model_path = latest
        else:
            candidates = sorted(MODELS_DIR.glob("best_model_*.pkl"), reverse=True)
            if not candidates:
                raise FileNotFoundError(
                    f"Nenhum modelo encontrado em {MODELS_DIR}. "
                    "Execute primeiro: uv run python ml_lab/run_lab.py"
                )
            model_path = candidates[0]

    logger.info(f"Carregando modelo: {model_path.name}")
    with open(model_path, "rb") as f:
        bundle = pickle.load(f)

    # Compatibilidade com modelos antigos (pickle direto, sem bundle)
    if not isinstance(bundle, dict):
        bundle = {"model": bundle, "scaler": None, "feature_names": FEATURE_COLS_LEGACY}
        logger.warning("Modelo antigo (sem bundle) — usando FEATURE_COLS_LEGACY com leakage!")

    return bundle, model_path


def load_active_markets() -> pd.DataFrame:
    """
    Carrega o snapshot mais recente de mercados ativos.
    Usa o Parquet com prefixo 'markets_all_' ou 'markets_incremental_'.
    """
    candidates = sorted(
        list(RAW_DIR.glob("markets_all_*.parquet")) +
        list(RAW_DIR.glob("markets_incremental_*.parquet")),
        key=lambda p: p.stat().st_mtime,  # ordena por tempo de modificação, não por nome
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            f"Nenhum arquivo de mercados em {RAW_DIR}. "
            "Execute primeiro: uv run python pipeline/fetch_markets.py"
        )
    path = candidates[0]
    logger.info(f"Carregando mercados: {path.name}")
    df = pd.read_parquet(path)
    logger.info(f"  {len(df):,} mercados carregados")
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reconstrói as mesmas features usadas no treino (dataset_builder.FEATURE_COLS).
    Nomes e transformações devem ser idênticos ao treinamento para evitar feature drift.
    """
    out = pd.DataFrame(index=df.index)

    # --- Preço ---
    out["yes_price"] = pd.to_numeric(df.get("yes_price", df.get("lastTradePrice", 0.5)), errors="coerce").fillna(0.5)

    # --- Volume (log) ---
    volume    = pd.to_numeric(df.get("volume",    0), errors="coerce").fillna(0)
    liquidity = pd.to_numeric(df.get("liquidity", 0), errors="coerce").fillna(0)
    v24h      = pd.to_numeric(df.get("volume24hr", 0), errors="coerce").fillna(0)
    v1wk      = pd.to_numeric(df.get("volume1wk",  0), errors="coerce").fillna(0)

    out["log_volume"]    = np.log1p(volume)
    out["log_liquidity"] = np.log1p(liquidity)
    out["log_volume24hr"]= np.log1p(v24h)
    out["log_volume1wk"] = np.log1p(v1wk)

    # --- Recência de volume ---
    end_raw   = pd.to_datetime(df.get("endDate"),   errors="coerce", utc=True)
    start_raw = pd.to_datetime(df.get("startDate", df.get("createdAt")), errors="coerce", utc=True)
    now       = pd.Timestamp.now(tz="UTC")

    days_ran  = (end_raw - start_raw).dt.days.fillna(-1).clip(lower=0)
    out["days_ran"] = days_ran

    # Proporção do volume recente em relação à média diária histórica
    avg_daily = volume / days_ran.replace(0, 1)
    recency   = v24h / (avg_daily + 1e-9)
    out["volume_recency_ratio"] = recency.clip(upper=100.0)

    # --- Spread ---
    out["spread"] = pd.to_numeric(df.get("spread", 0), errors="coerce").fillna(0)

    # --- Flags ---
    out["has_resolution_source"] = df.get("resolutionSource", "").apply(
        lambda x: int(bool(x and str(x).strip()))
    )
    out["is_extreme_price"] = ((out["yes_price"] > 0.85) | (out["yes_price"] < 0.10)).astype(int)

    # --- Variações de preço ---
    out["price_1d_change"] = pd.to_numeric(df.get("oneDayPriceChange",  0), errors="coerce").fillna(0)
    out["price_1w_change"] = pd.to_numeric(df.get("oneWeekPriceChange", 0), errors="coerce").fillna(0)

    # Imputa NaN com mediana (como no treinamento)
    for col in out.columns:
        if out[col].isnull().any():
            out[col] = out[col].fillna(out[col].median())

    return out[FEATURE_COLS_LEGACY]


def compute_signals(
    df: pd.DataFrame,
    features: pd.DataFrame,
    model,
    edge_threshold: float = 0.04,
    min_liquidity: float = 5_000,
    min_volume24h: float = 500,
    min_days_left: int = 1,
) -> pd.DataFrame:
    """
    Aplica o modelo, calcula edge e filtra sinais relevantes.

    Args:
        edge_threshold:  edge mínimo (em probabilidade) para considerar sinal
        min_liquidity:   liquidez mínima em USDC
        min_volume24h:   volume nas últimas 24h mínimo em USDC
        min_days_left:   dias mínimos até resolução (evita mercados expirando hoje)

    Returns:
        DataFrame de sinais filtrados, rankeados por |edge| decrescente.
    """
    # Predição de probabilidade P(YES)
    prob_yes = model.predict_proba(features)[:, 1]

    now     = pd.Timestamp.now(tz="UTC")
    end_raw = pd.to_datetime(df.get("endDate"), errors="coerce", utc=True)
    days_to_end = (end_raw - now).dt.days.fillna(-1)

    yes_price = pd.to_numeric(df.get("yes_price", df.get("lastTradePrice", 0.5)), errors="coerce").fillna(0.5)
    liquidity = pd.to_numeric(df.get("liquidity", 0), errors="coerce").fillna(0)
    volume24h = pd.to_numeric(df.get("volume24hr", 0), errors="coerce").fillna(0)

    edge = prob_yes - yes_price.values

    signals = pd.DataFrame({
        "condition_id":   df["conditionId"].values,
        "question":       df.get("question", pd.Series("", index=df.index)).values,
        "category":       df.get("category", pd.Series("", index=df.index)).values,
        "yes_price":      yes_price.values,
        "prob_yes":       prob_yes.round(4),
        "edge":           edge.round(4),
        "abs_edge":       np.abs(edge).round(4),
        "direction":      np.where(edge > 0, "BUY_YES", "BUY_NO"),
        "liquidity":      liquidity.values,
        "volume_24h":     volume24h.values,
        "days_to_end":    days_to_end.values,
        "end_date":       end_raw.dt.strftime("%Y-%m-%d").values,
    })

    # Filtros de qualidade
    mask = (
        (signals["abs_edge"]    >= edge_threshold) &
        (signals["liquidity"]   >= min_liquidity)  &
        (signals["volume_24h"]  >= min_volume24h)  &
        (signals["days_to_end"] >= min_days_left)
    )
    filtered = signals[mask].sort_values("abs_edge", ascending=False).reset_index(drop=True)
    return filtered


def print_signals_table(signals: pd.DataFrame, top_n: int = 20) -> None:
    """Exibe tabela de sinais no terminal via Rich."""
    console.print("\n[bold cyan]══════════════════════════════════════════════════════════════════[/bold cyan]")
    console.print("[bold cyan]   SINAIS DE TRADING — POLYMARKET QUANT                           [/bold cyan]")
    console.print("[bold cyan]══════════════════════════════════════════════════════════════════[/bold cyan]\n")

    if signals.empty:
        console.print("[yellow]Nenhum sinal encontrado com os filtros atuais.[/yellow]")
        console.print("Tente reduzir --edge-threshold ou --min-liquidity.\n")
        return

    display = signals.head(top_n)
    table = Table(box=box.ROUNDED, show_lines=True)
    table.add_column("#",           width=3,  justify="right")
    table.add_column("Questão",     width=45)
    table.add_column("Direção",     width=9,  justify="center")
    table.add_column("P(Mercado)", width=10,  justify="right")
    table.add_column("P(Modelo)",  width=10,  justify="right")
    table.add_column("Edge",        width=8,  justify="right", style="bold")
    table.add_column("Liquidez",   width=10,  justify="right")
    table.add_column("Vol24h",      width=10,  justify="right")
    table.add_column("Dias",        width=5,  justify="right")

    for i, row in display.iterrows():
        edge_color = "green" if row["edge"] > 0 else "red"
        dir_color  = "green" if row["direction"] == "BUY_YES" else "magenta"
        table.add_row(
            str(i + 1),
            str(row["question"])[:44],
            f"[{dir_color}]{row['direction']}[/{dir_color}]",
            f"{row['yes_price']:.3f}",
            f"{row['prob_yes']:.3f}",
            f"[{edge_color}]{row['edge']:+.3f}[/{edge_color}]",
            f"${row['liquidity']:,.0f}",
            f"${row['volume_24h']:,.0f}",
            str(int(row["days_to_end"])),
        )

    console.print(table)
    console.print(f"\n[bold]Total de sinais:[/bold] {len(signals):,}  |  Exibindo top {min(top_n, len(signals))}")
    console.print(f"[bold]Edge médio (abs):[/bold] {signals['abs_edge'].mean():.3f}")
    console.print(f"[bold]Liquidez média:[/bold] ${signals['liquidity'].mean():,.0f}\n")


def save_signals(signals: pd.DataFrame, tag: str = "") -> Path:
    """Salva sinais em CSV com timestamp."""
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_{tag}" if tag else ""
    path = REPORTS_DIR / f"signals{suffix}_{ts}.csv"
    signals.to_csv(path, index=False)
    logger.info(f"Sinais salvos: {path}")
    return path


def _confidence_score(
    abs_divergence: float,
    hours_left: float,
    liquidity: float,
    overround: float = 1.02,
    iv: float | None = None,
    n_books: int = 1,
    consensus_spread: float = 0.0,
) -> float:
    """
    Score de confiança composto (0–1) que escala o tamanho de posição.

    Componentes:
      - divergence_score:   maior divergência → maior confiança, mas com teto
        (acima de 15pp o ganho marginal é pequeno e pode indicar erro de modelo)
      - time_score:         muito pouco tempo até expiração → IV intraday é ruidosa
      - liquidity_score:    liquidez Polymarket mínima para execução confiável
      - book_score:         penaliza bookmakers com overround alto (e.g. DraftKings ~1.06)
        Pinnacle ~1.02 → 1.0 | DraftKings ~1.06 → ~0.5 | acima de 1.10 → 0.1
      - iv_score:           [modo deribit] IV alta = B-S fair_prob menos confiável
        IV < 70% → 1.0 | IV 100% → 0.50 | IV 130%+ → 0.30
      - consensus_score:    [modo odds] quantos books concordam na mesma direção?
        n_books=1 → 0.70 | n_books=2 → 0.85 | n_books>=3 → 1.0
        consensus_spread alto (books divergem) → penaliza adicionalmente

    O produto dos componentes dá um score conservador que evita over-conviction.
    """
    # Divergência: escala de 0.03 → 0.0 a 0.15 → 1.0, com teto em 1.0
    div_score = float(np.clip((abs_divergence - 0.03) / (0.15 - 0.03), 0, 1))

    # Tempo: < 2h é muito arriscado (γ alto), escala até 24h onde já é estável.
    # Mercados muito longos (> 30 dias = 720h) recebem penalidade crescente porque
    # o modelo B-S risk-neutral (r=0, sem drift) subestima cada vez mais a
    # probabilidade real à medida que o horizonte aumenta.
    # < 2h → 0.1 | 24h → 1.0 | 720h (30d) → 1.0 | 4380h (6m) → 0.7 | 8760h (1a) → 0.5
    short_penalty = float(np.clip(hours_left / 24.0, 0.1, 1.0))
    long_penalty  = float(np.clip(1.0 - max(hours_left - 720, 0) / 8040, 0.5, 1.0))
    time_score    = short_penalty * long_penalty

    # Liquidez: < $5k = 0.2, $50k+ = 1.0
    liq_score = float(np.clip(np.log10(max(liquidity, 1)) / np.log10(50_000), 0.2, 1.0))

    # Qualidade do bookmaker: overround 1.02 (Pinnacle) → 1.0; 1.10+ → 0.1
    # Penaliza agressivamente vig alta pois contamina o cálculo de edge real
    book_score = float(np.clip(1.0 - (overround - 1.02) / 0.08, 0.1, 1.0))

    # IV (modo deribit): quanto mais alta a IV, maior o erro padrão da fair_prob B-S.
    # IV normal BTC/ETH: 50-70%. Acima de 70% penaliza linearmente até 0.30.
    # iv=None (modo odds): sem penalidade (iv_score=1.0).
    if iv is not None:
        iv_score = float(np.clip(1.0 - max(iv - 0.70, 0.0) / 0.60, 0.30, 1.0))
    else:
        iv_score = 1.0

    # Consenso multi-book (modo odds):
    # Mais books confirmando o mesmo preço → fair_prob mais robusta → maior confiança.
    # n_books=1: usamos um único book (menos confiável) → penaliza 30%
    # n_books=2: dois books → 15% de penalidade
    # n_books>=3: consenso pleno → sem penalidade
    # consensus_spread alto (livros divergem muito): penalidade adicional proporcional
    if n_books >= 3:
        n_score = 1.0
    elif n_books == 2:
        n_score = 0.85
    else:
        n_score = 0.70

    # Spread: cada 1pp de desvio padrão entre books reduz confiança em ~5%
    # Ex: spread=0.02 (2pp) → penalidade de 10% → n_score *= 0.90
    spread_penalty = float(np.clip(1.0 - consensus_spread * 5.0, 0.50, 1.0))
    consensus_score = n_score * spread_penalty

    return round(div_score * time_score * liq_score * book_score * iv_score * consensus_score, 3)


# ──────────────────────────────────────────────────────────
# MODO ODDS — sinais baseados em divergência vs. bookmakers
# ──────────────────────────────────────────────────────────

def load_latest_odds() -> pd.DataFrame:
    """
    Carrega o arquivo de odds matched mais recente de data/raw/odds/.
    Gerado por: uv run python pipeline/odds_collector.py
    """
    candidates = sorted(RAW_ODDS_DIR.glob("odds_matched_*.parquet"), reverse=True)
    if not candidates:
        raise FileNotFoundError(
            f"Nenhum arquivo de odds em {RAW_ODDS_DIR}. "
            "Execute: uv run python pipeline/odds_collector.py"
        )
    path = candidates[0]
    # total_seconds(), não .seconds — este último dá wrap a cada 24h no log
    age_minutes = int((datetime.now() - datetime.fromtimestamp(path.stat().st_mtime)).total_seconds() // 60)
    logger.info(f"Carregando odds: {path.name}  (gerado há {age_minutes}min)")
    return pd.read_parquet(path)


def generate_odds_signals(
    min_divergence: float = 0.08,
    min_liquidity: float = 5_000,
    min_volume24h: float = 500,
    min_days_left: int = 1,
    min_match_score: float = 0.25,
    top_n: int = 20,
    save: bool = True,
    fetch_fresh: bool = False,
) -> pd.DataFrame:
    """
    Gera sinais a partir de divergência Polymarket vs. bookmakers (modo odds).

    Args:
        min_divergence:  edge mínimo |polymarket_price - fair_prob|
        min_liquidity:   liquidez mínima do mercado em USDC
        min_volume24h:   volume 24h mínimo em USDC
        min_days_left:   dias mínimos até resolução
        min_match_score: score mínimo de matching para confiar no par
        top_n:           quantos sinais exibir
        save:            salvar CSV
        fetch_fresh:     se True, re-executa o odds_collector antes de gerar sinais

    Returns:
        DataFrame de sinais com edge real calculado contra bookmakers.
    """
    if fetch_fresh:
        logger.info("Coletando odds frescos...")
        sys.path.insert(0, str(Path(__file__).parent.parent / "pipeline"))
        from odds_collector import run as collect_odds
        collect_odds(min_divergence=0.0, save=True)

    odds_df = load_latest_odds()

    now = pd.Timestamp.now(tz="UTC")

    # Usa commence_time (hora do jogo) para o cálculo de dias restantes —
    # mais preciso que end_date para mercados esportivos intraday.
    # Outrights vêm com commence_time="" (torneio, não jogo) → fallback por linha
    # para end_date; sem isso, hours_to_start=-1 descartava TODOS os outrights.
    if "commence_time" in odds_df.columns:
        time_ref = pd.to_datetime(odds_df["commence_time"], errors="coerce", utc=True)
        if "end_date" in odds_df.columns:
            end_ref  = pd.to_datetime(odds_df["end_date"], errors="coerce", utc=True)
            time_ref = time_ref.fillna(end_ref)
    else:
        time_ref = pd.to_datetime(odds_df.get("end_date"), errors="coerce", utc=True)

    hours_to_start = ((time_ref - now).dt.total_seconds() / 3600).fillna(-1)

    odds_df = odds_df.copy()
    odds_df["days_to_end"]     = (hours_to_start / 24).clip(lower=-1)
    odds_df["hours_to_start"]  = hours_to_start.round(1)

    # Filtros de qualidade
    # min_days_left em horas para não cortar jogos de hoje à noite:
    # min_days_left=1 → pelo menos 24h → só jogos de amanhã em diante
    # min_days_left=0 → aceita jogos de hoje (recomendado para intraday)
    min_hours = min_days_left * 24
    mask = (
        (odds_df["abs_divergence"]  >= min_divergence) &
        (odds_df["liquidity"]       >= min_liquidity)  &
        (odds_df.get("volume_24h", pd.Series(9999, index=odds_df.index)) >= min_volume24h) &
        (odds_df["hours_to_start"]  >= min_hours)      &
        (odds_df["match_score"]     >= min_match_score)
    )

    signals = odds_df[mask].sort_values("abs_divergence", ascending=False).reset_index(drop=True)

    if signals.empty:
        return pd.DataFrame()

    # Renomeia para interface uniforme
    if "fair_prob_yes" in signals.columns:
        signals = signals.rename(columns={
            "fair_prob_yes": "prob_yes",
            "abs_divergence": "abs_edge",
        })
    # CONVENÇÃO ÚNICA (2026-07): edge = prob_yes − yes_price.
    # Positivo → mercado subprecifica YES → BUY_YES. O parquet do collector guarda
    # "divergence" com o sinal oposto (yes − fair, legado) — por isso recalculamos aqui.
    signals["edge"] = (signals["prob_yes"] - signals["yes_price"]).round(4)
    signals["signal_source"] = "odds"

    # Classifica trade_type:
    #   momentum — jogo em até 48h + mercado líquido (>= $30k): apostar na convergência
    #              de preço antes do evento (saída antecipada em 1.5×)
    #   value    — longo prazo ou outrights: aguardar resolução (saída em 2.5×)
    def _classify_trade(row) -> str:
        hours = float(row.get("hours_to_start", 9999))
        liq   = float(row.get("liquidity", 0))
        mtype = str(row.get("market_type", "h2h"))
        if mtype == "outright":
            return "value"
        if hours <= 48 and liq >= 30_000:
            return "momentum"
        return "value"

    signals["trade_type"] = signals.apply(_classify_trade, axis=1)

    # Score de confiança: escala posição por divergência, tempo e liquidez
    signals["confidence"] = signals.apply(
        lambda r: _confidence_score(
            abs_divergence=r.get("abs_edge", 0),
            hours_left=r.get("hours_to_start", 24),
            liquidity=r.get("liquidity", 0),
            overround=r.get("overround", 1.02),
            n_books=int(r.get("n_books", 1)),
            consensus_spread=float(r.get("consensus_spread", 0.0)),
        ), axis=1
    )

    _print_odds_signals_table(signals, top_n=top_n)

    if save and not signals.empty:
        path = save_signals(signals, tag="odds")
        console.print(f"  Relatório CSV: [dim]{path}[/dim]")

    return signals


def _print_odds_signals_table(signals: pd.DataFrame, top_n: int = 20) -> None:
    """Exibe tabela de sinais (modo odds) no terminal."""
    console.print("\n[bold cyan]══════════════════════════════════════════════════════════════════[/bold cyan]")
    console.print("[bold cyan]   SINAIS DE TRADING — ODDS DIVERGENCE (Poly vs. Bookmakers)    [/bold cyan]")
    console.print("[bold cyan]══════════════════════════════════════════════════════════════════[/bold cyan]\n")

    if signals.empty:
        console.print("[yellow]Nenhum sinal encontrado com os filtros atuais.[/yellow]")
        console.print("Sugestões:")
        console.print("  • Reduza --edge-threshold")
        console.print("  • Execute: uv run python pipeline/odds_collector.py  (atualiza odds)")
        console.print("  • Execute: uv run python pipeline/fetch_markets.py   (atualiza mercados)\n")
        return

    display = signals.head(top_n)
    table = Table(box=box.ROUNDED, show_lines=True)
    table.add_column("#",         width=3,  justify="right")
    table.add_column("Questão",   width=44)
    table.add_column("Dir.",      width=9,  justify="center")
    table.add_column("Poly",      width=7,  justify="right")
    table.add_column("Fair",      width=7,  justify="right")
    table.add_column("Edge",      width=8,  justify="right", style="bold")
    table.add_column("Book",      width=11)
    table.add_column("Dias",      width=5,  justify="right")
    table.add_column("Liquid.",   width=10, justify="right")

    for i, row in display.iterrows():
        edge = row.get("edge", 0)
        # edge = prob − preço: positivo → YES subprecificado → BUY_YES (verde)
        edge_color = "green" if edge > 0 else "red"
        dir_color  = "magenta" if row.get("direction") == "BUY_NO" else "green"
        table.add_row(
            str(i + 1),
            str(row["question"])[:43],
            f"[{dir_color}]{row.get('direction', '?')}[/{dir_color}]",
            f"{row['yes_price']:.3f}",
            f"{row.get('prob_yes', 0):.3f}",
            f"[{edge_color}]{edge:+.3f}[/{edge_color}]",
            str(row.get("bookmaker", ""))[:10],
            str(int(row.get("days_to_end", -1))),
            f"${row.get('liquidity', 0):,.0f}",
        )

    console.print(table)
    console.print(f"\n[bold]Total de sinais:[/bold] {len(signals):,}  |  Exibindo top {min(top_n, len(signals))}")
    console.print(f"[bold]Edge médio (abs):[/bold] {signals['abs_edge'].mean():.3f}")
    console.print(f"[bold]Fonte:[/bold] odds de bookmakers (edge real contra mercado externo)\n")


# ──────────────────────────────────────────────────────────
# MODO DERIBIT — Black-Scholes IV vs. Polymarket (crypto)
# ──────────────────────────────────────────────────────────

def generate_deribit_signals(
    min_divergence: float = 0.05,  # threshold menor: B-S risk-neutral subestima drift
    min_liquidity: float = 5_000,
    min_hours_left: float = 8.0,
    max_hours_left: float = 8_760.0,  # 1 ano — captura mercados EOY/mensais
    max_delta_days: int = 30,
    top_n: int = 20,
    save: bool = True,
    fetch_fresh: bool = True,   # sempre busca mercados frescos para deribit
) -> pd.DataFrame:
    """
    Gera sinais crypto via divergência Black-Scholes (Deribit IV) vs. Polymarket.

    O fair_prob é calculado com a medida risk-neutral (Q) do Black-Scholes.
    Isso significa que NÃO é a probabilidade real do mundo (P) — há um prêmio
    de risco de volatilidade embutido. Por isso usamos o score de confiança para
    escalar posições: divergências pequenas (< 5pp) podem ser apenas esse prêmio.

    Thresholds práticos baseados na incerteza do modelo:
      < 5pp  → ruído / prêmio de risco → não operar
      5-10pp → sinal fraco → posição pequena (confidence ~0.3-0.6)
      > 10pp → sinal forte → posição normal (confidence > 0.6)
      T < 4h → confiança reduzida automaticamente (IV intraday ruidosa)

    Args:
        min_divergence:  |poly - bs_prob| mínimo — recomendado >= 0.05
        min_liquidity:   liquidez mínima do mercado Polymarket em USDC
        min_hours_left:  horas mínimas até expiração
        max_delta_days:  janela para matching de expiração Deribit vs. Polymarket
        top_n:           quantos sinais exibir
        save:            salvar CSV
        fetch_fresh:     re-executa deribit_collector antes de gerar sinais

    Returns:
        DataFrame de sinais com fair_prob, divergence e confidence score.
    """
    sys.path.insert(0, str(Path(__file__).parent.parent / "pipeline"))
    from deribit_collector import run as collect_deribit

    if fetch_fresh:
        logger.info("Coletando dados frescos do Deribit...")
        df = collect_deribit(
            min_divergence=min_divergence,
            min_liquidity=min_liquidity,
            min_hours_left=min_hours_left,
            max_hours_left=max_hours_left,
            max_delta_days=max_delta_days,
            save=save,
        )
    else:
        # fetch_fresh=False usa o parquet mais recente do collector — antes este
        # flag era ignorado e re-coletava sempre (chamadas Deribit desperdiçadas)
        cached = sorted(RAW_ODDS_DIR.glob("deribit_signals_*.parquet"),
                        key=lambda p: p.stat().st_mtime)
        if not cached:
            logger.warning("Sem cache de sinais deribit — coletando frescos...")
            df = collect_deribit(
                min_divergence=min_divergence,
                min_liquidity=min_liquidity,
                min_hours_left=min_hours_left,
                max_hours_left=max_hours_left,
                max_delta_days=max_delta_days,
                save=save,
            )
        else:
            age_min = (datetime.now().timestamp() - cached[-1].stat().st_mtime) / 60
            logger.info(f"Usando cache: {cached[-1].name} ({age_min:.0f}min atrás)")
            if age_min > 60:
                logger.warning("Cache deribit com mais de 60min — considere --fetch-fresh")
            df = pd.read_parquet(cached[-1])
            if "abs_divergence" in df.columns:
                df = df[df["abs_divergence"] >= min_divergence].reset_index(drop=True)

    if df.empty:
        _print_deribit_signals_table(pd.DataFrame(), top_n=top_n)
        return pd.DataFrame()

    # Normaliza colunas para interface uniforme.
    # "direction" no deribit_collector = "above"/"below" (direção do mercado)
    # "signal" = "BUY_YES"/"BUY_NO" (ação de trading) → renomeia para "direction"
    df = df.rename(columns={
        "direction":      "market_direction",   # "above" ou "below"
        "signal":         "direction",          # "BUY_YES" ou "BUY_NO"
        "fair_prob":      "prob_yes",
        "abs_divergence": "abs_edge",
    })
    # CONVENÇÃO ÚNICA (2026-07): edge = prob_yes − yes_price (positivo → BUY_YES).
    # A coluna "divergence" do collector tem o sinal oposto (legado).
    df["edge"] = (df["prob_yes"] - df["yes_price"]).round(4)
    df["signal_source"] = "deribit"

    # Score de confiança: tempo curto (γ alto) + IV alta (B-S incerto) reduzem confiança
    df["confidence"] = df.apply(
        lambda r: _confidence_score(
            abs_divergence=r["abs_edge"],
            hours_left=r.get("hours_left", 24),
            liquidity=r.get("liquidity", 0),
            iv=r.get("iv"),  # penaliza IV alta: IV 100% → iv_score=0.50
        ), axis=1
    )

    _print_deribit_signals_table(df, top_n=top_n)

    if save and not df.empty:
        path = save_signals(df, tag="deribit")
        console.print(f"  Relatório CSV: [dim]{path}[/dim]")

    return df


def _print_deribit_signals_table(signals: pd.DataFrame, top_n: int = 20) -> None:
    """Exibe tabela de sinais Deribit com colunas de confiança."""
    console.print("\n[bold cyan]══════════════════════════════════════════════════════════════════[/bold cyan]")
    console.print("[bold cyan]   SINAIS CRYPTO — BLACK-SCHOLES (Deribit IV) vs. POLYMARKET      [/bold cyan]")
    console.print("[bold cyan]══════════════════════════════════════════════════════════════════[/bold cyan]\n")

    if signals.empty:
        console.print("[yellow]Nenhum sinal encontrado.[/yellow]")
        console.print("Sugestões:")
        console.print("  • Reduza --edge-threshold (mínimo recomendado: 0.05)")
        console.print("  • Execute: uv run python pipeline/fetch_markets.py  (atualiza mercados)")
        console.print("  • Execute: uv run python signals/run_signals.py --mode deribit --fetch-fresh\n")
        return

    console.print(
        "[dim]Nota: fair_prob = probabilidade risk-neutral (Black-Scholes). "
        "Divergências < 5pp podem refletir prêmio de risco, não mispricing real.[/dim]\n"
    )

    display = signals.head(top_n)
    table = Table(box=box.ROUNDED, show_lines=True)
    table.add_column("#",         width=3,  justify="right")
    table.add_column("Questão",   width=40)
    table.add_column("Dir.",      width=9,  justify="center")
    table.add_column("Poly",      width=7,  justify="right")
    table.add_column("B-S Fair",  width=8,  justify="right")
    table.add_column("Edge",      width=8,  justify="right", style="bold")
    table.add_column("IV",        width=7,  justify="right")
    table.add_column("Conf.",     width=6,  justify="right")
    table.add_column("Hrs",       width=5,  justify="right")

    for i, row in display.iterrows():
        edge      = float(row.get("edge", 0))
        conf      = float(row.get("confidence", 0))
        conf_color = "green" if conf >= 0.5 else ("yellow" if conf >= 0.25 else "red")
        # edge = prob − preço: positivo → BUY_YES (verde)
        edge_color = "green" if edge > 0 else "red"
        dir_color  = "magenta" if str(row.get("direction", "")) == "BUY_NO" else "green"

        table.add_row(
            str(i + 1),
            str(row["question"])[:39],
            f"[{dir_color}]{row.get('direction', '?')}[/{dir_color}]",
            f"{row['yes_price']:.3f}",
            f"{row.get('prob_yes', 0):.3f}",
            f"[{edge_color}]{edge:+.3f}[/{edge_color}]",
            row.get("iv_pct", "—"),
            f"[{conf_color}]{conf:.2f}[/{conf_color}]",
            str(int(row.get("hours_left", -1))),
        )

    console.print(table)
    avg_conf = signals["confidence"].mean()
    console.print(f"\n[bold]Total de sinais:[/bold] {len(signals):,}  |  Exibindo top {min(top_n, len(signals))}")
    console.print(f"[bold]Edge médio (abs):[/bold] {signals['abs_edge'].mean():.3f}")
    console.print(f"[bold]Confiança média:[/bold] {avg_conf:.2f}  "
                  f"({'alta' if avg_conf >= 0.5 else 'moderada' if avg_conf >= 0.25 else 'baixa'})")
    console.print(f"[bold]IV média:[/bold]         {signals['iv'].mean()*100:.1f}%\n")


# ──────────────────────────────────────────────────────────
# MODO ML — sinais baseados no modelo treinado
# ──────────────────────────────────────────────────────────

def generate_signals(
    model_path: Path | None = None,
    edge_threshold: float = 0.04,
    min_liquidity: float = 5_000,
    min_volume24h: float = 500,
    min_days_left: int = 1,
    top_n: int = 20,
    save: bool = True,
) -> pd.DataFrame:
    """
    ⚠️ CONGELADO (ADR-008, 2026-07) — removido do CLI. Além do data leakage
    conhecido, a inferência está incompatível com o bundle v2 (passa o dict como
    modelo, não aplica scaler, features divergem das do treino). NÃO usar em
    produção sem consertar; mantido apenas para eventual retomada do ML Lab.
    """
    logger.warning(
        "Modo ML ativo. O modelo atual tem data leakage — use generate_odds_signals() "
        "para sinais com edge real em mercados esportivos."
    )
    model, model_file = load_best_model(model_path)
    console.print(f"  Modelo: [cyan]{model_file.name}[/cyan]")

    df = load_active_markets()
    # Filtra apenas mercados ativos (não fechados nem arquivados)
    if "active" in df.columns:
        df = df[df["active"] == True].copy()
    if "closed" in df.columns:
        df = df[df["closed"] == False].copy()
    logger.info(f"  Mercados ativos após filtros: {len(df):,}")

    features = build_features(df.reset_index(drop=True))
    df = df.reset_index(drop=True)

    signals = compute_signals(
        df, features, model,
        edge_threshold=edge_threshold,
        min_liquidity=min_liquidity,
        min_volume24h=min_volume24h,
        min_days_left=min_days_left,
    )

    print_signals_table(signals, top_n=top_n)

    if save and not signals.empty:
        path = save_signals(signals)
        console.print(f"  Relatório CSV: [dim]{path}[/dim]")

    return signals
