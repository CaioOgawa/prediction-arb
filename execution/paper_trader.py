"""
paper_trader.py
Simulador de paper trading para Polymarket.

Simula execução de ordens com:
  - Capital virtual ($1.000 USDC por padrão)
  - Preços reais do mercado (bid/ask do Parquet mais recente)
  - Slippage realista: compras no ask, vendas no bid
  - Kelly fracionado × confidence score (via risk/risk_manager.py)
  - Persistência de portfólio em SQLite
  - Resolução automática de posições quando mercado fecha (P&L real)

Fontes de sinal suportadas:
  --mode odds     → sinais de divergência Polymarket vs. Pinnacle (esportes)
  --mode deribit  → sinais de divergência Polymarket vs. Black-Scholes IV (crypto)
  --mode all      → combina odds + deribit, rankeados por edge × confidence

Arb estrutural (independente do --mode): baskets GARANTIDOS do scanner
(signals_structural_*.csv) são executados ATOMICAMENTE via open_basket() —
todas as pernas na mesma transação, trade_type='arb', hold até resolução.
"""

import contextlib
import sqlite3
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from rich.console import Console
from rich.table import Table
from rich import box

DB_PATH      = Path("data/db/paper_trading.db")
REPORTS_DIR  = Path("outputs/reports")
RAW_MKT_DIR  = Path("data/raw/markets")
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

console = Console()

SCHEMA = """
CREATE TABLE IF NOT EXISTS portfolio (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at      TEXT    NOT NULL DEFAULT (datetime('now')),
    initial_capital REAL    NOT NULL,
    current_cash    REAL    NOT NULL,
    note            TEXT
);

CREATE TABLE IF NOT EXISTS positions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    opened_at       TEXT    NOT NULL DEFAULT (datetime('now')),
    closed_at       TEXT,
    condition_id    TEXT    NOT NULL,
    question        TEXT,
    category        TEXT,
    signal_source   TEXT,
    direction       TEXT    NOT NULL,
    entry_price     REAL    NOT NULL,
    exit_price      REAL,
    shares          REAL    NOT NULL,
    cost_usdc       REAL    NOT NULL,
    pnl_usdc        REAL    DEFAULT 0,
    status          TEXT    NOT NULL DEFAULT 'open',
    edge_at_entry   REAL,
    prob_at_entry   REAL,
    confidence      REAL,
    end_date        TEXT,
    event_slug      TEXT
);

CREATE TABLE IF NOT EXISTS trades_log (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       TEXT    NOT NULL DEFAULT (datetime('now')),
    action          TEXT    NOT NULL,
    condition_id    TEXT    NOT NULL,
    direction       TEXT    NOT NULL,
    price           REAL    NOT NULL,
    shares          REAL    NOT NULL,
    usdc_amount     REAL    NOT NULL,
    note            TEXT
);
"""


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    # P1-18: default do sqlite3 é 5s. Cinco processos disputam o mesmo
    # arquivo (run_cycle, run_execution, ws_feed.exit_writer,
    # ws_feed.history_writer commitando a 10Hz, dashboard) — 5s estoura fácil
    # e sobe OperationalError sem nenhum try/except no meio do ciclo.
    conn.execute("PRAGMA busy_timeout = 30000")
    return conn


# P1-18: WHERE id = ? com portfolio["id"] em cache já causou débito no
# portfólio errado depois de reset (9 resets até agora) — o crédito
# correspondente sempre vai pro portfólio ATUAL via get_or_create_portfolio().
# Todo UPDATE portfolio usa este subquery em vez do id em Python.
_CURRENT_PORTFOLIO_ID_SQL = "(SELECT MAX(id) FROM portfolio)"


def init_db() -> None:
    with contextlib.closing(get_connection()) as conn:
        conn.executescript(SCHEMA)
        # Migrations: adiciona colunas ausentes sem recriar a tabela
        existing = {r[1] for r in conn.execute("PRAGMA table_info(positions)").fetchall()}
        for col_name, typedef in [
            ("signal_source", "TEXT"),
            ("confidence", "REAL"),
            ("trade_type", "TEXT DEFAULT 'value'"),
            ("event_slug", "TEXT DEFAULT ''"),
            ("arb_group", "TEXT DEFAULT ''"),
            ("underlying", "TEXT DEFAULT ''"),  # P1-11: ativo real (BTC/ETH, sport_key:matchup)
            ("needs_manual_resolution", "INTEGER DEFAULT 0"),  # P1-16: fechou sem outcome parseável
        ]:
            if col_name not in existing:
                # Bug pré-existente: o ALTER TABLE só usava {col}, descartando
                # {typedef} — colunas migradas ficavam sem tipo/default (NULL
                # em vez de '' ou 'value'), provavelmente a origem real do
                # P1-17 (trade_type NULL/NaN em posições antigas).
                conn.execute(f"ALTER TABLE positions ADD COLUMN {col_name} {typedef}")
                logger.info(f"Migração: coluna '{col_name}' adicionada à tabela positions")
        conn.commit()


def get_or_create_portfolio(initial_capital: float = 1_000.0) -> dict:
    with contextlib.closing(get_connection()) as conn:
        row = conn.execute(
            "SELECT * FROM portfolio ORDER BY id DESC LIMIT 1"
        ).fetchone()
        if row:
            return dict(row)
        conn.execute(
            "INSERT INTO portfolio (initial_capital, current_cash) VALUES (?, ?)",
            (initial_capital, initial_capital),
        )
        conn.commit()
        return {
            "id": conn.execute("SELECT last_insert_rowid()").fetchone()[0],
            "initial_capital": initial_capital,
            "current_cash":    initial_capital,
        }


def get_open_positions() -> pd.DataFrame:
    with contextlib.closing(get_connection()) as conn:
        rows = conn.execute(
            "SELECT * FROM positions WHERE status = 'open'"
        ).fetchall()
    df = pd.DataFrame([dict(r) for r in rows]) if rows else pd.DataFrame()
    if not df.empty and "trade_type" in df.columns:
        # P1-17: normaliza na leitura — cobre tanto NaN/None quanto a string
        # literal 'nan' que ficou gravada em posições antigas (fillna() não
        # pega, porque 'nan' já é uma string válida do ponto de vista do
        # pandas). Sem isso, os limites por trade_type (MAX_MOMENTUM_POS/
        # MAX_VALUE_POS) nunca contam essas posições — o filtro não casa NaN.
        from risk.risk_manager import normalize_trade_type
        df["trade_type"] = df["trade_type"].apply(lambda x: normalize_trade_type(x, warn=False))
    return df


def get_traded_condition_ids() -> set[str]:
    """
    Retorna condition_ids de todos os mercados já operados no portfólio atual
    (abertas + fechadas). Evita reabrir posições em mercados que já resolveram
    ou que foram encerrados por edge flip/profit target.
    """
    with contextlib.closing(get_connection()) as conn:
        portfolio_start = conn.execute(
            "SELECT created_at FROM portfolio ORDER BY id DESC LIMIT 1"
        ).fetchone()
        if not portfolio_start:
            return set()
        rows = conn.execute(
            "SELECT DISTINCT condition_id FROM positions WHERE opened_at >= ?",
            (portfolio_start[0],),
        ).fetchall()
    return {r[0] for r in rows} if rows else set()


MAX_MARKET_SNAPSHOT_AGE_MIN = 90


def load_current_markets() -> pd.DataFrame:
    """Carrega snapshot mais recente de mercados para mark-to-market e resolução."""
    candidates = sorted(
        list(RAW_MKT_DIR.glob("markets_all_*.parquet")) +
        list(RAW_MKT_DIR.glob("markets_incremental_*.parquet")),
        key=lambda p: p.stat().st_mtime,  # ordena por tempo de modificação, não por nome
        reverse=True,
    )
    if not candidates:
        return pd.DataFrame()

    newest = candidates[0]
    age_min = int((datetime.now().timestamp() - newest.stat().st_mtime) // 60)
    if age_min > MAX_MARKET_SNAPSHOT_AGE_MIN:
        logger.warning(
            f"Snapshot de mercados STALE ({newest.name}, {age_min}min > "
            f"{MAX_MARKET_SNAPSHOT_AGE_MIN}min) — ignorado. Execute o pipeline "
            "para gerar mercados frescos."
        )
        return pd.DataFrame()

    return pd.read_parquet(newest)


def load_signals(mode: str = "odds", min_edge: float | None = None) -> pd.DataFrame:
    """
    Carrega os sinais mais recentes do modo especificado.

    mode:
      'odds'    → signals_odds_*.csv    (Pinnacle divergence)
      'deribit' → signals_deribit_*.csv (Black-Scholes IV)
      'all'     → combina ambos, rankeados por abs_edge × confidence
    """
    MAX_SIGNAL_AGE_MINUTES = 60

    if min_edge is None:
        from risk.risk_manager import MIN_EDGE_ABS
        min_edge = MIN_EDGE_ABS  # pré-filtro coarse; o gate por fonte é do kelly_size

    def _load_pattern(pattern: str) -> pd.DataFrame:
        files = sorted(REPORTS_DIR.glob(pattern), reverse=True)
        if not files:
            return pd.DataFrame()
        age_sec = (datetime.now() - datetime.fromtimestamp(files[0].stat().st_mtime)).total_seconds()
        age_min = int(age_sec // 60)
        if age_min > MAX_SIGNAL_AGE_MINUTES:
            logger.warning(
                f"  {pattern}: sinal STALE ({age_min}min > {MAX_SIGNAL_AGE_MINUTES}min) — ignorado. "
                "Execute o pipeline para gerar sinais frescos."
            )
            return pd.DataFrame()
        df = pd.read_csv(files[0])
        logger.info(f"  {pattern}: {len(df)} sinais ({files[0].name}, {age_min}min atrás)")
        return df

    if mode == "odds":
        df = _load_pattern("signals_odds_*.csv")
    elif mode == "deribit":
        df = _load_pattern("signals_deribit_*.csv")
    elif mode == "all":
        odds_df    = _load_pattern("signals_odds_*.csv")
        deribit_df = _load_pattern("signals_deribit_*.csv")
        frames = [f for f in [odds_df, deribit_df] if not f.empty]
        if not frames:
            return pd.DataFrame()
        df = pd.concat(frames, ignore_index=True)
    else:
        raise ValueError(f"Modo desconhecido: {mode}. Use 'odds', 'deribit' ou 'all'.")

    if df.empty:
        return df

    # Garante colunas essenciais
    if "abs_edge" not in df.columns and "abs_divergence" in df.columns:
        df = df.rename(columns={"abs_divergence": "abs_edge"})
    if "confidence" not in df.columns:
        df["confidence"] = 0.5  # fallback se arquivo antigo sem confidence

    # Score composto para ranking: edge × confidence
    df["rank_score"] = df.get("abs_edge", 0) * df.get("confidence", 0.5)
    df = df[df.get("abs_edge", pd.Series(0, index=df.index)) >= min_edge]
    df = df.sort_values("rank_score", ascending=False).reset_index(drop=True)

    return df


def open_position(
    portfolio: dict,
    signal: dict,
    dry_run: bool = False,
    current_markets: pd.DataFrame | None = None,
    open_positions: pd.DataFrame | None = None,
    traded_ids: set[str] | None = None,
) -> dict | None:
    """
    Tenta abrir uma posição com base no sinal.
    Usa o risk_manager para sizing e verificação de limites.
    Retorna o dict da posição ou None se rejeitada.

    P1-19: current_markets/open_positions/traded_ids são opcionais — quando o
    chamador itera vários sinais no mesmo ciclo (run_paper_trading,
    run_execution), passar os três evita reler o parquet e reconsultar o
    banco a cada candidato. None (default) mantém o comportamento antigo,
    buscando tudo internamente — é o que os testes existentes fazem.
    """
    from risk.risk_manager import kelly_size, check_exposure, MIN_EDGE_ABS, normalize_trade_type
    from pipeline.market_pricing import entry_price_and_net_edge

    direction    = str(signal.get("direction", ""))
    yes_price    = float(signal.get("yes_price", 0.5))
    edge         = float(signal.get("abs_edge", signal.get("edge", 0)))
    prob_yes     = float(signal.get("prob_yes", 0.5))
    confidence   = float(signal.get("confidence", 0.5))
    signal_source= str(signal.get("signal_source", "odds"))
    trade_type   = normalize_trade_type(signal.get("trade_type"), context="open_position")
    spread       = max(float(signal.get("spread") or 0), 0.005)
    cid          = str(signal.get("condition_id", ""))

    if direction not in ("BUY_YES", "BUY_NO"):
        return None

    # P1-25: preço executável real (bestAsk/1-bestBid) calculado pelo gerador
    # de sinal — não a reconstrução yes_price(lastTradePrice) + spread/2.
    # Pode ficar stale até 60min (idade do sinal); re-cotação abaixo tenta
    # uma versão mais fresca antes de cair pro valor do sinal.
    real_entry_price = signal.get("entry_price")
    real_entry_price = float(real_entry_price) if pd.notna(real_entry_price) else None

    # ── Re-cotação: verifica preço atual antes de abrir ──────────────────
    # O sinal pode ter sido gerado há até 60min; re-cotamos para garantir
    # que o edge ainda existe com o preço corrente do mercado.
    MIN_EDGE = MIN_EDGE_ABS  # do risk_manager — não duplicar constantes
    current_mkts = current_markets if current_markets is not None else load_current_markets()
    if cid and not current_mkts.empty:
        id_col = "conditionId" if "conditionId" in current_mkts.columns else "condition_id"
        match = current_mkts[current_mkts[id_col] == cid]
        if not match.empty:
            row = match.iloc[0]
            fresh_yes   = float(row.get("yes_price") or yes_price)
            fresh_spread = max(float(row.get("spread") or 0), 0.005)
            # Reconstrói fair_prob do sinal (não muda — é previsão do modelo)
            fair_prob = (yes_price + edge) if direction == "BUY_YES" else (yes_price - edge)
            # Edge com preço atual
            refreshed_edge = (fair_prob - fresh_yes) if direction == "BUY_YES" else (fresh_yes - fair_prob)
            if refreshed_edge < MIN_EDGE:
                logger.info(
                    f"Re-cotação: edge desapareceu {edge:.1%} → {refreshed_edge:.1%} "
                    f"(yes {yes_price:.3f}→{fresh_yes:.3f}) — pulando {cid[:12]}"
                )
                return None
            # Atualiza com dados frescos
            if abs(fresh_yes - yes_price) > 0.005:
                logger.debug(
                    f"Re-cotação aplicada: yes {yes_price:.3f}→{fresh_yes:.3f}, "
                    f"edge {edge:.1%}→{refreshed_edge:.1%}"
                )
            yes_price = fresh_yes
            spread    = fresh_spread
            edge      = refreshed_edge

            # Preço executável fresco do snapshot atual, se o book der pra usar
            # (mesmo cálculo bestAsk/1-bestBid do gerador de sinal) — mais
            # recente que o entry_price gravado no sinal, então tem prioridade.
            # fair_prob acima já é fair_yes nas duas direções (yes_price±edge
            # reconstrói o mesmo fair_yes, só o sinal do edge muda com a direção).
            fresh_priced = entry_price_and_net_edge(
                row, direction, fair_prob, 1.0 - fair_prob, source=signal_source,
            )
            if fresh_priced is not None:
                real_entry_price = fresh_priced[0]

    # Preço de entrada base (sem market impact por enquanto — recalculado após Kelly)
    liquidity = float(signal.get("liquidity") or 0)
    if real_entry_price is not None:
        entry_price = min(real_entry_price, 0.98)
    elif direction == "BUY_YES":
        entry_price = min(yes_price + spread / 2, 0.98)
    else:
        entry_price = min((1.0 - yes_price) + spread / 2, 0.98)

    cash = float(portfolio["current_cash"])

    # P1-19: uma leitura só de open_positions pro candidato inteiro — antes
    # eram duas (aqui e no check_exposure mais abaixo), cada uma um SELECT
    # completo na tabela positions.
    open_pos_snapshot = open_positions if open_positions is not None else get_open_positions()

    # Capital efetivo: caixa + 50% do valor MTM das posições abertas.
    # Posições abertas têm liquidez limitada mas não são zero — usar 50% como proxy.
    # Isso evita que o Kelly encolha progressivamente conforme alocamos capital.
    if not open_pos_snapshot.empty and not current_mkts.empty:
        mtm_tmp = mark_to_market(open_pos_snapshot, current_mkts)
        open_value = float(mtm_tmp["current_value"].sum()) if "current_value" in mtm_tmp.columns else 0.0
    else:
        open_value = 0.0
    effective_capital = cash + open_value * 0.5

    # P1-11d: quantas posições já abertas compartilham o mesmo underlying
    # (BTC, ETH, sport_key:matchup) — Kelly independente sobre apostas
    # correlacionadas superaposta risco por ~√n.
    underlying = str(signal.get("underlying", "")).lower()
    n_correlated = 1
    if underlying and not open_pos_snapshot.empty and "underlying" in open_pos_snapshot.columns:
        n_correlated = 1 + int(
            (open_pos_snapshot["underlying"].fillna("").str.lower() == underlying).sum()
        )

    # Kelly com confidence modulation — sizing varia por trade_type
    size_usdc = kelly_size(
        edge=edge,
        entry_price=entry_price,
        capital=effective_capital,
        confidence=confidence,
        signal_source=signal_source,
        trade_type=trade_type,
        n_correlated=n_correlated,
    )

    # Recalcula entry_price com market impact agora que size_usdc é conhecido.
    # Com real_entry_price, o meio-spread já está embutido no preço — somar
    # spread/2 de novo seria o mesmo bug de desconto duplo já corrigido no
    # ws_feed/early_exit_positions (P1-14); só o impacto de mercado do nosso
    # próprio tamanho de ordem entra aqui.
    if liquidity > 0:
        impact = size_usdc / (liquidity * 0.10)
        effective_spread = spread * 2.0 if size_usdc > liquidity * 0.05 else spread
        slippage = min(impact, spread) if real_entry_price is not None else (effective_spread / 2 + min(impact, spread))
    else:
        slippage = 0.0 if real_entry_price is not None else spread / 2
    if real_entry_price is not None:
        entry_price = min(real_entry_price + slippage, 0.98)
    elif direction == "BUY_YES":
        entry_price = min(yes_price + slippage, 0.98)
    else:
        entry_price = min((1.0 - yes_price) + slippage, 0.98)

    _q = str(signal.get("question", ""))[:40]

    if size_usdc < 2.0:
        motivo = f"correlacao n={n_correlated}" if n_correlated > 1 else "kelly_pequeno"
        logger.info(f"REJEITADO [{motivo} ${size_usdc:.2f}] {_q}")
        return None

    # Não reabrir mercados já operados neste portfólio (evita loop em mercados resolvidos)
    traded = traded_ids if traded_ids is not None else get_traded_condition_ids()
    if cid and cid in traded:
        logger.debug(f"REJEITADO [já_operado] {cid[:16]}")
        return None

    # Verificações de diversificação — P1-11b: denominador é cash + MTM
    # (total_value), não initial_capital, senão os caps folgam depois de um
    # drawdown em vez de apertar.
    total_value = cash + open_value
    # P1-16: check_exposure exclui posição travada (needs_manual_resolution=1)
    # só do contador de slot direcional — passa a lista completa aqui de
    # propósito, pra manter a checagem de duplicata e os caps de
    # categoria/underlying vendo o capital que continua comprometido nela.
    can_trade, reason = check_exposure(signal, open_pos_snapshot, portfolio, size_usdc, total_value=total_value)
    if not can_trade:
        logger.info(f"REJEITADO [risco: {reason}] {_q}")
        return None

    shares = size_usdc / entry_price

    position = {
        "condition_id":   signal.get("condition_id", ""),
        "question":       str(signal.get("question", ""))[:100],
        "category":       str(signal.get("category", "")),
        "underlying":     str(signal.get("underlying", "")),
        "signal_source":  signal_source,
        "trade_type":     trade_type,
        "direction":      direction,
        "entry_price":    round(entry_price, 4),
        "shares":         round(shares, 4),
        "cost_usdc":      round(size_usdc, 2),
        "edge_at_entry":  round(edge, 4),
        "prob_at_entry":  round(prob_yes, 4),
        "confidence":     round(confidence, 3),
        "end_date":       str(signal.get("end_date", signal.get("commence_time", ""))),
        "event_slug":     str(signal.get("event_slug", "")),
    }

    if dry_run:
        return position

    # P1-18: única gravação de dinheiro sem BEGIN IMMEDIATE/retry até aqui —
    # com o history_writer do ws_feed commitando a 10Hz, um open_position
    # azarado estourava o busy_timeout e a OperationalError subia sem
    # try/except, matando o subprocesso no meio do ciclo. ALTER TABLE saiu
    # daqui — já é feito uma vez por init_db(), redundante dentro da
    # transação de dinheiro. Ordem cash-primeiro-com-guarda, posição depois,
    # igual open_basket/rebalance_positions.
    MAX_DB_RETRIES = 3
    for attempt in range(1, MAX_DB_RETRIES + 1):
        try:
            with contextlib.closing(get_connection()) as conn:
                conn.execute("BEGIN IMMEDIATE")
                cur = conn.execute(
                    "UPDATE portfolio SET current_cash = current_cash - ? "
                    f"WHERE id = {_CURRENT_PORTFOLIO_ID_SQL} AND current_cash >= ?",
                    (size_usdc, size_usdc),
                )
                if cur.rowcount == 0:
                    conn.rollback()
                    logger.warning(
                        f"Race condition: caixa insuficiente para ${size_usdc:.2f} "
                        f"— posição cancelada (outro processo abriu posição simultaneamente?)"
                    )
                    return None
                conn.execute("""
                    INSERT INTO positions
                      (condition_id, question, category, underlying, signal_source, trade_type, direction,
                       entry_price, shares, cost_usdc, edge_at_entry, prob_at_entry,
                       confidence, end_date, event_slug)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    position["condition_id"], position["question"], position["category"],
                    position["underlying"], position["signal_source"], position["trade_type"],
                    position["direction"], position["entry_price"], position["shares"],
                    position["cost_usdc"], position["edge_at_entry"], position["prob_at_entry"],
                    position["confidence"], position["end_date"], position["event_slug"],
                ))
                conn.execute("""
                    INSERT INTO trades_log (action, condition_id, direction, price, shares, usdc_amount, note)
                    VALUES ('OPEN', ?, ?, ?, ?, ?, ?)
                """, (
                    position["condition_id"], direction, position["entry_price"],
                    position["shares"], position["cost_usdc"],
                    f"source={signal_source} edge={edge:.3f} conf={confidence:.2f}",
                ))
                conn.commit()
            return position
        except sqlite3.OperationalError as e:
            if "locked" in str(e).lower() and attempt < MAX_DB_RETRIES:
                logger.warning(f"DB travado ao abrir posição, tentativa {attempt}/{MAX_DB_RETRIES}: {e}")
                time.sleep(0.5 * attempt)
                continue
            logger.exception(
                f"open_position falhou após {attempt} tentativa(s) — sinal descartado, "
                "ciclo continua"
            )
            return None


# ──────────────────────────────────────────────────────────
# Arb estrutural — execução atômica de baskets (Fase 2)
# ──────────────────────────────────────────────────────────

# Sinais estruturais mais velhos que isso não são executados. O scanner roda
# imediatamente antes do paper trader no run_cycle (~1-2min de idade típica);
# arb depende de preço de book — mais estrito que os 60min dos sinais de valor.
MAX_ARB_SIGNAL_AGE_MINUTES = 30


def load_structural_signals() -> pd.DataFrame:
    """
    Carrega o CSV mais recente do scanner estrutural (uma linha por perna,
    agrupadas por arb_group). Só retorna oportunidades GARANTIDAS — as
    condicionais (YES-basket sem guarda-chuva, book cruzado) exigem
    verificação humana e continuam report-only.
    """
    files = sorted(REPORTS_DIR.glob("signals_structural_*.csv"), reverse=True)
    if not files:
        return pd.DataFrame()
    age_min = (datetime.now() - datetime.fromtimestamp(files[0].stat().st_mtime)).total_seconds() / 60
    if age_min > MAX_ARB_SIGNAL_AGE_MINUTES:
        logger.debug(f"Sinais estruturais stale ({age_min:.0f}min) — ignorados")
        return pd.DataFrame()
    df = pd.read_csv(files[0])
    if df.empty or "guaranteed" not in df.columns:
        return pd.DataFrame()
    df = df[df["guaranteed"] == True]  # noqa: E712
    logger.info(f"  signals_structural: {df['arb_group'].nunique() if not df.empty else 0} "
                f"baskets garantidos ({files[0].name}, {age_min:.0f}min atrás)")
    return df


def open_basket(
    portfolio: dict,
    group: pd.DataFrame,
    current_markets: pd.DataFrame,
    dry_run: bool = False,
) -> dict | None:
    """
    Abre TODAS as pernas de um basket de arb estrutural ATOMICAMENTE.

    Ou todas as pernas entram na mesma transação SQLite (com o débito único de
    cash), ou nenhuma entra — perna rejeitada isolada viraria posição
    direcional nua, exatamente o que a garantia estrutural elimina.

    Regras (roadmap Fase 2):
      - Só baskets guaranteed=True (a exaustividade/exclusão mútua é lógica)
      - Re-verificação de preço no parquet atual: se o arb evaporou, pula
      - Sizing por basket (ARB_MAX_BASKET_PCT do capital inicial) limitado
        pela liquidez da perna mais fina — Kelly não se aplica (edge garantido
        daria fração infinita)
      - Mesmo nº de shares em todas as pernas (a matemática do arb exige)
      - trade_type='arb' → EARLY_EXIT nunca dispara (hold até resolução)

    Retorna dict-resumo do basket ou None se rejeitado.
    """
    from risk.risk_manager import (
        ARB_MIN_PROFIT, ARB_MAX_BASKET_PCT, MAX_ARB_BASKETS,
        ARB_LIQUIDITY_FRAC, MAX_SOURCE_PCT,
    )

    first      = group.iloc[0]
    arb_group  = str(first["arb_group"])
    arb_kind   = str(first["arb_kind"])
    payout_min = float(first["payout_min"])
    n_legs     = int(first["n_legs"])

    # CSV truncado/inconsistente → basket incompleto não pode ser executado
    if len(group) != n_legs or n_legs < 2:
        logger.warning(f"BASKET rejeitado [pernas {len(group)}/{n_legs} no CSV] {arb_group}")
        return None

    legs = []
    for _, r in group.iterrows():
        direction = str(r["direction"])
        price     = float(r["leg_price"])
        if direction not in ("BUY_YES", "BUY_NO") or not (0.0 < price < 1.0):
            logger.warning(f"BASKET rejeitado [perna inválida {direction}@{price}] {arb_group}")
            return None
        legs.append({
            "condition_id": str(r["condition_id"]),
            "question":     str(r.get("question", ""))[:100],
            "direction":    direction,
            "exec_price":   price,
            "liquidity":    float(r.get("liquidity") or 0),
            "event_slug":   str(r.get("event_slug") or ""),
            "underlying":   str(r.get("underlying") or ""),
        })

    # Dedupe: qualquer perna já operada neste portfólio cancela o basket inteiro
    traded = get_traded_condition_ids()
    if any(l["condition_id"] in traded for l in legs):
        logger.debug(f"BASKET pulado [perna já operada] {arb_group}")
        return None

    # Limite de baskets simultâneos (as pernas não contam no MAX_OPEN_POSITIONS)
    open_pos = get_open_positions()
    if not open_pos.empty and "trade_type" in open_pos.columns:
        arb_open = open_pos[open_pos["trade_type"] == "arb"]
        n_baskets = arb_open["arb_group"].nunique() if "arb_group" in arb_open.columns else len(arb_open)
        if n_baskets >= MAX_ARB_BASKETS:
            logger.info(f"BASKET pulado [limite de {MAX_ARB_BASKETS} baskets] {arb_group}")
            return None

    # ── Re-verificação: o arb ainda existe com os preços do parquet atual? ──
    # Pernas fora do parquet (filtro de volume) mantêm o preço do CSV — que é
    # fresco por construção (gate MAX_ARB_SIGNAL_AGE_MINUTES).
    if not current_markets.empty and "conditionId" in current_markets.columns:
        mkt = current_markets.drop_duplicates("conditionId", keep="last").set_index("conditionId")
        for leg in legs:
            if leg["condition_id"] not in mkt.index:
                continue
            row = mkt.loc[leg["condition_id"]]
            try:
                bid, ask = float(row.get("bestBid")), float(row.get("bestAsk"))
            except (TypeError, ValueError):
                continue
            if not (0.0 < bid < 1.0 and 0.0 < ask < 1.0):
                continue
            leg["exec_price"] = ask if leg["direction"] == "BUY_YES" else round(1.0 - bid, 4)

    basket_cost = sum(l["exec_price"] for l in legs)
    profit      = payout_min - basket_cost
    if profit < ARB_MIN_PROFIT * payout_min:
        logger.info(
            f"BASKET pulado [arb evaporou na re-cotação: lucro ${profit:.3f} "
            f"< {ARB_MIN_PROFIT:.0%} × payout {payout_min:.0f}] {arb_group}"
        )
        return None

    # ── Sizing por basket (não é Kelly — lucro garantido daria fração ∞) ──
    initial = float(portfolio.get("initial_capital", 1000))
    cash    = float(portfolio["current_cash"])

    cap_usdc = initial * ARB_MAX_BASKET_PCT
    # Respeita o teto da fonte structural (mesmo MAX_SOURCE_PCT das demais)
    if not open_pos.empty and "signal_source" in open_pos.columns:
        src_cost = float(open_pos[open_pos["signal_source"] == "structural"]["cost_usdc"].sum())
        cap_usdc = min(cap_usdc, initial * MAX_SOURCE_PCT - src_cost)
    cap_usdc = min(cap_usdc, cash - 5.0)
    if cap_usdc < 2.0:
        logger.info(f"BASKET pulado [sem capital: cap ${cap_usdc:.2f}] {arb_group}")
        return None

    shares = cap_usdc / basket_cost
    # Liquidez da perna mais fina limita o basket inteiro (shares são iguais)
    for leg in legs:
        if leg["liquidity"] > 0:
            shares = min(shares, (leg["liquidity"] * ARB_LIQUIDITY_FRAC) / leg["exec_price"])
    shares = round(shares, 4)
    # Débito = soma exata dos custos arredondados por perna — evita drift de
    # centavos entre portfolio.current_cash e Σ positions.cost_usdc
    leg_costs  = [round(shares * l["exec_price"], 2) for l in legs]
    total_cost = round(sum(leg_costs), 2)
    if total_cost < 2.0:
        logger.info(f"BASKET pulado [tamanho mínimo: ${total_cost:.2f}] {arb_group}")
        return None

    summary = {
        "arb_group":         arb_group,
        "arb_kind":          arb_kind,
        "n_legs":            n_legs,
        "shares":            shares,
        "total_cost":        total_cost,
        "payout_total":      round(shares * payout_min, 2),
        "guaranteed_profit": round(shares * profit, 2),
        "edge":              round(profit / payout_min, 4),
    }
    if dry_run:
        return summary

    # ── Abertura ATÔMICA: débito único + todas as pernas na mesma transação ──
    with contextlib.closing(get_connection()) as conn:
        conn.execute("BEGIN IMMEDIATE")
        try:
            cur = conn.execute(
                "UPDATE portfolio SET current_cash = current_cash - ? "
                "WHERE id = (SELECT MAX(id) FROM portfolio) AND current_cash >= ?",
                (total_cost, total_cost),
            )
            if cur.rowcount == 0:
                conn.rollback()
                logger.warning(f"BASKET cancelado [caixa insuficiente p/ ${total_cost:.2f}] {arb_group}")
                return None
            for leg, leg_cost in zip(legs, leg_costs):
                # P1-11a: category NÃO recebe arb_kind mais — eram dois
                # conceitos sobrepostos na mesma coluna (categoria de mercado
                # da Gamma API vs. tipo de arb). arb_kind já vive em
                # arb_group (prefixo "mono:"/"neg:"); underlying carrega o
                # ativo real (BTC/ETH) para o cap de underlying pegar
                # posições correlacionadas mesmo vindas de baskets.
                conn.execute("""
                    INSERT INTO positions
                      (condition_id, question, category, underlying, signal_source, trade_type,
                       direction, entry_price, shares, cost_usdc, edge_at_entry,
                       prob_at_entry, confidence, end_date, event_slug, arb_group)
                    VALUES (?, ?, '', ?, 'structural', 'arb', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    leg["condition_id"], leg["question"], leg["underlying"],
                    leg["direction"], round(leg["exec_price"], 4), shares, leg_cost,
                    summary["edge"], round(leg["exec_price"], 4),
                    float(first.get("confidence", 0.9)), "", leg["event_slug"], arb_group,
                ))
                conn.execute("""
                    INSERT INTO trades_log (action, condition_id, direction, price, shares, usdc_amount, note)
                    VALUES ('OPEN_BASKET', ?, ?, ?, ?, ?, ?)
                """, (
                    leg["condition_id"], leg["direction"], round(leg["exec_price"], 4),
                    shares, leg_cost,
                    f"arb_group={arb_group} lucro_garantido=${summary['guaranteed_profit']:.2f}",
                ))
            conn.commit()
        except Exception:
            conn.rollback()
            logger.exception(f"BASKET falhou no meio da transação — rollback completo {arb_group}")
            return None

    logger.info(
        f"BASKET ABERTO {arb_group}: {n_legs} pernas × {shares:.2f} sh, "
        f"custo ${total_cost:.2f} → payout ${summary['payout_total']:.2f} "
        f"(lucro garantido ${summary['guaranteed_profit']:.2f})"
    )
    return summary


def open_arb_baskets(
    portfolio: dict,
    current_markets: pd.DataFrame,
    dry_run: bool = False,
) -> list[dict]:
    """
    Executa todos os baskets garantidos do CSV estrutural mais recente,
    do maior edge para o menor. Cada basket é atômico (ver open_basket).
    """
    df = load_structural_signals()
    if df.empty:
        return []

    opened = []
    groups = (
        df.groupby("arb_group")["edge"].first()
        .sort_values(ascending=False)
        .index.tolist()
    )
    for arb_group in groups:
        result = open_basket(portfolio, df[df["arb_group"] == arb_group],
                             current_markets, dry_run=dry_run)
        if result:
            opened.append(result)
            if not dry_run:
                # Refresh do cash — o débito do basket anterior já aconteceu
                portfolio = get_or_create_portfolio(float(portfolio.get("initial_capital", 1000)))
    return opened


MIN_LIQUIDITY_MTM = 500.0  # USDC — abaixo disso, preço é considerado stale/ilíquido

def mark_to_market(open_positions: pd.DataFrame, current_markets: pd.DataFrame) -> pd.DataFrame:
    """Atualiza valor de mercado das posições abertas.

    Para mercados com liquidez < MIN_LIQUIDITY_MTM, usa cost basis como proxy
    (current_price = entry_price) em vez do preço stale/fantasma.
    """
    if open_positions.empty or current_markets.empty:
        return open_positions

    cols_needed = [c for c in ("yes_price", "spread", "bestBid", "bestAsk", "liquidity")
                   if c in current_markets.columns]

    prices = (
        current_markets.drop_duplicates("conditionId", keep="last")
        .set_index("conditionId")[cols_needed]
        .to_dict(orient="index")
    )

    has_liq_col = "liquidity" in current_markets.columns

    rows = []
    for _, pos in open_positions.iterrows():
        info      = prices.get(pos["condition_id"], {})
        liquidity = float(info.get("liquidity") or 0)
        # Sem informação de liquidez no snapshot (coluna existe mas valor 0/ausente)
        # → tratar como ILÍQUIDO: preço pode ser stale/fantasma. Antes, liquidez
        # ausente era tratada como líquida e o MTM usava o preço fantasma.
        illiquid  = has_liq_col and liquidity < MIN_LIQUIDITY_MTM

        if illiquid:
            # Sem liquidez — usa cost basis para não inflar P&L com preço fantasma
            current_price = float(pos["entry_price"])
            spread = 0.0
            logger.debug(
                f"MTM ilíquido ({liquidity:.0f} USDC < {MIN_LIQUIDITY_MTM:.0f}) "
                f"— usando cost basis para {pos['condition_id'][:12]}"
            )
        else:
            best_bid = info.get("bestBid")
            best_ask = info.get("bestAsk")
            if best_bid is not None and best_ask is not None \
                    and pd.notna(best_bid) and pd.notna(best_ask):
                # Book real disponível — mesma fórmula de
                # risk_manager.early_exit_positions/ws_feed.evaluate_exit.
                # Vender a mercado executa no bid; "yes_price - spread/2"
                # (fórmula legada, ramo abaixo) descontava o spread duas vezes.
                spread = max(float(best_ask) - float(best_bid), 0.0)
                if pos["direction"] == "BUY_YES":
                    current_price = max(float(best_bid), 0.001)
                else:
                    current_price = max(1.0 - float(best_ask), 0.001)
            else:
                # Fallback legado — snapshot sem bestBid/bestAsk (fixtures de
                # teste, ou colunas ausentes no snapshot).
                yes_price = float(info.get("yes_price") or pos["entry_price"])
                spread    = max(float(info.get("spread") or 0), 0.005)
                if pos["direction"] == "BUY_YES":
                    current_price = max(yes_price - spread / 2, 0.001)
                else:
                    current_price = max((1.0 - yes_price) - spread / 2, 0.001)

        current_value  = current_price * float(pos["shares"])
        unrealized_pnl = current_value - float(pos["cost_usdc"])

        row = dict(pos)
        row["current_price"]  = round(current_price, 4)
        row["spread"]         = round(spread, 4)  # P1-15b: rebalance precisa do ask, não só do bid
        row["current_value"]  = round(current_value, 2)
        row["unrealized_pnl"] = round(unrealized_pnl, 2)
        row["illiquid"]       = illiquid
        rows.append(row)

    return pd.DataFrame(rows)


def rebalance_positions(
    open_positions_mtm: pd.DataFrame,
    portfolio: dict,
    dry_run: bool = False,
) -> list[dict]:
    """
    Aumenta posições abertas quando o edge cresceu desde a entrada (item 4.3).

    Para cada posição aberta com MTM disponível:
      1. Recalcula edge com preço atual (fair_prob = prob_at_entry, não muda)
      2. Recalcula Kelly com o novo edge e preço atual
      3. Se Kelly_novo > cost_usdc × 1.5 → adiciona a diferença
         Cap: adicionar no máximo cost_usdc (total ≤ 2× custo original)

    Retorna lista de dicts com as adições realizadas.
    """
    from risk.risk_manager import kelly_size, normalize_trade_type

    if open_positions_mtm.empty or "current_price" not in open_positions_mtm.columns:
        return []

    cash    = float(portfolio["current_cash"])
    initial = float(portfolio.get("initial_capital", 1000))
    added   = []

    for _, pos in open_positions_mtm.iterrows():
        if pos.get("illiquid"):
            continue

        trade_type   = normalize_trade_type(pos.get("trade_type"), context="rebalance_positions")
        if trade_type == "arb":
            # P1-15a: pernas de arb precisam de shares IGUAIS entre si — é o
            # que torna o basket riskless (open_basket impõe essa invariante
            # na abertura). Rebalancear uma perna isolada quebra a invariante
            # e transforma um arb garantido numa aposta direcional nua sem
            # que ninguém perceba (prob_at_entry de uma perna de arb É o
            # preço de execução, então qualquer queda de preço parece "edge"
            # positivo pro cálculo abaixo).
            continue

        direction    = str(pos["direction"])
        entry_price  = float(pos["entry_price"])
        cost_usdc    = float(pos["cost_usdc"])
        fair_prob    = float(pos.get("prob_at_entry") or 0.5)
        current_price= float(pos["current_price"])
        signal_source= str(pos.get("signal_source", "odds"))
        confidence   = float(pos.get("confidence") or 0.5)
        cid          = str(pos["condition_id"])

        # P1-15b: current_price (de mark_to_market) é o lado BID — o preço
        # que eu receberia se vendesse agora. Aportar significa COMPRAR mais,
        # que executa no ASK. O código antigo usava current_price direto como
        # "new_entry" (comentário dizia "proxy conservador"); na prática é
        # anti-conservador por um spread inteiro — supõe que compra no bid.
        spread    = float(pos.get("spread") or 0.01)
        new_entry = min(current_price + spread, 0.98)

        if direction == "BUY_YES":
            new_edge = fair_prob - current_price
        else:
            # Para BUY_NO: prob_at_entry é a fair_prob de YES ser baixo
            # Edge = (1 - yes_price) - (1 - fair_prob) = fair_prob_no - current_no_price
            # Simplificado: edge = (1 - current_price_no) > fair_prob_yes → yes subiu além do fair
            new_edge = (1.0 - fair_prob) - current_price

        if new_edge <= 0:
            continue  # edge virou negativo — não reforça posição perdedora

        # Kelly com o novo edge e preço atual
        new_kelly = kelly_size(
            edge=new_edge,
            entry_price=new_entry,
            capital=cash,
            confidence=confidence,
            signal_source=signal_source,
            trade_type=trade_type,
        )

        # Só adiciona se Kelly novo > 1.5× posição atual
        if new_kelly <= cost_usdc * 1.5:
            continue

        # Quanto adicionar: diferença entre Kelly novo e posição atual
        # Cap: não pode exceder custo original (total ≤ 2× original)
        max_add    = cost_usdc  # cap em 1× adicional → total = 2×
        add_usdc   = min(new_kelly - cost_usdc, max_add)
        add_usdc   = min(add_usdc, cash * 0.02, cash - 5)  # respeita cash disponível

        if add_usdc < 2.0:
            continue

        add_shares = round(add_usdc / new_entry, 4)
        new_cost   = round(cost_usdc + add_usdc, 2)
        new_shares = round(float(pos["shares"]) + add_shares, 4)
        # P1-15c: entry_price vira custo médio ponderado, não mais o preço da
        # abertura original — de propósito. resolve_positions calcula pnl como
        # (exit-entry)×shares com o shares JÁ somado do aporte; se entry_price
        # ficasse congelado no valor original, esse pnl não reconciliaria com
        # cost_usdc. edge_at_entry (não atualizado aqui) fica como registro do
        # edge da abertura original — é só exibido em log, nada financeiro lê.
        new_avg_entry = round(new_cost / new_shares, 4)

        logger.info(
            f"REBALANCE +${add_usdc:.2f} → {str(pos['question'])[:40]} "
            f"(edge {new_edge:.1%}, Kelly {new_kelly:.2f} > {cost_usdc*1.5:.2f})"
        )

        if not dry_run:
            with contextlib.closing(get_connection()) as conn:
                # Ordem importa: debita o cash PRIMEIRO (com guard) e só então
                # aumenta a posição. Antes, se o guard de cash falhasse, a posição
                # ficava com shares/custo aumentados sem o débito correspondente.
                conn.execute("BEGIN IMMEDIATE")
                cur = conn.execute(
                    "UPDATE portfolio SET current_cash = current_cash - ? "
                    "WHERE id = (SELECT MAX(id) FROM portfolio) AND current_cash >= ?",
                    (add_usdc, add_usdc),
                )
                if cur.rowcount == 0:
                    conn.rollback()
                    logger.warning(f"REBALANCE cancelado [caixa insuficiente] {cid[:16]}")
                    continue
                cur = conn.execute(
                    "UPDATE positions SET shares=?, cost_usdc=?, entry_price=? "
                    "WHERE id=? AND status='open'",
                    (new_shares, new_cost, new_avg_entry, int(pos["id"])),
                )
                if cur.rowcount == 0:
                    conn.rollback()  # posição fechada por outro processo — desfaz o débito
                    logger.info(f"REBALANCE cancelado [posição já fechada] {cid[:16]}")
                    continue
                conn.execute("""
                    INSERT INTO trades_log (action, condition_id, direction, price, shares, usdc_amount, note)
                    VALUES ('REBALANCE', ?, ?, ?, ?, ?, ?)
                """, (cid, direction, new_entry, add_shares, add_usdc,
                      f"edge={new_edge:.3f} kelly_novo={new_kelly:.2f}"))
                conn.commit()

        cash -= add_usdc
        added.append({
            "condition_id": cid,
            "question":     str(pos["question"])[:60],
            "add_usdc":     round(add_usdc, 2),
            "new_edge":     round(new_edge, 4),
            "new_cost":     new_cost,
        })

    return added


def print_portfolio(portfolio: dict, positions_mtm: pd.DataFrame) -> None:
    """Exibe estado do portfólio com P&L e posições abertas."""
    cash      = float(portfolio["current_cash"])
    initial   = float(portfolio["initial_capital"])
    pos_value = float(positions_mtm["current_value"].sum()) if not positions_mtm.empty and "current_value" in positions_mtm.columns else 0.0
    total     = cash + pos_value
    pnl       = total - initial
    pnl_color = "green" if pnl >= 0 else "red"

    console.print("\n[bold cyan]══════════════════════════════════════════════════[/bold cyan]")
    console.print("[bold cyan]   PAPER TRADING — PORTFÓLIO ATUAL               [/bold cyan]")
    console.print("[bold cyan]══════════════════════════════════════════════════[/bold cyan]\n")
    console.print(f"  Capital inicial:  [bold]${initial:>10,.2f}[/bold]")
    console.print(f"  Caixa disponível: [bold]${cash:>10,.2f}[/bold]")
    console.print(f"  Valor posições:   [bold]${pos_value:>10,.2f}[/bold]")
    console.print(f"  Total portfólio:  [bold]${total:>10,.2f}[/bold]")
    console.print(f"  P&L total:        [{pnl_color}][bold]${pnl:>+10,.2f}  ({pnl/initial:+.2%})[/bold][/{pnl_color}]\n")

    if positions_mtm.empty:
        console.print("[dim]Nenhuma posição aberta.[/dim]\n")
        return

    table = Table(box=box.ROUNDED, show_lines=True, title="Posições Abertas")
    table.add_column("#",        width=3,  justify="right")
    table.add_column("Questão",  width=34)
    table.add_column("Fonte",    width=8,  justify="center")
    table.add_column("Dir.",     width=9,  justify="center")
    table.add_column("Entrada",  width=8,  justify="right")
    table.add_column("Atual",    width=8,  justify="right")
    table.add_column("Custo",    width=8,  justify="right")
    table.add_column("P&L",      width=10, justify="right", style="bold")
    table.add_column("Conf.",    width=5,  justify="right")

    for i, row in positions_mtm.iterrows():
        pnl_u   = float(row.get("unrealized_pnl", 0))
        pnl_c   = "green" if pnl_u >= 0 else "red"
        dir_c   = "green" if row["direction"] == "BUY_YES" else "magenta"
        src     = str(row.get("signal_source", "?"))[:7]
        conf    = float(row.get("confidence", 0))
        table.add_row(
            str(i + 1),
            str(row["question"])[:33],
            f"[dim]{src}[/dim]",
            f"[{dir_c}]{row['direction']}[/{dir_c}]",
            f"{float(row['entry_price']):.3f}",
            f"{float(row.get('current_price', row['entry_price'])):.3f}",
            f"${float(row['cost_usdc']):.2f}",
            f"[{pnl_c}]${pnl_u:+.2f}[/{pnl_c}]",
            f"{conf:.2f}",
        )

    console.print(table)


def run_paper_trading(
    initial_capital: float = 1_000.0,
    max_positions: int | None = None,
    edge_threshold: float | None = None,
    min_liquidity: float | None = None,
    dry_run: bool = False,
    top_signals: int | None = None,
    mode: str = "odds",
) -> None:
    """
    Ciclo completo do paper trader:
      1. Resolve posições de mercados fechados (P&L real)
      2. Verifica stop loss por drawdown
      3. Carrega sinais reais (odds/deribit/all)
      4. Abre novas posições via Kelly × confidence
      5. Mark-to-market e exibe portfólio
    """
    from risk.risk_manager import (
        resolve_positions, early_exit_positions, check_drawdown_stop,
        portfolio_risk_summary, MAX_OPEN_POSITIONS,
        MIN_SIGNAL_LIQUIDITY, MAX_SIGNALS_PER_CYCLE,
        reclassify_orphan_arb_legs, find_positions_needing_manual_resolution,
    )

    # Defaults vêm do risk_manager — o CLI antigo travava em 10 posições
    # enquanto o limite real era 20 (constante duplicada, bug da auditoria).
    if max_positions is None:
        max_positions = MAX_OPEN_POSITIONS
    if min_liquidity is None:
        min_liquidity = MIN_SIGNAL_LIQUIDITY
    if top_signals is None:
        top_signals = MAX_SIGNALS_PER_CYCLE

    init_db()
    portfolio = get_or_create_portfolio(initial_capital)
    console.print(f"\n[bold]Paper Trader — capital: ${float(portfolio['initial_capital']):,.0f} USDC | modo: {mode.upper()}[/bold]")
    if dry_run:
        console.print("[yellow][DRY RUN] Nenhuma posição será salva.[/yellow]\n")

    current_markets = load_current_markets()
    open_pos = get_open_positions()

    # ── 1. Resolve posições ─────────────────────────────
    if not open_pos.empty and not current_markets.empty:
        resolved = resolve_positions(open_pos, current_markets, DB_PATH, dry_run=dry_run)
        if resolved:
            console.print(f"[bold]Posições resolvidas: {len(resolved)}[/bold]")
            for r in resolved:
                pnl_c = "green" if r["pnl_usdc"] >= 0 else "red"
                console.print(
                    f"  [{pnl_c}]{r['status'].upper()}[/{pnl_c}] "
                    f"{r['question'][:50]} | "
                    f"[{pnl_c}]P&L=${r['pnl_usdc']:+.2f}[/{pnl_c}]"
                )
            # Recarrega após resolução
            open_pos  = get_open_positions()
            portfolio = get_or_create_portfolio(initial_capital)

    # ── 1b. Pernas de arb órfãs (P0-5) ──────────────────
    # Resolução acima pode ter fechado só parte de um basket — a(s) perna(s)
    # remanescente(s) ficaria(m) nua(s) e sem stop-loss para sempre
    # (EARLY_EXIT["arb"] é hold-forever por design). Reclassifica para 'value'.
    orphans = reclassify_orphan_arb_legs(DB_PATH, dry_run=dry_run)
    if orphans:
        console.print(f"[bold yellow]Pernas de arb órfãs reclassificadas: {len(orphans)}[/bold yellow]")
        try:
            from notify import alert
            groups = sorted({o["arb_group"] for o in orphans})
            alert(
                f"{len(orphans)} perna(s) de arb órfã(s) reclassificada(s) para 'value' "
                f"(basket parcialmente resolvido): {', '.join(groups)}",
                cycle="paper_trader",
            )
        except Exception:
            logger.exception("Falha ao notificar reclassificação de pernas órfãs")
        open_pos  = get_open_positions()
        portfolio = get_or_create_portfolio(initial_capital)

    # ── 1c. Posições precisando de revisão manual (P1-16) ──
    # resolve_positions deixa ABERTA (não fecha com resultado fabricado)
    # qualquer posição que fechou sem outcomePrices parseável. Sem correção
    # automática possível (é decisão do operador), então alerta aqui — só
    # no ciclo de 30min, não no run_execution de 5min, pra não spammar
    # Telegram sobre a mesma posição parada a cada 5 minutos.
    stuck = find_positions_needing_manual_resolution(DB_PATH)
    if not stuck.empty:
        console.print(f"[bold red]Posições precisando de revisão manual: {len(stuck)}[/bold red]")
        try:
            from notify import alert
            questions = "; ".join(str(q)[:40] for q in stuck["question"].tolist())
            alert(
                f"{len(stuck)} posição(ões) fechada(s) sem outcome parseável, "
                f"precisando revisão manual: {questions}",
                cycle="paper_trader",
            )
        except Exception:
            logger.exception("Falha ao notificar posições precisando de revisão manual")

    # ── 2. Saída antecipada ────────────────────────────
    open_pos = get_open_positions()
    if not open_pos.empty and not current_markets.empty:
        early_exits = early_exit_positions(open_pos, current_markets, DB_PATH, dry_run=dry_run)
        if early_exits:
            console.print(f"[bold]Saídas antecipadas: {len(early_exits)}[/bold]")
            for e in early_exits:
                pnl_c = "green" if e["pnl_usdc"] >= 0 else "red"
                console.print(
                    f"  [{pnl_c}]EXIT[/{pnl_c}] {e['question'][:50]}"
                    f" | [{pnl_c}]P&L=${e['pnl_usdc']:+.2f}[/{pnl_c}]"
                    f" | {e['trigger']}"
                )
            open_pos  = get_open_positions()
            portfolio = get_or_create_portfolio(initial_capital)

    # ── 3. Verifica stop loss (valor total, antes de rebalance/baskets/aberturas) ──
    # P1-12: precisa vir ANTES do rebalance — senão um portfólio em halt
    # continua recebendo mais capital em posições já perdedoras via
    # rebalance_positions, e o stop nunca protege delas (só bloqueava
    # abertura de posição nova). total_value = cash + MTM alimenta o cap de
    # drawdown do pico (check_drawdown_stop grava em portfolio_equity); os
    # stops de P&L realizado (semanal/diário) continuam checados também.
    open_pos = get_open_positions()
    mtm_for_stop = mark_to_market(open_pos, current_markets) if not open_pos.empty and not current_markets.empty else open_pos
    pos_value_for_stop = float(mtm_for_stop["current_value"].sum()) if "current_value" in mtm_for_stop.columns else float(open_pos.get("cost_usdc", pd.Series([0])).sum()) if not open_pos.empty else 0.0
    total_value_for_stop = float(portfolio["current_cash"]) + pos_value_for_stop

    stop, reason = check_drawdown_stop(portfolio, DB_PATH, total_value=total_value_for_stop, record=True)
    if stop:
        console.print(f"\n[bold red]STOP LOSS ATIVADO: {reason}[/bold red]")
        console.print("[dim]Novas posições, rebalance e baskets suspensos. Apenas monitorando portfólio atual.[/dim]\n")
        print_portfolio(portfolio, mtm_for_stop if not mtm_for_stop.empty else pd.DataFrame())
        return

    # ── 3b. Rebalanceamento de posições ─────────────────
    open_pos = get_open_positions()
    if not open_pos.empty and not current_markets.empty:
        open_pos_mtm = mark_to_market(open_pos, current_markets)
        rebalanced = rebalance_positions(open_pos_mtm, portfolio, dry_run=dry_run)
        if rebalanced:
            console.print(f"[bold]Rebalanceamentos: {len(rebalanced)}[/bold]")
            for r in rebalanced:
                console.print(
                    f"  [cyan]+${r['add_usdc']:.2f}[/cyan] {r['question'][:50]}"
                    f" | edge={r['new_edge']:.1%}"
                )
            open_pos  = get_open_positions()
            portfolio = get_or_create_portfolio(initial_capital)

    # ── 4.5 Baskets de arb estrutural (execução atômica) ──
    # Só baskets GARANTIDOS do CSV do scanner; cada basket abre todas as
    # pernas na mesma transação ou nenhuma (perna solta = direcional nua).
    baskets = open_arb_baskets(portfolio, current_markets, dry_run=dry_run)
    if baskets:
        console.print(f"[bold]Baskets estruturais abertos: {len(baskets)}[/bold]")
        for b in baskets:
            console.print(
                f"  [green]BASKET[/green] {b['arb_group']}"
                f" | {b['n_legs']} pernas × {b['shares']:.1f} sh"
                f" | custo ${b['total_cost']:.2f} → payout ${b['payout_total']:.2f}"
                f" | [green]lucro garantido ${b['guaranteed_profit']:.2f}[/green]"
            )
        open_pos  = get_open_positions()
        portfolio = get_or_create_portfolio(initial_capital)

    # ── 5. Carrega sinais ───────────────────────────────
    # min_edge é só pré-filtro coarse (MIN_EDGE_ABS por default); o gate fino
    # por fonte (MIN_EDGE_TO_TRADE) é aplicado pelo kelly_size ao abrir.
    console.print(f"\nCarregando sinais ({mode.upper()})...")
    signals_df = load_signals(mode=mode, min_edge=edge_threshold)

    if signals_df.empty:
        console.print(f"[yellow]Nenhum sinal disponível para modo '{mode}'.[/yellow]")
        console.print(f"[dim]Execute: uv run python -m signals.run_signals --mode {mode}[/dim]\n")
    else:
        # Filtra por liquidez
        if "liquidity" in signals_df.columns:
            signals_df = signals_df[
                pd.to_numeric(signals_df["liquidity"], errors="coerce").fillna(0) >= min_liquidity
            ]

        signals_top = signals_df.head(top_signals)
        # Pernas de arb não ocupam slots direcionais (mesma regra do check_exposure);
        # nem posições travadas esperando revisão manual (P1-16) — mercado já fechou.
        open_pos_for_slots = open_pos
        if not open_pos.empty and "needs_manual_resolution" in open_pos.columns:
            open_pos_for_slots = open_pos[open_pos["needs_manual_resolution"].fillna(0) != 1]
        if not open_pos_for_slots.empty and "trade_type" in open_pos_for_slots.columns:
            n_open = int((open_pos_for_slots["trade_type"].fillna("value") != "arb").sum())
        else:
            n_open = len(open_pos_for_slots) if not open_pos_for_slots.empty else 0
        slots   = max(0, max_positions - n_open)
        cash    = float(portfolio["current_cash"])

        console.print(f"  Sinais disponíveis: {len(signals_top)} | Posições: {n_open}/{max_positions} | Caixa: ${cash:,.2f}\n")

        # ── 6. Abre posições ────────────────────────────
        # P1-19: current_markets/traded_ids/open_pos içados pro chamador —
        # sem isso, open_position relia o parquet inteiro e reconsultava o
        # banco a cada candidato (até 30 leituras de parquet por ciclo).
        # open_pos e traded_ids são atualizados aqui a cada abertura pra não
        # perder o efeito de uma posição aberta 2 candidatos atrás nesse
        # mesmo ciclo (caps de diversificação, checagem de duplicata). A
        # linha apendada em open_pos não tem "id" (só existe depois do
        # INSERT) — serve só pra leitura de caps/duplicata neste loop, não
        # pra um caminho de escrita (mark_to_market/early_exit/resolve
        # precisam do id de verdade e rodam com open_pos recarregado do
        # banco no próximo passo do ciclo, não com este).
        traded_ids = get_traded_condition_ids()
        opened = skipped = 0
        for _, sig in signals_top.iterrows():
            if slots <= 0 or cash < 5:
                break
            pos = open_position(
                portfolio, sig.to_dict(), dry_run=dry_run,
                current_markets=current_markets, open_positions=open_pos, traded_ids=traded_ids,
            )
            if pos:
                dir_c = "green" if pos["direction"] == "BUY_YES" else "magenta"
                console.print(
                    f"  [{dir_c}]ABERTA[/{dir_c}] {str(sig.get('question',''))[:48]}"
                    f" | ${pos['cost_usdc']:.2f} @ {pos['entry_price']:.3f}"
                    f" | edge={pos['edge_at_entry']:.3f} conf={pos['confidence']:.2f}"
                )
                opened += 1
                slots  -= 1
                cash   -= pos["cost_usdc"]
                portfolio["current_cash"] = cash
                if not dry_run:
                    traded_ids.add(str(pos["condition_id"]))
                    open_pos = pd.concat(
                        [open_pos, pd.DataFrame([{**pos, "status": "open", "needs_manual_resolution": 0}])],
                        ignore_index=True,
                    )
            else:
                skipped += 1

        if opened:
            console.print(f"\n[green]{opened} posição(ões) aberta(s)[/green]  ({skipped} rejeitadas — ver logs INFO para motivo)\n")
        else:
            console.print(f"[dim]Nenhuma posição nova — {skipped} sinais rejeitados. Motivos nos logs (INFO).[/dim]\n")

    # ── 7. Mark-to-market e exibição ───────────────────
    open_pos = get_open_positions()
    if not open_pos.empty and not current_markets.empty:
        positions_mtm = mark_to_market(open_pos, current_markets)
    else:
        positions_mtm = open_pos if not open_pos.empty else pd.DataFrame()

    with contextlib.closing(get_connection()) as conn:
        row = conn.execute("SELECT * FROM portfolio ORDER BY id DESC LIMIT 1").fetchone()
        portfolio = dict(row)

    print_portfolio(portfolio, positions_mtm)

    # Sumário de risco — usa as posições COM mark-to-market, senão o
    # unrealized_pnl sai sempre $0 (bug da auditoria)
    risk = portfolio_risk_summary(portfolio, positions_mtm, DB_PATH)
    if risk["category_exposure"]:
        console.print("[bold]Exposição por categoria:[/bold]")
        for cat, pct in sorted(risk["category_exposure"].items(), key=lambda x: -x[1]):
            bar = "█" * int(pct * 20)
            console.print(f"  {cat:<15} {bar} {pct:.1%}")
    if risk["source_exposure"]:
        console.print("\n[bold]Exposição por fonte:[/bold]")
        for src, pct in sorted(risk["source_exposure"].items(), key=lambda x: -x[1]):
            console.print(f"  {src:<10} {pct:.1%}")

    # Salva CSV
    if not positions_mtm.empty:
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = REPORTS_DIR / f"portfolio_{ts}.csv"
        positions_mtm.to_csv(path, index=False)
        console.print(f"\n[dim]Portfólio salvo: {path}[/dim]")
