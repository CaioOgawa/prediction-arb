"""
risk_manager.py
Gestão de risco centralizada para o paper trader e (futuramente) live trader.

Responsabilidades:
  1. Kelly criterion fracionado com confidence modulation
  2. Limites de exposição: por posição, categoria e fonte de sinal
  3. Stop loss semanal por drawdown
  4. Resolução de posições: detecta mercados fechados e calcula P&L real

Design principle:
  Cada camada de risco é independente — falha em uma não impede as outras.
  O objetivo é sobreviver, não maximizar: errar pequeno é preferível a acertar
  grande mas explodir a conta num drawdown.
"""

import json
import sqlite3
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

# ──────────────────────────────────────────────────────────
# Parâmetros de risco — modificar aqui para ajustar o perfil
# ──────────────────────────────────────────────────────────

# Tamanho de posição
MIN_EDGE_TO_TRADE = {
    "odds":    0.08,   # Bookmakers: edge bem calibrado, exige mais
    "deribit": 0.05,   # B-S risk-neutral subestima drift → threshold menor
    "ml":      0.08,   # Modelo tem data leakage → conservador
}
KELLY_MAX_FRAC    = 0.25   # Quarter-Kelly — reduz variância mantendo EV

# Piso absoluto de edge usado como pré-filtro coarse (load_signals e re-cotação
# do paper_trader). O gate fino por fonte é MIN_EDGE_TO_TRADE, aplicado no kelly_size.
MIN_EDGE_ABS      = 0.03

# Tamanho por tipo de trade
# Momentum: posições menores, rotação rápida (saída antes da resolução)
# Value: posições maiores, hold até resolução (edge fundamental)
MAX_POSITION_PCT = {
    "momentum": 0.015,   # 1.5% — trades rápidos, muita rotação
    "value":    0.030,   # 3.0% — hold longer, mais convicção
}
MAX_POSITION_PCT_DEFAULT = 0.020   # fallback para sinais sem trade_type

# Diversificação
MAX_CATEGORY_PCT   = 0.30   # 30% do capital por categoria
MAX_SOURCE_PCT     = 0.60   # 60% por fonte de sinal
MAX_OPEN_POSITIONS = 20     # Mais posições simultâneas para diversificar
MAX_MOMENTUM_POS   = 12     # Máx de posições momentum abertas (rotação rápida)
MAX_VALUE_POS      = 10     # Máx de posições value abertas (hold longo)

# Drawdown
WEEKLY_STOP_PCT   = 0.10   # Halt de novas posições se perder 10% em 7 dias
DAILY_STOP_PCT    = 0.05   # Halt diário se perder 5% no dia

# ── Arb estrutural (baskets multi-perna, trade_type="arb") ──
# O lucro é garantido pela estrutura lógica dos mercados, não por modelo —
# por isso os limites são por basket/liquidez, não por Kelly (edge garantido
# daria Kelly infinito). Só baskets GARANTIDOS são executados (guaranteed=True).
ARB_MIN_PROFIT     = 0.02  # lucro mínimo por $1 de payout (report no scanner + re-verificação na abertura)
ARB_MAX_BASKET_PCT = 0.05  # custo máximo de um basket: 5% do capital inicial
MAX_ARB_BASKETS    = 5     # baskets simultâneos (limite próprio; pernas não contam no MAX_OPEN_POSITIONS)
ARB_LIQUIDITY_FRAC = 0.05  # basket consome no máx 5% da liquidez da perna mais fina

# ── Saída antecipada por tipo de trade ─────────────────────
#
# MOMENTUM: apostamos na convergência de preço antes do evento.
#   - Sai cedo quando o preço se move a favor (1.5× já é excelente para intraday)
#   - Edge flip apertado: se o mercado vai contra 12pp, thesis foi invalidada rápido
#   - Min hold menor: preços mudam rápido perto de eventos
#
# VALUE: apostamos na probabilidade fundamental (hold até resolução).
#   - Profit target mais alto (2.5×) — queremos deixar correr
#   - Edge flip mais folgado (25pp): aguenta ruído intraday, só sai se inversão real
#   - Min hold maior: ignora spike de curto prazo
#
EARLY_EXIT = {
    "momentum": {
        "profit_target_mult": 1.5,   # Sai quando valor = 1.5× custo
        "edge_flip_delta":    0.12,  # Sai se YES moveu 12pp contra a tese
        "min_hold_hours":     1.0,   # Aguarda 1h antes de early exit
    },
    "value": {
        "profit_target_mult": 2.5,   # Sai quando valor = 2.5× custo
        "edge_flip_delta":    0.25,  # Sai só se inversão forte (25pp)
        "min_hold_hours":     4.0,   # Aguarda 4h — ignora ruído de curto prazo
    },
    # ARB: pernas de basket estrutural NUNCA saem antes da resolução.
    # Fechar uma perna isolada destrói a garantia e vira posição direcional nua
    # — o P&L de cada perna não significa nada, só o basket completo importa.
    "arb": {
        "profit_target_mult": float("inf"),
        "edge_flip_delta":    float("inf"),
        "min_hold_hours":     float("inf"),
    },
}
# Defaults para posições sem trade_type
EARLY_EXIT_DEFAULT = {
    "profit_target_mult": 2.0,
    "edge_flip_delta":    0.20,
    "min_hold_hours":     2.0,
}

# Stop por posição — corta a perda quando o valor cai abaixo de (1 - X) do custo.
# FONTE ÚNICA: o ws_feed importa daqui (antes era hardcoded lá, fora do risk_manager).
# Respeita min_hold_hours do trade_type — um book momentaneamente vazio/largo não
# pode executar a posição (incidente 2026-05: 51 posições mortas a 0.001 por isso).
POSITION_STOP_LOSS = 0.50

# Confiança mínima por fonte
MIN_CONFIDENCE = {
    "odds":    0.15,   # Odds Pinnacle são confiáveis mesmo com conf baixa
    "deribit": 0.08,   # Mercados EOY têm confiança estruturalmente menor (drift mismatch)
    "ml":      0.50,   # Modelo com leakage — só opera com confiança muito alta
}


# ──────────────────────────────────────────────────────────
# Kelly Criterion
# ──────────────────────────────────────────────────────────

def kelly_size(
    edge: float,
    entry_price: float,
    capital: float,
    confidence: float = 1.0,
    signal_source: str = "odds",
    trade_type: str = "value",
) -> float:
    """
    Calcula o tamanho da posição em USDC via Kelly fracionado.

    Fórmula para mercado binário (aposta num token de prediction market):
        f* = edge / (ganho_por_unidade)
           = edge / ((1 / entry_price) - 1)
           = edge * entry_price / (1 - entry_price)

    O edge aqui é a diferença entre a probabilidade real estimada e o preço
    pago pelo token. Exemplo: se o token YES custa 0.40 e a fair prob é 0.55,
    edge = 0.15 e o ganho por unidade = (1/0.40) - 1 = 1.5x.

    Ajustes conservadores:
      - Multiplica pelo KELLY_MAX_FRAC (quarter-Kelly padrão)
      - Multiplica pelo confidence score (0–1) da fonte de sinal
      - Aplica hard cap de MAX_POSITION_PCT do capital

    Args:
        edge:          |fair_prob - market_price|, esperança de lucro por unidade
        entry_price:   preço pago pelo token (já com spread/slippage)
        capital:       caixa disponível em USDC
        confidence:    score de confiança do sinal (0–1), calculado em signal_generator
        signal_source: 'odds' | 'deribit' | 'ml'

    Returns:
        Tamanho da posição em USDC (0 se não deve operar).
    """
    min_edge = MIN_EDGE_TO_TRADE.get(signal_source, 0.08) if isinstance(MIN_EDGE_TO_TRADE, dict) else MIN_EDGE_TO_TRADE
    if edge < min_edge:
        return 0.0
    if entry_price <= 0 or entry_price >= 1:
        return 0.0
    if capital <= 0:
        return 0.0

    # Garante confiança mínima por fonte
    min_conf = MIN_CONFIDENCE.get(signal_source, 0.20)
    if confidence < min_conf:
        logger.debug(f"Confiança {confidence:.2f} abaixo do mínimo {min_conf} para {signal_source}")
        return 0.0

    # Full Kelly fraction
    odds_ratio = (1.0 / entry_price) - 1.0        # ganho por unidade apostada
    kelly_full = edge / odds_ratio                  # f* = edge / odds

    # Clamp kelly_full antes de aplicar confidence:
    # tokens deep-OTM (entry_price << 0.10) geram odds_ratio > 9 e kelly_full absurdo.
    # Cap em 0.10 (10% do capital como fração máxima pré-ajuste) evita posições gigantes.
    kelly_full = min(kelly_full, 0.10)

    # Cap de posição por trade_type
    pos_pct  = MAX_POSITION_PCT.get(trade_type, MAX_POSITION_PCT_DEFAULT)

    # Aplica frações conservadoras
    kelly_used = kelly_full * KELLY_MAX_FRAC * confidence
    kelly_used = float(np.clip(kelly_used, 0.0, pos_pct))

    size     = capital * kelly_used
    hard_cap = capital * pos_pct

    result = round(min(size, hard_cap), 2)

    # Posições abaixo do mínimo executável são DESCARTADAS, não infladas.
    # O floor antigo subia $0.50 → $3, dando boost justamente aos sinais em que
    # o Kelly×confidence tinha menos convicção — inversão da lógica de risco.
    min_size = 3.0 if signal_source == "deribit" else 2.0
    if result < min_size:
        return 0.0

    return result


# ──────────────────────────────────────────────────────────
# Verificações de exposição
# ──────────────────────────────────────────────────────────

def check_exposure(
    signal: dict,
    open_positions: pd.DataFrame,
    portfolio: dict,
    size_usdc: float,
) -> tuple[bool, str]:
    """
    Verifica se abrir esta posição viola os limites de diversificação.

    Returns:
        (pode_operar, motivo_se_nao)
    """
    initial_capital = float(portfolio.get("initial_capital", 1000))

    if open_positions.empty:
        return True, ""

    # 1. Posição duplicada no mesmo mercado
    cid = signal.get("condition_id", "")
    if cid and cid in open_positions["condition_id"].values:
        return False, f"posição já aberta em {cid[:16]}"

    # 2. Limite global de posições simultâneas
    # Pernas de arb (trade_type='arb') não contam: têm limite próprio
    # (MAX_ARB_BASKETS) e não podem espremer os slots direcionais.
    if "trade_type" in open_positions.columns:
        n_directional = int((open_positions["trade_type"].fillna("value") != "arb").sum())
    else:
        n_directional = len(open_positions)
    if n_directional >= MAX_OPEN_POSITIONS:
        return False, f"limite global de {MAX_OPEN_POSITIONS} posições atingido"

    # 3. Limite por trade_type
    trade_type = str(signal.get("trade_type", "value")).lower()
    if "trade_type" in open_positions.columns:
        type_count = (open_positions["trade_type"] == trade_type).sum()
        type_limit = MAX_MOMENTUM_POS if trade_type == "momentum" else MAX_VALUE_POS
        if type_count >= type_limit:
            return False, f"limite de posições {trade_type} ({type_limit}) atingido"

    # 4. Limite por categoria
    category = str(signal.get("category", "")).lower()
    if category and "category" in open_positions.columns:
        cat_cost = open_positions[
            open_positions["category"].str.lower() == category
        ]["cost_usdc"].sum()
        if (cat_cost + size_usdc) / initial_capital > MAX_CATEGORY_PCT:
            return False, f"limite de categoria '{category}' ({MAX_CATEGORY_PCT:.0%}) atingido"

    # 5. Limite por fonte de sinal
    source = str(signal.get("signal_source", "")).lower()
    if source and "signal_source" in open_positions.columns:
        src_cost = open_positions[
            open_positions["signal_source"].str.lower() == source
        ]["cost_usdc"].sum()
        if (src_cost + size_usdc) / initial_capital > MAX_SOURCE_PCT:
            return False, f"limite de fonte '{source}' ({MAX_SOURCE_PCT:.0%}) atingido"

    # 6. Correlação: mesmo evento (event_slug) já tem posição aberta
    # Evita abrir dois lados opostos do mesmo jogo/evento (ex: Lakers win + Celtics win).
    # ISENÇÃO para source=structural: baskets negRisk são, por construção,
    # N pernas do MESMO evento — a correlação é exatamente o que garante o lucro.
    source_is_structural = str(signal.get("signal_source", "")).lower() == "structural"
    event_slug = str(signal.get("event_slug", "")).strip()
    if event_slug and not source_is_structural and "event_slug" in open_positions.columns:
        same_event = open_positions[
            open_positions["event_slug"].fillna("") == event_slug
        ]
        if not same_event.empty:
            existing_dirs = same_event["direction"].unique().tolist()
            return False, (
                f"evento '{event_slug[:40]}' já tem posição aberta "
                f"({', '.join(existing_dirs)}) — correlação bloqueada"
            )

    return True, ""


def check_drawdown_stop(portfolio: dict, db_path: Path) -> tuple[bool, str]:
    """
    Verifica se o drawdown semanal/diário atingiu o stop.

    Returns:
        (deve_parar, motivo)
    """
    try:
        conn = sqlite3.connect(db_path); conn.execute("PRAGMA journal_mode=WAL")
        conn.row_factory = sqlite3.Row

        now      = datetime.now(timezone.utc)
        week_ago = (now - timedelta(days=7)).strftime("%Y-%m-%d %H:%M:%S")
        day_ago  = (now - timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S")

        # P&L realizado nos últimos 7 dias (posições fechadas)
        weekly_pnl = conn.execute("""
            SELECT COALESCE(SUM(pnl_usdc), 0)
            FROM positions
            WHERE status IN ('closed', 'expired')
              AND closed_at >= ?
        """, (week_ago,)).fetchone()[0] or 0.0

        # P&L realizado hoje
        daily_pnl = conn.execute("""
            SELECT COALESCE(SUM(pnl_usdc), 0)
            FROM positions
            WHERE status IN ('closed', 'expired')
              AND closed_at >= ?
        """, (day_ago,)).fetchone()[0] or 0.0

        conn.close()

        initial = float(portfolio.get("initial_capital", 1000))

        if weekly_pnl / initial < -WEEKLY_STOP_PCT:
            return True, f"stop loss semanal ativado: P&L={weekly_pnl:+.2f} ({weekly_pnl/initial:+.1%})"

        if daily_pnl / initial < -DAILY_STOP_PCT:
            return True, f"stop loss diário ativado: P&L={daily_pnl:+.2f} ({daily_pnl/initial:+.1%})"

    except Exception as e:
        logger.warning(f"Erro ao verificar drawdown: {e}")

    return False, ""


# ──────────────────────────────────────────────────────────
# Resolução de posições
# ──────────────────────────────────────────────────────────

def _parse_outcome_prices(raw) -> tuple[float, float] | None:
    """
    Extrai (p_yes, p_no) de outcomePrices (pode ser string JSON ou lista).
    Retorna None se inválido.
    """
    if raw is None:
        return None
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except Exception:
            return None
    if not isinstance(raw, list) or len(raw) < 2:
        return None
    try:
        return float(raw[0]), float(raw[1])
    except (ValueError, TypeError):
        return None


def resolve_positions(
    open_positions: pd.DataFrame,
    current_markets: pd.DataFrame,
    db_path: Path,
    dry_run: bool = False,
) -> list[dict]:
    """
    Verifica se alguma posição aberta tem mercado resolvido e fecha com P&L real.

    Um mercado está resolvido quando:
      - closed=True  OU  outcomePrices convergiu (>= 0.95 em um lado)

    P&L calculado:
      - BUY_YES + resolved YES:  lucro = (1.0 - entry_price) × shares
      - BUY_YES + resolved NO:   perda = -entry_price × shares
      - BUY_NO  + resolved NO:   lucro = (1.0 - entry_price) × shares
      - BUY_NO  + resolved YES:  perda = -entry_price × shares

    Returns:
        Lista de dicts com posições resolvidas e seus P&Ls.
    """
    if open_positions.empty or current_markets.empty:
        return []

    # P0-3: preço de mercado NÃO é resolução. Um YES a 0.03 é só "quase todo mundo
    # acha que não" — não significa que o evento já aconteceu. Tratar isso como
    # resolução fechava posições vivas a -100% meses antes do vencimento (posição
    # ETH-$10k resolvida "NO" 31min após abertura, vencimento em dezembro).
    # Preço extremo só conta como resolução quando o mercado TAMBÉM já venceu.
    RESOLVE_PRICE_THRESHOLD = 0.999

    # Indexa mercados pelo conditionId
    mkt_index = current_markets.drop_duplicates("conditionId").set_index("conditionId")

    now_utc = datetime.now(timezone.utc)
    resolved = []
    conn = sqlite3.connect(db_path) if not dry_run else None

    try:
        for _, pos in open_positions.iterrows():
            cid = pos["condition_id"]
            if cid not in mkt_index.index:
                continue

            mkt = mkt_index.loc[cid]

            # Verifica se fechou
            is_closed = bool(mkt.get("closed", False))

            end_date_raw = mkt.get("endDate") or mkt.get("end_date")
            end_date = pd.to_datetime(end_date_raw, utc=True, errors="coerce")
            past_end_date = bool(pd.notna(end_date) and end_date < now_utc)

            outcome_raw = mkt.get("outcomePrices")
            prices = _parse_outcome_prices(outcome_raw)

            outcome = None  # 1 = YES ganhou, 0 = NO ganhou
            if prices is not None and (is_closed or past_end_date):
                p_yes, p_no = prices
                if p_yes >= RESOLVE_PRICE_THRESHOLD:
                    outcome = 1
                elif p_no >= RESOLVE_PRICE_THRESHOLD:
                    outcome = 0

            if not is_closed and outcome is None:
                continue  # mercado ainda aberto (ou vencido sem convergência clara)

            # Calcula P&L
            direction   = pos["direction"]
            entry_price = float(pos["entry_price"])
            shares      = float(pos["shares"])
            cost_usdc   = float(pos["cost_usdc"])

            if outcome is None:
                # Mercado fechado sem resolução clara — marca como expirado e
                # DEVOLVE o custo integral (P&L = 0). Sem outcome não há como
                # calcular ganho/perda; exit_price = entry é só o registro disso.
                exit_price = entry_price
                pnl_usdc   = 0.0
                status     = "expired"
            else:
                # YES ganhou = token YES vale 1.0, token NO vale 0.0
                yes_won = (outcome == 1)
                if (direction == "BUY_YES" and yes_won) or (direction == "BUY_NO" and not yes_won):
                    exit_price = 1.0
                    pnl_usdc   = round((exit_price - entry_price) * shares, 2)
                else:
                    exit_price = 0.0
                    pnl_usdc   = round(-cost_usdc, 2)
                status = "closed"

            resolved_pos = {
                "id":          int(pos["id"]),
                "condition_id": cid,
                "question":    pos.get("question", "")[:60],
                "direction":   direction,
                "entry_price": entry_price,
                "exit_price":  exit_price,
                "pnl_usdc":    pnl_usdc,
                "status":      status,
            }
            resolved.append(resolved_pos)

            if not dry_run and conn:
                # Transação por posição: crash no meio do batch não deixa
                # posição fechada sem o cash creditado (bug de atomicidade).
                now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
                conn.execute("BEGIN IMMEDIATE")
                cur = conn.execute("""
                    UPDATE positions
                    SET closed_at=?, exit_price=?, pnl_usdc=?, status=?
                    WHERE id=? AND status NOT IN ('closed', 'expired')
                """, (now_str, exit_price, pnl_usdc, status, int(pos["id"])))
                if cur.rowcount == 0:
                    conn.rollback()
                    logger.warning(f"Posição {pos['id']} já resolvida — pulando atualização de cash")
                    resolved.pop()  # não reportar resolução que não aconteceu
                    continue
                conn.execute("""
                    UPDATE portfolio
                    SET current_cash = current_cash + ?
                    WHERE id = (SELECT MAX(id) FROM portfolio)
                """, (cost_usdc + pnl_usdc,))
                conn.execute("""
                    INSERT INTO trades_log
                      (action, condition_id, direction, price, shares, usdc_amount, note)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    status.upper(), cid, direction,
                    exit_price, shares, cost_usdc + pnl_usdc,
                    f"resolved outcome={'YES' if outcome==1 else 'NO' if outcome==0 else 'expired'}",
                ))
                conn.commit()

    finally:
        if conn:
            conn.close()

    return resolved


def find_orphan_arb_legs(db_path: Path) -> pd.DataFrame:
    """
    P0-5: `open_basket` abre todas as pernas numa única transação, mas a
    resolução é por posição — se uma perna do arb_group resolve e outra não,
    a perna remanescente vira posição direcional NUA. `EARLY_EXIT["arb"]` é
    hold-forever por design (correto enquanto o basket está intacto: fechar
    uma perna isolada destruiria a garantia), então uma perna órfã nunca teria
    stop-loss se ninguém a tirasse do grupo "arb".

    Retorna as pernas 'arb'/'open' cujo arb_group já tem alguma perna
    'closed'/'expired' — i.e., a garantia do basket já foi rompida.
    """
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query("""
            SELECT id, arb_group, condition_id, status
            FROM positions
            WHERE trade_type = 'arb' AND arb_group IS NOT NULL AND arb_group != ''
        """, conn)
    finally:
        conn.close()

    if df.empty:
        return df

    counts = df.groupby("arb_group")["status"].agg(
        n_total="count",
        n_open=lambda s: (s == "open").sum(),
    )
    orphan_groups = counts[(counts["n_open"] > 0) & (counts["n_open"] < counts["n_total"])].index
    return df[df["arb_group"].isin(orphan_groups) & (df["status"] == "open")].copy()


def reclassify_orphan_arb_legs(db_path: Path, dry_run: bool = False) -> list[dict]:
    """
    Reclassifica pernas órfãs (ver find_orphan_arb_legs) de trade_type='arb'
    para 'value' — reativa profit_target/stop_loss/edge_flip do EARLY_EXIT,
    que a perna nua nunca teria enquanto marcada como 'arb'.

    Idempotente: uma vez reclassificada, a perna some do próximo
    find_orphan_arb_legs (deixa de ter trade_type='arb').
    """
    orphans = find_orphan_arb_legs(db_path)
    if orphans.empty:
        return []

    reclassified = []
    conn = None if dry_run else sqlite3.connect(db_path)
    try:
        for _, leg in orphans.iterrows():
            leg_id = int(leg["id"])
            if conn:
                conn.execute("BEGIN IMMEDIATE")
                cur = conn.execute("""
                    UPDATE positions SET trade_type = 'value'
                    WHERE id = ? AND status = 'open' AND trade_type = 'arb'
                """, (leg_id,))
                if cur.rowcount == 0:
                    conn.rollback()
                    continue
                conn.commit()
            reclassified.append({
                "id":           leg_id,
                "arb_group":    leg["arb_group"],
                "condition_id": leg["condition_id"],
            })
            logger.warning(
                f"P0-5: perna órfã reclassificada arb→value | posição {leg_id} "
                f"arb_group={leg['arb_group']} — basket parcialmente resolvido, "
                f"perna remanescente ganha stop-loss"
            )
    finally:
        if conn:
            conn.close()

    return reclassified


# ──────────────────────────────────────────────────────────
# Saída antecipada de posições
# ──────────────────────────────────────────────────────────

def early_exit_positions(
    open_positions: pd.DataFrame,
    current_markets: pd.DataFrame,
    db_path: Path,
    dry_run: bool = False,
) -> list[dict]:
    """
    Verifica posições abertas e fecha antecipadamente quando um dos gatilhos é ativado:

      1. Profit target  — valor atual >= custo × PROFIT_TARGET_MULT (padrão: 5×)
         Ex: comprou NO a 0.006 por $1.48; NO subiu a 0.031 → valor $7.65 → 5.2× → sair.

      2. Stop por posição — valor atual <= custo × (1 - POSITION_STOP_LOSS) (padrão: 50%)
         Ex: custo $1.48, valor caiu para $0.74 → cortar perda.

      3. Edge flip — mercado invalidou a tese original (relativo ao entry):
         - BUY_NO:  YES subiu mais de EDGE_FLIP_DELTA pp acima do entry_yes
         - BUY_YES: YES caiu mais de EDGE_FLIP_DELTA pp abaixo do entry_yes
         Ex: compramos NO (entry_no=0.40, entry_yes=0.60); YES foi a 0.81 → saída (+21pp).
         Nota: para posições onde YES já era alto (e.g. 0.88), o threshold fica > 1.0 →
         edge_flip não dispara e o stop_loss por posição cuida de perdas severas.

    Proteção contra ruído intraday: early exits não disparam dentro das primeiras
    MIN_HOLD_HOURS horas após abertura (evita stop-out por spike temporário de preço).

    O preço de saída é o current_price do mark-to-market (bid/ask com spread).
    P&L é calculado como: (exit_price - entry_price) × shares.

    Returns:
        Lista de dicts com as posições encerradas antecipadamente.
    """
    if open_positions.empty or current_markets.empty:
        return []

    # Indexa preços atuais por conditionId
    mkt_prices = (
        current_markets.drop_duplicates("conditionId", keep="last")
        .set_index("conditionId")[["yes_price", "spread"]]
        .to_dict(orient="index")
    )

    exits = []
    conn  = sqlite3.connect(db_path) if not dry_run else None
    now   = datetime.now(timezone.utc)

    try:
        for _, pos in open_positions.iterrows():
            cid = pos["condition_id"]
            info = mkt_prices.get(cid, {})
            if not info:
                continue

            # Parâmetros de early exit dependem do trade_type da posição
            trade_type  = str(pos.get("trade_type", "value"))
            exit_params = EARLY_EXIT.get(trade_type, EARLY_EXIT_DEFAULT)
            profit_mult = exit_params["profit_target_mult"]
            flip_delta  = exit_params["edge_flip_delta"]
            min_hold    = exit_params["min_hold_hours"]

            # Calcula tempo de hold (usado para bloquear edge_flip prematuros)
            hold_hours = None
            opened_at_str = pos.get("opened_at", "")
            if opened_at_str:
                try:
                    opened_at = datetime.fromisoformat(
                        str(opened_at_str).replace("Z", "+00:00")
                    )
                    if opened_at.tzinfo is None:
                        opened_at = opened_at.replace(tzinfo=timezone.utc)
                    hold_hours = (now - opened_at).total_seconds() / 3600
                except Exception:
                    pass

            yes_price   = float(info.get("yes_price") or 0.5)
            spread      = max(float(info.get("spread") or 0), 0.005)
            direction   = pos["direction"]
            entry_price = float(pos["entry_price"])
            shares      = float(pos["shares"])
            cost_usdc   = float(pos["cost_usdc"])

            # Preço de saída (bid com spread)
            if direction == "BUY_YES":
                exit_price = max(yes_price - spread / 2, 0.001)
            else:
                exit_price = max((1.0 - yes_price) - spread / 2, 0.001)

            current_value = exit_price * shares
            pnl_usdc      = round((exit_price - entry_price) * shares, 2)

            # YES implícito no entry (entry_price é o token que compramos)
            if direction == "BUY_YES":
                entry_yes = entry_price
            else:
                entry_yes = 1.0 - entry_price  # entry_price = NO token

            # ── Avalia gatilhos ─────────────────────────────
            trigger = None

            # profit_target: dispara sempre — spike de lucro não é ruído
            if current_value >= cost_usdc * profit_mult:
                trigger = f"profit_target ({current_value/cost_usdc:.1f}× custo, tipo={trade_type})"

            # stop_loss e edge_flip: só após min_hold_hours — evita execução
            # por ruído intraday ou preço fantasma de book vazio
            elif hold_hours is not None and hold_hours >= min_hold:
                if current_value <= cost_usdc * (1.0 - POSITION_STOP_LOSS):
                    trigger = (
                        f"stop_loss (${current_value:.2f} = "
                        f"{current_value/cost_usdc:.0%} do custo, tipo={trade_type})"
                    )
                elif direction == "BUY_NO" and yes_price > entry_yes + flip_delta:
                    trigger = (
                        f"edge_flip_no (YES={yes_price:.2f}, entry_yes={entry_yes:.2f}, "
                        f"Δ={yes_price - entry_yes:+.2f} > {flip_delta:.0%}, tipo={trade_type})"
                    )
                elif direction == "BUY_YES" and yes_price < entry_yes - flip_delta:
                    trigger = (
                        f"edge_flip_yes (YES={yes_price:.2f}, entry_yes={entry_yes:.2f}, "
                        f"Δ={entry_yes - yes_price:+.2f} > {flip_delta:.0%}, tipo={trade_type})"
                    )

            if trigger is None:
                continue

            exit_record = {
                "id":           int(pos["id"]),
                "condition_id": cid,
                "question":     str(pos.get("question", ""))[:60],
                "direction":    direction,
                "entry_price":  entry_price,
                "exit_price":   round(exit_price, 4),
                "pnl_usdc":     pnl_usdc,
                "trigger":      trigger,
                "status":       "closed",
            }
            exits.append(exit_record)

            if not dry_run and conn:
                # Guard de status + transação por posição: o ws_feed e o
                # run_execution também fecham posições — sem o guard, um exit
                # concorrente creditaria o cash em dobro.
                now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
                conn.execute("BEGIN IMMEDIATE")
                cur = conn.execute("""
                    UPDATE positions
                    SET closed_at=?, exit_price=?, pnl_usdc=?, status='closed'
                    WHERE id=? AND status='open'
                """, (now_str, round(exit_price, 4), pnl_usdc, int(pos["id"])))
                if cur.rowcount == 0:
                    conn.rollback()
                    logger.info(f"Posição {pos['id']} já fechada por outro processo — pulando")
                    exits.pop()  # não reportar exit que não aconteceu
                    continue
                conn.execute("""
                    UPDATE portfolio
                    SET current_cash = current_cash + ?
                    WHERE id = (SELECT MAX(id) FROM portfolio)
                """, (cost_usdc + pnl_usdc,))
                conn.execute("""
                    INSERT INTO trades_log
                      (action, condition_id, direction, price, shares, usdc_amount, note)
                    VALUES ('EARLY_EXIT', ?, ?, ?, ?, ?, ?)
                """, (
                    cid, direction, round(exit_price, 4), shares,
                    cost_usdc + pnl_usdc, trigger,
                ))
                conn.commit()

    finally:
        if conn:
            conn.close()

    return exits


# ──────────────────────────────────────────────────────────
# Sumário de risco do portfólio
# ──────────────────────────────────────────────────────────

def portfolio_risk_summary(
    portfolio: dict,
    open_positions: pd.DataFrame,
    db_path: Path,
) -> dict:
    """
    Retorna um sumário de métricas de risco do portfólio atual.
    Útil para exibir no terminal e para decisões de sizing.
    """
    initial = float(portfolio.get("initial_capital", 1000))
    cash    = float(portfolio.get("current_cash", 0))

    if open_positions.empty:
        pos_value = 0.0
        category_exposure = {}
        source_exposure   = {}
    else:
        # Usa current_value (mark-to-market) se disponível — calculado pelo paper_trader.
        # Se não disponível (chamada direta sem MTM prévio), cai back para cost_usdc.
        if "current_value" in open_positions.columns:
            pos_value = float(open_positions["current_value"].sum())
        else:
            pos_value = float(open_positions.get("cost_usdc", pd.Series([0])).sum())
        category_exposure = (
            open_positions.groupby("category")["cost_usdc"].sum()
            .apply(lambda x: round(x / initial, 3))
            .to_dict()
        ) if "category" in open_positions.columns else {}
        source_exposure = (
            open_positions.groupby("signal_source")["cost_usdc"].sum()
            .apply(lambda x: round(x / initial, 3))
            .to_dict()
        ) if "signal_source" in open_positions.columns else {}

    # P&L realizado total
    try:
        conn = sqlite3.connect(db_path); conn.execute("PRAGMA journal_mode=WAL")
        realized_pnl = conn.execute(
            "SELECT COALESCE(SUM(pnl_usdc), 0) FROM positions WHERE status IN ('closed','expired')"
        ).fetchone()[0] or 0.0
        conn.close()
    except Exception:
        realized_pnl = 0.0

    stop_triggered, stop_reason = check_drawdown_stop(portfolio, db_path)

    return {
        "initial_capital":    initial,
        "current_cash":       cash,
        "positions_value":    pos_value,
        "total_value":        cash + pos_value,
        "realized_pnl":       round(float(realized_pnl), 2),
        "unrealized_pnl":     round(
            (pos_value - float(open_positions["cost_usdc"].sum()))
            if not open_positions.empty and "current_value" in open_positions.columns
            else 0.0,
        2),
        "n_open_positions":   len(open_positions) if not open_positions.empty else 0,
        "category_exposure":  category_exposure,
        "source_exposure":    source_exposure,
        "stop_triggered":     stop_triggered,
        "stop_reason":        stop_reason,
        "available_slots":    MAX_OPEN_POSITIONS - (len(open_positions) if not open_positions.empty else 0),
    }
