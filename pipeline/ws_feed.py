"""
ws_feed.py
Feed de preços em tempo real via WebSocket CLOB Polymarket — versão otimizada.

Arquitetura de baixa latência:
  WebSocket recv()
      │
      ▼
  [ check_exit_trigger ]  ← caminho crítico: ~0.01ms, tudo in-memory
      │ (se trigger)
      ▼
  [ exit_writer coroutine ]  ← SQLite com conexão persistente, ~1ms
      │
  [ tick_queue asyncio ]  ← fila não-bloqueante para price_history
      │
  [ history_writer coroutine ]  ← batch insert a cada 100ms, nunca bloqueia exit

Resultado: early exit detectado no mesmo tick que o preço muda.
Latência total: recv() + ~0.01ms (vs. recv() + 0.55ms no design anterior).

Execução:
    uv run python -m pipeline.ws_feed
    uv run python -m pipeline.ws_feed --dry-run
"""

import asyncio
import json
import signal
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple

import click
import pandas as pd
import websockets
from loguru import logger

ROOT    = Path(__file__).parent.parent
DB_PATH = ROOT / "data/db/paper_trading.db"
WS_URI  = "wss://ws-subscriptions-clob.polymarket.com/ws/market"

TOP_MARKETS_TO_WATCH = 30
REFRESH_INTERVAL_S   = 300
MAX_TOKENS_PER_SUB   = 50
BACKOFF_MIN, BACKOFF_MAX = 2, 60
HISTORY_BATCH_MS     = 100   # flush price_history a cada 100ms

# Importa parâmetros de early exit do risk_manager para manter consistência.
# Fallback para valores conservadores se o import falhar.
try:
    from risk.risk_manager import EARLY_EXIT as _EARLY_EXIT, POSITION_STOP_LOSS
    # Usa parâmetros do trade_type "value" como baseline (posições mais longas)
    _ee_value    = _EARLY_EXIT.get("value",    {})
    _ee_momentum = _EARLY_EXIT.get("momentum", {})
    PROFIT_TARGET_MULT_VALUE    = float(_ee_value.get("profit_target_mult", 2.5))
    PROFIT_TARGET_MULT_MOMENTUM = float(_ee_momentum.get("profit_target_mult", 1.5))
    EDGE_FLIP_DELTA_VALUE    = float(_ee_value.get("edge_flip_delta", 0.25))
    EDGE_FLIP_DELTA_MOMENTUM = float(_ee_momentum.get("edge_flip_delta", 0.12))
    MIN_HOLD_HOURS_VALUE     = float(_ee_value.get("min_hold_hours", 4.0))
    MIN_HOLD_HOURS_MOMENTUM  = float(_ee_momentum.get("min_hold_hours", 1.0))
except Exception:
    PROFIT_TARGET_MULT_VALUE    = 2.5
    PROFIT_TARGET_MULT_MOMENTUM = 1.5
    EDGE_FLIP_DELTA_VALUE    = 0.25
    EDGE_FLIP_DELTA_MOMENTUM = 0.12
    MIN_HOLD_HOURS_VALUE     = 4.0
    MIN_HOLD_HOURS_MOMENTUM  = 1.0
    POSITION_STOP_LOSS       = 0.50

# Book com spread acima disso é vazio/ilíquido (ex: só ordens-poeira 0.001/0.999) —
# não é preço real e NÃO pode disparar exit. Incidente 2026-05: books assim
# executaram 51 posições a 0.001 (agravado por bids[0]/asks[0] invertidos).
MAX_EXIT_SPREAD = 0.10

PRICE_HISTORY_SCHEMA = """
CREATE TABLE IF NOT EXISTS price_history (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    ts           TEXT    NOT NULL,
    asset_id     TEXT    NOT NULL,
    condition_id TEXT,
    best_bid     REAL,
    best_ask     REAL,
    mid          REAL,
    spread       REAL,
    side         TEXT,
    size         REAL
);
CREATE INDEX IF NOT EXISTS idx_ph_asset_id ON price_history(asset_id);
CREATE INDEX IF NOT EXISTS idx_ph_ts       ON price_history(ts);
"""


class AssetInfo(NamedTuple):
    asset_id:     str
    condition_id: str
    question:     str
    direction:    str
    entry_price:  float
    shares:       float
    cost_usdc:    float
    position_id:  int
    trade_type:   str = "value"  # "value" | "momentum" — define thresholds de early exit
    opened_ts:    float = 0.0    # epoch UTC da abertura — gate de min_hold nos exits


class Tick(NamedTuple):
    ts:           str
    asset_id:     str
    condition_id: str
    best_bid:     float | None
    best_ask:     float | None
    side:         str | None
    size:         float | None


# ── Inicialização ───────────────────────────────────────────

def init_db() -> None:
    conn = sqlite3.connect(DB_PATH)
    conn.executescript(PRICE_HISTORY_SCHEMA)
    conn.commit()
    conn.close()


def build_asset_map() -> dict[str, AssetInfo]:
    """Constrói asset_id → AssetInfo para posições abertas e top mercados."""
    try:
        raw_dir = ROOT / "data/raw/markets"
        files = sorted(
            list(raw_dir.glob("markets_all_*.parquet")) +
            list(raw_dir.glob("markets_incremental_*.parquet")),
            key=lambda p: p.stat().st_mtime,  # ordena por mtime, não por nome (P0-2)
            reverse=True,
        )
        if not files:
            logger.warning("Sem arquivos de mercado. Execute fetch_markets.py.")
            return {}
        markets = pd.read_parquet(files[0])
    except Exception as e:
        logger.error(f"Erro ao carregar mercados: {e}")
        return {}

    def parse_tokens(raw) -> list[str]:
        if isinstance(raw, str):
            try:
                return json.loads(raw)
            except Exception:
                return []
        return list(raw) if raw else []

    cid_to_tokens: dict[str, list[str]] = {}
    cid_to_question: dict[str, str] = {}
    for _, row in markets.iterrows():
        cid    = str(row.get("conditionId", ""))
        tokens = parse_tokens(row.get("clobTokenIds", "[]"))
        if cid and tokens:
            cid_to_tokens[cid]    = tokens
            cid_to_question[cid]  = str(row.get("question", ""))[:80]

    asset_map: dict[str, AssetInfo] = {}

    # Posições abertas — caminho crítico, monitorar com atenção
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM positions WHERE status='open'").fetchall()
    conn.close()
    def _opened_ts(raw) -> float:
        """Converte opened_at (texto UTC do SQLite) para epoch. 0.0 se inválido."""
        try:
            dt = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.timestamp()
        except Exception:
            return 0.0

    for pos in rows:
        cid    = str(pos["condition_id"])
        tokens = cid_to_tokens.get(cid, [])
        if not tokens:
            continue
        direction = str(pos["direction"])
        asset_id  = tokens[0] if direction == "BUY_YES" else (tokens[1] if len(tokens) > 1 else tokens[0])
        asset_map[asset_id] = AssetInfo(
            asset_id=asset_id,
            condition_id=cid,
            question=cid_to_question.get(cid, "")[:60],
            direction=direction,
            entry_price=float(pos["entry_price"]),
            shares=float(pos["shares"]),
            cost_usdc=float(pos["cost_usdc"]),
            position_id=int(pos["id"]),
            trade_type=str(pos["trade_type"]) if "trade_type" in pos.keys() else "value",
            opened_ts=_opened_ts(pos["opened_at"] if "opened_at" in pos.keys() else None),
        )

    # Top mercados ativos (só monitorar preços — sem position_id)
    try:
        active = markets[
            (markets["active"] == True) &  # noqa: E712
            (markets.get("yes_price", pd.Series(dtype=float)).between(0.05, 0.95))
        ].nlargest(TOP_MARKETS_TO_WATCH, "liquidity")
        for _, row in active.iterrows():
            cid    = str(row.get("conditionId", ""))
            tokens = cid_to_tokens.get(cid, [])
            if tokens and tokens[0] not in asset_map:
                asset_map[tokens[0]] = AssetInfo(
                    asset_id=tokens[0], condition_id=cid,
                    question=cid_to_question.get(cid, "")[:60],
                    direction="WATCH", entry_price=0.0,
                    shares=0.0, cost_usdc=0.0, position_id=0,
                )
    except Exception:
        pass

    n_pos  = sum(1 for v in asset_map.values() if v.position_id > 0)
    n_watch = sum(1 for v in asset_map.values() if v.position_id == 0)
    logger.info(f"Asset map: {n_pos} posições + {n_watch} mercados monitorados")
    return asset_map


# ── Caminho crítico: exit check (in-memory, ~0.01ms) ────────

def best_levels(bids: list[dict], asks: list[dict]) -> tuple[float | None, float | None]:
    """
    Extrai (best_bid, best_ask) de um book snapshot da CLOB.

    A CLOB NÃO garante que o índice 0 é o melhor nível (bids vêm ascendentes:
    bids[0] é o PIOR bid, ex: ordem-poeira a 0.001). Melhor bid = max, melhor
    ask = min. Usar bids[0]/asks[0] executou todas as posições a 0.001 (2026-05).
    """
    bb = max((float(b["price"]) for b in bids), default=None)
    ba = min((float(a["price"]) for a in asks), default=None)
    return bb, ba


def evaluate_exit(info: AssetInfo, best_bid: float, best_ask: float) -> tuple[float, str] | None:
    """
    Avalia gatilhos de saída — pura lógica in-memory, sem I/O.
    Usa parâmetros do risk_manager por trade_type (value vs momentum).
    Retorna (exit_price, trigger_str) ou None se nenhum gatilho ativado.

    Proteções (pós-incidente 2026-05):
      - Book vazio/ilíquido (spread > MAX_EXIT_SPREAD) NUNCA dispara exit —
        preço de ordem-poeira não é preço de mercado.
      - exit_price = best_bid (vender a mercado executa no bid; o antigo
        `bid - spread/2` descontava o spread duas vezes e ia a ~0).
      - stop_loss e edge_flip respeitam min_hold_hours do trade_type;
        só o profit_target dispara a qualquer momento.
    """
    if info.position_id == 0 or info.direction == "WATCH":
        return None

    # Perna de basket estrutural (arb): hold até resolução, SEM exceção.
    # Fechar uma perna isolada quebra a garantia do basket e vira posição
    # direcional nua — nenhum gatilho (nem profit_target) pode disparar.
    if info.trade_type == "arb":
        return None

    spread = best_ask - best_bid
    if spread <= 0 or spread > MAX_EXIT_SPREAD:
        return None  # book cruzado ou vazio — sem preço confiável para decidir

    is_momentum   = info.trade_type == "momentum"
    profit_mult   = PROFIT_TARGET_MULT_MOMENTUM if is_momentum else PROFIT_TARGET_MULT_VALUE
    edge_flip_d   = EDGE_FLIP_DELTA_MOMENTUM    if is_momentum else EDGE_FLIP_DELTA_VALUE
    min_hold_h    = MIN_HOLD_HOURS_MOMENTUM     if is_momentum else MIN_HOLD_HOURS_VALUE

    exit_price    = max(best_bid, 0.001)
    current_value = exit_price * info.shares
    cost          = info.cost_usdc

    yes_price_implied = (1.0 - best_bid) if info.direction == "BUY_NO" else best_bid

    if current_value >= cost * profit_mult:
        return exit_price, f"profit_target ({current_value/cost:.1f}× custo, tipo={info.trade_type})"

    # Gate de min_hold: sem opened_ts confiável, não arrisca exit por perda
    hold_ok = info.opened_ts > 0 and (time.time() - info.opened_ts) / 3600 >= min_hold_h
    if not hold_ok:
        return None

    if current_value <= cost * (1.0 - POSITION_STOP_LOSS):
        return exit_price, f"stop_loss (${current_value:.2f} = {current_value/cost:.0%} do custo, tipo={info.trade_type})"

    # Edge flip: entry_price está incorporado — yes_price saiu da faixa favorável
    # BUY_YES: vendemos se yes caiu abaixo de entry_price - edge_flip_delta
    # BUY_NO:  vendemos se yes subiu acima de (1-entry_price) + edge_flip_delta
    if info.direction == "BUY_YES":
        flip_threshold = info.entry_price - edge_flip_d
        if yes_price_implied < flip_threshold:
            return exit_price, f"edge_flip_yes (YES={yes_price_implied:.2f} < {flip_threshold:.2f})"
    else:
        flip_threshold = (1.0 - info.entry_price) + edge_flip_d
        if yes_price_implied > flip_threshold:
            return exit_price, f"edge_flip_no (YES={yes_price_implied:.2f} > {flip_threshold:.2f})"

    return None


# ── Writers assíncronos (fora do caminho crítico) ────────────

async def exit_writer(
    exit_queue: asyncio.Queue,
    dry_run: bool,
    stop_event: asyncio.Event,
) -> None:
    """
    Drena exit_queue e aplica saídas antecipadas no SQLite.
    Usa conexão persistente com WAL — ~1ms por exit.
    Roda em paralelo com o WebSocket loop, nunca bloqueia recv().
    """
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")

    try:
        while not stop_event.is_set() or not exit_queue.empty():
            try:
                info, exit_price, trigger = await asyncio.wait_for(exit_queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue

            pnl_usdc = round((exit_price - info.entry_price) * info.shares, 2)
            cost     = info.cost_usdc
            now_str  = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

            if dry_run:
                logger.info(f"[DRY-RUN] EXIT #{info.position_id} {info.question[:40]} P&L=${pnl_usdc:+.2f} | {trigger}")
                exit_queue.task_done()
                continue

            try:
                # BEGIN IMMEDIATE: trava de escrita desde o início — o run_cycle
                # e o run_execution também fecham posições neste banco.
                conn.execute("BEGIN IMMEDIATE")
                cur = conn.execute("""
                    UPDATE positions
                    SET closed_at=?, exit_price=?, pnl_usdc=?, status='closed'
                    WHERE id=? AND status='open'
                """, (now_str, round(exit_price, 4), pnl_usdc, info.position_id))
                if cur.rowcount == 0:
                    # Outro processo já fechou esta posição — NÃO creditar de novo
                    # (task_done fica por conta do finally abaixo)
                    conn.rollback()
                    logger.info(f"Exit #{info.position_id} já processado por outro processo — pulando")
                    continue
                conn.execute("""
                    UPDATE portfolio SET current_cash = current_cash + ?
                    WHERE id = (SELECT MAX(id) FROM portfolio)
                """, (cost + pnl_usdc,))
                conn.execute("""
                    INSERT INTO trades_log (action, condition_id, direction, price, shares, usdc_amount, note)
                    VALUES ('EARLY_EXIT', ?, ?, ?, ?, ?, ?)
                """, (info.condition_id, info.direction, round(exit_price, 4),
                      info.shares, cost + pnl_usdc, trigger))
                conn.commit()
                logger.success(
                    f"EARLY EXIT #{info.position_id} {info.question[:40]}"
                    f" | P&L=${pnl_usdc:+.2f} | {trigger}"
                )
            except Exception as e:
                logger.error(f"Erro ao salvar exit #{info.position_id}: {e}")
                try:
                    conn.rollback()  # não deixar transação aberta para a próxima iteração
                except Exception:
                    pass
            finally:
                exit_queue.task_done()
    finally:
        conn.close()


async def history_writer(
    tick_queue: asyncio.Queue,
    stop_event: asyncio.Event,
) -> None:
    """
    Drena tick_queue em batches e insere no price_history.
    Flush a cada HISTORY_BATCH_MS ms — nunca bloqueia o loop de exit.
    """
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")

    INSERT = (
        "INSERT INTO price_history "
        "(ts, asset_id, condition_id, best_bid, best_ask, mid, spread, side, size) "
        "VALUES (?,?,?,?,?,?,?,?,?)"
    )

    try:
        while not stop_event.is_set() or not tick_queue.empty():
            batch = []
            deadline = asyncio.get_event_loop().time() + HISTORY_BATCH_MS / 1000

            # Drena a fila até o prazo
            while asyncio.get_event_loop().time() < deadline:
                try:
                    tick = tick_queue.get_nowait()
                    mid    = ((tick.best_bid or 0) + (tick.best_ask or 0)) / 2 if (tick.best_bid and tick.best_ask) else None
                    spread = round((tick.best_ask or 0) - (tick.best_bid or 0), 5) if (tick.best_bid and tick.best_ask) else None
                    batch.append((tick.ts, tick.asset_id, tick.condition_id,
                                  tick.best_bid, tick.best_ask, mid, spread,
                                  tick.side, tick.size))
                    tick_queue.task_done()
                except asyncio.QueueEmpty:
                    break

            if batch:
                try:
                    conn.executemany(INSERT, batch)
                    conn.commit()
                except Exception as e:
                    logger.debug(f"Erro ao gravar batch de {len(batch)} ticks: {e}")

            remaining = deadline - asyncio.get_event_loop().time()
            if remaining > 0:
                await asyncio.sleep(remaining)
    finally:
        conn.close()


# ── WebSocket loop principal ────────────────────────────────

async def ws_loop(
    dry_run: bool,
    stop_event: asyncio.Event,
    exit_queue: asyncio.Queue,
    tick_queue: asyncio.Queue,
) -> None:
    asset_map:    dict[str, AssetInfo] = {}
    exited_ids:   set[int]             = set()   # evita duplo-exit na mesma sessão
    last_refresh: float                = 0.0
    backoff:      float                = BACKOFF_MIN

    while not stop_event.is_set():
        try:
            async with websockets.connect(
                WS_URI,
                open_timeout=15,
                ping_interval=20,
                ping_timeout=10,
            ) as ws:
                logger.info(f"Conectado: {WS_URI}")
                backoff = BACKOFF_MIN

                while not stop_event.is_set():
                    now = asyncio.get_event_loop().time()

                    # Refresh periódico: novas posições abertas, novos mercados
                    if now - last_refresh > REFRESH_INTERVAL_S:
                        asset_map   = build_asset_map()
                        exited_ids &= {v.position_id for v in asset_map.values()}  # limpa IDs antigos
                        tokens       = list(asset_map.keys())[:MAX_TOKENS_PER_SUB]
                        if tokens:
                            await ws.send(json.dumps({"assets_ids": tokens, "type": "market"}))
                            logger.info(f"Subscrito a {len(tokens)} tokens")
                        last_refresh = now

                    # Aguarda próximo frame — retorna imediatamente se há mensagem
                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=1.0)
                    except asyncio.TimeoutError:
                        continue

                    try:
                        msgs = json.loads(raw)
                        if not isinstance(msgs, list):
                            msgs = [msgs]
                    except json.JSONDecodeError:
                        continue

                    ts_now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f")

                    for msg in msgs:
                        # ── Book snapshot ──────────────────────────────
                        if "bids" in msg and "asks" in msg:
                            aid  = msg.get("asset_id", "")
                            bb, ba = best_levels(msg.get("bids", []), msg.get("asks", []))
                            info = asset_map.get(aid)
                            if info:
                                # Persiste histórico só de POSIÇÕES — ticks de mercados
                                # WATCH geravam ~1 GB/dia sem consumidor (2026-07)
                                if info.position_id > 0:
                                    tick_queue.put_nowait(Tick(ts_now, aid, info.condition_id, bb, ba, None, None))
                                if bb and ba and info.position_id not in exited_ids:
                                    result = evaluate_exit(info, bb, ba)
                                    if result:
                                        exited_ids.add(info.position_id)
                                        await exit_queue.put((info, result[0], result[1]))

                        # ── Price change ticks ─────────────────────────
                        for change in msg.get("price_changes", []):
                            aid  = change.get("asset_id", "")
                            bb_s = change.get("best_bid")
                            ba_s = change.get("best_ask")
                            bb   = float(bb_s) if bb_s else None
                            ba   = float(ba_s) if ba_s else None
                            side = change.get("side")
                            sz   = float(change["size"]) if change.get("size") else None
                            info = asset_map.get(aid)
                            if info:
                                # ★ Exit check ANTES de enfileirar tick (caminho crítico primeiro)
                                if bb and ba and info.position_id not in exited_ids:
                                    result = evaluate_exit(info, bb, ba)
                                    if result:
                                        exited_ids.add(info.position_id)
                                        await exit_queue.put((info, result[0], result[1]))
                                # Histórico: só de posições, não bloqueia, drop se fila cheia
                                if info.position_id > 0:
                                    try:
                                        tick_queue.put_nowait(Tick(ts_now, aid, info.condition_id, bb, ba, side, sz))
                                    except asyncio.QueueFull:
                                        pass  # drop tick de histórico, exit já foi processado

        except (websockets.ConnectionClosed, OSError) as e:
            if stop_event.is_set():
                break
            logger.warning(f"Desconectado: {e}. Reconectando em {backoff:.0f}s...")
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, BACKOFF_MAX)
        except Exception as e:
            if stop_event.is_set():
                break
            logger.error(f"Erro inesperado: {e}. Reconectando em {backoff:.0f}s...")
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, BACKOFF_MAX)

    logger.info("ws_loop encerrado.")


# ── Entry point ─────────────────────────────────────────────

@click.command()
@click.option("--dry-run", is_flag=True, default=False,
              help="Detecta exits mas não os aplica no banco.")
def main(dry_run: bool) -> None:
    """Feed de preços em tempo real com early exit de baixa latência."""
    logger.remove()
    logger.add(sys.stderr, level="INFO",
               format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}")
    logger.add(ROOT / "logs/ws_feed.log", level="DEBUG",
               rotation="50 MB", retention=5, compression="gz")

    init_db()
    logger.info(f"ws_feed v2 iniciado | dry_run={dry_run}")

    stop_event  = asyncio.Event()
    exit_queue  = asyncio.Queue()
    tick_queue  = asyncio.Queue(maxsize=10_000)   # drop ticks se acumular (histórico pode perder, exit não)

    def _shutdown(sig, frame):
        logger.info(f"Sinal {sig} — encerrando...")
        stop_event.set()

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT,  _shutdown)

    async def run_all():
        await asyncio.gather(
            ws_loop(dry_run, stop_event, exit_queue, tick_queue),
            exit_writer(exit_queue, dry_run, stop_event),
            history_writer(tick_queue, stop_event),
        )

    asyncio.run(run_all())


if __name__ == "__main__":
    main()
