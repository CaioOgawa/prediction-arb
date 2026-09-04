"""
gamma_collector.py
Coleta metadados de mercados via Gamma API (sem autenticação).
Suporta paginação automática, filtros e armazenamento incremental.
"""

import json
import time
import sqlite3
from pathlib import Path
from datetime import datetime, timezone

import requests
import pandas as pd
from loguru import logger

from db import init_db, get_connection

GAMMA_BASE = "https://gamma-api.polymarket.com"
RAW_DIR = Path("data/raw/markets")
RAW_DIR.mkdir(parents=True, exist_ok=True)

# Piso de sanidade: coletas abaixo disso indicam problema de paginação/API,
# não um universo genuinamente pequeno (bug de 2026-05→07: universo travado em 100).
MIN_EXPECTED_MARKETS = 300

# Retry por página: uma falha de rede transitória (DNS, timeout) não pode
# derrubar a coleta inteira e virar um snapshot vazio (P0-6).
PAGE_MAX_RETRIES  = 3
PAGE_BACKOFF_BASE = 2.0  # segundos: 2, 4, 8


class EmptySnapshotError(Exception):
    """Levantado quando a coleta não retornou nenhum mercado — ver save_snapshot()."""


def _as_list(raw) -> list:
    """
    A Gamma API retorna campos como clobTokenIds e outcomePrices como STRING JSON
    (ex: '["0.42", "0.58"]'), não como lista. Decodifica com segurança.
    """
    if raw is None:
        return []
    if isinstance(raw, str):
        try:
            decoded = json.loads(raw)
            return decoded if isinstance(decoded, list) else []
        except (json.JSONDecodeError, ValueError):
            return []
    try:
        return list(raw)
    except TypeError:
        return []

# Colunas base do payload da Gamma API
KEEP_COLS = [
    "id", "conditionId", "question", "category", "volume", "liquidity",
    "endDate", "active", "closed", "archived",
    # P2-34: sem isso, quem consome o snapshot (sim_backtest.py) não tem como
    # filtrar mercado que ainda nem existia em ts — look-ahead duro.
    "startDate", "createdAt",
    "outcomePrices", "outcomes", "clobTokenIds",
    "restricted", "resolutionSource", "description",
    # Campos de preço em tempo real (presentes apenas em mercados CLOB ativos)
    "lastTradePrice", "bestBid", "bestAsk", "spread",
    # Evento multi-outcome mutuamente exclusivo (soma das YES deve ser 1)
    # — usado pelo scanner de arb estrutural (signals/structural_arb.py)
    "negRisk",
    # Volumes por janela temporal (úteis como features)
    "volume24hr", "volume1wk", "volume1mo",
    # Variação de preço por período
    "oneDayPriceChange", "oneWeekPriceChange", "oneMonthPriceChange",
]


def _parse_markets(raw: list[dict]) -> pd.DataFrame:
    """
    Normaliza lista bruta de mercados para DataFrame padronizado.
    Prioriza lastTradePrice para yes_price; fallback em outcomePrices.
    """
    rows = []
    for m in raw:
        row = {col: m.get(col) for col in KEEP_COLS}

        # Preço YES: usa lastTradePrice quando disponível (mercados CLOB ativos)
        last_trade = m.get("lastTradePrice")
        if last_trade is not None:
            try:
                row["yes_price"] = float(last_trade)
            except (ValueError, TypeError):
                row["yes_price"] = None
        else:
            # Fallback: primeiro valor de outcomePrices (pode ser "0" em mercados resolvidos)
            prices = _as_list(m.get("outcomePrices"))
            if prices:
                try:
                    val = float(prices[0])
                    row["yes_price"] = val if val > 0 else None
                except (ValueError, TypeError):
                    row["yes_price"] = None
            else:
                row["yes_price"] = None

        # Categoria: campo direto ou herdado do evento pai
        category = m.get("category")
        if not category:
            events = m.get("events", [])
            if events and isinstance(events, list):
                category = events[0].get("category") or events[0].get("tag") or events[0].get("slug", "").split("-")[0]
        row["category"] = category or "uncategorized"

        # Slug do evento pai (útil para agrupar mercados relacionados)
        events = m.get("events", [])
        row["event_slug"]  = events[0].get("slug")  if events else None
        row["event_title"] = events[0].get("title") if events else None

        # Extrai token IDs (YES, NO) — clobTokenIds vem como string JSON da API
        token_ids = _as_list(m.get("clobTokenIds"))
        row["token_yes"] = token_ids[0] if len(token_ids) > 0 else None
        row["token_no"]  = token_ids[1] if len(token_ids) > 1 else None

        # Garante tipos numéricos
        row["volume"]     = float(m.get("volume", 0) or 0)
        row["liquidity"]  = float(m.get("liquidity", 0) or 0)
        row["volume24hr"] = float(m.get("volume24hr", 0) or 0)
        row["volume1wk"]  = float(m.get("volume1wk", 0) or 0)
        row["volume1mo"]  = float(m.get("volume1mo", 0) or 0)

        rows.append(row)

    return pd.DataFrame(rows)


def fetch_markets(
    active: bool = True,
    min_volume: float = 0,
    limit: int = 100,
    max_pages: int = 200,
) -> pd.DataFrame:
    """
    Busca todos os mercados via /markets/keyset com paginação por cursor.

    Por que keyset e não offset (mudança de 2026-07):
      - O endpoint /markets passou a capar `limit` em 100 e `offset` em ~2000,
        truncando o universo silenciosamente (bug: coleta travada em 100 mercados).
      - /markets/keyset pagina por `after_cursor` sem limite de profundidade e
        aceita `volume_num_min` server-side (filtra ANTES de fatiar a página).
      - Única condição de parada confiável: página vazia ou next_cursor ausente.
        NÃO usar `len(batch) < limit` — o /markets antigo retornava páginas
        parciais legítimas no meio do stream.

    Args:
        active: se True, retorna apenas mercados ainda abertos
        min_volume: volume mínimo em USDC (aplicado server-side E client-side)
        limit: registros por página (a API atende até 100)
        max_pages: trava de segurança contra loop infinito de cursor
    """
    all_markets: list[dict] = []
    cursor: str | None = None
    page = 0

    logger.info(f"Iniciando coleta Gamma API (keyset) | active={active} min_volume=${min_volume:,.0f}")

    while page < max_pages:
        params: dict = {
            "active": str(active).lower(),
            "closed": "false",   # exclui mercados já encerrados
            "limit": limit,
        }
        if min_volume > 0:
            params["volume_num_min"] = min_volume
        if cursor:
            params["after_cursor"] = cursor

        payload = None
        for attempt in range(1, PAGE_MAX_RETRIES + 1):
            try:
                resp = requests.get(f"{GAMMA_BASE}/markets/keyset", params=params, timeout=20)
                resp.raise_for_status()
                payload = resp.json()
                break
            except requests.RequestException as e:
                if attempt < PAGE_MAX_RETRIES:
                    backoff = PAGE_BACKOFF_BASE * (2 ** (attempt - 1))
                    logger.warning(
                        f"Erro na Gamma API (página {page+1}, tentativa {attempt}/{PAGE_MAX_RETRIES}): "
                        f"{e} — retry em {backoff:.0f}s"
                    )
                    time.sleep(backoff)
                else:
                    logger.error(
                        f"Erro na Gamma API (página {page+1}) após {PAGE_MAX_RETRIES} tentativas: {e}"
                    )

        if payload is None:
            break

        if not isinstance(payload, dict):
            logger.error(f"Resposta inesperada da Gamma API (página {page+1}): {str(payload)[:120]}")
            break

        batch = payload.get("markets", [])
        if not batch:
            break

        all_markets.extend(batch)
        logger.debug(f"  Página {page+1}: {len(batch)} mercados (total acumulado: {len(all_markets)})")

        cursor = payload.get("next_cursor")
        page += 1

        if not cursor:
            break

        # Respeita rate limit (~100 req/min)
        time.sleep(0.15)

    if page >= max_pages:
        logger.warning(f"Coleta atingiu max_pages={max_pages} — universo pode estar truncado")

    df = _parse_markets(all_markets)

    # Filtra por volume mínimo (redundante com volume_num_min — cinto e suspensório)
    if min_volume > 0 and not df.empty:
        df = df[df["volume"] >= min_volume].reset_index(drop=True)

    logger.info(f"Coleta concluída: {len(df)} mercados (filtro volume >= ${min_volume:,.0f})")

    # Sanidade: coleta anormalmente pequena = paginação/API quebrada, não mercado calmo.
    # Entre 2026-05 e 2026-07 o universo ficou travado em 100 sem nenhum alerta.
    if len(df) < MIN_EXPECTED_MARKETS and max_pages > 1:
        logger.warning(
            f"⚠️ Universo suspeito: {len(df)} mercados (< {MIN_EXPECTED_MARKETS} esperados). "
            "Verificar paginação/mudança na Gamma API."
        )

    return df


def save_snapshot(df: pd.DataFrame, tag: str = "all") -> Path:
    """
    Salva snapshot de mercados em Parquet com timestamp.
    Retorna o caminho do arquivo salvo.

    P0-6: um snapshot vazio (0 mercados — falha transitória de rede, DNS, etc.)
    NUNCA é gravado sob o nome `markets_{tag}_*` que todo loader do sistema
    (paper_trader, signal_generator, deribit_collector, ws_feed...) busca por
    mtime mais recente. Gravar vazio ali "envenena" a leitura de todo mundo
    pelos próximos ~30min (ou até o próximo ciclo bem-sucedido) — foi o que
    parou o pipeline inteiro em 2026-09 (100% dos ciclos do dia com 0 mercados,
    fail-open silencioso). O snapshot vazio ainda é salvo, mas com um prefixo
    que nenhum glob de loader casa, e a função levanta para o chamador tratar
    como falha de ciclo (ver EmptySnapshotError).
    """
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    if df.empty:
        path = RAW_DIR / f"markets_partial_{tag}_{ts}.parquet"
        df.to_parquet(path, index=False, compression="snappy")
        logger.error(
            f"Coleta vazia — NÃO publicada como markets_{tag}_* (poison do fail-open). "
            f"Salva para inspeção em {path}"
        )
        raise EmptySnapshotError(f"Gamma API devolveu 0 mercados — snapshot descartado ({path})")

    path = RAW_DIR / f"markets_{tag}_{ts}.parquet"
    df.to_parquet(path, index=False, compression="snappy")
    logger.info(f"Snapshot salvo: {path} ({len(df)} linhas)")
    return path


def upsert_to_db(df: pd.DataFrame) -> int:
    """
    Insere ou atualiza mercados no market_registry do SQLite.
    Retorna número de registros processados.
    """
    init_db()
    conn = get_connection()
    now = datetime.now(timezone.utc).isoformat()

    upserted = 0
    for _, row in df.iterrows():
        cid = row.get("conditionId")
        if not cid:
            continue
        conn.execute("""
            INSERT INTO market_registry
                (condition_id, question, category, token_yes, token_no,
                 volume, liquidity, yes_price, end_date, first_seen, last_seen)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(condition_id) DO UPDATE SET
                volume    = excluded.volume,
                liquidity = excluded.liquidity,
                yes_price = excluded.yes_price,
                last_seen = excluded.last_seen
        """, (
            cid,
            row.get("question"),
            row.get("category"),
            row.get("token_yes"),
            row.get("token_no"),
            row.get("volume", 0),
            row.get("liquidity", 0),
            row.get("yes_price"),
            row.get("endDate"),
            now,
            now,
        ))
        upserted += 1

    conn.commit()
    conn.close()
    return upserted


def run(min_volume: float = 5_000) -> pd.DataFrame:
    """
    Executa coleta completa: Gamma API → Parquet + SQLite.
    Ponto de entrada padrão para o pipeline de mercados.
    """
    df = fetch_markets(active=True, min_volume=min_volume)
    save_snapshot(df, tag="all")
    n = upsert_to_db(df)
    logger.info(f"DB atualizado: {n} mercados no market_registry")
    return df


if __name__ == "__main__":
    df = run(min_volume=5_000)
    print(f"\nTop 10 mercados por volume:")
    cols = ["question", "category", "volume", "yes_price", "liquidity"]
    print(df[cols].sort_values("volume", ascending=False).head(10).to_string(index=False))
