"""
structural_arb.py
Scanner de statistical arbitrage ESTRUTURAL intra-Polymarket (Fase 2, ADR-008).

Diferente dos modos odds/deribit (valor relativo contra referência externa),
aqui o lucro é garantido pela ESTRUTURA lógica dos mercados — sem modelo:

  1. NEGRISK BASKET — eventos multi-outcome mutuamente exclusivos (negRisk=true).
     Exatamente um outcome resolve YES, logo:
       Σ ask(YES_i) < 1  → comprar 1 YES de cada outcome: custo < 1, payout = 1
       Σ bid(YES_i) > 1  → comprar 1 NO de cada outcome:
                           custo = Σ(1 − bid_i) = n − Σbid, payout = n − 1
                           lucro = Σbid − 1

  2. MONOTONICIDADE — mercados crypto aninhados do mesmo ativo:
       touch:    tocar $80k até dezembro ⊇ tocar $90k até novembro
       europeia: acima de $80k em D ⊇ acima de $90k em D (mesma data)
     Se o dominado (j) custa MAIS que o dominante (i):
       comprar YES_i (ask_i) + NO_j (1 − bid_j) → payout mínimo 1
       lucro garantido = bid_j − ask_i

  3. BOOK CRUZADO — bestAsk < bestBid no mesmo mercado (glitch de dados ou
     arb intramercado). Reportado para investigação.

Saída: tabela no terminal + CSV signals_structural_*.csv (uma linha por perna,
agrupadas por arb_group) + alerta Telegram para oportunidades GARANTIDAS.
Baskets garantidos são executados ATOMICAMENTE pelo paper trader via
open_basket() (todas as pernas na mesma transação); oportunidades condicionais
(YES-basket sem guarda-chuva, book cruzado) continuam report-only.

Uso:
    uv run python signals/structural_arb.py
    uv run python signals/structural_arb.py --margin 0.01 --max-events 60
"""

import re
import sys
import time
from datetime import datetime
from pathlib import Path

import click
import pandas as pd
import requests
from loguru import logger
from rich.console import Console
from rich.table import Table
from rich import box

sys.path.insert(0, str(Path(__file__).parent.parent / "pipeline"))
sys.path.insert(0, str(Path(__file__).parent.parent / "risk"))
sys.path.insert(0, str(Path(__file__).parent.parent))
from deribit_collector import parse_crypto_market, get_spot_price
from risk_manager import ARB_MIN_PROFIT

GAMMA_BASE   = "https://gamma-api.polymarket.com"
RAW_MKT_DIR  = Path("data/raw/markets")
REPORTS_DIR  = Path("outputs/reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

console = Console()

# Lucro mínimo garantido (por $1 de payout) para reportar uma oportunidade.
# Cobre slippage de execução + risco de resolução ambígua. Abaixo disso é ruído.
# Fonte única no risk_manager — a execução re-verifica com o mesmo threshold.
MARGIN_MIN = ARB_MIN_PROFIT

# Máximo de eventos negRisk consultados na API por execução (1 request cada).
MAX_EVENTS_PER_RUN = 40


# ──────────────────────────────────────────────────────────
# Universo
# ──────────────────────────────────────────────────────────

def load_universe() -> pd.DataFrame:
    """Snapshot mais recente de mercados (mesmo padrão dos outros geradores)."""
    candidates = sorted(
        list(RAW_MKT_DIR.glob("markets_all_*.parquet")) +
        list(RAW_MKT_DIR.glob("markets_incremental_*.parquet")),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            f"Nenhum parquet de mercados em {RAW_MKT_DIR}. "
            "Execute: uv run python pipeline/fetch_markets.py"
        )
    df = pd.read_parquet(candidates[0])
    logger.info(f"Universo: {len(df):,} mercados ({candidates[0].name})")
    return df


def _f(x) -> float | None:
    """float ou None (campos bid/ask podem vir vazios/str)."""
    try:
        v = float(x)
        return v if v > 0 else None
    except (TypeError, ValueError):
        return None


# ──────────────────────────────────────────────────────────
# Detector 1 — NegRisk basket
# ──────────────────────────────────────────────────────────

def _fetch_event(slug: str) -> dict | None:
    """Evento completo (todos os mercados, preços frescos) via Gamma /events."""
    try:
        resp = requests.get(f"{GAMMA_BASE}/events", params={"slug": slug}, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        return data[0] if isinstance(data, list) and data else None
    except requests.RequestException as e:
        logger.debug(f"  /events?slug={slug}: {e}")
        return None


# Perguntas "guarda-chuva" que fecham o conjunto de outcomes ("someone else",
# "another candidate"...). Sem uma delas, o evento pode não ser EXAUSTIVO e o
# YES-basket deixa de ser garantido (ex: Nobel — se nenhum listado ganhar,
# todas as YES viram 0 e o basket perde tudo).
_CATCHALL_RE = re.compile(
    r"\b(another|other|someone else|anyone else|any other|none of|no one|nobody)\b",
    re.IGNORECASE,
)


def evaluate_negrisk_event(markets: list[dict], margin: float = MARGIN_MIN) -> dict | None:
    """
    Avalia um evento negRisk (lista de mercados ativos com bid/ask).
    Retorna a oportunidade (dict) ou None. Puro — testável sem API.

    Garantias por tipo:
      NO-basket  — só exige exclusão mútua (no máx. 1 YES): payout ≥ n−1. GARANTIDO.
      YES-basket — exige EXAUSTIVIDADE (pelo menos 1 YES). Só é garantido se o
                   evento tem outcome guarda-chuva; senão sai como *_conditional.
    """
    legs = []
    for m in markets:
        bid, ask = _f(m.get("bestBid")), _f(m.get("bestAsk"))
        if bid is None or ask is None:
            return None  # sem book completo, a soma não é confiável
        legs.append({
            "condition_id": m.get("conditionId", ""),
            "question":     str(m.get("question", ""))[:80],
            "bid": bid, "ask": ask,
            "liquidity": _f(m.get("liquidity")) or 0.0,
        })

    if len(legs) < 2:
        return None

    sum_ask = sum(l["ask"] for l in legs)
    sum_bid = sum(l["bid"] for l in legs)
    n = len(legs)
    exhaustive = any(_CATCHALL_RE.search(l["question"]) for l in legs)

    # YES basket: custo Σask, payout 1 SE exaustivo
    yes_profit = 1.0 - sum_ask
    # NO basket: custo n − Σbid, payout ≥ n − 1 → lucro ≥ Σbid − 1
    no_profit = sum_bid - 1.0

    if yes_profit >= margin:
        return {
            "kind": "negrisk_yes" if exhaustive else "negrisk_yes_conditional",
            "guaranteed": exhaustive,
            "n_legs": n,
            "basket_cost": round(sum_ask, 4), "payout_min": 1.0,
            "profit": round(yes_profit, 4),
            "edge": round(yes_profit / 1.0, 4),          # lucro por $1 de payout
            "legs": [{**l, "direction": "BUY_YES", "leg_price": l["ask"]} for l in legs],
        }
    if no_profit >= margin * (n - 1):
        return {
            "kind": "negrisk_no",
            "guaranteed": True,
            "n_legs": n,
            "basket_cost": round(n - sum_bid, 4), "payout_min": float(n - 1),
            "profit": round(no_profit, 4),
            "edge": round(no_profit / (n - 1), 4),
            "legs": [{**l, "direction": "BUY_NO", "leg_price": round(1 - l["bid"], 4)} for l in legs],
        }
    return None


def scan_negrisk(
    df: pd.DataFrame,
    margin: float = MARGIN_MIN,
    max_events: int = MAX_EVENTS_PER_RUN,
) -> list[dict]:
    """
    Varre eventos negRisk do universo, consultando o evento COMPLETO na API
    (o parquet é filtrado por volume — a soma exige todos os outcomes).
    """
    if "negRisk" not in df.columns or "event_slug" not in df.columns:
        logger.warning("Universo sem colunas negRisk/event_slug — re-rode fetch_markets.")
        return []

    neg = df[(df["negRisk"] == True) & df["event_slug"].notna()]  # noqa: E712
    if neg.empty:
        return []

    # Prioriza eventos por liquidez agregada (mais executáveis primeiro)
    slugs = (
        neg.groupby("event_slug")["liquidity"].sum()
        .sort_values(ascending=False)
        .head(max_events)
        .index.tolist()
    )
    logger.info(f"NegRisk: avaliando {len(slugs)} eventos (de {neg['event_slug'].nunique()} no universo)")

    found = []
    for slug in slugs:
        ev = _fetch_event(slug)
        if ev is None or not ev.get("negRisk"):
            continue
        active = [
            m for m in ev.get("markets", [])
            if m.get("active") and not m.get("closed") and not m.get("archived")
        ]
        opp = evaluate_negrisk_event(active, margin=margin)
        if opp:
            opp["arb_group"]  = f"neg:{slug}"
            opp["event_slug"] = slug
            found.append(opp)
            logger.info(f"  ARB {opp['kind']}: {slug} → lucro ${opp['profit']:.3f} ({opp['edge']:.1%})")
        time.sleep(0.15)

    return found


# ──────────────────────────────────────────────────────────
# Detector 2 — Monotonicidade (crypto aninhados)
# ──────────────────────────────────────────────────────────

def scan_monotonicity(df: pd.DataFrame, margin: float = MARGIN_MIN) -> list[dict]:
    """
    Detecta pares dominante/dominado com preços invertidos.

    Dominância (P_i ≥ P_j garantido por lógica, não por modelo):
      touch:    level_i ≤ level_j  e  expiry_i ≥ expiry_j
      europeia: level_i ≤ level_j  e  expiry_i == expiry_j
    onde level = strike para 'above' e −strike para 'below' (tocar barreira mais
    distante do spot implica ter tocado a mais próxima).

    Arb quando bid_j − ask_i ≥ margin: comprar YES_i (ask_i) + NO_j (1 − bid_j),
    payout mínimo $1 em qualquer cenário.

    Exclusões (fontes de falso arb):
      - Mercados de JANELA ("reach $72k July 6-12"): a barreira só vale dentro
        da janela, não em [agora, expiração] — quebra a relação de dominância.
      - Expiração vem do endDate da API (autoritativo); o parser de texto pode
        errar o ano em perguntas sem ano explícito.
    """
    _WINDOW_RE = re.compile(r"[A-Za-z]{3,9}\.?\s+\d{1,2}\s*[-–]\s*\d{1,2}\b")
    now = pd.Timestamp.now(tz="UTC")

    # P0-4 correlato: mercados de touch precisam do spot ATUAL para saber se a
    # barreira é upward ou downward (keyword sozinha erra "hit $50k" com spot
    # em $100k, que é um dip). Cache por asset — no máximo 2 chamadas (BTC/ETH).
    spot_cache: dict[str, float | None] = {}

    def _spot_for(asset: str) -> float | None:
        if asset not in spot_cache:
            spot_cache[asset] = get_spot_price(asset)
        return spot_cache[asset]

    recs = []
    for _, row in df.iterrows():
        q = str(row.get("question", ""))
        parsed = parse_crypto_market(q)
        if parsed is None or parsed.get("is_between"):
            continue
        if parsed.get("is_touch"):
            spot = _spot_for(parsed["asset"])
            if spot is not None:
                reparsed = parse_crypto_market(q, spot=spot)
                if reparsed is not None:
                    parsed = reparsed
        if _WINDOW_RE.search(q):
            continue  # janela ≠ barreira desde já — dominância não vale
        bid, ask = _f(row.get("bestBid")), _f(row.get("bestAsk"))
        if bid is None or ask is None:
            continue
        expiry = pd.to_datetime(row.get("endDate"), errors="coerce", utc=True)
        if pd.isna(expiry) or expiry <= now:
            continue
        direction = parsed["direction"]
        recs.append({
            "condition_id": str(row.get("conditionId", "")),
            "question":  q[:80],
            "asset":     parsed["asset"],
            "is_touch":  bool(parsed["is_touch"]),
            "direction": direction,
            "level":     parsed["strike"] if direction == "above" else -parsed["strike"],
            "expiry":    expiry,
            "bid": bid, "ask": ask,
            "liquidity": _f(row.get("liquidity")) or 0.0,
        })

    found = []
    for i in range(len(recs)):
        for j in range(len(recs)):
            if i == j:
                continue
            a, b = recs[i], recs[j]  # a = dominante (P_a ≥ P_b), b = dominado
            if a["asset"] != b["asset"] or a["is_touch"] != b["is_touch"]:
                continue
            if a["direction"] != b["direction"]:
                continue
            if a["is_touch"]:
                dominates = a["level"] <= b["level"] and a["expiry"] >= b["expiry"]
            else:
                dominates = (a["level"] <= b["level"]
                             and a["expiry"].date() == b["expiry"].date())
            # Exige dominância estrita em pelo menos uma dimensão
            if not dominates or (a["level"] == b["level"] and a["expiry"] == b["expiry"]):
                continue

            profit = b["bid"] - a["ask"]  # comprar YES_a + NO_b, payout ≥ 1
            if profit < margin:
                continue

            found.append({
                "kind": "monotonicity", "guaranteed": True, "n_legs": 2,
                "arb_group": f"mono:{a['condition_id'][:10]}>{b['condition_id'][:10]}",
                "event_slug": "",
                "basket_cost": round(a["ask"] + (1 - b["bid"]), 4),
                "payout_min": 1.0,
                "profit": round(profit, 4),
                "edge": round(profit, 4),
                "legs": [
                    {**a, "direction": "BUY_YES", "leg_price": a["ask"]},
                    {**b, "direction": "BUY_NO",  "leg_price": round(1 - b["bid"], 4)},
                ],
            })

    if found:
        logger.info(f"Monotonicidade: {len(found)} violações ≥ {margin:.1%}")
    return found


# ──────────────────────────────────────────────────────────
# Detector 3 — Book cruzado
# ──────────────────────────────────────────────────────────

def scan_crossed_books(df: pd.DataFrame, margin: float = 0.0) -> list[dict]:
    """bestAsk < bestBid — glitch de dados ou arb intramercado. Só reporta."""
    found = []
    for _, row in df.iterrows():
        bid, ask = _f(row.get("bestBid")), _f(row.get("bestAsk"))
        if bid is None or ask is None or ask >= bid - margin:
            continue
        found.append({
            # Snapshot pode estar stale — verificar o book ao vivo antes de agir
            "kind": "crossed_book", "guaranteed": False, "n_legs": 1,
            "arb_group": f"cross:{str(row.get('conditionId',''))[:12]}",
            "event_slug": str(row.get("event_slug") or ""),
            "basket_cost": ask, "payout_min": bid,
            "profit": round(bid - ask, 4), "edge": round(bid - ask, 4),
            "legs": [{
                "condition_id": str(row.get("conditionId", "")),
                "question": str(row.get("question", ""))[:80],
                "direction": "BUY_YES", "leg_price": ask,
                "bid": bid, "ask": ask,
                "liquidity": _f(row.get("liquidity")) or 0.0,
            }],
        })
    return found


# ──────────────────────────────────────────────────────────
# Output
# ──────────────────────────────────────────────────────────

def opportunities_to_frame(opps: list[dict]) -> pd.DataFrame:
    """Uma linha por PERNA, agrupadas por arb_group."""
    rows = []
    for opp in opps:
        for k, leg in enumerate(opp["legs"]):
            rows.append({
                "arb_group":    opp["arb_group"],
                "arb_kind":     opp["kind"],
                "leg":          k + 1,
                "n_legs":       opp["n_legs"],
                "condition_id": leg["condition_id"],
                "question":     leg["question"],
                "direction":    leg["direction"],
                "leg_price":    leg["leg_price"],
                "basket_cost":  opp["basket_cost"],
                "payout_min":   opp["payout_min"],
                "profit":       opp["profit"],
                "edge":         opp["edge"],
                "abs_edge":     abs(opp["edge"]),
                "guaranteed":   bool(opp.get("guaranteed", False)),
                "liquidity":    leg.get("liquidity", 0.0),
                "event_slug":   opp.get("event_slug", ""),
                "signal_source": "structural",
                "trade_type":    "arb",
                "confidence":    0.9,  # execução/resolução, não incerteza de modelo
            })
    return pd.DataFrame(rows)


def print_opportunities(opps: list[dict], top_n: int = 15) -> None:
    console.print("\n[bold cyan]══════════════════════════════════════════════════════════════════[/bold cyan]")
    console.print("[bold cyan]   ARB ESTRUTURAL — INTRA-POLYMARKET                              [/bold cyan]")
    console.print("[bold cyan]══════════════════════════════════════════════════════════════════[/bold cyan]\n")

    if not opps:
        console.print("[dim]Nenhuma oportunidade ≥ margem — mercados estruturalmente bem precificados.[/dim]")
        console.print("[dim]Isso é o esperado na maior parte do tempo; o scanner caça exceções.[/dim]\n")
        return

    table = Table(box=box.ROUNDED, show_lines=True)
    table.add_column("Tipo",      width=12)
    table.add_column("Grupo/Perna", width=44)
    table.add_column("Dir.",      width=8, justify="center")
    table.add_column("Preço",     width=7, justify="right")
    table.add_column("Lucro",     width=8, justify="right", style="bold green")
    table.add_column("Edge",      width=7, justify="right")
    table.add_column("Liquid.",   width=10, justify="right")

    for opp in sorted(opps, key=lambda o: (-o.get("guaranteed", False), -o["edge"]))[:top_n]:
        tag = "" if opp.get("guaranteed") else " [yellow]⚠ condicional[/yellow]"
        table.add_row(
            opp["kind"], f"[bold]{opp['arb_group']}[/bold]{tag}  (custo {opp['basket_cost']:.3f} → payout {opp['payout_min']:.0f})",
            "", "", f"${opp['profit']:.3f}", f"{opp['edge']:.1%}", "",
        )
        for leg in opp["legs"]:
            table.add_row(
                "", f"  {leg['question'][:42]}",
                leg["direction"].replace("BUY_", ""),
                f"{leg['leg_price']:.3f}", "", "",
                f"${leg.get('liquidity', 0):,.0f}",
            )
    console.print(table)

    guaranteed  = [o for o in opps if o.get("guaranteed")]
    conditional = [o for o in opps if not o.get("guaranteed")]
    console.print(
        f"\n[bold]Garantidas:[/bold] {len(guaranteed)}  "
        f"(lucro total 1 sh/perna: ${sum(o['profit'] for o in guaranteed):.2f})  |  "
        f"[bold]Condicionais:[/bold] {len(conditional)} "
        f"[dim](YES-basket sem outcome guarda-chuva / book stale — verificar antes de agir)[/dim]\n"
    )


def run(
    margin: float = MARGIN_MIN,
    max_events: int = MAX_EVENTS_PER_RUN,
    save: bool = True,
    include_negrisk: bool = True,
) -> pd.DataFrame:
    """Pipeline completo do scanner. Retorna DataFrame de pernas."""
    df = load_universe()

    opps: list[dict] = []
    opps += scan_monotonicity(df, margin=margin)
    opps += scan_crossed_books(df)
    if include_negrisk:
        opps += scan_negrisk(df, margin=margin, max_events=max_events)

    print_opportunities(opps)

    # Alerta Telegram para oportunidades GARANTIDAS (dedupe por arb_group no notify)
    try:
        from notify import arb_alert
        arb_alert(opps)
    except Exception as e:
        logger.warning(f"Alerta Telegram falhou (scanner segue normal): {e}")

    frame = opportunities_to_frame(opps)
    if save and not frame.empty:
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = REPORTS_DIR / f"signals_structural_{ts}.csv"
        frame.to_csv(path, index=False)
        logger.info(f"Sinais estruturais salvos: {path}")
    return frame


@click.command()
@click.option("--margin",     default=MARGIN_MIN, type=float, show_default=True,
              help="Lucro mínimo garantido (por $1 de payout) para reportar.")
@click.option("--max-events", default=MAX_EVENTS_PER_RUN, type=int, show_default=True,
              help="Máximo de eventos negRisk consultados na API por execução.")
@click.option("--no-negrisk", is_flag=True, default=False,
              help="Pula o detector negRisk (sem chamadas à API de eventos).")
@click.option("--no-save",    is_flag=True, default=False,
              help="Não salva CSV.")
def main(margin: float, max_events: int, no_negrisk: bool, no_save: bool) -> None:
    """Scanner de arb estrutural intra-Polymarket (NegRisk, monotonicidade, book cruzado)."""
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="<green>{time:HH:mm:ss}</green> | {message}")
    run(margin=margin, max_events=max_events, save=not no_save,
        include_negrisk=not no_negrisk)


if __name__ == "__main__":
    main()
