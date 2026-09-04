"""
market_pricing.py
Preço executável e edge líquido de custo — compartilhado entre odds_collector.py
e deribit_collector.py (P1-25/P1-26).

A camada de sinal historicamente usava `yes_price` (lastTradePrice — uma
impressão do último trade, não um preço que dá para pegar) para calcular
divergência. O meio-spread sozinho é rotineiramente 1-3pp nesses books —
contra um threshold de 0.03-0.08, o "edge" filtrado é majoritariamente custo
de execução, não alpha. `entry_price_and_net_edge()` usa bestBid/bestAsk
(já coletados pelo gamma_collector, nunca usados para isso) para calcular o
preço real de entrada e o edge que sobra depois do spread e de fees.
"""

import csv
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "risk"))
from risk_manager import MIN_EDGE_TO_TRADE, TRANSACTION_FEE_PCT, MAX_SPREAD_TO_MIN_EDGE_RATIO

RAW_ODDS_DIR = Path("data/raw/odds")
RAW_ODDS_DIR.mkdir(parents=True, exist_ok=True)

REJECTED_EDGES_LOG = RAW_ODDS_DIR / "rejected_implausible_edges.csv"


def log_rejected_implausible_edge(
    question: str, yes_price: float, fair_prob: float, divergence: float, source: str,
) -> None:
    """
    P1-26: MAX_PLAUSIBLE_EDGE descarta o topo da cauda — exatamente onde vivem
    os bugs de matching (P1-21, P1-22). Um logger.debug perde a rejeição pra
    sempre; gravar num CSV revisável transforma o filtro em detector de bug.
    """
    is_new = not REJECTED_EDGES_LOG.exists()
    with open(REJECTED_EDGES_LOG, "a", newline="") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(["timestamp", "source", "question", "yes_price", "fair_prob", "divergence"])
        writer.writerow([
            datetime.now(timezone.utc).isoformat(), source, question[:120],
            round(yes_price, 4), round(fair_prob, 4), divergence,
        ])


def _safe_float(x) -> float | None:
    try:
        if x is None or pd.isna(x):
            return None
        return float(x)
    except (TypeError, ValueError):
        return None


def entry_price_and_net_edge(
    mkt, direction: str, fair_yes: float, fair_no: float, source: str,
) -> tuple[float, float, float] | None:
    """
    P1-25: preço executável de verdade — bestAsk pra BUY_YES, 1-bestBid pra
    BUY_NO — em vez de yes_price (lastTradePrice: uma impressão, não um preço
    que dá pra pegar). net_edge já desconta o custo real (spread + fees).

    Args:
        mkt: linha de mercado (Series ou dict) com bestBid/bestAsk/spread.
        source: chave em risk_manager.MIN_EDGE_TO_TRADE ("odds" ou "deribit")
                — usada pra decidir quanto de spread é aceitável.

    Retorna (entry_price, spread, net_edge), ou None quando o book não tem o
    lado necessário pra essa direção, ou o spread sozinho já consome o edge
    mínimo da fonte.
    """
    best_bid = _safe_float(mkt.get("bestBid"))
    best_ask = _safe_float(mkt.get("bestAsk"))

    if direction == "BUY_YES":
        if best_ask is None:
            return None
        entry_price, fair_for_direction = best_ask, fair_yes
    else:
        if best_bid is None:
            return None
        entry_price, fair_for_direction = 1.0 - best_bid, fair_no

    spread = (
        (best_ask - best_bid) if (best_ask is not None and best_bid is not None)
        else float(mkt.get("spread") or 0)
    )
    if spread > MAX_SPREAD_TO_MIN_EDGE_RATIO * MIN_EDGE_TO_TRADE.get(source, 0.05):
        return None  # spread sozinho já consome o edge mínimo da fonte

    net_edge = round(fair_for_direction - entry_price - TRANSACTION_FEE_PCT, 4)
    return entry_price, spread, net_edge


def apply_edge_shrinkage(df: pd.DataFrame) -> pd.DataFrame:
    """
    P1-26 (maldição do vencedor): ordenar milhares de estimativas ruidosas por
    net_edge bruto seleciona o topo da distribuição de ERRO da referência, não
    de mispricing real — não há shrinkage, erro-padrão por sinal, nem correção
    de múltiplos testes.

    `consensus_spread` (desvio-padrão entre bookmakers, só existe em sinais de
    odds) já é um proxy de incerteza calculado e disponível — hoje só usado
    como multiplicador de confidence. Encolhe net_edge em direção a zero
    proporcionalmente a ele:

        shrunk = net_edge × var_edge / (var_edge + consensus_spread²)

    onde var_edge é a variância amostral do próprio lote de net_edge (batch
    atual) — não há prior externo calibrado (ver P1-27), então usamos a
    variância observada como estimador de quanto os net_edge "verdadeiros"
    variam entre si. Um sinal com consensus_spread baixo (bookmakers
    concordam) é pouco encolhido; um com consensus_spread alto (bookmakers
    discordam — ruído, não sinal) é puxado quase a zero.

    Sem `consensus_spread` (deribit, sem múltiplos books — nenhum proxy de
    ruído equivalente existe hoje) a função devolve net_edge sem encolher.
    """
    out = df.copy()
    if "net_edge" not in out.columns or "consensus_spread" not in out.columns:
        out["shrunk_edge"] = out.get("net_edge", out.get("abs_divergence", 0.0))
        return out

    var_edge = out["net_edge"].var()
    if pd.isna(var_edge) or var_edge <= 0:
        out["shrunk_edge"] = out["net_edge"]
        return out

    consensus = pd.to_numeric(out["consensus_spread"], errors="coerce").fillna(0.0)
    out["shrunk_edge"] = (out["net_edge"] * var_edge / (var_edge + consensus ** 2)).round(4)
    return out
