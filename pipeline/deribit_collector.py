"""
deribit_collector.py
Calcula probabilidades implícitas de opções Deribit para mercados crypto do Polymarket.

Edge = divergência entre preço Polymarket e P(S_T > K) calculada via Black-Scholes
       usando a IV (implied volatility) interpolada do mercado de opções.

Por que funciona:
  O mercado de opções Deribit é altamente líquido e eficiente — ele reflete o
  consenso de traders sofisticados sobre a distribuição futura de preços.
  Quando o Polymarket diverge dessa probabilidade, há edge explorável.

Fluxo:
  1. Detecta mercados Polymarket do tipo "Will BTC be above $X on date Y?"
  2. Para cada mercado, obtém IV interpolada das opções Deribit (mesma strike/expiry)
  3. Calcula P(S_T > K) via Black-Scholes (medida risk-neutral)
  4. Compara com yes_price do Polymarket → divergência = edge

API: https://docs.deribit.com (pública, sem autenticação)
"""

import re
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from loguru import logger
from rich.console import Console
from rich.table import Table
from rich import box
from scipy.stats import norm

from pipeline.market_pricing import entry_price_and_net_edge

RAW_MKT_DIR   = Path("data/raw/markets")
RAW_ODDS_DIR  = Path("data/raw/odds")
RAW_ODDS_DIR.mkdir(parents=True, exist_ok=True)

DERIBIT_BASE  = "https://www.deribit.com/api/v2"
console       = Console()

# Activos suportados: nome Deribit → variações de texto em perguntas Polymarket
ASSET_ALIASES: dict[str, list[str]] = {
    "BTC": ["bitcoin", "btc"],
    "ETH": ["ethereum", "eth", "ether"],
}

# Filtro de moneyness: o strike deve estar dentro de MAX_MONEYNESS_SIGMA desvios-padrão
# do spot para que o B-S tenha poder preditivo real.
#
# Por que isso importa:
#   B-S usa medida risk-neutral (Q) com drift r≈0. Para opções deep-OTM com T pequeno,
#   σ√T ≈ 2-3%, e qualquer strike além de 2σ√T do spot recebe probabilidade quase zero
#   independentemente do momentum real. O Polymarket, ao contrário, precifica momentum
#   corretamente → gera BUY_NO falso que perde quando o preço cruza o strike.
#
# Calibração empírica: com os 4 trades perdidos, todos tinham |S-K|/K > 2×σ√T.
# 1.5σ garante que só operamos perto do ATM, onde B-S e mercado concordam melhor.
MAX_MONEYNESS_SIGMA = 1.5

# Limite de yes_price para sinais BUY_NO.
# Quando o mercado precifica YES > 70%, ele está incorporando momentum/informação
# que B-S (sem drift) não captura. Lutar contra um consenso de 70%+ é sistematicamente
# perdedor com a medida risk-neutral.
MAX_YES_FOR_BUY_NO = 0.70

# Drift histórico anualizado por ativo (log-return, medida real P).
#
# Por que isso importa:
#   B-S puro usa r=0 (medida risk-neutral Q, drift esperado = 0). Mas o Polymarket
#   precifica probabilidades reais onde BTC/ETH têm drift positivo histórico.
#   Para mercados curtos (< 30 dias), a diferença é pequena (< 2pp). Para EOY
#   (270 dias com σ=80%), a diferença chega a 8-12pp — viés sistemático que gera
#   sinais BUY_NO falsos contra o drift de mercado.
#
# Valores baseados em retornos históricos 2020-2025 (período bull-bear completo):
#   BTC: ~55% anualizado (log-return); ETH: ~45%.
# Fonte: CoinMetrics / Messari historical returns.
ASSET_DRIFT: dict[str, float] = {
    "BTC": 0.55,
    "ETH": 0.45,
}


# ──────────────────────────────────────────────────────────
# Deribit API
# ──────────────────────────────────────────────────────────

def _deribit_get(method: str, params: dict) -> dict | None:
    """Chamada genérica à API pública do Deribit."""
    url = f"{DERIBIT_BASE}/public/{method}"
    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        if data.get("error"):
            logger.warning(f"Deribit error ({method}): {data['error']}")
            return None
        return data.get("result")
    except requests.RequestException as e:
        logger.error(f"Deribit request failed ({method}): {e}")
        return None


def get_spot_price(currency: str) -> float | None:
    """
    Retorna o preço spot atual (index price) de BTC ou ETH em USD.
    Usa o index da Deribit, que é média de múltiplas exchanges — mais estável que um único book.
    """
    index_name = f"{currency.lower()}_usd"
    result = _deribit_get("get_index_price", {"index_name": index_name})
    if result is None:
        return None
    return float(result.get("index_price", 0)) or None


def get_active_options(currency: str) -> list[dict]:
    """
    Retorna todos os instrumentos de opção ativos para a moeda dada.
    Cada item contém: instrument_name, strike, expiration_timestamp, option_type.
    """
    result = _deribit_get("get_instruments", {
        "currency": currency,
        "kind": "option",
        "expired": "false",
    })
    return result or []


def get_option_ticker(instrument_name: str) -> dict | None:
    """
    Retorna dados de mercado de uma opção específica.
    Campos relevantes: mark_iv (IV implícita), mark_price, bid_iv, ask_iv.
    """
    return _deribit_get("ticker", {"instrument_name": instrument_name})


# ──────────────────────────────────────────────────────────
# Parsing de perguntas Polymarket
# ──────────────────────────────────────────────────────────

# Mês abreviado → número
_MONTH_MAP = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
    # Versões em português (compatibilidade)
    "fev": 2, "abr": 4, "mai": 5, "ago": 8, "set": 9, "out": 10, "dez": 12,
}

_DATE_PATTERN = re.compile(
    r"(?:on\s+|by\s+|before\s+)?(?P<month>[A-Za-z]+)\s+(?P<day>\d{1,2})(?:,?\s*(?P<year>\d{4}))?",
    re.IGNORECASE,
)
_PRICE_PATTERN = re.compile(r"\$\s*(?P<num>[\d,]+(?:\.\d+)?)\s*(?P<suf>[kKmM])?\b")
_BETWEEN_PATTERN = re.compile(r"between", re.IGNORECASE)

_SUFFIX_MULT = {"k": 1_000, "m": 1_000_000}


def _parse_strike(text: str) -> float | None:
    """
    Extrai o valor de strike da pergunta. Ex: '$68,000' → 68000.0, '$100k' → 100000.0.

    P0-4: a versão antiga tinha duas regras de ×1000 sobrepostas (janela de
    ±2 caracteres + checagem de sufixo) que disparavam as DUAS para números de
    um dígito, multiplicando por 1000 duas vezes: "$3k" virava 3.000.000 em vez
    de 3.000. O sufixo agora entra na própria captura do regex e multiplica
    exatamente uma vez.

    Pergunta com múltiplos preços (ex: "between $X and $Y") não tem um strike
    único e não-ambíguo — rejeita em vez de inventar uma média. Foi uma média
    assim que produziu o strike de 3.000.000 usado no único "arb garantido"
    (falso) já executado em produção.
    """
    matches = _PRICE_PATTERN.findall(text)
    if len(matches) != 1:
        return None

    num_raw, suffix = matches[0]
    val = float(num_raw.replace(",", ""))
    if suffix:
        val *= _SUFFIX_MULT[suffix.lower()]
    return val


def _parse_expiry(text: str, reference_year: int | None = None) -> datetime | None:
    """
    Extrai a data de expiração da pergunta.
    Ex: 'on April 2' → datetime(<ano atual>, 4, 2, 16, 0, tzinfo=UTC)
    Polymarket usa 16:00 UTC como horário de resolução dos mercados diários.

    Sem ano explícito na pergunta, assume o ano corrente; se a data já passou
    (> 2 dias atrás), rola para o próximo ano — antes o ano era hardcoded 2026
    e em janeiro/2027 "by March 31" viraria uma data no passado.
    """
    match = _DATE_PATTERN.search(text)
    if not match:
        return None

    month_str = match.group("month").lower()[:3]
    month_num = _MONTH_MAP.get(month_str)
    if month_num is None:
        return None

    now  = datetime.now(timezone.utc)
    day  = int(match.group("day"))
    year_explicit = match.group("year") is not None
    year = int(match.group("year")) if year_explicit else (reference_year or now.year)

    try:
        # Polymarket resolve às 16:00 UTC nos mercados diários
        dt = datetime(year, month_num, day, 16, 0, 0, tzinfo=timezone.utc)
    except ValueError:
        return None

    if not year_explicit and reference_year is None and dt < now - timedelta(days=2):
        try:
            dt = datetime(year + 1, month_num, day, 16, 0, 0, tzinfo=timezone.utc)
        except ValueError:
            return None  # ex: 29 de fevereiro em ano não bissexto

    return dt


def _detect_asset(text: str) -> str | None:
    """Detecta BTC ou ETH na pergunta usando word boundaries para evitar falsos positivos."""
    import re as _re
    text_lower = text.lower()
    for currency, aliases in ASSET_ALIASES.items():
        for alias in aliases:
            # word boundary: alias não pode estar embutido em outra palavra
            if _re.search(rf"\b{_re.escape(alias)}\b", text_lower):
                return currency
    return None


def _detect_direction(text: str) -> tuple[str | None, bool]:
    """
    Detecta se é mercado 'above'/'below' e se é touch option (reach/hit/by).

    Returns:
        (direction, is_touch) onde is_touch=True significa que o mercado precisa
        apenas tocar o nível em algum ponto (barreira), não terminar acima/abaixo.
        Exemplos:
          "be above $X on date"   → ("above", False) — europeia
          "reach $X by date"      → ("above", True)  — barreira
          "hit $X by date"        → ("above", True)  — barreira
    """
    text_lower = text.lower()
    is_touch = any(w in text_lower for w in ["reach", "hit", "dip"])
    if any(w in text_lower for w in ["above", "exceed", "greater than", "more than", "over", "reach", "hit"]):
        return "above", is_touch
    if any(w in text_lower for w in ["below", "less than", "under", "beneath", "dip"]):
        return "below", is_touch
    return None, False


def parse_crypto_market(question: str, spot: float | None = None) -> dict | None:
    """
    Extrai (asset, strike, expiry, direction, is_touch) de uma pergunta Polymarket.
    Retorna None se não conseguir parsear com confiança.

    Exemplos suportados:
      "Will the price of Bitcoin be above $68,000 on April 2?"     → europeia
      "Will Bitcoin reach $100,000 by December 31, 2026?"          → touch/barreira
      "Will Bitcoin hit $150k by December 31, 2026?"               → touch/barreira
      "Will the price of Ethereum be greater than $2,500 on April 2?"
      "Will the price of Bitcoin be less than $62,000 on April 1?"

    spot: preço à vista atual do ativo, se disponível. Para mercados de touch,
    a direção real é dada pelo strike vs. spot ATUAL, não pela keyword da
    pergunta — "Will BTC hit $50k?" com spot em $100k é barreira DOWNWARD (o
    preço precisa CAIR até 50k), mas "hit/reach" por si só sugere upward.
    Usar a keyword sozinha classificava esse caso como upward, e a fórmula
    upward com K < S satura em P≈1 (BUY_YES espúrio — auditoria 2026-07).
    Sem spot, fica a direção inferida por keyword (é o que P0-4 corrigiu no
    parser de strike, mas a correção de direção por spot é P0-4 correlato —
    todo consumidor que tiver spot disponível deve passá-lo aqui, em vez de
    reimplementar "strike > spot" no próprio call site).
    """
    asset              = _detect_asset(question)
    direction, is_touch = _detect_direction(question)
    strike             = _parse_strike(question)
    expiry             = _parse_expiry(question)

    if not asset or not direction or strike is None or expiry is None:
        return None
    if strike <= 0:
        return None

    if is_touch and spot is not None and spot > 0:
        direction = "above" if strike > spot else "below"

    return {
        "asset":      asset,
        "direction":  direction,
        "strike":     strike,
        "expiry":     expiry,
        "is_touch":   is_touch,
        "is_between": bool(_BETWEEN_PATTERN.search(question)),
    }


# ──────────────────────────────────────────────────────────
# Interpolação de IV pelo smile de volatilidade
# ──────────────────────────────────────────────────────────

def _group_options_by_expiry(options: list[dict]) -> dict[datetime, list[dict]]:
    """Agrupa opções por data de expiração (arredondada ao dia)."""
    groups: dict[datetime, list[dict]] = {}
    for opt in options:
        ts_ms = opt.get("expiration_timestamp", 0)
        expiry = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).replace(
            hour=8, minute=0, second=0, microsecond=0
        )
        groups.setdefault(expiry, []).append(opt)
    return groups


def _find_nearest_expiry(
    groups: dict[datetime, list[dict]],
    target: datetime,
    max_delta_days: int = 5,
) -> datetime | None:
    """
    Retorna a expiração Deribit mais próxima do target.
    Só aceita expirações FUTURAS (>= target - 12h) dentro da janela máxima.
    """
    now = datetime.now(timezone.utc)
    candidates = [
        exp for exp in groups
        if exp >= now - timedelta(hours=12)  # não expirado
    ]
    if not candidates:
        return None

    # Prefere expirações que "englobem" o target (exp >= target)
    # Fallback: a mais próxima em valor absoluto dentro da janela
    valid = [exp for exp in candidates if abs((exp - target).days) <= max_delta_days]
    if not valid:
        return None

    return min(valid, key=lambda exp: abs((exp - target).total_seconds()))


def interpolate_iv(
    options_at_expiry: list[dict],
    target_strike: float,
    option_type: str = "C",
) -> float | None:
    """
    Interpola a IV para um strike específico a partir das opções disponíveis.

    Estratégia:
      1. Filtra opções do tipo especificado (C=call, P=put)
      2. Ordena por strike
      3. Interpola linearmente no espaço log(K) entre os dois strikes mais próximos
      4. Fallback: IV da opção mais próxima (se não houver dois candidatos)

    Returns:
        IV anualizada (e.g., 0.65 para 65%) ou None se indisponível.
    """
    # Deribit retorna "call"/"put" (lowercase) — normaliza para C/P
    _type_map = {"c": "call", "p": "put", "call": "call", "put": "put"}
    want = _type_map.get(option_type.lower(), "call")
    other_want = "put" if want == "call" else "call"

    candidates = [
        opt for opt in options_at_expiry
        if opt.get("option_type", "").lower() == want
    ]
    if not candidates:
        candidates = [
            opt for opt in options_at_expiry
            if opt.get("option_type", "").lower() == other_want
        ]
    if not candidates:
        return None

    # Ordena por strike
    candidates.sort(key=lambda o: float(o.get("strike", 0)))
    strikes = [float(o["strike"]) for o in candidates]

    # Encontra os dois strikes vizinhos do target (um acima, um abaixo)
    idx_above = next((i for i, s in enumerate(strikes) if s >= target_strike), None)

    if idx_above is None:
        # Target acima de todos os strikes disponíveis — usa o mais alto
        return _fetch_iv(candidates[-1]["instrument_name"])

    if idx_above == 0:
        # Target abaixo de todos os strikes disponíveis — usa o mais baixo
        return _fetch_iv(candidates[0]["instrument_name"])

    # Interpolação linear em log(K) entre os dois vizinhos
    k_lo = strikes[idx_above - 1]
    k_hi = strikes[idx_above]

    iv_lo = _fetch_iv(candidates[idx_above - 1]["instrument_name"])
    iv_hi = _fetch_iv(candidates[idx_above]["instrument_name"])

    if iv_lo is not None and iv_hi is not None and k_lo != k_hi:
        w = (np.log(target_strike) - np.log(k_lo)) / (np.log(k_hi) - np.log(k_lo))
        w = float(np.clip(w, 0, 1))
        return (1 - w) * iv_lo + w * iv_hi

    return iv_lo or iv_hi


def _fetch_iv(instrument_name: str) -> float | None:
    """Busca mark_iv do ticker Deribit para um instrumento. Retorna IV em fração (0.65 = 65%)."""
    ticker = get_option_ticker(instrument_name)
    if ticker is None:
        return None
    # mark_iv já vem em % na Deribit (ex: 65.0 = 65%)
    raw_iv = ticker.get("mark_iv")
    if raw_iv is None or raw_iv == 0:
        # Fallback: tenta bid_iv / ask_iv
        bid_iv = ticker.get("greeks", {}).get("iv") or ticker.get("bid_iv")
        ask_iv = ticker.get("greeks", {}).get("iv") or ticker.get("ask_iv")
        if bid_iv and ask_iv:
            raw_iv = (float(bid_iv) + float(ask_iv)) / 2
        elif bid_iv:
            raw_iv = float(bid_iv)
        elif ask_iv:
            raw_iv = float(ask_iv)
        else:
            return None
    return float(raw_iv) / 100.0  # converte % → fração


# ──────────────────────────────────────────────────────────
# Black-Scholes: probabilidade log-normal
# ──────────────────────────────────────────────────────────

def bs_prob(S: float, K: float, T: float, sigma: float,
            r: float = 0.0, above: bool = True, touch: bool = False) -> float:
    """
    Probabilidade de preço via Black-Scholes com drift real (medida P).

    Dois modos:
      touch=False (europeia): P(S_T > K) — o ativo deve estar acima do strike *na expiração*.
                              Usado para "Will BTC be above $X on [date]?".

      touch=True (barreira):  primeiro tempo de passagem — o ativo deve tocar o nível
                              *em algum ponto* antes de T.
                              Usado para "Will BTC reach/hit $X by [date]?".

    Fórmulas (first-passage time com drift μ = r - σ²/2; Harrison,
    "Brownian Motion and Stochastic Flow Systems", §1.8):
      Upward (above=True):
        P(max S_t ≥ K) = N(d₂) + exp(2μ ln(K/S) / σ²) × N(-d₂ - 2ln(K/S)/(σ√T))

      Downward (above=False):
        P(min S_t ≤ K) = N(-d₂) + exp(2μ ln(K/S) / σ²) × N(d₂ + 2ln(K/S)/(σ√T))

      Onde d₂ = [ln(S/K) + μT] / (σ√T)

    Args:
        S:     preço spot atual
        K:     strike (limiar da pergunta)
        T:     tempo até expiração em anos
        sigma: volatilidade implícita anualizada (ex: 0.65)
        r:     drift anualizado do ativo (use ASSET_DRIFT para medida real P)
        above: True → acima do strike, False → abaixo
        touch: True → probabilidade de tocar (barreira), False → europeia
    """
    if T <= 0:
        return (1.0 if S > K else 0.0) if above else (1.0 if S < K else 0.0)
    if sigma <= 0:
        return (1.0 if S > K else 0.0) if above else (1.0 if S < K else 0.0)

    sig_sqrtT = sigma * np.sqrt(T)              # σ√T — não √T, o nome antigo "sqrtT" enganava
    mu        = r - 0.5 * sigma ** 2            # drift do log-preço
    d2        = (np.log(S / K) + mu * T) / sig_sqrtT

    if not touch:
        # Opção europeia: P(S_T > K) = N(d₂) sob medida P com drift r
        return float(norm.cdf(d2) if above else norm.cdf(-d2))

    # Opção barreira (first-passage time com drift)
    log_KS = np.log(K / S)
    factor = np.exp(2.0 * mu * log_KS / sigma ** 2)
    arg    = d2 + 2.0 * log_KS / sig_sqrtT
    if above:
        # P(max_{0,T} S_t ≥ K) = N(d₂) + exp(2μ ln(K/S)/σ²) × N(-arg)
        p_touch = float(norm.cdf(d2) + factor * norm.cdf(-arg))
    else:
        # P(min_{0,T} S_t ≤ K) = N(-d₂) + exp(2μ ln(K/S)/σ²) × N(arg)
        p_touch = float(norm.cdf(-d2) + factor * norm.cdf(arg))

    return float(np.clip(p_touch, 0.0, 1.0))


# ──────────────────────────────────────────────────────────
# Pipeline principal
# ──────────────────────────────────────────────────────────

def load_active_markets(min_liquidity: float = 5_000) -> pd.DataFrame:
    """Carrega o snapshot mais recente de mercados ativos do Polymarket."""
    candidates = sorted(
        list(RAW_MKT_DIR.glob("markets_all_*.parquet")) +
        list(RAW_MKT_DIR.glob("markets_incremental_*.parquet")),
        key=lambda p: p.stat().st_mtime,  # ordena por tempo de modificação, não por nome
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            f"Nenhum arquivo de mercados em {RAW_MKT_DIR}. "
            "Execute: uv run python -m pipeline.fetch_markets"
        )
    df = pd.read_parquet(candidates[0])
    if "active" in df.columns:
        df = df[df["active"] == True]
    if "closed" in df.columns:
        df = df[df["closed"] == False]
    if "liquidity" in df.columns:
        df = df[pd.to_numeric(df["liquidity"], errors="coerce").fillna(0) >= min_liquidity]
    return df.reset_index(drop=True)


def run(
    min_divergence: float = 0.03,
    min_liquidity: float = 5_000,
    min_hours_left: float = 8.0,
    max_hours_left: float = 8_760.0,   # 1 ano — captura mercados EOY/mensais
    max_delta_days: int = 30,          # Deribit expiração mais próxima pode diferir em semanas
    save: bool = True,
) -> pd.DataFrame:
    """
    Pipeline completo: carrega mercados → obtém IV Deribit → calcula divergência.

    Args:
        min_divergence:  |polymarket_price - bs_prob| mínimo para incluir sinal
        min_liquidity:   liquidez mínima do mercado Polymarket em USDC
        min_hours_left:  horas mínimas até expiração (padrão: 8h)
                         Evita as últimas horas onde gamma é muito alto e IV intraday ruidosa.
                         Combinado com MIN_HOLD_HOURS=2 no risk_manager, dá margem suficiente.
        max_hours_left:  horas máximas até expiração (padrão: 168h = 7 dias)
                         Evita mercados muito distantes onde Deribit não tem opções matching.
        max_delta_days:  janela máxima (dias) entre expiração Deribit e Polymarket
        save:            salva resultado em Parquet

    Returns:
        DataFrame com sinais rankeados por |divergência|.
    """
    now = datetime.now(timezone.utc)
    markets = load_active_markets(min_liquidity=min_liquidity)

    # O snapshot mais recente por mtime pode ser um markets_all_* gravado
    # antes do guard de coleta vazia existir (P0-6) ou, mesmo depois dele,
    # um arquivo sem "question" por outro motivo — sem isso, a Gamma API fora
    # do ar não vira "sem sinal deribit este ciclo", vira exceção não tratada
    # que run_cycle.py registra como falha de etapa.
    if markets.empty or "question" not in markets.columns:
        logger.warning(
            f"load_active_markets() devolveu {len(markets)} mercados sem coluna 'question' "
            "utilizável — sem sinal deribit este ciclo (Gamma API fora do ar ou snapshot "
            "vazio/corrompido). Verifique fetch_markets.py."
        )
        return pd.DataFrame()

    # Filtra apenas mercados de preço específico (above/below/reach $X) para BTC/ETH
    price_mask = markets["question"].str.contains(
        r"above|below|greater than|less than|exceed|over|under|reach|hit",
        case=False, na=False, regex=True,
    )
    # word-boundary para evitar falsos positivos: "Ethena" não é ETH
    crypto_mask = markets["question"].str.contains(
        r"\bbitcoin\b|\bbtc\b|\bethereum\b|\beth\b", case=False, na=False, regex=True
    )
    # Exclui mercados "between" por enquanto (requer tratamento especial)
    between_mask = ~markets["question"].str.contains(r"between", case=False, na=False)

    candidates = markets[price_mask & crypto_mask & between_mask].copy()
    logger.info(f"Mercados crypto de preço específico: {len(candidates):,}")

    # Filtra por janela de tempo e preço genuinamente incerto
    end_raw = pd.to_datetime(candidates["endDate"], errors="coerce", utc=True)
    hours_left = ((end_raw - pd.Timestamp(now)) / pd.Timedelta(hours=1)).fillna(-1)
    candidates = candidates[
        (hours_left >= min_hours_left) &
        (hours_left <= max_hours_left) &
        (pd.to_numeric(candidates["yes_price"], errors="coerce").between(0.03, 0.97))
    ].copy()
    logger.info(f"Após filtros de tempo ({min_hours_left:.0f}h–{max_hours_left:.0f}h) e preço incerto: {len(candidates):,}")

    if candidates.empty:
        logger.warning("Nenhum mercado crypto elegível encontrado.")
        return pd.DataFrame()

    # Obtém preços spot e opções por ativo
    assets_needed = set()
    parsed_markets = []
    for _, row in candidates.iterrows():
        parsed = parse_crypto_market(str(row.get("question", "")))
        if parsed:
            assets_needed.add(parsed["asset"])
            parsed_markets.append((row, parsed))

    logger.info(f"Mercados parseados com sucesso: {len(parsed_markets):,}")

    # Busca spot e opções para cada ativo (cache por ativo)
    spot_cache: dict[str, float] = {}
    options_cache: dict[str, list[dict]] = {}

    for asset in assets_needed:
        logger.info(f"Buscando spot e opções para {asset}...")
        spot = get_spot_price(asset)
        if spot:
            spot_cache[asset] = spot
            logger.info(f"  {asset} spot: ${spot:,.0f}")
        else:
            logger.warning(f"  Spot indisponível para {asset}")

        options = get_active_options(asset)
        if options:
            options_cache[asset] = options
            logger.info(f"  {asset}: {len(options):,} opções ativas")
        else:
            logger.warning(f"  Opções indisponíveis para {asset}")
        time.sleep(0.2)

    # Processa cada mercado
    rows = []
    api_calls = 0

    for market_row, parsed in parsed_markets:
        asset     = parsed["asset"]
        strike    = parsed["strike"]
        expiry    = parsed["expiry"]
        direction = parsed["direction"]
        is_touch  = parsed.get("is_touch", False)

        spot = spot_cache.get(asset)
        opts = options_cache.get(asset, [])

        if spot is None or not opts:
            continue

        # Touch/barreira: a direção real é dada pelo strike vs. spot ATUAL, não
        # pela keyword da primeira parse — spot só fica disponível depois
        # (duas passadas: parse geral, depois busca de spot por ativo). A
        # MESMA regra agora mora em parse_crypto_market (P0-4 correlato) para
        # que outros consumidores com spot disponível NO MOMENTO DO PARSE
        # (como structural_arb) herdem automaticamente — aqui, sem spot na
        # primeira passada, aplica-se direto sobre os valores já extraídos.
        if is_touch:
            direction = "above" if strike > spot else "below"
            parsed["direction"] = direction

        # Agrupa opções por expiração e encontra a mais próxima
        groups = _group_options_by_expiry(opts)
        nearest_expiry = _find_nearest_expiry(groups, expiry, max_delta_days=max_delta_days)
        if nearest_expiry is None:
            logger.debug(f"  Sem expiração Deribit próxima para {expiry.date()} — pulando")
            continue

        opts_at_expiry = groups[nearest_expiry]

        # Tipo de opção: call para "above", put para "below"
        opt_type = "C" if direction == "above" else "P"
        iv = interpolate_iv(opts_at_expiry, strike, option_type=opt_type)
        api_calls += 2  # Estimativa de chamadas à API
        time.sleep(0.1)  # Rate limit suave

        if iv is None or iv <= 0:
            logger.debug(f"  IV indisponível para {asset} K={strike:,.0f} — pulando")
            continue

        # Tempo até expiração em anos
        T = max((expiry - now).total_seconds() / (365.25 * 24 * 3600), 1 / 8760)

        # Filtro de moneyness: |S - K| / K deve ser < MAX_MONEYNESS_SIGMA × σ√T
        # Fora dessa faixa, B-S (sem drift) atribui probabilidades próximas de 0 ou 1
        # independentemente do momentum real → sinal espúrio.
        moneyness_pct = abs(spot - strike) / strike
        bs_reach      = MAX_MONEYNESS_SIGMA * iv * np.sqrt(T)
        if moneyness_pct > bs_reach:
            logger.debug(
                f"  Moneyness {moneyness_pct:.1%} > {MAX_MONEYNESS_SIGMA}×σ√T={bs_reach:.1%} "
                f"({asset} spot=${spot:,.0f} K=${strike:,.0f}) — pulando"
            )
            continue

        # Probabilidade Black-Scholes com drift histórico do ativo (medida real P)
        # is_touch=True → fórmula de barreira P(max/min S_t > K) — para "reach/hit by date"
        # is_touch=False → fórmula europeia P(S_T > K)           — para "be above on date"
        drift = ASSET_DRIFT.get(asset, 0.0)
        fair_prob = bs_prob(
            S=spot,
            K=strike,
            T=T,
            sigma=iv,
            r=drift,
            above=(direction == "above"),
            touch=is_touch,
        )

        yes_price  = float(market_row.get("yes_price") or 0.5)
        divergence = round(yes_price - fair_prob, 4)

        # Não fazer BUY_NO quando o mercado já precifica YES > MAX_YES_FOR_BUY_NO.
        # O Polymarket incorpora momentum e informação direcional que B-S ignora.
        # Sinal BUY_NO contra um consenso de 70%+ é sistematicamente perdedor.
        if divergence > 0 and yes_price > MAX_YES_FOR_BUY_NO:
            logger.debug(
                f"  BUY_NO vetado: yes_price={yes_price:.1%} > {MAX_YES_FOR_BUY_NO:.0%} "
                f"({asset} K=${strike:,.0f})"
            )
            continue

        trade_signal = "BUY_NO" if divergence > 0 else "BUY_YES"

        # P1-25: entry_price executável (bestAsk/1-bestBid), não yes_price
        # (lastTradePrice). net_edge já desconta spread real e fees.
        priced = entry_price_and_net_edge(
            market_row, trade_signal, fair_prob, 1.0 - fair_prob, source="deribit",
        )
        if priced is None:
            continue  # sem book confiável nesse lado, ou spread come o edge mínimo
        entry_price, book_spread, net_edge = priced

        end_dt     = pd.to_datetime(market_row.get("endDate"), errors="coerce", utc=True)
        hrs_left   = ((end_dt - pd.Timestamp(now)) / pd.Timedelta(hours=1)) if pd.notna(end_dt) else -1

        rows.append({
            "condition_id":   market_row.get("conditionId"),
            "question":       str(market_row.get("question", ""))[:80],
            "category":       str(market_row.get("category", "")),
            "underlying":     asset,   # P1-11: ativo real p/ cap de exposição por underlying
            "asset":          asset,
            "strike":         strike,
            "direction":      direction,
            "yes_price":      round(yes_price, 4),
            "entry_price":    round(entry_price, 4),
            "spread":         round(book_spread, 4),
            "fair_prob":      round(fair_prob, 4),
            "divergence":     divergence,           # legado, mid-based — só p/ display
            "abs_divergence": abs(divergence),
            "net_edge":       net_edge,             # edge de verdade: fair - entry_price - fee
            "signal":         trade_signal,
            "is_touch":       is_touch,
            "spot_price":     round(spot, 2),
            "iv":             round(iv, 4),
            "iv_pct":         f"{iv*100:.1f}%",
            "T_days":         round(T * 365.25, 1),
            "moneyness_pct":  round(moneyness_pct, 4),  # |S-K|/K
            "bs_reach":       round(bs_reach, 4),        # MAX_SIGMA × σ√T
            "deribit_expiry": nearest_expiry.strftime("%Y-%m-%d"),
            "expiry_delta_d": abs((nearest_expiry - expiry).days),
            "liquidity":      float(market_row.get("liquidity") or 0),
            "volume_24h":     float(market_row.get("volume24hr") or 0),
            "hours_left":     round(float(hrs_left), 1),
            "end_date":       market_row.get("endDate", ""),
        })

    logger.info(f"Chamadas à API Deribit (estimado): {api_calls}")

    if not rows:
        logger.warning("Nenhum sinal calculado. Verifique disponibilidade da Deribit API.")
        return pd.DataFrame()

    df = (
        pd.DataFrame(rows)
        .sort_values("net_edge", ascending=False)
        .reset_index(drop=True)
    )

    # P1-25: net_edge já é líquido de spread/fees — ordenar/filtrar por ele
    # em vez de abs_divergence (mid-based, ignora se o book dá pra executar).
    # Sem consensus_spread equivalente aqui, sem shrinkage (P1-26) — só odds tem.
    result = df[df["net_edge"] >= min_divergence].reset_index(drop=True)
    logger.info(f"Sinais calculados: {len(df):,} | Com net_edge >= {min_divergence}: {len(result):,}")

    if save and not result.empty:
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = RAW_ODDS_DIR / f"deribit_signals_{ts}.parquet"
        result.to_parquet(path, index=False, compression="snappy")
        logger.info(f"Salvo em {path}")

    return result


# ──────────────────────────────────────────────────────────
# Display
# ──────────────────────────────────────────────────────────

def print_signals_table(df: pd.DataFrame, top_n: int = 20) -> None:
    """Exibe tabela de sinais Deribit no terminal."""
    console.print("\n[bold cyan]══════════════════════════════════════════════════════════════════[/bold cyan]")
    console.print("[bold cyan]   SINAIS CRYPTO — DERIBIT IV vs. POLYMARKET                      [/bold cyan]")
    console.print("[bold cyan]══════════════════════════════════════════════════════════════════[/bold cyan]\n")

    if df.empty:
        console.print("[yellow]Nenhum sinal encontrado.[/yellow]")
        console.print("  • Verifique se há mercados crypto ativos com preço entre 5% e 95%")
        console.print("  • Execute: uv run python -m pipeline.fetch_markets\n")
        return

    display = df.head(top_n)
    table = Table(box=box.ROUNDED, show_lines=True)
    table.add_column("#",        width=3,  justify="right")
    table.add_column("Questão",  width=44)
    table.add_column("Sinal",    width=9,  justify="center")
    table.add_column("Poly",     width=7,  justify="right")
    table.add_column("BS Fair",  width=7,  justify="right")
    table.add_column("Edge",     width=8,  justify="right", style="bold")
    table.add_column("IV",       width=7,  justify="right")
    table.add_column("Spot",     width=11, justify="right")
    table.add_column("Hrs",      width=5,  justify="right")

    for i, row in display.iterrows():
        div        = row["divergence"]
        div_color  = "red" if div > 0 else "green"
        sig_color  = "magenta" if row["signal"] == "BUY_NO" else "green"
        table.add_row(
            str(i + 1),
            str(row["question"])[:43],
            f"[{sig_color}]{row['signal']}[/{sig_color}]",
            f"{row['yes_price']:.3f}",
            f"{row['fair_prob']:.3f}",
            f"[{div_color}]{div:+.3f}[/{div_color}]",
            row["iv_pct"],
            f"${row['spot_price']:,.0f}",
            str(int(row["hours_left"])),
        )

    console.print(table)
    console.print(f"\n[bold]Total de sinais:[/bold] {len(df):,}  |  Exibindo top {min(top_n, len(df))}")
    console.print(f"[bold]Edge médio (abs):[/bold] {df['abs_divergence'].mean():.3f}")
    console.print(f"[bold]IV média:[/bold]         {df['iv'].mean()*100:.1f}%")
    console.print(f"[bold]Fonte:[/bold] Black-Scholes (Deribit implied volatility)\n")


# ──────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import click

    @click.command()
    @click.option("--min-divergence",  default=0.03,  type=float, show_default=True,
                  help="Divergência mínima |poly - fair_prob| para exibir sinal.")
    @click.option("--min-liquidity",   default=5_000, type=float, show_default=True,
                  help="Liquidez mínima do mercado Polymarket em USDC.")
    @click.option("--min-hours-left",  default=1.0,   type=float, show_default=True,
                  help="Horas mínimas até expiração do mercado.")
    @click.option("--max-delta-days",  default=5,     type=int,   show_default=True,
                  help="Janela máxima (dias) para aceitar expiração Deribit vs. Polymarket.")
    @click.option("--top-n",           default=20,    type=int,   show_default=True,
                  help="Quantos sinais exibir na tabela.")
    @click.option("--no-save",         is_flag=True,  default=False,
                  help="Não salvar resultado em Parquet.")
    def main(min_divergence, min_liquidity, min_hours_left, max_delta_days, top_n, no_save):
        """Calcula edge em mercados crypto do Polymarket via IV do Deribit."""
        logger.remove()
        logger.add(sys.stderr, level="INFO", format="<green>{time:HH:mm:ss}</green> | {message}")

        console.print("\n[bold]Polymarket Quant — Deribit IV Collector[/bold]")
        console.print("Comparando probabilidades implícitas (Black-Scholes) com Polymarket\n")

        df = run(
            min_divergence=min_divergence,
            min_liquidity=min_liquidity,
            min_hours_left=min_hours_left,
            max_delta_days=max_delta_days,
            save=not no_save,
        )
        print_signals_table(df, top_n=top_n)

        if not df.empty:
            console.print("[dim]Para integrar ao signal_generator:[/dim]")
            console.print("[dim]  uv run python -m signals.run_signals --mode deribit[/dim]")

    main()
