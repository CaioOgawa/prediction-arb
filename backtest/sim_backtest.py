"""
sim_backtest.py
Simulador walk-forward histórico do modelo Deribit / Black-Scholes.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
AVISO — Preços Sintéticos + Desconto de Mercado
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
A API CLOB do Polymarket não retém histórico de preços para mercados
já resolvidos. Por isso, esta simulação usa a probabilidade Black-Scholes
como base, com um fator de desconto opcional para simular a ineficiência
histórica observada do Polymarket.

  market_price = BS_prob × (1 − market_discount)

  --market-discount 0.00 → sem edge (fair price = BS_prob). Resultado: 0 trades
                            porque Kelly = 0 quando entry_price = fair_value.
  --market-discount 0.15-0.25 → cenário de ineficiência assumida, ESCOLHIDO,
                            não medido — não há citação nem dado que sustente
                            um número específico aqui (P2-32). Como
                            entry_price = BS_prob × (1 − desconto), edge =
                            desconto × BS_prob por construção: qualquer
                            desconto > 0 garante EV positivo simulado, sempre
                            a favor. O simulador não consegue produzir P&L
                            negativo nesse regime, nem tem como distinguir
                            "modelo bom" de "desconto generoso". Rode
                            `--discount-sweep` pra ver os dois números
                            (P&L, Sharpe) subirem junto com o parâmetro, e
                            compare com o Brier score / erro de calibração
                            (esses dois não mudam com o desconto — são a
                            parte deste simulador que mede alguma coisa real).

O que se testa com isso (P2-32 reenquadrou pra deixar isso honesto):
  1. Calibração: BS_prob prevista vs. resultado real (Brier score, erro por
     faixa) — o único resultado deste simulador que não depende do desconto.
  2. Kelly sizing: adequação do tamanho dado o desconto assumido.
  3. Early exits: profit targets / edge flips mudam o mix de saídas.
  4. Sensibilidade ao desconto: `--discount-sweep` — não é uma pergunta em
     aberto, é a demonstração de que P&L/Sharpe são função do parâmetro.

Dados utilizados:
  - Spot BTC/ETH: CoinGecko API — diário (1d), até 365 dias
  - Volatilidade implícita: Deribit DVOL — diário (1d), até 365 dias
  - Mercados: markets_all_*.parquet mais ANTIGO em data/raw/markets/ (por
    mtime, não ordem alfabética — P2-34) → mercados BTC/ETH com resultado
    conhecido (outcomePrices)

Janela de simulação: depende do que sobrar em disco. A retenção de
db_maintenance.py (P2-37) apagava markets_all_* mais velho que 14 dias antes
de excluir esse padrão do escopo dela (2026-09-04) — o histórico original de
~abril a dezembro de 2025 foi perdido nesse incidente e não há como recuperá-lo
localmente. Na data do incidente, o snapshot mais antigo restante em disco era
de ~21 de agosto de 2026 — checar `data/raw/markets/` pra saber o que há hoje.
Passo: 1 dia (limitado pela granularidade do DVOL).

Uso:
  uv run python backtest/sim_backtest.py
  uv run python backtest/sim_backtest.py --days 200 --min-conviction 0.08
  uv run python backtest/sim_backtest.py --html
  uv run python backtest/sim_backtest.py --discount-sweep
"""

import json
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import click
import numpy as np
import pandas as pd
import requests
from loguru import logger
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).parent.parent / "pipeline"))
sys.path.insert(0, str(Path(__file__).parent.parent / "risk"))
from deribit_collector import (
    ASSET_DRIFT,
    MAX_MONEYNESS_SIGMA,
    MAX_YES_FOR_BUY_NO,
    bs_prob,
    parse_crypto_market,
)
# Constantes de risco vêm do risk_manager — a simulação deve usar OS MESMOS
# parâmetros da produção, senão o backtest mede outra estratégia (bug da auditoria:
# valores duplicados aqui divergiam, ex: MAX_POSITION_PCT 0.10 vs 0.03 real).
from risk_manager import (
    EARLY_EXIT as _EARLY_EXIT,
    KELLY_MAX_FRAC,
    kelly_size as _kelly_size_shared,
)

RESULTS_DIR   = Path("backtest/results")
MARKETS_DIR   = Path("data/raw/markets")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

PROFIT_TARGET_VALUE    = float(_EARLY_EXIT["value"]["profit_target_mult"])      # 2.5
PROFIT_TARGET_MOMENTUM = float(_EARLY_EXIT["momentum"]["profit_target_mult"])   # 1.5
EDGE_FLIP_DELTA        = float(_EARLY_EXIT["value"]["edge_flip_delta"])         # 0.25 (era 0.10 duplicado)
KELLY_FRACTION         = KELLY_MAX_FRAC                                          # 0.25


# ──────────────────────────────────────────────────────────
# Data fetching
# ──────────────────────────────────────────────────────────

def _coingecko_get(coin_id: str, days: int) -> list[list]:
    """Retorna preços diários de CoinGecko: [[ts_ms, price], ...]"""
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    params = {"vs_currency": "usd", "days": days}
    for attempt in range(4):
        try:
            r = requests.get(url, params=params, timeout=20)
            r.raise_for_status()
            data = r.json()
            return data.get("prices", [])
        except requests.RequestException as e:
            wait = [5, 15, 30][min(attempt, 2)]  # 5s, 15s, 30s
            if attempt < 3:
                logger.debug(f"CoinGecko rate limit/error, aguardando {wait}s...")
                time.sleep(wait)
            else:
                raise e
    return []


def fetch_spot_daily(asset: str, days: int = 365) -> pd.Series:
    """
    Retorna série temporal diária do preço spot (close) para BTC ou ETH.
    Index: datetime UTC, Values: preço USD.
    """
    coin_map = {"BTC": "bitcoin", "ETH": "ethereum"}
    coin_id = coin_map.get(asset.upper())
    if not coin_id:
        raise ValueError(f"Asset não suportado: {asset}")

    logger.info(f"Buscando spot {asset} (CoinGecko, {days}d)...")
    raw = _coingecko_get(coin_id, days)
    if not raw:
        raise RuntimeError(f"Sem dados de spot para {asset}")

    ts = [datetime.fromtimestamp(p[0] / 1000, tz=timezone.utc) for p in raw]
    prices = [p[1] for p in raw]
    s = pd.Series(prices, index=ts, name=f"{asset}_spot")
    # Normaliza para horário 00:00 UTC (dados diários já vêm assim)
    s.index = s.index.normalize()
    return s.sort_index()


def fetch_dvol_daily(asset: str, days: int = 365) -> pd.Series:
    """
    Retorna série temporal diária do Deribit DVOL (IV implícita, em fração).
    Index: datetime UTC, Values: IV anualizada (ex: 0.65 = 65%).
    """
    end_ts   = int(time.time() * 1000)
    start_ts = end_ts - days * 86400 * 1000

    url = "https://www.deribit.com/api/v2/public/get_volatility_index_data"
    params = {
        "currency":        asset.upper(),
        "start_timestamp": start_ts,
        "end_timestamp":   end_ts,
        "resolution":      86400,
    }
    logger.info(f"Buscando DVOL {asset} (Deribit, {days}d)...")
    try:
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        data = r.json().get("result", {}).get("data", [])
    except requests.RequestException as e:
        raise RuntimeError(f"Falha ao buscar DVOL {asset}: {e}")

    if not data:
        raise RuntimeError(f"Sem dados de DVOL para {asset}")

    ts  = [datetime.fromtimestamp(d[0] / 1000, tz=timezone.utc) for d in data]
    ivs = [d[4] / 100.0 for d in data]  # close, converte % → fração
    s   = pd.Series(ivs, index=ts, name=f"{asset}_dvol")
    s.index = s.index.normalize()
    return s.sort_index()


def load_sim_markets(markets_dir: Path = MARKETS_DIR) -> pd.DataFrame:
    """
    Carrega mercados BTC/ETH parseáveis dos arquivos parquet brutos da Gamma API.
    Retorna DataFrame com colunas:
      market_id, question, asset, strike, expiry, direction, is_touch,
      end_date, created_at, resolved_yes (0/1/None), yes_price_current

    P2-34: usa o parquet MAIS VELHO disponível (não o mais recente) — este é
    o único consumidor do sistema que quer a janela histórica mais ampla
    possível pra simulação walk-forward, ao contrário de
    load_current_markets()/dashboard, que sempre querem o mais recente. Por
    isso markets_all_*.parquet fica de fora da retenção automática do
    db_maintenance.py (achado em 2026-09-04: uma rodada de retenção sem essa
    exceção apagou a janela histórica original sem arquivamento equivalente).
    """
    raw_files = sorted(
        markets_dir.glob("markets_all_*.parquet"), key=lambda p: p.stat().st_mtime,
    )
    raw_files_with_enddate = []
    for f in raw_files:
        try:
            df = pd.read_parquet(f)
            if "endDate" in df.columns:
                raw_files_with_enddate.append(f)
        except Exception:
            continue

    if not raw_files_with_enddate:
        raise RuntimeError(f"Nenhum parquet com metadados completos em {markets_dir}")

    df = pd.read_parquet(raw_files_with_enddate[0])

    # Filtra BTC/ETH com barrier keywords
    mask = (
        df["question"].str.lower().str.contains(r"btc|bitcoin|eth|ethereum", na=False, regex=True) &
        df["question"].str.lower().str.contains(r"above|below|reach|hit|dip", na=False, regex=True)
    )
    df = df[mask].copy()

    rows = []
    for _, row in df.iterrows():
        parsed = parse_crypto_market(row["question"])
        if parsed is None:
            continue

        # Usa endDate da API como data autoritativa (mais confiável que o parser)
        end_date_str = row.get("endDate") or ""
        try:
            end_date = datetime.fromisoformat(
                end_date_str.replace("Z", "+00:00")
            )
        except (ValueError, AttributeError):
            end_date = parsed["expiry"]

        # P2-34: startDate/createdAt só existem em snapshots coletados depois
        # do fix (2026-09-04) — None aqui é "não sei quando abriu", não "abriu
        # antes de qualquer coisa". run() trata os dois casos diferente.
        created_at_str = row.get("startDate") or row.get("createdAt") or ""
        try:
            created_at = datetime.fromisoformat(created_at_str.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            created_at = None

        # Resolução: outcomePrices[0] ≈ 1 → YES, ≈ 0 → NO
        resolved_yes = None
        prices_raw = row.get("outcomePrices")
        if prices_raw:
            try:
                if isinstance(prices_raw, str):
                    prices_list = json.loads(prices_raw)
                else:
                    prices_list = prices_raw
                yes_val = float(prices_list[0])
                resolved_yes = 1 if yes_val >= 0.5 else 0
            except (ValueError, TypeError, IndexError):
                pass

        rows.append({
            "market_id":       str(row.get("conditionId", row.get("id", ""))),
            "question":        row["question"],
            "asset":           parsed["asset"],
            "strike":          parsed["strike"],
            "expiry":          parsed["expiry"],   # parsed (pode ter ano errado)
            "end_date":        end_date,            # da API (autoritativo)
            "created_at":      created_at,           # None se o snapshot for anterior ao P2-34
            "direction":       parsed["direction"],
            "is_touch":        parsed["is_touch"],
            "resolved_yes":    resolved_yes,
            "yes_price":       row.get("yes_price"),
            "is_active":       row.get("active", False),
            "is_closed":       row.get("closed", True),
        })

    result = pd.DataFrame(rows)
    logger.info(f"Mercados carregados: {len(result)} ({result['resolved_yes'].notna().sum()} com resolução conhecida)")
    return result


# ──────────────────────────────────────────────────────────
# Posição simulada
# ──────────────────────────────────────────────────────────

@dataclass
class SimPosition:
    market_id:    str
    question:     str
    asset:        str
    strike:       float
    direction:    str        # "above" ou "below"
    is_touch:     bool
    side:         str        # "BUY_YES" ou "BUY_NO"
    entry_ts:     datetime
    entry_prob:   float      # BS_prob no momento de entrada
    entry_price:  float      # preço da ação comprada (YES ou NO)
    cost_usdc:    float      # capital investido
    shares:       float      # = cost_usdc / entry_price
    end_date:     datetime   # resolução oficial do mercado
    resolved_yes: Optional[int]  # 0, 1 ou None (ainda aberto)

    # Preenchido ao fechar
    exit_ts:      Optional[datetime] = None
    exit_price:   Optional[float]    = None
    pnl_usdc:     Optional[float]    = None
    exit_reason:  Optional[str]      = None  # "profit_target", "edge_flip", "resolved", "expired"
    status:       str                = "open"


# ──────────────────────────────────────────────────────────
# Motor da simulação
# ──────────────────────────────────────────────────────────

class WalkForwardSimulator:
    """
    Simula o modelo Deribit/B-S em dados históricos passo a passo (diário).

    Preços sintéticos: BS_prob é usado como proxy para o yes_price de mercado.
    Isso remove o "edge" de mispricing, mas permite testar:
      - Calibração: win_rate vs. convicção prevista
      - Kelly sizing: adequação do tamanho das posições
      - Early exits: profit targets e edge flips melhoram Sharpe?
      - Timing: o modelo entra em momentos favoráveis?
    """

    def __init__(
        self,
        sim_start:       datetime,
        sim_end:         datetime,
        initial_capital: float = 1000.0,
        min_conviction:  float = 0.08,
        kelly_fraction:  float = KELLY_FRACTION,
        profit_mult:     float = PROFIT_TARGET_VALUE,
        edge_flip_delta: float = EDGE_FLIP_DELTA,
        market_discount: float = 0.20,
    ):
        self.sim_start       = sim_start
        self.sim_end         = sim_end
        self.initial_capital = initial_capital
        self.cash            = initial_capital
        self.min_conviction  = min_conviction
        self.kelly_fraction  = kelly_fraction
        self.profit_mult     = profit_mult
        self.edge_flip_delta = edge_flip_delta
        # Desconto sintético: market_price = prob × (1 − market_discount)
        # Simula a ineficiência histórica do Polymarket vs. BS_prob com drift.
        self.market_discount = market_discount

        self.open_positions:   list[SimPosition] = []
        self.closed_positions: list[SimPosition] = []
        self.pnl_curve:        list[dict]        = []
        # Mercados que já foram negociados (sem re-entrada)
        self._traded_markets: set[str]           = set()

        # Dados históricos (preenchidos em run())
        self.spot:   dict[str, pd.Series] = {}   # asset → pd.Series
        self.dvol:   dict[str, pd.Series] = {}   # asset → pd.Series
        self.markets: pd.DataFrame        = pd.DataFrame()

    # ── Helpers ───────────────────────────────────────────

    @staticmethod
    def _is_market_open_at(mkt: "pd.Series", ts: datetime) -> bool:
        """
        P2-34: sem isso, mercado criado em novembro é negociável desde abril
        — look-ahead duro, condiciona no futuro. mkt["created_at"] é None em
        snapshots coletados antes do P2-34 (2026-09-04) — não bloqueia
        (não dá pra saber), em vez de vetar a simulação inteira até um
        snapshot novo com startDate/createdAt ser coletado.
        """
        created_at = mkt.get("created_at")
        if created_at is None or pd.isna(created_at):
            return True
        return bool(ts >= created_at)

    def _mark_open_positions(self, ts: datetime) -> float:
        """
        P2-35: marcar posições abertas a custo (cost_usdc fixo) deixa a
        curva de P&L constante por partes, só pulando nas saídas —
        subestima a volatilidade diária e infla o Sharpe por construção (o
        max_dd calculado em cima também ignora toda excursão não
        realizada). Reusa _compute_prob, já chamado no mesmo loop pra
        fechamentos forçados no fim da simulação.
        """
        total = 0.0
        for p in self.open_positions:
            prob = self._compute_prob(p.asset, p.strike, p.direction == "above", p.is_touch, ts, p.end_date)
            if prob is not None:
                fair = prob if p.side == "BUY_YES" else (1.0 - prob)
                mtm_price = max(0.01, fair * (1.0 - self.market_discount))
            else:
                mtm_price = p.entry_price  # sem spot/IV disponível em ts — mantém ao custo
            total += p.shares * mtm_price
        return total

    def _get_spot(self, asset: str, ts: datetime) -> Optional[float]:
        """Spot price mais recente disponível em ts."""
        s = self.spot.get(asset)
        if s is None or s.empty:
            return None
        day = ts.normalize() if hasattr(ts, "normalize") else ts.replace(hour=0, minute=0, second=0, microsecond=0)
        available = s[s.index <= day]
        return float(available.iloc[-1]) if not available.empty else None

    def _get_iv(self, asset: str, ts: datetime) -> Optional[float]:
        """IV mais recente disponível em ts."""
        s = self.dvol.get(asset)
        if s is None or s.empty:
            return None
        day = ts.normalize() if hasattr(ts, "normalize") else ts.replace(hour=0, minute=0, second=0, microsecond=0)
        available = s[s.index <= day]
        return float(available.iloc[-1]) if not available.empty else None

    def _years_to_expiry(self, end_date: datetime, current_ts: datetime) -> float:
        """Anos até expiração (mínimo 0)."""
        delta = (end_date - current_ts).total_seconds()
        return max(0.0, delta / (365.25 * 86400))

    def _compute_prob(
        self, asset: str, strike: float, above: bool, is_touch: bool,
        ts: datetime, end_date: datetime
    ) -> Optional[float]:
        """Computa BS_prob para um mercado no momento ts."""
        S     = self._get_spot(asset, ts)
        sigma = self._get_iv(asset, ts)
        T     = self._years_to_expiry(end_date, ts)

        if S is None or sigma is None or T <= 0:
            return None

        # Filtro de moneyness
        sigma_sqrtT = sigma * np.sqrt(T)
        moneyness   = abs(S - strike) / strike
        if moneyness > MAX_MONEYNESS_SIGMA * sigma_sqrtT:
            return None

        drift = ASSET_DRIFT.get(asset, 0.0)
        return bs_prob(S, strike, T, sigma, r=drift, above=above, touch=is_touch)

    def _kelly_size(self, entry_price: float, prob: float) -> float:
        """
        P2-33: delega para risk_manager.kelly_size — a mesma função que a
        produção usa. A versão anterior reimplementava a fórmula aqui (de
        forma correta, ao contrário da produção pré-P1-9) e mais nada do
        resto do stack de risco (MIN_EDGE_TO_TRADE, MIN_CONFIDENCE) — dois
        caminhos que podiam divergir silenciosamente no arquivo que promete
        paridade com produção no cabeçalho.

        prob é a probabilidade de win do lado comprado (fair_price no call
        site); edge = prob - entry_price segue a mesma convenção do
        kelly_size. self.kelly_fraction fica sem efeito aqui — kelly_size já
        aplica KELLY_MAX_FRAC internamente, e nada no CLI expõe um
        --kelly-fraction que precise de override por instância.
        """
        edge = prob - entry_price
        return _kelly_size_shared(
            edge=edge,
            entry_price=entry_price,
            capital=self.cash,
            confidence=1.0,
            signal_source="deribit",
            trade_type="value",
        )

    def _check_touch_resolution(
        self, pos: SimPosition, ts: datetime
    ) -> bool:
        """
        Para mercados TOUCH: verifica se o spot atual cruzou a barreira.
        Retorna True se o mercado deve ser resolvido como YES pelo toque.
        """
        if not pos.is_touch:
            return False
        S = self._get_spot(pos.asset, ts)
        if S is None:
            return False
        if pos.direction == "above" and S >= pos.strike:
            return True
        if pos.direction == "below" and S <= pos.strike:
            return True
        return False

    # ── Abertura de posições ──────────────────────────────

    def _try_open(self, market: pd.Series, ts: datetime) -> Optional[SimPosition]:
        """
        Tenta abrir posição para um mercado. Retorna SimPosition ou None.
        """
        asset     = market["asset"]
        strike    = market["strike"]
        above     = market["direction"] == "above"
        is_touch  = market["is_touch"]
        end_date  = market["end_date"]

        # Não abre se muito próximo da expiração
        T_days = self._years_to_expiry(end_date, ts) * 365.25
        if T_days < 7 or T_days > 400:
            return None

        prob = self._compute_prob(asset, strike, above, is_touch, ts, end_date)
        if prob is None:
            return None

        # Convicção: distância de 0.5
        conviction = abs(prob - 0.5)
        if conviction < self.min_conviction:
            return None

        # Direção da posição e preço sintético de mercado
        # market_price = prob × (1 − discount), simulando subprecificação
        if prob >= 0.5 + self.min_conviction:
            side        = "BUY_YES"
            fair_price  = prob
            entry_price = max(0.02, fair_price * (1.0 - self.market_discount))
        else:
            side        = "BUY_NO"
            fair_price  = 1.0 - prob
            entry_price = max(0.02, fair_price * (1.0 - self.market_discount))
            # Limita BUY_NO quando mercado já precifica YES muito alto
            if (1.0 - fair_price) > MAX_YES_FOR_BUY_NO:
                return None

        # Kelly: win_prob = fair_price (model's true probability), b = upside odds
        size_usdc = self._kelly_size(entry_price, fair_price)
        if size_usdc <= 0:
            return None
        if size_usdc > self.cash:
            return None

        # Marca mercado como negociado (sem re-entrada)
        mkt_id = str(market["market_id"])
        self._traded_markets.add(mkt_id)

        self.cash -= size_usdc
        return SimPosition(
            market_id    = str(market["market_id"]),
            question     = str(market["question"]),
            asset        = asset,
            strike       = strike,
            direction    = market["direction"],
            is_touch     = is_touch,
            side         = side,
            entry_ts     = ts,
            entry_prob   = prob,
            entry_price  = entry_price,
            cost_usdc    = size_usdc,
            shares       = size_usdc / entry_price,
            end_date     = end_date,
            resolved_yes = market.get("resolved_yes"),
        )

    # ── Fechamento de posições ────────────────────────────

    def _close_position(
        self, pos: SimPosition, ts: datetime,
        exit_price: float, reason: str
    ) -> None:
        """Fecha posição, computa P&L, move para closed_positions."""
        pnl = pos.shares * exit_price - pos.cost_usdc
        pos.exit_ts     = ts
        pos.exit_price  = exit_price
        pos.pnl_usdc    = pnl
        pos.exit_reason = reason
        pos.status      = "closed"
        self.cash      += pos.cost_usdc + pnl
        self.open_positions.remove(pos)
        self.closed_positions.append(pos)

    def _check_exits(self, pos: SimPosition, ts: datetime) -> bool:
        """
        Verifica condições de saída para uma posição aberta.
        Retorna True se a posição foi fechada.
        """
        # Resolução por toque (touch markets)
        if pos.is_touch and self._check_touch_resolution(pos, ts):
            exit_price = 1.0 if pos.side == "BUY_YES" else 0.0
            self._close_position(pos, ts, exit_price, "resolved_touch")
            return True

        # Resolução na data de expiração
        if ts >= pos.end_date:
            # Usa resultado real se disponível, senão usa BS_prob final
            if pos.resolved_yes is not None:
                won = (pos.resolved_yes == 1 and pos.side == "BUY_YES") or \
                      (pos.resolved_yes == 0 and pos.side == "BUY_NO")
                exit_price = 1.0 if won else 0.0
            else:
                # Mercado sem resultado conhecido: usa BS_prob atual como MtM
                prob = self._compute_prob(
                    pos.asset, pos.strike,
                    pos.direction == "above", pos.is_touch,
                    ts, pos.end_date
                )
                if prob is not None:
                    exit_price = prob if pos.side == "BUY_YES" else (1.0 - prob)
                else:
                    exit_price = pos.entry_price
            self._close_position(pos, ts, exit_price, "resolved")
            return True

        # Profit target e edge flip (usa BS_prob atual com desconto)
        prob = self._compute_prob(
            pos.asset, pos.strike,
            pos.direction == "above", pos.is_touch,
            ts, pos.end_date
        )
        if prob is None:
            return False

        # Preço de saída = BS_prob atual × (1 - desconto) — mesmo desconto da entrada
        fair_current = prob if pos.side == "BUY_YES" else (1.0 - prob)
        current_price = max(0.01, fair_current * (1.0 - self.market_discount))

        # Profit target: quando a posição valoriza >= profit_mult × entry
        if current_price >= pos.entry_price * self.profit_mult:
            self._close_position(pos, ts, current_price, "profit_target")
            return True

        # Edge flip: quando a probabilidade cruza o limiar oposto
        if pos.side == "BUY_YES" and prob < (pos.entry_prob - self.edge_flip_delta):
            self._close_position(pos, ts, current_price, "edge_flip")
            return True
        if pos.side == "BUY_NO" and prob > (pos.entry_prob + self.edge_flip_delta):
            self._close_position(pos, ts, current_price, "edge_flip")
            return True

        return False

    # ── Loop principal ────────────────────────────────────

    def run(self, markets: pd.DataFrame, verbose: bool = True) -> dict:
        """
        Executa a simulação walk-forward.

        Args:
            markets: DataFrame de mercados (de load_sim_markets())
            verbose: exibe progresso no terminal

        Returns:
            dict com P&L curve, posições, métricas.
        """
        self.markets = markets

        # Identifica ativos necessários
        assets = markets["asset"].unique().tolist()

        # Busca dados históricos — pula ativo já presente em self.spot/self.dvol
        # (run_discount_sweep pré-popula os dois pra não refazer 1 fetch de
        # CoinGecko/Deribit por desconto testado, quando um basta).
        days = max(366, int((self.sim_end - self.sim_start).days) + 30)
        for asset in assets:
            if asset in self.spot and asset in self.dvol:
                continue
            try:
                self.spot[asset] = fetch_spot_daily(asset, days=min(days, 365))
                self.dvol[asset] = fetch_dvol_daily(asset, days=min(days, 365))
            except RuntimeError as e:
                logger.warning(f"Dados {asset} indisponíveis: {e}")

        # Remove mercados sem dados disponíveis
        markets = markets[markets["asset"].isin(self.spot.keys())].copy()
        logger.info(f"Simulando {len(markets)} mercados de {self.sim_start.date()} a {self.sim_end.date()}")

        # Gera sequência de datas diárias
        day_range = pd.date_range(
            start=self.sim_start,
            end=self.sim_end,
            freq="1D",
            tz=timezone.utc,
        )

        # Set de market_ids já com posição aberta
        open_ids = set()

        for ts in day_range:
            # 1. Verifica early exits e resoluções
            for pos in list(self.open_positions):
                self._check_exits(pos, ts)
                # Atualiza open_ids
            open_ids = {p.market_id for p in self.open_positions}

            # 2. Tenta abrir novas posições (sem re-entrada em mercados já negociados)
            for _, mkt in markets.iterrows():
                mkt_id = str(mkt["market_id"])
                if mkt_id in open_ids:
                    continue
                if mkt_id in self._traded_markets:
                    continue   # sem re-entrada após fechar
                if ts >= mkt["end_date"]:
                    continue
                if not self._is_market_open_at(mkt, ts):
                    continue

                pos = self._try_open(mkt, ts)
                if pos is not None:
                    self.open_positions.append(pos)
                    open_ids.add(pos.market_id)
                    if verbose:
                        logger.info(
                            f"  [{pos.side}] {pos.question[:55]} "
                            f"@ {pos.entry_price:.3f} ${pos.cost_usdc:.2f} "
                            f"(prob={pos.entry_prob:.3f})"
                        )

            # 3. Registra ponto na curva de P&L — marcado a mercado, não a
            # custo (P2-35, ver _mark_open_positions).
            open_value = self._mark_open_positions(ts)
            self.pnl_curve.append({
                "date":          ts.date(),
                "cash":          round(self.cash, 2),
                "open_value":    round(open_value, 2),
                "portfolio":     round(self.cash + open_value, 2),
                "n_open":        len(self.open_positions),
                "n_closed":      len(self.closed_positions),
            })

        # 3. Força fechamento das posições ainda abertas ao final
        for pos in list(self.open_positions):
            prob = self._compute_prob(
                pos.asset, pos.strike,
                pos.direction == "above", pos.is_touch,
                self.sim_end, pos.end_date
            )
            if prob is not None:
                fair = prob if pos.side == "BUY_YES" else (1.0 - prob)
                exit_price = max(0.01, fair * (1.0 - self.market_discount))
            else:
                exit_price = pos.entry_price
            self._close_position(pos, self.sim_end, exit_price, "sim_end")

        return self._summary()

    # ── Métricas ──────────────────────────────────────────

    def _summary(self) -> dict:
        """Compila métricas da simulação."""
        closed = self.closed_positions
        n      = len(closed)
        wins   = sum(1 for p in closed if p.pnl_usdc and p.pnl_usdc > 0)
        total_pnl = sum(p.pnl_usdc or 0 for p in closed)
        final_portfolio = self.cash

        # Sharpe (usando variância dos retornos diários de P&L)
        curve_df = pd.DataFrame(self.pnl_curve)
        sharpe   = float("nan")
        max_dd   = 0.0
        if not curve_df.empty:
            daily_ret = curve_df["portfolio"].pct_change().dropna()
            if len(daily_ret) > 5 and daily_ret.std() > 0:
                sharpe = float(daily_ret.mean() / daily_ret.std() * np.sqrt(252))
            # Max drawdown
            port = curve_df["portfolio"].values
            peak = np.maximum.accumulate(port)
            dd   = (port - peak) / peak
            max_dd = float(dd.min())

        exit_reasons = {}
        for p in closed:
            r = p.exit_reason or "?"
            exit_reasons[r] = exit_reasons.get(r, 0) + 1

        # Calibração: agrupa por faixa de BS_prob na entrada vs resultado real
        calibration = []
        closed_with_outcome = [p for p in closed if p.resolved_yes is not None]
        if closed_with_outcome:
            bins = [(0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.01)]
            for lo, hi in bins:
                # Para BUY_YES: prob_entry; para BUY_NO: 1-prob_entry
                bucket = [
                    p for p in closed_with_outcome
                    if lo <= (p.entry_prob if p.side == "BUY_YES" else 1 - p.entry_prob) < hi
                ]
                if bucket:
                    n_b = len(bucket)
                    wins_b = sum(
                        1 for p in bucket
                        if (p.resolved_yes == 1 and p.side == "BUY_YES") or
                           (p.resolved_yes == 0 and p.side == "BUY_NO")
                    )
                    avg_prob = sum(
                        p.entry_prob if p.side == "BUY_YES" else 1 - p.entry_prob
                        for p in bucket
                    ) / n_b
                    calibration.append({
                        "prob_bin":   f"{lo:.0%}–{hi:.0%}",
                        "n":          n_b,
                        "avg_prob":   round(avg_prob, 3),
                        "actual_wr":  round(wins_b / n_b, 3),
                        "calib_err":  round(wins_b / n_b - avg_prob, 3),
                    })

        # P2-32: Brier score e erro de calibração médio, não os bins por faixa
        # — isso é o que este simulador consegue de fato falsear (o desconto
        # de mercado não entra em nenhum dos dois). market_discount desloca
        # entry_price, não entry_prob nem resolved_yes.
        brier_score = None
        mean_abs_calib_err = None
        if closed_with_outcome:
            brier_terms = []
            for p in closed_with_outcome:
                model_prob = p.entry_prob if p.side == "BUY_YES" else 1 - p.entry_prob
                won = 1 if (
                    (p.resolved_yes == 1 and p.side == "BUY_YES") or
                    (p.resolved_yes == 0 and p.side == "BUY_NO")
                ) else 0
                brier_terms.append((model_prob - won) ** 2)
            brier_score = round(sum(brier_terms) / len(brier_terms), 4)
        if calibration:
            total_n = sum(b["n"] for b in calibration)
            mean_abs_calib_err = round(
                sum(abs(b["calib_err"]) * b["n"] for b in calibration) / total_n, 4
            )

        return {
            "sim_start":        self.sim_start.date().isoformat(),
            "sim_end":          self.sim_end.date().isoformat(),
            "initial_capital":  self.initial_capital,
            "market_discount":  self.market_discount,
            "final_portfolio":  round(final_portfolio, 2),
            "total_pnl":        round(total_pnl, 2),
            "total_return_pct": round((final_portfolio - self.initial_capital) / self.initial_capital * 100, 2),
            "n_trades":         n,
            "n_wins":           wins,
            "win_rate":         round(wins / n, 3) if n > 0 else None,
            "sharpe":           round(sharpe, 3) if not np.isnan(sharpe) else None,
            "max_drawdown_pct": round(max_dd * 100, 2),
            "exit_reasons":     exit_reasons,
            "calibration":      calibration,
            "brier_score":      brier_score,
            "mean_abs_calib_err": mean_abs_calib_err,
            "positions":        closed,
            "pnl_curve":        pd.DataFrame(self.pnl_curve),
        }


# ──────────────────────────────────────────────────────────
# Relatório terminal
# ──────────────────────────────────────────────────────────

def print_sim_summary(result: dict) -> None:
    from rich.console import Console
    from rich.table import Table
    from rich import box

    console = Console()
    console.print()
    console.print("[bold]── Simulation Summary (Walk-Forward) ───────────────[/bold]")
    console.print(f"  Período: {result['sim_start']} → {result['sim_end']}")
    console.print(f"  Capital inicial:  ${result['initial_capital']:,.2f}")

    n = result["n_trades"]
    console.print(f"  Trades fechados:  {n}")

    # P2-32: calibração primeiro — é a única coisa que este simulador pode
    # de fato falsear. market_discount não entra em brier_score/mean_abs_calib_err.
    console.print()
    console.print("[bold]── Calibração (o que este simulador mede de verdade) ─[/bold]")
    brier = result.get("brier_score")
    mace  = result.get("mean_abs_calib_err")
    if brier is not None:
        b_color = "green" if brier < 0.20 else "yellow" if brier < 0.25 else "red"
        console.print(f"  Brier score:     [{b_color}]{brier:.4f}[/{b_color}]  (0=perfeito, 0.25=chute em 50%)")
    if mace is not None:
        m_color = "green" if mace < 0.10 else "yellow" if mace < 0.20 else "red"
        console.print(f"  Erro calib. médio: [{m_color}]{mace:+.1%}[/{m_color}]  (ponderado por N, por faixa)")
    if brier is None and mace is None:
        console.print("  [dim]Sem trades com resolução conhecida.[/dim]")

    cal = result.get("calibration", [])
    if cal:
        console.print()
        cal_tbl = Table(box=box.SIMPLE, show_header=True, header_style="bold cyan")
        cal_tbl.add_column("Faixa Prob", width=12)
        cal_tbl.add_column("N",          justify="right", width=6)
        cal_tbl.add_column("BS_prob",    justify="right", width=9)
        cal_tbl.add_column("Win Rate",   justify="right", width=10)
        cal_tbl.add_column("Erro Calib", justify="right", width=11)
        for row in cal:
            err = row["calib_err"]
            err_color = "green" if abs(err) < 0.10 else "yellow" if abs(err) < 0.20 else "red"
            cal_tbl.add_row(
                row["prob_bin"],
                str(row["n"]),
                f"{row['avg_prob']:.1%}",
                f"{row['actual_wr']:.1%}",
                f"[{err_color}]{err:+.1%}[/{err_color}]",
            )
        console.print(cal_tbl)
        console.print(
            "  [dim]Erro calib. = win_rate_real − BS_prob_prevista. "
            "Próximo de 0% = modelo bem calibrado.[/dim]"
        )

    # P&L sintético — depende de market_discount, não é edge medido
    discount = result.get("market_discount", 0)
    console.print()
    console.print(f"[bold yellow]── P&L sintético (desconto assumido = {discount:.0%}) ──[/bold yellow]")
    console.print(
        "  [dim italic]market_discount é um parâmetro escolhido, não medido — o P&L abaixo "
        "é monotônico nele por construção (rode --discount-sweep para ver). "
        "Não é uma previsão de retorno real.[/dim italic]"
    )

    port = result["final_portfolio"]
    ret  = result["total_return_pct"]
    pnl  = result["total_pnl"]
    color = "green" if ret >= 0 else "red"
    console.print(f"  Portfólio final:  ${port:,.2f}")
    console.print(f"  P&L realizado:   [{color}]${pnl:+.2f}[/{color}]")
    console.print(f"  Retorno:         [{color}]{ret:+.2f}%[/{color}]")

    wr = result.get("win_rate")
    if wr is not None:
        wr_color = "green" if wr >= 0.5 else "yellow"
        console.print(f"  Win rate:        [{wr_color}]{wr:.1%}[/{wr_color}]  ({result['n_wins']}/{n})")
    if result["sharpe"] is not None:
        sh_color = "green" if result["sharpe"] >= 1 else "yellow"
        console.print(f"  Sharpe (anual.): [{sh_color}]{result['sharpe']:.2f}[/{sh_color}]")
    if result["max_drawdown_pct"] < 0:
        console.print(f"  Max drawdown:    [red]{result['max_drawdown_pct']:.2f}%[/red]")

    # Exit reasons
    reasons = result.get("exit_reasons", {})
    if reasons:
        console.print()
        console.print("  Motivos de saída:")
        for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
            console.print(f"    {reason:20s}: {count}")

    # Trades table
    positions = result.get("positions", [])
    if positions:
        console.print()
        tbl = Table(
            box=box.SIMPLE, show_header=True, header_style="bold cyan",
            title="Trades da Simulação"
        )
        tbl.add_column("Side",     width=9)
        tbl.add_column("Mercado",  width=52)
        tbl.add_column("Asset",    width=5)
        tbl.add_column("Strike",   justify="right", width=10)
        tbl.add_column("Entry",    justify="right", width=7)
        tbl.add_column("Exit",     justify="right", width=7)
        tbl.add_column("Cost $",   justify="right", width=8)
        tbl.add_column("P&L $",    justify="right", width=9)
        tbl.add_column("Reason",   width=14)

        for pos in sorted(positions, key=lambda p: p.entry_ts):
            pnl = pos.pnl_usdc or 0
            pnl_color = "green" if pnl >= 0 else "red"
            tbl.add_row(
                pos.side,
                pos.question[:50],
                pos.asset,
                f"${pos.strike:,.0f}",
                f"{pos.entry_price:.3f}",
                f"{pos.exit_price:.3f}" if pos.exit_price is not None else "—",
                f"${pos.cost_usdc:.2f}",
                f"[{pnl_color}]${pnl:+.2f}[/{pnl_color}]",
                pos.exit_reason or "?",
            )
        console.print(tbl)


# ──────────────────────────────────────────────────────────
# Relatório HTML
# ──────────────────────────────────────────────────────────

def generate_html_report(result: dict, output_path: Optional[Path] = None) -> Path:
    try:
        import plotly.graph_objects as go
        import plotly.io as pio
        HAS_PLOTLY = True
    except ImportError:
        HAS_PLOTLY = False

    curve = result["pnl_curve"]
    positions = result.get("positions", [])
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    charts_html = ""
    if HAS_PLOTLY and not curve.empty:
        baseline = result["initial_capital"]
        pnl_pct  = (curve["portfolio"] / baseline - 1) * 100

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=curve["date"].astype(str),
            y=pnl_pct.round(2),
            mode="lines",
            name="Retorno acumulado (%)",
            line=dict(color="#00d4aa", width=2),
            fill="tozeroy",
            fillcolor="rgba(0,212,170,0.12)",
        ))
        fig.add_hline(y=0, line_dash="dot", line_color="#555", line_width=1)
        fig.update_layout(
            title="Retorno Acumulado — Walk-Forward Simulation",
            xaxis_title="Data", yaxis_title="Retorno (%)",
            template="plotly_dark", height=350,
            margin=dict(l=40, r=20, t=50, b=40),
        )
        charts_html += pio.to_html(fig, full_html=False, include_plotlyjs=False)

    # Trades table
    if positions:
        rows = ""
        for pos in sorted(positions, key=lambda p: p.entry_ts):
            pnl   = pos.pnl_usdc or 0
            color = "#00d4aa" if pnl >= 0 else "#ff6b6b"
            rows += f"""<tr>
              <td>{pos.side}</td>
              <td title="{pos.question}">{pos.question[:50]}</td>
              <td>{pos.asset}</td>
              <td>${pos.strike:,.0f}</td>
              <td>{pos.entry_price:.3f}</td>
              <td>{f"{pos.exit_price:.3f}" if pos.exit_price is not None else "—"}</td>
              <td>${pos.cost_usdc:.2f}</td>
              <td style="color:{color}">${pnl:+.2f}</td>
              <td>{pos.exit_reason or "?"}</td>
            </tr>"""
        trades_table = f"""<table>
          <thead><tr>
            <th>Side</th><th>Mercado</th><th>Asset</th><th>Strike</th>
            <th>Entry</th><th>Exit</th><th>Cost</th><th>P&L</th><th>Reason</th>
          </tr></thead><tbody>{rows}</tbody></table>"""
    else:
        trades_table = "<p><em>Nenhum trade.</em></p>"

    # Calibration table (P2-32) — o que este simulador consegue de fato falsear
    cal = result.get("calibration", [])
    if cal:
        cal_rows = "".join(
            f"""<tr>
              <td>{row['prob_bin']}</td><td>{row['n']}</td>
              <td>{row['avg_prob']:.1%}</td><td>{row['actual_wr']:.1%}</td>
              <td style="color:{'#00d4aa' if abs(row['calib_err']) < 0.10 else '#ffd93d' if abs(row['calib_err']) < 0.20 else '#ff6b6b'}">{row['calib_err']:+.1%}</td>
            </tr>"""
            for row in cal
        )
        calibration_table = f"""<table>
          <thead><tr>
            <th>Faixa Prob</th><th>N</th><th>BS_prob</th><th>Win Rate</th><th>Erro Calib</th>
          </tr></thead><tbody>{cal_rows}</tbody></table>"""
    else:
        calibration_table = "<p><em>Sem trades com resolução conhecida.</em></p>"

    wr    = result.get("win_rate")
    sh    = result.get("sharpe")
    ret   = result["total_return_pct"]
    pnl   = result["total_pnl"]
    brier = result.get("brier_score")
    mace  = result.get("mean_abs_calib_err")
    discount = result.get("market_discount", 0)
    plotly_cdn = '<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>' if HAS_PLOTLY else ""

    html = f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
  <meta charset="UTF-8">
  <title>Polymarket — Walk-Forward Simulation</title>
  {plotly_cdn}
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ font-family: 'Segoe UI', sans-serif; background: #0d1117; color: #e9ecef; padding: 24px; }}
    h1 {{ font-size: 1.6rem; margin-bottom: 4px; color: #f8f9fa; }}
    h2 {{ font-size: 1.1rem; margin: 24px 0 10px; color: #adb5bd; border-bottom: 1px solid #2d3748; padding-bottom: 6px; }}
    .subtitle {{ color: #6c757d; font-size: 0.85rem; margin-bottom: 12px; }}
    .warn {{ color: #ffd93d; font-size: 0.8rem; background: #1a1600; border: 1px solid #ffd93d33;
             padding: 8px 14px; border-radius: 6px; margin-bottom: 20px; }}
    .warn-strong {{ color: #ffd93d; font-size: 0.85rem; background: #1a1600; border: 1px solid #ffd93d55;
             padding: 10px 16px; border-radius: 6px; margin: 24px 0 14px; font-weight: 600; }}
    .stats-grid {{ display: flex; flex-wrap: wrap; gap: 12px; margin-bottom: 24px; }}
    .stat-card {{ background: #161b22; border: 1px solid #2d3748; border-radius: 8px; padding: 14px 18px; min-width: 130px; }}
    .stat-value {{ font-size: 1.4rem; font-weight: 700; }}
    .stat-label {{ font-size: 0.75rem; color: #6c757d; margin-top: 4px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 0.82rem; margin-bottom: 8px; }}
    th {{ background: #21262d; padding: 8px 12px; text-align: left; color: #adb5bd; font-weight: 600; }}
    td {{ padding: 7px 12px; border-top: 1px solid #21262d; }}
    tr:hover td {{ background: #161b22; }}
  </style>
</head>
<body>
  <h1>Polymarket Quant — Walk-Forward Simulation</h1>
  <p class="subtitle">Gerado em {now_str} &nbsp;|&nbsp; {result['sim_start']} → {result['sim_end']}</p>
  <p class="warn">⚠ Preços sintéticos: BS_prob usada como proxy para yes_price (dados históricos do Polymarket indisponíveis para mercados resolvidos).</p>

  <h2>Calibração (o que este simulador mede de verdade)</h2>
  <div class="stats-grid">
    <div class="stat-card"><div class="stat-value">{result['n_trades']}</div><div class="stat-label">Trades Fechados</div></div>
    <div class="stat-card"><div class="stat-value">{f"{brier:.4f}" if brier is not None else "—"}</div><div class="stat-label">Brier Score (0=perfeito)</div></div>
    <div class="stat-card"><div class="stat-value">{f"{mace:+.1%}" if mace is not None else "—"}</div><div class="stat-label">Erro Calib. Médio</div></div>
  </div>
  {calibration_table}

  <p class="warn-strong">
    ⚠ P&L sintético — desconto assumido = {discount:.0%}. market_discount é um parâmetro
    escolhido, não medido: entry_price = BS_prob × (1 − desconto), então edge = desconto × BS_prob,
    sempre positivo por construção. Os números abaixo são monotônicos nesse parâmetro
    (rode <code>--discount-sweep</code> pra ver) e não são uma previsão de retorno real —
    só a calibração acima é.
  </p>

  <h2>Métricas Gerais (P&L Sintético)</h2>
  <div class="stats-grid">
    <div class="stat-card"><div class="stat-value">${result['initial_capital']:,.0f}</div><div class="stat-label">Capital Inicial</div></div>
    <div class="stat-card"><div class="stat-value">${result['final_portfolio']:,.2f}</div><div class="stat-label">Portfólio Final</div></div>
    <div class="stat-card"><div class="stat-value" style="color:{'#00d4aa' if pnl >= 0 else '#ff6b6b'}">${pnl:+.2f}</div><div class="stat-label">P&L Realizado</div></div>
    <div class="stat-card"><div class="stat-value" style="color:{'#00d4aa' if ret >= 0 else '#ff6b6b'}">{ret:+.2f}%</div><div class="stat-label">Retorno</div></div>
    <div class="stat-card"><div class="stat-value">{f"{wr:.1%}" if wr is not None else "—"}</div><div class="stat-label">Win Rate</div></div>
    <div class="stat-card"><div class="stat-value">{f"{sh:.2f}" if sh is not None else "—"}</div><div class="stat-label">Sharpe (anual.)</div></div>
    <div class="stat-card"><div class="stat-value" style="color:{'#ff6b6b' if result['max_drawdown_pct'] < 0 else '#e9ecef'}">{result['max_drawdown_pct']:.2f}%</div><div class="stat-label">Max Drawdown</div></div>
  </div>

  <h2>Retorno Acumulado</h2>
  {charts_html if HAS_PLOTLY else "<p><em>Instale plotly para ver o gráfico.</em></p>"}

  <h2>Trades ({result['n_trades']})</h2>
  {trades_table}

  <h2>Motivos de Saída</h2>
  <table>
    <thead><tr><th>Motivo</th><th>Trades</th></tr></thead>
    <tbody>
      {"".join(f"<tr><td>{r}</td><td>{c}</td></tr>" for r,c in sorted(result.get('exit_reasons',{}).items(), key=lambda x:-x[1]))}
    </tbody>
  </table>
</body>
</html>"""

    if output_path is None:
        ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = RESULTS_DIR / f"sim_backtest_{ts_str}.html"

    output_path.write_text(html, encoding="utf-8")
    logger.success(f"Relatório HTML salvo: {output_path}")
    return output_path


# ──────────────────────────────────────────────────────────
# CSV export
# ──────────────────────────────────────────────────────────

def save_results(result: dict) -> Path:
    """Salva trades e curva de P&L em CSV."""
    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Trades CSV
    positions = result.get("positions", [])
    if positions:
        trades_df = pd.DataFrame([{
            "entry_ts":     p.entry_ts.date().isoformat(),
            "exit_ts":      p.exit_ts.date().isoformat() if p.exit_ts else "",
            "side":         p.side,
            "question":     p.question,
            "asset":        p.asset,
            "strike":       p.strike,
            "is_touch":     p.is_touch,
            "entry_price":  round(p.entry_price, 4),
            "exit_price":   round(p.exit_price, 4) if p.exit_price is not None else None,
            "cost_usdc":    round(p.cost_usdc, 2),
            "pnl_usdc":     round(p.pnl_usdc, 2) if p.pnl_usdc is not None else None,
            "exit_reason":  p.exit_reason,
            "resolved_yes": p.resolved_yes,
        } for p in positions])
        trades_path = RESULTS_DIR / f"sim_trades_{ts_str}.csv"
        trades_df.to_csv(trades_path, index=False)
        logger.info(f"Trades salvos: {trades_path}")

    # P&L curve CSV
    curve = result.get("pnl_curve")
    if curve is not None and not curve.empty:
        curve_path = RESULTS_DIR / f"sim_pnl_curve_{ts_str}.csv"
        curve.to_csv(curve_path, index=False)
        logger.info(f"P&L curve salva: {curve_path}")

    return RESULTS_DIR


# ──────────────────────────────────────────────────────────
# Sensibilidade ao market_discount (P2-32)
# ──────────────────────────────────────────────────────────

DEFAULT_SWEEP_DISCOUNTS = (0.0, 0.05, 0.10, 0.15, 0.20, 0.25)


def run_discount_sweep(
    markets: pd.DataFrame,
    sim_start: datetime,
    sim_end: datetime,
    capital: float,
    min_conviction: float,
    profit_mult: float,
    discounts: tuple[float, ...] = DEFAULT_SWEEP_DISCOUNTS,
    verbose: bool = False,
) -> list[dict]:
    """
    Roda a mesma janela sob vários market_discount pra deixar explícita a
    circularidade que a auditoria aponta: entry_price = BS_prob × (1 − desconto),
    logo edge = desconto × BS_prob, sempre positivo por construção. P&L subindo
    de forma monotônica com o desconto não é um resultado — é a definição da
    função rodada de novo com outro parâmetro.
    """
    # Busca spot/DVOL uma vez só e compartilha entre os discounts — sem isso
    # seriam N fetches de CoinGecko/Deribit pra dados que não mudam com
    # market_discount, incluindo o rate-limit/retry de _coingecko_get.
    spot_cache: dict[str, pd.Series] = {}
    dvol_cache: dict[str, pd.Series] = {}
    if not markets.empty:
        days = max(366, int((sim_end - sim_start).days) + 30)
        for asset in markets["asset"].unique().tolist():
            try:
                spot_cache[asset] = fetch_spot_daily(asset, days=min(days, 365))
                dvol_cache[asset] = fetch_dvol_daily(asset, days=min(days, 365))
            except RuntimeError as e:
                logger.warning(f"Dados {asset} indisponíveis: {e}")

    rows = []
    for discount in discounts:
        sim = WalkForwardSimulator(
            sim_start=sim_start, sim_end=sim_end, initial_capital=capital,
            min_conviction=min_conviction, profit_mult=profit_mult,
            market_discount=discount,
        )
        sim.spot = dict(spot_cache)
        sim.dvol = dict(dvol_cache)
        result = sim.run(markets.copy(), verbose=verbose)
        rows.append({
            "market_discount":  discount,
            "n_trades":         result["n_trades"],
            "total_pnl":        result["total_pnl"],
            "total_return_pct": result["total_return_pct"],
            "sharpe":           result["sharpe"],
            "brier_score":      result["brier_score"],
        })
    return rows


def print_discount_sweep(rows: list[dict]) -> None:
    from rich.console import Console
    from rich.table import Table
    from rich import box

    console = Console()
    console.print()
    console.print("[bold]── Sensibilidade ao market_discount ──────────────────[/bold]")
    console.print(
        "[dim]edge = desconto × BS_prob por construção — o simulador não "
        "produz P&L negativo pra desconto > 0 com convicção mínima fixa. "
        "brier_score não muda entre linhas: calibração não depende do desconto.[/dim]"
    )
    tbl = Table(box=box.SIMPLE, show_header=True, header_style="bold cyan")
    tbl.add_column("Desconto",  justify="right", width=9)
    tbl.add_column("Trades",    justify="right", width=7)
    tbl.add_column("P&L $",     justify="right", width=10)
    tbl.add_column("Retorno %", justify="right", width=10)
    tbl.add_column("Sharpe",    justify="right", width=8)
    tbl.add_column("Brier",     justify="right", width=8)
    for row in rows:
        tbl.add_row(
            f"{row['market_discount']:.0%}",
            str(row["n_trades"]),
            f"${row['total_pnl']:+.2f}",
            f"{row['total_return_pct']:+.2f}%",
            f"{row['sharpe']:.2f}" if row["sharpe"] is not None else "—",
            f"{row['brier_score']:.4f}" if row["brier_score"] is not None else "—",
        )
    console.print(tbl)


# ──────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────

@click.command()
@click.option("--days",            default=None, type=int,
              help="Duração da simulação em dias (default: toda a janela disponível).")
@click.option("--min-conviction",  default=0.08, type=float,
              help="Convicção mínima |BS_prob - 0.5| para abrir posição (default: 0.08).")
@click.option("--capital",         default=1000.0, type=float,
              help="Capital inicial em USDC (default: 1000).")
@click.option("--profit-mult",     default=PROFIT_TARGET_VALUE, type=float,
              help=f"Multiplicador do profit target (default: {PROFIT_TARGET_VALUE}).")
@click.option("--market-discount", default=0.20, type=float,
              help=(
                  "Desconto sintético: simula que Polymarket precificou X% abaixo "
                  "do BS_prob com drift. 0.0 = preço justo (nenhum trade). "
                  "0.15-0.25 = range empírico observado para touch markets. (default: 0.20)"
              ))
@click.option("--html",            is_flag=True, default=False,
              help="Gera relatório HTML ao final.")
@click.option("--save-csv",        is_flag=True, default=False,
              help="Salva trades e P&L curve em CSV.")
@click.option("--quiet",           is_flag=True, default=False,
              help="Suprime logs de abertura de posição.")
@click.option("--discount-sweep",  is_flag=True, default=False,
              help=(
                  "P2-32: em vez de uma simulação, roda a mesma janela sob vários "
                  "market_discount e imprime a tabela de sensibilidade "
                  "(--market-discount é ignorado). Torna explícito que P&L é "
                  "monotônico no parâmetro escolhido, não um resultado medido."
              ))
def main(
    days:            Optional[int],
    min_conviction:  float,
    capital:         float,
    profit_mult:     float,
    market_discount: float,
    html:            bool,
    save_csv:        bool,
    quiet:           bool,
    discount_sweep:  bool,
) -> None:
    """
    Simulador walk-forward histórico do modelo Deribit/BS.

    Usa preços sintéticos: entry_price = BS_prob × (1 − market_discount).
    market_discount=0 → sem edge (Kelly=0, nenhum trade).
    market_discount=0.20 → Polymarket subprecificou 20% vs. nosso modelo.
    """
    logger.info("=== Walk-Forward Simulation iniciada ===")

    # Carrega mercados
    markets = load_sim_markets()

    if markets.empty:
        logger.error("Nenhum mercado disponível para simulação.")
        raise SystemExit(1)

    # Define janela de simulação
    # Usa a data mais antiga do spot (CoinGecko retorna ~365 dias)
    earliest_data = datetime.now(timezone.utc) - timedelta(days=364)

    # Filtra mercados dentro da janela de dados disponíveis
    markets = markets[markets["end_date"] >= earliest_data].copy()
    if markets.empty:
        logger.error("Nenhum mercado dentro da janela de dados disponíveis (±365d).")
        raise SystemExit(1)

    # sim_start: 365 dias atrás OU days atrás
    if days is not None:
        sim_start = datetime.now(timezone.utc) - timedelta(days=days)
    else:
        sim_start = earliest_data

    # sim_end: hoje ou última data de expiração dos mercados (o que vier primeiro)
    sim_end = min(
        datetime.now(timezone.utc),
        markets["end_date"].max() + timedelta(days=1),
    )

    sim_start = sim_start.replace(hour=0, minute=0, second=0, microsecond=0)
    sim_end   = sim_end.replace(hour=0, minute=0, second=0, microsecond=0)

    logger.info(f"Janela: {sim_start.date()} → {sim_end.date()} ({(sim_end-sim_start).days}d)")
    logger.info(f"Mercados disponíveis: {len(markets)} "
                f"({markets['resolved_yes'].notna().sum()} com resolução conhecida)")

    if discount_sweep:
        rows = run_discount_sweep(
            markets=markets, sim_start=sim_start, sim_end=sim_end,
            capital=capital, min_conviction=min_conviction, profit_mult=profit_mult,
            verbose=not quiet,
        )
        print_discount_sweep(rows)
        return

    logger.info(f"Desconto de mercado: {market_discount:.0%} "
                f"({'sem edge — 0 trades esperados' if market_discount == 0 else 'simulação com ineficiência sintética'})")

    # Executa simulação
    sim = WalkForwardSimulator(
        sim_start       = sim_start,
        sim_end         = sim_end,
        initial_capital = capital,
        min_conviction  = min_conviction,
        profit_mult     = profit_mult,
        market_discount = market_discount,
    )

    result = sim.run(markets, verbose=not quiet)

    # Output
    print_sim_summary(result)

    if save_csv:
        save_results(result)

    if html:
        generate_html_report(result)


if __name__ == "__main__":
    main()
