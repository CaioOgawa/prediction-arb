"""
odds_collector.py
Coleta odds de casas de apostas via The Odds API e faz matching com mercados Polymarket.

Edge real = divergência entre preço Polymarket e probabilidade implícita das odds (sem vig).

Estratégia:
  1. Busca eventos esportivos ativos na The Odds API (bookmakers: Pinnacle, DraftKings, FanDuel)
  2. Converte odds decimais → probabilidade implícita (remove overround/vig)
  3. Faz matching com mercados Polymarket ativos usando nome de times + data
  4. Calcula divergência: polymarket_price - fair_prob
  5. Salva pares matched em Parquet para uso no signal_generator

Uso:
    uv run python pipeline/odds_collector.py
    uv run python pipeline/odds_collector.py --sport mlb --min-divergence 0.05

API key gratuita em: https://the-odds-api.com (500 req/mês no plano free)
Coloque ODDS_API_KEY no .env
"""

import os
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

try:
    from rapidfuzz import fuzz as _rfuzz
    def _seq_ratio(a: str, b: str) -> float:
        return _rfuzz.token_set_ratio(a, b) / 100.0
except ImportError:
    from difflib import SequenceMatcher
    def _seq_ratio(a: str, b: str) -> float:  # type: ignore[misc]
        return SequenceMatcher(None, a, b).ratio()

import pandas as pd
import requests
from dotenv import load_dotenv
from loguru import logger
from rich.console import Console
from rich.table import Table
from rich import box

load_dotenv()

ODDS_API_BASE = "https://api.the-odds-api.com/v4"
RAW_ODDS_DIR  = Path("data/raw/odds")
RAW_MKT_DIR   = Path("data/raw/markets")
RAW_ODDS_DIR.mkdir(parents=True, exist_ok=True)

console = Console()

# Mapeamento: categoria Polymarket → sport keys da Odds API
# Referência: https://the-odds-api.com/sports-odds-data/sports-apis.html
#
# Quota: cada sport_key = 1 request à API. Free tier = 500 req/mês.
# Ciclo full roda 6×/dia → cada sport_key custa ~180 req/mês.
# Manter o total de sport_keys únicos ativados por ciclo em <= 3-4.
CATEGORY_TO_SPORTS: dict[str, list[str]] = {
    # Esportes americanos
    "nba":     ["basketball_nba"],
    "mlb":     ["baseball_mlb"],
    "nfl":     ["americanfootball_nfl"],
    "nhl":     ["icehockey_nhl"],
    "mls":     ["soccer_usa_mls"],
    "ncaab":   ["basketball_ncaab"],
    "ncaaf":   ["americanfootball_ncaaf"],
    "cfl":     ["americanfootball_cfl"],

    # Soccer europeu — categorias específicas do Polymarket
    "epl":     ["soccer_epl"],
    "lal":     ["soccer_spain_la_liga"],          # La Liga (cat. Polymarket: "lal")
    "fl1":     ["soccer_france_ligue_one"],        # Ligue 1 (cat. Polymarket: "fl1")
    "ucl":     ["soccer_uefa_champs_league"],      # UCL partidas (cat. Polymarket: "ucl")
    "bundesliga": ["soccer_germany_bundesliga"],

    # Soccer genérico — fallback para categorias "soccer" não mapeadas acima.
    # Inclui apenas EPL + Bundesliga para não explodir quota da API.
    # Liga específica (lal, fl1, ucl) já são cobertas pelos mapeamentos acima.
    "soccer":  ["soccer_epl", "soccer_germany_bundesliga"],

    # Cricket
    "crint":   ["cricket_ipl"],

    # Fallback para categorias "sports" genéricas
    "sports":  ["basketball_nba", "baseball_mlb", "americanfootball_nfl",
                "icehockey_nhl", "soccer_usa_mls"],
}

# Mapeamento: categoria Polymarket → sport key da Odds API para outrights (vencedor)
# Outrights = odds de quem vai GANHAR o torneio ("Will X win the Premier League?")
# Quota: cada chave = 1 request. Outrights são raros (não mudam diariamente) →
# buscar apenas 1×/dia no ciclo full.
#
# Para descobrir sport_keys disponíveis: GET /v4/sports?apiKey=...
# Chaves de outrights têm sufixo "_winner" ou são o esporte base com markets=outrights.
OUTRIGHT_CATEGORY_TO_SPORTS: dict[str, str] = {
    # Copa do Mundo FIFA 2026 — $85M+ em mercados Polymarket
    "2026":       "soccer_fifa_world_cup_winner",

    # Ligas europeias — vencedores de temporada
    "english":    "soccer_epl",
    "epl":        "soccer_epl",
    "uefa":       "soccer_uefa_champs_league",
    "ucl":        "soccer_uefa_champs_league",
    "lal":        "soccer_spain_la_liga",
    "fl1":        "soccer_france_ligue_one",
    "bundesliga": "soccer_germany_bundesliga",

    # Tênis (Grand Slams) — detectado por palavra-chave na pergunta
    "tennis":     "tennis_atp_aus_open",
}

# Regex para extrair o sujeito de perguntas "Will [TEAM] win [TOURNAMENT]?"
import re as _re
_WILL_WIN_RE = _re.compile(
    r"^will\s+(?:the\s+)?(.+?)\s+win\b",
    _re.IGNORECASE,
)

# Bookmakers preferidos por ordem de confiabilidade
# Pinnacle é o mais preciso ("sharp book"), mas nem sempre disponível no plano free
PREFERRED_BOOKS = ["pinnacle", "draftkings", "fanduel", "betmgm", "caesars", "pointsbetus"]

# Books considerados "sharp" — overround baixo (~1.02-1.03), preços eficientes
# Signals com esses books têm edge real; outros têm overround alto demais para servir
# de referência confiável de probabilidade verdadeira.
SHARP_BOOKS = {"pinnacle", "betfair_ex_eu", "betfair_ex_uk", "matchbook"}

# Overround máximo aceito para usar como referência de probabilidade.
# Pinnacle: ~1.02 | Betfair: ~1.02 | DraftKings: ~1.06 | FanDuel: ~1.07
# Acima de 1.05 a vig consome o edge real e o sinal vira ruído.
MAX_OVERROUND = 1.05


# ──────────────────────────────────────────────────────────
# Odds API — fetch
# ──────────────────────────────────────────────────────────

def fetch_sport_odds(sport_key: str, api_key: str) -> list[dict]:
    """
    Busca eventos com odds H2H para um esporte específico.
    Retorna lista de eventos com bookmaker odds.
    """
    url = f"{ODDS_API_BASE}/sports/{sport_key}/odds"
    params = {
        "apiKey":   api_key,
        "regions":  "us,eu",
        "markets":  "h2h",
        "oddsFormat": "decimal",
        "dateFormat": "iso",
    }
    try:
        resp = requests.get(url, params=params, timeout=15)
        if resp.status_code == 401:
            raise ValueError("ODDS_API_KEY inválida ou ausente no .env")
        if resp.status_code == 422:
            logger.warning(f"Esporte não disponível no plano atual: {sport_key}")
            return []
        resp.raise_for_status()
        data = resp.json()
        remaining = resp.headers.get("x-requests-remaining", "?")
        logger.debug(f"  {sport_key}: {len(data)} eventos | requests restantes: {remaining}")
        return data
    except requests.RequestException as e:
        logger.error(f"Erro ao buscar {sport_key}: {e}")
        return []


def get_available_sports(api_key: str) -> list[str]:
    """Lista esportes com odds disponíveis no plano atual."""
    url = f"{ODDS_API_BASE}/sports"
    try:
        resp = requests.get(url, params={"apiKey": api_key}, timeout=10)
        resp.raise_for_status()
        return [s["key"] for s in resp.json() if s.get("active")]
    except requests.RequestException as e:
        logger.error(f"Erro ao listar esportes: {e}")
        return []


# ──────────────────────────────────────────────────────────
# Probabilidade implícita (remoção de vig)
# ──────────────────────────────────────────────────────────

def decimal_to_implied_prob(odds: float) -> float:
    """Converte odds decimais para probabilidade implícita bruta."""
    if odds <= 1.0:
        return 0.0
    return 1.0 / odds


def remove_vig(probs: list[float]) -> list[float]:
    """
    Remove overround (vig) — método multiplicativo (normalização simples).
    Divide cada probabilidade pelo total. Exato para mercados binários.
    Ex: [0.526, 0.526] → [0.500, 0.500]
    """
    total = sum(probs)
    if total <= 0:
        return probs
    return [p / total for p in probs]


def remove_vig_power(probs: list[float], tol: float = 1e-9, max_iter: int = 500) -> list[float]:
    """
    Remove overround pelo método Power (Shin/Jullien).

    Com overround > 1 (sum(p_i) > 1), as probabilidades implícitas bruta estão
    infladas pela vig. O método Power encontra o expoente n > 1 tal que
    sum(p_i^n) = 1 e retorna [p_i^n] como as probabilidades fair.

    Por que n > 1 reduz a soma: para p_i ∈ (0,1), p^n decresce com n.
    Logo sum(p^n) < sum(p) quando n > 1 — útil exatamente quando sum(p) > 1.

    Vantagem vs. método multiplicativo (normalização simples):
      - Multiplicativo: divide cada p_i pelo total → redistribui vig uniformemente
      - Power: aplica expoente → mantém razões de prob (favorito 2× > draw permanece 2×)
      - Para mercados 3-way (futebol), diferença chega a 0.5-1pp na prob do empate

    Referências:
      - Shin (1992): "Prices of State Contingent Claims..."
      - The Pinnacle article: "How to remove the juice from sports odds"
    """
    if len(probs) < 2:
        return probs

    total = sum(probs)
    if abs(total - 1.0) < tol:
        return probs  # sem vig, retorna direto

    # Busca binária em n ∈ [1, 10]:
    # n=1 → sum = total > 1 (muito alto)
    # n→∞ → sum → 0 (muito baixo)
    # Queremos n tal que sum(p^n) = 1
    lo, hi = 1.0, 10.0

    for _ in range(max_iter):
        mid = (lo + hi) / 2.0
        s   = sum(p ** mid for p in probs)
        if abs(s - 1.0) < tol:
            break
        if s > 1.0:
            lo = mid   # sum ainda > 1 → n precisa crescer para reduzir
        else:
            hi = mid   # sum < 1 → n precisa diminuir

    n       = (lo + hi) / 2.0
    powered = [p ** n for p in probs]
    # powered já suma ≈ 1 por construção; normalizamos para garantir precisão numérica
    s_pow   = sum(powered)
    return [p / s_pow for p in powered]


def _choose_remove_vig(probs: list[float]) -> tuple[list[float], str]:
    """
    Escolhe o método de remoção de vig conforme o número de resultados:
      - 2-way (binário): multiplicativo (idêntico ao power, mais rápido)
      - 3-way+ (futebol, etc.): power method (mais preciso)

    Retorna (fair_probs, método_usado).
    """
    if len(probs) == 2:
        return remove_vig(probs), "multiplicative"
    return remove_vig_power(probs), "power"


def extract_fair_probs(event: dict) -> dict[str, float] | None:
    """
    Extrai probabilidades justas (sem vig) usando CONSENSO de múltiplos bookmakers.

    Estratégia:
      - Coleta todos os bookmakers com overround <= MAX_OVERROUND
      - Pondera pelo inverso do overround (casas mais sharp → peso maior)
      - Retorna média ponderada + n_books + consensus_spread (incerteza entre casas)

    Retorna: {
        "home", "away", "draw"(opcional),
        "bookmaker"(melhor casa),
        "overround"(melhor),
        "is_sharp", "n_books", "consensus_spread"
    } ou None se nenhuma casa válida.
    """
    bookmakers = event.get("bookmakers", [])
    if not bookmakers:
        return None

    home_team = event.get("home_team", "")
    away_team = event.get("away_team", "")

    # Coleta todos os bookmakers válidos (overround <= MAX_OVERROUND)
    valid_books: list[dict] = []

    for bm in bookmakers:
        for market in bm.get("markets", []):
            if market.get("key") != "h2h":
                continue
            outcomes = market.get("outcomes", [])
            if len(outcomes) < 2:
                continue

            odds_map: dict[str, float] = {}
            for o in outcomes:
                name  = o.get("name", "")
                price = float(o.get("price", 0))
                if price > 1.0:
                    odds_map[name] = price

            if len(odds_map) < 2:
                continue

            teams      = list(odds_map.keys())
            raw_probs  = [decimal_to_implied_prob(odds_map[t]) for t in teams]
            overround  = round(sum(raw_probs), 4)

            if overround > MAX_OVERROUND:
                logger.debug(f"  Skipping {bm.get('key')} (overround={overround:.3f} > {MAX_OVERROUND})")
                continue

            fair_probs, _vig_method = _choose_remove_vig(raw_probs)

            book_entry: dict = {
                "bookmaker":  bm.get("key", "unknown"),
                "overround":  overround,
                "is_sharp":   bm.get("key", "") in SHARP_BOOKS,
                "vig_method": _vig_method,
            }
            for i, team in enumerate(teams):
                if team == home_team:
                    book_entry["home"] = fair_probs[i]
                elif team == away_team:
                    book_entry["away"] = fair_probs[i]
                else:
                    book_entry["draw"] = fair_probs[i]

            if "home" not in book_entry and len(fair_probs) >= 1:
                book_entry["home"] = fair_probs[0]
            if "away" not in book_entry and len(fair_probs) >= 2:
                book_entry["away"] = fair_probs[1]

            valid_books.append(book_entry)
            break  # um mercado H2H por bookmaker

    if not valid_books:
        return None

    # Melhor casa (menor overround)
    best = min(valid_books, key=lambda b: b["overround"])

    # Média ponderada: peso = 1/overround (mais sharp → mais peso)
    weights    = [1.0 / b["overround"] for b in valid_books]
    total_w    = sum(weights)

    def _wavg(key: str) -> float | None:
        vals = [b.get(key) for b in valid_books]
        if any(v is None for v in vals):
            return None
        return sum(w * v for w, v in zip(weights, vals)) / total_w  # type: ignore[arg-type]

    home_avg = _wavg("home")
    away_avg = _wavg("away")

    if home_avg is None or away_avg is None:
        # Fallback: usa melhor casa
        home_avg = best.get("home", 0.5)
        away_avg = best.get("away", 0.5)

    result: dict = {
        "bookmaker":        best["bookmaker"],
        "overround":        best["overround"],
        "is_sharp":         any(b["is_sharp"] for b in valid_books),
        "n_books":          len(valid_books),
        "home":             round(home_avg, 4),
        "away":             round(away_avg, 4),
    }

    draw_avg = _wavg("draw")
    if draw_avg is not None:
        result["draw"] = round(draw_avg, 4)

    # Método de remoção de vig usado (power para 3-way, multiplicative para 2-way)
    # Usa o da melhor casa como referência (todas usam o mesmo critério de escolha)
    result["vig_method"] = best.get("vig_method", "multiplicative")

    # Spread do consenso: desvio padrão das probabilidades "home" entre casas
    # Alto spread → books discordam → fair_prob menos confiável
    if len(valid_books) > 1:
        home_vals = [b["home"] for b in valid_books if "home" in b]
        if len(home_vals) > 1:
            mean_h = sum(home_vals) / len(home_vals)
            variance = sum((v - mean_h) ** 2 for v in home_vals) / len(home_vals)
            result["consensus_spread"] = round(variance ** 0.5, 4)
        else:
            result["consensus_spread"] = 0.0
    else:
        result["consensus_spread"] = 0.0

    return result


# ──────────────────────────────────────────────────────────
# Matching Polymarket ↔ Odds API
# ──────────────────────────────────────────────────────────

# Mapa de aliases: apelidos / abreviações → nome canônico (lowercase).
# Aplicado antes do fuzzy matching para aumentar recall de 0.3% → alvo 3-5%.
# Adicione entradas conforme novos mismatches forem encontrados nos logs.
TEAM_ALIASES: dict[str, str] = {
    # NBA
    "lakers":           "los angeles lakers",
    "la lakers":        "los angeles lakers",
    "clippers":         "los angeles clippers",
    "la clippers":      "los angeles clippers",
    "knicks":           "new york knicks",
    "ny knicks":        "new york knicks",
    "nets":             "brooklyn nets",
    "sixers":           "philadelphia 76ers",
    "76ers":            "philadelphia 76ers",
    "warriors":         "golden state warriors",
    "gsw":              "golden state warriors",
    "celtics":          "boston celtics",
    "bucks":            "milwaukee bucks",
    "nuggets":          "denver nuggets",
    "heat":             "miami heat",
    "suns":             "phoenix suns",
    "mavs":             "dallas mavericks",
    "mavericks":        "dallas mavericks",
    "thunder":          "oklahoma city thunder",
    "okc":              "oklahoma city thunder",
    "raptors":          "toronto raptors",
    "hawks":            "atlanta hawks",
    "bulls":            "chicago bulls",
    "pacers":           "indiana pacers",
    # NFL
    "pats":             "new england patriots",
    "patriots":         "new england patriots",
    "ne patriots":      "new england patriots",
    "cowboys":          "dallas cowboys",
    "eagles":           "philadelphia eagles",
    "ny giants":        "new york giants",
    "ny jets":          "new york jets",
    "chiefs":           "kansas city chiefs",
    "kc chiefs":        "kansas city chiefs",
    "niners":           "san francisco 49ers",
    "49ers":            "san francisco 49ers",
    "sf 49ers":         "san francisco 49ers",
    "ravens":           "baltimore ravens",
    "browns":           "cleveland browns",
    "steelers":         "pittsburgh steelers",
    "bengals":          "cincinnati bengals",
    "packers":          "green bay packers",
    "gb packers":       "green bay packers",
    "bears":            "chicago bears",
    "lions":            "detroit lions",
    "vikings":          "minnesota vikings",
    "seahawks":         "seattle seahawks",
    "rams":             "los angeles rams",
    "la rams":          "los angeles rams",
    "chargers":         "los angeles chargers",
    "la chargers":      "los angeles chargers",
    "raiders":          "las vegas raiders",
    "lv raiders":       "las vegas raiders",
    "broncos":          "denver broncos",
    "texans":           "houston texans",
    "colts":            "indianapolis colts",
    "jaguars":          "jacksonville jaguars",
    "jags":             "jacksonville jaguars",
    "titans":           "tennessee titans",
    "dolphins":         "miami dolphins",
    "bills":            "buffalo bills",
    "az cardinals":     "arizona cardinals",
    "saints":           "new orleans saints",
    "falcons":          "atlanta falcons",
    "bucs":             "tampa bay buccaneers",
    "buccaneers":       "tampa bay buccaneers",
    "tb buccaneers":    "tampa bay buccaneers",
    "commanders":       "washington commanders",
    # MLB
    "yankees":          "new york yankees",
    "ny yankees":       "new york yankees",
    "mets":             "new york mets",
    "ny mets":          "new york mets",
    "red sox":          "boston red sox",
    "dodgers":          "los angeles dodgers",
    "la dodgers":       "los angeles dodgers",
    "cubs":             "chicago cubs",
    "white sox":        "chicago white sox",
    "astros":           "houston astros",
    "braves":           "atlanta braves",
    "phillies":         "philadelphia phillies",
    "sf giants":        "san francisco giants",
    "padres":           "san diego padres",
    "mariners":         "seattle mariners",
    "tx rangers":       "texas rangers",
    "angels":           "los angeles angels",
    "la angels":        "los angeles angels",
    "athletics":        "oakland athletics",
    "a's":              "oakland athletics",
    "blue jays":        "toronto blue jays",
    "tor blue jays":    "toronto blue jays",
    "twins":            "minnesota twins",
    "tigers":           "detroit tigers",
    "orioles":          "baltimore orioles",
    "nationals":        "washington nationals",
    "reds":             "cincinnati reds",
    "pirates":          "pittsburgh pirates",
    "brewers":          "milwaukee brewers",
    "rockies":          "colorado rockies",
    "rays":             "tampa bay rays",
    "tb rays":          "tampa bay rays",
    "royals":           "kansas city royals",
    "kc royals":        "kansas city royals",
    "marlins":          "miami marlins",
    "diamondbacks":     "arizona diamondbacks",
    "d-backs":          "arizona diamondbacks",
    # Premier League / Soccer
    "man utd":          "manchester united",
    "man united":       "manchester united",
    "man city":         "manchester city",
    "manchester city":  "manchester city",
    "tottenham":        "tottenham hotspur",
    "arsenal":          "arsenal",
    "chelsea":          "chelsea",
    "liverpool":        "liverpool",
    "wolves":           "wolverhampton wanderers",
    "wolverhampton":    "wolverhampton wanderers",
    "villa":            "aston villa",
    "newcastle":        "newcastle united",
    "leeds":            "leeds united",
    "brighton":         "brighton & hove albion",
    "brentford":        "brentford",
    "fulham":           "fulham",
    "everton":          "everton",
    "west ham":         "west ham united",
    "crystal palace":   "crystal palace",
    "nottm forest":     "nottingham forest",
    "notts forest":     "nottingham forest",
    "leicester":        "leicester city",
    "southampton":      "southampton",
    "ipswich":          "ipswich town",
    # NHL
    "leafs":            "toronto maple leafs",
    "maple leafs":      "toronto maple leafs",
    "bruins":           "boston bruins",
    "habs":             "montreal canadiens",
    "canadiens":        "montreal canadiens",
    "ny rangers":       "new york rangers",
    "flyers":           "philadelphia flyers",
    "penguins":         "pittsburgh penguins",
    "pens":             "pittsburgh penguins",
    "caps":             "washington capitals",
    "capitals":         "washington capitals",
    "lightning":        "tampa bay lightning",
    "tb lightning":     "tampa bay lightning",
    "oilers":           "edmonton oilers",
    "flames":           "calgary flames",
    "canucks":          "vancouver canucks",
    "senators":         "ottawa senators",
    "ducks":            "anaheim ducks",
    "kings":            "los angeles kings",
    "la kings":         "los angeles kings",
    "sharks":           "san jose sharks",
    "golden knights":   "vegas golden knights",
    "vgk":              "vegas golden knights",
    "wild":             "minnesota wild",
    "blues":            "st. louis blues",
    "predators":        "nashville predators",
    "preds":            "nashville predators",
    "hurricanes":       "carolina hurricanes",
    "canes":            "carolina hurricanes",
    "blue jackets":     "columbus blue jackets",
    "cbj":              "columbus blue jackets",
    "avalanche":        "colorado avalanche",
    "avs":              "colorado avalanche",
    "stars":            "dallas stars",
    "coyotes":          "arizona coyotes",
    "az coyotes":       "arizona coyotes",
    "islanders":        "new york islanders",
    "ny islanders":     "new york islanders",
    "sabres":           "buffalo sabres",
    "devils":           "new jersey devils",
    "nj devils":        "new jersey devils",
    "red wings":        "detroit red wings",
    "blackhawks":       "chicago blackhawks",
}

# Aliases AMBÍGUOS entre esportes — resolvidos pelo prefixo do sport_key do evento.
# Não podem viver no dict global: chaves duplicadas em dict literal se sobrescrevem
# silenciosamente (bug 2026-07: "spurs" global apontava para Tottenham e o
# San Antonio Spurs sumiu do matching da NBA).
SPORT_ALIASES: dict[str, dict[str, str]] = {
    "basketball": {
        "spurs":      "san antonio spurs",
    },
    "americanfootball": {
        "cardinals":  "arizona cardinals",
        "giants":     "new york giants",
        "jets":       "new york jets",
        "panthers":   "carolina panthers",
        "washington": "washington commanders",
    },
    "baseball": {
        "cardinals":  "st. louis cardinals",
        "giants":     "san francisco giants",
        "rangers":    "texas rangers",
    },
    "icehockey": {
        "rangers":    "new york rangers",
        "jets":       "winnipeg jets",
        "panthers":   "florida panthers",
    },
    "soccer": {
        "spurs":      "tottenham hotspur",
    },
}

# Mapa reverso: canônico → conjunto de apelidos. Usado para comparar a pergunta
# contra TODAS as variantes do time — substitui o antigo str.replace na pergunta,
# que corrompia substrings ("washington nationals" → "...commanders nationals").
_REVERSE_ALIASES: dict[str, set[str]] = {}
for _alias, _canon in TEAM_ALIASES.items():
    _REVERSE_ALIASES.setdefault(_canon, set()).add(_alias)
for _sport_map in SPORT_ALIASES.values():
    for _alias, _canon in _sport_map.items():
        _REVERSE_ALIASES.setdefault(_canon, set()).add(_alias)


def _sport_group(sport_key: str) -> str:
    """'basketball_nba' → 'basketball'; '' → ''."""
    return sport_key.split("_", 1)[0] if sport_key else ""


def _normalize_team(name: str, sport_key: str = "") -> str:
    """
    Normaliza nome de time: lowercase → aliases do esporte → aliases globais.
    Permite que 'lakers' e 'los angeles lakers' tenham similaridade máxima.
    """
    normalized = name.lower().strip()
    sport_map = SPORT_ALIASES.get(_sport_group(sport_key), {})
    if normalized in sport_map:
        return sport_map[normalized]
    return TEAM_ALIASES.get(normalized, normalized)


def _team_variants(team_raw: str, sport_key: str = "") -> set[str]:
    """Todas as formas conhecidas de um time: nome bruto, canônico e apelidos."""
    canonical = _normalize_team(team_raw, sport_key)
    variants = {team_raw.lower().strip(), canonical}
    variants |= _REVERSE_ALIASES.get(canonical, set())
    return variants


def _underlying_key(sport_key: str, team_a: str, team_b: str = "") -> str:
    """
    P1-11: chave de correlação para o cap de exposição por underlying —
    identifica o jogo/participante real, não a categoria solta da Gamma API.
    Usa nomes canônicos (via _normalize_team) para que apelidos diferentes do
    mesmo time (ex: "Lakers" vs "LA Lakers") caiam na mesma chave. Ordem dos
    times não importa: "A vs B" e "B vs A" são o mesmo jogo.
    """
    a = _normalize_team(team_a, sport_key)
    if not team_b:
        return f"{sport_key}:{a}"
    b = _normalize_team(team_b, sport_key)
    lo, hi = sorted([a, b])
    return f"{sport_key}:{lo}v{hi}"


def _token_similarity(a: str, b: str) -> float:
    """Similaridade por sobreposição de tokens (case-insensitive)."""
    tokens_a = set(a.lower().split())
    tokens_b = set(b.lower().split())
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    return len(intersection) / len(union)


def _name_similarity(a: str, b: str) -> float:
    """Combina Jaccard de tokens com SequenceMatcher/rapidfuzz para melhor precisão."""
    jaccard = _token_similarity(a, b)
    seq = _seq_ratio(a.lower(), b.lower())
    return 0.6 * jaccard + 0.4 * seq


def _match_score(question: str, home_team: str, away_team: str, sport_key: str = "") -> float:
    """
    Score de matching entre uma pergunta Polymarket e um evento esportivo.
    Compara a pergunta contra todas as variantes conhecidas de cada time
    (nome bruto, canônico e apelidos do esporte) — sem mutar a pergunta.
    Retorna valor entre 0 e 1 (1 = match perfeito).
    """
    q_lower = question.lower()

    def _best_score(team_raw: str) -> float:
        return max(
            max(_name_similarity(q_lower, v), _token_similarity(q_lower, v))
            for v in _team_variants(team_raw, sport_key)
        )

    home_score = _best_score(home_team)
    away_score = _best_score(away_team)

    # Perguntas do Polymarket citam UM time na maioria dos casos
    # ("Will Manchester United FC win on 2026-05-09?") — exigir os dois times
    # mataria o formato dominante. Regra: um time com match forte é obrigatório;
    # o segundo, quando presente, dá bônus (média). A janela de data do
    # match_markets_to_odds desambigua qual jogo do time é o evento certo.
    primary   = max(home_score, away_score)
    secondary = min(home_score, away_score)

    if primary < 0.45:
        return 0.0
    if secondary >= 0.2:
        return (home_score + away_score) / 2
    return primary


def match_markets_to_odds(
    markets_df: pd.DataFrame,
    odds_events: list[dict],
    min_score: float = 0.25,
    date_window_days: int = 2,
) -> pd.DataFrame:
    """
    Faz matching entre mercados Polymarket e eventos da Odds API.

    Args:
        markets_df:      DataFrame de mercados Polymarket ativos
        odds_events:     Lista de eventos retornados pela Odds API
        min_score:       Score mínimo de similaridade de nome (0–1)
        date_window_days: Janela de datas para considerar match válido

    Returns:
        DataFrame de pares matched com colunas de divergência.
    """
    if markets_df.empty or not odds_events:
        return pd.DataFrame()

    now = datetime.now(timezone.utc)
    rows = []

    skipped_ou = 0
    for _, mkt in markets_df.iterrows():
        question  = str(mkt.get("question", ""))
        yes_price = float(mkt.get("yes_price") or 0.5)
        end_date  = pd.to_datetime(mkt.get("endDate"), errors="coerce", utc=True)

        if pd.isna(end_date):
            continue
        if yes_price < 0.05 or yes_price > 0.95:
            continue  # Mercado já praticamente resolvido — sem edge real

        # Classifica o tipo de pergunta para determinar a probabilidade correta
        q_lower_check = question.lower()

        # Pula mercados Over/Under — odds H2H (win/loss) não são comparáveis
        if " o/u " in q_lower_check or ": o/u" in q_lower_check or "over/under" in q_lower_check:
            skipped_ou += 1
            continue

        # Detecta mercados de empate — precisam da prob. de draw, não de vitória
        is_draw_market = any(w in q_lower_check for w in ["draw", "tie", "empate"])

        best_match = None
        best_score = 0.0

        for event in odds_events:
            home_team    = event.get("home_team", "")
            away_team    = event.get("away_team", "")
            commence_str = event.get("commence_time", "")

            try:
                commence = pd.to_datetime(commence_str, utc=True)
            except Exception:
                continue

            # Só jogos futuros — exclui jogos em andamento ou já finalizados
            # (odds pré-jogo vs. preço Polymarket in-play seria comparação inválida)
            if commence <= now:
                continue

            # Filtro de data: mercado deve resolver próximo ao início do evento
            date_diff = abs((end_date - commence).total_seconds() / 86400)
            if date_diff > date_window_days:
                continue

            score = _match_score(question, home_team, away_team,
                                 sport_key=event.get("sport_key", ""))
            if score > best_score:
                best_score = score
                best_match = event

        if best_score < min_score or best_match is None:
            continue

        # Extrai probabilidades justas do melhor match
        fair = extract_fair_probs(best_match)
        if fair is None:
            continue

        home_team = best_match.get("home_team", "")
        away_team = best_match.get("away_team", "")
        q_lower   = question.lower()

        if is_draw_market:
            # Mercado de empate: compara contra a probabilidade de draw do bookmaker.
            # Se o bookmaker não forneceu draw (esportes sem empate, e.g. NBA), pula.
            fair_draw = fair.get("draw")
            if fair_draw is None:
                continue
            fair_yes = fair_draw
            fair_no  = round(1.0 - fair_draw, 4)
            yes_team = "draw"
        else:
            # Mercado de vitória: YES = time que aparece PRIMEIRO na pergunta.
            # Padrão Polymarket: "[Time A] vs. [Time B]" → YES = Time A.
            def _first_token_pos(team: str, text: str) -> int:
                tokens = team.lower().split()
                positions = [text.find(tok) for tok in tokens if tok in text]
                return min(positions) if positions else len(text)

            home_pos = _first_token_pos(home_team, q_lower)
            away_pos = _first_token_pos(away_team, q_lower)

            if home_pos <= away_pos:
                fair_yes = fair.get("home", 0.5)
                fair_no  = fair.get("away", 0.5)
                yes_team = home_team
            else:
                fair_yes = fair.get("away", 0.5)
                fair_no  = fair.get("home", 0.5)
                yes_team = away_team

        divergence = round(yes_price - fair_yes, 4)

        # Sanity check: edge > 25% quase sempre indica erro de matching.
        # Pinnacle não fica tão desalinhado do Polymarket em mercados líquidos.
        MAX_PLAUSIBLE_EDGE = 0.25
        if abs(divergence) > MAX_PLAUSIBLE_EDGE:
            logger.debug(
                f"  Edge implausível {divergence:+.2%} descartado: '{question[:50]}' "
                f"(yes={yes_price:.2f} fair={fair_yes:.2f})"
            )
            continue

        sport_key_val = best_match.get("sport_key", "")
        rows.append({
            "condition_id":     mkt.get("conditionId"),
            "question":         question,
            "category":         mkt.get("category", ""),
            "underlying":       _underlying_key(sport_key_val, home_team, away_team),
            "yes_price":        round(yes_price, 4),
            "fair_prob_yes":    round(fair_yes, 4),
            "fair_prob_no":     round(fair_no, 4),
            "divergence":       divergence,           # + → Poly overpriced YES, apostar NO
            "abs_divergence":   abs(divergence),      # magnitude do edge
            "direction":        "BUY_NO" if divergence > 0 else "BUY_YES",
            "yes_team":         yes_team,
            "home_team":        home_team,
            "away_team":          away_team,
            "bookmaker":          fair.get("bookmaker", "unknown"),
            "overround":          fair.get("overround", 1.0),
            "is_sharp":           fair.get("is_sharp", False),
            "n_books":            int(fair.get("n_books", 1)),
            "consensus_spread":   float(fair.get("consensus_spread", 0.0)),
            "vig_method":         fair.get("vig_method", "multiplicative"),
            "match_score":        round(best_score, 3),
            "sport_key":          best_match.get("sport_key", ""),
            "commence_time":      best_match.get("commence_time", ""),
            "end_date":           mkt.get("endDate", ""),
            "liquidity":          float(mkt.get("liquidity") or 0),
            "volume_24h":         float(mkt.get("volume24hr") or 0),
        })

    if skipped_ou:
        logger.debug(f"  Mercados O/U ignorados (incompatível com H2H): {skipped_ou}")

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).sort_values("abs_divergence", ascending=False).reset_index(drop=True)
    return df


# ──────────────────────────────────────────────────────────
# Outrights — odds de vencedor de torneio
# ──────────────────────────────────────────────────────────

def fetch_outright_odds(sport_key: str, api_key: str) -> list[dict]:
    """
    Busca odds de vencedor (outrights) para um esporte/torneio.
    Retorna lista de eventos onde cada evento é o torneio inteiro com outcomes = participantes.
    """
    url = f"{ODDS_API_BASE}/sports/{sport_key}/odds"
    params = {
        "apiKey":     api_key,
        "regions":    "us,eu",
        "markets":    "outrights",
        "oddsFormat": "decimal",
        "dateFormat": "iso",
    }
    try:
        resp = requests.get(url, params=params, timeout=15)
        if resp.status_code == 401:
            raise ValueError("ODDS_API_KEY inválida ou ausente no .env")
        if resp.status_code == 422:
            logger.warning(f"Outrights não disponíveis para: {sport_key}")
            return []
        resp.raise_for_status()
        data = resp.json()
        remaining = resp.headers.get("x-requests-remaining", "?")
        logger.debug(f"  {sport_key} (outrights): {len(data)} eventos | requests restantes: {remaining}")
        return data
    except requests.RequestException as e:
        logger.error(f"Erro ao buscar outrights {sport_key}: {e}")
        return []


def _extract_outright_subject(question: str) -> str | None:
    """
    Extrai o nome do sujeito de perguntas "Will [TEAM] win [TOURNAMENT]?".
    Retorna None se a pergunta não seguir o padrão.
    """
    m = _WILL_WIN_RE.match(question.strip())
    return m.group(1).strip() if m else None


def match_outright_markets(
    markets_df: pd.DataFrame,
    outright_events: list[dict],
    min_score: float = 0.35,
) -> pd.DataFrame:
    """
    Faz matching entre mercados "Will X win Y?" e odds de outrights.

    Estratégia:
      1. Extrai o sujeito da pergunta ("Will [SUBJECT] win...?")
      2. Para cada evento outright, procura [SUBJECT] nas outcomes (participantes)
      3. Calcula fair_prob do sujeito e divergência vs. yes_price

    Args:
        markets_df:      DataFrame de mercados Polymarket filtrados
        outright_events: Eventos de outrights da Odds API
        min_score:       Similaridade mínima sujeito ↔ outcome (0–1)

    Returns:
        DataFrame de pares matched com colunas de divergência (mesmo formato H2H).
    """
    if markets_df.empty or not outright_events:
        return pd.DataFrame()

    # Pré-processa: para cada evento, extrai todas as outcomes com seus bookmakers
    # Estrutura: {sport_key → list[{event_name, outcomes: {team: fair_prob}, bookmaker, overround, is_sharp}]}
    def _get_outright_probs(event: dict) -> dict | None:
        """
        Extrai fair_probs de todos os participantes usando CONSENSO de múltiplos bookmakers.
        Retorna prob_map como média ponderada (peso = 1/overround) entre casas válidas.
        """
        bookmakers = event.get("bookmakers", [])
        if not bookmakers:
            return None

        valid_books: list[tuple[str, float, bool, dict[str, float]]] = []
        # (bookmaker_key, overround, is_sharp, prob_map)

        for bm in bookmakers:
            for market in bm.get("markets", []):
                if market.get("key") != "outrights":
                    continue
                outcomes = market.get("outcomes", [])
                if len(outcomes) < 2:
                    continue

                odds_map: dict[str, float] = {
                    o["name"]: float(o["price"])
                    for o in outcomes
                    if o.get("price", 0) > 1.0
                }
                if len(odds_map) < 2:
                    continue

                raw_probs = [decimal_to_implied_prob(p) for p in odds_map.values()]
                overround = round(sum(raw_probs), 4)
                if overround > MAX_OVERROUND:
                    logger.debug(f"  Outright {bm.get('key')} overround={overround:.3f} > {MAX_OVERROUND} — skip")
                    continue

                fair_list, _ = _choose_remove_vig(raw_probs)
                prob_map  = dict(zip(odds_map.keys(), fair_list))
                valid_books.append((bm.get("key", "unknown"), overround,
                                    bm.get("key", "") in SHARP_BOOKS, prob_map))
                break  # um mercado por bookmaker

        if not valid_books:
            return None

        # Ponderação: peso = 1/overround
        weights   = [1.0 / b[1] for b in valid_books]
        total_w   = sum(weights)
        best_bm   = min(valid_books, key=lambda b: b[1])

        # Consenso: para cada participante, média ponderada das probabilidades entre casas
        # Usa todos os nomes que aparecem em pelo menos um book
        all_teams: set[str] = set()
        for _, _, _, pm in valid_books:
            all_teams.update(pm.keys())

        consensus_map: dict[str, float] = {}
        for team in all_teams:
            w_sum  = sum(w * b[3].get(team, 0.0) for w, b in zip(weights, valid_books))
            # Normaliza pelo peso total dos books que têm este participante
            w_have = sum(w for w, b in zip(weights, valid_books) if team in b[3])
            consensus_map[team] = w_sum / w_have if w_have > 0 else 0.0

        return {
            "prob_map":         consensus_map,
            "bookmaker":        best_bm[0],
            "overround":        best_bm[1],
            "is_sharp":         any(b[2] for b in valid_books),
            "n_books":          len(valid_books),
            "sport_key":        event.get("sport_key", ""),
            "event_name":       event.get("sport_title", event.get("sport_key", "")),
        }

    # Cache de fair_probs por evento
    event_probs = [_get_outright_probs(ev) for ev in outright_events]
    valid_events = [(ev, ep) for ev, ep in zip(outright_events, event_probs) if ep is not None]
    logger.info(f"Eventos outright com odds válidas: {len(valid_events):,}")

    rows = []
    for _, mkt in markets_df.iterrows():
        question  = str(mkt.get("question", ""))
        yes_price = float(mkt.get("yes_price") or 0.5)

        if yes_price < 0.02 or yes_price > 0.98:
            continue  # Praticamente resolvido

        subject = _extract_outright_subject(question)
        if subject is None:
            continue  # Pergunta não segue padrão "Will X win"

        best_prob   = None
        best_score  = 0.0
        best_ep     = None
        best_team   = None

        for _ev, ep in valid_events:
            for team_name, fair_prob in ep["prob_map"].items():
                score = _name_similarity(subject.lower(), team_name.lower())
                if score > best_score:
                    best_score = score
                    best_prob  = fair_prob
                    best_ep    = ep
                    best_team  = team_name

        if best_score < min_score or best_prob is None or best_ep is None:
            continue

        divergence = round(yes_price - best_prob, 4)

        rows.append({
            "condition_id":  mkt.get("conditionId"),
            "question":      question,
            "category":      mkt.get("category", ""),
            "underlying":    _underlying_key(best_ep["sport_key"], subject),
            "yes_price":     round(yes_price, 4),
            "fair_prob_yes": round(best_prob, 4),
            "fair_prob_no":  round(1.0 - best_prob, 4),
            "divergence":    divergence,
            "abs_divergence": abs(divergence),
            "direction":     "BUY_NO" if divergence > 0 else "BUY_YES",
            "yes_team":      subject,
            "home_team":     best_team,   # outcome name no bookmaker
            "away_team":     "",
            "bookmaker":       best_ep["bookmaker"],
            "overround":       best_ep["overround"],
            "is_sharp":        best_ep["is_sharp"],
            "n_books":         int(best_ep.get("n_books", 1)),
            "consensus_spread": 0.0,  # outrights: spread calculado por participante, não por mercado
            "match_score":     round(best_score, 3),
            "sport_key":       best_ep["sport_key"],
            "market_type":     "outright",
            "commence_time": "",
            "end_date":      mkt.get("endDate", ""),
            "liquidity":     float(mkt.get("liquidity") or 0),
            "volume_24h":    float(mkt.get("volume24hr") or 0),
        })

    if not rows:
        return pd.DataFrame()

    return (
        pd.DataFrame(rows)
        .sort_values("abs_divergence", ascending=False)
        .reset_index(drop=True)
    )


# ──────────────────────────────────────────────────────────
# Pipeline principal
# ──────────────────────────────────────────────────────────

def load_active_markets(min_liquidity: float = 1_000) -> pd.DataFrame:
    """Carrega o snapshot mais recente de mercados ativos."""
    candidates = sorted(
        list(RAW_MKT_DIR.glob("markets_all_*.parquet")) +
        list(RAW_MKT_DIR.glob("markets_incremental_*.parquet")),
        key=lambda p: p.stat().st_mtime,  # ordena por tempo de modificação, não por nome
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            f"Nenhum arquivo de mercados em {RAW_MKT_DIR}. "
            "Execute: uv run python pipeline/fetch_markets.py"
        )
    df = pd.read_parquet(candidates[0])

    # Filtra apenas mercados ativos e com liquidez mínima
    if "active" in df.columns:
        df = df[df["active"] == True]
    if "closed" in df.columns:
        df = df[df["closed"] == False]
    if "liquidity" in df.columns:
        df = df[pd.to_numeric(df["liquidity"], errors="coerce").fillna(0) >= min_liquidity]

    logger.info(f"Mercados ativos carregados: {len(df):,} (liquidity >= ${min_liquidity:,.0f})")
    return df.reset_index(drop=True)


def collect_odds_for_sports(
    categories: list[str] | None = None,
    api_key: str | None = None,
) -> tuple[list[dict], set[str]]:
    """
    Coleta odds de todos os esportes relevantes para as categorias dadas.

    Returns:
        (lista de eventos, set de sport_keys consultados)
    """
    if api_key is None:
        api_key = os.getenv("ODDS_API_KEY", "")
    if not api_key:
        raise ValueError(
            "ODDS_API_KEY não encontrada. Adicione ao .env e obtenha em https://the-odds-api.com"
        )

    # Determina quais esportes buscar
    if categories:
        sport_keys: set[str] = set()
        for cat in categories:
            cat_lower = cat.lower()
            for pattern, sports in CATEGORY_TO_SPORTS.items():
                if pattern in cat_lower or cat_lower in pattern:
                    sport_keys.update(sports)
        if not sport_keys:
            # Fallback: tenta com o nome direto da categoria
            sport_keys = set(categories)
    else:
        # Sem filtro: coleta todos os esportes mapeados
        sport_keys = set(s for sports in CATEGORY_TO_SPORTS.values() for s in sports)

    all_events: list[dict] = []
    consulted: set[str] = set()

    for sport_key in sorted(sport_keys):
        logger.info(f"Buscando odds: {sport_key}...")
        events = fetch_sport_odds(sport_key, api_key)
        all_events.extend(events)
        consulted.add(sport_key)
        if len(sport_keys) > 1:
            time.sleep(0.5)  # Respeita rate limit

    logger.info(f"Total de eventos com odds: {len(all_events):,}")
    return all_events, consulted


def run(
    min_divergence: float = 0.03,
    min_liquidity: float = 1_000,
    min_match_score: float = 0.25,
    categories: list[str] | None = None,
    api_key: str | None = None,
    save: bool = True,
    include_outrights: bool = True,
) -> pd.DataFrame:
    """
    Pipeline completo: coleta odds → carrega mercados → matching → divergência.

    Args:
        min_divergence:  edge mínimo (|polymarket - fair_prob|) para incluir no output
        min_liquidity:   liquidez mínima do mercado Polymarket em USDC
        min_match_score: score mínimo de similaridade para aceitar match
        categories:      lista de categorias Polymarket a filtrar (None = todas)
        api_key:         ODDS_API_KEY (None = lê do .env)
        save:            se True, salva resultado em Parquet

    Returns:
        DataFrame de mercados matched com divergência calculada.
    """
    markets = load_active_markets(min_liquidity=min_liquidity)

    # Filtra categorias se especificado
    if categories:
        cat_mask = markets["category"].str.lower().isin([c.lower() for c in categories])
        # Inclui também categorias que contenham os termos (ex: "mlb" em "sports_mlb")
        for cat in categories:
            cat_mask = cat_mask | markets["category"].str.lower().str.contains(cat.lower(), na=False)
        markets_filtered = markets[cat_mask]
        logger.info(f"Mercados após filtro de categoria: {len(markets_filtered):,}")
    else:
        # Filtra para esportes conhecidos automaticamente
        known_cats = set(CATEGORY_TO_SPORTS.keys())
        cat_mask = markets["category"].str.lower().apply(
            lambda c: any(k in str(c).lower() for k in known_cats)
        )
        markets_filtered = markets[cat_mask]
        logger.info(f"Mercados esportivos detectados automaticamente: {len(markets_filtered):,}")

    if markets_filtered.empty:
        logger.warning("Nenhum mercado esportivo encontrado. Verifique as categorias disponíveis.")
        return pd.DataFrame()

    all_cats = markets_filtered["category"].str.lower().unique().tolist()

    # ── H2H matching ─────────────────────────────────────────
    resolved_key = _api_key(api_key)
    events, _ = collect_odds_for_sports(categories=all_cats, api_key=resolved_key)

    h2h_matched = pd.DataFrame()
    if events:
        logger.info("Matching H2H (partida a partida)...")
        h2h_matched = match_markets_to_odds(
            markets_filtered, events, min_score=min_match_score,
        )
        if not h2h_matched.empty:
            h2h_matched["market_type"] = "h2h"

    # ── Outrights matching ────────────────────────────────────
    outright_matched = pd.DataFrame()
    if include_outrights:
        # Determina sport_keys de outrights relevantes para as categorias presentes
        outright_keys: set[str] = set()
        for cat in all_cats:
            for pattern, sk in OUTRIGHT_CATEGORY_TO_SPORTS.items():
                if pattern in cat or cat in pattern:
                    outright_keys.add(sk)

        if outright_keys:
            logger.info(f"Buscando outrights: {sorted(outright_keys)}...")
            all_outright_events: list[dict] = []
            for sk in sorted(outright_keys):
                evts = fetch_outright_odds(sk, resolved_key)
                all_outright_events.extend(evts)
                if len(outright_keys) > 1:
                    time.sleep(0.3)

            if all_outright_events:
                logger.info(f"Matching outrights ({len(all_outright_events)} torneios)...")
                outright_matched = match_outright_markets(
                    markets_filtered,
                    all_outright_events,
                    min_score=min_match_score,
                )

    # ── Consolida H2H + outrights ────────────────────────────
    parts = [df for df in [h2h_matched, outright_matched] if not df.empty]
    if not parts:
        logger.warning("Nenhum match encontrado (H2H nem outrights). Tente reduzir --min-match-score.")
        return pd.DataFrame()

    matched = (
        pd.concat(parts, ignore_index=True)
        .sort_values("abs_divergence", ascending=False)
        .reset_index(drop=True)
    )

    result = matched[matched["abs_divergence"] >= min_divergence].reset_index(drop=True)
    n_h2h = (result.get("market_type", pd.Series()) == "h2h").sum()
    n_out = (result.get("market_type", pd.Series()) == "outright").sum()
    logger.info(
        f"Matched: {len(matched):,} total | "
        f"Com edge >= {min_divergence}: {len(result):,} "
        f"(H2H={n_h2h}, outrights={n_out})"
    )

    if save and not result.empty:
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = RAW_ODDS_DIR / f"odds_matched_{ts}.parquet"
        result.to_parquet(path, index=False, compression="snappy")
        logger.info(f"Salvo em {path}")

    return result


def _api_key(api_key: str | None) -> str:
    """Retorna api_key fornecida ou lê do .env."""
    if api_key:
        return api_key
    key = os.getenv("ODDS_API_KEY", "")
    if not key:
        raise ValueError(
            "ODDS_API_KEY não encontrada. Adicione ao .env e obtenha em https://the-odds-api.com"
        )
    return key


def print_matched_table(df: pd.DataFrame, top_n: int = 20) -> None:
    """Exibe tabela de matches e divergências no terminal."""
    console.print("\n[bold cyan]══════════════════════════════════════════════════════════════[/bold cyan]")
    console.print("[bold cyan]   ODDS DIVERGENCE — POLYMARKET vs. BOOKMAKERS               [/bold cyan]")
    console.print("[bold cyan]══════════════════════════════════════════════════════════════[/bold cyan]\n")

    if df.empty:
        console.print("[yellow]Nenhum match encontrado.[/yellow]")
        return

    display = df.head(top_n)
    table = Table(box=box.ROUNDED, show_lines=True)
    table.add_column("#",          width=3,  justify="right")
    table.add_column("Questão",    width=40)
    table.add_column("Dir.",       width=9,  justify="center")
    table.add_column("Poly",       width=7,  justify="right")
    table.add_column("Fair",       width=7,  justify="right")
    table.add_column("Diverg.",    width=8,  justify="right", style="bold")
    table.add_column("Book",       width=12)
    table.add_column("Score",      width=6,  justify="right")
    table.add_column("Liquid.",    width=10, justify="right")

    for i, row in display.iterrows():
        div = row["divergence"]
        div_color  = "red" if div > 0 else "green"
        dir_color  = "magenta" if row["direction"] == "BUY_NO" else "green"
        table.add_row(
            str(i + 1),
            str(row["question"])[:39],
            f"[{dir_color}]{row['direction']}[/{dir_color}]",
            f"{row['yes_price']:.3f}",
            f"{row['fair_prob_yes']:.3f}",
            f"[{div_color}]{div:+.3f}[/{div_color}]",
            str(row["bookmaker"])[:11],
            f"{row['match_score']:.2f}",
            f"${row['liquidity']:,.0f}",
        )

    console.print(table)
    console.print(f"\n[bold]Total matched:[/bold] {len(df):,}  |  Exibindo top {min(top_n, len(df))}")
    if not df.empty:
        console.print(f"[bold]Divergência média:[/bold] {df['abs_divergence'].mean():.3f}")
        console.print(f"[bold]Overround médio:[/bold]   {df['overround'].mean():.3f} "
                      f"(vig ≈ {(df['overround'].mean() - 1) * 100:.1f}%)\n")


# ──────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import click

    @click.command()
    @click.option("--sport",           default=None,  help="Categoria/esporte a filtrar (ex: mlb, nba, nfl).")
    @click.option("--min-divergence",  default=0.03,  type=float, show_default=True,
                  help="Divergência mínima |poly - fair| para exibir.")
    @click.option("--min-liquidity",   default=1_000, type=float, show_default=True,
                  help="Liquidez mínima do mercado Polymarket em USDC.")
    @click.option("--min-match-score", default=0.25,  type=float, show_default=True,
                  help="Score mínimo de similaridade para aceitar match (0–1).")
    @click.option("--top-n",           default=20,    type=int,   show_default=True,
                  help="Quantos matches exibir na tabela.")
    @click.option("--no-save",         is_flag=True,  default=False,
                  help="Não salvar resultado em Parquet.")
    def main(sport, min_divergence, min_liquidity, min_match_score, top_n, no_save):
        """Coleta odds e detecta divergências com mercados Polymarket."""
        logger.remove()
        logger.add(sys.stderr, level="INFO", format="<green>{time:HH:mm:ss}</green> | {message}")

        console.print("\n[bold]Polymarket Quant — Odds Collector[/bold]")
        console.print("Comparando preços Polymarket com odds de bookmakers\n")

        categories = [sport] if sport else None
        df = run(
            min_divergence=min_divergence,
            min_liquidity=min_liquidity,
            min_match_score=min_match_score,
            categories=categories,
            save=not no_save,
        )
        print_matched_table(df, top_n=top_n)

        if not df.empty:
            console.print(
                "\n[dim]Para usar esses sinais no paper trader:[/dim]"
                "\n[dim]  uv run python signals/run_signals.py --mode odds[/dim]"
            )

    main()
