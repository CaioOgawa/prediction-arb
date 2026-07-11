"""
conftest.py
Fixtures compartilhadas entre todos os testes do projeto.
"""

import sqlite3
import pytest
from pathlib import Path


# ---------------------------------------------------------------------------
# Fixtures de ambiente isolado
# ---------------------------------------------------------------------------

@pytest.fixture()
def tmp_db(tmp_path: Path, monkeypatch):
    """
    Banco SQLite temporário isolado do banco real.
    Faz patch de DB_PATH em db.py para apontar ao diretório temporário.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "pipeline"))

    db_path = tmp_path / "test_polymarket.db"
    import db as db_module
    monkeypatch.setattr(db_module, "DB_PATH", db_path)
    db_module.init_db()
    return db_path


@pytest.fixture()
def tmp_raw_dir(tmp_path: Path, monkeypatch):
    """
    Diretório temporário para arquivos Parquet gerados nos testes.
    Faz patch de RAW_DIR em gamma_collector.py.
    """
    raw_dir = tmp_path / "raw" / "markets"
    raw_dir.mkdir(parents=True)

    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "pipeline"))
    import gamma_collector
    monkeypatch.setattr(gamma_collector, "RAW_DIR", raw_dir)
    return raw_dir


# ---------------------------------------------------------------------------
# Dados mock que simulam o payload da Gamma API
# ---------------------------------------------------------------------------

MOCK_MARKET_1 = {
    "id": "1",
    "conditionId": "0xabc001",
    "question": "Will Bitcoin reach $100k by end of 2026?",
    "category": None,
    "volume": 500_000.0,
    "liquidity": 25_000.0,
    "endDate": "2026-12-31T23:59:59Z",
    "active": True,
    "closed": False,
    "archived": False,
    # ATENÇÃO: a Gamma API real retorna estes campos como STRING JSON, não lista.
    # Fixtures com lista mascararam o bug token_yes='[' em produção (2026-07).
    "outcomePrices": '["0.42", "0.58"]',
    "outcomes": '["Yes", "No"]',
    "clobTokenIds": '["token_yes_001", "token_no_001"]',
    "lastTradePrice": 0.42,
    "bestBid": 0.415,
    "bestAsk": 0.425,
    "spread": 0.01,
    "volume24hr": 12_000.0,
    "volume1wk": 80_000.0,
    "volume1mo": 300_000.0,
    "oneDayPriceChange": 0.02,
    "oneWeekPriceChange": -0.05,
    "oneMonthPriceChange": 0.10,
    "restricted": False,
    "resolutionSource": "Coinbase",
    "description": "Resolves YES if BTC closes above $100k.",
    "events": [{"slug": "bitcoin-price-2026", "title": "Bitcoin Price 2026", "category": "Crypto"}],
}

MOCK_MARKET_2 = {
    "id": "2",
    "conditionId": "0xabc002",
    "question": "Will the Fed cut rates in Q2 2026?",
    "category": None,
    "volume": 200_000.0,
    "liquidity": 10_000.0,
    "endDate": "2026-06-30T23:59:59Z",
    "active": True,
    "closed": False,
    "archived": False,
    "outcomePrices": '["0.65", "0.35"]',
    "outcomes": '["Yes", "No"]',
    "clobTokenIds": '["token_yes_002", "token_no_002"]',
    "lastTradePrice": 0.65,
    "bestBid": 0.645,
    "bestAsk": 0.655,
    "spread": 0.01,
    "volume24hr": 8_000.0,
    "volume1wk": 50_000.0,
    "volume1mo": 180_000.0,
    "oneDayPriceChange": -0.01,
    "oneWeekPriceChange": 0.03,
    "oneMonthPriceChange": 0.08,
    "restricted": False,
    "resolutionSource": "Fed announcement",
    "description": "Resolves YES if the Fed announces a rate cut.",
    "events": [{"slug": "fed-rates-2026", "title": "Fed Rates 2026", "category": None}],
}

MOCK_MARKET_3 = {
    "id": "3",
    "conditionId": "0xabc003",
    "question": "Will Ethereum hit $5k in 2026?",
    "category": "Crypto",
    "volume": 750_000.0,
    "liquidity": 40_000.0,
    "endDate": "2026-12-31T23:59:59Z",
    "active": True,
    "closed": False,
    "archived": False,
    "outcomePrices": '["0.30", "0.70"]',
    "outcomes": '["Yes", "No"]',
    "clobTokenIds": '["token_yes_003", "token_no_003"]',
    "lastTradePrice": 0.30,
    "bestBid": 0.295,
    "bestAsk": 0.305,
    "spread": 0.01,
    "volume24hr": 20_000.0,
    "volume1wk": 120_000.0,
    "volume1mo": 450_000.0,
    "oneDayPriceChange": 0.01,
    "oneWeekPriceChange": -0.02,
    "oneMonthPriceChange": 0.05,
    "restricted": False,
    "resolutionSource": "Coinbase",
    "description": "Resolves YES if ETH closes above $5k.",
    "events": [],
}


@pytest.fixture()
def mock_markets():
    """Lista de mercados mock representando um payload real da Gamma API."""
    return [MOCK_MARKET_1, MOCK_MARKET_2, MOCK_MARKET_3]
