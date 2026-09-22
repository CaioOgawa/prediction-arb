"""
test_paper_trader.py
Testes de valor conhecido para load_current_markets() — P0-2 da auditoria de
2026-09-03: a função escolhia o snapshot por NOME (ordem alfabética), não por
mtime, então "markets_incremental_*" sempre vencia "markets_all_*" mesmo
quando o segundo era muito mais recente.
"""

import os
import time
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).parent.parent

from execution import paper_trader


def _write_snapshot(path: Path, mtime: float, n_rows: int = 1) -> None:
    df = pd.DataFrame({"conditionId": [f"0x{i}" for i in range(n_rows)]})
    df.to_parquet(path)
    os.utime(path, (mtime, mtime))


@pytest.fixture()
def tmp_mkt_dir(tmp_path: Path, monkeypatch):
    raw_dir = tmp_path / "raw" / "markets"
    raw_dir.mkdir(parents=True)
    monkeypatch.setattr(paper_trader, "RAW_MKT_DIR", raw_dir)
    return raw_dir


class TestLoadCurrentMarkets:
    def test_escolhe_por_mtime_nao_por_nome(self, tmp_mkt_dir):
        """
        'markets_incremental_*' é lexicograficamente maior que 'markets_all_*'
        ('i' > 'a'), mas aqui o markets_all é o mais NOVO por mtime — tem que
        vencer mesmo assim.
        """
        now = time.time()
        _write_snapshot(tmp_mkt_dir / "markets_incremental_20260401_031129.parquet", now - 155 * 86400, n_rows=1)
        _write_snapshot(tmp_mkt_dir / "markets_all_20260903_213907.parquet", now, n_rows=5)

        df = paper_trader.load_current_markets()

        assert len(df) == 5

    def test_fallback_para_incremental_sem_markets_all(self, tmp_mkt_dir):
        """Sem nenhum markets_all_*, o incremental mais recente ainda deve ser usado."""
        now = time.time()
        _write_snapshot(tmp_mkt_dir / "markets_incremental_20260903_100000.parquet", now, n_rows=3)

        df = paper_trader.load_current_markets()

        assert len(df) == 3

    def test_snapshot_stale_devolve_vazio(self, tmp_mkt_dir):
        """Snapshot mais recente disponível ainda é mais velho que o gate de frescor."""
        stale_ts = time.time() - (paper_trader.MAX_MARKET_SNAPSHOT_AGE_MIN + 30) * 60
        _write_snapshot(tmp_mkt_dir / "markets_all_20260401_000000.parquet", stale_ts, n_rows=5)

        df = paper_trader.load_current_markets()

        assert df.empty

    def test_snapshot_fresco_nao_e_descartado(self, tmp_mkt_dir):
        fresh_ts = time.time() - (paper_trader.MAX_MARKET_SNAPSHOT_AGE_MIN - 10) * 60
        _write_snapshot(tmp_mkt_dir / "markets_all_20260903_213907.parquet", fresh_ts, n_rows=7)

        df = paper_trader.load_current_markets()

        assert len(df) == 7

    def test_sem_candidatos_devolve_vazio(self, tmp_mkt_dir):
        df = paper_trader.load_current_markets()
        assert df.empty


class TestEnrichMarketsForResolution:
    """
    P3-46: current_markets vem de um fetch com closed=false — um mercado que
    já resolveu nunca aparece nele, então uma posição aberta nesse mercado
    fica invisível pra resolve_positions pra sempre. enrich_markets_for_resolution
    completa o buraco consultando a CLOB só pelos condition_id que faltam.
    """

    def test_completa_condition_id_ausente_via_clob(self, monkeypatch):
        open_positions = pd.DataFrame({"condition_id": ["0xaaa", "0xbbb"]})
        current_markets = pd.DataFrame({"conditionId": ["0xaaa"], "closed": [False]})

        def fake_fetch(cid):
            assert cid == "0xbbb"
            return {
                "conditionId": "0xbbb", "closed": True,
                "endDate": "2026-01-01T00:00:00Z", "outcomePrices": [1.0, 0.0],
                "question": "resolvido",
            }

        monkeypatch.setattr(
            "pipeline.clob_collector.fetch_market_by_condition_id", fake_fetch
        )

        result = paper_trader.enrich_markets_for_resolution(open_positions, current_markets)

        assert set(result["conditionId"]) == {"0xaaa", "0xbbb"}
        row = result[result["conditionId"] == "0xbbb"].iloc[0]
        assert bool(row["closed"]) is True
        assert row["outcomePrices"] == [1.0, 0.0]

    def test_sem_posicao_orfa_nao_chama_clob(self, monkeypatch):
        open_positions = pd.DataFrame({"condition_id": ["0xaaa"]})
        current_markets = pd.DataFrame({"conditionId": ["0xaaa"], "closed": [False]})

        def fail_if_called(cid):
            raise AssertionError("não deveria consultar a CLOB sem posição órfã")

        monkeypatch.setattr(
            "pipeline.clob_collector.fetch_market_by_condition_id", fail_if_called
        )

        result = paper_trader.enrich_markets_for_resolution(open_positions, current_markets)

        assert len(result) == 1

    def test_falha_na_clob_nao_quebra_e_ignora_posicao(self, monkeypatch):
        open_positions = pd.DataFrame({"condition_id": ["0xaaa", "0xbbb"]})
        current_markets = pd.DataFrame({"conditionId": ["0xaaa"], "closed": [False]})

        monkeypatch.setattr(
            "pipeline.clob_collector.fetch_market_by_condition_id", lambda cid: None
        )

        result = paper_trader.enrich_markets_for_resolution(open_positions, current_markets)

        assert set(result["conditionId"]) == {"0xaaa"}

    def test_open_positions_ou_current_markets_vazio_devolve_current_markets(self):
        current_markets = pd.DataFrame({"conditionId": ["0xaaa"], "closed": [False]})

        result = paper_trader.enrich_markets_for_resolution(pd.DataFrame(), current_markets)

        assert result is current_markets
