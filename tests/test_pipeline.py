"""
test_pipeline.py
Testes do pipeline de dados do Polymarket.

Estrutura:
  - TestDB:               schema SQLite, init, upsert, deduplicação
  - TestParsing:          normalização do payload da Gamma API
  - TestIncremental:      detecção de novo/atualizado/inalterado
  - TestParquet:          geração e leitura de arquivos Parquet
  - TestIntegration:      testes de ponta a ponta com a API real (marcados @integration)

Rodar apenas unitários (rápidos, sem rede):
    uv run pytest tests/ -v -m "not integration"

Rodar tudo incluindo integração (~60s):
    uv run pytest tests/ -v
"""

import sqlite3
import sys
from pathlib import Path

import pandas as pd
import pytest

# Garante que pipeline está no path
sys.path.insert(0, str(Path(__file__).parent.parent / "pipeline"))


# ===========================================================================
# TestDB — banco SQLite
# ===========================================================================

class TestDB:
    def test_init_creates_tables(self, tmp_db):
        """init_db deve criar as três tabelas do schema."""
        conn = sqlite3.connect(tmp_db)
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        conn.close()
        assert "pipeline_runs"        in tables
        assert "market_registry"      in tables
        assert "orderbook_snapshots"  in tables

    def test_init_is_idempotent(self, tmp_db):
        """Chamar init_db duas vezes não deve lançar erro nem duplicar tabelas."""
        import db
        db.init_db()
        db.init_db()  # segunda chamada — deve ser silenciosa

        conn = sqlite3.connect(tmp_db)
        # Exclui tabelas internas do SQLite (ex: sqlite_sequence criada pelo AUTOINCREMENT)
        count = conn.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        ).fetchone()[0]
        conn.close()
        assert count == 3

    def test_upsert_inserts_new_market(self, tmp_db, mock_markets):
        """upsert_to_db deve inserir mercados novos no market_registry."""
        import gamma_collector
        df = gamma_collector._parse_markets(mock_markets)
        n = gamma_collector.upsert_to_db(df)

        conn = sqlite3.connect(tmp_db)
        count = conn.execute("SELECT COUNT(*) FROM market_registry").fetchone()[0]
        conn.close()

        assert n == 3
        assert count == 3

    def test_upsert_no_duplicates(self, tmp_db, mock_markets):
        """Inserir os mesmos mercados duas vezes não deve duplicar registros."""
        import gamma_collector
        df = gamma_collector._parse_markets(mock_markets)
        gamma_collector.upsert_to_db(df)
        gamma_collector.upsert_to_db(df)  # segunda vez

        conn = sqlite3.connect(tmp_db)
        count = conn.execute("SELECT COUNT(*) FROM market_registry").fetchone()[0]
        conn.close()

        assert count == 3  # ainda 3, não 6

    def test_upsert_updates_existing_price(self, tmp_db, mock_markets):
        """upsert deve atualizar yes_price se o mercado já existia."""
        import gamma_collector

        # Insere versão inicial
        df_initial = gamma_collector._parse_markets(mock_markets)
        gamma_collector.upsert_to_db(df_initial)

        # Cria versão atualizada com preço diferente
        updated = [dict(mock_markets[0])]
        updated[0]["lastTradePrice"] = 0.99
        df_updated = gamma_collector._parse_markets(updated)
        gamma_collector.upsert_to_db(df_updated)

        conn = sqlite3.connect(tmp_db)
        row = conn.execute(
            "SELECT yes_price FROM market_registry WHERE condition_id = ?",
            ("0xabc001",),
        ).fetchone()
        conn.close()

        assert abs(row[0] - 0.99) < 0.001

    def test_upsert_preserves_first_seen(self, tmp_db, mock_markets):
        """first_seen não deve ser sobrescrito em upserts subsequentes."""
        import gamma_collector
        df = gamma_collector._parse_markets(mock_markets)
        gamma_collector.upsert_to_db(df)

        conn = sqlite3.connect(tmp_db)
        first = conn.execute(
            "SELECT first_seen FROM market_registry WHERE condition_id = ?",
            ("0xabc001",),
        ).fetchone()[0]
        conn.close()

        # Segundo upsert
        gamma_collector.upsert_to_db(df)

        conn = sqlite3.connect(tmp_db)
        first_after = conn.execute(
            "SELECT first_seen FROM market_registry WHERE condition_id = ?",
            ("0xabc001",),
        ).fetchone()[0]
        conn.close()

        assert first == first_after  # não mudou


# ===========================================================================
# TestParsing — normalização do payload
# ===========================================================================

class TestParsing:
    def test_parse_returns_dataframe(self, mock_markets):
        """_parse_markets deve retornar um DataFrame não vazio."""
        import gamma_collector
        df = gamma_collector._parse_markets(mock_markets)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3

    def test_parse_yes_price_from_last_trade(self, mock_markets):
        """yes_price deve vir de lastTradePrice quando disponível."""
        import gamma_collector
        df = gamma_collector._parse_markets([mock_markets[0]])
        assert abs(df.iloc[0]["yes_price"] - 0.42) < 0.001

    def test_parse_yes_price_fallback_outcome_prices(self):
        """Sem lastTradePrice, deve usar outcomePrices[0] como fallback."""
        import gamma_collector
        market = {
            "conditionId": "0xfallback",
            "question": "Fallback test?",
            "outcomePrices": ["0.75", "0.25"],
            "lastTradePrice": None,
            "clobTokenIds": ["t1", "t2"],
            "volume": 1000.0,
            "liquidity": 500.0,
            "events": [],
        }
        df = gamma_collector._parse_markets([market])
        assert abs(df.iloc[0]["yes_price"] - 0.75) < 0.001

    def test_parse_token_ids_extracted(self, mock_markets):
        """token_yes e token_no devem ser extraídos de clobTokenIds."""
        import gamma_collector
        df = gamma_collector._parse_markets([mock_markets[0]])
        assert df.iloc[0]["token_yes"] == "token_yes_001"
        assert df.iloc[0]["token_no"]  == "token_no_001"

    def test_parse_category_from_event(self, mock_markets):
        """Categoria deve ser herdada do evento pai quando ausente no mercado."""
        import gamma_collector
        df = gamma_collector._parse_markets([mock_markets[0]])
        assert df.iloc[0]["category"] == "Crypto"

    def test_parse_category_direct(self, mock_markets):
        """Categoria direta no mercado deve ser usada sem consultar evento."""
        import gamma_collector
        df = gamma_collector._parse_markets([mock_markets[2]])
        assert df.iloc[0]["category"] == "Crypto"

    def test_parse_numeric_volumes(self, mock_markets):
        """volume, liquidity e volume24hr devem ser float."""
        import gamma_collector
        df = gamma_collector._parse_markets(mock_markets)
        assert df["volume"].dtype == float
        assert df["liquidity"].dtype == float
        assert df["volume24hr"].dtype == float

    def test_parse_missing_condition_id(self):
        """Mercado sem conditionId deve ser incluído no DataFrame mas sem cid."""
        import gamma_collector
        market = {"question": "No ID?", "volume": 100.0, "events": []}
        df = gamma_collector._parse_markets([market])
        assert len(df) == 1
        assert df.iloc[0].get("conditionId") is None


# ===========================================================================
# TestIncremental — detecção de novo/atualizado/inalterado
# ===========================================================================

class TestIncremental:
    def _make_known(self, df: pd.DataFrame) -> dict:
        """Simula o estado do DB a partir de um DataFrame."""
        return {
            row["conditionId"]: {
                "volume":    row["volume"],
                "liquidity": row["liquidity"],
                "yes_price": row["yes_price"],
            }
            for _, row in df.iterrows()
            if row.get("conditionId")
        }

    def test_all_new_when_db_empty(self, mock_markets):
        """Com DB vazio, todos os mercados devem ser detectados como novos."""
        import gamma_collector
        from run_pipeline import _diff_markets

        df = gamma_collector._parse_markets(mock_markets)
        new_df, updated_df, unchanged_df = _diff_markets(df, known={})

        assert len(new_df)       == 3
        assert len(updated_df)   == 0
        assert len(unchanged_df) == 0

    def test_unchanged_after_same_fetch(self, mock_markets):
        """Após upsert inicial, re-buscar os mesmos dados → tudo inalterado."""
        import gamma_collector
        from run_pipeline import _diff_markets

        df = gamma_collector._parse_markets(mock_markets)
        known = self._make_known(df)

        new_df, updated_df, unchanged_df = _diff_markets(df, known)

        assert len(new_df)       == 0
        assert len(updated_df)   == 0
        assert len(unchanged_df) == 3

    def test_detects_price_change(self, mock_markets):
        """Mudança de yes_price >= 0.001 deve ser detectada como atualização."""
        import gamma_collector
        from run_pipeline import _diff_markets

        df_original = gamma_collector._parse_markets(mock_markets)
        known = self._make_known(df_original)

        # Altera preço do primeiro mercado
        updated_raw = [dict(mock_markets[0])]
        updated_raw[0]["lastTradePrice"] = 0.42 + 0.05  # mudança de 0.05
        df_updated = gamma_collector._parse_markets(updated_raw + mock_markets[1:])

        new_df, updated_df, unchanged_df = _diff_markets(df_updated, known)

        assert len(new_df)       == 0
        assert len(updated_df)   == 1
        assert len(unchanged_df) == 2
        assert updated_df.iloc[0]["conditionId"] == "0xabc001"

    def test_detects_volume_change(self, mock_markets):
        """Aumento de volume >= 1 USDC deve ser detectado como atualização."""
        import gamma_collector
        from run_pipeline import _diff_markets

        df = gamma_collector._parse_markets(mock_markets)
        known = self._make_known(df)

        updated_raw = [dict(mock_markets[1])]
        updated_raw[0]["volume"] = mock_markets[1]["volume"] + 10_000
        df_updated = gamma_collector._parse_markets([mock_markets[0]] + updated_raw + [mock_markets[2]])

        new_df, updated_df, unchanged_df = _diff_markets(df_updated, known)

        assert len(updated_df) == 1
        assert updated_df.iloc[0]["conditionId"] == "0xabc002"

    def test_detects_new_market(self, mock_markets):
        """Mercado com conditionId desconhecido deve aparecer como novo."""
        import gamma_collector
        from run_pipeline import _diff_markets

        df_first = gamma_collector._parse_markets(mock_markets[:2])
        known = self._make_known(df_first)

        # Adiciona terceiro mercado que ainda não estava no DB
        df_all = gamma_collector._parse_markets(mock_markets)
        new_df, updated_df, unchanged_df = _diff_markets(df_all, known)

        assert len(new_df) == 1
        assert new_df.iloc[0]["conditionId"] == "0xabc003"

    def test_noise_below_threshold_not_detected(self, mock_markets):
        """Mudança de volume < 1 USDC não deve ser tratada como atualização."""
        import gamma_collector
        from run_pipeline import _diff_markets

        df = gamma_collector._parse_markets(mock_markets)
        known = self._make_known(df)

        # Variação de 0.5 USDC — abaixo do threshold de 1.0
        updated_raw = [dict(mock_markets[0])]
        updated_raw[0]["volume"] = mock_markets[0]["volume"] + 0.5
        df_noise = gamma_collector._parse_markets(updated_raw + mock_markets[1:])

        new_df, updated_df, unchanged_df = _diff_markets(df_noise, known)

        assert len(updated_df) == 0
        assert len(unchanged_df) == 3


# ===========================================================================
# TestParquet — persistência em Parquet
# ===========================================================================

class TestParquet:
    def test_snapshot_creates_file(self, tmp_raw_dir, tmp_db, mock_markets):
        """save_snapshot deve criar um arquivo .parquet no diretório correto."""
        import gamma_collector
        df = gamma_collector._parse_markets(mock_markets)
        path = gamma_collector.save_snapshot(df, tag="test")

        assert path.exists()
        assert path.suffix == ".parquet"
        assert "test" in path.name

    def test_parquet_roundtrip(self, tmp_raw_dir, tmp_db, mock_markets):
        """DataFrame salvo em Parquet deve ser idêntico ao relido."""
        import gamma_collector
        df = gamma_collector._parse_markets(mock_markets)
        path = gamma_collector.save_snapshot(df, tag="roundtrip")

        df_read = pd.read_parquet(path)
        assert len(df_read) == len(df)
        assert set(df_read["conditionId"]) == {"0xabc001", "0xabc002", "0xabc003"}

    def test_parquet_preserves_numeric_types(self, tmp_raw_dir, tmp_db, mock_markets):
        """Tipos numéricos devem ser preservados após salvar/ler Parquet."""
        import gamma_collector
        df = gamma_collector._parse_markets(mock_markets)
        path = gamma_collector.save_snapshot(df, tag="types")
        df_read = pd.read_parquet(path)

        assert df_read["volume"].dtype    in [float, "float64"]
        assert df_read["yes_price"].dtype in [float, "float64"]


# ===========================================================================
# TestPipelineRun — pipeline_runs log
# ===========================================================================

class TestPipelineRun:
    def test_run_log_created(self, tmp_db):
        """_log_run_start deve criar registro com status 'running'."""
        from run_pipeline import _log_run_start, _log_run_end
        run_id = _log_run_start("test_run")

        conn = sqlite3.connect(tmp_db)
        row = conn.execute(
            "SELECT status, run_type FROM pipeline_runs WHERE id = ?", (run_id,)
        ).fetchone()
        conn.close()

        assert row[0] == "running"
        assert row[1] == "test_run"

    def test_run_log_finished(self, tmp_db):
        """_log_run_end deve atualizar status para 'ok'."""
        from run_pipeline import _log_run_start, _log_run_end
        run_id = _log_run_start("test_run")
        _log_run_end(run_id, n_records=42, status="ok", notes="test")

        conn = sqlite3.connect(tmp_db)
        row = conn.execute(
            "SELECT status, n_records, notes FROM pipeline_runs WHERE id = ?", (run_id,)
        ).fetchone()
        conn.close()

        assert row[0] == "ok"
        assert row[1] == 42
        assert "test" in row[2]


# ===========================================================================
# TestDbMaintenance — retenção (P2-37)
# ===========================================================================

class TestDbMaintenance:
    """P2-37: outputs/reports e data/raw/markets nunca tinham retenção —
    load_current_markets()/load_signals() só leem o arquivo mais recente de
    cada padrão, então tudo mais velho é lixo puro. Chegou a 15 GB /
    11.556 arquivos em outputs/reports."""

    def _touch(self, path: Path, days_old: float):
        path.write_text("x")
        import os
        old_ts = __import__("time").time() - days_old * 86400
        os.utime(path, (old_ts, old_ts))

    def test_prune_report_files_remove_so_padroes_conhecidos_e_antigos(self, tmp_path):
        import db_maintenance
        reports = tmp_path / "reports"
        raw = tmp_path / "raw"
        reports.mkdir()
        raw.mkdir()

        old_eda = reports / "eda_markets_20260101_000000.csv"
        self._touch(old_eda, days_old=30)
        new_eda = reports / "eda_markets_20260901_000000.csv"
        self._touch(new_eda, days_old=1)
        # Não está em REPORT_PATTERNS — não deve ser tocado mesmo sendo velho.
        old_unrelated = reports / "notas_manuais.txt"
        self._touch(old_unrelated, days_old=30)

        old_resolved = raw / "resolved_markets_v2_20260401_000000.parquet"
        self._touch(old_resolved, days_old=200)
        old_incremental = raw / "markets_incremental_20260101_000000.parquet"
        self._touch(old_incremental, days_old=30)
        # markets_all_* fora de RAW_MARKET_PATTERNS de propósito — sim_backtest.py
        # lê o mais VELHO desses de propósito (janela histórica walk-forward);
        # apagar por idade já destruiu essa janela uma vez (2026-09-04) sem
        # nenhum arquivamento equivalente pra recuperar depois.
        old_full_snapshot = raw / "markets_all_20260101_000000.parquet"
        self._touch(old_full_snapshot, days_old=30)

        result = db_maintenance.prune_report_files(
            reports_dir=reports, raw_markets_dir=raw, days=14, dry_run=False,
        )

        assert not old_eda.exists()
        assert new_eda.exists()
        assert old_unrelated.exists()      # padrão desconhecido — nunca apagado
        assert old_resolved.exists()       # dataset do ml_lab — fora dos padrões, nunca apagado
        assert not old_incremental.exists()
        assert old_full_snapshot.exists()  # markets_all_* nunca é apagado automaticamente
        assert result["files_removed"] == 2

    def test_prune_report_files_dry_run_nao_apaga(self, tmp_path):
        import db_maintenance
        reports = tmp_path / "reports"
        reports.mkdir()
        old_eda = reports / "eda_markets_20260101_000000.csv"
        self._touch(old_eda, days_old=30)

        result = db_maintenance.prune_report_files(
            reports_dir=reports, raw_markets_dir=tmp_path / "raw", days=14, dry_run=True,
        )

        assert old_eda.exists()
        assert result["files_removed"] == 1
        assert result["removed"] == [old_eda.name]

    def test_prune_price_history_dry_run_nao_apaga_nem_faz_vacuum(self, tmp_path):
        import db_maintenance
        db = tmp_path / "paper.db"
        conn = sqlite3.connect(db)
        conn.execute("""
            CREATE TABLE price_history (
                condition_id TEXT, ts TEXT, best_bid REAL, best_ask REAL
            )
        """)
        conn.execute(
            "INSERT INTO price_history VALUES ('0xa', datetime('now', '-30 days'), 0.4, 0.42)"
        )
        conn.execute(
            "INSERT INTO price_history VALUES ('0xb', datetime('now'), 0.001, 0.999)"
        )  # corrompido (spread > 0.5), mas recente
        conn.commit()
        conn.close()

        result = db_maintenance.prune_price_history(days=14, db_path=db, dry_run=True)

        conn = sqlite3.connect(db)
        n = conn.execute("SELECT COUNT(*) FROM price_history").fetchone()[0]
        conn.close()
        assert n == 2  # nada removido
        assert result["removed_old"] == 1
        assert result["removed_corrupt"] == 1

    def test_prune_price_history_apaga_antigas_e_corrompidas(self, tmp_path):
        import db_maintenance
        db = tmp_path / "paper.db"
        conn = sqlite3.connect(db)
        conn.execute("""
            CREATE TABLE price_history (
                condition_id TEXT, ts TEXT, best_bid REAL, best_ask REAL
            )
        """)
        conn.execute(
            "INSERT INTO price_history VALUES ('0xa', datetime('now', '-30 days'), 0.4, 0.42)"
        )
        conn.execute(
            "INSERT INTO price_history VALUES ('0xb', datetime('now'), 0.001, 0.999)"
        )
        conn.execute(
            "INSERT INTO price_history VALUES ('0xc', datetime('now'), 0.40, 0.42)"
        )  # recente e sã — sobrevive
        conn.commit()
        conn.close()

        db_maintenance.prune_price_history(days=14, db_path=db, dry_run=False)

        conn = sqlite3.connect(db)
        rows = conn.execute("SELECT condition_id FROM price_history").fetchall()
        conn.close()
        assert [r[0] for r in rows] == ["0xc"]


# ===========================================================================
# TestIntegration — testes com a API real (requerem rede)
# ===========================================================================

@pytest.mark.integration
class TestIntegration:
    def test_gamma_api_returns_markets(self):
        """Gamma API deve retornar ao menos 10 mercados ativos."""
        import gamma_collector
        df = gamma_collector.fetch_markets(active=True, min_volume=0, limit=20, max_pages=1)
        assert len(df) >= 10, "Gamma API retornou menos de 10 mercados"

    def test_gamma_api_has_required_columns(self):
        """DataFrame retornado deve ter as colunas essenciais para o pipeline."""
        import gamma_collector
        df = gamma_collector.fetch_markets(active=True, min_volume=0, limit=10, max_pages=1)
        required = {"conditionId", "question", "volume", "liquidity", "yes_price"}
        assert required.issubset(set(df.columns))

    def test_gamma_api_prices_in_range(self):
        """yes_price deve estar entre 0 e 1 para mercados CLOB ativos."""
        import gamma_collector
        df = gamma_collector.fetch_markets(active=True, min_volume=1_000, limit=50, max_pages=1)
        prices = df["yes_price"].dropna()
        assert (prices >= 0).all(),  "yes_price com valor negativo encontrado"
        assert (prices <= 1).all(),  "yes_price acima de 1.0 encontrado"

    def test_full_pipeline_incremental_run(self, tmp_db, tmp_raw_dir):
        """
        Pipeline completo com a API real:
        1ª run → tudo novo
        2ª run → tudo inalterado (mesmo fetch)
        """
        from run_pipeline import run_incremental

        metrics1 = run_incremental(min_volume=100_000, dry_run=False)
        assert metrics1["n_new"]   > 0
        assert metrics1["n_total"] > 0

        # Segunda execução imediata — sem mudanças reais, deve ser tudo inalterado
        # (pequenas variações de preço podem gerar alguns "updated")
        metrics2 = run_incremental(min_volume=100_000, dry_run=False)
        assert metrics2["n_new"] == 0
        assert metrics2["n_total"] == metrics1["n_total"]


# ===========================================================================
# TestValidateConnection — P2-42: validador de .env e conectividade
# ===========================================================================

class _FakeResp:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code
    def raise_for_status(self):
        if self.status_code >= 400:
            raise Exception(f"HTTP {self.status_code}")
    def json(self):
        return self._payload


class TestValidateConnection:
    """P2-42: test_thegraph devolvia True em todo caminho, inclusive falha —
    o __main__ então anunciava "todas as conexões validadas com sucesso"
    mesmo com o TheGraph fora do ar. E check_env_vars checava
    POLY_PROXY_WALLET/NEWSAPI_KEY (sem leitor nenhum no código) mas não
    ODDS_API_KEY (a única credencial cuja ausência de fato derruba
    odds_collector.py)."""

    def test_thegraph_ok_devolve_true(self, monkeypatch):
        import validate_connection as vc
        monkeypatch.setattr(
            vc.requests, "post",
            lambda *a, **kw: _FakeResp({"data": {"fixedProductMarketMakers": []}}),
        )
        assert vc.test_thegraph() is True

    def test_thegraph_erro_de_rede_devolve_false(self, monkeypatch):
        import validate_connection as vc
        def boom(*a, **kw):
            raise Exception("Connection refused")
        monkeypatch.setattr(vc.requests, "post", boom)
        assert vc.test_thegraph() is False

    def test_thegraph_resposta_com_errors_devolve_false(self, monkeypatch):
        import validate_connection as vc
        monkeypatch.setattr(
            vc.requests, "post",
            lambda *a, **kw: _FakeResp({"errors": ["subgraph indisponível"]}),
        )
        assert vc.test_thegraph() is False

    def test_check_env_vars_inclui_odds_api_key(self, capsys):
        import validate_connection as vc
        vc.check_env_vars()
        out = capsys.readouterr().out
        assert "ODDS_API_KEY" in out

    def test_check_env_vars_nao_checa_vars_mortas(self, capsys):
        import validate_connection as vc
        vc.check_env_vars()
        out = capsys.readouterr().out
        assert "POLY_PROXY_WALLET" not in out
        assert "NEWSAPI_KEY" not in out
