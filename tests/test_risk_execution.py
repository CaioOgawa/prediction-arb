"""
test_risk_execution.py
Testes de risco e execução — cada classe cobre um incidente REAL de 2026-05/07.

Origem (auditoria 2026-07-09, ver AUDITORIA_2026-07-09.md):
  - TestBookParsing / TestEvaluateExit → ws_feed executou 47 posições a 0.001
    (bids[0] tratado como best bid + stop loss sem min_hold em book vazio)
  - TestGammaKeyset → universo travado em 100 mercados por 2 meses
    (offset capado + páginas parciais legítimas no meio do stream)
  - TestAliases → chaves duplicadas no TEAM_ALIASES se sobrescreviam
    ("spurs" → Tottenham engoliu o San Antonio Spurs)
  - TestTouchDirection → "hit $50k" com spot acima era barreira upward P≈1
  - TestDoubleCredit → exits concorrentes creditavam o cash em dobro
"""

import ast
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
sys.path.insert(0, str(ROOT / "risk"))

import ws_feed
import gamma_collector
import odds_collector
import deribit_collector
import market_pricing
import risk_manager


# ──────────────────────────────────────────────────────────
# ws_feed — parsing do book e gatilhos de exit
# ──────────────────────────────────────────────────────────

class TestBookParsing:
    def test_best_levels_ignora_ordem_dos_niveis(self):
        """bids[0] é o PIOR bid na CLOB — best = max(bids)/min(asks)."""
        bids = [{"price": "0.001"}, {"price": "0.44"}, {"price": "0.45"}]
        asks = [{"price": "0.999"}, {"price": "0.47"}, {"price": "0.46"}]
        assert ws_feed.best_levels(bids, asks) == (0.45, 0.46)

    def test_best_levels_book_vazio(self):
        assert ws_feed.best_levels([], []) == (None, None)


def _pos(direction="BUY_YES", entry=0.49, shares=13.0, cost=6.4,
         hold_h=0.1, trade_type="value") -> "ws_feed.AssetInfo":
    return ws_feed.AssetInfo(
        "tok", "cid", "pergunta", direction, entry, shares, cost, 1,
        trade_type, time.time() - hold_h * 3600,
    )


class TestEvaluateExit:
    def test_book_poeira_nunca_dispara_exit(self):
        """O cenário exato que matou as 47 posições: book 0.001/0.999."""
        assert ws_feed.evaluate_exit(_pos(hold_h=100), 0.001, 0.999) is None

    def test_book_largo_nunca_dispara_exit(self):
        """Spread > MAX_EXIT_SPREAD = book ilíquido, não preço real."""
        assert ws_feed.evaluate_exit(_pos(hold_h=100), 0.30, 0.55) is None

    def test_stop_loss_apos_min_hold_sai_no_bid(self):
        r = ws_feed.evaluate_exit(_pos(entry=0.49, hold_h=5), 0.18, 0.19)
        assert r is not None and "stop_loss" in r[1]
        assert r[0] == 0.18  # exit no BID, não bid - spread/2

    def test_stop_loss_bloqueado_dentro_do_min_hold(self):
        assert ws_feed.evaluate_exit(_pos(entry=0.49, hold_h=0.5), 0.18, 0.19) is None

    def test_profit_target_dispara_a_qualquer_momento(self):
        r = ws_feed.evaluate_exit(
            _pos(entry=0.20, shares=32, cost=6.4, hold_h=0.1), 0.55, 0.56
        )
        assert r is not None and "profit_target" in r[1]

    def test_sem_opened_ts_nao_arrisca_exit_por_perda(self):
        info = ws_feed.AssetInfo("t", "c", "q", "BUY_YES", 0.49, 13, 6.4, 1, "value", 0.0)
        assert ws_feed.evaluate_exit(info, 0.18, 0.19) is None

    def test_watch_nunca_dispara(self):
        info = ws_feed.AssetInfo("t", "c", "q", "WATCH", 0, 0, 0, 0)
        assert ws_feed.evaluate_exit(info, 0.18, 0.19) is None


# ──────────────────────────────────────────────────────────
# gamma_collector — paginação keyset
# ──────────────────────────────────────────────────────────

class _FakeResp:
    def __init__(self, payload):
        self._payload = payload
    def raise_for_status(self):
        pass
    def json(self):
        return self._payload


def _mk_market(i: int) -> dict:
    return {
        "conditionId": f"0x{i:04d}",
        "question": f"Mercado {i}?",
        "volume": 10_000.0,
        "liquidity": 5_000.0,
        "clobTokenIds": f'["tok_yes_{i}", "tok_no_{i}"]',
        "outcomePrices": '["0.50", "0.50"]',
        "lastTradePrice": 0.5,
        "events": [],
    }


class TestGammaKeyset:
    def test_pagina_parcial_no_meio_nao_encerra(self, monkeypatch):
        """A API filtra DEPOIS de fatiar: página com 40 itens no meio é normal.
        O bug antigo (len(batch) < limit → break) truncava o universo."""
        pages = [
            {"markets": [_mk_market(i) for i in range(100)],      "next_cursor": "c1"},
            {"markets": [_mk_market(i) for i in range(100, 140)], "next_cursor": "c2"},
            {"markets": [_mk_market(i) for i in range(140, 200)], "next_cursor": None},
        ]
        calls = []

        def fake_get(url, params=None, timeout=None):
            assert "/markets/keyset" in url
            calls.append(dict(params or {}))
            idx = len(calls) - 1
            return _FakeResp(pages[min(idx, len(pages) - 1)])

        monkeypatch.setattr(gamma_collector.requests, "get", fake_get)
        monkeypatch.setattr(gamma_collector.time, "sleep", lambda s: None)

        df = gamma_collector.fetch_markets(min_volume=0)
        assert len(df) == 200          # 100 + 40 + 60 — nada truncado
        assert len(calls) == 3
        assert calls[1]["after_cursor"] == "c1"
        assert calls[2]["after_cursor"] == "c2"

    def test_para_em_pagina_vazia(self, monkeypatch):
        pages = [
            {"markets": [_mk_market(i) for i in range(30)], "next_cursor": "c1"},
            {"markets": [], "next_cursor": "c2"},
        ]
        calls = []

        def fake_get(url, params=None, timeout=None):
            calls.append(1)
            return _FakeResp(pages[min(len(calls) - 1, len(pages) - 1)])

        monkeypatch.setattr(gamma_collector.requests, "get", fake_get)
        monkeypatch.setattr(gamma_collector.time, "sleep", lambda s: None)

        df = gamma_collector.fetch_markets(min_volume=0)
        assert len(df) == 30 and len(calls) == 2

    def test_resposta_de_erro_nao_explode(self, monkeypatch):
        monkeypatch.setattr(
            gamma_collector.requests, "get",
            lambda url, params=None, timeout=None: _FakeResp(
                {"type": "validation error", "error": "offset too large"}
            ),
        )
        df = gamma_collector.fetch_markets(min_volume=0)
        assert df.empty

    def test_falha_transitoria_recupera_via_retry(self, monkeypatch):
        """P0-6: uma falha de rede isolada não pode derrubar a coleta inteira
        — só a página falha, tenta de novo, e segue de onde parou."""
        import requests as _requests

        pages = [
            {"markets": [_mk_market(i) for i in range(50)], "next_cursor": None},
        ]
        calls = {"n": 0}

        def fake_get(url, params=None, timeout=None):
            calls["n"] += 1
            if calls["n"] == 1:
                raise _requests.exceptions.ConnectionError("Failed to resolve host")
            return _FakeResp(pages[0])

        monkeypatch.setattr(gamma_collector.requests, "get", fake_get)
        monkeypatch.setattr(gamma_collector.time, "sleep", lambda s: None)
        df = gamma_collector.fetch_markets(min_volume=0)
        assert len(df) == 50
        assert calls["n"] == 2  # 1 falha + 1 sucesso no retry

    def test_falha_persistente_nao_trava_alem_do_limite(self, monkeypatch):
        """Após esgotar PAGE_MAX_RETRIES, desiste da página (não trava para sempre)."""
        import requests as _requests

        calls = {"n": 0}

        def fake_get(url, params=None, timeout=None):
            calls["n"] += 1
            raise _requests.exceptions.ConnectionError("Failed to resolve host")

        monkeypatch.setattr(gamma_collector.requests, "get", fake_get)
        monkeypatch.setattr(gamma_collector.time, "sleep", lambda s: None)
        df = gamma_collector.fetch_markets(min_volume=0)
        assert df.empty
        assert calls["n"] == gamma_collector.PAGE_MAX_RETRIES


class TestSaveSnapshotNaoEnvenena:
    """
    P0-6: um snapshot vazio (falha transitória da Gamma API) gravado sob
    markets_all_*/markets_incremental_* vira "o mais recente" para TODO
    consumidor do sistema por até o próximo ciclo bem-sucedido — foi
    exatamente isso que parou o pipeline inteiro (100% dos ciclos de um
    dia com universo zerado, sem nenhum alerta).
    """

    def test_df_vazio_nao_grava_sob_nome_padrao(self, tmp_path, monkeypatch):
        monkeypatch.setattr(gamma_collector, "RAW_DIR", tmp_path)
        with pytest.raises(gamma_collector.EmptySnapshotError):
            gamma_collector.save_snapshot(pd.DataFrame(), tag="all")

        # Nada sob o nome que os loaders (mtime mais recente) enxergam
        assert list(tmp_path.glob("markets_all_*.parquet")) == []
        # O snapshot vazio ainda existe, mas sob um nome que ninguém glob-a
        assert list(tmp_path.glob("markets_partial_all_*.parquet"))

    def test_df_nao_vazio_grava_normalmente(self, tmp_path, monkeypatch):
        monkeypatch.setattr(gamma_collector, "RAW_DIR", tmp_path)
        df = pd.DataFrame([_mk_market(0)])
        path = gamma_collector.save_snapshot(df, tag="all")
        assert path.exists()
        assert list(tmp_path.glob("markets_all_*.parquet")) == [path]


# ──────────────────────────────────────────────────────────
# odds_collector — aliases e matching
# ──────────────────────────────────────────────────────────

class TestAliases:
    def test_sem_chaves_duplicadas_nos_dict_literais(self):
        """Chave duplicada em dict literal se sobrescreve em silêncio — foi
        assim que 'spurs' virou Tottenham e o San Antonio sumiu."""
        src = (ROOT / "pipeline/odds_collector.py").read_text()
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.Dict):
                keys = [k.value for k in node.keys if isinstance(k, ast.Constant)]
                dupes = {k for k in keys if keys.count(k) > 1}
                assert not dupes, f"chaves duplicadas em dict (linha {node.lineno}): {dupes}"

    def test_alias_ambiguo_resolvido_por_esporte(self):
        assert odds_collector._normalize_team("spurs", "basketball_nba") == "san antonio spurs"
        assert odds_collector._normalize_team("spurs", "soccer_epl") == "tottenham hotspur"
        assert odds_collector._normalize_team("rangers", "baseball_mlb") == "texas rangers"
        assert odds_collector._normalize_team("rangers", "icehockey_nhl") == "new york rangers"

    def test_pergunta_de_um_time_passa(self):
        """Formato dominante do Polymarket: 'Will X win on DATE?' cita só um time."""
        s = odds_collector._match_score(
            "Will Manchester United FC win on 2026-05-09?",
            "Manchester United", "Liverpool", "soccer_epl",
        )
        assert s >= 0.25

    def test_pergunta_nao_relacionada_falha(self):
        s = odds_collector._match_score(
            "Will Bitcoin reach 100k?", "Manchester United", "Liverpool", "soccer_epl",
        )
        assert s == 0.0

    def test_underlying_key_independe_da_ordem_dos_times(self):
        # P1-11: "A vs B" e "B vs A" são o mesmo jogo — mesma chave de correlação.
        k1 = odds_collector._underlying_key("basketball_nba", "Lakers", "Celtics")
        k2 = odds_collector._underlying_key("basketball_nba", "Celtics", "Lakers")
        assert k1 == k2

    def test_underlying_key_usa_nome_canonico(self):
        # "Spurs" ambíguo — o esporte já resolve para o time canônico certo,
        # então dois apelidos do mesmo time caem na mesma chave.
        k1 = odds_collector._underlying_key("basketball_nba", "Spurs", "Lakers")
        k2 = odds_collector._underlying_key("basketball_nba", "San Antonio Spurs", "Lakers")
        assert k1 == k2 == "basketball_nba:los angeles lakersvsan antonio spurs"

    def test_underlying_key_outright_sem_oponente(self):
        k = odds_collector._underlying_key("basketball_nba", "Lakers")
        assert k == "basketball_nba:los angeles lakers"


# ──────────────────────────────────────────────────────────
# deribit_collector — direção de touch options
# ──────────────────────────────────────────────────────────

class TestTouchDirection:
    def test_dip_detectado_como_below_touch(self):
        assert deribit_collector._detect_direction(
            "Will Bitcoin dip to $50,000 by December 31?"
        ) == ("below", True)

    def test_reach_detectado_como_touch(self):
        direction, is_touch = deribit_collector._detect_direction(
            "Will Bitcoin reach $150,000 by June 30?"
        )
        assert is_touch is True

    def test_barreira_downward_nao_da_prob_1(self):
        """Bug original: 'hit $50k' com spot $100k usava fórmula upward → P≈1."""
        p = deribit_collector.bs_prob(
            S=100_000, K=50_000, T=0.5, sigma=0.6, r=0.55, above=False, touch=True,
        )
        assert 0.0 < p < 0.5


# ──────────────────────────────────────────────────────────
# deribit_collector — _parse_strike (P0-4: "$3k" virava 3.000.000, erro 1000×)
# ──────────────────────────────────────────────────────────

class TestParseStrike:
    """Tabela de repro da auditoria: duas regras de ×1000 sobrepostas disparavam
    as duas para números de 1 dígito seguidos de 'k' — "$3k" virava 3.000.000
    em vez de 3.000. Foi esse erro que gerou o único "arb garantido" (falso) já
    executado em produção ($50, 5% do capital): o strike de "$3k" virou
    3.000.000 e passou a "dominar" logicamente qualquer outro strike real."""

    @pytest.mark.parametrize("text,expected", [
        ("$3k",      3_000.0),
        ("$4k",      4_000.0),
        ("$100k",    100_000.0),
        ("$3.5k",    3_500.0),
        ("$120K",    120_000.0),
        ("$68,000",  68_000.0),
        ("$2m",      2_000_000.0),
    ])
    def test_tabela_de_repro(self, text, expected):
        assert deribit_collector._parse_strike(text) == expected

    def test_sem_preco_retorna_none(self):
        assert deribit_collector._parse_strike("Will it happen?") is None

    def test_multiplos_precos_rejeita_em_vez_de_tirar_media(self):
        """Antes: "between $60,000 and $70,000" virava a média (65.000) —
        um strike fabricado que não corresponde a mercado nenhum. Agora
        rejeita: sem um strike único e não-ambíguo, não há B-S para rodar."""
        assert deribit_collector._parse_strike(
            "Will BTC be between $60,000 and $70,000 on June 1?"
        ) is None


class TestParseCryptoMarketSpotDirection:
    """P0-4 correlato: structural_arb.py usava parsed['direction'] sem a
    correção de direção por spot que deribit_collector já aplicava no seu
    próprio loop — dois consumidores, dois comportamentos. Agora a correção
    mora dentro de parse_crypto_market(question, spot=...) e qualquer
    consumidor com spot disponível herda o mesmo resultado."""

    def test_sem_spot_usa_keyword(self):
        # "hit" sozinho sugere upward — sem spot, fica nisso.
        parsed = deribit_collector.parse_crypto_market(
            "Will Bitcoin hit $50,000 by December 31, 2026?"
        )
        assert parsed["direction"] == "above"

    def test_com_spot_acima_do_strike_corrige_para_dip(self):
        # Spot em $100k: "hit $50k" só é possível se o preço CAIR — downward.
        parsed = deribit_collector.parse_crypto_market(
            "Will Bitcoin hit $50,000 by December 31, 2026?", spot=100_000.0,
        )
        assert parsed["direction"] == "below"

    def test_com_spot_abaixo_do_strike_mantem_upward(self):
        parsed = deribit_collector.parse_crypto_market(
            "Will Bitcoin hit $150,000 by December 31, 2026?", spot=100_000.0,
        )
        assert parsed["direction"] == "above"

    def test_mercado_europeu_ignora_spot(self):
        # Não-touch: a direção vem da keyword da pergunta, não do spot —
        # "above $68,000" já é explícito, não é uma barreira a tocar.
        parsed = deribit_collector.parse_crypto_market(
            "Will the price of Bitcoin be above $68,000 on April 2?", spot=10_000.0,
        )
        assert parsed["direction"] == "above"


# ──────────────────────────────────────────────────────────
# deribit_collector — bs_prob (P0-1: barreira saturava em 1.0 para todo touch upward)
# ──────────────────────────────────────────────────────────

class TestBSProbBarrier:
    """
    bs_prob() é função pura de first-passage time. Antes do fix ela tinha dois
    erros de sinal (fórmula construída a partir de d1 em vez de d2, e o fator
    exponencial usava ln(S/K) em vez de ln(K/S)) que faziam TODO touch upward
    saturar em 1.000000, independente de quão longe o strike estivesse do spot.

    Valores de referência: Monte Carlo com 200k trajetórias e 1000 passos por
    ano (auditoria MELHORIAS_2026-09-03.md, P0-1). MC discreto tem viés
    negativo conhecido para probabilidade de barreira (pode pular o toque
    entre passos), por isso a tolerância é generosa (3pp).
    """

    @pytest.mark.parametrize("K, above, mc_prob", [
        (105, True, 0.8312),
        (110, True, 0.6988),
        (130, True, 0.3237),
        (90, False, 0.7506),
        (70, False, 0.2718),
    ])
    def test_contra_monte_carlo(self, K, above, mc_prob):
        p = deribit_collector.bs_prob(S=100, K=K, T=0.25, sigma=0.6, r=0.0, above=above, touch=True)
        assert abs(p - mc_prob) < 0.03

    def test_nao_satura_em_1_para_strikes_distantes(self):
        """Bug original: todo K acima do spot dava exatamente 1.0."""
        probs = [
            deribit_collector.bs_prob(S=100, K=K, T=0.25, sigma=0.6, r=0.0, above=True, touch=True)
            for K in [105, 110, 130, 200, 500]
        ]
        assert all(p < 0.999 for p in probs)
        # monotonicamente decrescente conforme o strike se afasta do spot
        assert probs == sorted(probs, reverse=True)

    def test_atm_da_probabilidade_1(self):
        """S == K: o preço já está na barreira, toque é certo."""
        p = deribit_collector.bs_prob(S=100, K=100, T=0.25, sigma=0.6, r=0.0, above=True, touch=True)
        assert p == pytest.approx(1.0, abs=1e-9)

    def test_europeia_nao_regride(self):
        """Ramo europeu (touch=False) já estava correto — não pode ter mudado."""
        p_above = deribit_collector.bs_prob(S=100, K=110, T=0.25, sigma=0.6, r=0.0, above=True, touch=False)
        p_below = deribit_collector.bs_prob(S=100, K=110, T=0.25, sigma=0.6, r=0.0, above=False, touch=False)
        assert p_above == pytest.approx(1.0 - p_below, abs=1e-9)
        assert 0.0 < p_above < 0.5  # K acima do spot, sem drift → P(S_T > K) < 50%


# ──────────────────────────────────────────────────────────
# risk_manager — double-credit e Kelly
# ──────────────────────────────────────────────────────────

@pytest.fixture()
def trading_db(tmp_path):
    """Banco de paper trading isolado com portfólio e uma posição aberta."""
    db = tmp_path / "paper.db"
    conn = sqlite3.connect(db)
    # Schema real do paper_trader (import tardio para evitar efeitos colaterais)
    sys.path.insert(0, str(ROOT / "execution"))
    import paper_trader
    conn.executescript(paper_trader.SCHEMA)
    # trade_type/needs_manual_resolution são colunas de migração (não estão no SCHEMA base)
    conn.execute("ALTER TABLE positions ADD COLUMN trade_type TEXT DEFAULT 'value'")
    conn.execute("ALTER TABLE positions ADD COLUMN needs_manual_resolution INTEGER DEFAULT 0")
    conn.execute(
        "INSERT INTO portfolio (initial_capital, current_cash) VALUES (1000, 1000)"
    )
    conn.execute("""
        INSERT INTO positions
          (condition_id, question, direction, entry_price, shares, cost_usdc,
           status, trade_type, opened_at)
        VALUES ('0xaaa', 'Teste?', 'BUY_YES', 0.10, 10.0, 1.0,
                'open', 'value', '2026-01-01 00:00:00')
    """)
    conn.commit()
    conn.close()
    return db


def _open_positions_df(db) -> pd.DataFrame:
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM positions WHERE status='open'").fetchall()
    conn.close()
    return pd.DataFrame([dict(r) for r in rows])


def _cash(db) -> float:
    conn = sqlite3.connect(db)
    v = conn.execute("SELECT current_cash FROM portfolio ORDER BY id DESC LIMIT 1").fetchone()[0]
    conn.close()
    return float(v)


class TestDoubleCredit:
    # Mercado que dispara profit_target (não depende de min_hold):
    # BUY_YES entry 0.10, yes atual 0.50 → valor 10 × ~0.49 ≈ 4.9 ≥ 2.5 × custo 1.0
    MKT = pd.DataFrame([{
        "conditionId": "0xaaa", "yes_price": 0.50, "spread": 0.01,
    }])

    def test_early_exit_credita_cash_uma_unica_vez(self, trading_db):
        open_pos = _open_positions_df(trading_db)
        cash0 = _cash(trading_db)

        exits1 = risk_manager.early_exit_positions(open_pos, self.MKT, trading_db)
        assert len(exits1) == 1
        cash1 = _cash(trading_db)
        assert cash1 > cash0  # creditou custo + P&L

        # Segundo processo com snapshot STALE (posição ainda parecia aberta):
        # guard status='open' → rowcount 0 → NÃO credita de novo
        exits2 = risk_manager.early_exit_positions(open_pos, self.MKT, trading_db)
        assert exits2 == []
        assert _cash(trading_db) == cash1

    def test_resolve_nao_credita_duas_vezes(self, trading_db):
        open_pos = _open_positions_df(trading_db)
        mkt = pd.DataFrame([{
            "conditionId": "0xaaa", "closed": True,
            "outcomePrices": '["1.0", "0.0"]',
        }])
        r1 = risk_manager.resolve_positions(open_pos, mkt, trading_db)
        assert len(r1) == 1
        cash1 = _cash(trading_db)

        r2 = risk_manager.resolve_positions(open_pos, mkt, trading_db)
        assert r2 == []
        assert _cash(trading_db) == cash1


class TestExitPriceGuards:
    """
    P1-14: early_exit_positions não tinha guarda de spread nem de liquidez —
    o ws_feed foi endurecido após o incidente de 2026-05 (51 posições a 0.001),
    mas esse caminho paralelo continuou aberto. 47 posições no DB com
    exit_price <= 0.0015 até este fix. Posição de teste: BUY_YES, entry=0.10,
    shares=10, custo=$1.0, trade_type='value' (min_hold 4h, aberta há meses).
    """

    def test_book_largo_nunca_dispara_exit_mesmo_com_preco_de_stop(self, trading_db):
        # bestBid=0.02 sozinho dispararia stop_loss (valor $0.20 <= 50% do custo),
        # mas spread=0.18 > MAX_EXIT_SPREAD=0.10 — book não é confiável.
        open_pos = _open_positions_df(trading_db)
        mkt = pd.DataFrame([{
            "conditionId": "0xaaa", "bestBid": 0.02, "bestAsk": 0.20, "liquidity": 10_000.0,
        }])
        assert risk_manager.early_exit_positions(open_pos, mkt, trading_db) == []

    def test_liquidez_insuficiente_nunca_dispara_exit(self, trading_db):
        # Book estreito e preço de stop, mas liquidez abaixo do piso executável.
        open_pos = _open_positions_df(trading_db)
        mkt = pd.DataFrame([{
            "conditionId": "0xaaa", "bestBid": 0.02, "bestAsk": 0.03, "liquidity": 100.0,
        }])
        assert risk_manager.early_exit_positions(open_pos, mkt, trading_db) == []

    def test_book_valido_sai_no_bid_sem_desconto_duplo(self, trading_db):
        # value=0.30×10=3.0 ≥ 2.5×custo(1.0) → profit_target. exit_price deve
        # ser exatamente bestBid — não (bid+ask)/2 nem bid-spread/2.
        open_pos = _open_positions_df(trading_db)
        mkt = pd.DataFrame([{
            "conditionId": "0xaaa", "bestBid": 0.30, "bestAsk": 0.32, "liquidity": 10_000.0,
        }])
        exits = risk_manager.early_exit_positions(open_pos, mkt, trading_db)
        assert len(exits) == 1
        assert exits[0]["exit_price"] == 0.30

    def test_edge_flip_funciona_com_bid_ask_real(self, trading_db):
        # Posição própria: BUY_YES momentum, entry_yes=0.50, flip_delta=0.12 →
        # threshold=0.38. mid=(0.30+0.34)/2=0.32 < 0.38 → edge_flip_yes.
        # current_value=$3.0, fora das faixas de stop_loss ($2.5) e profit ($7.5).
        conn = sqlite3.connect(trading_db)
        conn.execute("""
            INSERT INTO positions
              (condition_id, question, direction, entry_price, shares, cost_usdc,
               status, trade_type, opened_at)
            VALUES ('0xbbb', 'Momentum?', 'BUY_YES', 0.50, 10.0, 5.0,
                    'open', 'momentum', '2026-01-01 00:00:00')
        """)
        conn.commit(); conn.close()
        open_pos = _open_positions_df(trading_db)
        open_pos = open_pos[open_pos["condition_id"] == "0xbbb"]
        mkt = pd.DataFrame([{
            "conditionId": "0xbbb", "bestBid": 0.30, "bestAsk": 0.34, "liquidity": 10_000.0,
        }])
        exits = risk_manager.early_exit_positions(open_pos, mkt, trading_db)
        assert len(exits) == 1
        assert exits[0]["trigger"].startswith("edge_flip_yes")


class TestResolveThresholdNaoEhPreco:
    """
    P0-3: preço de mercado != resolução. RESOLVE_THRESHOLD=0.95 fechava posições
    vivas a -100% (posição ETH-$10k, vencimento dezembro, "resolvida NO" 31min
    após abertura porque YES caiu abaixo de 0.05). Preço extremo só pode contar
    como resolução quando o mercado TAMBÉM já venceu (endDate no passado).
    """

    def test_preco_extremo_sem_closed_e_sem_vencimento_nao_resolve(self, trading_db):
        # YES a 0.03 (p_no=0.97) teria disparado o RESOLVE_THRESHOLD antigo de 0.95.
        # Mercado não fechou e vence só em 2027 → precisa continuar aberto.
        open_pos = _open_positions_df(trading_db)
        mkt = pd.DataFrame([{
            "conditionId": "0xaaa", "closed": False,
            "outcomePrices": '["0.03", "0.97"]',
            "endDate": "2027-01-01T00:00:00Z",
        }])
        resolved = risk_manager.resolve_positions(open_pos, mkt, trading_db)
        assert resolved == []
        assert not _open_positions_df(trading_db).empty

    def test_preco_extremo_com_vencimento_passado_resolve(self, trading_db):
        # Mesmo preço extremo, mas endDate já passou → convergência é confiável.
        open_pos = _open_positions_df(trading_db)
        mkt = pd.DataFrame([{
            "conditionId": "0xaaa", "closed": False,
            "outcomePrices": '["0.001", "0.999"]',
            "endDate": "2020-01-01T00:00:00Z",
        }])
        resolved = risk_manager.resolve_positions(open_pos, mkt, trading_db)
        assert len(resolved) == 1
        assert resolved[0]["status"] == "closed"

    def test_closed_flag_ainda_resolve_sem_endDate(self, trading_db):
        # closed=True da API é sinal de resolução válido por si só, com ou sem endDate.
        open_pos = _open_positions_df(trading_db)
        mkt = pd.DataFrame([{
            "conditionId": "0xaaa", "closed": True,
            "outcomePrices": '["1.0", "0.0"]',
        }])
        resolved = risk_manager.resolve_positions(open_pos, mkt, trading_db)
        assert len(resolved) == 1
        assert resolved[0]["exit_price"] == 1.0

    def test_closed_com_preco_nao_extremo_resolve_pelo_lado_vencedor(self, trading_db):
        # closed=True mas outcomePrices ainda não pinou em 0/1 exato (comum sob
        # disputa/settlement da UMA) — o piso de 0.999 do path por vencimento NÃO
        # pode vazar para cá, senão isto cai no ramo "expired" (P1-16) e devolve
        # o custo integral em vez de resolver pelo lado vencedor.
        open_pos = _open_positions_df(trading_db)
        mkt = pd.DataFrame([{
            "conditionId": "0xaaa", "closed": True,
            "outcomePrices": '["0.98", "0.02"]',
        }])
        resolved = risk_manager.resolve_positions(open_pos, mkt, trading_db)
        assert resolved[0]["status"] == "closed"
        assert resolved[0]["exit_price"] == 1.0

    def test_preco_0_95_nao_basta_mais(self, trading_db):
        # O threshold antigo (0.95) não deve mais disparar resolução sozinho.
        open_pos = _open_positions_df(trading_db)
        mkt = pd.DataFrame([{
            "conditionId": "0xaaa", "closed": False,
            "outcomePrices": '["0.04", "0.96"]',
            "endDate": "2027-01-01T00:00:00Z",
        }])
        resolved = risk_manager.resolve_positions(open_pos, mkt, trading_db)
        assert resolved == []


class TestExpiredNaoDevolveCustoIntegral:
    """
    P1-16: mercado fechado (closed=True) sem outcomePrices parseável não
    pode virar status='expired' com P&L=0 (devolução integral) — é uma
    garantia que não existe, um dos lados sempre vale 0 na resolução real.
    Fica ABERTA e marcada needs_manual_resolution, sem tocar cash.
    """

    def test_closed_sem_outcome_prices_nao_fecha_nem_devolve_custo(self, trading_db):
        cash_antes = _cash(trading_db)
        open_pos = _open_positions_df(trading_db)
        mkt = pd.DataFrame([{
            "conditionId": "0xaaa", "closed": True,
            "outcomePrices": None,
        }])
        resolved = risk_manager.resolve_positions(open_pos, mkt, trading_db)

        assert resolved == []
        assert _cash(trading_db) == cash_antes  # nenhum crédito fabricado
        still_open = _open_positions_df(trading_db)
        assert len(still_open) == 1 and still_open.iloc[0]["status"] == "open"
        assert int(still_open.iloc[0]["needs_manual_resolution"]) == 1

    def test_closed_com_outcome_prices_json_invalido_nao_fecha(self, trading_db):
        open_pos = _open_positions_df(trading_db)
        mkt = pd.DataFrame([{
            "conditionId": "0xaaa", "closed": True,
            "outcomePrices": "not-json-at-all",
        }])
        resolved = risk_manager.resolve_positions(open_pos, mkt, trading_db)
        assert resolved == []
        assert _open_positions_df(trading_db).iloc[0]["status"] == "open"

    def test_dry_run_nao_grava_a_flag(self, trading_db):
        open_pos = _open_positions_df(trading_db)
        mkt = pd.DataFrame([{"conditionId": "0xaaa", "closed": True, "outcomePrices": None}])
        risk_manager.resolve_positions(open_pos, mkt, trading_db, dry_run=True)
        # dry_run não deve alterar o banco de jeito nenhum
        assert int(_open_positions_df(trading_db).iloc[0]["needs_manual_resolution"]) == 0

    def test_find_positions_needing_manual_resolution_acha_a_posicao_flagada(self, trading_db):
        open_pos = _open_positions_df(trading_db)
        mkt = pd.DataFrame([{"conditionId": "0xaaa", "closed": True, "outcomePrices": None}])
        risk_manager.resolve_positions(open_pos, mkt, trading_db)

        stuck = risk_manager.find_positions_needing_manual_resolution(trading_db)
        assert len(stuck) == 1
        assert stuck.iloc[0]["condition_id"] == "0xaaa"

    def test_posicao_normal_nao_aparece_na_busca(self, trading_db):
        # Sanity: sem nenhuma resolução ambígua, a busca não acha nada.
        stuck = risk_manager.find_positions_needing_manual_resolution(trading_db)
        assert stuck.empty

    def test_flag_some_quando_outcome_prices_chega_num_ciclo_seguinte(self, trading_db):
        # Ciclo 1: fecha sem outcomePrices parseável, fica travada.
        open_pos = _open_positions_df(trading_db)
        mkt_sem_outcome = pd.DataFrame([{"conditionId": "0xaaa", "closed": True, "outcomePrices": None}])
        risk_manager.resolve_positions(open_pos, mkt_sem_outcome, trading_db)
        assert int(_open_positions_df(trading_db).iloc[0]["needs_manual_resolution"]) == 1

        # Ciclo 2: a API finalmente devolve outcomePrices — resolve normal,
        # e a flag não pode sobreviver numa posição que já fechou de verdade.
        open_pos = _open_positions_df(trading_db)
        mkt_com_outcome = pd.DataFrame([{
            "conditionId": "0xaaa", "closed": True, "outcomePrices": "[1, 0]",
        }])
        resolved = risk_manager.resolve_positions(open_pos, mkt_com_outcome, trading_db)

        assert len(resolved) == 1
        assert risk_manager.find_positions_needing_manual_resolution(trading_db).empty


class TestKelly:
    def test_edge_abaixo_do_minimo_da_zero(self):
        assert risk_manager.kelly_size(
            edge=0.01, entry_price=0.5, capital=1000, confidence=0.9,
            signal_source="odds",
        ) == 0.0

    def test_preco_invalido_da_zero(self):
        assert risk_manager.kelly_size(0.10, 0.0, 1000, 0.9, "odds") == 0.0
        assert risk_manager.kelly_size(0.10, 1.0, 1000, 0.9, "odds") == 0.0

    def test_sizing_respeita_cap_por_trade_type(self):
        size = risk_manager.kelly_size(
            edge=0.30, entry_price=0.5, capital=1000, confidence=1.0,
            signal_source="odds", trade_type="value",
        )
        assert 0 < size <= 1000 * risk_manager.MAX_POSITION_PCT["value"] + 0.01

    def test_formula_bate_com_a_tabela_da_auditoria(self):
        # p=0.20, edge=0.10 -> f*=edge/(1-p)=0.125 (a fórmula antiga dava
        # edge*p/(1-p)=0.025, 5x menor). kelly_full clampa em KELLY_FULL_CAP
        # (0.10) antes do quarter-Kelly: 0.10*0.25*1.0=0.025 -> $25 em $1000.
        size = risk_manager.kelly_size(
            edge=0.10, entry_price=0.20, capital=1000, confidence=1.0,
            signal_source="odds", trade_type="value",
        )
        assert size == 25.0

    def test_formula_sem_clamp_bate_com_edge_sobre_1_menos_p(self):
        # p=0.10, edge=0.08: kelly_full=0.08/0.90=0.0889 (< KELLY_FULL_CAP,
        # não clampa) -> kelly_used=0.0889*0.25=0.02222 (< pos_pct 'value'
        # 0.03, não clampa) -> $22.22 em $1000. Isola a fórmula pura.
        size = risk_manager.kelly_size(
            edge=0.08, entry_price=0.10, capital=1000, confidence=1.0,
            signal_source="odds", trade_type="value",
        )
        assert abs(size - 22.22) < 0.01

    def test_clamp_do_kelly_full_binda_em_preco_alto_nao_baixo(self):
        # P1-10: o clamp existe para p ALTO (token quase certo), onde
        # (1-p)->0 faz f* explodir — não para p baixo (deep-OTM), onde
        # f*=edge/(1-p) fica pequeno e limitado por construção. As duas
        # posições NÃO podem dar o mesmo tamanho: se o clamp estivesse do
        # lado errado (ou a fórmula tivesse regredido para edge*p/(1-p)),
        # esses dois valores não bateriam.
        edge = 0.08

        # p baixo (0.05): kelly_full=0.08/0.95=0.0842 (< KELLY_FULL_CAP, não
        # clampa) -> kelly_used=0.0842*0.25=0.02105 (< pos_pct 0.03) -> $21.05
        size_low_p = risk_manager.kelly_size(
            edge=edge, entry_price=0.05, capital=1000, confidence=1.0,
            signal_source="odds", trade_type="value",
        )
        assert size_low_p == 21.05

        # p alto (0.95): kelly_full=0.08/0.05=1.6 >> KELLY_FULL_CAP -> clampa
        # em 0.10 -> kelly_used=0.10*0.25=0.025 (< pos_pct 0.03) -> $25.00,
        # travado no KELLY_FULL_CAP e não na fórmula bruta.
        size_high_p = risk_manager.kelly_size(
            edge=edge, entry_price=0.95, capital=1000, confidence=1.0,
            signal_source="odds", trade_type="value",
        )
        assert size_high_p == 25.0


# ──────────────────────────────────────────────────────────
# sim_backtest — paridade de Kelly com produção (P2-33)
# ──────────────────────────────────────────────────────────

sys.path.insert(0, str(ROOT / "backtest"))
import sim_backtest


class TestSimBacktestKellyParity:
    """
    P2-33: sim_backtest tinha sua própria _kelly_size — com a fórmula CORRETA
    (ao contrário da produção pré-P1-9), mas isolada. O cabeçalho do arquivo
    afirma paridade com produção; só passou a ser verdade quando as duas
    funções viraram a mesma função. Testa que o simulador não pode mais
    divergir silenciosamente da produção.
    """

    def _sim(self, capital=1000.0):
        return sim_backtest.WalkForwardSimulator(
            sim_start=datetime(2026, 1, 1, tzinfo=timezone.utc),
            sim_end=datetime(2026, 2, 1, tzinfo=timezone.utc),
            initial_capital=capital,
        )

    @pytest.mark.parametrize("entry_price,prob", [
        (0.20, 0.30),   # exemplo da auditoria: edge=0.10
        (0.10, 0.18),   # edge=0.08, sem clamp de KELLY_FULL_CAP
        (0.50, 0.65),   # edge=0.15, meio da distribuição
    ])
    def test_bate_exatamente_com_risk_manager_kelly_size(self, entry_price, prob):
        sim = self._sim()
        edge = prob - entry_price
        expected = risk_manager.kelly_size(
            edge=edge, entry_price=entry_price, capital=sim.cash,
            confidence=1.0, signal_source="deribit", trade_type="value",
        )
        assert sim._kelly_size(entry_price, prob) == expected

    def test_edge_abaixo_do_minimo_deribit_da_zero(self):
        # MIN_EDGE_TO_TRADE["deribit"]=0.05 — o simulador agora herda esse
        # piso, que antes não existia aqui (P2-33 nota isso como lacuna).
        sim = self._sim()
        assert sim._kelly_size(entry_price=0.50, prob=0.53) == 0.0  # edge=0.03


# ──────────────────────────────────────────────────────────
# structural_arb — detectores puros (Fase 2)
# ──────────────────────────────────────────────────────────

sys.path.insert(0, str(ROOT / "signals"))
import structural_arb
import signal_generator


class TestNegRiskBasket:
    def _mkts(self, bidasks):
        return [
            {"conditionId": f"0x{i}", "question": f"Outcome {i}?",
             "bestBid": b, "bestAsk": a, "liquidity": 10_000}
            for i, (b, a) in enumerate(bidasks)
        ]

    def test_soma_asks_abaixo_de_1_sem_guarda_chuva_e_condicional(self):
        # Σ ask = 0.95 mas SEM outcome "someone else": exaustividade não
        # garantida (ex: Nobel — ninguém listado ganha → todas as YES viram 0)
        opp = structural_arb.evaluate_negrisk_event(
            self._mkts([(0.28, 0.30), (0.30, 0.32), (0.31, 0.33)]), margin=0.02,
        )
        assert opp is not None and opp["kind"] == "negrisk_yes_conditional"
        assert opp["guaranteed"] is False
        assert abs(opp["profit"] - 0.05) < 1e-9
        assert all(l["direction"] == "BUY_YES" for l in opp["legs"])

    def test_yes_basket_com_guarda_chuva_e_garantido(self):
        mkts = self._mkts([(0.28, 0.30), (0.30, 0.32)])
        mkts.append({"conditionId": "0xo", "question": "Will someone else win?",
                     "bestBid": 0.31, "bestAsk": 0.33, "liquidity": 10_000})
        opp = structural_arb.evaluate_negrisk_event(mkts, margin=0.02)
        assert opp is not None and opp["kind"] == "negrisk_yes"
        assert opp["guaranteed"] is True

    def test_soma_bids_acima_de_1_gera_no_basket(self):
        # Σ bid = 1.11 → comprar todas as NO: lucro Σbid − 1 = 0.11 (payout n−1=2)
        opp = structural_arb.evaluate_negrisk_event(
            self._mkts([(0.40, 0.42), (0.36, 0.38), (0.35, 0.37)]), margin=0.02,
        )
        assert opp is not None and opp["kind"] == "negrisk_no"
        assert abs(opp["profit"] - 0.11) < 1e-9

    def test_evento_bem_precificado_nao_gera_nada(self):
        # Σ ask = 1.03, Σ bid = 0.97 — sem arb (caso real típico)
        opp = structural_arb.evaluate_negrisk_event(
            self._mkts([(0.32, 0.34), (0.32, 0.34), (0.33, 0.35)]), margin=0.02,
        )
        assert opp is None

    def test_book_incompleto_invalida_a_soma(self):
        mkts = self._mkts([(0.28, 0.30), (0.30, 0.32)])
        mkts.append({"conditionId": "0xz", "question": "Sem book?",
                     "bestBid": None, "bestAsk": None, "liquidity": 0})
        assert structural_arb.evaluate_negrisk_event(mkts, margin=0.0) is None


class TestMonotonicity:
    # endDate autoritativo da API — o parser de texto não decide expiração
    DEC = "2098-12-31T16:00:00Z"
    NOV = "2098-11-30T16:00:00Z"

    @pytest.fixture(autouse=True)
    def _fixed_spot(self, monkeypatch):
        # P0-4 correlato: scan_monotonicity agora busca spot para corrigir a
        # direção de mercados touch. Sem mock, isso bateria na Deribit de
        # verdade a cada teste — quebra a garantia "sem rede" da suíte e torna
        # o resultado dependente do preço real no momento do run. Spot fixo
        # abaixo de todos os strikes usados aqui (BTC min=72k, ETH min=9k)
        # preserva a semântica "above" que estes testes já assumiam.
        spots = {"BTC": 50_000.0, "ETH": 3_000.0}
        monkeypatch.setattr(structural_arb, "get_spot_price", lambda asset: spots.get(asset))

    def _universe(self, rows):
        base = {"liquidity": 10_000.0}
        return pd.DataFrame([{**base, **r} for r in rows])

    def test_violacao_touch_detectada(self):
        # Tocar 80k até dez (dominante) precisa valer ≥ tocar 90k até nov (dominado).
        # Aqui o dominado tem bid 0.60 > ask 0.50 do dominante → arb de 0.10.
        df = self._universe([
            {"conditionId": "0xa", "question": "Will Bitcoin reach $80,000 by December 31?",
             "endDate": self.DEC, "bestBid": 0.48, "bestAsk": 0.50},
            {"conditionId": "0xb", "question": "Will Bitcoin reach $90,000 by November 30?",
             "endDate": self.NOV, "bestBid": 0.60, "bestAsk": 0.62},
        ])
        opps = structural_arb.scan_monotonicity(df, margin=0.02)
        assert len(opps) == 1
        opp = opps[0]
        assert opp["guaranteed"] is True
        assert abs(opp["profit"] - 0.10) < 1e-9
        dirs = {l["direction"] for l in opp["legs"]}
        assert dirs == {"BUY_YES", "BUY_NO"}

    def test_precos_coerentes_nao_geram_arb(self):
        # Dominante mais caro que dominado — ordem correta, sem arb
        df = self._universe([
            {"conditionId": "0xa", "question": "Will Bitcoin reach $80,000 by December 31?",
             "endDate": self.DEC, "bestBid": 0.70, "bestAsk": 0.72},
            {"conditionId": "0xb", "question": "Will Bitcoin reach $90,000 by November 30?",
             "endDate": self.NOV, "bestBid": 0.50, "bestAsk": 0.52},
        ])
        assert structural_arb.scan_monotonicity(df, margin=0.02) == []

    def test_ativos_diferentes_nao_comparam(self):
        df = self._universe([
            {"conditionId": "0xa", "question": "Will Bitcoin reach $80,000 by December 31?",
             "endDate": self.DEC, "bestBid": 0.48, "bestAsk": 0.50},
            {"conditionId": "0xb", "question": "Will Ethereum reach $9,000 by November 30?",
             "endDate": self.NOV, "bestBid": 0.60, "bestAsk": 0.62},
        ])
        assert structural_arb.scan_monotonicity(df, margin=0.02) == []

    def test_mercado_de_janela_excluido(self):
        """'reach $72k July 6-12' vale só na janela — não domina nada.
        Falso arb real do primeiro run do scanner (2026-07-11)."""
        df = self._universe([
            {"conditionId": "0xa", "question": "Will Bitcoin reach $72,000 July 6-12?",
             "endDate": self.DEC, "bestBid": 0.48, "bestAsk": 0.50},
            {"conditionId": "0xb", "question": "Will Bitcoin reach $150,000 by November 30?",
             "endDate": self.NOV, "bestBid": 0.60, "bestAsk": 0.62},
        ])
        assert structural_arb.scan_monotonicity(df, margin=0.02) == []

    def test_sem_enddate_nao_participa(self):
        df = self._universe([
            {"conditionId": "0xa", "question": "Will Bitcoin reach $80,000 by December 31?",
             "endDate": None, "bestBid": 0.48, "bestAsk": 0.50},
            {"conditionId": "0xb", "question": "Will Bitcoin reach $90,000 by November 30?",
             "endDate": self.NOV, "bestBid": 0.60, "bestAsk": 0.62},
        ])
        assert structural_arb.scan_monotonicity(df, margin=0.02) == []


class TestCrossedBook:
    def test_book_cruzado_reportado(self):
        df = pd.DataFrame([{
            "conditionId": "0xa", "question": "Glitch?", "event_slug": "",
            "bestBid": 0.55, "bestAsk": 0.50, "liquidity": 1000.0,
        }])
        opps = structural_arb.scan_crossed_books(df)
        assert len(opps) == 1 and abs(opps[0]["profit"] - 0.05) < 1e-9


# ──────────────────────────────────────────────────────────
# open_basket — execução atômica de baskets (Fase 2)
# ──────────────────────────────────────────────────────────

sys.path.insert(0, str(ROOT / "execution"))
import paper_trader


@pytest.fixture()
def basket_db(tmp_path, monkeypatch):
    """DB isolado + DB_PATH monkeypatched para as funções do paper_trader."""
    db = tmp_path / "paper.db"
    monkeypatch.setattr(paper_trader, "DB_PATH", db)
    paper_trader.init_db()
    conn = sqlite3.connect(db)
    conn.execute(
        "INSERT INTO portfolio (initial_capital, current_cash, created_at) "
        "VALUES (1000, 1000, '2026-01-01 00:00:00')"
    )
    conn.commit()
    conn.close()
    return db


def _basket_group(arb_group="mono:0xa>0xb", legs=None) -> pd.DataFrame:
    """Grupo de pernas no formato do CSV do scanner (monotonicidade 2 pernas)."""
    if legs is None:
        legs = [
            {"condition_id": "0xa", "direction": "BUY_YES", "leg_price": 0.50},
            {"condition_id": "0xb", "direction": "BUY_NO",  "leg_price": 0.38},
        ]
    rows = []
    for i, leg in enumerate(legs):
        rows.append({
            "arb_group": arb_group, "arb_kind": "monotonicity",
            "leg": i + 1, "n_legs": len(legs),
            "condition_id": leg["condition_id"],
            "question": f"Perna {i}?", "direction": leg["direction"],
            "leg_price": leg["leg_price"],
            "basket_cost": sum(l["leg_price"] for l in legs),
            "payout_min": 1.0,
            "profit": 1.0 - sum(l["leg_price"] for l in legs),
            "edge": 1.0 - sum(l["leg_price"] for l in legs),
            "abs_edge": abs(1.0 - sum(l["leg_price"] for l in legs)),
            "guaranteed": True, "liquidity": 10_000.0, "event_slug": "ev-x",
            "signal_source": "structural", "trade_type": "arb", "confidence": 0.9,
            "underlying": leg.get("underlying", ""),
        })
    return pd.DataFrame(rows)


class TestOpenBasket:
    EMPTY_MKTS = pd.DataFrame()

    def test_abre_todas_as_pernas_com_debito_unico(self, basket_db):
        portfolio = paper_trader.get_or_create_portfolio(1000)
        result = paper_trader.open_basket(
            portfolio, _basket_group(), self.EMPTY_MKTS
        )
        assert result is not None
        pos = _open_positions_df(basket_db)
        assert len(pos) == 2
        assert set(pos["trade_type"]) == {"arb"}
        assert set(pos["signal_source"]) == {"structural"}
        assert set(pos["arb_group"]) == {"mono:0xa>0xb"}
        # P1-11a: category não recebe mais arb_kind ("monotonicity") — os
        # dois eram conceitos sobrepostos na mesma coluna.
        assert set(pos["category"]) == {""}
        # Mesmo nº de shares nas duas pernas (a matemática do arb exige)
        assert pos["shares"].nunique() == 1
        # Débito único = soma dos custos das pernas
        assert abs((1000 - _cash(basket_db)) - result["total_cost"]) < 0.02

    def test_underlying_da_perna_e_persistido(self, basket_db):
        # P1-11: monotonicidade tem asset real (BTC/ETH) por perna — precisa
        # sobreviver até a tabela positions para o cap de underlying funcionar.
        legs = [
            {"condition_id": "0xa", "direction": "BUY_YES", "leg_price": 0.50, "underlying": "BTC"},
            {"condition_id": "0xb", "direction": "BUY_NO",  "leg_price": 0.38, "underlying": "BTC"},
        ]
        portfolio = paper_trader.get_or_create_portfolio(1000)
        result = paper_trader.open_basket(portfolio, _basket_group(legs=legs), self.EMPTY_MKTS)
        assert result is not None
        pos = _open_positions_df(basket_db)
        assert set(pos["underlying"]) == {"BTC"}

    def test_caixa_insuficiente_nao_abre_nenhuma_perna(self, basket_db):
        """Atomicidade: guard de cash falhou → rollback, zero posições órfãs."""
        conn = sqlite3.connect(basket_db)
        conn.execute("UPDATE portfolio SET current_cash = 1.0")
        conn.commit(); conn.close()
        # Portfolio dict STALE (acha que tem $1000) — simula corrida entre processos
        portfolio = {"id": 1, "initial_capital": 1000, "current_cash": 1000}
        result = paper_trader.open_basket(portfolio, _basket_group(), self.EMPTY_MKTS)
        assert result is None
        assert _open_positions_df(basket_db).empty
        assert _cash(basket_db) == 1.0  # nada debitado

    def test_perna_ja_operada_cancela_o_basket_inteiro(self, basket_db):
        conn = sqlite3.connect(basket_db)
        conn.execute("""
            INSERT INTO positions
              (condition_id, question, direction, entry_price, shares, cost_usdc,
               status, opened_at)
            VALUES ('0xb', 'Já operado?', 'BUY_YES', 0.5, 10, 5, 'closed',
                    '2026-06-01 00:00:00')
        """)
        conn.commit(); conn.close()
        portfolio = paper_trader.get_or_create_portfolio(1000)
        result = paper_trader.open_basket(portfolio, _basket_group(), self.EMPTY_MKTS)
        assert result is None
        assert _open_positions_df(basket_db).empty

    def test_arb_evaporado_na_recotacao_e_pulado(self, basket_db):
        """Preços frescos do parquet pioraram → lucro < ARB_MIN_PROFIT → skip."""
        mkts = pd.DataFrame([
            {"conditionId": "0xa", "bestBid": 0.58, "bestAsk": 0.60},
            {"conditionId": "0xb", "bestBid": 0.42, "bestAsk": 0.44},
        ])  # custo re-cotado: 0.60 + (1-0.42) = 1.18 > payout 1 → sem arb
        portfolio = paper_trader.get_or_create_portfolio(1000)
        result = paper_trader.open_basket(portfolio, _basket_group(), mkts)
        assert result is None
        assert _open_positions_df(basket_db).empty

    def test_csv_truncado_nao_executa(self, basket_db):
        group = _basket_group()
        group.loc[:, "n_legs"] = 3  # CSV diz 3 pernas mas só há 2 → incompleto
        portfolio = paper_trader.get_or_create_portfolio(1000)
        assert paper_trader.open_basket(portfolio, group, self.EMPTY_MKTS) is None

    def test_dry_run_nao_persiste(self, basket_db):
        portfolio = paper_trader.get_or_create_portfolio(1000)
        result = paper_trader.open_basket(
            portfolio, _basket_group(), self.EMPTY_MKTS, dry_run=True
        )
        assert result is not None
        assert _open_positions_df(basket_db).empty
        assert _cash(basket_db) == 1000.0

    def test_sizing_respeita_cap_de_basket_e_liquidez(self, basket_db):
        portfolio = paper_trader.get_or_create_portfolio(1000)
        result = paper_trader.open_basket(portfolio, _basket_group(), self.EMPTY_MKTS)
        assert result["total_cost"] <= 1000 * risk_manager.ARB_MAX_BASKET_PCT + 0.02
        # Liquidez 10k × 5% = $500/perna — não é o binding constraint aqui
        assert result["guaranteed_profit"] > 0


class TestArbHoldUntilResolution:
    """Pernas de arb nunca saem antes da resolução — em NENHUM dos 2 caminhos."""

    MKT_ADVERSO = pd.DataFrame([{
        "conditionId": "0xa", "yes_price": 0.05, "spread": 0.01,
    }])  # BUY_YES entry 0.50 → valor a 9% do custo: stop_loss dispararia se fosse value

    def _arb_pos(self) -> pd.DataFrame:
        return pd.DataFrame([{
            "id": 1, "condition_id": "0xa", "question": "Perna arb?",
            "direction": "BUY_YES", "entry_price": 0.50, "shares": 10.0,
            "cost_usdc": 5.0, "status": "open", "trade_type": "arb",
            "opened_at": "2026-01-01 00:00:00",  # hold de meses — min_hold não protege
        }])

    def test_early_exit_ignora_arb(self, tmp_path):
        db = tmp_path / "x.db"
        exits = risk_manager.early_exit_positions(self._arb_pos(), self.MKT_ADVERSO, db)
        assert exits == []

    def test_profit_target_tambem_nao_dispara(self, tmp_path):
        mkt = pd.DataFrame([{"conditionId": "0xa", "yes_price": 0.99, "spread": 0.01}])
        db = tmp_path / "x.db"
        exits = risk_manager.early_exit_positions(self._arb_pos(), mkt, db)
        assert exits == []

    def test_ws_feed_ignora_arb(self):
        info = ws_feed.AssetInfo(
            "tok", "0xa", "Perna arb?", "BUY_YES", 0.50, 10.0, 5.0, 1,
            "arb", time.time() - 100 * 3600,
        )
        # Lucro de 1.9× e prejuízo de 90% — nada dispara para arb
        assert ws_feed.evaluate_exit(info, 0.95, 0.96) is None
        assert ws_feed.evaluate_exit(info, 0.05, 0.06) is None


class TestOrphanArbLegs:
    """
    P0-5: abertura de basket é atômica (open_basket), resolução não era —
    quando uma perna resolve e a outra fica aberta, a perna remanescente virava
    posição direcional NUA (EARLY_EXIT["arb"] é hold-forever por design).
    Caso real em produção: basket mono:0x201f51d2>0x3c16fd3f, perna de $2.33
    fechada pelo bug do RESOLVE_THRESHOLD (P0-3), perna de $47.67 (22% do
    livro) ficou aberta e sem stop-loss.
    """

    def _insert_leg(self, db, condition_id, arb_group, status):
        conn = sqlite3.connect(db)
        conn.execute("""
            INSERT INTO positions
              (condition_id, question, direction, entry_price, shares, cost_usdc,
               status, trade_type, arb_group, opened_at)
            VALUES (?, 'Perna arb?', 'BUY_YES', 0.5, 10, 5, ?, 'arb', ?, '2026-01-01 00:00:00')
        """, (condition_id, status, arb_group))
        conn.commit()
        conn.close()

    def test_perna_com_irma_resolvida_e_orfa(self, basket_db):
        self._insert_leg(basket_db, "0xa", "mono:0xa>0xb", "closed")
        self._insert_leg(basket_db, "0xb", "mono:0xa>0xb", "open")

        orphans = risk_manager.find_orphan_arb_legs(basket_db)
        assert list(orphans["condition_id"]) == ["0xb"]

    def test_basket_intacto_nao_gera_orfa(self, basket_db):
        self._insert_leg(basket_db, "0xa", "mono:0xa>0xb", "open")
        self._insert_leg(basket_db, "0xb", "mono:0xa>0xb", "open")

        assert risk_manager.find_orphan_arb_legs(basket_db).empty

    def test_basket_totalmente_resolvido_nao_gera_orfa(self, basket_db):
        self._insert_leg(basket_db, "0xa", "mono:0xa>0xb", "closed")
        self._insert_leg(basket_db, "0xb", "mono:0xa>0xb", "closed")

        assert risk_manager.find_orphan_arb_legs(basket_db).empty

    def test_reclassifica_perna_orfa_para_value(self, basket_db):
        self._insert_leg(basket_db, "0xa", "mono:0xa>0xb", "closed")
        self._insert_leg(basket_db, "0xb", "mono:0xa>0xb", "open")

        result = risk_manager.reclassify_orphan_arb_legs(basket_db)
        assert len(result) == 1
        assert result[0]["condition_id"] == "0xb"

        pos = _open_positions_df(basket_db)
        row = pos[pos["condition_id"] == "0xb"].iloc[0]
        assert row["trade_type"] == "value"

        # Idempotente: a perna já não é mais 'arb', some da próxima varredura
        assert risk_manager.reclassify_orphan_arb_legs(basket_db) == []

    def test_reclassificacao_reativa_stop_loss(self, basket_db):
        """Depois de reclassificada, a perna passa a poder sair por stop_loss —
        exatamente o que EARLY_EXIT['arb'] proibia (hold-forever)."""
        self._insert_leg(basket_db, "0xa", "mono:0xa>0xb", "closed")
        self._insert_leg(basket_db, "0xb", "mono:0xa>0xb", "open")
        risk_manager.reclassify_orphan_arb_legs(basket_db)

        open_pos = _open_positions_df(basket_db)
        mkt = pd.DataFrame([{"conditionId": "0xb", "yes_price": 0.05, "spread": 0.01}])
        exits = risk_manager.early_exit_positions(open_pos, mkt, basket_db)
        assert len(exits) == 1
        assert exits[0]["trigger"].startswith("stop_loss")

    def test_dry_run_nao_persiste(self, basket_db):
        self._insert_leg(basket_db, "0xa", "mono:0xa>0xb", "closed")
        self._insert_leg(basket_db, "0xb", "mono:0xa>0xb", "open")

        result = risk_manager.reclassify_orphan_arb_legs(basket_db, dry_run=True)
        assert len(result) == 1
        pos = _open_positions_df(basket_db)
        assert pos[pos["condition_id"] == "0xb"].iloc[0]["trade_type"] == "arb"


class TestStructuralExposure:
    def _open_arb_legs(self) -> pd.DataFrame:
        return pd.DataFrame([
            {"condition_id": f"0x{i}", "direction": "BUY_YES", "cost_usdc": 5.0,
             "category": "monotonicity", "signal_source": "structural",
             "trade_type": "arb", "event_slug": "ev-x"}
            for i in range(25)  # mais que MAX_OPEN_POSITIONS
        ])

    def test_structural_isento_do_bloqueio_de_event_slug(self):
        open_pos = pd.DataFrame([{
            "condition_id": "0x1", "direction": "BUY_YES", "cost_usdc": 5.0,
            "category": "", "signal_source": "structural", "trade_type": "arb",
            "event_slug": "ev-x",
        }])
        ok, _ = risk_manager.check_exposure(
            {"condition_id": "0x2", "signal_source": "structural",
             "trade_type": "arb", "event_slug": "ev-x"},
            open_pos, {"initial_capital": 1000}, 5.0,
        )
        assert ok

    def test_sinal_normal_continua_bloqueado_por_event_slug(self):
        open_pos = pd.DataFrame([{
            "condition_id": "0x1", "direction": "BUY_YES", "cost_usdc": 5.0,
            "category": "", "signal_source": "odds", "trade_type": "value",
            "event_slug": "ev-x",
        }])
        ok, reason = risk_manager.check_exposure(
            {"condition_id": "0x2", "signal_source": "odds",
             "trade_type": "value", "event_slug": "ev-x"},
            open_pos, {"initial_capital": 1000}, 5.0,
        )
        assert not ok and "correlação" in reason

    def test_pernas_arb_nao_contam_no_limite_global(self):
        """25 pernas de arb abertas não podem bloquear um sinal direcional."""
        ok, reason = risk_manager.check_exposure(
            {"condition_id": "0xnew", "signal_source": "odds",
             "trade_type": "value", "event_slug": "outro-ev", "category": "sports"},
            self._open_arb_legs(), {"initial_capital": 1000}, 5.0,
        )
        assert ok, reason


class TestUnderlyingExposure:
    """
    P1-11: `category` não captura correlação real. Livro real da auditoria:
    BTC acima de $62k/$64k/$74k/$78k + "reach $70k/$75k/$100k/$110k" — 9
    posições na MESMA variável (spot do BTC) espalhadas por categorias
    diferentes, cada uma abaixo do cap de 30%. underlying agrupa pelo ativo
    real; MAX_UNDERLYING_PCT (15%) é o check que pegaria isso.
    """

    def _btc_positions(self, n, cost_each=50.0):
        return pd.DataFrame([
            {"condition_id": f"0x{i}", "direction": "BUY_YES", "cost_usdc": cost_each,
             "category": f"cat-{i % 3}", "underlying": "BTC", "signal_source": "deribit",
             "trade_type": "value", "event_slug": f"ev-{i}"}
            for i in range(n)
        ])

    def test_underlying_bloqueia_quando_categoria_nao_bloquearia(self):
        # 5 posições BTC de $50 = $250, espalhadas por 3 categorias — nenhuma
        # categoria isolada passa de 30% de $1000, mas juntas são 25% do
        # underlying BTC. Nova posição de $50 empurra para $300 = 30% > 15%.
        open_pos = self._btc_positions(5)
        ok, reason = risk_manager.check_exposure(
            {"condition_id": "0xnew", "signal_source": "deribit", "trade_type": "value",
             "category": "cat-9", "underlying": "BTC", "event_slug": "ev-novo"},
            open_pos, {"initial_capital": 1000}, 50.0,
        )
        assert not ok and "underlying" in reason

    def test_underlyings_diferentes_nao_se_bloqueiam(self):
        open_pos = self._btc_positions(5)
        ok, reason = risk_manager.check_exposure(
            {"condition_id": "0xnew", "signal_source": "deribit", "trade_type": "value",
             "category": "cat-9", "underlying": "ETH", "event_slug": "ev-novo"},
            open_pos, {"initial_capital": 1000}, 50.0,
        )
        assert ok, reason

    def test_denominador_usa_total_value_nao_initial_capital(self):
        # Drawdown para $500: 2 posições BTC de $50 = $100 = 20% de $500 (já
        # > 15%), mas só 10% de initial_capital=$1000 — sem total_value o
        # bug antigo deixaria passar.
        open_pos = self._btc_positions(2)
        ok, reason = risk_manager.check_exposure(
            {"condition_id": "0xnew", "signal_source": "deribit", "trade_type": "value",
             "category": "cat-9", "underlying": "BTC", "event_slug": "ev-novo"},
            open_pos, {"initial_capital": 1000}, 5.0, total_value=500.0,
        )
        assert not ok and "underlying" in reason


class TestDirectionalSkew:
    """P1-11e: 20 das 21 posições do livro real eram BUY_YES."""

    def _positions(self, n_yes, n_no, trade_type="value"):
        rows = [
            {"condition_id": f"y{i}", "direction": "BUY_YES", "cost_usdc": 10.0,
             "trade_type": trade_type, "event_slug": f"ev-y{i}"}
            for i in range(n_yes)
        ] + [
            {"condition_id": f"n{i}", "direction": "BUY_NO", "cost_usdc": 10.0,
             "trade_type": trade_type, "event_slug": f"ev-n{i}"}
            for i in range(n_no)
        ]
        return pd.DataFrame(rows)

    def test_livro_pequeno_nao_bloqueia_por_skew(self):
        # 3 posições BUY_YES — abaixo de MIN_POSITIONS_FOR_SKEW_CHECK (5),
        # não pode bloquear a 4ª só por ainda não ter diversidade.
        open_pos = self._positions(n_yes=3, n_no=0)
        ok, reason = risk_manager.check_exposure(
            {"condition_id": "0xnew", "direction": "BUY_YES", "trade_type": "value",
             "event_slug": "ev-novo"},
            open_pos, {"initial_capital": 1000}, 10.0,
        )
        assert ok, reason

    def test_livro_grande_e_desbalanceado_bloqueia_mesmo_lado(self):
        # 6 BUY_YES + 1 BUY_NO (>= 5 direcionais) — mais um BUY_YES estoura
        # MAX_DIRECTIONAL_SKEW_PCT (75%).
        open_pos = self._positions(n_yes=6, n_no=1)
        ok, reason = risk_manager.check_exposure(
            {"condition_id": "0xnew", "direction": "BUY_YES", "trade_type": "value",
             "event_slug": "ev-novo"},
            open_pos, {"initial_capital": 1000}, 10.0,
        )
        assert not ok and "skew" in reason

    def test_livro_desbalanceado_nao_bloqueia_lado_oposto(self):
        # Mesmo livro, mas BUY_NO ajuda a equilibrar — não deve bloquear.
        open_pos = self._positions(n_yes=6, n_no=1)
        ok, reason = risk_manager.check_exposure(
            {"condition_id": "0xnew", "direction": "BUY_NO", "trade_type": "value",
             "event_slug": "ev-novo"},
            open_pos, {"initial_capital": 1000}, 10.0,
        )
        assert ok, reason

    def test_pernas_arb_nao_contam_no_skew(self):
        # Livro majoritariamente BUY_YES, mas via arb (hedge estrutural) —
        # não é uma aposta de direção, não deve entrar no cálculo.
        open_pos = self._positions(n_yes=6, n_no=1, trade_type="arb")
        ok, reason = risk_manager.check_exposure(
            {"condition_id": "0xnew", "direction": "BUY_YES", "trade_type": "value",
             "event_slug": "ev-novo"},
            open_pos, {"initial_capital": 1000}, 10.0,
        )
        assert ok, reason


class TestNormalizeTradeType:
    """P1-17: str(float('nan')) == 'nan' — string, não None, não casa em
    nenhum dict keyed por trade_type. 17 posições reais contaminadas em
    produção antes deste fix (backfill aplicado em 2026-09-04)."""

    def test_nan_float_vira_value(self):
        assert risk_manager.normalize_trade_type(float("nan")) == "value"

    def test_none_vira_value(self):
        assert risk_manager.normalize_trade_type(None) == "value"

    def test_string_literal_nan_vira_value(self):
        # A contaminação real: a string 'nan' já gravada no banco.
        assert risk_manager.normalize_trade_type("nan") == "value"

    def test_string_vazia_vira_value(self):
        assert risk_manager.normalize_trade_type("") == "value"

    def test_typo_vira_value(self):
        assert risk_manager.normalize_trade_type("valeu") == "value"

    def test_valores_validos_passam_intactos(self):
        for t in ("momentum", "value", "arb"):
            assert risk_manager.normalize_trade_type(t) == t
        assert risk_manager.normalize_trade_type("MOMENTUM") == "momentum"  # case-insensitive

    def test_warn_false_nao_loga(self, caplog):
        # get_open_positions usa warn=False — não pode spammar log a cada
        # leitura de posições já conhecidas (o warning importa na escrita).
        import io
        from loguru import logger as loguru_logger
        buf = io.StringIO()
        sink_id = loguru_logger.add(buf, level="WARNING")
        try:
            risk_manager.normalize_trade_type("nan", warn=False)
        finally:
            loguru_logger.remove(sink_id)
        assert buf.getvalue() == ""


class TestGetOpenPositionsNormalizaTradeType:
    def test_string_nan_do_banco_vira_value_na_leitura(self, tmp_path, monkeypatch):
        sys.path.insert(0, str(ROOT / "execution"))
        import paper_trader
        db = tmp_path / "paper.db"
        conn = sqlite3.connect(db)
        conn.executescript(paper_trader.SCHEMA)
        conn.execute("ALTER TABLE positions ADD COLUMN trade_type TEXT DEFAULT 'value'")
        conn.execute("""
            INSERT INTO positions
              (condition_id, question, direction, entry_price, shares, cost_usdc,
               status, trade_type, opened_at)
            VALUES ('0xaaa', 'Teste?', 'BUY_YES', 0.10, 10.0, 1.0,
                    'open', 'nan', '2026-01-01 00:00:00')
        """)
        conn.commit()
        conn.close()

        monkeypatch.setattr(paper_trader, "DB_PATH", db)
        df = paper_trader.get_open_positions()
        assert df.iloc[0]["trade_type"] == "value"


class TestRebalancePositionsCorrigido:
    """
    P1-15: rebalance_positions tinha três bugs no mesmo bloco:
      15a — sem guarda de arb, uma perna de basket podia receber mais shares
            sozinha, destruindo a invariante de shares iguais que torna o
            basket riskless.
      15b — usava o preço BID (de mark_to_market) como preço de compra do
            aporte; aportar é COMPRAR, que executa no ASK.
      15c — sobrescrevia entry_price pro custo médio (mantido — é o correto
            pra P&L/thresholds ancorados em custo; ver nota no commit).
    """

    def _seed_db(self, tmp_path, trade_type="value", cost_usdc=10.0, shares=100.0,
                 entry_price=0.10, prob_at_entry=0.60, confidence=1.0, signal_source="odds"):
        sys.path.insert(0, str(ROOT / "execution"))
        import paper_trader
        db = tmp_path / "paper.db"
        conn = sqlite3.connect(db)
        conn.executescript(paper_trader.SCHEMA)
        conn.execute("ALTER TABLE positions ADD COLUMN trade_type TEXT DEFAULT 'value'")
        conn.execute("INSERT INTO portfolio (initial_capital, current_cash) VALUES (1000, 1000)")
        conn.execute("""
            INSERT INTO positions
              (condition_id, question, direction, entry_price, shares, cost_usdc,
               prob_at_entry, confidence, signal_source, status, trade_type, opened_at)
            VALUES ('0xaaa', 'Teste?', 'BUY_YES', ?, ?, ?, ?, ?, ?, 'open', ?, '2026-01-01 00:00:00')
        """, (entry_price, shares, cost_usdc, prob_at_entry, confidence, signal_source, trade_type))
        conn.commit()
        conn.close()
        return db

    # yes_price=0.31, spread=0.02 → current_price (bid) = 0.31-0.01 = 0.30;
    # new_entry (ask, pós-fix) = 0.30+0.02 = 0.32.
    MKT = pd.DataFrame([{
        "conditionId": "0xaaa", "yes_price": 0.31, "spread": 0.02, "liquidity": 10_000.0,
    }])

    def test_pula_pernas_de_arb(self, tmp_path, monkeypatch):
        sys.path.insert(0, str(ROOT / "execution"))
        import paper_trader
        db = self._seed_db(tmp_path, trade_type="arb")
        monkeypatch.setattr(paper_trader, "DB_PATH", db)

        open_pos = _open_positions_df(db)
        mtm = paper_trader.mark_to_market(open_pos, self.MKT)
        portfolio = {"current_cash": 1000.0, "initial_capital": 1000.0}
        result = paper_trader.rebalance_positions(mtm, portfolio, dry_run=True)

        assert result == []
        # Confere que shares/cost_usdc no banco continuam intocados —
        # a invariante de shares iguais entre pernas de arb sobrevive.
        pos = _open_positions_df(db).iloc[0]
        assert float(pos["shares"]) == 100.0
        assert float(pos["cost_usdc"]) == 10.0

    def test_compra_no_ask_nao_no_bid(self, tmp_path, monkeypatch):
        sys.path.insert(0, str(ROOT / "execution"))
        import paper_trader
        db = self._seed_db(tmp_path, trade_type="value")
        monkeypatch.setattr(paper_trader, "DB_PATH", db)

        open_pos = _open_positions_df(db)
        mtm = paper_trader.mark_to_market(open_pos, self.MKT)
        assert mtm.iloc[0]["current_price"] == pytest.approx(0.30)  # bid — sanity do fixture

        portfolio = {"current_cash": 1000.0, "initial_capital": 1000.0}
        result = paper_trader.rebalance_positions(mtm, portfolio, dry_run=False)

        assert len(result) == 1
        pos = _open_positions_df(db).iloc[0]
        # new_entry = ask (0.30+0.02=0.32), não o bid (0.30) que o código
        # antigo usava. add_usdc=10 → add_shares = 10/0.32 = 31.25, não
        # 10/0.30 = 33.33 (o que o bug antigo teria comprado).
        add_usdc = result[0]["add_usdc"]
        assert add_usdc == pytest.approx(10.0)
        expected_shares = 100.0 + round(add_usdc / 0.32, 4)
        assert float(pos["shares"]) == pytest.approx(expected_shares)
        assert float(pos["shares"]) < 100.0 + round(add_usdc / 0.30, 4)  # menos shares que o bug antigo daria
        assert float(pos["cost_usdc"]) == pytest.approx(20.0)

    def test_trade_type_invalido_na_posicao_nao_quebra_e_vira_value(self, tmp_path, monkeypatch):
        # P1-17: uma posição com trade_type='nan' (contaminação real já
        # encontrada em produção) não pode travar o rebalance nem ser tratada
        # como arb por acidente — normalize_trade_type cai pra 'value'.
        sys.path.insert(0, str(ROOT / "execution"))
        import paper_trader
        db = self._seed_db(tmp_path, trade_type="nan")
        monkeypatch.setattr(paper_trader, "DB_PATH", db)

        open_pos = _open_positions_df(db)
        mtm = paper_trader.mark_to_market(open_pos, self.MKT)
        portfolio = {"current_cash": 1000.0, "initial_capital": 1000.0}
        result = paper_trader.rebalance_positions(mtm, portfolio, dry_run=True)

        assert len(result) == 1  # tratado como 'value', não pulado como se fosse 'arb'


class TestKellyCorrelacao:
    """P1-11d: Kelly independente sobre apostas correlacionadas (mesmo
    underlying) superaposta risco — escala por 1/n_correlated."""

    def test_n_correlated_1_nao_muda_nada(self):
        base = risk_manager.kelly_size(
            edge=0.10, entry_price=0.20, capital=1000, confidence=1.0,
            signal_source="odds", trade_type="value",
        )
        scaled = risk_manager.kelly_size(
            edge=0.10, entry_price=0.20, capital=1000, confidence=1.0,
            signal_source="odds", trade_type="value", n_correlated=1,
        )
        assert base == scaled

    def test_terceira_posicao_no_mesmo_underlying_arrisca_um_terco(self):
        # edge=0.08, p=0.10 (longe do KELLY_FULL_CAP e do pos_pct — isola a
        # escala): sem correlação, $22.22; com n_correlated=3, $22.22/3=$7.41.
        unscaled = risk_manager.kelly_size(
            edge=0.08, entry_price=0.10, capital=1000, confidence=1.0,
            signal_source="odds", trade_type="value",
        )
        scaled = risk_manager.kelly_size(
            edge=0.08, entry_price=0.10, capital=1000, confidence=1.0,
            signal_source="odds", trade_type="value", n_correlated=3,
        )
        assert abs(scaled - unscaled / 3) < 0.02

    def test_rejeicao_por_correlacao_loga_motivo_distinto_de_kelly_pequeno(self, tmp_path, monkeypatch):
        # P1-11d (revisão advisor): quando o Kelly some por causa de n_correlated
        # (não por edge/capital pequenos), o log precisa dizer "correlacao",
        # senão "kelly_pequeno" vira motivo genérico pra dois bugs diferentes.
        sys.path.insert(0, str(ROOT / "execution"))
        import paper_trader
        from loguru import logger as loguru_logger

        logged: list[str] = []
        sink_id = loguru_logger.add(lambda msg: logged.append(str(msg)), level="INFO")

        db = tmp_path / "paper.db"
        conn = sqlite3.connect(db)
        conn.executescript(paper_trader.SCHEMA)
        conn.execute("ALTER TABLE positions ADD COLUMN trade_type TEXT DEFAULT 'value'")
        conn.execute("ALTER TABLE positions ADD COLUMN underlying TEXT DEFAULT ''")
        conn.execute("INSERT INTO portfolio (initial_capital, current_cash) VALUES (30, 30)")
        for i in range(2):
            conn.execute("""
                INSERT INTO positions
                  (condition_id, question, underlying, direction, entry_price,
                   shares, cost_usdc, status, trade_type, opened_at)
                VALUES (?, 'BTC já aberto', 'btc', 'BUY_YES', 0.10, 10.0, 1.0,
                        'open', 'value', '2026-01-01 00:00:00')
            """, (f"0xopen{i}",))
        conn.commit()
        conn.close()

        monkeypatch.setattr(paper_trader, "DB_PATH", db)
        monkeypatch.setattr(paper_trader, "load_current_markets", lambda: pd.DataFrame())

        portfolio = {"current_cash": 30.0}
        # edge/capital pequenos o bastante pra só serem cortados quando divididos por n_correlated=3.
        signal = {
            "condition_id": "0xnovo", "question": "Bitcoin novo?", "underlying": "BTC",
            "direction": "BUY_YES", "yes_price": 0.10, "edge": 0.05, "prob_yes": 0.15,
            "confidence": 1.0, "signal_source": "deribit", "trade_type": "value", "spread": 0.01,
        }
        try:
            result = paper_trader.open_position(portfolio, signal)
        finally:
            loguru_logger.remove(sink_id)
        output = "".join(logged)

        assert result is None
        assert "correlacao n=3" in output
        assert "kelly_pequeno" not in output


class TestDrawdownPico:
    """P1-12: os stops de WEEKLY_STOP_PCT/DAILY_STOP_PCT só enxergam P&L
    REALIZADO — um portfólio pode estar -40% em mark-to-market sem nenhum
    fechamento e o stop nunca dispara. check_drawdown_stop(..., total_value=X)
    adiciona um terceiro stop sobre valor total contra o pico já visto,
    persistido em portfolio_equity."""

    @pytest.fixture()
    def empty_db(self, tmp_path):
        sys.path.insert(0, str(ROOT / "execution"))
        import paper_trader
        db = tmp_path / "paper.db"
        conn = sqlite3.connect(db)
        conn.executescript(paper_trader.SCHEMA)
        conn.execute("INSERT INTO portfolio (id, initial_capital, current_cash) VALUES (1, 1000, 1000)")
        conn.commit()
        conn.close()
        return db

    def test_sem_total_value_nao_mexe_em_portfolio_equity(self, empty_db):
        # Compatibilidade: chamador que não passa total_value (nenhum hoje,
        # mas pode existir código legado) não cria a tabela nem quebra.
        portfolio = {"id": 1, "initial_capital": 1000.0}
        stop, reason = risk_manager.check_drawdown_stop(portfolio, empty_db)
        assert stop is False
        conn = sqlite3.connect(empty_db)
        tables = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}
        conn.close()
        assert "portfolio_equity" not in tables

    def test_primeira_leitura_vira_o_proprio_pico_nao_dispara(self, empty_db):
        # Sem histórico, o valor atual É o pico — drawdown=0, não dispara.
        portfolio = {"id": 1, "initial_capital": 1000.0}
        stop, reason = risk_manager.check_drawdown_stop(portfolio, empty_db, total_value=1000.0, record=True)
        assert stop is False

    def test_queda_alem_do_cap_dispara_mesmo_sem_pnl_realizado(self, empty_db):
        # Cenário do P1-12: portfólio -40% em MTM, zero posições fechadas.
        # Sem total_value isso passaria batido; com ele, dispara.
        portfolio = {"id": 1, "initial_capital": 1000.0}
        risk_manager.check_drawdown_stop(portfolio, empty_db, total_value=1000.0, record=True)  # fixa o pico
        stop, reason = risk_manager.check_drawdown_stop(portfolio, empty_db, total_value=600.0, record=True)
        assert stop is True
        assert "pico" in reason

    def test_queda_dentro_do_cap_nao_dispara(self, empty_db):
        portfolio = {"id": 1, "initial_capital": 1000.0}
        risk_manager.check_drawdown_stop(portfolio, empty_db, total_value=1000.0, record=True)
        stop, reason = risk_manager.check_drawdown_stop(portfolio, empty_db, total_value=850.0, record=True)  # -15%, cap é 20%
        assert stop is False

    def test_recuperacao_nao_reseta_o_pico(self, empty_db):
        # O pico é o MAIOR valor já visto, não o mais recente — subir de novo
        # depois de cair não "perdoa" o drawdown contra o teto histórico.
        portfolio = {"id": 1, "initial_capital": 1000.0}
        risk_manager.check_drawdown_stop(portfolio, empty_db, total_value=1000.0, record=True)
        risk_manager.check_drawdown_stop(portfolio, empty_db, total_value=1200.0, record=True)  # novo pico
        stop, reason = risk_manager.check_drawdown_stop(portfolio, empty_db, total_value=950.0, record=True)  # -20.8% do pico 1200
        assert stop is True

    def test_portfolios_diferentes_nao_compartilham_pico(self, empty_db):
        # Reset de portfólio (novo id) não deve herdar o high-water mark do anterior.
        conn = sqlite3.connect(empty_db)
        conn.execute("INSERT INTO portfolio (id, initial_capital, current_cash) VALUES (2, 500, 500)")
        conn.commit()
        conn.close()

        risk_manager.check_drawdown_stop({"id": 1, "initial_capital": 1000.0}, empty_db, total_value=1000.0, record=True)
        # Portfólio novo com valor bem menor não deve disparar contra o pico do #1.
        stop, reason = risk_manager.check_drawdown_stop(
            {"id": 2, "initial_capital": 500.0}, empty_db, total_value=500.0, record=True,
        )
        assert stop is False

    def test_check_sem_record_nao_grava_mas_ainda_compara_com_pico_existente(self, empty_db):
        # portfolio_risk_summary usa record=False (é leitura, não pode definir
        # o pico) mas ainda precisa DETECTAR um halt já em curso — senão o
        # sumário de risco mostra stop_triggered=False durante um halt real.
        portfolio = {"id": 1, "initial_capital": 1000.0}
        risk_manager.check_drawdown_stop(portfolio, empty_db, total_value=1000.0, record=True)  # fixa o pico
        stop, _ = risk_manager.check_drawdown_stop(portfolio, empty_db, total_value=600.0, record=False)
        assert stop is True

        conn = sqlite3.connect(empty_db)
        n_rows = conn.execute("SELECT COUNT(*) FROM portfolio_equity").fetchone()[0]
        conn.close()
        assert n_rows == 1  # só o record=True acima gravou — a checagem não


class TestRunExecutionGuardas:
    """P1-13: run_execution.py negociava sem check_drawdown_stop, piso de
    liquidez, budget de slots ou cap de sinais por ciclo — as quatro guardas
    que run_paper_trading já tinha. Teste de fumaça sobre o código-fonte
    (main() é um comando click com efeitos colaterais de import pesados
    demais pra rodar ponta a ponta em unit test)."""

    SRC = (ROOT / "run_execution.py").read_text()

    def test_chama_check_drawdown_stop_com_total_value(self):
        assert "check_drawdown_stop" in self.SRC
        assert "total_value=total_value_for_stop" in self.SRC

    def test_filtra_por_liquidez_minima(self):
        # Comportamental: reproduz a expressão de filtro do arquivo (mesma
        # forma que run_paper_trading usa) e confirma que ela de fato corta
        # sinais abaixo do piso — não só que a constante aparece no arquivo.
        assert "MIN_SIGNAL_LIQUIDITY" in self.SRC
        signals = pd.DataFrame([
            {"question": "ilíquido", "liquidity": 100.0},
            {"question": "líquido", "liquidity": risk_manager.MIN_SIGNAL_LIQUIDITY + 1},
        ])
        filtered = signals[
            pd.to_numeric(signals["liquidity"], errors="coerce").fillna(0) >= risk_manager.MIN_SIGNAL_LIQUIDITY
        ]
        assert list(filtered["question"]) == ["líquido"]

    def test_respeita_budget_de_slots(self):
        assert "MAX_OPEN_POSITIONS" in self.SRC
        assert "slots -= 1" in self.SRC

    def test_capa_sinais_por_ciclo(self):
        assert "MAX_SIGNALS_PER_CYCLE" in self.SRC
        assert "signals.head(MAX_SIGNALS_PER_CYCLE)" in self.SRC

    def test_stop_bloqueia_rebalance_nao_so_abertura(self):
        # O bug original: o halt só impedia abrir posição nova, rebalance
        # rodava sem checar nada. A guarda de stop precisa vir ANTES do
        # bloco de rebalanceamento no arquivo.
        i_stop = self.SRC.index("check_drawdown_stop(portfolio, DB_PATH")
        i_rebalance = self.SRC.index("rebalance_positions(open_pos_mtm, portfolio")
        assert i_stop < i_rebalance


class TestPaperTraderStopAntesDoRebalance:
    """P1-12: mesmo bug de ordem existia em run_paper_trading — rebalance
    (que injeta mais capital em posições já perdedoras) rodava ANTES da
    checagem de stop loss, então um halt nunca protegia o rebalance."""

    SRC = (ROOT / "execution" / "paper_trader.py").read_text()

    def test_stop_vem_antes_do_rebalance_no_ciclo(self):
        i_stop = self.SRC.index("check_drawdown_stop(portfolio, DB_PATH")
        i_rebalance = self.SRC.index("rebalance_positions(open_pos_mtm, portfolio, dry_run=dry_run)")
        assert i_stop < i_rebalance


class TestOpenPositionUsaEntryPriceReal:
    """
    P1-25 (achado na revisão): o gerador de sinal já calcula entry_price
    executável (bestAsk/1-bestBid), mas open_position ignorava a coluna e
    reconstruía yes_price(lastTradePrice) + spread/2 — o filtro dizia "cobre
    custo a 0.53" e a posição era gravada a um preço diferente. Sem isso, o
    P1-25 nunca chegava no cost_usdc/shares de verdade.
    """

    def _portfolio_and_signal(self, entry_price=None, spread=0.06, liquidity=0.0):
        portfolio = {"current_cash": 1000.0}
        signal = {
            "condition_id": "0xnovo", "question": "Teste?", "underlying": "",
            "direction": "BUY_YES", "yes_price": 0.50, "edge": 0.15, "prob_yes": 0.65,
            "confidence": 1.0, "signal_source": "odds", "trade_type": "value",
            "spread": spread, "liquidity": liquidity,
        }
        if entry_price is not None:
            signal["entry_price"] = entry_price
        return portfolio, signal

    def test_com_entry_price_real_nao_reconstroi_de_yes_price(self, monkeypatch):
        sys.path.insert(0, str(ROOT / "execution"))
        import paper_trader
        monkeypatch.setattr(paper_trader, "load_current_markets", lambda: pd.DataFrame())
        monkeypatch.setattr(paper_trader, "get_open_positions", lambda: pd.DataFrame())

        portfolio, signal = self._portfolio_and_signal(entry_price=0.53)
        pos = paper_trader.open_position(portfolio, signal, dry_run=True)

        assert pos is not None
        # yes_price(0.50) + spread/2(0.03) = 0.53 coincide por acidente aqui —
        # o que importa é que NÃO veio de yes_price+spread/2 (sem meio-spread
        # de novo): com liquidity=0 e real_entry_price setado, slippage=0.
        assert pos["entry_price"] == pytest.approx(0.53)

    def test_sem_entry_price_cai_pro_legado_yes_price_mais_meio_spread(self, monkeypatch):
        sys.path.insert(0, str(ROOT / "execution"))
        import paper_trader
        monkeypatch.setattr(paper_trader, "load_current_markets", lambda: pd.DataFrame())
        monkeypatch.setattr(paper_trader, "get_open_positions", lambda: pd.DataFrame())

        portfolio, signal = self._portfolio_and_signal(entry_price=None, spread=0.06)
        pos = paper_trader.open_position(portfolio, signal, dry_run=True)

        assert pos is not None
        # Legado: yes_price(0.50) + spread/2(0.03) = 0.53 — mesmo valor do
        # teste acima, mas chegando pelo caminho antigo (sem coluna entry_price).
        assert pos["entry_price"] == pytest.approx(0.53)

    def test_nao_desconta_spread_duas_vezes_com_entry_price_real(self, monkeypatch):
        # Regressão específica: liquidity > 0 soma impacto de mercado. Com
        # real_entry_price, NÃO deve somar effective_spread/2 de novo por cima.
        sys.path.insert(0, str(ROOT / "execution"))
        import paper_trader
        monkeypatch.setattr(paper_trader, "load_current_markets", lambda: pd.DataFrame())
        monkeypatch.setattr(paper_trader, "get_open_positions", lambda: pd.DataFrame())

        portfolio, signal = self._portfolio_and_signal(entry_price=0.53, spread=0.06, liquidity=1_000_000.0)
        pos = paper_trader.open_position(portfolio, signal, dry_run=True)

        assert pos is not None
        # liquidity gigante → impact ≈ 0 → slippage ≈ 0 → entry_price ≈ real_entry_price.
        # Se o spread/2 (0.03) tivesse sido somado de novo, daria ~0.56.
        assert pos["entry_price"] == pytest.approx(0.53, abs=0.01)

    def test_re_cotacao_prefere_book_fresco_sobre_entry_price_do_sinal(self, monkeypatch):
        # current_mkts tem um bestAsk mais barato que o gravado no sinal (book
        # melhorou) — a versão fresca deve vencer, não o valor stale do sinal.
        sys.path.insert(0, str(ROOT / "execution"))
        import paper_trader
        fresh_mkts = pd.DataFrame([{
            "conditionId": "0xnovo", "yes_price": 0.50, "spread": 0.02,
            "bestBid": 0.485, "bestAsk": 0.495,
        }])
        monkeypatch.setattr(paper_trader, "load_current_markets", lambda: fresh_mkts)
        monkeypatch.setattr(paper_trader, "get_open_positions", lambda: pd.DataFrame())

        # entry_price "stale" do sinal é bem pior (0.60) que o book fresco (0.495).
        portfolio, signal = self._portfolio_and_signal(entry_price=0.60, spread=0.06)
        pos = paper_trader.open_position(portfolio, signal, dry_run=True)

        assert pos is not None
        assert pos["entry_price"] < 0.55  # veio do book fresco, não do 0.60 stale


class TestOpenPositionIgnoraPosicoesTravadas:
    """
    P1-16 (revisão do advisor): sem isso, uma posição needs_manual_resolution=1
    fica 'open' pra sempre (mercado fechou, ninguém nunca resolve) e ocupava
    slot/cap de exposição indefinidamente — sinais novos eram rejeitados por
    causa de uma posição que não é mais uma aposta ativa.
    """

    def _travadas(self, n):
        return pd.DataFrame(
            [
                # metade BUY_YES, metade BUY_NO — só pra não disparar o cap de
                # skew direcional (item 7 de check_exposure), que é ortogonal
                # ao que este teste cobre.
                {"condition_id": f"0x{i}", "direction": "BUY_YES" if i % 2 == 0 else "BUY_NO",
                 "cost_usdc": 5.0, "trade_type": "value", "needs_manual_resolution": 1}
                for i in range(n)
            ]
        )

    SIGNAL = {
        "condition_id": "0xnovo", "direction": "BUY_YES", "trade_type": "value",
        "signal_source": "odds", "category": "", "underlying": "",
    }

    def test_posicoes_travadas_nao_contam_no_limite_global(self):
        from risk_manager import check_exposure, MAX_OPEN_POSITIONS

        ok, reason = check_exposure(
            self.SIGNAL, self._travadas(MAX_OPEN_POSITIONS),
            {"initial_capital": 1000.0}, 5.0,
        )
        # Antes da correção: MAX_OPEN_POSITIONS posições travadas bloqueavam
        # qualquer sinal novo ("limite global atingido") mesmo com todo mundo
        # esperando revisão manual, não disputando capital de verdade.
        assert ok, reason

    def test_duplicata_de_posicao_travada_continua_bloqueada(self):
        # A exclusão do limite global NÃO pode furar a checagem de duplicata —
        # a posição travada continua com capital de verdade comprometido nela.
        from risk_manager import check_exposure, MAX_OPEN_POSITIONS

        signal = {**self.SIGNAL, "condition_id": "0x0"}  # mesmo cid de uma posição travada
        ok, reason = check_exposure(
            signal, self._travadas(MAX_OPEN_POSITIONS - 1),
            {"initial_capital": 1000.0}, 5.0,
        )
        assert not ok
        assert "já aberta" in reason


class TestArbAlert:
    OPP = {
        "kind": "monotonicity", "guaranteed": True, "arb_group": "mono:0xa>0xb",
        "n_legs": 2, "basket_cost": 0.88, "payout_min": 1.0,
        "profit": 0.12, "edge": 0.12,
    }

    @pytest.fixture()
    def notify_mod(self, tmp_path, monkeypatch):
        sys.path.insert(0, str(ROOT))
        import notify
        monkeypatch.setattr(notify, "ARB_ALERT_STATE", tmp_path / "state.json")
        sent = []
        monkeypatch.setattr(notify, "_send", lambda text: sent.append(text) or True)
        return notify, sent

    def test_alerta_enviado_para_garantida(self, notify_mod):
        notify, sent = notify_mod
        assert notify.arb_alert([self.OPP]) is True
        assert len(sent) == 1 and "mono:0xa>0xb" in sent[0]

    def test_cooldown_evita_spam_no_ciclo_seguinte(self, notify_mod):
        notify, sent = notify_mod
        notify.arb_alert([self.OPP])
        assert notify.arb_alert([self.OPP]) is False  # 30min depois: dedupe
        assert len(sent) == 1

    def test_condicional_nao_alerta(self, notify_mod):
        notify, sent = notify_mod
        opp = {**self.OPP, "guaranteed": False}
        assert notify.arb_alert([opp]) is False
        assert sent == []

    def test_falha_de_envio_nao_grava_cooldown(self, notify_mod, monkeypatch):
        notify, sent = notify_mod
        monkeypatch.setattr(notify, "_send", lambda text: False)
        assert notify.arb_alert([self.OPP]) is False
        # Próximo ciclo (envio volta a funcionar) deve re-tentar
        monkeypatch.setattr(notify, "_send", lambda text: sent.append(text) or True)
        assert notify.arb_alert([self.OPP]) is True


# ──────────────────────────────────────────────────────────
# market_pricing — preço executável e net edge (P1-25/P1-26)
# ──────────────────────────────────────────────────────────

class TestEntryPriceNetEdge:
    """
    P1-25: yes_price na camada de sinal é lastTradePrice — uma impressão, não
    um preço executável. entry_price_and_net_edge usa bestBid/bestAsk de
    verdade (já coletados, nunca usados pra isso) e desconta o custo real.
    """

    def test_buy_yes_usa_best_ask(self):
        mkt = {"bestBid": 0.40, "bestAsk": 0.45, "spread": 0.05}
        result = market_pricing.entry_price_and_net_edge(
            mkt, "BUY_YES", fair_yes=0.55, fair_no=0.45, source="odds",
        )
        assert result is not None
        entry_price, spread, net_edge = result
        assert entry_price == 0.45
        assert spread == pytest.approx(0.05)
        assert net_edge == pytest.approx(0.55 - 0.45)  # fee=0 hoje

    def test_buy_no_usa_um_menos_best_bid(self):
        mkt = {"bestBid": 0.40, "bestAsk": 0.45, "spread": 0.05}
        result = market_pricing.entry_price_and_net_edge(
            mkt, "BUY_NO", fair_yes=0.30, fair_no=0.70, source="odds",
        )
        assert result is not None
        entry_price, spread, net_edge = result
        assert entry_price == pytest.approx(0.60)
        assert net_edge == pytest.approx(0.70 - 0.60)

    def test_sem_ask_pra_buy_yes_retorna_none(self):
        mkt = {"bestBid": 0.40, "bestAsk": None, "spread": 0.05}
        assert market_pricing.entry_price_and_net_edge(mkt, "BUY_YES", 0.55, 0.45, source="odds") is None

    def test_sem_bid_pra_buy_no_retorna_none(self):
        mkt = {"bestBid": None, "bestAsk": 0.45, "spread": 0.05}
        assert market_pricing.entry_price_and_net_edge(mkt, "BUY_NO", 0.30, 0.70, source="odds") is None

    def test_spread_largo_demais_descarta_o_candidato(self):
        # MIN_EDGE_TO_TRADE["odds"]=0.08 × MAX_SPREAD_TO_MIN_EDGE_RATIO=1.0 → veto acima de 0.08
        mkt = {"bestBid": 0.30, "bestAsk": 0.45, "spread": 0.15}  # spread real = 0.15
        assert market_pricing.entry_price_and_net_edge(mkt, "BUY_YES", 0.55, 0.45, source="odds") is None

    def test_spread_dentro_do_limite_passa(self):
        mkt = {"bestBid": 0.42, "bestAsk": 0.45, "spread": 0.03}
        result = market_pricing.entry_price_and_net_edge(mkt, "BUY_YES", 0.55, 0.45, source="odds")
        assert result is not None

    def test_spread_do_veto_vem_da_coluna_quando_bid_falta(self):
        # Sem bestBid não dá pra calcular ask-bid — cai pro campo spread bruto.
        mkt = {"bestBid": None, "bestAsk": 0.10, "spread": 0.02}
        result = market_pricing.entry_price_and_net_edge(mkt, "BUY_YES", 0.15, 0.85, source="odds")
        assert result is not None
        _, spread, _ = result
        assert spread == pytest.approx(0.02)

    def test_piso_de_spread_e_por_fonte(self):
        # deribit tem MIN_EDGE_TO_TRADE menor (0.05) — mesmo spread que passaria
        # em odds (0.08) é vetado em deribit.
        mkt = {"bestBid": 0.40, "bestAsk": 0.47, "spread": 0.07}
        assert market_pricing.entry_price_and_net_edge(mkt, "BUY_YES", 0.55, 0.45, source="odds") is not None
        assert market_pricing.entry_price_and_net_edge(mkt, "BUY_YES", 0.55, 0.45, source="deribit") is None


class TestApplyEdgeShrinkage:
    """P1-26: maldição do vencedor — ordenar por net_edge bruto entre milhares
    de estimativas ruidosas seleciona o topo da distribuição de erro, não de
    mispricing real. Encolhe proporcionalmente ao consensus_spread."""

    def test_sem_net_edge_faz_passthrough_do_abs_divergence(self):
        df = pd.DataFrame([{"abs_divergence": 0.12}, {"abs_divergence": 0.08}])
        out = market_pricing.apply_edge_shrinkage(df)
        assert list(out["shrunk_edge"]) == [0.12, 0.08]

    def test_variancia_degenerada_nao_encolhe(self):
        # net_edge idênticos → var_edge=0 → sem correção possível, passa direto.
        df = pd.DataFrame([
            {"net_edge": 0.10, "consensus_spread": 0.01},
            {"net_edge": 0.10, "consensus_spread": 0.05},
        ])
        out = market_pricing.apply_edge_shrinkage(df)
        assert list(out["shrunk_edge"]) == [0.10, 0.10]

    def test_consensus_spread_alto_encolhe_mais_que_baixo(self):
        # Terceira linha só pra dar variância não-degenerada ao lote (sem ela
        # var_edge=0 e nada encolhe, ver test_variancia_degenerada_nao_encolhe).
        df = pd.DataFrame([
            {"net_edge": 0.20, "consensus_spread": 0.001},  # books concordam — quase intacto
            {"net_edge": 0.20, "consensus_spread": 0.30},   # books discordam — puxado a zero
            {"net_edge": 0.05, "consensus_spread": 0.05},
        ])
        out = market_pricing.apply_edge_shrinkage(df)
        low_spread_shrunk, high_spread_shrunk = out["shrunk_edge"].iloc[0], out["shrunk_edge"].iloc[1]
        assert low_spread_shrunk > high_spread_shrunk
        assert low_spread_shrunk == pytest.approx(0.20, abs=0.02)
        assert high_spread_shrunk < 0.20 * 0.5

    def test_math_bate_com_a_formula_manual(self):
        rows = [
            {"net_edge": 0.10, "consensus_spread": 0.02},
            {"net_edge": 0.15, "consensus_spread": 0.05},
            {"net_edge": 0.05, "consensus_spread": 0.10},
        ]
        df = pd.DataFrame(rows)
        var_edge = df["net_edge"].var()
        expected = [
            round(r["net_edge"] * var_edge / (var_edge + r["consensus_spread"] ** 2), 4)
            for r in rows
        ]
        out = market_pricing.apply_edge_shrinkage(df)
        assert list(out["shrunk_edge"]) == expected


# ──────────────────────────────────────────────────────────
# odds_collector — matching com entry_price/net_edge (P1-25)
# ──────────────────────────────────────────────────────────

def _fake_h2h_event(home_team, away_team, home_odds, away_odds, sport_key="basketball_nba",
                     hours_from_now=24, bookmaker="pinnacle"):
    commence = datetime.now(timezone.utc) + pd.Timedelta(hours=hours_from_now)
    return {
        "sport_key": sport_key,
        "home_team": home_team,
        "away_team": away_team,
        "commence_time": commence.isoformat(),
        "bookmakers": [{
            "key": bookmaker,
            "markets": [{
                "key": "h2h",
                "outcomes": [
                    {"name": home_team, "price": home_odds},
                    {"name": away_team, "price": away_odds},
                ],
            }],
        }],
    }


def _fake_market_row(condition_id, question, yes_price, best_bid, best_ask, spread,
                      hours_from_now=24, liquidity=10_000.0):
    end_date = datetime.now(timezone.utc) + pd.Timedelta(hours=hours_from_now)
    return {
        "conditionId": condition_id,
        "question":    question,
        "category":    "sports",
        "yes_price":   yes_price,
        "bestBid":     best_bid,
        "bestAsk":     best_ask,
        "spread":      spread,
        "endDate":     end_date.isoformat(),
        "liquidity":   liquidity,
        "volume24hr":  50_000.0,
    }


class TestExtractFairProbsSemDrawFantasma:
    """P1-22: nomes de outcome cosmeticamente diferentes de home_team/away_team
    (comparação exata antes deste fix) caíam no else e viravam "draw" —
    inclusive em esporte de 2 vias, sem empate possível (NBA)."""

    def _event(self, home_team, away_team, outcome_names, odds=(1.60, 2.50), sport_key="basketball_nba"):
        return {
            "sport_key": sport_key,
            "home_team": home_team,
            "away_team": away_team,
            "bookmakers": [{
                "key": "pinnacle",
                "markets": [{
                    "key": "h2h",
                    "outcomes": [
                        {"name": outcome_names[0], "price": odds[0]},
                        {"name": outcome_names[1], "price": odds[1]},
                    ],
                }],
            }],
        }

    def test_nome_com_alias_conhecido_nao_fabrica_draw(self):
        # event.home_team usa a forma abreviada; o outcome do bookmaker usa a
        # forma canônica completa — ambos resolvem pro mesmo _normalize_team.
        event = self._event(
            home_team="LA Lakers", away_team="LA Clippers",
            outcome_names=["Los Angeles Lakers", "Los Angeles Clippers"],
        )
        fair = odds_collector.extract_fair_probs(event)
        assert fair is not None
        assert "draw" not in fair
        assert "home" in fair and "away" in fair

    def test_nome_sem_alias_conhecido_descarta_o_book_em_vez_de_dar_draw(self):
        # "LAL" não está em TEAM_ALIASES/SPORT_ALIASES — não bate com home
        # nem away por nome, e não é "draw"/"tie"/"empate" literal. Antes do
        # fix isso virava book_entry["draw"]; agora o book inteiro é descartado.
        event = self._event(
            home_team="Los Angeles Lakers", away_team="Los Angeles Clippers",
            outcome_names=["LAL", "LAC"],
        )
        fair = odds_collector.extract_fair_probs(event)
        assert fair is None  # único book, descartado → nenhum book válido

    def test_draw_literal_em_futebol_ainda_funciona(self):
        # Outcome "Draw" de verdade (esporte de 3 vias) continua sendo
        # reconhecido — o fix não deve quebrar o caso legítimo.
        event = {
            "sport_key": "soccer_epl",
            "home_team": "Arsenal",
            "away_team": "Chelsea",
            "bookmakers": [{
                "key": "pinnacle",
                "markets": [{
                    "key": "h2h",
                    "outcomes": [
                        {"name": "Arsenal", "price": 2.10},
                        {"name": "Chelsea", "price": 3.40},
                        {"name": "Draw", "price": 3.60},
                    ],
                }],
            }],
        }
        fair = odds_collector.extract_fair_probs(event)
        assert fair is not None
        assert "draw" in fair and "home" in fair and "away" in fair

    def test_terceiro_outcome_nao_literal_ainda_vira_draw_por_eliminacao(self):
        # Revisão advisor: um bookmaker que rotula o empate como "Tie (90 mins)"
        # em vez de "Draw" exato não pode derrubar o book inteiro — home e away
        # já bateram por nome, e num mercado de 3 vias o único outcome que sobra
        # só pode ser o empate. Descartar o book aqui jogaria fora probabilidades
        # boas de home/away por causa de um rótulo de terceira via não reconhecido.
        event = {
            "sport_key": "soccer_epl",
            "home_team": "Arsenal",
            "away_team": "Chelsea",
            "bookmakers": [{
                "key": "pinnacle",
                "markets": [{
                    "key": "h2h",
                    "outcomes": [
                        {"name": "Arsenal", "price": 2.10},
                        {"name": "Chelsea", "price": 3.40},
                        {"name": "Tie (90 mins)", "price": 3.60},
                    ],
                }],
            }],
        }
        fair = odds_collector.extract_fair_probs(event)
        assert fair is not None
        assert "draw" in fair and "home" in fair and "away" in fair

    def test_away_nao_identificado_descarta_mesmo_com_home_batendo(self):
        # Só o home bate por nome ("Arsenal") — away não bate com "Chelsea"
        # de jeito nenhum ("XYZ FC"). A eliminação por contagem exige home E
        # away batidos antes de assumir a sobra como draw; com away ausente,
        # nenhuma das probabilidades desse book é confiável — descarta.
        event = {
            "sport_key": "soccer_epl",
            "home_team": "Arsenal",
            "away_team": "Chelsea",
            "bookmakers": [{
                "key": "pinnacle",
                "markets": [{
                    "key": "h2h",
                    "outcomes": [
                        {"name": "Arsenal", "price": 2.10},
                        {"name": "XYZ FC", "price": 3.40},
                        {"name": "Tie (90 mins)", "price": 3.60},
                    ],
                }],
            }],
        }
        fair = odds_collector.extract_fair_probs(event)
        assert fair is None


class TestOddsDirecaoConfrontosMesmaCidade:
    """
    P1-21: a versão antiga escolhia YES por posição de TOKEN isolado — em
    confrontos da mesma cidade, tokens de cidade compartilhados ("los
    angeles") empatavam as duas posições e o desempate escolhia sempre o
    mandante, invertendo o lado. Quando nenhum time aparecia no texto, os
    dois empatavam em len(text) e fair=0.5 (default) fabricava edge do nada.
    """

    def _match(self, mkt_row, event):
        markets_df = pd.DataFrame([mkt_row])
        return odds_collector.match_markets_to_odds(markets_df, [event], min_score=0.25)

    def test_confronto_mesma_cidade_resolve_pro_time_certo_nao_pro_mandante(self):
        # home=Clippers, away=Lakers — a pergunta menciona o Clippers primeiro.
        # A versão antiga inverteria (tokens "los"/"angeles" empatados →
        # escolhe home só por ser home). O nome completo "los angeles
        # clippers" só bate na posição onde ELE aparece.
        event = _fake_h2h_event("Los Angeles Clippers", "Los Angeles Lakers", 1.60, 2.50)
        mkt = _fake_market_row(
            "0xabc", "Will the Los Angeles Clippers beat the Los Angeles Lakers?",
            yes_price=0.50, best_bid=0.47, best_ask=0.53, spread=0.06,
        )
        df = self._match(mkt, event)
        assert not df.empty
        assert df.iloc[0]["yes_team"] == "Los Angeles Clippers"

    def test_nenhum_time_no_texto_nao_fabrica_edge(self):
        # Pergunta genérica que não cita nenhum dos dois times pelo nome —
        # antes, os dois empatavam em len(text) e caía no ramo "home" com
        # fair=0,5 (fabricado). Agora não dá pra saber quem é YES → descarta.
        event = _fake_h2h_event("Los Angeles Clippers", "Los Angeles Lakers", 1.60, 2.50)
        mkt = _fake_market_row(
            "0xabc", "Will the home team win tonight's game?",
            yes_price=0.50, best_bid=0.47, best_ask=0.53, spread=0.06,
        )
        df = self._match(mkt, event)
        assert df.empty

    def test_time_visitante_citado_primeiro_resolve_pro_visitante(self):
        # home=Lakers, away=Celtics, mas a pergunta cita o Celtics primeiro —
        # YES precisa ser o Celtics (away), não o mandante por default.
        event = _fake_h2h_event("Los Angeles Lakers", "Boston Celtics", 1.60, 2.50)
        mkt = _fake_market_row(
            "0xabc", "Will the Boston Celtics beat the Los Angeles Lakers?",
            yes_price=0.50, best_bid=0.47, best_ask=0.53, spread=0.06,
        )
        df = self._match(mkt, event)
        assert not df.empty
        assert df.iloc[0]["yes_team"] == "Boston Celtics"

    def test_repro_mets_yankees_do_documento_de_auditoria(self):
        # Segunda linha da tabela de repro do P1-21 (a primeira é o teste do
        # Lakers/Clippers acima): "Will the New York Mets beat the New York
        # Yankees?" invertia pro mesmo motivo (tokens de cidade empatados).
        event = _fake_h2h_event("New York Mets", "New York Yankees", 1.60, 2.50, sport_key="baseball_mlb")
        mkt = _fake_market_row(
            "0xabc", "Will the New York Mets beat the New York Yankees?",
            yes_price=0.50, best_bid=0.47, best_ask=0.53, spread=0.06,
        )
        df = self._match(mkt, event)
        assert not df.empty
        assert df.iloc[0]["yes_team"] == "New York Mets"

    # Rivais reais da mesma cidade que jogam entre si de verdade — pares
    # cross-sport (ex: "Washington Capitals" x "Washington Commanders", NHL x
    # NFL) nunca aparecem no mesmo mercado, então não entram aqui: variam o
    # sport_key certo do par, não uma cidade qualquer.
    SAME_CITY_RIVALS = [
        ("Los Angeles Lakers", "Los Angeles Clippers", "basketball_nba"),
        ("New York Knicks", "Brooklyn Nets", "basketball_nba"),
        ("New York Mets", "New York Yankees", "baseball_mlb"),
        ("Los Angeles Dodgers", "Los Angeles Angels", "baseball_mlb"),
        ("Chicago Cubs", "Chicago White Sox", "baseball_mlb"),
        ("New York Giants", "New York Jets", "americanfootball_nfl"),
        ("Los Angeles Rams", "Los Angeles Chargers", "americanfootball_nfl"),
        ("New York Rangers", "New York Islanders", "icehockey_nhl"),
    ]

    def test_rivais_reais_da_mesma_cidade_resolvem_pro_time_citado_primeiro(self):
        # Revisão advisor: em vez de só o par escolhido à mão acima, varre
        # vários rivais reais de mesma cidade — se algum alias curto
        # colidisse como substring do nome completo do rival, apareceria
        # aqui como inversão ou queda inesperada num desses pares.
        failures = []
        for team_a, team_b, sport_key in self.SAME_CITY_RIVALS:
            event = _fake_h2h_event(team_a, team_b, 1.60, 2.50, sport_key=sport_key)
            mkt = _fake_market_row(
                "0xabc", f"Will the {team_a} beat the {team_b}?",
                yes_price=0.50, best_bid=0.47, best_ask=0.53, spread=0.06,
            )
            df = self._match(mkt, event)
            if df.empty or df.iloc[0]["yes_team"] != team_a:
                failures.append((team_a, team_b, df.iloc[0]["yes_team"] if not df.empty else "VAZIO"))

        assert failures == [], f"pares que não resolveram pro time citado primeiro: {failures}"

    def test_apelido_de_sport_alias_nao_vaza_pra_outro_esporte(self):
        # Achado na revisão: _REVERSE_ALIASES antes despejava os apelidos de
        # TODOS os esportes no mesmo dict global — "washington" (apelido só
        # do americanfootball, "Washington Commanders") virava variante de
        # busca mesmo com sport_key de outro esporte, colidindo com QUALQUER
        # time começando com "Washington" nesse esporte (ex: Washington
        # Capitals, NHL — nunca jogam entre si, mas se aparecessem no mesmo
        # texto por acidente, empatavam por posição). Escopado por esporte:
        # "washington" não pode ser variante de "Washington Commanders" fora
        # do sport_key de futebol americano.
        variants_wrong_sport = odds_collector._team_variants("Washington Commanders", "icehockey_nhl")
        assert "washington" not in variants_wrong_sport

        variants_right_sport = odds_collector._team_variants("Washington Commanders", "americanfootball_nfl")
        assert "washington" in variants_right_sport


class TestMatchMarketsToOddsEntryPrice:
    def _match(self, mkt_row, event):
        markets_df = pd.DataFrame([mkt_row])
        return odds_collector.match_markets_to_odds(markets_df, [event], min_score=0.25)

    def test_linha_matched_carrega_entry_price_e_net_edge(self):
        # Fair (sem vig) pra odds [1.60, 2.50] favorece o home moderadamente.
        # yes_price=0.50 empurra a divergência mid-based pra BUY_NO ou BUY_YES
        # dependendo do fair — o teste só precisa que as colunas apareçam e
        # sejam consistentes com bestBid/bestAsk, não de um valor fixo de fair.
        event = _fake_h2h_event("Los Angeles Lakers", "Boston Celtics", 1.60, 2.50)
        mkt = _fake_market_row(
            "0xabc", "Los Angeles Lakers vs. Boston Celtics",
            yes_price=0.50, best_bid=0.48, best_ask=0.52, spread=0.04,
        )
        df = self._match(mkt, event)
        assert not df.empty
        row = df.iloc[0]
        assert {"entry_price", "spread", "net_edge"}.issubset(df.columns)
        if row["direction"] == "BUY_YES":
            assert row["entry_price"] == pytest.approx(0.52)
        else:
            assert row["entry_price"] == pytest.approx(1.0 - 0.48)

    def test_sem_book_do_lado_necessario_descarta_a_linha(self):
        # fair_home ≈ 0.60 (odds 1.65/2.50, overround baixo — passa o gate de
        # plausibilidade); yes_price=0.40 dá divergência -0.20 → direção
        # BUY_YES. Sem bestAsk, a linha não pode ser precificada de verdade.
        event = _fake_h2h_event("Los Angeles Lakers", "Boston Celtics", 1.65, 2.50)
        mkt = _fake_market_row(
            "0xabc", "Los Angeles Lakers vs. Boston Celtics",
            yes_price=0.40, best_bid=0.38, best_ask=None, spread=0.05,
        )
        df = self._match(mkt, event)
        assert df.empty

    def test_spread_largo_demais_descarta_a_linha(self):
        event = _fake_h2h_event("Los Angeles Lakers", "Boston Celtics", 1.60, 2.50)
        mkt = _fake_market_row(
            "0xabc", "Los Angeles Lakers vs. Boston Celtics",
            yes_price=0.50, best_bid=0.30, best_ask=0.70, spread=0.40,  # spread=0.40 >> 0.08
        )
        df = self._match(mkt, event)
        assert df.empty

    def test_net_edge_menor_que_divergence_mid_quando_ha_spread(self):
        # O ponto inteiro do P1-25: o edge de verdade é MENOR que a divergência
        # calculada contra o mid/last, porque paga o spread. fair_home≈0.60
        # (odds 1.65/2.50); yes_price=0.50 → divergência mid = -0.10,
        # direção BUY_YES; net_edge = fair - bestAsk(0.53) = 0.60-0.53≈0.07.
        # spread=0.06 fica sob o piso de veto (0.08) — precisa passar, não só existir.
        event = _fake_h2h_event("Los Angeles Lakers", "Boston Celtics", 1.65, 2.50)
        mkt = _fake_market_row(
            "0xabc", "Los Angeles Lakers vs. Boston Celtics",
            yes_price=0.50, best_bid=0.47, best_ask=0.53, spread=0.06,
        )
        df = self._match(mkt, event)
        assert not df.empty
        row = df.iloc[0]
        assert abs(row["net_edge"]) < row["abs_divergence"]


class TestMatchOutrightEntryPrice:
    def test_outright_carrega_entry_price_e_net_edge(self):
        event = {
            "sport_key": "basketball_nba_championship_winner",
            "sport_title": "NBA Championship Winner",
            "bookmakers": [{
                "key": "pinnacle",
                "markets": [{
                    "key": "outrights",
                    "outcomes": [
                        {"name": "Boston Celtics", "price": 3.5},
                        {"name": "Los Angeles Lakers", "price": 8.0},
                        {"name": "Denver Nuggets", "price": 12.0},
                    ],
                }],
            }],
        }
        mkt = pd.DataFrame([{
            "conditionId": "0xout1",
            "question":    "Will the Boston Celtics win the championship?",
            "category":    "sports",
            "yes_price":   0.25,
            "bestBid":     0.22,
            "bestAsk":     0.28,
            "spread":      0.06,
            "endDate":     (datetime.now(timezone.utc) + pd.Timedelta(days=60)).isoformat(),
            "liquidity":   20_000.0,
            "volume24hr":  100_000.0,
        }])
        df = odds_collector.match_outright_markets(mkt, [event], min_score=0.35)
        assert not df.empty
        row = df.iloc[0]
        assert {"entry_price", "spread", "net_edge"}.issubset(df.columns)


# ──────────────────────────────────────────────────────────
# signal_generator — edge de verdade chega no Kelly (P1-25/P1-26)
# ──────────────────────────────────────────────────────────

class TestSignalGeneratorUsaEdgeDeVerdade:
    """
    O fix do P1-25/P1-26 só importa se o `edge` que o Kelly usa pra dimensionar
    a posição vier do net_edge/shrunk_edge — não da divergência bruta contra
    yes_price (lastTradePrice). Testa a costura final: parquet do collector →
    campo "edge" consumido por risk_manager.kelly_size.
    """

    def _odds_row(self, **overrides):
        now = datetime.now(timezone.utc)
        row = {
            "condition_id": "0xabc", "question": "Lakers vs Celtics", "category": "sports",
            "underlying": "basketball_nba:x", "yes_price": 0.50, "entry_price": 0.55,
            "spread": 0.10, "fair_prob_yes": 0.65, "fair_prob_no": 0.35,
            "divergence": -0.15, "abs_divergence": 0.15, "net_edge": 0.10,
            "shrunk_edge": 0.07,  # encolhido — menor que net_edge bruto
            "direction": "BUY_YES", "yes_team": "Lakers", "home_team": "Lakers",
            "away_team": "Celtics", "bookmaker": "pinnacle", "overround": 1.02,
            "is_sharp": True, "n_books": 3, "consensus_spread": 0.02,
            "vig_method": "multiplicative", "match_score": 0.9, "sport_key": "basketball_nba",
            "commence_time": (now + pd.Timedelta(hours=24)).isoformat(),
            "end_date": (now + pd.Timedelta(hours=26)).isoformat(),
            "liquidity": 10_000.0, "volume_24h": 50_000.0, "market_type": "h2h",
        }
        row.update(overrides)
        return row

    def test_edge_final_e_o_shrunk_edge_nao_o_mid_bruto(self, tmp_path, monkeypatch):
        monkeypatch.setattr(signal_generator, "RAW_ODDS_DIR", tmp_path)
        pd.DataFrame([self._odds_row()]).to_parquet(tmp_path / "odds_matched_20260101_000000.parquet")

        signals = signal_generator.generate_odds_signals(
            min_divergence=0.03, min_liquidity=1_000, min_volume24h=0,
            min_days_left=0, min_match_score=0.25, save=False,
        )
        assert not signals.empty
        row = signals.iloc[0]
        assert row["edge"] == pytest.approx(0.07)          # shrunk_edge, não 0.15 (mid) nem 0.10 (net cru)
        assert row["abs_edge"] == pytest.approx(0.07)

    def test_compat_parquet_antigo_sem_net_edge_cai_pro_calculo_legado(self, tmp_path, monkeypatch):
        monkeypatch.setattr(signal_generator, "RAW_ODDS_DIR", tmp_path)
        legacy_row = self._odds_row()
        del legacy_row["net_edge"]
        del legacy_row["shrunk_edge"]
        pd.DataFrame([legacy_row]).to_parquet(tmp_path / "odds_matched_20260101_000000.parquet")

        signals = signal_generator.generate_odds_signals(
            min_divergence=0.03, min_liquidity=1_000, min_volume24h=0,
            min_days_left=0, min_match_score=0.25, save=False,
        )
        assert not signals.empty
        row = signals.iloc[0]
        # legado: prob_yes - yes_price = 0.65 - 0.50 = 0.15
        assert row["edge"] == pytest.approx(0.15)

    def test_deribit_signals_sempre_carregam_trade_type_value(self, tmp_path, monkeypatch):
        # P1-17: generate_deribit_signals nunca setava trade_type — em
        # load_signals(mode="all"), o concat com sinais odds (que TÊM a
        # coluna) sobrava NaN pros deribit, e str(nan)=='nan' rio abaixo.
        # 17 posições reais em produção ficaram com trade_type='nan' por
        # causa disso (backfill aplicado em 2026-09-04).
        monkeypatch.setattr(signal_generator, "RAW_ODDS_DIR", tmp_path)
        now = datetime.now(timezone.utc)
        row = {
            "condition_id": "0xdef", "question": "Bitcoin acima de $70k?",
            "category": "crypto", "underlying": "BTC", "asset": "BTC", "strike": 70_000.0,
            "direction": "above", "yes_price": 0.40, "entry_price": 0.42, "spread": 0.04,
            "fair_prob": 0.55, "divergence": -0.15, "abs_divergence": 0.15,
            "net_edge": 0.13, "signal": "BUY_YES", "is_touch": False,
            "spot_price": 68_000.0, "iv": 0.55, "iv_pct": "55.0%", "T_days": 10.0,
            "moneyness_pct": 0.03, "bs_reach": 0.10,
            "deribit_expiry": (now + pd.Timedelta(days=10)).strftime("%Y-%m-%d"),
            "expiry_delta_d": 0, "liquidity": 10_000.0, "volume_24h": 50_000.0,
            "hours_left": 240.0, "end_date": (now + pd.Timedelta(hours=240)).isoformat(),
        }
        path = tmp_path / "deribit_signals_20260101_000000.parquet"
        pd.DataFrame([row]).to_parquet(path)

        signals = signal_generator.generate_deribit_signals(
            min_divergence=0.05, min_liquidity=1_000, min_hours_left=1.0,
            save=False, fetch_fresh=False,
        )
        assert not signals.empty
        assert (signals["trade_type"] == "value").all()


def _valid_open_signal(condition_id="0xnovo"):
    return {
        "condition_id": condition_id, "question": "Teste?", "underlying": "",
        "direction": "BUY_YES", "yes_price": 0.50, "edge": 0.15, "prob_yes": 0.65,
        "confidence": 1.0, "signal_source": "odds", "trade_type": "value",
        "spread": 0.06, "liquidity": 0.0,
    }


class TestOpenPositionDebitaOPortfolioAtual:
    """P1-18: open_position mirava WHERE id = ? com portfolio["id"] em cache
    — já houve 9 resets em produção; depois de um, o débito ia pro portfólio
    aposentado e o crédito de fechamentos futuros caía no portfólio novo,
    perdendo dinheiro silenciosamente. A correção usa
    WHERE id = (SELECT MAX(id) FROM portfolio), igual open_basket/
    rebalance_positions/resolve_positions já faziam."""

    def test_debita_do_portfolio_mais_novo_nao_do_id_em_cache(self, tmp_path, monkeypatch):
        db = tmp_path / "paper.db"
        monkeypatch.setattr(paper_trader, "DB_PATH", db)
        paper_trader.init_db()

        conn = sqlite3.connect(db)
        conn.execute("INSERT INTO portfolio (id, initial_capital, current_cash) VALUES (1, 1000, 1000)")
        # Simula um reset: portfólio novo, id maior, cash menor.
        conn.execute("INSERT INTO portfolio (id, initial_capital, current_cash) VALUES (2, 500, 500)")
        conn.commit()
        conn.close()

        # portfolio["id"]=1 é a referência EM CACHE (stale) que um chamador de
        # vida longa carregaria de antes do reset — current_cash reflete o
        # portfólio 2 real, como get_or_create_portfolio() devolveria hoje.
        stale_portfolio = {"id": 1, "current_cash": 500.0}
        pos = paper_trader.open_position(stale_portfolio, _valid_open_signal(), dry_run=False)

        assert pos is not None
        conn = sqlite3.connect(db)
        cash_id1 = conn.execute("SELECT current_cash FROM portfolio WHERE id=1").fetchone()[0]
        cash_id2 = conn.execute("SELECT current_cash FROM portfolio WHERE id=2").fetchone()[0]
        conn.close()
        assert cash_id1 == 1000.0  # portfólio aposentado intocado
        assert cash_id2 == pytest.approx(500.0 - pos["cost_usdc"])  # débito no atual


class TestOpenPositionBeginImmediateERetry:
    """P1-18: única gravação de dinheiro sem BEGIN IMMEDIATE/retry — um
    OperationalError('database is locked') subia sem try/except e matava o
    subprocesso no meio do ciclo."""

    SRC = (ROOT / "execution" / "paper_trader.py").read_text()

    def test_open_position_usa_begin_immediate(self):
        i_def = self.SRC.index("def open_position(")
        i_next_def = self.SRC.index("\ndef ", i_def + 1)
        body = self.SRC[i_def:i_next_def]
        assert "BEGIN IMMEDIATE" in body
        assert "ADD COLUMN" not in body  # P1-18: ALTER TABLE saiu da transação de dinheiro

    def test_get_connection_seta_busy_timeout(self):
        assert "PRAGMA busy_timeout" in self.SRC

    def test_retry_em_database_locked(self, tmp_path, monkeypatch):
        db = tmp_path / "paper.db"
        monkeypatch.setattr(paper_trader, "DB_PATH", db)
        paper_trader.init_db()
        conn = sqlite3.connect(db)
        conn.execute("INSERT INTO portfolio (initial_capital, current_cash) VALUES (1000, 1000)")
        conn.commit()
        conn.close()

        class _FlakyConn:
            """Proxy que falha o BEGIN IMMEDIATE uma vez só, depois se comporta normal.
            sqlite3.Connection não aceita sobrescrever .execute na instância
            (atributo read-only do tipo C), daí o proxy em vez de monkeypatch direto."""
            def __init__(self, real):
                object.__setattr__(self, "_real", real)
                object.__setattr__(self, "_armed", True)

            def execute(self, sql, *a, **kw):
                if object.__getattribute__(self, "_armed") and sql.strip() == "BEGIN IMMEDIATE":
                    object.__setattr__(self, "_armed", False)
                    raise sqlite3.OperationalError("database is locked")
                return self._real.execute(sql, *a, **kw)

            def __getattr__(self, name):
                return getattr(self._real, name)

            def __setattr__(self, name, value):
                setattr(self._real, name, value)

        state = {"n": 0, "flaky": None}
        real_get_connection = paper_trader.get_connection

        def flaky_get_connection():
            state["n"] += 1
            real_conn = real_get_connection()
            if state["n"] == 1:
                state["flaky"] = _FlakyConn(real_conn)
                return state["flaky"]
            return real_conn

        monkeypatch.setattr(paper_trader, "get_connection", flaky_get_connection)
        monkeypatch.setattr(paper_trader.time, "sleep", lambda *_: None)  # não espera de verdade

        pos = paper_trader.open_position(
            {"current_cash": 1000.0}, _valid_open_signal(), dry_run=False,
            current_markets=pd.DataFrame(), open_positions=pd.DataFrame(), traded_ids=set(),
        )

        assert pos is not None  # segunda tentativa vingou
        assert state["n"] == 2


class TestOpenPositionRecebeSnapshotsDoChamador:
    """P1-19: por candidato a sinal, open_position relia o parquet inteiro
    (load_current_markets) e reconsultava o banco (get_open_positions,
    get_traded_condition_ids) — até 30 leituras de parquet por ciclo com
    top_signals=30, ilimitado no run_execution. Os três agora são opcionais:
    quando o chamador passa, open_position usa o que foi passado sem tocar
    disco/banco de novo."""

    def test_nao_chama_load_current_markets_quando_current_markets_passado(self, tmp_path, monkeypatch):
        db = tmp_path / "paper.db"
        monkeypatch.setattr(paper_trader, "DB_PATH", db)
        paper_trader.init_db()
        conn = sqlite3.connect(db)
        conn.execute("INSERT INTO portfolio (initial_capital, current_cash) VALUES (1000, 1000)")
        conn.commit()
        conn.close()

        def boom():
            raise AssertionError("load_current_markets não deveria ser chamado")
        monkeypatch.setattr(paper_trader, "load_current_markets", boom)

        pos = paper_trader.open_position(
            {"current_cash": 1000.0}, _valid_open_signal(), dry_run=True,
            current_markets=pd.DataFrame(), open_positions=pd.DataFrame(), traded_ids=set(),
        )
        assert pos is not None

    def test_nao_chama_get_open_positions_nem_get_traded_ids_quando_passados(self, tmp_path, monkeypatch):
        db = tmp_path / "paper.db"
        monkeypatch.setattr(paper_trader, "DB_PATH", db)
        paper_trader.init_db()
        conn = sqlite3.connect(db)
        conn.execute("INSERT INTO portfolio (initial_capital, current_cash) VALUES (1000, 1000)")
        conn.commit()
        conn.close()

        def boom(*a, **kw):
            raise AssertionError("não deveria reconsultar o banco — snapshot já foi passado")
        monkeypatch.setattr(paper_trader, "get_open_positions", boom)
        monkeypatch.setattr(paper_trader, "get_traded_condition_ids", boom)

        pos = paper_trader.open_position(
            {"current_cash": 1000.0}, _valid_open_signal(), dry_run=True,
            current_markets=pd.DataFrame(), open_positions=pd.DataFrame(), traded_ids=set(),
        )
        assert pos is not None

    def test_traded_ids_passado_ainda_bloqueia_duplicata(self, tmp_path, monkeypatch):
        db = tmp_path / "paper.db"
        monkeypatch.setattr(paper_trader, "DB_PATH", db)
        paper_trader.init_db()
        conn = sqlite3.connect(db)
        conn.execute("INSERT INTO portfolio (initial_capital, current_cash) VALUES (1000, 1000)")
        conn.commit()
        conn.close()

        pos = paper_trader.open_position(
            {"current_cash": 1000.0}, _valid_open_signal("0xjatem"), dry_run=True,
            current_markets=pd.DataFrame(), open_positions=pd.DataFrame(),
            traded_ids={"0xjatem"},
        )
        assert pos is None

    def test_linha_apendada_no_loop_do_chamador_tem_status_open(self):
        # Achado na revisão do advisor: pd.concat com um dict de 15 chaves
        # (o que open_position devolve) contra open_pos (SELECT * — 20+
        # colunas, incluindo status) sem "status" no dict apendado deixava
        # NaN em status pra essa linha — a mesma classe de bug do P1-17
        # (str(nan)=='nan' furando .get/comparação rio abaixo). Smoke test
        # de código-fonte: os dois pontos onde o loop apenda a posição
        # recém-aberta precisam fixar status="open" explicitamente.
        for path, marker in [
            (ROOT / "execution" / "paper_trader.py", '{**pos, "status": "open"'),
            (ROOT / "run_execution.py", '{**result, "status": "open"'),
        ]:
            assert marker in path.read_text(), f"{path.name} sem status fixo na linha apendada"


class TestRunCycleTimeoutELock:
    """P1-20: subprocess.run sem timeout= deixava uma chamada de rede
    pendurada travar o ciclo indefinidamente, e o StartInterval fixo do
    launchd empilhava ciclos sobrepostos em cima do subprocesso travado —
    o mesmo cenário contra o qual P1-18 protege no lado do banco. Teste de
    fumaça sobre o código-fonte (main() tem efeitos colaterais de processo
    — chdir, flock, subprocess — pesados demais pra rodar ponta a ponta)."""

    SRC = (ROOT / "run_cycle.py").read_text()

    def test_subprocess_run_tem_timeout(self):
        i_run = self.SRC.index("def run(cmd")
        i_next_def = self.SRC.index("\ndef ", i_run + 1)
        body = self.SRC[i_run:i_next_def]
        assert "timeout=timeout" in body
        assert "TimeoutExpired" in body

    def test_usa_flock_nao_bloqueante(self):
        assert "fcntl.flock" in self.SRC
        assert "LOCK_EX | fcntl.LOCK_NB" in self.SRC

    def test_fetch_markets_falho_nao_aborta_o_resto_do_ciclo(self):
        # P1-20 cogitou abortar o ciclo inteiro numa falha de fetch_markets.
        # Revertido: logs/cycle_light_error.log mostra ~27% de falha
        # histórica (Gamma API/DNS piscando por segundos), e P0-6/P0-2 já
        # fecham o buraco de dado sem precisar de abort — abortar custaria
        # 1/4 dos ciclos de trading pra pouco ganho. Fica soft, sinais/
        # paper_trader seguem rodando (e degradam pra no-op sozinhos se o
        # snapshot ficar velho demais).
        i_fetch = self.SRC.index('run(uv + ["pipeline/fetch_markets.py"')
        i_odds_step = self.SRC.index("# ── 2. Sinais odds")
        assert i_fetch < i_odds_step
        body_between = self.SRC[i_fetch:i_odds_step]
        assert "critical" not in body_between.lower()
        assert "return" not in body_between  # não sai da função nem pula passos

    def test_db_maintenance_pulado_em_dry_run(self):
        i_step7 = self.SRC.index("Retenção do price_history")
        body = self.SRC[i_step7:i_step7 + 400]
        assert "if dry_run:" in body

    def test_chdir_pro_diretorio_do_script(self):
        assert "os.chdir(Path(__file__).resolve().parent)" in self.SRC

    def test_fetch_markets_roda_sem_relatorio_eda(self):
        # P2-37: eda_markets_*.csv + top_markets_*.csv (--report, default do
        # fetch_markets.py) é pra exploração manual, não pro ciclo
        # automatizado — virou 3.106 arquivos / 15 GB sozinho em
        # outputs/reports sem isso.
        assert '"pipeline/fetch_markets.py", "--no-report"' in self.SRC

    def test_log_proprio_com_rotacao(self):
        i_add = self.SRC.index("logger.add(")
        body = self.SRC[i_add:i_add + 200]
        assert 'rotation="1 day"' in body
        assert 'retention="7 days"' in body
