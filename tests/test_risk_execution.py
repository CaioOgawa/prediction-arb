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
    # trade_type é coluna de migração (não está no SCHEMA base)
    conn.execute("ALTER TABLE positions ADD COLUMN trade_type TEXT DEFAULT 'value'")
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


# ──────────────────────────────────────────────────────────
# structural_arb — detectores puros (Fase 2)
# ──────────────────────────────────────────────────────────

sys.path.insert(0, str(ROOT / "signals"))
import structural_arb


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
        # Mesmo nº de shares nas duas pernas (a matemática do arb exige)
        assert pos["shares"].nunique() == 1
        # Débito único = soma dos custos das pernas
        assert abs((1000 - _cash(basket_db)) - result["total_cost"]) < 0.02

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
