# prediction-arb

Automated trading system for [Polymarket](https://polymarket.com) prediction markets. It hunts for mispricing against external references (bookmaker odds, crypto options) and for structural arbitrage inside Polymarket itself, then paper-trades the signals with strict risk limits.

Everything runs locally on a Mac via launchd: a light cycle every 30 minutes, a full cycle every 4 hours, and a WebSocket daemon watching open positions in real time.

**Status:** paper trading only. The portfolio was reset in July 2026 after a data bug (not a strategy failure — see [Lessons](#lessons)) and the system is now in a measurement phase: accumulating trades per signal source to find out which of them, if any, has real edge after costs.

## How it works

```
pipeline/    market data: Gamma API (keyset pagination), CLOB, Deribit,
             The Odds API, WebSocket price feed
signals/     signal generation — the three engines below
execution/   paper trader: Kelly sizing, atomic basket execution, MTM
risk/        risk_manager.py — single source of truth for every limit
backtest/    forward performance tracking per signal source
dashboard/   Streamlit monitoring
features/    feature engineering (mostly idle since the ML freeze)
ml_lab/      archived ML experiments (see below)
```

`run_cycle.py` orchestrates: fetch markets → generate signals → paper trade → snapshot prices → update performance metrics → health checks. Failures and silent degradation (shrinking market universe, stale signals, suspicious exits) are reported via Telegram.

## Signal engines

**Bookmaker odds.** Sharp bookmaker odds via The Odds API, vig removed with the power method, consensus across books, fuzzy-matched to Polymarket sports markets. Divergence between implied probability and market price is the edge.

**Deribit options.** Risk-neutral probabilities for BTC/ETH price targets from Deribit's IV surface — barrier (one-touch) formula for "will X reach $Y" markets, plain digital for fixed-date ones. Honest caveat: with a drift assumption this is closer to a model bet than pure arbitrage; whether it stays is a decision for the measurement phase.

**Structural arbitrage.** No external reference, no model — profit guaranteed by the logic of the markets themselves:

- *NegRisk baskets*: mutually exclusive multi-outcome events where Σask < 1 (buy every YES) or Σbid > 1 (buy every NO)
- *Monotonicity*: nested crypto markets priced inconsistently (touching $80k by December must be at least as likely as touching $90k by November)
- *Crossed books*: ask < bid, reported for manual inspection

Guaranteed opportunities trigger a Telegram alert and are executed as **atomic baskets** — every leg opens in a single transaction or none do, since a partially-filled basket is just a naked directional position. Basket legs are held to resolution; no early exit can touch them.

## Risk management

All limits live in `risk/risk_manager.py` and are imported everywhere else — never duplicated.

- Quarter-Kelly × confidence score, capped per trade type (1.5% momentum / 3% value of capital)
- Max 20 open directional positions, 30% per category, 60% per signal source
- Arb baskets sized separately (5% of capital, bounded by the thinnest leg's liquidity — Kelly is undefined for guaranteed edge)
- Stop loss per position (−50%), daily (−5%) and weekly (−10%) drawdown halts
- Early exits respect a minimum hold time, so an empty order book can't stop-out a position on a phantom price

## Running it

```bash
uv sync
cp .env.example .env        # fill in API keys (all optional except what you use)

uv run python pipeline/validate_connection.py   # sanity-check the APIs
uv run python run_cycle.py --dry-run            # full cycle, nothing persisted
uv run python run_cycle.py                      # light cycle for real
uv run python execution/run_paper_trader.py --status
uv run streamlit run dashboard/app.py           # http://localhost:8501

uv run pytest                                   # 81 tests
```

Positions and trades live in SQLite (`data/db/paper_trading.db`), market snapshots in Parquet (`data/raw/`), signal CSVs in `outputs/reports/`. All paths are relative to the repo root.

## Lessons

Things this project got wrong first and fixed later, kept here because they're the actual point:

- **Order books don't come sorted.** The CLOB WebSocket sends bid levels in ascending order; treating `bids[0]` as the best bid marked every position at dust prices and stop-lossed the entire portfolio at 0.001. Exits now require a sane spread and a minimum hold time.
- **Pagination can fail silently for months.** The Gamma API caps offset pagination at 100 results; the market universe quietly shrank from thousands to 100 and signal generation starved for two months before anyone noticed. The fix was keyset pagination plus health checks that alert on the *artifacts* (universe size, signal freshness), not on exit codes.
- **A great AUC is a smell, not a result.** The original ML models scored 0.98 AUC because features leaked the outcome. The ML lab is archived in `ml_lab/` and the project pivoted to arbitrage-style signals, where each trade is a falsifiable hypothesis.
- **Concurrent processes will double-credit cash** unless every exit path guards on position status inside an immediate transaction.

## License

MIT
