# Brent Crude Futures — Book PnL & Risk Dashboard

A Streamlit dashboard for a Brent crude futures book: mark-to-market P&L,
Value-at-Risk / Expected Shortfall, and aggregated options Greeks —
built on top of a FIFO position-keeping engine and a Black-76 pricing
engine.

The dashboard reads positions from `trades.xlsx` / `options_trades.xlsx`
(see **Adding / updating positions** below) — these ship pre-populated
with sample data so the dashboard isn't empty on first run, but they're
meant to be edited with your actual positions. Market data (settlement
prices) still comes from Yahoo Finance live, falling back to a bundled
CSV — see **Data source** below for that limitation.

## What it does

**Portfolio Analysis**
- FIFO realized P&L and current open positions, per contract
- Daily mark-to-market P&L (variation-margin style), by contract or total book
- On-request "live PnL" against a manually entered or simulated live price

**Risk Metrics**
- Historical VaR / Expected Shortfall (full revaluation, no distributional assumption)
- Parametric (delta-normal) VaR / Expected Shortfall
- Both at 95% and 99% confidence, toggleable 1-day / 10-day horizon

**Greeks**
- A full options book (calls/puts on the same futures contracts) priced with Black-76
- FIFO realized P&L on closed option trades
- Live Delta, Gamma, Vega, Theta, Rho — per position and aggregated by underlying
- A strike/expiry-dependent implied volatility surface (not flat vol) driving the Greeks
- Greeks-vs-strike charts that bake in the actual vol smile

## Project structure

```
dashboard.py                    Streamlit app — entry point, three tabs
pnl_engine.py                   FIFO positions, daily MTM, live PnL (futures)
risk_engine.py                  Historical + parametric VaR/ES (portfolio-agnostic)
greeks_engine.py                Black-76 pricing/Greeks, FIFO positions (options)
market_data.py                  Live settlement price fetch (Yahoo Finance) + CSV fallback
generate_dummy_data.py          Generates the four sandbox files below (safe to re-run any time)

trades.xlsx                     REAL futures trade blotter — the dashboard reads this. Edit it.
options_trades.xlsx             REAL options trade blotter — the dashboard reads this. Edit it.

trades_dummy.xlsx               Sandbox/test futures blotter (generate_dummy_data.py's output only)
settlement_prices.csv           Sandbox settlement prices + live-fetch fallback
options_trades_dummy.xlsx       Sandbox/test options blotter (generate_dummy_data.py's output only)
volatility_surface_dummy.csv    Sandbox implied volatility surface

requirements.txt                Python dependencies
```

## Adding / updating positions

The dashboard reads positions from two files, which you edit directly in
Excel:

- **`trades.xlsx`** (sheet `Trades`) — futures trades. Columns: `TradeID`,
  `TradeDate`, `Contract`, `BuySell` (exactly `BUY` or `SELL`), `Quantity`,
  `Price`, `Trader`, `LotSize`, `Currency`.
- **`options_trades.xlsx`** (sheet `OptionTrades`) — options trades.
  Columns: `TradeID`, `TradeDate`, `UnderlyingContract`, `OptionType`
  (`CALL`/`PUT`), `Strike`, `ExpiryDate`, `BuySell`, `Quantity`,
  `Premium`, `Trader`, `LotSize`, `Currency`.

**Workflow:** open the file in Excel, add a new row for the trade, save,
then commit and push the updated file to GitHub (`git add`, `git commit`,
`git push`, or drag-and-drop the updated file on github.com to replace
it). Streamlit Community Cloud auto-redeploys on every push, so the new
trade appears on the live dashboard shortly after.

**Why not edit these inline on github.com?** `.xlsx` is a binary format —
GitHub's web editor can only edit plain text files inline, so Excel files
have to be edited locally and pushed back. This is a deliberate tradeoff:
it keeps Excel's own validation/dropdowns for data entry, at the cost of
one extra local step per update.

**`BuySell` must be exactly `BUY` or `SELL`** (case-insensitive, no extra
spaces) — anything else is rejected with a clear error rather than
silently misinterpreted, by design (see `pnl_engine.py`/`greeks_engine.py`
for why this matters).

`trades_dummy.xlsx` / `options_trades_dummy.xlsx` are untouched by this
workflow — they're sandbox files `generate_dummy_data.py` regenerates for
testing, and the dashboard never reads them.

## Running locally

```bash
pip install -r requirements.txt
streamlit run dashboard.py
```

The dummy data files are already included in this repo, so no setup
beyond installing dependencies is needed. If you want to regenerate them
(e.g. different trades, a longer price history), run:

```bash
python generate_dummy_data.py
```

This overwrites all four dummy data files listed above.

## Data source

Settlement prices are fetched live from Yahoo Finance (`BZ=F`, the Brent
front-month continuous contract) on each run, and fall back automatically
to the bundled `settlement_prices.csv` if the live fetch fails for any
reason (no network, rate limit, etc.). A banner at the top of the
dashboard shows which source is actually in use.

**Known limitation:** Yahoo Finance doesn't expose per-expiry Brent
contract prices for free, so all contract months are currently priced off
the same single front-month series. This understates diversification in
the Risk Metrics tab's parametric VaR (contracts appear near-perfectly
correlated). A real deployment would replace this with a licensed
per-contract feed (ICE Data, Bloomberg, Refinitiv) — see the comments at
the top of `market_data.py`.

## Deploying

Pushed to a public GitHub repo, this deploys directly on
[Streamlit Community Cloud](https://streamlit.io/cloud) (free): connect
your GitHub account, point it at this repo and `dashboard.py`, and it
builds automatically. See Streamlit's own
[deployment docs](https://docs.streamlit.io/deploy) for current limits
and options.

## Disclaimer

`trades.xlsx` / `options_trades.xlsx` ship with sample data but are meant
to hold real positions if you choose to edit them. That said, the engines
behind this dashboard (FIFO matching, VaR/ES, Black-76 Greeks) are
portfolio/demo code — not independently audited, not connected to any
licensed market data feed for per-contract pricing (see **Data source**
above), and not intended as the sole basis for real trading or risk
decisions.
