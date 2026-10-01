# PnL---Dashboard
# Brent Crude Futures — Book PnL & Risk Dashboard

A Streamlit dashboard for a Brent crude futures book: mark-to-market P&L,
Value-at-Risk / Expected Shortfall, and aggregated options Greeks —
built on top of a FIFO position-keeping engine and a Black-76 pricing
engine.

This is demo/portfolio code using **dummy trade and market data** — not
connected to any real trading system or market data vendor.

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
generate_dummy_data.py          Generates all dummy data files below (run once, optional)

trades_dummy.xlsx               Dummy futures trade blotter
settlement_prices.csv           Dummy daily settlement prices
options_trades_dummy.xlsx       Dummy options trade blotter
volatility_surface_dummy.csv    Dummy implied volatility surface

requirements.txt                Python dependencies
```

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

Dummy data, for demonstration purposes only. Not connected to any real
trading, risk, or market data system, and not intended to inform real
trading or risk decisions.
