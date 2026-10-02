"""
generate_dummy_data.py
=======================
SANDBOX / TEST DATA ONLY. Creates four dummy inputs for testing the
engines and self-tests:

1. trades_dummy.xlsx            - a dummy futures trade blotter
2. settlement_prices.csv        - dummy daily settlement prices
3. options_trades_dummy.xlsx    - a dummy options trade blotter
4. volatility_surface_dummy.csv - a dummy implied vol surface

IMPORTANT: this script only ever writes to the four _dummy-suffixed (or,
for settlement_prices.csv, market-data-fed) files above. It NEVER touches
trades.xlsx or options_trades.xlsx -- those are the REAL, user-maintained
blotters the dashboard actually reads (see README.md). Running this
script to regenerate test data is always safe and will never overwrite
your real positions.

In real life trades.xlsx/options_trades.xlsx would come from your
OMS/back-office system, and settlement_prices.csv from your market data
vendor. Everything downstream (pnl_engine.py, greeks_engine.py) only
cares about the shape of these files, so you can swap any of them for a
real data source later without touching the engine logic.
"""

import numpy as np
import pandas as pd

np.random.seed(42)  # reproducible "randomness" for a dummy dataset

LOT_SIZE = 1000  # barrels per ICE Brent futures lot

# ---------------------------------------------------------------------
# 1. Trade blotter
# ---------------------------------------------------------------------
# A handful of trades across three Brent futures contract months, spread
# over the last few weeks, some of which partially offset each other so
# the PnL engine has realized PnL to compute as well as open positions.
trades = [
    # TradeID, TradeDate,   Contract,    BuySell, Quantity(lots), Price(usd/bbl), Trader
    ("T001", "2026-07-06", "BRN Sep26", "BUY",  10, 82.15, "Rayane"),
    ("T002", "2026-07-08", "BRN Sep26", "BUY",   5, 82.60, "Rayane"),
    ("T003", "2026-07-10", "BRN Oct26", "SELL",  8, 81.90, "Rayane"),
    ("T004", "2026-07-13", "BRN Sep26", "SELL",  6, 83.05, "Rayane"),   # partial close -> realized PnL
    ("T005", "2026-07-15", "BRN Dec26", "BUY",  12, 80.75, "Rayane"),
    ("T006", "2026-07-17", "BRN Oct26", "BUY",   3, 82.20, "Rayane"),   # partial close -> realized PnL
    ("T007", "2026-07-20", "BRN Sep26", "BUY",   4, 83.40, "Rayane"),
    ("T008", "2026-07-22", "BRN Dec26", "SELL",  5, 81.50, "Rayane"),   # partial close -> realized PnL
    ("T009", "2026-07-24", "BRN Oct26", "SELL",  2, 82.95, "Rayane"),
    ("T010", "2026-07-27", "BRN Sep26", "SELL",  7, 84.10, "Rayane"),  # partial close -> realized PnL
    ("T011", "2026-07-29", "BRN Dec26", "BUY",   6, 81.10, "Rayane"),
    ("T012", "2026-07-31", "BRN Oct26", "BUY",   4, 83.30, "Rayane"),
]

trades_df = pd.DataFrame(
    trades,
    columns=["TradeID", "TradeDate", "Contract", "BuySell", "Quantity", "Price", "Trader"],
)
trades_df["TradeDate"] = pd.to_datetime(trades_df["TradeDate"])
trades_df["LotSize"] = LOT_SIZE
trades_df["Currency"] = "USD"

trades_df.to_excel("trades_dummy.xlsx", sheet_name="Trades", index=False)
print(f"Wrote trades_dummy.xlsx with {len(trades_df)} trades")

# ---------------------------------------------------------------------
# 2. Simulated daily settlement prices
# ---------------------------------------------------------------------
# One random-walk price series per contract, running from that contract's
# first trade date up to "today". This stands in for a settlement price
# feed from ICE / your market data vendor.
today = pd.Timestamp("2026-08-01")
rows = []
for contract, start_price in [("BRN Sep26", 82.15), ("BRN Oct26", 81.90), ("BRN Dec26", 80.75)]:
    first_trade_date = trades_df.loc[trades_df.Contract == contract, "TradeDate"].min()
    business_days = pd.bdate_range(first_trade_date, today)
    # small daily moves, ~1% daily vol, mean-reverting slightly around start_price
    shocks = np.random.normal(loc=0.0, scale=0.55, size=len(business_days))
    prices = start_price + np.cumsum(shocks)
    prices = np.round(prices, 2)
    for d, p in zip(business_days, prices):
        rows.append((d, contract, p))

prices_df = pd.DataFrame(rows, columns=["Date", "Contract", "SettlePrice"])
prices_df.to_csv("settlement_prices.csv", index=False)
print(f"Wrote settlement_prices.csv with {len(prices_df)} rows")

# ---------------------------------------------------------------------
# 3. Options trade blotter
# ---------------------------------------------------------------------
# European-style options on the same three Brent futures contracts, priced
# off the futures settlement as the forward (Black-76). Mix of calls/puts,
# some partially closed (so the FIFO engine has realized option PnL to
# compute, same pattern as the futures book), rest left open past "today"
# so greeks_engine.py has live, un-expired positions to revalue.
option_trades = [
    # TradeID, TradeDate,   Underlying,   Type,  Strike, ExpiryDate,   BuySell, Qty, Premium, Trader
    ("O001", "2026-07-07", "BRN Sep26", "CALL", 84.00, "2026-09-15", "BUY",  10, 1.85, "Rayane"),
    ("O002", "2026-07-09", "BRN Sep26", "PUT",  80.00, "2026-09-15", "BUY",   8, 1.40, "Rayane"),
    ("O003", "2026-07-11", "BRN Oct26", "CALL", 83.00, "2026-09-30", "SELL",  6, 1.55, "Rayane"),
    ("O004", "2026-07-14", "BRN Sep26", "CALL", 84.00, "2026-09-15", "SELL",  4, 2.10, "Rayane"),  # partial close of O001
    ("O005", "2026-07-16", "BRN Dec26", "PUT",  79.00, "2026-10-15", "BUY",  10, 1.70, "Rayane"),
    ("O006", "2026-07-18", "BRN Oct26", "CALL", 83.00, "2026-09-30", "BUY",   3, 1.75, "Rayane"),  # partial close of O003
    ("O007", "2026-07-21", "BRN Sep26", "PUT",  80.00, "2026-09-15", "SELL",  5, 1.95, "Rayane"),  # partial close of O002
    ("O008", "2026-07-23", "BRN Dec26", "CALL", 82.00, "2026-10-15", "BUY",   6, 1.60, "Rayane"),
    ("O009", "2026-07-27", "BRN Dec26", "PUT",  79.00, "2026-10-15", "SELL",  4, 1.55, "Rayane"),  # partial close of O005
    ("O010", "2026-07-30", "BRN Oct26", "PUT",  80.00, "2026-09-30", "BUY",   7, 1.30, "Rayane"),
]

option_trades_df = pd.DataFrame(
    option_trades,
    columns=["TradeID", "TradeDate", "UnderlyingContract", "OptionType", "Strike",
             "ExpiryDate", "BuySell", "Quantity", "Premium", "Trader"],
)
option_trades_df["TradeDate"] = pd.to_datetime(option_trades_df["TradeDate"])
option_trades_df["ExpiryDate"] = pd.to_datetime(option_trades_df["ExpiryDate"])
option_trades_df["LotSize"] = LOT_SIZE
option_trades_df["Currency"] = "USD"

option_trades_df.to_excel("options_trades_dummy.xlsx", sheet_name="OptionTrades", index=False)
print(f"Wrote options_trades_dummy.xlsx with {len(option_trades_df)} option trades")

# ---------------------------------------------------------------------
# 4. Volatility surface (strike x expiry x date, per underlying)
# ---------------------------------------------------------------------
# One expiry per underlying in this dummy book (realistic enough for a
# demo; greeks_engine.py's lookup function supports more per underlying
# if the book grows). Strikes are generated as a grid AROUND THAT DAY'S
# ATM (the underlying's own settle price), so the smile recenters as the
# forward moves -- exactly like a real vol surface is quoted relative to
# moneyness, not to a fixed price level. A traded option's fixed strike
# is then interpolated against that day's grid in greeks_engine.py.
vol_rows = []
underlying_expiry = {
    "BRN Sep26": pd.Timestamp("2026-09-15"),
    "BRN Oct26": pd.Timestamp("2026-09-30"),
    "BRN Dec26": pd.Timestamp("2026-10-15"),
}
strike_offsets = [-6, -4, -2, 0, 2, 4, 6]  # $/bbl around that day's ATM
base_vol = 0.30     # 30% flat level
skew = -0.015        # puts slightly bid vs calls (typical commodity skew direction varies; illustrative only)
curvature = 0.08     # smile convexity

for contract, expiry in underlying_expiry.items():
    cprices = prices_df[prices_df["Contract"] == contract]
    for _, row in cprices.iterrows():
        f = row["SettlePrice"]
        noise = np.random.normal(0, 0.01)  # small daily vol-of-vol noise
        for offset in strike_offsets:
            k = round(f + offset, 2)
            moneyness = (k - f) / f
            implied_vol = base_vol + skew * moneyness + curvature * moneyness ** 2 + noise
            implied_vol = max(implied_vol, 0.05)  # floor so noise can't produce a nonsensical negative vol
            vol_rows.append({
                "Date": row["Date"], "UnderlyingContract": contract,
                "ExpiryDate": expiry, "Strike": k, "ImpliedVol": round(implied_vol, 4),
            })

vol_surface_df = pd.DataFrame(vol_rows)
vol_surface_df.to_csv("volatility_surface_dummy.csv", index=False)
print(f"Wrote volatility_surface_dummy.csv with {len(vol_surface_df)} rows")
