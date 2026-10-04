"""
generate_dummy_data.py
======================
Builds ONE internally consistent dummy dataset for checking the maths of
the dashboard (P&L, VaR, Black-76 Greeks) -- not for trading.

Everything is derived from a single simulated price path, so the files
can't contradict each other the way the previous set did (Yahoo prices
in the settlement CSV, an old random walk in the vol surface, hand-typed
option premiums below intrinsic value):

  settlement_prices_dummy.csv   one settle per contract month per business day
  volatility_surface_dummy.csv  per contract month, centred on THAT day's settle
  trades_dummy.xlsx             futures trades priced within ~$0.15 of settle
  options_trades_dummy.xlsx     premiums = Black-76 off the surface on trade date

Output files all end in _dummy, so nothing here can overwrite the real
blotters, and market_data.py's live fetch never touches them.

Run:  python generate_dummy_data.py           (deterministic, seed below)
"""

import numpy as np
import pandas as pd

from greeks_engine import black76_price, RISK_FREE_RATE_DEFAULT

SEED = 42
HISTORY_START = "2026-03-02"     # ~5 months of history so VaR has >100 observations
AS_OF = "2026-07-31"             # last settlement date in the dummy world
LOT_SIZE = 1000

# Contract months: chosen so all are still trading on AS_OF.
# ICE Brent futures expire on the last business day of month M-2; options
# expire 3 business days before their future.
CONTRACTS = {
    #  name          futures expiry   spread to front   base ATM vol
    "BRN Oct26": dict(fut_expiry="2026-08-31", spread=0.00, atm=0.31),
    "BRN Nov26": dict(fut_expiry="2026-09-30", spread=-0.45, atm=0.30),
    "BRN Dec26": dict(fut_expiry="2026-10-30", spread=-0.85, atm=0.29),
}

START_PRICE = 80.0
PRICE_VOL = 0.28                 # annualised, for the common factor
STRIKE_MONEYNESS = np.round(np.arange(0.75, 1.2501, 0.05), 2)   # 11 strikes, ±25%

# Smile in log-moneyness x = ln(K/F):  sigma = ATM + SKEW*x + CURVE*x^2
# -> about +3.6 vol pts at 80% strike, -1 pt at 120%: a typical oil put skew.
SKEW, CURVE = -0.10, 0.25


def option_expiry(fut_expiry: str) -> pd.Timestamp:
    return pd.Timestamp(fut_expiry) - pd.offsets.BDay(3)


def simulate_settlements(rng) -> pd.DataFrame:
    dates = pd.bdate_range(HISTORY_START, AS_OF)
    n = len(dates)
    dt = 1 / 252
    shocks = rng.normal(0, PRICE_VOL * np.sqrt(dt), n)
    shocks[0] = 0.0
    front = START_PRICE * np.exp(np.cumsum(shocks - 0.5 * PRICE_VOL ** 2 * dt))

    rows = []
    for name, c in CONTRACTS.items():
        # Small mean-reverting idiosyncratic spread so the months aren't
        # perfectly correlated (VaR diversification becomes visible).
        noise = np.zeros(n)
        for i in range(1, n):
            noise[i] = 0.9 * noise[i - 1] + rng.normal(0, 0.06)
        px = front + c["spread"] + noise
        rows += [{"Date": d, "Contract": name, "SettlePrice": round(float(p), 2), "Source": "dummy"}
                 for d, p in zip(dates, px)]
    return pd.DataFrame(rows)


def build_vol_surface(settles: pd.DataFrame, rng) -> pd.DataFrame:
    dates = sorted(settles["Date"].unique())
    # Common slow-moving vol factor (AR(1)), so ATM vol drifts realistically
    # instead of jumping 3-4 points a day as in the old file.
    vf = np.zeros(len(dates))
    for i in range(1, len(dates)):
        vf[i] = 0.97 * vf[i - 1] + rng.normal(0, 0.003)
    vol_factor = dict(zip(dates, vf))

    rows = []
    for row in settles.itertuples():
        c = CONTRACTS[row.Contract]
        expiry = option_expiry(c["fut_expiry"])
        if row.Date >= expiry:
            continue
        atm = np.clip(c["atm"] + vol_factor[row.Date], 0.15, 0.60)
        F = row.SettlePrice
        for m in STRIKE_MONEYNESS:
            K = round(F * m, 2)
            x = np.log(K / F)
            rows.append({"Date": row.Date, "UnderlyingContract": row.Contract, "ExpiryDate": expiry,
                         "Strike": K, "Moneyness": round(K / F, 6),
                         "ImpliedVol": round(float(atm + SKEW * x + CURVE * x * x), 6)})
    return pd.DataFrame(rows)


# Futures blotter design: (date, contract, side, qty). Prices come from the path.
FUTURES_PLAN = [
    ("2026-07-01", "BRN Oct26", "BUY", 10), ("2026-07-03", "BRN Oct26", "BUY", 5),
    ("2026-07-07", "BRN Nov26", "SELL", 8), ("2026-07-09", "BRN Oct26", "SELL", 6),
    ("2026-07-13", "BRN Dec26", "BUY", 12), ("2026-07-15", "BRN Nov26", "BUY", 3),
    ("2026-07-17", "BRN Oct26", "BUY", 4), ("2026-07-21", "BRN Dec26", "SELL", 5),
    ("2026-07-23", "BRN Nov26", "SELL", 2), ("2026-07-27", "BRN Oct26", "SELL", 7),
    ("2026-07-29", "BRN Dec26", "BUY", 6), ("2026-07-31", "BRN Nov26", "BUY", 4),
]

# Options blotter design: (date, contract, type, target moneyness OR ref to an
# earlier trade to close against, side, qty). Includes partial and full closes
# so FIFO realized option P&L gets exercised.
OPTIONS_PLAN = [
    ("2026-07-02", "BRN Oct26", "CALL", 1.05, "BUY", 10),
    ("2026-07-06", "BRN Oct26", "PUT", 0.95, "BUY", 8),
    ("2026-07-08", "BRN Nov26", "CALL", 1.00, "SELL", 6),
    ("2026-07-10", "BRN Dec26", "PUT", 0.90, "BUY", 10),
    ("2026-07-14", "BRN Oct26", "CALL", "O001", "SELL", 4),   # partial close of O001
    ("2026-07-16", "BRN Nov26", "PUT", 0.95, "BUY", 5),
    ("2026-07-20", "BRN Dec26", "CALL", 1.10, "SELL", 8),
    ("2026-07-22", "BRN Oct26", "PUT", "O002", "SELL", 8),    # full close of O002
    ("2026-07-24", "BRN Nov26", "CALL", "O003", "BUY", 2),    # partial close of short O003
    ("2026-07-27", "BRN Dec26", "PUT", 0.95, "BUY", 6),
    ("2026-07-28", "BRN Dec26", "CALL", 1.00, "BUY", 4),
    ("2026-07-30", "BRN Nov26", "PUT", 1.05, "SELL", 3),
]


def surface_vol(surface, contract, date, strike):
    s = surface[(surface["UnderlyingContract"] == contract) & (surface["Date"] == date)].sort_values("Strike")
    return float(np.interp(strike, s["Strike"], s["ImpliedVol"]))


def build_futures_trades(settles, rng) -> pd.DataFrame:
    px = settles.set_index(["Date", "Contract"])["SettlePrice"]
    rows = []
    for i, (d, c, side, q) in enumerate(FUTURES_PLAN, 1):
        d = pd.Timestamp(d)
        price = round(px[(d, c)] + rng.normal(0, 0.12), 2)
        rows.append({"TradeID": f"T{i:03d}", "TradeDate": d, "Contract": c, "BuySell": side,
                     "Quantity": q, "Price": price, "Trader": "Dummy", "LotSize": LOT_SIZE, "Currency": "USD"})
    return pd.DataFrame(rows)


def build_option_trades(settles, surface, rng) -> pd.DataFrame:
    px = settles.set_index(["Date", "Contract"])["SettlePrice"]
    r = RISK_FREE_RATE_DEFAULT
    rows, by_id = [], {}
    for i, (d, c, typ, m, side, q) in enumerate(OPTIONS_PLAN, 1):
        d = pd.Timestamp(d)
        F = px[(d, c)]
        expiry = option_expiry(CONTRACTS[c]["fut_expiry"])
        strike = by_id[m]["Strike"] if isinstance(m, str) else round(F * m * 2) / 2   # nearest 0.50
        T = (expiry - d).days / 365.0
        sigma = surface_vol(surface, c, d, strike)
        theo = black76_price(F, strike, T, r, sigma, typ)
        # Tiny execution noise (±0.5%), floored at discounted intrinsic value.
        intrinsic = np.exp(-r * T) * max((F - strike) if typ == "CALL" else (strike - F), 0.0)
        premium = round(max(theo * (1 + rng.normal(0, 0.005)), intrinsic), 2)
        row = {"TradeID": f"O{i:03d}", "TradeDate": d, "UnderlyingContract": c, "OptionType": typ,
               "Strike": strike, "ExpiryDate": expiry, "BuySell": side, "Quantity": q,
               "Premium": premium, "Trader": "Dummy", "LotSize": LOT_SIZE, "Currency": "USD"}
        rows.append(row)
        by_id[row["TradeID"]] = row
    return pd.DataFrame(rows)


def main():
    rng = np.random.default_rng(SEED)
    settles = simulate_settlements(rng)
    surface = build_vol_surface(settles, rng)
    fut = build_futures_trades(settles, rng)
    opt = build_option_trades(settles, surface, rng)

    settles.to_csv("settlement_prices_dummy.csv", index=False)
    surface.to_csv("volatility_surface_dummy.csv", index=False)
    with pd.ExcelWriter("trades_dummy.xlsx") as xw:
        fut.to_excel(xw, sheet_name="Trades", index=False)
    with pd.ExcelWriter("options_trades_dummy.xlsx") as xw:
        opt.to_excel(xw, sheet_name="OptionTrades", index=False)

    print(f"settlement_prices_dummy.csv  {len(settles)} rows, {settles['Date'].min().date()} -> {settles['Date'].max().date()}")
    print(f"volatility_surface_dummy.csv {len(surface)} rows")
    print(f"trades_dummy.xlsx            {len(fut)} trades")
    print(f"options_trades_dummy.xlsx    {len(opt)} trades")


if __name__ == "__main__":
    main()
