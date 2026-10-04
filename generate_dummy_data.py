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
  option_chains_dummy.csv       end-of-day SETTLEMENT chains for Brent (BRN) and
                                WTI (CL) futures options -- calls and puts, every
                                strike, per contract month -- shaped like what a
                                real exchange / vendor feed would deliver. In dummy
                                mode the dashboard backs the implied vol surface out
                                of these prices (Black-76), exactly as it would with
                                real Brent/WTI option settlements.

volatility_surface_dummy.csv is the "true" input smile used to price the
chains and the option trades; test_math.py checks that calibrating the
chains recovers it.

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
    """ICE Brent options: 3 business days before the future expires."""
    return pd.Timestamp(fut_expiry) - pd.offsets.BDay(3)


# --------------------------------------------------------------------------
# WTI (CL) -- only used for the dummy option chains
# --------------------------------------------------------------------------
# CME WTI futures stop trading 3 business days before the 25th calendar day
# of the month before the contract month (4 if the 25th isn't a business
# day); WTI (LO) options expire 1 business day before the future.
def wti_futures_expiry(contract_month: str) -> pd.Timestamp:
    first = pd.Timestamp(contract_month + "-01")
    d25 = (first - pd.DateOffset(months=1)).replace(day=25)
    if d25.weekday() >= 5:
        d25 = d25 - pd.offsets.BDay(1)
    return d25 - pd.offsets.BDay(3)


WTI_CONTRACTS = {
    #  name        contract month   spread to WTI front   base ATM vol
    "CL Oct26": dict(month="2026-10", spread=0.00, atm=0.34),
    "CL Nov26": dict(month="2026-11", spread=-0.40, atm=0.33),
    "CL Dec26": dict(month="2026-12", spread=-0.75, atm=0.32),
}
WTI_DISCOUNT_TO_BRENT = -3.60      # WTI front trades ~$3.6 under Brent
WTI_SKEW, WTI_CURVE = -0.13, 0.30  # slightly steeper put skew than Brent

CHAIN_START = "2026-06-01"         # chains from a month before the first trade
CHAIN_STRIKE_RANGE = (0.75, 1.25)  # listed strikes within ±25% of that day's futures settle
CHAIN_STRIKE_STEP = 1.00           # $1 strike increments
MIN_SETTLE = 0.01                  # exchange tick; options settling below this aren't listed


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


def simulate_wti_settlements(brent_settles: pd.DataFrame, rng) -> pd.DataFrame:
    """WTI follows the Brent front month (shared factor) plus its own noise."""
    front = brent_settles[brent_settles["Contract"] == "BRN Oct26"].set_index("Date")["SettlePrice"]
    n = len(front)
    rows = []
    basis = np.zeros(n)
    for i in range(1, n):                       # Brent-WTI spread wanders a little
        basis[i] = 0.95 * basis[i - 1] + rng.normal(0, 0.10)
    for name, c in WTI_CONTRACTS.items():
        noise = np.zeros(n)
        for i in range(1, n):
            noise[i] = 0.9 * noise[i - 1] + rng.normal(0, 0.06)
        px = front.values + WTI_DISCOUNT_TO_BRENT + basis + c["spread"] + noise
        rows += [{"Date": d, "Contract": name, "SettlePrice": round(float(p), 2)} for d, p in zip(front.index, px)]
    return pd.DataFrame(rows)


def _chain_rows(date, commodity, contract, F, expiry, atm, skew, curve, rng):
    T = (expiry - date).days / 365.0
    r = RISK_FREE_RATE_DEFAULT
    lo, hi = CHAIN_STRIKE_RANGE
    strikes = np.arange(np.ceil(F * lo), np.floor(F * hi) + 1e-9, CHAIN_STRIKE_STEP)
    out = []
    for K in strikes:
        x = np.log(K / F)
        vol = atm + skew * x + curve * x * x
        dist = abs(x)
        for typ in ("CALL", "PUT"):
            px = round(black76_price(F, K, T, r, vol, typ), 2)
            if px < MIN_SETTLE:
                continue
            oi = int(4000 * np.exp(-dist * 12) * rng.uniform(0.6, 1.4)) + 5
            out.append({"Date": date, "Commodity": commodity, "UnderlyingContract": contract,
                        "FuturesSettle": F, "ExpiryDate": expiry, "OptionType": typ, "Strike": float(K),
                        "SettlePrice": px, "Volume": int(oi * rng.uniform(0.0, 0.15)), "OpenInterest": oi})
    return out


def build_option_chains(settles: pd.DataFrame, surface: pd.DataFrame, wti: pd.DataFrame, rng) -> pd.DataFrame:
    """
    End-of-day settlement chains. Brent uses the SAME true smile as
    volatility_surface_dummy.csv (ATM vol read from it, same SKEW/CURVE), so
    the options book, the surface and the chains all agree.
    """
    atm_brent = (surface[np.isclose(surface["Moneyness"], 1.0, atol=1e-4)]
                 .set_index(["Date", "UnderlyingContract"])["ImpliedVol"])
    rows = []
    start = pd.Timestamp(CHAIN_START)
    for s in settles[settles["Date"] >= start].itertuples():
        expiry = option_expiry(CONTRACTS[s.Contract]["fut_expiry"])
        if s.Date >= expiry or (s.Date, s.Contract) not in atm_brent.index:
            continue
        rows += _chain_rows(s.Date, "BRENT", s.Contract, s.SettlePrice, expiry,
                            atm_brent[(s.Date, s.Contract)], SKEW, CURVE, rng)

    # WTI ATM vol: its own slow-moving factor
    dates = sorted(wti["Date"].unique())
    vf = np.zeros(len(dates))
    for i in range(1, len(dates)):
        vf[i] = 0.97 * vf[i - 1] + rng.normal(0, 0.003)
    wti_vf = dict(zip(dates, vf))
    for w in wti[wti["Date"] >= start].itertuples():
        c = WTI_CONTRACTS[w.Contract]
        expiry = wti_futures_expiry(c["month"]) - pd.offsets.BDay(1)
        if w.Date >= expiry:
            continue
        atm = float(np.clip(c["atm"] + wti_vf[w.Date], 0.15, 0.60))
        rows += _chain_rows(w.Date, "WTI", w.Contract, w.SettlePrice, expiry, atm, WTI_SKEW, WTI_CURVE, rng)
    return pd.DataFrame(rows)


def main():
    rng = np.random.default_rng(SEED)
    settles = simulate_settlements(rng)
    surface = build_vol_surface(settles, rng)
    fut = build_futures_trades(settles, rng)
    opt = build_option_trades(settles, surface, rng)

    # Option chains use their OWN random stream, so adding them leaves every
    # file above byte-for-byte identical to before.
    rng_chain = np.random.default_rng(SEED + 1)
    wti = simulate_wti_settlements(settles, rng_chain)
    chains = build_option_chains(settles, surface, wti, rng_chain)

    settles.to_csv("settlement_prices_dummy.csv", index=False)
    surface.to_csv("volatility_surface_dummy.csv", index=False)
    with pd.ExcelWriter("trades_dummy.xlsx") as xw:
        fut.to_excel(xw, sheet_name="Trades", index=False)
    with pd.ExcelWriter("options_trades_dummy.xlsx") as xw:
        opt.to_excel(xw, sheet_name="OptionTrades", index=False)
    chains.to_csv("option_chains_dummy.csv", index=False)

    print(f"settlement_prices_dummy.csv  {len(settles)} rows, {settles['Date'].min().date()} -> {settles['Date'].max().date()}")
    print(f"volatility_surface_dummy.csv {len(surface)} rows")
    print(f"trades_dummy.xlsx            {len(fut)} trades")
    print(f"options_trades_dummy.xlsx    {len(opt)} trades")
    print(f"option_chains_dummy.csv      {len(chains)} rows "
          f"({chains['UnderlyingContract'].nunique()} contracts: {', '.join(sorted(chains['UnderlyingContract'].unique()))}; "
          f"{chains['Date'].min().date()} -> {chains['Date'].max().date()})")


if __name__ == "__main__":
    main()
