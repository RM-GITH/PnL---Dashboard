"""
greeks_engine.py
=================
Options engine for the Brent book: Black-76 pricing/Greeks, FIFO realized
P&L on closed option trades, and live aggregated Greeks on the currently
open option positions, using a strike/expiry-dependent volatility surface.

Why Black-76, not Black-Scholes
-------------------------------
Every option in this book is written on a FUTURES contract (BRN Sep26,
Oct26, Dec26), not on spot crude. Black-76 takes the forward price F
directly (here: that contract's settlement price) rather than a spot
price S and a cost-of-carry adjustment -- the natural model for exchange-
traded commodity futures options, and what a real Brent options desk
would actually use. Options-Black76.py (Phase 1 script) supplied the
pricing math this module is built from, just as library functions
instead of an interactive CLI.

Reusing the futures FIFO engine for options
--------------------------------------------
An option "contract" that can net against itself is the tuple
(UnderlyingContract, OptionType, Strike, ExpiryDate) -- a BUY and a SELL
only close each other if ALL FOUR match (you can't close a Sep26 84 call
with a Sep26 85 call). pnl_engine.compute_fifo_positions() already
implements exactly this kind of single-key FIFO matching; it just assumes
the key column is called "Contract". Rather than duplicate that matching
logic, this module collapses the 4-part option key into one composite
string and hands the trades to pnl_engine's existing, already-tested FIFO
function -- same pattern as risk_engine.py calling into pnl_engine for
positions, just reusing a different function from it.

Volatility
----------
Uses a strike/expiry-dependent implied vol surface (see
volatility_surface_dummy.csv / generate_dummy_data.py), not a single flat
number -- so the Greeks below pick up a real smile/skew shape rather than
assuming Black-76's flat-vol idealization. Lookup matches the nearest
surface date to the requested as-of date, and the nearest ExpiryDate on
the surface for that underlying, then interpolates across strike.

Known simplification: the dummy book is constructed so every still-open
lot has an ExpiryDate after the data's last settlement date -- i.e.
nothing in the sample book has quietly expired without a closing trade.
Production code should additionally handle expiry settlement (intrinsic
value payout) for any open lot whose ExpiryDate has passed.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm

RISK_FREE_RATE_DEFAULT = 0.04  # flat 4%; swap for a real curve in production


# ==========================================================================
# 1. Black-76 pricing & Greeks (pure functions, no I/O)
# ==========================================================================

def black76_price(F: float, K: float, T: float, r: float, sigma: float, option_type: str = "call") -> float:
    """Black-76 price of a European option on a futures/forward F."""
    option_type = option_type.lower()
    if T <= 0:
        intrinsic = max(F - K, 0.0) if option_type == "call" else max(K - F, 0.0)
        return intrinsic
    if sigma <= 0:
        intrinsic = max(F - K, 0.0) if option_type == "call" else max(K - F, 0.0)
        return np.exp(-r * T) * intrinsic

    d1 = (np.log(F / K) + 0.5 * sigma ** 2 * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "call":
        return np.exp(-r * T) * (F * norm.cdf(d1) - K * norm.cdf(d2))
    else:
        return np.exp(-r * T) * (K * norm.cdf(-d2) - F * norm.cdf(-d1))


def black76_greeks(F: float, K: float, T: float, r: float, sigma: float, option_type: str = "call") -> dict:
    """
    Black-76 Greeks, PER UNIT (one barrel / one unit of F), not scaled by
    lot size or position quantity -- that scaling happens one layer up in
    compute_live_option_greeks(), same separation pnl_engine.py keeps
    between per-unit price math and position-level aggregation.

    Theta is returned PER DAY (divided by 365) since that's the usual way
    desks read it; Vega is returned per 1.00 (100 vol points) change in
    sigma -- divide by 100 yourself if you want "per vol point".
    """
    option_type = option_type.lower()
    if T <= 0 or sigma <= 0:
        # Expired or degenerate -- no time value, Greeks collapse to the
        # intrinsic-value limit. Only Delta is non-trivial in that limit.
        if option_type == "call":
            delta = 1.0 if F > K else 0.0
        else:
            delta = -1.0 if F < K else 0.0
        return {"Delta": delta, "Gamma": 0.0, "Vega": 0.0, "Theta": 0.0, "Rho": 0.0}

    d1 = (np.log(F / K) + 0.5 * sigma ** 2 * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    discount = np.exp(-r * T)

    if option_type == "call":
        delta = discount * norm.cdf(d1)
    else:
        delta = discount * (norm.cdf(d1) - 1)

    gamma = discount * norm.pdf(d1) / (F * sigma * np.sqrt(T))
    vega = discount * F * np.sqrt(T) * norm.pdf(d1)

    if option_type == "call":
        theta_annual = (-discount * F * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
                         + r * discount * (F * norm.cdf(d1) - K * norm.cdf(d2)))
        rho = -T * discount * (F * norm.cdf(d1) - K * norm.cdf(d2))
    else:
        theta_annual = (-discount * F * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
                         + r * discount * (K * norm.cdf(-d2) - F * norm.cdf(-d1)))
        rho = -T * discount * (K * norm.cdf(-d2) - F * norm.cdf(-d1))

    return {"Delta": delta, "Gamma": gamma, "Vega": vega, "Theta": theta_annual / 365.0, "Rho": rho}


# ==========================================================================
# 2. Loading data
# ==========================================================================

def load_option_trades(path: str) -> pd.DataFrame:
    """Read the options blotter and normalize dtypes/sign convention."""
    df = pd.read_excel(path, sheet_name="OptionTrades")
    df["TradeDate"] = pd.to_datetime(df["TradeDate"])
    df["ExpiryDate"] = pd.to_datetime(df["ExpiryDate"])
    df["SignedQty"] = np.where(df["BuySell"].str.upper() == "BUY", df["Quantity"], -df["Quantity"])
    return df.sort_values("TradeDate").reset_index(drop=True)


def load_volatility_surface(path: str) -> pd.DataFrame:
    """Read the dummy implied-vol surface."""
    df = pd.read_csv(path, parse_dates=["Date", "ExpiryDate"])
    return df.sort_values(["UnderlyingContract", "ExpiryDate", "Date", "Strike"]).reset_index(drop=True)


# ==========================================================================
# 3. FIFO realized P&L on options (reuses pnl_engine's generic matcher)
# ==========================================================================

_KEY_SEP = "|"


def _option_key(underlying: str, option_type: str, strike: float, expiry: pd.Timestamp) -> str:
    """Composite key: a BUY and a SELL only net against each other if all four match."""
    return f"{underlying}{_KEY_SEP}{option_type.upper()}{_KEY_SEP}{strike:.2f}{_KEY_SEP}{expiry.strftime('%Y-%m-%d')}"


def _decode_option_key(key: str) -> dict:
    underlying, option_type, strike, expiry = key.split(_KEY_SEP)
    return {"UnderlyingContract": underlying, "OptionType": option_type,
            "Strike": float(strike), "ExpiryDate": pd.Timestamp(expiry)}


def compute_option_fifo_positions(option_trades: pd.DataFrame):
    """
    FIFO-match option trades, grouped by (Underlying, Type, Strike, Expiry).
    Delegates the actual matching algorithm to pnl_engine.compute_fifo_positions
    by building a synthetic "Contract" column from the composite key and
    "Price" column from Premium -- see module docstring for why.

    Returns
    -------
    open_lots       : dict[option_key] -> deque[Lot]   (still-open option lots)
    realized_df     : DataFrame of matched closes, with the composite key
                       split back out into UnderlyingContract/OptionType/
                       Strike/ExpiryDate columns for readability.
    """
    from pnl_engine import compute_fifo_positions  # local import, see risk_engine.py for the same pattern

    prepped = option_trades.copy()
    prepped["Contract"] = [
        _option_key(row.UnderlyingContract, row.OptionType, row.Strike, row.ExpiryDate)
        for row in prepped.itertuples()
    ]
    prepped["Price"] = prepped["Premium"]  # FIFO matcher is generic on "Price"; premium plays that role here

    open_lots, realized_df = compute_fifo_positions(prepped)

    if not realized_df.empty:
        decoded = realized_df["Contract"].apply(_decode_option_key).apply(pd.Series)
        realized_df = pd.concat([decoded, realized_df.drop(columns=["Contract"])], axis=1)
        realized_df = realized_df.rename(columns={"EntryPrice": "EntryPremium", "ExitPrice": "ExitPremium"})

    return open_lots, realized_df


def open_option_position_summary(open_lots: dict) -> pd.DataFrame:
    """Collapse open option lots into one row per (Underlying, Type, Strike, Expiry): net qty & avg premium."""
    rows = []
    for key, lots in open_lots.items():
        if not lots:
            continue
        net_qty = sum(l.qty for l in lots)
        if net_qty == 0:
            continue
        avg_premium = sum(l.qty * l.price for l in lots) / net_qty
        rows.append({**_decode_option_key(key), "NetQty": net_qty, "AvgEntryPremium": avg_premium})
    return pd.DataFrame(rows)


# ==========================================================================
# 4. Volatility surface lookup
# ==========================================================================

def lookup_implied_vol(vol_surface: pd.DataFrame, underlying: str, expiry: pd.Timestamp,
                        strike: float, as_of_date: pd.Timestamp) -> float:
    """
    Nearest-date, nearest-expiry, strike-interpolated vol lookup.

    - Date: nearest available surface date to `as_of_date` (handles the
      surface not having a row for a weekend/holiday).
    - Expiry: nearest ExpiryDate on the surface for this underlying (the
      dummy surface has exactly one expiry per underlying, so this is
      exact in practice; the nearest-match logic is what would let a
      richer surface with multiple tenors per underlying work unchanged).
    - Strike: linearly interpolated across that day's strike grid
      (np.interp also handles extrapolation by clamping to the grid ends).
    """
    sub = vol_surface[vol_surface["UnderlyingContract"] == underlying]
    if sub.empty:
        raise ValueError(f"No volatility surface data for underlying '{underlying}'.")

    nearest_expiry = min(sub["ExpiryDate"].unique(), key=lambda e: abs((pd.Timestamp(e) - expiry).days))
    sub = sub[sub["ExpiryDate"] == nearest_expiry]

    nearest_date = min(sub["Date"].unique(), key=lambda d: abs((pd.Timestamp(d) - as_of_date).days))
    day_slice = sub[sub["Date"] == nearest_date].sort_values("Strike")

    return float(np.interp(strike, day_slice["Strike"].values, day_slice["ImpliedVol"].values))


# ==========================================================================
# 5. Live portfolio Greeks on currently open positions
# ==========================================================================

def compute_live_option_greeks(option_trades: pd.DataFrame, settles: pd.DataFrame,
                                vol_surface: pd.DataFrame, as_of_date: pd.Timestamp = None,
                                r: float = RISK_FREE_RATE_DEFAULT) -> pd.DataFrame:
    """
    Revalue every currently open option lot (net, per Underlying/Type/
    Strike/Expiry) using Black-76 and the vol surface, as of `as_of_date`
    (defaults to the latest settlement date available).

    Lots whose ExpiryDate has already passed as_of_date are EXCLUDED --
    see the module docstring's note on expiry handling.

    Position-level Greeks are the per-unit Greek x NetQty x LotSize, i.e.
    expressed as the actual $ (or $/vol-point, $/day) sensitivity of the
    position, not a per-contract unit figure.

    Returns one row per open option position with Price, per-unit Greeks,
    and position-scaled Greeks (prefixed "Position").
    """
    open_lots, _ = compute_option_fifo_positions(option_trades)
    positions = open_option_position_summary(open_lots)
    if positions.empty:
        return pd.DataFrame()

    if as_of_date is None:
        as_of_date = settles["Date"].max()
    last_settle_by_underlying = settles.sort_values("Date").groupby("Contract")["SettlePrice"].last().to_dict()
    lot_size_by_underlying = option_trades.groupby("UnderlyingContract")["LotSize"].first().to_dict()

    rows = []
    for pos in positions.itertuples():
        days_to_expiry = (pos.ExpiryDate - as_of_date).days
        if days_to_expiry <= 0:
            continue  # expired -- see module docstring's known simplification

        F = last_settle_by_underlying.get(pos.UnderlyingContract)
        if F is None:
            continue
        T = days_to_expiry / 365.0
        sigma = lookup_implied_vol(vol_surface, pos.UnderlyingContract, pos.ExpiryDate, pos.Strike, as_of_date)
        lot_size = lot_size_by_underlying.get(pos.UnderlyingContract, 1000)

        price = black76_price(F, pos.Strike, T, r, sigma, pos.OptionType)
        greeks = black76_greeks(F, pos.Strike, T, r, sigma, pos.OptionType)

        scale = pos.NetQty * lot_size
        rows.append({
            "UnderlyingContract": pos.UnderlyingContract, "OptionType": pos.OptionType,
            "Strike": pos.Strike, "ExpiryDate": pos.ExpiryDate, "DaysToExpiry": days_to_expiry,
            "NetQty": pos.NetQty, "AvgEntryPremium": pos.AvgEntryPremium,
            "Forward": F, "ImpliedVol": sigma, "TheoPrice": price,
            "Delta": greeks["Delta"], "Gamma": greeks["Gamma"], "Vega": greeks["Vega"],
            "Theta": greeks["Theta"], "Rho": greeks["Rho"],
            "PositionDelta": scale * greeks["Delta"], "PositionGamma": scale * greeks["Gamma"],
            "PositionVega": scale * greeks["Vega"], "PositionTheta": scale * greeks["Theta"],
            "PositionRho": scale * greeks["Rho"], "PositionValue": pos.NetQty * lot_size * price,
        })

    return pd.DataFrame(rows)


def portfolio_greeks_summary(live_greeks: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate compute_live_option_greeks()'s per-position rows into one
    row per underlying plus a TOTAL row -- options book only, deliberately
    NOT netted against the futures book's Delta (see pnl_engine.py /
    dashboard.py for that separate figure).
    """
    if live_greeks.empty:
        return pd.DataFrame()

    cols = ["PositionDelta", "PositionGamma", "PositionVega", "PositionTheta", "PositionRho", "PositionValue"]
    by_underlying = live_greeks.groupby("UnderlyingContract")[cols].sum().reset_index()

    total = by_underlying[cols].sum().to_frame().T
    total.insert(0, "UnderlyingContract", "TOTAL (options book)")

    return pd.concat([by_underlying, total], ignore_index=True)


# ==========================================================================
# Self-test
# ==========================================================================
if __name__ == "__main__":
    # Sanity check #1: Black-76 call/put parity -- C - P = DF*(F - K)
    F, K, T, r, sigma = 82.0, 80.0, 0.25, 0.04, 0.30
    c = black76_price(F, K, T, r, sigma, "call")
    p = black76_price(F, K, T, r, sigma, "put")
    parity_lhs = c - p
    parity_rhs = np.exp(-r * T) * (F - K)
    print(f"Call-put parity check: C-P={parity_lhs:.6f}  DF*(F-K)={parity_rhs:.6f}  diff={abs(parity_lhs-parity_rhs):.2e}")
    assert abs(parity_lhs - parity_rhs) < 1e-8, "Call-put parity violated -- bug in black76_price!"
    print("OK: call-put parity holds.\n")

    # Sanity check #2: delta should be near 0.5 ATM, near 0/1 deep ITM/OTM
    atm_greeks = black76_greeks(80.0, 80.0, 0.25, 0.04, 0.30, "call")
    deep_itm_greeks = black76_greeks(100.0, 80.0, 0.25, 0.04, 0.30, "call")
    deep_otm_greeks = black76_greeks(60.0, 80.0, 0.25, 0.04, 0.30, "call")
    print(f"ATM call delta:      {atm_greeks['Delta']:.3f} (expect ~0.5)")
    print(f"Deep ITM call delta: {deep_itm_greeks['Delta']:.3f} (expect close to 1)")
    print(f"Deep OTM call delta: {deep_otm_greeks['Delta']:.3f} (expect close to 0)")
    assert 0.4 < atm_greeks["Delta"] < 0.6
    assert deep_itm_greeks["Delta"] > 0.9
    assert deep_otm_greeks["Delta"] < 0.1
    print("OK: delta behaves as expected across moneyness.\n")

    # Full pipeline on the actual (dummy) options book, if files are present
    try:
        from pnl_engine import load_settlement_prices

        option_trades = load_option_trades("options_trades_dummy.xlsx")
        settles = load_settlement_prices("settlement_prices.csv")
        vol_surface = load_volatility_surface("volatility_surface_dummy.csv")

        open_lots, realized_df = compute_option_fifo_positions(option_trades)
        print(f"Realized option PnL events: {len(realized_df)}")
        if not realized_df.empty:
            print(realized_df[["UnderlyingContract", "OptionType", "Strike", "QtyClosed", "RealizedPnL"]]
                  .to_string(index=False))
            print(f"Total realized option PnL: {realized_df['RealizedPnL'].sum():,.2f} USD\n")

        live_greeks = compute_live_option_greeks(option_trades, settles, vol_surface)
        print(f"Open option positions: {len(live_greeks)}")
        print(live_greeks[["UnderlyingContract", "OptionType", "Strike", "DaysToExpiry", "NetQty",
                            "ImpliedVol", "TheoPrice", "Delta", "Gamma", "Vega", "Theta"]].round(4).to_string(index=False))

        print("\nPortfolio Greeks summary (options book only, not netted vs futures):")
        print(portfolio_greeks_summary(live_greeks).round(2).to_string(index=False))

    except FileNotFoundError as e:
        print(f"\n(Skipping full pipeline test -- data file not found: {e})")
