"""
market_vol_surface.py
======================
Builds a REAL, market-calibrated implied volatility surface for crude
oil, using listed options on USO (WTI proxy) and BNO (Brent proxy) --
the closest free substitute for licensed ICE Brent / CME WTI futures
options data (see market_data.py's docstring for the identical licensing
constraint on settlement prices: real per-contract futures options data
lives behind Bloomberg/Refinitiv/ICE Data, not anywhere free).

This is real market data, not synthetic -- unlike volatility_surface_
dummy.csv (a hand-built smile shape), every point here is calibrated
from an actual traded/quoted option price via Black-76. The tradeoff is
coverage: these are ETF options, not futures options, so there's real
basis (fund roll costs, tracking error) between this and a true Brent/
WTI futures vol surface, and -- especially for the thinly-traded BNO --
listed liquidity may not reach anywhere near the full target tenor
ladder. That's real market structure, not a bug in this code.

Method
------
For each ticker (USO, BNO):
  1. Pull every listed expiry's full option chain (calls + puts).
  2. Keep only quotes with a genuine two-sided market (bid>0, ask>0) AND
     actual trading activity (volume>0 or openInterest>0) -- a resting
     zero-size quote with no interest isn't a real price.
  3. Drop any expiry that doesn't have enough surviving quotes to build
     a smile (MIN_QUOTES_PER_EXPIRY).
  4. For each surviving (expiry, strike), take the OTM side's price
     (call if strike >= forward, put if strike < forward -- OTM options
     are typically more liquid / tighter-spread than the ITM option at
     the same strike) as the bid/ask midpoint, and back out Black-76
     implied vol via greeks_engine.implied_vol() -- using
     F = spot * exp(r*T), the standard spot-to-forward conversion that
     makes Black-76 equivalent to Black-Scholes for pricing a spot
     instrument. This reuses the EXACT SAME Black-76 implementation that
     prices the actual book, so calibration and pricing can never
     silently disagree on convention.
  5. Interpolate across STRIKE (linear in log-moneyness) onto the
     target 9-point moneyness grid (70%-130%), per usable expiry. A
     target point outside the actually-quoted moneyness range for that
     expiry is DROPPED -- NO EXTRAPOLATION.
  6. Interpolate across MATURITY onto the target 8-point tenor ladder
     (1M,3M,6M,9M,12M,18M,24M,36M), per moneyness column, in TOTAL
     VARIANCE space (sigma^2 * T) -- the standard, arbitrage-consistent
     way to interpolate a vol term structure (naive linear interpolation
     of sigma itself can imply a negative forward variance between two
     tenors, a calendar-arbitrage violation). A target tenor outside
     [min_usable_T, max_usable_T] is DROPPED -- NO EXTRAPOLATION. In
     practice this means USO likely reaches several tenor points out;
     BNO, being much thinner, may only populate the near end of the
     ladder. The output grid is allowed to be incomplete.

Output schema matches volatility_surface_dummy.csv exactly (Date,
UnderlyingContract, ExpiryDate, Strike, ImpliedVol) so
greeks_engine.lookup_implied_vol() works against it unchanged.
UnderlyingContract is the COMMODITY label ("BRENT"/"WTI"), not a specific
futures contract month -- see greeks_engine.map_underlying_to_commodity()
for how a book's "BRN Sep26"-style contract maps onto this surface.
"""

import numpy as np
import pandas as pd

from greeks_engine import implied_vol, RISK_FREE_RATE_DEFAULT

try:
    import yfinance as yf
except ImportError:
    yf = None

STRIKE_MONEYNESS_GRID = np.array([0.70, 0.775, 0.85, 0.925, 1.00, 1.075, 1.15, 1.225, 1.30])  # 9 points, 70%-130%
MATURITY_MONTHS_GRID = [1, 3, 6, 9, 12, 18, 24, 36]  # 8 points
MIN_QUOTES_PER_EXPIRY = 4  # minimum surviving liquid quotes to attempt a smile for that expiry
MIN_DAYS_TO_EXPIRY = 7      # sub-1-week expiries give unstable IVs (tiny T, pin risk, stale mids)
MAX_REL_SPREAD = 0.50       # drop quotes whose (ask-bid)/mid > 50% -- the mid is noise, not a price
LAST_PRICE_MAX_AGE_DAYS = 5  # last-trade fallback: only trades from the last 5 calendar days (covers a weekend + holiday)


# ==========================================================================
# 1. Raw data fetch (I/O layer)
# ==========================================================================

def _get_spot_price(ticker_obj) -> float:
    """Current price for the ETF, with a fallback if fast_info is unavailable/incomplete."""
    try:
        price = ticker_obj.fast_info.get("last_price") or ticker_obj.fast_info.get("lastPrice")
        if price:
            return float(price)
    except Exception:
        pass
    hist = ticker_obj.history(period="5d")
    if hist.empty or hist["Close"].dropna().empty:
        raise RuntimeError("Could not determine a spot price (fast_info and 5-day history both failed/empty).")
    return float(hist["Close"].dropna().iloc[-1])


def _num(x) -> float:
    """yfinance gives NaN/None for missing fields; `float(nan or 0)` stays NaN, so handle it explicitly."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return 0.0
    return 0.0 if np.isnan(v) else v


def fetch_raw_option_chain(ticker: str) -> tuple:
    """
    Pull every listed expiry's full chain (calls+puts) for `ticker`, plus
    the current spot price.

    Returns (raw_df, spot) where raw_df has columns: Expiry, OptionType
    ('call'/'put'), Strike, Bid, Ask, LastPrice, LastTradeDate, Volume,
    OpenInterest.
    """
    if yf is None:
        raise ImportError("yfinance is not installed. Run: pip install yfinance")

    tk = yf.Ticker(ticker)
    spot = _get_spot_price(tk)
    expiries = tk.options
    if not expiries:
        raise RuntimeError(f"{ticker}: no listed option expiries returned.")

    rows = []
    for expiry_str in expiries:
        chain = tk.option_chain(expiry_str)
        expiry = pd.Timestamp(expiry_str)
        for option_type, df in (("call", chain.calls), ("put", chain.puts)):
            if df.empty:
                continue
            for _, r in df.iterrows():
                ltd = pd.to_datetime(r.get("lastTradeDate"), utc=True, errors="coerce")
                rows.append({
                    "Expiry": expiry, "OptionType": option_type, "Strike": float(r["strike"]),
                    "Bid": _num(r.get("bid")), "Ask": _num(r.get("ask")),
                    "LastPrice": _num(r.get("lastPrice")),
                    "LastTradeDate": ltd.tz_localize(None) if pd.notna(ltd) else pd.NaT,
                    "Volume": _num(r.get("volume")), "OpenInterest": _num(r.get("openInterest")),
                })
    if not rows:
        raise RuntimeError(f"{ticker}: listed expiries exist but returned zero option rows.")
    return pd.DataFrame(rows), spot


# ==========================================================================
# 2. Liquidity filtering
# ==========================================================================

def filter_liquid_quotes(raw: pd.DataFrame, as_of_date: pd.Timestamp = None,
                         max_last_age_days: int = LAST_PRICE_MAX_AGE_DAYS) -> pd.DataFrame:
    """
    Pick ONE price per option, from real market data only:

      1. "mid"  -- a genuine two-sided quote (bid>0, ask>=bid, spread <= 50%
                   of mid) with real activity (volume or open interest).
      2. "last" -- ONLY if there is no usable two-sided quote: the last
                   traded price, if that trade happened within the last
                   `max_last_age_days` calendar days.

    Why the fallback: outside US market hours -- and all weekend -- Yahoo
    returns bid = ask = 0 for every listed option. With mids only, BOTH the
    BNO and USO surfaces then fail to build ("no expiry had >= 4 liquid
    quotes"). The last traded price is still a real transaction; it's just
    less precise than a live mid (different strikes may have last traded at
    different times), so every point carries its PriceBasis.

    The output column is still called "Mid" (= the price used) so the rest
    of the pipeline is unchanged.
    """
    if as_of_date is None:
        as_of_date = pd.Timestamp.now().normalize()
    active = (raw["Volume"] > 0) | (raw["OpenInterest"] > 0)

    mid = (raw["Bid"] + raw["Ask"]) / 2.0
    two_sided = (raw["Bid"] > 0) & (raw["Ask"] > 0) & (raw["Ask"] >= raw["Bid"]) & active
    # Thin ETF chains (BNO especially) often quote 0.05/0.40-type markets;
    # the mid of such a quote is meaningless and creates the jagged smile.
    two_sided &= ((raw["Ask"] - raw["Bid"]) / mid.where(mid > 0)) <= MAX_REL_SPREAD

    if "LastPrice" in raw.columns:
        age_days = (as_of_date - pd.to_datetime(raw["LastTradeDate"])).dt.days
        last_ok = (~two_sided) & active & (raw["LastPrice"] > 0) & age_days.between(0, max_last_age_days)
    else:
        last_ok = pd.Series(False, index=raw.index)

    liquid = raw[two_sided | last_ok].copy()
    liquid["Mid"] = np.where(two_sided[liquid.index], mid[liquid.index], liquid.get("LastPrice", np.nan))
    liquid["PriceBasis"] = np.where(two_sided[liquid.index], "mid", "last")
    return liquid


# ==========================================================================
# 3. OTM selection + Black-76 implied vol back-out
# ==========================================================================

def compute_iv_points(liquid: pd.DataFrame, spot: float, r: float, as_of_date: pd.Timestamp) -> pd.DataFrame:
    """
    For each (Expiry, Strike), prefer the OTM side's quote, back out
    Black-76 implied vol via greeks_engine.implied_vol using
    F = spot * exp(r*T). Points where brentq fails to converge (bad/
    crossed quote, price outside no-arbitrage bounds) are DROPPED, not
    guessed at -- same "fail loudly / don't silently fabricate" posture
    as the rest of this codebase.
    """
    results = []
    for expiry, group in liquid.groupby("Expiry"):
        T = (expiry - as_of_date).days / 365.0
        if T < MIN_DAYS_TO_EXPIRY / 365.0:
            continue
        F = spot * np.exp(r * T)

        for strike, strike_group in group.groupby("Strike"):
            side = "call" if strike >= F else "put"
            row = strike_group[strike_group["OptionType"] == side]
            if row.empty:
                continue
            mid = row["Mid"].iloc[0]
            basis = row["PriceBasis"].iloc[0] if "PriceBasis" in row.columns else "mid"
            try:
                vol = implied_vol(mid, F, strike, T, r, option_type=side)
            except Exception:
                continue  # no-arb violation or non-convergence -- skip, don't guess
            if not (0.01 < vol < 3.0):  # sanity bound; drop nonsensical solves
                continue
            results.append({"Expiry": expiry, "T": T, "Strike": strike, "Forward": F,
                             "Moneyness": strike / F, "ImpliedVol": vol, "PriceBasis": basis})
    return pd.DataFrame(results)


def filter_usable_expiries(iv_points: pd.DataFrame, min_quotes: int = MIN_QUOTES_PER_EXPIRY) -> pd.DataFrame:
    """Drop any expiry without enough surviving quotes to build a believable smile."""
    if iv_points.empty:
        return iv_points
    counts = iv_points.groupby("Expiry").size()
    usable_expiries = counts[counts >= min_quotes].index
    return iv_points[iv_points["Expiry"].isin(usable_expiries)].copy()


# ==========================================================================
# 4. Strike interpolation (per expiry, no extrapolation)
# ==========================================================================

def interpolate_strikes(iv_points: pd.DataFrame, moneyness_grid: np.ndarray = STRIKE_MONEYNESS_GRID) -> pd.DataFrame:
    """
    Per usable expiry, linearly interpolate vol against LOG-moneyness
    onto the target grid. A target point outside the actually-quoted
    moneyness range for that expiry is dropped -- no extrapolation.

    Returns columns: Expiry, T, Moneyness, ImpliedVol (Strike is
    deliberately NOT carried over here -- the final absolute Strike is
    recomputed later against each TARGET maturity's own forward, not
    this expiry's forward, which would otherwise be inconsistent).
    """
    rows = []
    target_log_m = np.log(moneyness_grid)

    for expiry, group in iv_points.groupby("Expiry"):
        group = group.sort_values("Moneyness")
        log_m = np.log(group["Moneyness"].values)
        vols = group["ImpliedVol"].values
        T = group["T"].iloc[0]

        if log_m.max() == log_m.min():
            continue  # need at least two distinct moneyness points to interpolate at all

        in_range = (target_log_m >= log_m.min()) & (target_log_m <= log_m.max())
        if not in_range.any():
            continue
        interp_vols = np.interp(target_log_m, log_m, vols)

        for moneyness, vol, keep in zip(moneyness_grid, interp_vols, in_range):
            if not keep:
                continue
            rows.append({"Expiry": expiry, "T": T, "Moneyness": moneyness, "ImpliedVol": vol})

    return pd.DataFrame(rows)


# ==========================================================================
# 5. Maturity interpolation (per moneyness column, total-variance space, no extrapolation)
# ==========================================================================

def interpolate_maturities(strike_grid: pd.DataFrame, as_of_date: pd.Timestamp,
                            maturity_months_grid: list = MATURITY_MONTHS_GRID) -> pd.DataFrame:
    """
    Per moneyness point, linearly interpolate TOTAL VARIANCE (sigma^2*T)
    across the usable expiries' T values onto the target tenor ladder --
    see module docstring for why total-variance interpolation, not raw
    vol, is the correct approach. A target tenor outside
    [min_usable_T, max_usable_T] is dropped -- no extrapolation.

    Needs at least 2 usable expiries at a given moneyness to interpolate
    a term structure at all; a single-expiry moneyness point is dropped
    rather than held flat (holding flat would itself be a form of
    extrapolation).
    """
    rows = []
    for moneyness, group in strike_grid.groupby("Moneyness"):
        group = group.sort_values("T")
        if len(group) < 2:
            continue
        T_vals = group["T"].values
        total_var = (group["ImpliedVol"].values ** 2) * T_vals

        for months in maturity_months_grid:
            T_target = months / 12.0
            if T_target < T_vals.min() or T_target > T_vals.max():
                continue  # outside real coverage -- no extrapolation
            interp_total_var = np.interp(T_target, T_vals, total_var)
            vol_target = np.sqrt(interp_total_var / T_target)
            rows.append({
                "Moneyness": moneyness, "MaturityMonths": months, "T": T_target,
                "ExpiryDate": as_of_date + pd.DateOffset(months=months),
                "ImpliedVol": vol_target,
            })
    return pd.DataFrame(rows)


# ==========================================================================
# 6. Full pipeline
# ==========================================================================

def build_market_vol_surface(ticker: str, commodity_label: str, r: float = RISK_FREE_RATE_DEFAULT,
                              as_of_date: pd.Timestamp = None, report: dict = None) -> pd.DataFrame:
    """
    Full pipeline: fetch -> filter liquid -> compute IV (OTM, Black-76) ->
    filter usable expiries -> interpolate strikes -> interpolate
    maturities. Both interpolation steps are strictly non-extrapolating.

    Returns a DataFrame in the SAME schema as volatility_surface_dummy.csv
    (Date, UnderlyingContract, ExpiryDate, Strike, ImpliedVol), with
    UnderlyingContract = commodity_label. May have fewer than the full
    9x8=72 grid points -- expected, not a bug, especially for BNO.
    """
    if as_of_date is None:
        as_of_date = pd.Timestamp.now().normalize()

    rep = report if report is not None else {}
    raw, spot = fetch_raw_option_chain(ticker)
    rep.update({"Spot": round(spot, 2), "Expiries listed": raw["Expiry"].nunique(), "Raw quotes": len(raw),
                "Quotes with bid>0 & ask>0": int(((raw["Bid"] > 0) & (raw["Ask"] > 0)).sum())})
    liquid = filter_liquid_quotes(raw, as_of_date)
    rep.update({"Priced from live mid": int((liquid["PriceBasis"] == "mid").sum()),
                "Priced from last trade": int((liquid["PriceBasis"] == "last").sum())})
    iv_points = compute_iv_points(liquid, spot, r, as_of_date)
    rep["IV points solved"] = len(iv_points)
    usable = filter_usable_expiries(iv_points)
    rep["Usable expiries"] = usable["Expiry"].nunique() if not usable.empty else 0
    if usable.empty:
        hint = (" Bid/ask are all zero -- usually the market is closed (weekend / after hours) and no option "
                f"traded in the last {LAST_PRICE_MAX_AGE_DAYS} days."
                if rep["Quotes with bid>0 & ask>0"] == 0 else "")
        raise RuntimeError(f"{ticker}: no expiry had >= {MIN_QUOTES_PER_EXPIRY} usable quotes to build a smile.{hint}")

    strike_grid = interpolate_strikes(usable)
    if strike_grid.empty:
        raise RuntimeError(f"{ticker}: strike interpolation produced zero usable points.")

    maturity_grid = interpolate_maturities(strike_grid, as_of_date)
    rep["Final grid points"] = len(maturity_grid)
    if maturity_grid.empty:
        raise RuntimeError(f"{ticker}: maturity interpolation produced zero usable points "
                            f"(fewer than 2 usable expiries at every moneyness point).")
    basis = usable.get("PriceBasis", pd.Series(["mid"]))
    rep["Price basis"] = "live mid" if (basis == "mid").all() else (
        "last trade" if (basis == "last").all() else f"mixed ({(basis == 'last').mean():.0%} last trade)")

    # Final Strike is computed against the TARGET maturity's own forward
    # (spot * exp(r*T_target)) -- not the forward of whichever expiry
    # contributed to the interpolation, which would be inconsistent.
    maturity_grid["Strike"] = maturity_grid["Moneyness"] * spot * np.exp(r * maturity_grid["T"])
    maturity_grid["Date"] = as_of_date
    maturity_grid["UnderlyingContract"] = commodity_label
    maturity_grid["PriceBasis"] = rep["Price basis"]

    # Moneyness is kept alongside Strike (not just internally) because it's
    # the only axis that stays consistent across maturities for plotting a
    # 3D surface -- Strike itself shifts with the forward at each tenor, so
    # pivoting by raw Strike would NOT line up into a clean grid. Extra
    # column, fully backward-compatible: lookup_implied_vol() only reads
    # the columns it needs and ignores the rest.
    return maturity_grid[["Date", "UnderlyingContract", "ExpiryDate", "Strike", "Moneyness", "ImpliedVol",
                          "PriceBasis"]].reset_index(drop=True)


# ==========================================================================
# 7. Futures-options SETTLEMENT chains (Brent / WTI) -- no ETF proxy
# ==========================================================================

MIN_SETTLE_FOR_IV = 0.05   # below ~5 ticks, 0.01 rounding dominates the price -> IV is noise


def implied_vol_vectorized(price, F, K, T, r, is_call, tol=1e-10, max_iter=100):
    """
    Black-76 implied vol for whole arrays at once (bisection on the price,
    which is monotonic in sigma -- same answer as greeks_engine.implied_vol's
    brentq, ~100x faster for thousands of options). Returns NaN where the
    price is outside the no-arbitrage bounds.
    """
    from scipy.stats import norm
    price, F, K, T = (np.asarray(a, dtype=float) for a in (price, F, K, T))
    is_call = np.asarray(is_call, dtype=bool)
    df = np.exp(-r * T)
    intrinsic = df * np.where(is_call, np.maximum(F - K, 0), np.maximum(K - F, 0))
    upper = df * np.where(is_call, F, K)
    valid = (price > intrinsic) & (price < upper) & (T > 0)

    def b76(sig):
        sq = sig * np.sqrt(T)
        d1 = (np.log(F / K) + 0.5 * sq * sq) / sq
        d2 = d1 - sq
        call = df * (F * norm.cdf(d1) - K * norm.cdf(d2))
        return np.where(is_call, call, call - df * (F - K))      # put via put-call parity

    lo, hi = np.full(price.shape, 1e-6), np.full(price.shape, 5.0)
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        too_high = b76(mid) > price
        hi = np.where(too_high, mid, hi)
        lo = np.where(too_high, lo, mid)
        if np.nanmax(hi - lo) < tol:
            break
    return np.where(valid, 0.5 * (lo + hi), np.nan)


def build_surface_from_settlement_chain(chain: pd.DataFrame, r: float = RISK_FREE_RATE_DEFAULT,
                                        min_premium: float = MIN_SETTLE_FOR_IV,
                                        min_quotes: int = MIN_QUOTES_PER_EXPIRY) -> pd.DataFrame:
    """
    Implied vol surface straight from end-of-day settlement prices of options
    ON FUTURES (Brent BRN, WTI CL) -- the real instrument, not BNO/USO.

    Input columns (what an exchange / vendor EOD file provides):
        Date, Commodity, UnderlyingContract, FuturesSettle, ExpiryDate,
        OptionType (CALL/PUT), Strike, SettlePrice[, Volume, OpenInterest]

    Per (Date, contract month):
      - F = that contract month's futures settle, same day. No spot-to-
        forward conversion: these options are written on the future itself.
      - Out-of-the-money side only (call if K >= F, put if K < F): OTM
        prices carry no early-exercise premium, so Black-76 (European) is
        right even though Brent/WTI options are American-style.
      - Settles below `min_premium` are skipped: at a 0.01 tick, a 0.02
        option's implied vol is mostly rounding.
      - Contract months with fewer than `min_quotes` usable strikes that day
        are dropped rather than guessed.

    Output: the same schema the rest of the code already reads (Date,
    UnderlyingContract, ExpiryDate, Strike, Moneyness, ImpliedVol), plus
    Commodity and PriceBasis="settlement". Strikes are in $/bbl and keyed by
    contract month, so greeks_engine looks them up directly (no commodity
    fallback, no ETF unit conversion).
    """
    need = {"Date", "UnderlyingContract", "FuturesSettle", "ExpiryDate", "OptionType", "Strike", "SettlePrice"}
    missing = need - set(chain.columns)
    if missing:
        raise ValueError(f"Option chain is missing columns: {sorted(missing)}")

    c = chain.copy()
    c["Date"] = pd.to_datetime(c["Date"])
    c["ExpiryDate"] = pd.to_datetime(c["ExpiryDate"])
    c["OptionType"] = c["OptionType"].str.upper()
    otm = ((c["OptionType"] == "CALL") & (c["Strike"] >= c["FuturesSettle"])) | \
          ((c["OptionType"] == "PUT") & (c["Strike"] < c["FuturesSettle"]))
    c = c[otm & (c["SettlePrice"] >= min_premium) & (c["ExpiryDate"] > c["Date"])]

    T = (c["ExpiryDate"] - c["Date"]).dt.days.values / 365.0
    vols = implied_vol_vectorized(c["SettlePrice"].values, c["FuturesSettle"].values, c["Strike"].values,
                                  T, r, (c["OptionType"] == "CALL").values)
    ok = np.isfinite(vols) & (vols > 0.01) & (vols < 3.0)   # unsolvable (no-arb violation) -> dropped, not guessed
    c = c[ok]
    rows = pd.DataFrame({
        "Date": c["Date"].values, "UnderlyingContract": c["UnderlyingContract"].values,
        "Commodity": c["Commodity"].values if "Commodity" in c.columns else None,
        "ExpiryDate": c["ExpiryDate"].values, "Strike": c["Strike"].values,
        "Forward": c["FuturesSettle"].values, "Moneyness": (c["Strike"] / c["FuturesSettle"]).values,
        "ImpliedVol": vols[ok],
    })
    out = rows
    if out.empty:
        return out
    counts = out.groupby(["Date", "UnderlyingContract"])["Strike"].transform("size")
    out = out[counts >= min_quotes].copy()
    out["PriceBasis"] = "settlement"
    return out.sort_values(["UnderlyingContract", "Date", "Strike"]).reset_index(drop=True)


def build_full_market_vol_surface(as_of_date: pd.Timestamp = None) -> pd.DataFrame:
    """Builds and concatenates both commodities' surfaces (BNO->BRENT, USO->WTI)."""
    brent = build_market_vol_surface("BNO", "BRENT", as_of_date=as_of_date)
    wti = build_market_vol_surface("USO", "WTI", as_of_date=as_of_date)
    return pd.concat([brent, wti], ignore_index=True)


# ==========================================================================
# Self-test -- uses SYNTHETIC option chain data (no live network needed),
# built by generating real Black-76 prices from KNOWN vols and feeding
# them back through the full pipeline exactly as if they were a real
# yfinance chain. This verifies the calibration math (IV back-out,
# strike interpolation, maturity interpolation, no-extrapolation
# behavior) independent of any network access.
# ==========================================================================
if __name__ == "__main__":
    from greeks_engine import black76_price

    as_of = pd.Timestamp("2026-10-01")
    spot = 80.0
    r = RISK_FREE_RATE_DEFAULT

    # Build a synthetic chain: 4 expiries (so maturity interpolation has
    # something to work with), 9 strikes each, vols following a simple
    # known smile (so we can check recovered IVs against ground truth).
    synthetic_expiries_days = [30, 90, 180, 365]  # ~1M, 3M, 6M, 12M
    strikes_pct = np.linspace(0.75, 1.25, 9)
    true_vol_base = 0.30

    rows = []
    for days in synthetic_expiries_days:
        expiry = as_of + pd.Timedelta(days=days)
        T = days / 365.0
        F = spot * np.exp(r * T)
        for pct in strikes_pct:
            K = round(F * pct, 2)
            moneyness = (K - F) / F
            true_vol = true_vol_base + 0.05 * moneyness ** 2  # mild smile, known shape
            side = "call" if K >= F else "put"
            price = black76_price(F, K, T, r, true_vol, side)
            # Build a realistic bid/ask around the true price with real liquidity
            rows.append({"Expiry": expiry, "OptionType": side, "Strike": K,
                         "Bid": price * 0.98, "Ask": price * 1.02, "Volume": 100, "OpenInterest": 500})

    synthetic_raw = pd.DataFrame(rows)

    print("=== Self-test 1: full pipeline recovers the known synthetic smile ===")
    liquid = filter_liquid_quotes(synthetic_raw)
    iv_points = compute_iv_points(liquid, spot, r, as_of)
    usable = filter_usable_expiries(iv_points)
    print(f"Usable expiries: {usable['Expiry'].nunique()} (expected 4)")
    assert usable["Expiry"].nunique() == 4

    strike_grid = interpolate_strikes(usable)
    maturity_grid = interpolate_maturities(strike_grid, as_of)
    maturity_grid["Strike"] = maturity_grid["Moneyness"] * spot * np.exp(r * maturity_grid["T"])
    print(f"Final grid points: {len(maturity_grid)}")
    print(f"Maturities recovered: {sorted(maturity_grid['MaturityMonths'].unique())}")

    # Spot-check: ATM, 3-month point should recover close to true_vol_base
    atm_3m = maturity_grid[(maturity_grid["Moneyness"] == 1.0) & (maturity_grid["MaturityMonths"] == 3)]
    if not atm_3m.empty:
        recovered = atm_3m["ImpliedVol"].iloc[0]
        print(f"ATM 3M recovered vol: {recovered:.4f} (true: {true_vol_base:.4f})")
        assert abs(recovered - true_vol_base) < 0.02, "Recovered ATM vol too far from the known input -- calibration bug"
    print("OK: pipeline recovers the known synthetic smile within tolerance.\n")

    print("=== Self-test 2: no-extrapolation is actually enforced ===")
    # Maturities outside [1M, 12M] (our synthetic expiry range) must NOT appear.
    out_of_range = [m for m in maturity_grid["MaturityMonths"].unique() if m not in (1, 3, 6, 9, 12)]
    print(f"Maturities beyond synthetic coverage present in output: {out_of_range} (must be empty)")
    assert maturity_grid["MaturityMonths"].max() <= 12, "Maturity extrapolation occurred -- should never happen"
    assert maturity_grid["MaturityMonths"].min() >= 1
    print("OK: no maturity beyond real synthetic coverage (1M-12M) leaked into the output.\n")

    print("=== Self-test 3: a thin/illiquid expiry is correctly dropped ===")
    # Add one expiry with too few liquid quotes -- should never survive filter_usable_expiries.
    thin_expiry = as_of + pd.Timedelta(days=500)
    thin_rows = pd.DataFrame([
        {"Expiry": thin_expiry, "OptionType": "call", "Strike": 85.0,
         "Bid": 1.0, "Ask": 1.1, "Volume": 1, "OpenInterest": 1},
    ])
    combined_raw = pd.concat([synthetic_raw, thin_rows], ignore_index=True)
    liquid2 = filter_liquid_quotes(combined_raw)
    iv_points2 = compute_iv_points(liquid2, spot, r, as_of)
    usable2 = filter_usable_expiries(iv_points2)
    assert thin_expiry not in usable2["Expiry"].values, "An illiquid expiry (1 quote) should have been dropped"
    print("OK: an expiry with too few liquid quotes is correctly excluded.\n")

    print("=== Self-test 4: commodity mapping ===")
    from greeks_engine import map_underlying_to_commodity
    assert map_underlying_to_commodity("BRN Sep26") == "BRENT"
    assert map_underlying_to_commodity("CL Oct26") == "WTI"
    try:
        map_underlying_to_commodity("XYZ Foo")
        print("FAILED: should have raised on an unrecognized prefix")
    except ValueError:
        print("OK: unrecognized underlying prefix raises cleanly.\n")

    print("All synthetic self-tests passed. NOTE: the live yfinance fetch path "
          "(fetch_raw_option_chain) is NOT exercised by this self-test -- it requires "
          "live network access to Yahoo Finance, which this test suite does not assume.")
