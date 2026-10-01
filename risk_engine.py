"""
risk_engine.py
===============
Generic Value-at-Risk / Expected-Shortfall engine.

Deliberately knows NOTHING about Brent, futures, lot sizes, or contract
months. It only understands two shapes of data:

    returns   : wide DataFrame, index=Date, one column per instrument,
                values = daily price change (or % return -- see method
                arg on compute_returns).
    positions : Series, index=instrument (must match returns.columns),
                values = USD P&L per one unit of that instrument's return
                column (for a future: qty_lots * lot_size, i.e. barrels;
                long = positive, short = negative, same sign convention
                as SignedQty in pnl_engine.py).

Given those two, VaR/ES is pure math -- same functions would work for an
equity book, an FX book, or (once available) delta-equivalent option
exposures. Wiring the actual Brent book into that shape lives at the
bottom of this file, in `build_risk_inputs_from_book()`, kept separate
on purpose so the math above it stays reusable and independently
testable.

Two independent methods are provided, and both must be supplied:

  - HISTORICAL  : full revaluation of the CURRENT position against every
                  day in the historical return sample -- no distributional
                  assumption, whatever shape the tail actually had.
  - PARAMETRIC  : delta-normal method -- assumes returns are ~Normal,
                  uses the sample covariance matrix and a closed-form
                  Normal quantile / Normal tail-mean.

Sign convention: VaR and ES are reported as POSITIVE numbers representing
a loss (e.g. VaR=125,000 means "we expect to lose at most $125,000 at this
confidence level"), even though the underlying P&L distribution has losses
as negative numbers. This matches how VaR is conventionally quoted.

Horizon scaling: both methods accept `horizon_days`. The parametric method
scales analytically (portfolio_std * sqrt(horizon_days), standard under a
random-walk/iid assumption). The historical method scales the ENTIRE 1-day
P&L distribution by sqrt(horizon_days) before taking the percentile/tail
mean -- a simple square-root-of-time shortcut, not a rebuild from actual
overlapping h-day windows. That's a deliberate simplification given the
short dummy-data history available; it slightly understates fat tails that
proper overlapping windows would capture, and should be swapped for a
windowed approach once there's enough history (roughly 250+ days) to make
that statistically meaningful.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm


# ==========================================================================
# 1. Returns
# ==========================================================================

def compute_returns(prices: pd.DataFrame, method: str = "diff") -> pd.DataFrame:
    """
    Wide price DataFrame (Date index, one column per instrument) ->
    wide DataFrame of period-over-period changes, same shape minus the
    first row.

    method:
        "diff"   -- absolute price change (today - yesterday). Default,
                    and the right choice for the Brent book: positions
                    are expressed in barrels, so barrels * price_diff =
                    USD P&L directly, exactly like compute_daily_mtm()
                    in pnl_engine.py.
        "simple" -- percentage return (today / yesterday - 1). Useful if
                    `positions` is later expressed as a USD market value
                    rather than a physical quantity (e.g. equities).
        "log"    -- log return. Slightly nicer statistical properties for
                    the parametric/Normal assumption, same use case as
                    "simple".
    """
    prices = prices.sort_index()
    if method == "diff":
        returns = prices.diff()
    elif method == "simple":
        returns = prices.pct_change()
    elif method == "log":
        returns = np.log(prices / prices.shift(1))
    else:
        raise ValueError(f"Unknown method '{method}'; use 'diff', 'simple', or 'log'.")
    return returns.dropna(how="all").dropna(axis=0, how="any")


# ==========================================================================
# 2. Historical VaR / ES (full revaluation)
# ==========================================================================

def compute_historical_var_es(returns: pd.DataFrame, positions: pd.Series,
                               confidence: float = 0.95, horizon_days: int = 1) -> dict:
    """
    Revalue the CURRENT position against every historical day's return to
    build an empirical P&L distribution, then read VaR/ES off its tail.
    No assumption about the shape of that distribution.

    Returns
    -------
    dict with:
        VaR, ES              -- positive numbers = loss, in the same
                                 currency as positions
        confidence, horizon_days
        pnl_distribution      -- the full (horizon-scaled) historical P&L
                                  series, useful for plotting a histogram
        n_observations         -- how many historical days went into this
    """
    common = [c for c in returns.columns if c in positions.index]
    if not common:
        raise ValueError("No overlapping instruments between returns and positions.")

    r = returns[common]
    p = positions[common]

    daily_pnl = r.mul(p, axis=1).sum(axis=1)          # one USD P&L number per historical day
    scaled_pnl = daily_pnl * np.sqrt(horizon_days)      # sqrt(time) shortcut, see module docstring

    alpha = 1 - confidence
    var = -np.percentile(scaled_pnl, alpha * 100)

    tail_losses = scaled_pnl[scaled_pnl <= -var]
    es = -tail_losses.mean() if len(tail_losses) > 0 else var

    return {
        "Method": "Historical",
        "Confidence": confidence,
        "HorizonDays": horizon_days,
        "VaR": var,
        "ES": es,
        "pnl_distribution": scaled_pnl,
        "n_observations": len(scaled_pnl),
    }


# ==========================================================================
# 3. Parametric (delta-normal) VaR / ES
# ==========================================================================

def compute_covariance(returns: pd.DataFrame) -> pd.DataFrame:
    """Sample covariance matrix of the (1-day) returns, instrument x instrument."""
    return returns.cov()


def compute_parametric_var_es(returns: pd.DataFrame, positions: pd.Series,
                               confidence: float = 0.95, horizon_days: int = 1) -> dict:
    """
    Delta-normal method: assumes the portfolio's daily P&L is ~Normal.

        portfolio_variance = positions^T . Cov(returns) . positions
        VaR = z(confidence) * portfolio_std * sqrt(horizon_days)
        ES  = portfolio_std * sqrt(horizon_days) * phi(z) / (1 - confidence)

    where z = Normal quantile at `confidence` and phi = standard Normal
    pdf. The ES formula is the closed-form Normal tail mean -- no
    simulation needed, unlike the historical method.

    Returns the same dict shape as compute_historical_var_es (minus
    pnl_distribution / n_observations, plus portfolio_std) so both can be
    concatenated easily in risk_summary().
    """
    common = [c for c in returns.columns if c in positions.index]
    if not common:
        raise ValueError("No overlapping instruments between returns and positions.")

    cov = returns[common].cov()
    p = positions[common].values

    portfolio_variance = p @ cov.values @ p.T
    portfolio_std = float(np.sqrt(max(portfolio_variance, 0.0)))

    z = norm.ppf(confidence)
    var = z * portfolio_std * np.sqrt(horizon_days)
    es = portfolio_std * np.sqrt(horizon_days) * norm.pdf(z) / (1 - confidence)

    return {
        "Method": "Parametric",
        "Confidence": confidence,
        "HorizonDays": horizon_days,
        "VaR": var,
        "ES": es,
        "portfolio_std": portfolio_std,
    }


# ==========================================================================
# 4. Summary wrapper -- runs both methods across every requested combo
# ==========================================================================

def risk_summary(returns: pd.DataFrame, positions: pd.Series,
                  confidence_levels=(0.95, 0.99), horizon_days_list=(1, 10)) -> pd.DataFrame:
    """
    Runs both Historical and Parametric VaR/ES across every
    (confidence, horizon) combination requested. Plays the same
    "tie it all together into one tidy table" role that book_summary()
    plays in pnl_engine.py.

    Returns a DataFrame: Method, Confidence, HorizonDays, VaR, ES
    """
    rows = []
    for h in horizon_days_list:
        for c in confidence_levels:
            hist = compute_historical_var_es(returns, positions, confidence=c, horizon_days=h)
            rows.append({k: hist[k] for k in ("Method", "Confidence", "HorizonDays", "VaR", "ES")})

            param = compute_parametric_var_es(returns, positions, confidence=c, horizon_days=h)
            rows.append({k: param[k] for k in ("Method", "Confidence", "HorizonDays", "VaR", "ES")})

    return pd.DataFrame(rows)


# ==========================================================================
# 5. Adapter: wire the generic engine to the actual Brent book
# ==========================================================================

def build_risk_inputs_from_book(trades: pd.DataFrame, settles: pd.DataFrame,
                                 price_method: str = "diff") -> tuple[pd.DataFrame, pd.Series]:
    """
    Turns pnl_engine.py's trades/settles DataFrames into the generic
    (returns, positions) shape this engine expects. This is the ONLY
    function in this file that knows Brent-specific things (contract
    names, lot sizes) -- everything above it is portfolio-agnostic.

    positions are expressed in barrels (qty_lots * lot_size), signed
    long/short, taken from the FIFO open position (same convention as
    pnl_engine.open_position_summary) -- so positions * price_diff =
    USD P&L directly.
    """
    # Local import to avoid a hard dependency between the two engines at
    # module-load time; risk_engine.py stays usable standalone.
    from pnl_engine import compute_fifo_positions, open_position_summary

    open_lots, _ = compute_fifo_positions(trades)
    pos_df = open_position_summary(open_lots)  # Contract, NetQty, AvgEntryPrice
    lot_size_by_contract = trades.groupby("Contract")["LotSize"].first()

    positions = pd.Series(
        {row.Contract: row.NetQty * lot_size_by_contract[row.Contract] for row in pos_df.itertuples()},
        dtype=float,
    )

    prices_wide = settles.pivot(index="Date", columns="Contract", values="SettlePrice").sort_index()
    returns = compute_returns(prices_wide, method=price_method)

    return returns, positions


# ==========================================================================
# Self-test: historical and parametric should broadly agree on synthetic
# Normal data (they're built on different assumptions, so they won't
# match exactly like pnl_engine's two PnL methods do -- but on a Normal
# sample they should land in the same ballpark).
# ==========================================================================
if __name__ == "__main__":
    rng = np.random.default_rng(7)
    n_days = 500
    instruments = ["A", "B", "C"]

    # Correlated synthetic returns (shared factor + idiosyncratic noise),
    # so the covariance matrix isn't trivially diagonal.
    factor = rng.normal(0, 1.0, n_days)
    synth_returns = pd.DataFrame(
        {inst: 0.6 * factor + rng.normal(0, 0.8, n_days) for inst in instruments},
        index=pd.bdate_range("2024-01-01", periods=n_days),
    )
    synth_positions = pd.Series({"A": 1000.0, "B": -400.0, "C": 250.0})

    print("Synthetic self-test (Normal-ish data, 95% / 1-day):")
    hist = compute_historical_var_es(synth_returns, synth_positions, confidence=0.95, horizon_days=1)
    param = compute_parametric_var_es(synth_returns, synth_positions, confidence=0.95, horizon_days=1)
    print(f"  Historical  VaR={hist['VaR']:,.2f}  ES={hist['ES']:,.2f}  (n={hist['n_observations']})")
    print(f"  Parametric  VaR={param['VaR']:,.2f}  ES={param['ES']:,.2f}  (std={param['portfolio_std']:,.2f})")
    rel_diff = abs(hist["VaR"] - param["VaR"]) / param["VaR"]
    print(f"  Relative VaR difference: {rel_diff:.1%} (expect small but nonzero on synthetic Normal data)")
    assert rel_diff < 0.25, "Historical and parametric VaR diverge more than expected on Normal data -- check logic."
    print("  OK: both methods broadly agree on Normal synthetic data.\n")

    print("Full risk_summary() on synthetic book:")
    print(risk_summary(synth_returns, synth_positions).to_string(index=False))

    # Now the real (dummy) Brent book, if the files are present.
    try:
        from pnl_engine import load_trades, load_settlement_prices

        trades = load_trades("trades_dummy.xlsx")
        settles = load_settlement_prices("settlement_prices.csv")
        returns, positions = build_risk_inputs_from_book(trades, settles)

        print(f"\nBrent book positions (barrels): \n{positions.to_string()}")
        print(f"\nReturns sample: {len(returns)} days, columns={list(returns.columns)}")
        print("\nRisk summary on actual (dummy) Brent book:")
        print(risk_summary(returns, positions).to_string(index=False))
    except FileNotFoundError as e:
        print(f"\n(Skipping Brent book test -- data file not found: {e})")