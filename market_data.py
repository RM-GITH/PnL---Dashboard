"""
market_data.py
===============
Fetches Brent futures settlement prices from a real source instead of the
simulated random walk in generate_dummy_data.py.

IMPORTANT — read this before trusting the numbers
---------------------------------------------------
Official ICE settlement prices, broken out PER CONTRACT MONTH (Sep26 vs
Oct26 vs Dec26 each having their own distinct daily settle), are licensed
data. You don't get that for free from anywhere. What a real Futures
Middle Office desk uses is one of:

  - ICE Data Services (End-of-Day / Settlement Data feed)
  - Bloomberg (via `blpapi`, ticker e.g. "COU26 Comdty" for Brent Oct26)
  - Refinitiv Eikon / Workspace API

Those require a paid subscription tied to your firm, so I can't call them
for you here. What I *can* wire up for free is Yahoo Finance's Brent
futures ticker "BZ=F" — but that's a single "front month, continuously
rolled" series, not one series per expiry. So this module broadcasts that
one real market price to every contract in your book. It's genuinely real
market data (not simulated), just not expiry-specific — good enough to
demo the pipeline end-to-end, not accurate enough to trade off.

If/when you get access to a real per-contract feed (at work, presumably),
replace `fetch_brent_settlements_yfinance()` below with a function that
calls that feed instead — as long as it returns a DataFrame with columns
[Date, Contract, SettlePrice], nothing else in the project needs to change.

Usage
-----
    python market_data.py
        -> overwrites settlement_prices.csv with real Yahoo Finance data

    python market_data.py --keep-dummy
        -> just prints what it *would* fetch, without touching the CSV
"""

import argparse
import sys
import warnings

import pandas as pd

try:
    import yfinance as yf
except ImportError:
    yf = None


def fetch_price_history(tickers: list[str], start_date, end_date=None) -> pd.DataFrame:
    """
    Generic multi-ticker fetch from Yahoo Finance. Not Brent-specific --
    this is the building block both the Brent-broadcast function below and
    risk_engine.py's live data path use.

    Returns a WIDE DataFrame: Date index, one column per ticker, values =
    daily close price. Missing days (holidays not shared across tickers,
    etc.) come back as NaN -- caller decides how to handle alignment.
    """
    if yf is None:
        raise ImportError("yfinance is not installed. Run: pip install yfinance")

    tickers = sorted(set(tickers))
    raw = yf.download(tickers, start=start_date, end=end_date, progress=False)
    if raw.empty:
        raise RuntimeError(
            f"yfinance returned no data for {tickers}. Check your network connection, "
            "the ticker symbol(s), or try again (Yahoo occasionally rate-limits)."
        )

    close = raw["Close"]
    # A single ticker can come back as a Series rather than a DataFrame,
    # depending on yfinance version -- normalize to DataFrame either way.
    if isinstance(close, pd.Series):
        close = close.to_frame(name=tickers[0])

    return close.sort_index()


def fetch_brent_settlements_yfinance(contracts: list[str], start_date, end_date=None,
                                      ticker: str = "BZ=F") -> pd.DataFrame:
    """
    Pull the real daily close for Yahoo Finance's Brent futures front-month
    proxy, and broadcast it to every contract name in `contracts` (since
    Yahoo doesn't expose individual expiries for ICE Brent).

    Returns a DataFrame shaped exactly like settlement_prices.csv:
        Date, Contract, SettlePrice
    """
    wide = fetch_price_history([ticker], start_date, end_date)
    close = wide[ticker].rename("SettlePrice").reset_index()
    close = close.rename(columns={close.columns[0]: "Date"})

    frames = []
    for contract in contracts:
        c = close.copy()
        c["Contract"] = contract
        frames.append(c)

    out = pd.concat(frames, ignore_index=True)
    return out[["Date", "Contract", "SettlePrice"]].sort_values(["Contract", "Date"]).reset_index(drop=True)


def get_settlement_prices(contract_ticker_map: dict, start_date, end_date=None,
                           csv_fallback_path: str = "settlement_prices.csv") -> pd.DataFrame:
    """
    THE function the dashboard should call for settlement prices.

    Tries a live Yahoo Finance fetch first, one ticker per contract as
    given in `contract_ticker_map` (e.g. {"BRN Sep26": "BZ=F",
    "BRN Oct26": "BZ=F", "BRN Dec26": "BZ=F"} -- reusing the same ticker
    per contract is the current Brent limitation described at the top of
    this file; swap in real per-expiry tickers here once available,
    nothing downstream needs to change).

    Falls back to the static CSV (same [Date, Contract, SettlePrice]
    shape) if the live fetch fails for ANY reason: yfinance not
    installed, no network, empty response, bad ticker, etc. Never raises
    on a live-fetch failure -- only raises if the fallback CSV is also
    unavailable.

    Returns [Date, Contract, SettlePrice, Source] where Source is
    "live" or "csv_fallback", so the dashboard can show the user which
    one they're actually looking at.
    """
    try:
        wide = fetch_price_history(list(contract_ticker_map.values()), start_date, end_date)
        rows = []
        for contract, ticker in contract_ticker_map.items():
            if ticker not in wide.columns:
                continue
            s = wide[ticker].dropna()
            for d, p in s.items():
                rows.append({"Date": d, "Contract": contract, "SettlePrice": float(p)})
        if not rows:
            raise RuntimeError("live fetch returned no usable rows for the requested contracts")
        out = pd.DataFrame(rows).sort_values(["Contract", "Date"]).reset_index(drop=True)
        out["Source"] = "live"
        return out

    except Exception as e:
        warnings.warn(
            f"Live price fetch failed ({e}); falling back to '{csv_fallback_path}'.",
            RuntimeWarning,
        )
        df = pd.read_csv(csv_fallback_path, parse_dates=["Date"])
        df["Source"] = "csv_fallback"
        return df.sort_values(["Contract", "Date"]).reset_index(drop=True)


def fetch_brent_settlements_institutional(contracts: list[str], start_date, end_date=None) -> pd.DataFrame:
    """
    STUB — fill this in once you have access to a real per-contract feed
    (ICE Data, Bloomberg, Refinitiv). Left unimplemented on purpose since it
    needs credentials I don't have. Example shape for a Bloomberg pull via
    `blpapi`/`xbbg`, for reference:

        from xbbg import blp
        # Brent contract month codes: F=Jan,G=Feb,H=Mar,J=Apr,K=May,M=Jun,
        # N=Jul,Q=Aug,U=Sep,V=Oct,X=Nov,Z=Dec
        bbg_tickers = {"BRN Sep26": "COU26 Comdty", "BRN Oct26": "COV26 Comdty", ...}
        df = blp.bdh(list(bbg_tickers.values()), "PX_SETTLE", start_date, end_date)
        # ... then reshape df into [Date, Contract, SettlePrice] and return it

    Must return a DataFrame with columns: Date, Contract, SettlePrice.
    """
    raise NotImplementedError(
        "Plug in your firm's real data feed here (ICE Data / Bloomberg / Refinitiv)."
    )


def main():
    parser = argparse.ArgumentParser(description="Fetch real Brent settlement prices.")
    parser.add_argument("--keep-dummy", action="store_true",
                         help="Print what would be fetched without overwriting settlement_prices.csv")
    args = parser.parse_args()

    trades = pd.read_excel("trades_dummy.xlsx", sheet_name="Trades")
    trades["TradeDate"] = pd.to_datetime(trades["TradeDate"])
    contracts = sorted(trades["Contract"].unique())
    start_date = trades["TradeDate"].min()

    print(f"Fetching real Brent (BZ=F) settlement prices from {start_date.date()} to today...")
    try:
        settlements = fetch_brent_settlements_yfinance(contracts, start_date)
    except Exception as e:
        print(f"Fetch failed: {e}", file=sys.stderr)
        sys.exit(1)

    print(settlements.groupby("Contract").agg(
        first_date=("Date", "min"), last_date=("Date", "max"),
        n_days=("Date", "count"), last_price=("SettlePrice", "last"),
    ))

    if args.keep_dummy:
        print("\n--keep-dummy set: settlement_prices.csv left untouched.")
    else:
        settlements.to_csv("settlement_prices.csv", index=False)
        print("\nWrote real settlement prices to settlement_prices.csv")


if __name__ == "__main__":
    main()