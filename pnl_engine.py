"""
pnl_engine.py
=============
Core PnL engine for a book of ICE Brent Crude futures trades.

--------------------------------------------------------------------------
How futures PnL actually works (quick refresher, since it drives the code)
--------------------------------------------------------------------------
Futures are marked-to-market every day by the exchange. That means your
daily cash PnL on a position you're still holding is NOT "current price
minus what I paid" -- it's "today's settle minus yesterday's settle",
because yesterday's move was already settled in cash. For a trade done
*today*, the first day's PnL is "today's settle minus my trade price".

So for a given contract, on day t:

    daily_pnl(t) = position(t-1) * lot_size * (settle(t) - settle(t-1))
                 + sum over today's trades: signed_qty * lot_size * (settle(t) - trade_price)

Summed over every day since inception, this equals total book PnL and
requires no separate "realized vs unrealized" split -- it's just cash.
That's `compute_daily_mtm()` below.

Separately, it's useful to know how much of your *current* PnL is already
locked in (realized) vs still sitting in open positions (unrealized). For
that we replay the trades through a FIFO lot matcher -- the same logic a
back-office P&L system uses. That's `compute_fifo_positions()` below.

Both methods must agree on total PnL at any snapshot date -- there's a
consistency check for that at the bottom of this file.
--------------------------------------------------------------------------

Public functions
-----------------
load_trades(path)                          -> DataFrame of trades
load_settlement_prices(path)               -> DataFrame of daily settle prices
compute_fifo_positions(trades)             -> open lots + realized PnL events
compute_daily_mtm(trades, settles)         -> daily & cumulative PnL per contract
compute_live_pnl(trades, settles, live_prices, as_of) -> on-request live PnL
book_summary(trades, settles, live_prices, as_of)     -> one-line-per-contract snapshot
"""

from collections import deque
from dataclasses import dataclass

import numpy as np
import pandas as pd

LOT_SIZE_DEFAULT = 1000  # barrels per ICE Brent futures lot


# ==========================================================================
# 1. Loading data
# ==========================================================================

def load_trades(path: str) -> pd.DataFrame:
    """Read the trade blotter (Excel) and normalize dtypes/sign convention."""
    df = pd.read_excel(path, sheet_name="Trades")
    df["TradeDate"] = pd.to_datetime(df["TradeDate"])
    # SignedQty > 0 for a BUY (long), < 0 for a SELL (short). Using a signed
    # quantity once here means every formula downstream is direction-agnostic.
    df["SignedQty"] = np.where(df["BuySell"].str.upper() == "BUY", df["Quantity"], -df["Quantity"])
    return df.sort_values("TradeDate").reset_index(drop=True)


def load_settlement_prices(path: str) -> pd.DataFrame:
    """Read the daily settlement price feed."""
    df = pd.read_csv(path, parse_dates=["Date"])
    return df.sort_values(["Contract", "Date"]).reset_index(drop=True)


# ==========================================================================
# 2. FIFO position & realized PnL
# ==========================================================================

@dataclass
class Lot:
    trade_id: str
    trade_date: pd.Timestamp
    qty: float      # signed, remaining quantity in this lot
    price: float


def compute_fifo_positions(trades: pd.DataFrame):
    """
    Replay every trade, contract by contract, matching closing trades
    against the OLDEST still-open lot first (FIFO).

    Returns
    -------
    open_lots : dict[contract] -> list[Lot]   (what's still on the book)
    realized_events : DataFrame              (one row per matched close)
    """
    open_lots: dict[str, deque] = {}
    realized_events = []

    for _, t in trades.iterrows():
        contract = t["Contract"]
        book = open_lots.setdefault(contract, deque())
        remaining = t["SignedQty"]
        price = t["Price"]

        # Match against existing lots only if they are on the OPPOSITE side
        # (a SELL closes a long book, a BUY closes a short book).
        while remaining != 0 and book and (book[0].qty * remaining < 0):
            lot = book[0]
            closed_qty = min(abs(lot.qty), abs(remaining)) * (1 if remaining > 0 else -1)
            # closed_qty carries the sign of the CLOSING trade

            pnl = (price - lot.price) * (-closed_qty) * t["LotSize"]
            # (-closed_qty) because "closing a long" means we're effectively
            # selling `abs(closed_qty)` of that lot -> pnl = (exit-entry)*qty_sold

            realized_events.append({
                "Contract": contract,
                "CloseDate": t["TradeDate"],
                "ClosingTradeID": t["TradeID"],
                "OpeningTradeID": lot.trade_id,
                "QtyClosed": abs(closed_qty),
                "EntryPrice": lot.price,
                "ExitPrice": price,
                "RealizedPnL": pnl,
            })

            lot.qty += closed_qty
            remaining -= closed_qty
            if lot.qty == 0:
                book.popleft()

        if remaining != 0:
            # Either the book was flat, or we've fully closed the opposite
            # side and still have quantity left over -> opens a new lot.
            book.append(Lot(t["TradeID"], t["TradeDate"], remaining, price))

    realized_df = pd.DataFrame(realized_events)
    return open_lots, realized_df


def open_position_summary(open_lots: dict) -> pd.DataFrame:
    """Collapse open lots into one row per contract: net qty & avg cost."""
    rows = []
    for contract, lots in open_lots.items():
        if not lots:
            continue
        net_qty = sum(l.qty for l in lots)
        avg_price = sum(l.qty * l.price for l in lots) / net_qty if net_qty else 0
        rows.append({"Contract": contract, "NetQty": net_qty, "AvgEntryPrice": avg_price})
    return pd.DataFrame(rows)


# ==========================================================================
# 3. Daily mark-to-market
# ==========================================================================

def compute_daily_mtm(trades: pd.DataFrame, settles: pd.DataFrame) -> pd.DataFrame:
    """
    Daily variation-margin style PnL per contract, per day, from each
    contract's first trade date up to the last settlement date available.

    Returns a tidy DataFrame:
        Date, Contract, Position, SettlePrice, DailyPnL, CumulativePnL
    """
    all_rows = []

    for contract, ctrades in trades.groupby("Contract"):
        cprices = settles[settles["Contract"] == contract].sort_values("Date")
        lot_size = ctrades["LotSize"].iloc[0]

        position = 0.0
        prev_settle = None
        cum_pnl = 0.0

        for _, day in cprices.iterrows():
            d = day["Date"]
            settle = day["SettlePrice"]

            # PnL earned overnight on the position we already had
            carry_pnl = 0.0
            if prev_settle is not None:
                carry_pnl = position * lot_size * (settle - prev_settle)

            # PnL on any trades executed *today*, from trade price to today's close
            todays_trades = ctrades[ctrades["TradeDate"] == d]
            trade_pnl = (todays_trades["SignedQty"] * lot_size * (settle - todays_trades["Price"])).sum()

            # Update running position with today's trades
            position += todays_trades["SignedQty"].sum()

            daily_pnl = carry_pnl + trade_pnl
            cum_pnl += daily_pnl

            all_rows.append({
                "Date": d,
                "Contract": contract,
                "Position": position,
                "SettlePrice": settle,
                "DailyPnL": daily_pnl,
                "CumulativePnL": cum_pnl,
            })

            prev_settle = settle

    return pd.DataFrame(all_rows).sort_values(["Contract", "Date"]).reset_index(drop=True)


# ==========================================================================
# 4. Live / on-request PnL
# ==========================================================================

def compute_live_pnl(trades: pd.DataFrame, live_prices: dict) -> pd.DataFrame:
    """
    Total PnL "right now", given a dict of live prices {contract: price}.

    This is deliberately built the same way as the RealizedPnL/UnrealizedPnL
    columns in book_summary() -- FIFO realized PnL, plus the still-open
    position revalued at a price you supply -- just using a live price
    instead of waiting for the next official settlement.

    Because it's built the same way, LivePnL and (RealizedPnL + Unrealized-
    PnL) from book_summary() will match EXACTLY whenever live_price equals
    the last settlement price -- that equality is a good sanity check.

    (An earlier version of this function tried to compute only the
    *incremental* PnL since last night's settle, by looking up "the most
    recent settlement price before today". That broke as soon as "today"
    drifted away from the book's last trade date -- which is basically
    always -- because it silently picked up a stale, days-old settle price
    instead of the actual last one. Pricing the whole open position from
    its FIFO average cost, like this version does, has no such date
    dependency.)

    Parameters
    ----------
    live_prices : dict[str, float]   e.g. {"BRN Sep26": 84.02, ...}
    """
    open_lots, realized_df = compute_fifo_positions(trades)
    positions = open_position_summary(open_lots)  # Contract, NetQty, AvgEntryPrice

    realized_by_contract = (
        realized_df.groupby("Contract")["RealizedPnL"].sum()
        if not realized_df.empty else pd.Series(dtype=float)
    )
    lot_size_by_contract = trades.groupby("Contract")["LotSize"].first()

    rows = []
    for contract, live_price in live_prices.items():
        pos_row = positions[positions["Contract"] == contract]
        net_qty = pos_row["NetQty"].iloc[0] if not pos_row.empty else 0.0
        avg_entry = pos_row["AvgEntryPrice"].iloc[0] if not pos_row.empty else None
        lot_size = lot_size_by_contract.get(contract, LOT_SIZE_DEFAULT)
        realized = realized_by_contract.get(contract, 0.0)

        unrealized_live = net_qty * lot_size * (live_price - avg_entry) if avg_entry is not None else 0.0

        rows.append({
            "Contract": contract,
            "LivePrice": live_price,
            "Position": net_qty,
            "AvgEntryPrice": avg_entry,
            "RealizedPnL": realized,
            "UnrealizedPnL": unrealized_live,
            "LivePnL": realized + unrealized_live,
        })

    return pd.DataFrame(rows)


# ==========================================================================
# 5. Book-level snapshot (combines everything above)
# ==========================================================================

def book_summary(trades: pd.DataFrame, settles: pd.DataFrame,
                  live_prices: dict = None, as_of=None) -> pd.DataFrame:
    """
    One row per contract: net position, avg entry, realized PnL to date,
    unrealized PnL at last settle, and (if live_prices given) live PnL.

    as_of : optional date (str or pd.Timestamp). If given, the book is
            valued as of that date: only trades on or before `as_of` count
            towards position/realized PnL, and "SettlePrice" is the most
            recent settlement price on or before `as_of` (not necessarily
            the very latest one in `settles`). Leave as None to use every
            trade and the latest available settlement, as before.
    """
    if as_of is not None:
        as_of = pd.Timestamp(as_of)
        trades = trades[trades["TradeDate"] <= as_of]
        settles = settles[settles["Date"] <= as_of]

    if trades.empty:
        # Nothing had traded yet as of this date -- return an empty, but
        # correctly-shaped, summary rather than letting the merges below
        # raise a confusing KeyError.
        cols = ["Contract", "Position", "AvgEntryPrice", "SettlePrice", "RealizedPnL",
                "UnrealizedPnL", "CumulativePnL"]
        if live_prices:
            cols += ["LivePrice", "LivePnL"]
        return pd.DataFrame(columns=cols)

    open_lots, realized_df = compute_fifo_positions(trades)
    positions = open_position_summary(open_lots)

    mtm = compute_daily_mtm(trades, settles)
    last_settle_row = mtm.sort_values("Date").groupby("Contract").last().reset_index()

    realized_by_contract = (
        realized_df.groupby("Contract")["RealizedPnL"].sum().reset_index()
        if not realized_df.empty else pd.DataFrame(columns=["Contract", "RealizedPnL"])
    )

    # `last_settle_row.Position` (running sum from compute_daily_mtm) and
    # `positions.NetQty` (FIFO open quantity) are computed two independent
    # ways and should always match -- keep only one, but this equality is a
    # useful sanity check while debugging.
    last_settle_row = last_settle_row.drop(columns=["Position"])

    summary = last_settle_row.merge(realized_by_contract, on="Contract", how="left")
    summary = summary.merge(positions, on="Contract", how="left")
    summary = summary.rename(columns={"NetQty": "Position"})
    summary["RealizedPnL"] = summary["RealizedPnL"].fillna(0.0)
    summary["UnrealizedPnL"] = summary["CumulativePnL"] - summary["RealizedPnL"]

    if live_prices:
        live = compute_live_pnl(trades, live_prices)
        summary = summary.merge(live[["Contract", "LivePrice", "LivePnL"]], on="Contract", how="left")

    cols = ["Contract", "Position", "AvgEntryPrice", "SettlePrice", "RealizedPnL",
            "UnrealizedPnL", "CumulativePnL"]
    if live_prices:
        cols += ["LivePrice", "LivePnL"]
    return summary[[c for c in cols if c in summary.columns]]


# ==========================================================================
# Self-test: the two PnL methods must agree
# ==========================================================================
if __name__ == "__main__":
    trades = load_trades("trades_dummy.xlsx")
    settles = load_settlement_prices("settlement_prices.csv")

    open_lots, realized_df = compute_fifo_positions(trades)
    total_realized = realized_df["RealizedPnL"].sum()

    mtm = compute_daily_mtm(trades, settles)
    total_mtm_pnl = mtm.groupby("Contract")["DailyPnL"].sum().sum()

    positions = open_position_summary(open_lots)
    last_prices = settles.sort_values("Date").groupby("Contract").last()["SettlePrice"]
    total_unrealized = sum(
        row.NetQty * LOT_SIZE_DEFAULT * (last_prices[row.Contract] - row.AvgEntryPrice)
        for row in positions.itertuples()
    )

    print(f"Total realized PnL (FIFO):           {total_realized:,.2f} USD")
    print(f"Total unrealized PnL (FIFO, open):   {total_unrealized:,.2f} USD")
    print(f"Realized + Unrealized:               {total_realized + total_unrealized:,.2f} USD")
    print(f"Total PnL (daily MTM cash method):   {total_mtm_pnl:,.2f} USD")
    diff = abs((total_realized + total_unrealized) - total_mtm_pnl)
    print(f"Difference (should be ~0):           {diff:,.6f} USD")
    assert diff < 1e-6, "PnL methods disagree -- bug!"
    print("\nOK: both PnL methods agree.")

    print("\nBook summary:")
    print(book_summary(trades, settles, live_prices={
        "BRN Sep26": 84.20, "BRN Oct26": 83.10, "BRN Dec26": 81.75
    }).to_string(index=False))

    # Sanity check: if the "live" price you feed in equals the last settle
    # price exactly, LivePnL must equal RealizedPnL + UnrealizedPnL (i.e.
    # today's total book PnL) -- since you haven't actually changed the
    # price you're marking at.
    live_at_settle = last_prices.to_dict()
    check = book_summary(trades, settles, live_prices=live_at_settle)
    check_diff = (check["LivePnL"] - check["CumulativePnL"]).abs().max()
    print(f"\nLive-PnL-equals-settle-PnL check (should be ~0): {check_diff:,.6f} USD")
    assert check_diff < 1e-6, "LivePnL should match total book PnL when live price == settle price!"
    print("OK: live PnL matches total book PnL when priced at the last settle.")