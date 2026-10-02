"""
dashboard.py
============
Streamlit dashboard for the Brent futures book. Run with:

    streamlit run dashboard.py

Tabs:
  1. Portfolio Analysis - book summary, daily MTM chart, live PnL
  2. Risk Metrics        - Historical and Parametric VaR / Expected
                            Shortfall at 95% and 99%, toggleable 1-day /
                            10-day horizon
  3. Greeks               - placeholder; full option book + aggregated
                            portfolio Greeks (Black-76) lands in Phase 3

Visual design: a dense, dark trading-terminal aesthetic (in the spirit of
Refinitiv Workspace / Bloomberg) -- monospace tabular numerals throughout,
hairline borders instead of card shadows, colour used only as a data
signal (green/red for P&L sign, amber for active/live state), never as
decoration. See TERMINAL_CSS below for the token system.
"""

import datetime as dt
import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from pnl_engine import (
    load_trades,
    compute_daily_mtm,
    book_summary,
    compute_fifo_positions,
    open_position_summary,
)
from market_data import get_settlement_prices
from market_vol_surface import build_market_vol_surface
from risk_engine import (
    build_risk_inputs_from_book,
    compute_historical_var_es,
    compute_parametric_var_es,
    risk_summary,
)
from greeks_engine import (
    load_option_trades,
    load_volatility_surface,
    compute_option_fifo_positions,
    compute_live_option_greeks,
    portfolio_greeks_summary,
    black76_greeks,
    lookup_implied_vol,
    resolve_surface_underlying,
    RISK_FREE_RATE_DEFAULT,
    VOL_SURFACE_STALENESS_WARNING_DAYS,
)

st.set_page_config(page_title="Brent Futures Book — PnL & Risk", layout="wide")

# ==========================================================================
# Design tokens (terminal aesthetic)
# ==========================================================================
COLOR = {
    "bg_base": "#0B0E14",
    "bg_panel": "#12161F",
    "bg_panel_alt": "#161B26",
    "border": "#232936",
    "text_primary": "#E4E7EC",
    "text_secondary": "#8891A1",
    "accent": "#FF7A1A",
    "positive": "#12B886",
    "negative": "#F0555A",
}

TERMINAL_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

html, body, [class*="css"], .stMarkdown, .stText {{
    font-family: 'IBM Plex Sans', -apple-system, sans-serif;
}}

.stApp {{
    background-color: {COLOR["bg_base"]};
    color: {COLOR["text_primary"]};
}}

h1, h2, h3 {{
    font-family: 'IBM Plex Sans', sans-serif;
    font-weight: 600;
    color: {COLOR["text_primary"]};
}}

/* Sidebar */
section[data-testid="stSidebar"] {{
    background-color: {COLOR["bg_panel"]};
    border-right: 1px solid {COLOR["border"]};
}}
section[data-testid="stSidebar"] .stMarkdown p {{
    color: {COLOR["text_secondary"]};
    font-size: 12.5px;
}}

/* Status bar */
.term-status-bar {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 7px 14px;
    background-color: {COLOR["bg_panel"]};
    border: 1px solid {COLOR["border"]};
    border-radius: 2px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px;
    color: {COLOR["text_secondary"]};
    margin-bottom: 14px;
}}
.term-status-bar .dot {{
    display: inline-block; width: 7px; height: 7px; border-radius: 50%;
    margin-right: 7px; position: relative; top: -1px;
}}
.term-status-bar .dot.live {{ background-color: {COLOR["positive"]}; box-shadow: 0 0 5px {COLOR["positive"]}; }}
.term-status-bar .dot.fallback {{ background-color: {COLOR["accent"]}; box-shadow: 0 0 5px {COLOR["accent"]}; }}

/* Watchlist rows (sidebar) */
.term-panel-title {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: {COLOR["text_secondary"]};
    border-bottom: 1px solid {COLOR["border"]};
    padding-bottom: 6px;
    margin: 14px 0 6px 0;
}}
.term-watchlist-row {{
    display: flex; justify-content: space-between; align-items: baseline;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12.5px;
    padding: 6px 2px;
    border-bottom: 1px solid {COLOR["border"]};
}}
.term-watchlist-row .tkr {{ color: {COLOR["text_primary"]}; font-weight: 500; }}
.term-watchlist-row .px {{ color: {COLOR["text_primary"]}; }}
.term-watchlist-row .chg {{ min-width: 62px; text-align: right; }}
.term-watchlist-row .chg.pos {{ color: {COLOR["positive"]}; }}
.term-watchlist-row .chg.neg {{ color: {COLOR["negative"]}; }}
.term-watchlist-row .chg.flat {{ color: {COLOR["text_secondary"]}; }}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {{
    gap: 4px;
    border-bottom: 1px solid {COLOR["border"]};
}}
.stTabs [data-baseweb="tab"] {{
    background-color: transparent;
    color: {COLOR["text_secondary"]};
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12.5px;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    padding: 10px 16px;
    border-bottom: 2px solid transparent;
}}
.stTabs [aria-selected="true"] {{
    color: {COLOR["text_primary"]} !important;
    border-bottom: 2px solid {COLOR["accent"]} !important;
}}

/* Metrics */
[data-testid="stMetric"] {{
    background-color: {COLOR["bg_panel"]};
    border: 1px solid {COLOR["border"]};
    border-radius: 2px;
    padding: 12px 14px 10px 14px;
}}
[data-testid="stMetricLabel"] {{
    color: {COLOR["text_secondary"]};
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}}
[data-testid="stMetricValue"] {{
    font-family: 'IBM Plex Mono', monospace;
    font-variant-numeric: tabular-nums;
    color: {COLOR["text_primary"]};
}}

/* DataFrames */
[data-testid="stDataFrame"] {{
    font-family: 'IBM Plex Mono', monospace;
}}

/* Expanders */
[data-testid="stExpander"] {{
    background-color: {COLOR["bg_panel"]};
    border: 1px solid {COLOR["border"]};
    border-radius: 2px;
}}

hr {{ border-color: {COLOR["border"]}; }}
</style>
"""
st.markdown(TERMINAL_CSS, unsafe_allow_html=True)


def apply_terminal_theme(fig: go.Figure) -> go.Figure:
    """
    Re-theme a Plotly figure to match the terminal palette. Uses
    update_xaxes/update_yaxes (rather than the layout.xaxis/yaxis dict)
    so this also themes every panel of a multi-subplot figure, not just
    a single-axis one.
    """
    fig.update_layout(
        paper_bgcolor=COLOR["bg_base"],
        plot_bgcolor=COLOR["bg_base"],
        font=dict(family="IBM Plex Mono, monospace", color=COLOR["text_secondary"], size=12),
        colorway=[COLOR["accent"], COLOR["positive"], "#4C9AFF", COLOR["negative"], "#B983FF", "#FFC53D"],
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=COLOR["text_secondary"])),
        margin=dict(t=40, b=20),
    )
    fig.update_xaxes(gridcolor=COLOR["border"], zerolinecolor=COLOR["border"], linecolor=COLOR["border"])
    fig.update_yaxes(gridcolor=COLOR["border"], zerolinecolor=COLOR["border"], linecolor=COLOR["border"])
    return fig


def style_signed_table(df: pd.DataFrame, signed_cols: list) -> "pd.io.formats.style.Styler":
    """Colour-code sign-varying P&L columns (green=profit, red=loss) and set the table to monospace tabular numerals."""
    def _color(val):
        try:
            v = float(val)
        except (TypeError, ValueError):
            return ""
        if v > 0:
            return f"color: {COLOR['positive']};"
        if v < 0:
            return f"color: {COLOR['negative']};"
        return f"color: {COLOR['text_secondary']};"

    styler = df.style
    for col in signed_cols:
        if col in df.columns:
            styler = styler.map(_color, subset=[col])
    styler = styler.set_properties(**{"font-family": "'IBM Plex Mono', monospace", "font-size": "12.5px"})
    styler = styler.set_table_styles([
        {"selector": "th", "props": [("font-family", "'IBM Plex Mono', monospace"),
                                      ("font-size", "11px"), ("text-transform", "uppercase"),
                                      ("letter-spacing", "0.04em"), ("color", COLOR["text_secondary"])]}
    ])
    return styler


def style_risk_table(df: pd.DataFrame, magnitude_cols: list) -> "pd.io.formats.style.Styler":
    """VaR/ES are always-positive loss magnitudes, not sign-varying P&L -- colour them uniformly as risk (amber), not green/red."""
    styler = df.style.set_properties(
        subset=[c for c in magnitude_cols if c in df.columns],
        **{"color": COLOR["accent"], "font-weight": "600"},
    )
    styler = styler.set_properties(**{"font-family": "'IBM Plex Mono', monospace", "font-size": "12.5px"})
    styler = styler.set_table_styles([
        {"selector": "th", "props": [("font-family", "'IBM Plex Mono', monospace"),
                                      ("font-size", "11px"), ("text-transform", "uppercase"),
                                      ("letter-spacing", "0.04em"), ("color", COLOR["text_secondary"])]}
    ])
    return styler


# ==========================================================================
# Data loading (cached so it doesn't re-fetch/re-read on every interaction)
# ==========================================================================
# NOTE on file naming: the dashboard reads trades.xlsx / options_trades.xlsx --
# these are the REAL, user-maintained blotters (edit them in Excel, commit,
# push). They are intentionally separate from trades_dummy.xlsx /
# options_trades_dummy.xlsx, which are generate_dummy_data.py's sandbox
# output -- re-running that script regenerates the _dummy files only and
# will never touch your real trades.xlsx / options_trades.xlsx. See
# README.md for the full workflow.
@st.cache_data(ttl=900)  # 15 min: fresh enough to pick up a live price, gentle enough on Yahoo
def get_data():
    trades = load_trades("trades.xlsx")
    contracts = sorted(trades["Contract"].unique())
    start_date = trades["TradeDate"].min()

    # All contracts currently map to the same Yahoo front-month proxy
    # ticker -- see the limitation explained at the top of market_data.py.
    # Swap in real per-expiry tickers here once available.
    contract_ticker_map = {c: "BZ=F" for c in contracts}
    settles = get_settlement_prices(
        contract_ticker_map, start_date=start_date, csv_fallback_path="settlement_prices.csv"
    )
    return trades, settles

BENCHMARK_TICKERS = {"Brent (BZ=F)": "BZ=F", "WTI (CL=F)": "CL=F", "TTF Gas (TTF=F)": "TTF=F"}


@st.cache_data(ttl=1800)  # 30 min: cheap (one batched yfinance call), fine to refresh often
def get_benchmark_prices(lookback_days: int = 365):
    """
    Live daily close history for Brent, WTI, and TTF via market_data.py's
    generic fetch_price_history -- same function, reused as-is, just with
    three tickers instead of one. No offline fallback exists for WTI/TTF
    (unlike settlement_prices.csv, which only ever covered the Brent
    dummy book) -- a live-fetch failure here returns an empty frame and
    a clear "unavailable" status rather than fabricating one.
    """
    from market_data import fetch_price_history
    start = (pd.Timestamp.now() - pd.Timedelta(days=lookback_days)).normalize()
    try:
        wide = fetch_price_history(list(BENCHMARK_TICKERS.values()), start_date=start)
        wide = wide.rename(columns={v: k for k, v in BENCHMARK_TICKERS.items()})
        return wide, "live"
    except Exception as e:
        warnings.warn(f"Live benchmark price fetch failed: {e}", RuntimeWarning)
        return pd.DataFrame(), "unavailable"


@st.cache_data(ttl=86400)  # once daily: a vol surface build makes ~30-40 option-chain calls to Yahoo, and doesn't need refreshing more often than that
def get_market_vol_surface():
    """
    Tries to build a REAL market-calibrated vol surface from BNO (Brent
    proxy) and USO (WTI proxy) listed options -- see market_vol_surface.py
    for the full method. Falls back to the static dummy surface if the
    live build fails for ANY reason (no network, no liquid expiries,
    yfinance error) -- never raises, same philosophy as
    get_settlement_prices() in market_data.py.
    """
    surfaces = []
    live_built = set()
    for ticker, commodity in (("BNO", "BRENT"), ("USO", "WTI")):
        try:
            s = build_market_vol_surface(ticker, commodity)
            if s.empty:
                raise RuntimeError("zero usable points after filtering/interpolation")
            s["Source"] = "market"
            surfaces.append(s)
            live_built.add(commodity)
        except Exception as e:
            # Independent per-commodity failure: a BNO (Brent) failure
            # must not discard a successful USO (WTI) fetch, or vice versa.
            warnings.warn(f"Live market vol surface build failed for {commodity} ({ticker}): {e}", RuntimeWarning)

    if "BRENT" not in live_built:
        # The dummy data only ever covered Brent (BRN Sep26/Oct26/Dec26) --
        # it's a real, honest fallback here. Keep it in its original
        # contract-month naming; the options book's positions ARE named
        # "BRN Sep26" etc., so this matches directly with no relabeling.
        warnings.warn("Falling back to volatility_surface_dummy.csv for BRENT.", RuntimeWarning)
        dummy = load_volatility_surface("volatility_surface_dummy.csv")
        dummy["Source"] = "dummy_fallback"
        surfaces.append(dummy)
    if "WTI" not in live_built:
        # No dummy WTI data exists to fall back to -- there are currently
        # no WTI positions in the options book, so this is silent-safe
        # today, but would surface as a clear lookup error the moment a
        # WTI option trade is actually added without live data available.
        warnings.warn("No live WTI vol surface available, and no dummy WTI fallback exists.", RuntimeWarning)

    if not surfaces:
        raise RuntimeError("No volatility surface data available at all (live fetch failed, no fallback exists).")

    return pd.concat(surfaces, ignore_index=True)


@st.cache_data(ttl=900)
def get_options_data():
    option_trades = load_option_trades("options_trades.xlsx")
    vol_surface = get_market_vol_surface()
    return option_trades, vol_surface


trades_all, settles_all = get_data()

# --------------------------------------------------------------------------
# As Of Date — lets the user view the book's state at any date between the
# first trade and the latest available settlement, not just "today". Every
# tab below reads from `trades`/`settles`, which are TRUNCATED to this date
# (any trade or settlement after it is excluded), so FIFO positions,
# realized P&L, VaR, and Greeks all recompute as of that point in time.
# --------------------------------------------------------------------------
min_as_of = trades_all["TradeDate"].min().date()
max_as_of = settles_all["Date"].max().date()

st.sidebar.markdown('<div class="term-panel-title">As of date</div>', unsafe_allow_html=True)
as_of_selected = st.sidebar.date_input(
    "Portfolio state as of", value=max_as_of, min_value=min_as_of, max_value=max_as_of,
)
as_of_date = pd.Timestamp(as_of_selected)
is_latest = as_of_date.date() == max_as_of

trades = trades_all[trades_all["TradeDate"] <= as_of_date].reset_index(drop=True)
settles = settles_all[settles_all["Date"] <= as_of_date].reset_index(drop=True)

if trades.empty:
    st.sidebar.warning("No trades on or before this date yet.")

contracts = sorted(trades["Contract"].unique()) if not trades.empty else sorted(trades_all["Contract"].unique())
last_settle_by_contract = (
    settles.sort_values("Date").groupby("Contract")["SettlePrice"].last().to_dict()
)
prev_settle_by_contract = {}
for c in contracts:
    c_prices = settles.loc[settles["Contract"] == c].sort_values("Date")["SettlePrice"]
    if len(c_prices) >= 2:
        prev_settle_by_contract[c] = c_prices.iloc[-2]
data_source = settles["Source"].iloc[0] if "Source" in settles.columns and len(settles) else "unknown"

# --------------------------------------------------------------------------
# Status bar
# --------------------------------------------------------------------------
dot_class = "live" if data_source == "live" else "fallback"
source_text = "LIVE — YAHOO FINANCE" if data_source == "live" else "STATIC FALLBACK — settlement_prices.csv"

# The actual last settlement date on/before the selected As Of date --
# can differ from as_of_date itself if that date falls on a weekend/
# holiday with no settlement row.
last_available_settle = settles["Date"].max() if len(settles) else None
as_of_str = (
    pd.Timestamp(last_available_settle).strftime("%Y-%m-%d") if pd.notna(last_available_settle) else "unknown"
)
now_str = dt.datetime.utcnow().strftime("%H:%M:%S UTC")

st.markdown(
    f"""
    <div class="term-status-bar">
        <div><span class="dot {dot_class}"></span>BRENT CRUDE (BRN) BOOK &nbsp;·&nbsp; {source_text}
             &nbsp;·&nbsp; AS OF {as_of_str}</div>
        <div>PAGE RENDERED {now_str}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.title("Brent Crude Futures — Book PnL & Risk")

# --------------------------------------------------------------------------
# Sidebar: watchlist + live price override
# --------------------------------------------------------------------------
st.sidebar.markdown('<div class="term-panel-title">Watchlist — last settle</div>', unsafe_allow_html=True)

for c in contracts:
    last_px = last_settle_by_contract[c]
    prev_px = prev_settle_by_contract.get(c)
    if prev_px is not None:
        chg = last_px - prev_px
        cls = "pos" if chg > 0 else ("neg" if chg < 0 else "flat")
        chg_str = f"{chg:+.2f}"
    else:
        cls, chg_str = "flat", "—"
    st.sidebar.markdown(
        f"""<div class="term-watchlist-row">
                <span class="tkr">{c}</span>
                <span class="px">{last_px:.2f}</span>
                <span class="chg {cls}">{chg_str}</span>
            </div>""",
        unsafe_allow_html=True,
    )

st.sidebar.markdown('<div class="term-panel-title">Manual price override</div>', unsafe_allow_html=True)
st.sidebar.caption("Feeds Live PnL on the Portfolio Analysis tab. Swap for a real feed in production.")

if "live_prices" not in st.session_state:
    st.session_state.live_prices = dict(last_settle_by_contract)

if st.sidebar.button("Simulate a live tick"):
    for c in contracts:
        base = last_settle_by_contract[c]
        st.session_state.live_prices[c] = round(base + np.random.normal(0, 0.35), 2)

live_prices = {}
for c in contracts:
    live_prices[c] = st.sidebar.number_input(
        c, value=float(st.session_state.live_prices.get(c, last_settle_by_contract[c])), step=0.01, format="%.2f"
    )
st.session_state.live_prices = live_prices

st.sidebar.button("Refresh live PnL", type="primary")

# --------------------------------------------------------------------------
# Tabs
# --------------------------------------------------------------------------
tab_portfolio, tab_risk, tab_greeks, tab_market = st.tabs(
    ["Portfolio Analysis", "Risk Metrics", "Greeks", "Market Data"]
)

# ==========================================================================
# TAB 1 — Portfolio Analysis
# ==========================================================================
with tab_portfolio:
    if not is_latest:
        st.info(
            f"Viewing historical snapshot as of {as_of_date.strftime('%Y-%m-%d')} — "
            "Live PnL is disabled here since the sidebar price override represents "
            "'right now', not this date. Realized/Unrealized/Cumulative P&L below "
            "reflect the book exactly as it stood on this date."
        )
    summary = book_summary(trades, settles, live_prices=live_prices if is_latest else None)

    total_realized = summary["RealizedPnL"].sum()
    total_unrealized = summary["UnrealizedPnL"].sum()
    total_live = summary["LivePnL"].sum() if "LivePnL" in summary else None

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Realized PnL", f"${total_realized:,.0f}")
    k2.metric("Unrealized PnL (last settle)", f"${total_unrealized:,.0f}")
    k3.metric("Total book PnL (last settle)", f"${total_realized + total_unrealized:,.0f}")
    if total_live is not None:
        delta = total_live - (total_realized + total_unrealized)
        k4.metric("Live PnL (on request)", f"${total_live:,.0f}", delta=f"${delta:,.0f} vs last settle")

    st.subheader("Position summary by contract")
    display_summary = summary.copy()
    for col in ["AvgEntryPrice", "SettlePrice", "RealizedPnL", "UnrealizedPnL", "CumulativePnL", "LivePrice", "LivePnL"]:
        if col in display_summary:
            display_summary[col] = display_summary[col].round(2)
    signed_cols = [c for c in ["RealizedPnL", "UnrealizedPnL", "CumulativePnL", "LivePnL"] if c in display_summary]
    st.dataframe(style_signed_table(display_summary, signed_cols), width="stretch", hide_index=True)

    st.subheader("Daily mark-to-market — cumulative PnL")
    mtm = compute_daily_mtm(trades, settles)
    view = st.radio("View", ["By contract", "Total book"], horizontal=True, key="mtm_view")

    fig = go.Figure()
    if view == "By contract":
        for c in contracts:
            cdata = mtm[mtm["Contract"] == c]
            fig.add_trace(go.Scatter(x=cdata["Date"], y=cdata["CumulativePnL"], mode="lines+markers", name=c))
    else:
        total = mtm.groupby("Date")["DailyPnL"].sum().cumsum().reset_index(name="CumulativePnL")
        fig.add_trace(go.Scatter(x=total["Date"], y=total["CumulativePnL"], mode="lines+markers", name="Total book"))

    fig.update_layout(xaxis_title="Date", yaxis_title="Cumulative PnL (USD)", hovermode="x unified", height=420)
    fig.add_hline(y=0, line_dash="dot", line_color=COLOR["border"])
    apply_terminal_theme(fig)
    st.plotly_chart(fig, width="stretch")

    with st.expander("Trade blotter"):
        st.dataframe(
            trades[["TradeID", "TradeDate", "Contract", "BuySell", "Quantity", "Price", "Trader"]],
            width="stretch", hide_index=True,
        )

    with st.expander("Daily settlement prices"):
        st.dataframe(settles, width="stretch", hide_index=True)

# ==========================================================================
# TAB 2 — Risk Metrics (Historical + Parametric VaR / ES)
# ==========================================================================
with tab_risk:
    st.subheader("Value at Risk & Expected Shortfall")
    st.write(
        "Both methods use the CURRENT open position (FIFO book) and the historical settlement "
        "price series shown in the sidebar. **Historical** makes no distributional assumption; "
        "**Parametric** (delta-normal) assumes Normal returns and uses the sample covariance matrix."
    )

    horizon_choice = st.radio("Horizon", ["1-day", "10-day"], horizontal=True, key="risk_horizon")
    horizon_days = 1 if horizon_choice == "1-day" else 10

    try:
        returns, positions = build_risk_inputs_from_book(trades, settles)
    except Exception as e:
        st.error(f"Could not build risk inputs from the current book: {e}")
        returns, positions = None, None

    if returns is not None and len(returns) < 20:
        st.warning(
            f"Only {len(returns)} overlapping historical return observations are available "
            "across all contracts (they started trading on different dates, and the dummy "
            "settlement history is short). VaR/ES below are directionally useful but not "
            "statistically robust yet — extending the settlement price history would fix this."
        )

    if returns is not None and positions is not None and len(returns) > 1:
        st.markdown('<div class="term-panel-title">Current position — barrels, signed long/short</div>', unsafe_allow_html=True)
        st.dataframe(positions.rename("Position (bbl)").to_frame().T, width="stretch")

        summary_df = risk_summary(
            returns, positions, confidence_levels=(0.95, 0.99), horizon_days_list=(horizon_days,)
        )
        display_risk = summary_df.copy()
        display_risk["Confidence"] = (display_risk["Confidence"] * 100).round(0).astype(int).astype(str) + "%"
        display_risk["VaR"] = display_risk["VaR"].round(0)
        display_risk["ES"] = display_risk["ES"].round(0)
        st.dataframe(style_risk_table(display_risk, ["VaR", "ES"]), width="stretch", hide_index=True)

        m1, m2, m3, m4 = st.columns(4)
        hist_95 = summary_df[(summary_df.Method == "Historical") & (summary_df.Confidence == 0.95)].iloc[0]
        param_95 = summary_df[(summary_df.Method == "Parametric") & (summary_df.Confidence == 0.95)].iloc[0]
        hist_99 = summary_df[(summary_df.Method == "Historical") & (summary_df.Confidence == 0.99)].iloc[0]
        param_99 = summary_df[(summary_df.Method == "Parametric") & (summary_df.Confidence == 0.99)].iloc[0]
        m1.metric(f"Historical VaR 95% ({horizon_choice})", f"${hist_95.VaR:,.0f}")
        m2.metric(f"Parametric VaR 95% ({horizon_choice})", f"${param_95.VaR:,.0f}")
        m3.metric(f"Historical VaR 99% ({horizon_choice})", f"${hist_99.VaR:,.0f}")
        m4.metric(f"Parametric VaR 99% ({horizon_choice})", f"${param_99.VaR:,.0f}")

        st.subheader("Historical P&L distribution")
        st.caption(
            f"Empirical distribution of {horizon_choice} book P&L, built by revaluing the current "
            "position against every historical day's price move (scaled by √horizon for 10-day)."
        )
        hist_result = compute_historical_var_es(returns, positions, confidence=0.95, horizon_days=horizon_days)
        pnl_dist = hist_result["pnl_distribution"]

        hist_fig = go.Figure()
        hist_fig.add_trace(go.Histogram(x=pnl_dist, nbinsx=30, name="P&L", marker_color=COLOR["accent"]))
        hist_fig.add_vline(x=-hist_95.VaR, line_dash="dot", line_color=COLOR["accent"],
                            annotation_text="95% VaR", annotation_position="top")
        hist_fig.add_vline(x=-hist_99.VaR, line_dash="dot", line_color=COLOR["negative"],
                            annotation_text="99% VaR", annotation_position="top")
        hist_fig.update_layout(xaxis_title="P&L (USD)", yaxis_title="Days", height=380)
        apply_terminal_theme(hist_fig)
        st.plotly_chart(hist_fig, width="stretch")

        with st.expander("Correlation matrix (return series, current lookback)"):
            corr = returns.corr()
            st.dataframe(corr.round(2), width="stretch")
            if corr.where(~np.eye(len(corr), dtype=bool)).max().max() > 0.95:
                st.info(
                    "Contracts are highly correlated because they're currently all priced off "
                    "the same broadcast ticker (see market_data.py). Parametric VaR won't show "
                    "much diversification benefit until each contract has an independent price series."
                )
    elif returns is not None:
        st.error("Not enough overlapping historical data to compute VaR/ES yet.")

# ==========================================================================
# TAB 3 — Greeks: full option book, aggregated portfolio Greeks (Black-76)
# ==========================================================================
with tab_greeks:
    st.subheader("Options Book — Aggregated Greeks (Black-76)")
    st.write(
        "European options on the Brent futures, priced with Black-76 using each contract's "
        "settlement price as the forward and a strike/expiry-dependent implied vol surface. "
        "Options Delta is reported **separately** from the futures position — not netted — "
        "per how this book is run."
    )

    try:
        option_trades_all, vol_surface = get_options_data()
        option_trades = option_trades_all[option_trades_all["TradeDate"] <= as_of_date].reset_index(drop=True)
    except FileNotFoundError as e:
        st.error(f"Options data not found: {e}")
        option_trades, vol_surface = None, None

    if option_trades is not None:
        if "Source" in vol_surface.columns and len(vol_surface):
            lines = []
            for underlying, grp in vol_surface.groupby("UnderlyingContract"):
                source = grp["Source"].iloc[0]
                coverage = f"{grp['ExpiryDate'].min().strftime('%Y-%m')} to {grp['ExpiryDate'].max().strftime('%Y-%m')}"
                dot = "🟢" if source == "market" else "🟡"
                label = "LIVE market-calibrated" if source == "market" else "STATIC fallback"
                lines.append(f"{dot} {underlying}: {label} ({coverage})")
            st.caption("Vol surface — " + " · ".join(lines))
        else:
            st.caption("🟡 Vol surface: STATIC fallback (volatility_surface_dummy.csv)")

        if not is_latest:
            st.info(f"Viewing historical snapshot as of {as_of_date.strftime('%Y-%m-%d')}.")

        # Futures Delta, shown alongside for reference only (see note above)
        open_lots_fut, _ = compute_fifo_positions(trades)
        fut_positions = open_position_summary(open_lots_fut)
        lot_size_by_contract = trades.groupby("Contract")["LotSize"].first()
        total_futures_delta = sum(
            row.NetQty * lot_size_by_contract[row.Contract] for row in fut_positions.itertuples()
        )

        _, realized_opt_df = compute_option_fifo_positions(option_trades)
        live_greeks = compute_live_option_greeks(option_trades, settles, vol_surface, as_of_date=as_of_date)
        port_summary = portfolio_greeks_summary(live_greeks)

        if not live_greeks.empty and live_greeks["VolStalenessDays"].max() > VOL_SURFACE_STALENESS_WARNING_DAYS:
            worst = live_greeks.loc[live_greeks["VolStalenessDays"].idxmax()]
            st.warning(
                f"Volatility surface data is stale for one or more open positions — up to "
                f"{int(live_greeks['VolStalenessDays'].max())} days away from the selected as-of date "
                f"(worst case: {worst.UnderlyingContract} {worst.OptionType} {worst.Strike:.2f}). "
                "Greeks below are still computed, but with the nearest available vol, not a current one. "
                "See 'VolStalenessDays' in the table below for the per-position detail."
            )

        total_realized_opt = realized_opt_df["RealizedPnL"].sum() if not realized_opt_df.empty else 0.0
        total_position_delta_opt = live_greeks["PositionDelta"].sum() if not live_greeks.empty else 0.0

        k1, k2, k3 = st.columns(3)
        k1.metric("Realized Option PnL (FIFO)", f"${total_realized_opt:,.0f}")
        k2.metric("Options Portfolio Delta (bbl-equiv)", f"{total_position_delta_opt:,.0f}")
        k3.metric("Futures Position Delta (bbl)", f"{total_futures_delta:,.0f}")
        st.caption("Futures Delta shown for reference only — the two are reported separately, not netted.")

        st.subheader("Open option positions — live Greeks")
        if live_greeks.empty:
            st.info("No open option positions as of the latest settlement date.")
        else:
            disp = live_greeks.copy()
            for col in ["Forward", "TheoPrice"]:
                disp[col] = disp[col].round(2)
            for col in ["ImpliedVol", "Delta", "Gamma", "Theta", "Rho"]:
                disp[col] = disp[col].round(4)
            for col in ["Vega", "PositionDelta", "PositionGamma", "PositionVega",
                        "PositionTheta", "PositionRho", "PositionValue"]:
                disp[col] = disp[col].round(2)
            st.dataframe(
                style_signed_table(disp, ["PositionDelta", "PositionTheta", "PositionRho"]),
                width="stretch", hide_index=True,
            )

        st.subheader("Portfolio Greeks — aggregated by underlying")
        if not port_summary.empty:
            st.dataframe(
                style_signed_table(port_summary.round(2),
                                    ["PositionDelta", "PositionGamma", "PositionVega", "PositionTheta", "PositionRho"]),
                width="stretch", hide_index=True,
            )

        if not realized_opt_df.empty:
            with st.expander("Realized option PnL (FIFO)"):
                disp_r = realized_opt_df.copy()
                disp_r["ExpiryDate"] = disp_r["ExpiryDate"].dt.date
                disp_r["CloseDate"] = disp_r["CloseDate"].dt.date
                st.dataframe(style_signed_table(disp_r.round(2), ["RealizedPnL"]), width="stretch", hide_index=True)

        with st.expander("Options trade blotter"):
            st.dataframe(
                option_trades[["TradeID", "TradeDate", "UnderlyingContract", "OptionType", "Strike",
                                "ExpiryDate", "BuySell", "Quantity", "Premium", "Trader"]],
                width="stretch", hide_index=True,
            )

        # ------------------------------------------------------------
        # Greeks vs Strike — smile-aware
        # ------------------------------------------------------------
        st.subheader("Greeks vs Strike — volatility smile aware")
        st.caption(
            "Unlike a flat-vol Greeks sweep, each strike here is priced against ITS OWN implied vol "
            "from the surface — so the curves below bake in the actual smile/skew shape, not a single "
            "constant sigma."
        )

        colA, colB = st.columns(2)
        sel_underlying = colA.selectbox("Underlying", contracts, key="greeks_underlying")
        sel_type = colB.radio("Option type", ["CALL", "PUT"], horizontal=True, key="greeks_option_type")

        as_of = settles["Date"].max()
        F = last_settle_by_contract[sel_underlying]
        try:
            resolved_underlying = resolve_surface_underlying(vol_surface, sel_underlying)
            expiry = vol_surface.loc[vol_surface["UnderlyingContract"] == resolved_underlying, "ExpiryDate"].iloc[0]
            days_to_expiry = (expiry - as_of).days
            vol_data_available = True
        except (ValueError, IndexError) as e:
            st.warning(f"No volatility surface data available for {sel_underlying}: {e}")
            vol_data_available = False
            days_to_expiry = 0

        if not vol_data_available:
            pass  # warning already shown above
        elif days_to_expiry <= 0:
            st.warning(f"{sel_underlying}'s option expiry has passed the latest settlement date — nothing to plot.")
        else:
            T = days_to_expiry / 365.0
            strike_grid = np.linspace(F * 0.85, F * 1.15, 60)
            vol_lookups = [lookup_implied_vol(vol_surface, sel_underlying, expiry, k, as_of) for k in strike_grid]
            sigmas = [v for v, _ in vol_lookups]
            chart_staleness = max(s for _, s in vol_lookups)
            if chart_staleness > VOL_SURFACE_STALENESS_WARNING_DAYS:
                st.warning(
                    f"This chart's vol surface lookup is {chart_staleness} days from the selected as-of "
                    "date — the smile shown may not reflect current conditions."
                )
            greek_curves = {g: [] for g in ["Delta", "Gamma", "Vega", "Theta"]}
            for k, sigma in zip(strike_grid, sigmas):
                g = black76_greeks(F, k, T, RISK_FREE_RATE_DEFAULT, sigma, sel_type)
                for name in greek_curves:
                    greek_curves[name].append(g[name])

            smile_fig = go.Figure()
            smile_fig.add_trace(go.Scatter(x=strike_grid, y=sigmas, mode="lines", name="Implied Vol"))
            smile_fig.add_vline(x=F, line_dash="dot", line_color=COLOR["accent"],
                                 annotation_text="Forward", annotation_position="top")
            smile_fig.update_layout(xaxis_title="Strike", yaxis_title="Implied Vol", height=260)
            apply_terminal_theme(smile_fig)
            st.plotly_chart(smile_fig, width="stretch")

            grid_fig = make_subplots(rows=2, cols=2, subplot_titles=["Delta", "Gamma", "Vega", "Theta"])
            positions_rc = [(1, 1), (1, 2), (2, 1), (2, 2)]
            for (name, values), (r, c) in zip(greek_curves.items(), positions_rc):
                grid_fig.add_trace(go.Scatter(x=strike_grid, y=values, mode="lines", name=name, showlegend=False),
                                    row=r, col=c)
                grid_fig.add_vline(x=F, line_dash="dot", line_color=COLOR["accent"], row=r, col=c)
            grid_fig.update_layout(height=520)
            apply_terminal_theme(grid_fig)
            st.plotly_chart(grid_fig, width="stretch")

# ==========================================================================
# TAB 4 — Market Data: benchmark prices (Brent/WTI/TTF) and the implied
# vol surface built in the Greeks tab, viewed as smile curves or a 3D mesh
# ==========================================================================
with tab_market:
    st.subheader("Benchmark Prices")
    lookback_choice = st.radio(
        "Lookback", ["6 months", "1 year", "2 years"], index=1, horizontal=True, key="benchmark_lookback"
    )
    lookback_days = {"6 months": 182, "1 year": 365, "2 years": 730}[lookback_choice]

    prices, price_source = get_benchmark_prices(lookback_days)
    if price_source != "live" or prices.empty:
        st.warning(
            "Live benchmark price data is currently unavailable (network or Yahoo Finance fetch failed). "
            "There's no offline fallback for WTI/TTF yet — settlement_prices.csv only ever covered the "
            "Brent dummy book."
        )
    else:
        oil_cols = [c for c in prices.columns if c.startswith("Brent") or c.startswith("WTI")]
        gas_cols = [c for c in prices.columns if c.startswith("TTF")]

        if oil_cols:
            oil_fig = go.Figure()
            for col in oil_cols:
                oil_fig.add_trace(go.Scatter(x=prices.index, y=prices[col], mode="lines", name=col))
            oil_fig.update_layout(
                title="Crude Oil — Brent vs WTI ($/bbl)", xaxis_title="Date", yaxis_title="$/bbl", height=380
            )
            apply_terminal_theme(oil_fig)
            st.plotly_chart(oil_fig, width="stretch")

        if gas_cols:
            gas_fig = go.Figure()
            for col in gas_cols:
                gas_fig.add_trace(go.Scatter(x=prices.index, y=prices[col], mode="lines", name=col,
                                              line=dict(color=COLOR["positive"])))
            gas_fig.update_layout(
                title="European Gas — Dutch TTF ($/MMBtu)", xaxis_title="Date", yaxis_title="$/MMBtu", height=320
            )
            apply_terminal_theme(gas_fig)
            st.plotly_chart(gas_fig, width="stretch")

        st.caption(
            "Brent and WTI share $/bbl units and are plotted together; TTF is priced in $/MMBtu "
            "(a different physical unit — gas vs. oil) and is kept on its own chart rather than a "
            "dual-axis overlay, to avoid implying a direct price comparison that doesn't exist."
        )

    st.subheader("Implied Volatility Surface")
    st.caption(
        "Same live, market-calibrated surface used by the Greeks tab (BNO/USO listed options, "
        "Black-76 implied vol, no extrapolation) — see the Greeks tab for per-commodity source "
        "and coverage details."
    )

    try:
        vol_surface_for_plot = get_market_vol_surface()
    except Exception as e:
        vol_surface_for_plot = pd.DataFrame()
        st.error(f"Could not load volatility surface data: {e}")

    if not vol_surface_for_plot.empty:
        view_choice = st.radio("View", ["Smile curves", "3D Surface"], horizontal=True, key="vol_surface_view")

        if view_choice == "Smile curves":
            smile_fig = go.Figure()
            for (underlying, expiry), grp in vol_surface_for_plot.groupby(["UnderlyingContract", "ExpiryDate"]):
                grp = grp.sort_values("Strike")
                label = f"{underlying} {pd.Timestamp(expiry).strftime('%Y-%m')}"
                smile_fig.add_trace(go.Scatter(x=grp["Strike"], y=grp["ImpliedVol"], mode="lines+markers", name=label))
            smile_fig.update_layout(xaxis_title="Strike", yaxis_title="Implied Vol", height=480)
            apply_terminal_theme(smile_fig)
            st.plotly_chart(smile_fig, width="stretch")

        else:
            # A 3D surface needs >=2 distinct maturities for a given underlying
            # to mean anything -- the dummy fallback (one expiry per contract
            # month) doesn't qualify, only a live multi-tenor market build does.
            expiry_counts = vol_surface_for_plot.groupby("UnderlyingContract")["ExpiryDate"].nunique()
            surface_candidates = expiry_counts[expiry_counts >= 2].index.tolist()

            if not surface_candidates:
                st.info(
                    "A 3D surface needs multiple maturities for one underlying to be meaningful — only "
                    "single-expiry data is currently available (e.g. dummy fallback mode, or thin live "
                    "coverage). Smile curves are more informative right now."
                )
            elif "Moneyness" not in vol_surface_for_plot.columns:
                st.info("This data source doesn't carry Moneyness (needed to align strikes across maturities) — showing smile curves instead is more reliable.")
            else:
                for underlying in surface_candidates:
                    grp = vol_surface_for_plot[vol_surface_for_plot["UnderlyingContract"] == underlying].copy()
                    grp = grp.dropna(subset=["Moneyness"])
                    grp["MaturityMonths"] = (
                        (pd.to_datetime(grp["ExpiryDate"]) - pd.to_datetime(grp["Date"])).dt.days / 30.44
                    ).round(1)
                    pivot = grp.pivot_table(index="MaturityMonths", columns="Moneyness", values="ImpliedVol")

                    surf_fig = go.Figure(data=[go.Surface(
                        z=pivot.values, x=pivot.columns, y=pivot.index, colorscale="Oranges",
                        colorbar=dict(title="Vol", tickfont=dict(color=COLOR["text_secondary"])),
                    )])
                    surf_fig.update_layout(
                        title=f"{underlying} — Implied Vol Surface",
                        scene=dict(
                            xaxis_title="Moneyness (Strike/Forward)", yaxis_title="Maturity (months)", zaxis_title="Implied Vol",
                            xaxis=dict(backgroundcolor=COLOR["bg_base"], gridcolor=COLOR["border"], color=COLOR["text_secondary"]),
                            yaxis=dict(backgroundcolor=COLOR["bg_base"], gridcolor=COLOR["border"], color=COLOR["text_secondary"]),
                            zaxis=dict(backgroundcolor=COLOR["bg_base"], gridcolor=COLOR["border"], color=COLOR["text_secondary"]),
                            bgcolor=COLOR["bg_base"],
                        ),
                        paper_bgcolor=COLOR["bg_base"],
                        font=dict(family="IBM Plex Mono, monospace", color=COLOR["text_secondary"]),
                        height=560,
                        margin=dict(t=50, b=20),
                    )
                    st.plotly_chart(surf_fig, width="stretch")
                    gaps = pivot.isna().sum().sum()
                    if gaps > 0:
                        st.caption(
                            f"{gaps} grid point(s) are missing (shown as holes in the mesh) — real listed "
                            "option coverage didn't reach that strike/maturity combination, and this surface "
                            "never extrapolates to fill it in."
                        )
    else:
        st.info("No volatility surface data available to plot.")
