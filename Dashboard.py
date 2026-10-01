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
    RISK_FREE_RATE_DEFAULT,
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
@st.cache_data(ttl=900)  # 15 min: fresh enough to pick up a live price, gentle enough on Yahoo
def get_data():
    trades = load_trades("trades_dummy.xlsx")
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

@st.cache_data(ttl=900)
def get_options_data():
    option_trades = load_option_trades("options_trades_dummy.xlsx")
    vol_surface = load_volatility_surface("volatility_surface_dummy.csv")
    return option_trades, vol_surface


trades, settles = get_data()
contracts = sorted(trades["Contract"].unique())
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
now_str = dt.datetime.utcnow().strftime("%H:%M:%S UTC")
st.markdown(
    f"""
    <div class="term-status-bar">
        <div><span class="dot {dot_class}"></span>BRENT CRUDE (BRN) BOOK &nbsp;·&nbsp; {source_text}</div>
        <div>UPDATED {now_str}</div>
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
tab_portfolio, tab_risk, tab_greeks = st.tabs(["Portfolio Analysis", "Risk Metrics", "Greeks"])

# ==========================================================================
# TAB 1 — Portfolio Analysis
# ==========================================================================
with tab_portfolio:
    summary = book_summary(trades, settles, live_prices=live_prices)

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
        option_trades, vol_surface = get_options_data()
    except FileNotFoundError as e:
        st.error(f"Options data not found: {e}")
        option_trades, vol_surface = None, None

    if option_trades is not None:
        # Futures Delta, shown alongside for reference only (see note above)
        open_lots_fut, _ = compute_fifo_positions(trades)
        fut_positions = open_position_summary(open_lots_fut)
        lot_size_by_contract = trades.groupby("Contract")["LotSize"].first()
        total_futures_delta = sum(
            row.NetQty * lot_size_by_contract[row.Contract] for row in fut_positions.itertuples()
        )

        _, realized_opt_df = compute_option_fifo_positions(option_trades)
        live_greeks = compute_live_option_greeks(option_trades, settles, vol_surface)
        port_summary = portfolio_greeks_summary(live_greeks)

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
        expiry = vol_surface.loc[vol_surface["UnderlyingContract"] == sel_underlying, "ExpiryDate"].iloc[0]
        days_to_expiry = (expiry - as_of).days

        if days_to_expiry <= 0:
            st.warning(f"{sel_underlying}'s option expiry has passed the latest settlement date — nothing to plot.")
        else:
            T = days_to_expiry / 365.0
            strike_grid = np.linspace(F * 0.85, F * 1.15, 60)
            sigmas = [lookup_implied_vol(vol_surface, sel_underlying, expiry, k, as_of) for k in strike_grid]
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
