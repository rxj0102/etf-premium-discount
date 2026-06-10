"""Streamlit dashboard for ETF premium/discount mean reversion monitoring.

Features:

* Sidebar: ETF selector and lookback period slider.
* Live premium/discount z-score gauge, colour coded (green < 1 sigma,
  yellow 1-2 sigma, red > 2 sigma).
* Historical premium/discount chart with +/-2 sigma signal overlays.
* Event-study average reversion path chart.
* Backtest summary statistics table.
* Raw data is refreshed via yfinance at most once per day (cached fetch on
  load; delete ``data/raw`` or wait for the TTL to force a refresh).

Run with::

    streamlit run dashboard/app.py
"""

from __future__ import annotations

import datetime as dt
import logging
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from analysis.reversion import path_summary, reversion_paths  # noqa: E402
from analysis.spread import build_spread_panel, load_close  # noqa: E402
from backtest.strategy import run_backtests  # noqa: E402
from data import fetch  # noqa: E402

logger = logging.getLogger(__name__)

ONE_DAY_SECONDS = 24 * 60 * 60


@st.cache_data(ttl=ONE_DAY_SECONDS, show_spinner="Refreshing market data...")
def refresh_data(today: dt.date) -> bool:
    """Refresh raw data via yfinance at most once per calendar day.

    Args:
        today: Current date; acts as the cache key so the underlying fetch
            re-runs when the day changes (i.e. data refreshes daily on load).

    Returns:
        True when the fetch step completed (data exists on disk).
    """
    marker = fetch.RAW_DIR / "vix.csv"
    stale = (
        not marker.exists()
        or dt.date.fromtimestamp(marker.stat().st_mtime) < today
    )
    if stale:
        fetch.main()
    return True


@st.cache_data(ttl=ONE_DAY_SECONDS, show_spinner="Computing analytics...")
def load_analytics(today: dt.date):
    """Build all derived datasets used by the dashboard (cached daily).

    Args:
        today: Current date, used as the cache key alongside the raw fetch.

    Returns:
        Tuple of (spreads, zscores, events, closes, backtest_summary).
    """
    refresh_data(today)
    spreads, zscores, events = build_spread_panel()
    closes = {t: load_close(t) for t in fetch.ETF_TICKERS}
    backtest_summary, _ = run_backtests()
    return spreads, zscores, events, closes, backtest_summary


def zscore_gauge(z_value: float, ticker: str) -> go.Figure:
    """Build the colour-coded z-score gauge.

    Args:
        z_value: Latest premium/discount z-score.
        ticker: ETF ticker for the title.

    Returns:
        Plotly figure with a gauge indicator (green < 1 sigma, yellow 1-2
        sigma, red > 2 sigma, based on |z|).
    """
    abs_z = abs(z_value)
    bar_color = "green" if abs_z < 1 else ("gold" if abs_z <= 2 else "red")
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=z_value,
            number={"valueformat": ".2f"},
            title={"text": f"{ticker} premium/discount z-score (63d)"},
            gauge={
                "axis": {"range": [-4, 4]},
                "bar": {"color": bar_color},
                "steps": [
                    {"range": [-4, -2], "color": "rgba(255,0,0,0.25)"},
                    {"range": [-2, -1], "color": "rgba(255,215,0,0.25)"},
                    {"range": [-1, 1], "color": "rgba(0,128,0,0.20)"},
                    {"range": [1, 2], "color": "rgba(255,215,0,0.25)"},
                    {"range": [2, 4], "color": "rgba(255,0,0,0.25)"},
                ],
                "threshold": {
                    "line": {"color": "black", "width": 3},
                    "value": z_value,
                },
            },
        )
    )
    fig.update_layout(height=300, margin=dict(t=60, b=10, l=30, r=30))
    return fig


def history_chart(
    spread: pd.Series, z: pd.Series, events: pd.DataFrame, ticker: str
) -> go.Figure:
    """Build the historical premium/discount chart with signal overlays.

    Args:
        spread: Premium/discount series (percent) for the selected ETF.
        z: Matching z-score series (drives the signal markers).
        events: Event table filtered to the lookback window.
        ticker: ETF ticker for labelling.

    Returns:
        Plotly figure with the spread line and discount/premium event
        markers.
    """
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=spread.index, y=spread, name="premium/discount (%)",
                   line=dict(color="steelblue", width=1))
    )
    ev = events[events["ticker"] == ticker]
    for side, color, symbol in (
        ("discount", "red", "triangle-down"),
        ("premium", "green", "triangle-up"),
    ):
        sub = ev[ev["side"] == side]
        if not sub.empty:
            fig.add_trace(
                go.Scatter(
                    x=sub["Date"], y=sub["premium_discount"], mode="markers",
                    name=f"{side} event (|z|>2)",
                    marker=dict(color=color, size=9, symbol=symbol),
                )
            )
    fig.add_hline(y=0, line_width=1, line_color="grey")
    fig.update_layout(
        title=f"{ticker} premium/discount history with signals",
        yaxis_title="Premium/discount (%)",
        height=380,
        margin=dict(t=50, b=20),
        legend=dict(orientation="h", y=1.12),
    )
    return fig


def reversion_chart(events: pd.DataFrame, closes: dict, ticker: str) -> go.Figure:
    """Build the event-study reversion path chart for one ETF.

    Args:
        events: Full event table.
        closes: Mapping of ticker to closing price series.
        ticker: Selected ETF.

    Returns:
        Plotly figure of the mean post-discount-event return path with a 95%
        confidence band (falls back to all-ETF events if the selected ETF has
        too few).
    """
    discount = events[events["side"] == "discount"]
    own = discount[discount["ticker"] == ticker]
    used, scope = (own, ticker) if len(own) >= 5 else (discount, "all ETFs")
    fig = go.Figure()
    if used.empty:
        fig.update_layout(title="No discount events in selected window")
        return fig
    summary = path_summary(reversion_paths(used, closes))
    fig.add_trace(
        go.Scatter(x=summary.index, y=summary["upper"], line=dict(width=0),
                   showlegend=False, hoverinfo="skip")
    )
    fig.add_trace(
        go.Scatter(x=summary.index, y=summary["lower"], fill="tonexty",
                   fillcolor="rgba(214,39,40,0.15)", line=dict(width=0),
                   name="95% CI")
    )
    fig.add_trace(
        go.Scatter(x=summary.index, y=summary["mean"],
                   line=dict(color="firebrick"),
                   name=f"mean path (n={int(summary['n'].iloc[0])})")
    )
    fig.add_hline(y=0, line_width=1, line_color="grey")
    fig.update_layout(
        title=f"Average return path after discount events ({scope})",
        xaxis_title="Trading days after event",
        yaxis_title="Cumulative return (%)",
        height=380,
        margin=dict(t=50, b=20),
    )
    return fig


def main() -> None:
    """Render the Streamlit dashboard."""
    st.set_page_config(page_title="ETF Premium/Discount Monitor", layout="wide")
    st.title("ETF Premium/Discount Mean Reversion Monitor")

    spreads, zscores, events, closes, backtest_summary = load_analytics(
        dt.date.today()
    )

    st.sidebar.header("Controls")
    ticker = st.sidebar.selectbox("ETF", fetch.ETF_TICKERS, index=0)
    lookback_months = st.sidebar.slider(
        "Lookback period (months)", min_value=6, max_value=60, value=36, step=6
    )
    cutoff = spreads.index.max() - pd.DateOffset(months=lookback_months)
    window_spread = spreads.loc[spreads.index >= cutoff, ticker]
    window_z = zscores.loc[zscores.index >= cutoff, ticker]
    window_events = events[events["Date"] >= cutoff]

    latest_z = float(zscores[ticker].dropna().iloc[-1])
    latest_spread = float(spreads[ticker].dropna().iloc[-1])

    col1, col2 = st.columns([1, 2])
    with col1:
        st.plotly_chart(zscore_gauge(latest_z, ticker), width="stretch")
        st.metric(
            "Latest premium/discount",
            f"{latest_spread:+.3f}%",
            help="(Price - NAV) / NAV. Negative = discount.",
        )
    with col2:
        st.plotly_chart(
            history_chart(window_spread, window_z, window_events, ticker),
            width="stretch",
        )

    st.plotly_chart(
        reversion_chart(window_events, closes, ticker), width="stretch"
    )

    st.subheader("Backtest summary (discount entry z < -2, exit z = 0)")
    st.dataframe(backtest_summary, width="stretch")
    st.caption(
        "Data refreshes at most once per day via yfinance on page load. "
        "NAV is simulated when no external NAV file is provided — see "
        "data/fetch.py for details."
    )


if __name__ == "__main__":
    # `streamlit run dashboard/app.py` executes this module as __main__.
    logging.basicConfig(level=logging.INFO)
    main()
