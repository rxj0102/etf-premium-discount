"""Premium/discount spread construction and event flagging.

Loads raw price and NAV data produced by ``data/fetch.py``, computes for each
ETF the premium/discount

    premium_discount = (Price - NAV) / NAV * 100   (percent)

its 63-day rolling z-score, and flags "events" where the z-score crosses the
+/-2 sigma threshold (a crossing, not merely being beyond the threshold, so
each episode is counted once at its onset).

Outputs (``data/processed/``):

* ``premium_discount.csv``  wide frame of premium/discount in percent
* ``zscore.csv``            wide frame of 63d rolling z-scores
* ``events.csv``            one row per threshold crossing
  (columns: Date, ticker, side, zscore, premium_discount)

A full spread history chart with stress-episode annotations (Mar 2020,
Oct 2022) is saved to ``outputs/spread_history.png``.

Usage::

    python analysis/spread.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # safe for headless runs; notebooks override this
import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from data.fetch import ETF_TICKERS, RAW_DIR  # noqa: E402

logger = logging.getLogger(__name__)

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

ZSCORE_WINDOW = 63       # ~ one quarter of trading days
EVENT_THRESHOLD = 2.0    # sigma

STRESS_EPISODES = {
    "COVID-19 (Mar 2020)": ("2020-02-20", "2020-04-15"),
    "Rate shock (Oct 2022)": ("2022-09-15", "2022-10-31"),
}


def load_close(ticker: str) -> pd.Series:
    """Load the closing price series for one ETF from ``data/raw/``.

    Args:
        ticker: ETF ticker symbol.

    Returns:
        Closing price series indexed by date.
    """
    df = pd.read_csv(RAW_DIR / f"{ticker}.csv", index_col=0, parse_dates=True)
    return df["Close"].rename(ticker)


def load_nav(ticker: str) -> pd.Series:
    """Load the NAV series for one ETF from ``data/raw/``.

    Args:
        ticker: ETF ticker symbol.

    Returns:
        NAV series indexed by date.
    """
    df = pd.read_csv(RAW_DIR / f"{ticker}_nav.csv", index_col=0, parse_dates=True)
    return df.iloc[:, 0].rename(ticker)


def load_vix() -> pd.Series:
    """Load the VIX close series from ``data/raw/``.

    Returns:
        VIX close series named ``vix`` indexed by date.
    """
    df = pd.read_csv(RAW_DIR / "vix.csv", index_col=0, parse_dates=True)
    return df.iloc[:, 0].rename("vix")


def compute_premium_discount(close: pd.Series, nav: pd.Series) -> pd.Series:
    """Compute the premium/discount in percent.

    Args:
        close: ETF closing price series.
        nav: ETF NAV series.

    Returns:
        Series of ``(Price - NAV) / NAV * 100`` aligned on common dates.
    """
    close, nav = close.align(nav, join="inner")
    return ((close - nav) / nav * 100).rename(close.name)


def rolling_zscore(series: pd.Series, window: int = ZSCORE_WINDOW) -> pd.Series:
    """Compute a rolling z-score of a series.

    Args:
        series: Input series.
        window: Rolling window length in trading days (default 63).

    Returns:
        ``(x - rolling_mean) / rolling_std`` with the same index.
    """
    mean = series.rolling(window).mean()
    std = series.rolling(window).std()
    return (series - mean) / std


def flag_events(
    zscores: pd.DataFrame,
    spreads: pd.DataFrame,
    threshold: float = EVENT_THRESHOLD,
) -> pd.DataFrame:
    """Flag dates where each ETF's z-score crosses the +/- threshold.

    A *discount* event is flagged when z crosses below ``-threshold`` (it was
    above on the prior day); a *premium* event when z crosses above
    ``+threshold``. Crossings rather than levels avoid double counting a
    single episode.

    Args:
        zscores: Wide frame of rolling z-scores, one column per ETF.
        spreads: Wide frame of premium/discount values (percent).
        threshold: Sigma threshold (default 2.0).

    Returns:
        DataFrame with columns Date, ticker, side ('discount'/'premium'),
        zscore and premium_discount, sorted by date.
    """
    records = []
    for ticker in zscores.columns:
        z = zscores[ticker]
        discount = (z <= -threshold) & (z.shift(1) > -threshold)
        premium = (z >= threshold) & (z.shift(1) < threshold)
        for side, mask in (("discount", discount), ("premium", premium)):
            for dt in z.index[mask]:
                records.append(
                    {
                        "Date": dt,
                        "ticker": ticker,
                        "side": side,
                        "zscore": z.loc[dt],
                        "premium_discount": spreads.loc[dt, ticker],
                    }
                )
    events = pd.DataFrame(records).sort_values("Date").reset_index(drop=True)
    logger.info(
        "Flagged %d events (%d discount, %d premium)",
        len(events),
        (events["side"] == "discount").sum() if not events.empty else 0,
        (events["side"] == "premium").sum() if not events.empty else 0,
    )
    return events


def build_spread_panel() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build premium/discount, z-score and event tables for all ETFs.

    Returns:
        Tuple of (spreads, zscores, events): wide premium/discount frame in
        percent, wide z-score frame, and the event table from
        :func:`flag_events`.
    """
    spreads = pd.DataFrame(
        {t: compute_premium_discount(load_close(t), load_nav(t)) for t in ETF_TICKERS}
    )
    zscores = spreads.apply(rolling_zscore)
    events = flag_events(zscores, spreads)
    return spreads, zscores, events


def plot_spread_history(
    spreads: pd.DataFrame,
    events: pd.DataFrame,
    save_path: Path | None = None,
) -> plt.Figure:
    """Plot the full premium/discount history with stress annotations.

    Args:
        spreads: Wide premium/discount frame (percent).
        events: Event table from :func:`flag_events`.
        save_path: Optional PNG destination; created if provided.

    Returns:
        The matplotlib figure.
    """
    fig, axes = plt.subplots(
        len(spreads.columns), 1, figsize=(12, 2.2 * len(spreads.columns)),
        sharex=True,
    )
    for ax, ticker in zip(axes, spreads.columns):
        ax.plot(spreads.index, spreads[ticker], lw=0.8, color="steelblue")
        ax.axhline(0, color="grey", lw=0.5)
        ev = events[events["ticker"] == ticker]
        ax.scatter(
            ev[ev["side"] == "discount"]["Date"],
            ev[ev["side"] == "discount"]["premium_discount"],
            color="red", s=15, zorder=3, label="discount event",
        )
        ax.scatter(
            ev[ev["side"] == "premium"]["Date"],
            ev[ev["side"] == "premium"]["premium_discount"],
            color="green", s=15, zorder=3, label="premium event",
        )
        for label, (start, end) in STRESS_EPISODES.items():
            # Skip episodes outside the sample (a rolling 5y window may no
            # longer include Mar 2020) so axvspan doesn't stretch the axis.
            if pd.Timestamp(end) < spreads.index.min():
                continue
            ax.axvspan(pd.Timestamp(start), pd.Timestamp(end), alpha=0.15,
                       color="orange")
        ax.set_ylabel(f"{ticker}\n% prem/disc", fontsize=8)
    for label, (start, _) in STRESS_EPISODES.items():
        if pd.Timestamp(start) < spreads.index.min():
            continue
        axes[0].annotate(
            label, xy=(pd.Timestamp(start), axes[0].get_ylim()[1]),
            fontsize=7, rotation=0, va="bottom", color="darkorange",
        )
    axes[0].legend(loc="upper right", fontsize=7)
    fig.suptitle("ETF premium/discount history (events at |z| > 2)")
    fig.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
        logger.info("Saved %s", save_path)
    return fig


def main() -> None:
    """Compute spreads/z-scores/events, save processed CSVs and the chart."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    spreads, zscores, events = build_spread_panel()
    spreads.to_csv(PROCESSED_DIR / "premium_discount.csv")
    zscores.to_csv(PROCESSED_DIR / "zscore.csv")
    events.to_csv(PROCESSED_DIR / "events.csv", index=False)
    logger.info("Saved processed spread data to %s", PROCESSED_DIR)
    plot_spread_history(spreads, events, OUTPUTS_DIR / "spread_history.png")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    main()
