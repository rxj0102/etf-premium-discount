"""Event study of mean reversion after +/-2 sigma premium/discount events.

For every threshold crossing flagged by ``analysis/spread.py`` this module
computes forward ETF price returns at 1, 3, 5, 10 and 20 trading-day
horizons, and the full day-0..20 cumulative return path. Average paths with
95% confidence bands are computed separately for discount events (z < -2,
the long candidates) and premium events (z > +2).

Outputs:

* ``data/processed/event_study.csv``  per-event forward returns
* ``data/processed/reversion_summary.csv``  mean forward returns by side
* ``outputs/reversion_paths.png``  average reversion path chart

Usage::

    python analysis/reversion.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from analysis.spread import (  # noqa: E402
    OUTPUTS_DIR,
    PROCESSED_DIR,
    build_spread_panel,
    load_adj_close,
)
from data.fetch import ETF_TICKERS  # noqa: E402

logger = logging.getLogger(__name__)

HORIZONS = [1, 3, 5, 10, 20]
MAX_HORIZON = 20


def forward_returns_for_events(
    events: pd.DataFrame, closes: dict[str, pd.Series]
) -> pd.DataFrame:
    """Attach forward price returns at each horizon to the event table.

    Args:
        events: Event table with Date/ticker/side columns
            (see ``analysis.spread.flag_events``).
        closes: Mapping of ticker to closing price series.

    Returns:
        Copy of ``events`` with columns ``ret_{h}d`` (simple returns, in
        percent) for each horizon in ``HORIZONS``. Events too close to the
        end of the sample have NaN for unavailable horizons.
    """
    out = events.copy()
    for h in HORIZONS:
        out[f"ret_{h}d"] = np.nan
    for i, row in out.iterrows():
        close = closes[row["ticker"]]
        if row["Date"] not in close.index:
            continue
        pos = close.index.get_loc(row["Date"])
        for h in HORIZONS:
            if pos + h < len(close):
                out.loc[i, f"ret_{h}d"] = (
                    close.iloc[pos + h] / close.iloc[pos] - 1
                ) * 100
    return out


def reversion_paths(
    events: pd.DataFrame,
    closes: dict[str, pd.Series],
    max_horizon: int = MAX_HORIZON,
) -> pd.DataFrame:
    """Build the day-0..N cumulative return path for every event.

    Args:
        events: Event table with Date and ticker columns.
        closes: Mapping of ticker to closing price series.
        max_horizon: Path length in trading days (default 20).

    Returns:
        DataFrame with one row per event and columns 0..max_horizon holding
        cumulative simple returns in percent (day 0 is always 0). Rows whose
        path would run past the end of the sample are dropped.
    """
    paths = []
    for _, row in events.iterrows():
        close = closes[row["ticker"]]
        if row["Date"] not in close.index:
            continue
        pos = close.index.get_loc(row["Date"])
        if pos + max_horizon >= len(close):
            continue
        window = close.iloc[pos : pos + max_horizon + 1]
        paths.append((window / window.iloc[0] - 1).to_numpy() * 100)
    return pd.DataFrame(paths, columns=range(max_horizon + 1))


def path_summary(paths: pd.DataFrame) -> pd.DataFrame:
    """Compute the mean path with a 95% confidence band.

    Args:
        paths: Per-event path frame from :func:`reversion_paths`.

    Returns:
        DataFrame indexed by horizon day with columns ``mean``, ``lower`` and
        ``upper`` (mean +/- 1.96 standard errors), plus ``n`` (event count).
    """
    n = len(paths)
    mean = paths.mean()
    sem = paths.std() / np.sqrt(max(n, 1))
    return pd.DataFrame(
        {"mean": mean, "lower": mean - 1.96 * sem, "upper": mean + 1.96 * sem,
         "n": n}
    )


def plot_reversion_paths(
    summaries: dict[str, pd.DataFrame], save_path: Path | None = None
) -> plt.Figure:
    """Plot average post-event return paths with confidence bands.

    Args:
        summaries: Mapping of side ('discount'/'premium') to the summary
            frame from :func:`path_summary`.
        save_path: Optional PNG destination.

    Returns:
        The matplotlib figure.
    """
    fig, axes = plt.subplots(1, len(summaries), figsize=(11, 4), sharey=True)
    axes = np.atleast_1d(axes)
    colors = {"discount": "tab:red", "premium": "tab:green"}
    for ax, (side, summary) in zip(axes, summaries.items()):
        color = colors.get(side, "tab:blue")
        ax.plot(summary.index, summary["mean"], color=color,
                label=f"mean (n={int(summary['n'].iloc[0])})")
        ax.fill_between(summary.index, summary["lower"], summary["upper"],
                        color=color, alpha=0.2, label="95% CI")
        ax.axhline(0, color="grey", lw=0.6)
        ax.set_title(f"{side} events (z {'<' if side == 'discount' else '>'} "
                     f"{'-2' if side == 'discount' else '+2'})")
        ax.set_xlabel("Trading days after event")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Cumulative price return (%)")
    fig.suptitle("Average post-event return path with 95% confidence bands")
    fig.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
        logger.info("Saved %s", save_path)
    return fig


def run_event_study() -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Run the full event study across all ETFs.

    Returns:
        Tuple of (event_returns, summaries): the per-event forward return
        table and a mapping of side to average path summary.
    """
    _, _, events = build_spread_panel()
    closes = {t: load_adj_close(t) for t in ETF_TICKERS}
    event_returns = forward_returns_for_events(events, closes)
    summaries = {}
    for side in ("discount", "premium"):
        side_events = events[events["side"] == side]
        if side_events.empty:
            logger.warning("No %s events in sample; skipping path", side)
            continue
        summaries[side] = path_summary(reversion_paths(side_events, closes))
    return event_returns, summaries


def main() -> None:
    """Run the event study, save processed tables and the path chart."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    event_returns, summaries = run_event_study()
    event_returns.to_csv(PROCESSED_DIR / "event_study.csv", index=False)

    summary = (
        event_returns.groupby("side")[[f"ret_{h}d" for h in HORIZONS]]
        .mean()
        .round(3)
    )
    summary.to_csv(PROCESSED_DIR / "reversion_summary.csv")
    logger.info("Mean forward returns by side (%%):\n%s", summary.to_string())

    plot_reversion_paths(summaries, OUTPUTS_DIR / "reversion_paths.png")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    main()
