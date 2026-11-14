"""Volatility regime analysis of premium/discount mean reversion.

Splits the +/-2 sigma events flagged by ``analysis/spread.py`` into high-VIX
(VIX > 20 at the event date) and low-VIX (VIX <= 20) regimes and compares:

* *Magnitude*: the average cumulative price return path (day 0..20) after
  discount events, with 95% confidence bands, per regime.
* *Speed*: the average number of trading days for the premium/discount
  z-score to first cross back through zero (censored at 20 days).

Outputs:

* ``data/processed/regime_summary.csv``  per-regime statistics
* ``outputs/regime_comparison.png``      side-by-side regime chart

Usage::

    python analysis/regime.py
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

from analysis.reversion import (  # noqa: E402
    HORIZONS,
    MAX_HORIZON,
    forward_returns_for_events,
    path_summary,
    reversion_paths,
)
from analysis.spread import (  # noqa: E402
    OUTPUTS_DIR,
    PROCESSED_DIR,
    build_spread_panel,
    load_adj_close,
    load_vix,
)
from data.fetch import ETF_TICKERS  # noqa: E402

logger = logging.getLogger(__name__)

VIX_THRESHOLD = 20.0


def tag_regimes(events: pd.DataFrame, vix: pd.Series) -> pd.DataFrame:
    """Tag each event with its VIX regime.

    Args:
        events: Event table with a Date column.
        vix: VIX close series (forward-filled onto event dates).

    Returns:
        Copy of ``events`` with ``vix`` and ``regime`` columns, where regime
        is ``'high_vix'`` if VIX > 20 on the event date else ``'low_vix'``.
    """
    out = events.copy()
    vix_aligned = vix.reindex(
        vix.index.union(pd.DatetimeIndex(out["Date"].unique()))
    ).ffill()
    out["vix"] = out["Date"].map(vix_aligned)
    out["regime"] = np.where(out["vix"] > VIX_THRESHOLD, "high_vix", "low_vix")
    logger.info(
        "Events per regime: %s", out["regime"].value_counts().to_dict()
    )
    return out


def days_to_zero_cross(
    events: pd.DataFrame, zscores: pd.DataFrame, max_horizon: int = MAX_HORIZON
) -> pd.Series:
    """Measure reversion speed: days until the z-score first crosses zero.

    Args:
        events: Event table with Date, ticker and side columns.
        zscores: Wide frame of rolling z-scores per ETF.
        max_horizon: Censoring horizon; events that have not reverted by then
            are recorded at ``max_horizon``.

    Returns:
        Series aligned to ``events.index`` with the (censored) number of
        trading days to the first zero crossing.
    """
    days = pd.Series(np.nan, index=events.index)
    for i, row in events.iterrows():
        z = zscores[row["ticker"]]
        if row["Date"] not in z.index:
            continue
        pos = z.index.get_loc(row["Date"])
        window = z.iloc[pos : pos + max_horizon + 1]
        # Discount events revert upward through 0; premium events downward.
        crossed = window >= 0 if row["side"] == "discount" else window <= 0
        hits = np.flatnonzero(crossed.to_numpy())
        days.loc[i] = float(hits[0]) if len(hits) else float(max_horizon)
    return days


def regime_summary(events: pd.DataFrame) -> pd.DataFrame:
    """Summarise reversion magnitude and speed per regime and side.

    Args:
        events: Event table that already has ``regime``, ``days_to_revert``
            and ``ret_{h}d`` columns.

    Returns:
        DataFrame indexed by (side, regime) with event counts, mean forward
        returns at each horizon, and mean days-to-reversion.
    """
    cols = [f"ret_{h}d" for h in HORIZONS] + ["days_to_revert"]
    summary = events.groupby(["side", "regime"]).agg(
        n=("Date", "size"), **{c: (c, "mean") for c in cols}
    )
    return summary.round(3)


def plot_regime_comparison(
    summaries: dict[str, pd.DataFrame],
    speed: pd.DataFrame,
    save_path: Path | None = None,
) -> plt.Figure:
    """Plot side-by-side regime comparison for discount events.

    Left/middle panels show the average post-event return path (with 95%
    bands) in the high- and low-VIX regimes; the right panel compares mean
    days-to-reversion per regime.

    Args:
        summaries: Mapping of regime name to a path summary frame
            (see ``analysis.reversion.path_summary``).
        speed: Frame indexed by regime with a ``days_to_revert`` column.
        save_path: Optional PNG destination.

    Returns:
        The matplotlib figure.
    """
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    colors = {"high_vix": "tab:red", "low_vix": "tab:blue"}
    for ax, (regime, summary) in zip(axes[:2], summaries.items()):
        color = colors.get(regime, "tab:grey")
        ax.plot(summary.index, summary["mean"], color=color,
                label=f"mean (n={int(summary['n'].iloc[0])})")
        ax.fill_between(summary.index, summary["lower"], summary["upper"],
                        color=color, alpha=0.2, label="95% CI")
        ax.axhline(0, color="grey", lw=0.6)
        ax.set_title(f"Discount events: {regime.replace('_', ' ')}")
        ax.set_xlabel("Trading days after event")
        ax.set_ylabel("Cumulative price return (%)")
        ax.legend(fontsize=8)
    bars = speed["days_to_revert"]
    axes[2].bar(bars.index, bars.values,
                color=[colors.get(r, "tab:grey") for r in bars.index])
    axes[2].set_title("Mean days for z-score to revert to 0")
    axes[2].set_ylabel("Trading days (censored at 20)")
    fig.suptitle(f"Mean reversion by VIX regime (threshold = {VIX_THRESHOLD:g})")
    fig.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
        logger.info("Saved %s", save_path)
    return fig


def run_regime_analysis() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the full regime analysis across all ETFs.

    Returns:
        Tuple of (tagged_events, summary): the event table with regime tags,
        forward returns and reversion speed, and the per-regime summary.
    """
    spreads, zscores, events = build_spread_panel()
    closes = {t: load_adj_close(t) for t in ETF_TICKERS}
    events = tag_regimes(events, load_vix())
    events = forward_returns_for_events(events, closes)
    events["days_to_revert"] = days_to_zero_cross(events, zscores)
    return events, regime_summary(events)


def main() -> None:
    """Run the regime analysis, save the summary table and comparison chart."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    events, summary = run_regime_analysis()
    summary.to_csv(PROCESSED_DIR / "regime_summary.csv")
    logger.info("Regime summary:\n%s", summary.to_string())

    closes = {t: load_adj_close(t) for t in ETF_TICKERS}
    discount = events[events["side"] == "discount"]
    summaries = {}
    for regime in ("high_vix", "low_vix"):
        subset = discount[discount["regime"] == regime]
        if subset.empty:
            logger.warning("No discount events in %s regime", regime)
            continue
        summaries[regime] = path_summary(reversion_paths(subset, closes))
    speed = (
        discount.groupby("regime")[["days_to_revert"]].mean()
        if not discount.empty
        else pd.DataFrame(columns=["days_to_revert"])
    )
    plot_regime_comparison(summaries, speed, OUTPUTS_DIR / "regime_comparison.png")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    main()
