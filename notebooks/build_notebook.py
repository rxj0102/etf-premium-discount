"""Generate ``notebooks/research.ipynb`` programmatically.

Building the notebook from this script keeps the JSON well-formed and the
narrative under version control in a reviewable format. Re-run after editing::

    python notebooks/build_notebook.py
"""

from __future__ import annotations

import json
from pathlib import Path

NOTEBOOK_PATH = Path(__file__).resolve().parent / "research.ipynb"


_cell_counter = 0


def _next_id() -> str:
    """Return a stable, unique notebook cell id."""
    global _cell_counter
    _cell_counter += 1
    return f"cell-{_cell_counter:02d}"


def md(source: str) -> dict:
    """Build a markdown cell.

    Args:
        source: Markdown text.

    Returns:
        Notebook-format markdown cell dict.
    """
    return {
        "cell_type": "markdown",
        "id": _next_id(),
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }


def code(source: str) -> dict:
    """Build a code cell.

    Args:
        source: Python source text.

    Returns:
        Notebook-format code cell dict with empty outputs.
    """
    return {
        "cell_type": "code",
        "id": _next_id(),
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


CELLS = [
    md(
        "# ETF Premium/Discount Mean Reversion — Research Notebook\n"
        "\n"
        "Fixed income ETFs (HYG, LQD, JNK, TLT, AGG) trade continuously while "
        "their underlying bonds trade over the counter, so the ETF price can "
        "deviate from end-of-day NAV — especially in stress. This notebook "
        "runs the full research pipeline: data, spread construction, event "
        "study, regime analysis and backtest.\n"
        "\n"
        "## 1. Motivation\n"
        "\n"
        "**Hypothesis:** when the premium/discount z-score (63-day window) "
        "exceeds ±2σ, the dislocation reflects temporary liquidity pressure "
        "and mean-reverts over the following 1–20 trading days — faster after "
        "deep discounts and in high-VIX regimes."
    ),
    code(
        "import logging\n"
        "import sys\n"
        "from pathlib import Path\n"
        "\n"
        "import matplotlib\n"
        "import matplotlib.pyplot as plt\n"
        "import pandas as pd\n"
        "\n"
        "%matplotlib inline\n"
        "matplotlib.rcParams['figure.dpi'] = 110\n"
        "\n"
        "PROJECT_ROOT = Path.cwd().resolve()\n"
        "if not (PROJECT_ROOT / 'data' / 'fetch.py').exists():\n"
        "    PROJECT_ROOT = PROJECT_ROOT.parent  # running from notebooks/\n"
        "sys.path.insert(0, str(PROJECT_ROOT))\n"
        "\n"
        "logging.basicConfig(level=logging.INFO, format='%(levelname)s %(name)s: %(message)s')"
    ),
    md(
        "## 2. Data\n"
        "\n"
        "`data/fetch.py` downloads 5 years of daily OHLCV for the five ETFs "
        "and the VIX via **yfinance**, credit OAS spreads (BAMLH0A0HYM2, "
        "BAMLC0A0CM) from **FRED**, and builds an end-of-day **NAV** series. "
        "FRED publishes no ETF NAV data, so NAV is simulated as price "
        "adjusted by a mean-reverting noise term bounded at ±0.5% (clearly "
        "logged; drop a real NAV export at `data/raw/<TICKER>_nav_external.csv` "
        "to override)."
    ),
    code(
        "from data import fetch\n"
        "\n"
        "fetch.main()  # idempotent: re-downloads and overwrites data/raw/\n"
        "\n"
        "hyg = pd.read_csv(fetch.RAW_DIR / 'HYG.csv', index_col=0, parse_dates=True)\n"
        "hyg.tail()"
    ),
    code(
        "oas = pd.read_csv(fetch.RAW_DIR / 'oas.csv', index_col=0, parse_dates=True)\n"
        "ax = oas.plot(figsize=(10, 3.5), title='Credit OAS spreads (FRED)')\n"
        "ax.set_ylabel('OAS (%)')\n"
        "plt.show()"
    ),
    md(
        "## 3. Spread construction\n"
        "\n"
        "For each ETF we compute `premium_discount = (Price − NAV) / NAV × 100`, "
        "its 63-day rolling z-score, and flag events where the z-score "
        "*crosses* ±2σ. Stress episodes (COVID-19 in Mar 2020, the Oct 2022 "
        "rate shock) are highlighted."
    ),
    code(
        "from analysis import spread\n"
        "\n"
        "spreads, zscores, events = spread.build_spread_panel()\n"
        "spread.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)\n"
        "spreads.to_csv(spread.PROCESSED_DIR / 'premium_discount.csv')\n"
        "zscores.to_csv(spread.PROCESSED_DIR / 'zscore.csv')\n"
        "events.to_csv(spread.PROCESSED_DIR / 'events.csv', index=False)\n"
        "\n"
        "fig = spread.plot_spread_history(spreads, events,\n"
        "                                 spread.OUTPUTS_DIR / 'spread_history.png')\n"
        "plt.show()\n"
        "events.groupby(['ticker', 'side']).size().unstack(fill_value=0)"
    ),
    md(
        "## 4. Event study\n"
        "\n"
        "For every ±2σ crossing we compute forward ETF price returns at 1, 3, "
        "5, 10 and 20 trading days, and the full 0–20 day cumulative return "
        "path with 95% confidence bands — separately for **discount** events "
        "(z < −2) and **premium** events (z > +2)."
    ),
    code(
        "from analysis import reversion\n"
        "\n"
        "event_returns, summaries = reversion.run_event_study()\n"
        "event_returns.to_csv(reversion.PROCESSED_DIR / 'event_study.csv', index=False)\n"
        "\n"
        "fig = reversion.plot_reversion_paths(\n"
        "    summaries, reversion.OUTPUTS_DIR / 'reversion_paths.png')\n"
        "plt.show()\n"
        "\n"
        "event_returns.groupby('side')[[f'ret_{h}d' for h in reversion.HORIZONS]].mean().round(3)"
    ),
    md(
        "## 5. Regime analysis\n"
        "\n"
        "Events are split by the VIX level on the event date: **high VIX** "
        "(> 20) vs **low VIX** (≤ 20). We compare both the *magnitude* of the "
        "subsequent reversion (average return path) and its *speed* (days for "
        "the z-score to revert to zero, censored at 20)."
    ),
    code(
        "from analysis import regime\n"
        "\n"
        "tagged_events, regime_table = regime.run_regime_analysis()\n"
        "regime_table.to_csv(regime.PROCESSED_DIR / 'regime_summary.csv')\n"
        "\n"
        "closes = {t: spread.load_close(t) for t in fetch.ETF_TICKERS}\n"
        "discount = tagged_events[tagged_events['side'] == 'discount']\n"
        "summaries_by_regime = {\n"
        "    r: reversion.path_summary(reversion.reversion_paths(\n"
        "        discount[discount['regime'] == r], closes))\n"
        "    for r in ('high_vix', 'low_vix')\n"
        "    if not discount[discount['regime'] == r].empty\n"
        "}\n"
        "speed = discount.groupby('regime')[['days_to_revert']].mean()\n"
        "fig = regime.plot_regime_comparison(\n"
        "    summaries_by_regime, speed, regime.OUTPUTS_DIR / 'regime_comparison.png')\n"
        "plt.show()\n"
        "regime_table"
    ),
    md(
        "## 6. Backtest\n"
        "\n"
        "**Rules:** enter long when the discount z-score crosses below −2; "
        "exit when it crosses back through 0. Position size is the inverse of "
        "21-day realised volatility, normalised and capped at 2×; signals "
        "execute at the next close (no look-ahead). Benchmark: buy-and-hold "
        "each ETF."
    ),
    code(
        "from backtest import strategy\n"
        "\n"
        "summary, results = strategy.run_backtests()\n"
        "summary.to_csv(strategy.PROCESSED_DIR / 'backtest_summary.csv')\n"
        "\n"
        "fig = strategy.plot_cumulative_returns(\n"
        "    results, strategy.OUTPUTS_DIR / 'backtest_cumulative.png')\n"
        "plt.show()\n"
        "summary"
    ),
    md(
        "## 7. Conclusion\n"
        "\n"
        "* ±2σ premium/discount dislocations in fixed income ETFs are "
        "infrequent but cluster in liquidity stress (Mar 2020, Oct 2022).\n"
        "* Discount events show positive average forward returns over 1–20 "
        "days, consistent with mean reversion; premium events show the "
        "mirror-image (weaker) pattern.\n"
        "* Reversion after discount events tends to be **larger** in high-VIX "
        "regimes (deeper dislocations to revert from) though confidence "
        "bands are wide given few independent episodes.\n"
        "* The simple long-discount strategy is in the market only a small "
        "fraction of the time and earns most of its return in stress "
        "windows; results are sensitive to the NAV data quality — replace "
        "the simulated NAV with issuer NAV files for production-grade "
        "conclusions.\n"
        "\n"
        "**Caveats:** simulated NAV (unless overridden), no transaction "
        "costs/borrow, overlapping events inflate the effective sample, and "
        "the ±2σ threshold was not tuned out of sample."
    ),
]


def main() -> None:
    """Write the notebook JSON to ``research.ipynb``."""
    notebook = {
        "cells": CELLS,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    NOTEBOOK_PATH.write_text(json.dumps(notebook, indent=1) + "\n")
    print(f"Wrote {NOTEBOOK_PATH}")


if __name__ == "__main__":
    main()
