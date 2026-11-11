"""Backtest of the discount mean-reversion strategy.

Strategy rules (long-only, per ETF):

* **Entry**: go long when the 63-day premium/discount z-score crosses below
  -2 (the fund trades at an unusually deep discount to NAV).
* **Exit**: flatten when the z-score crosses back up through 0.
* **Position sizing**: proportional to the inverse of 21-day rolling
  realised volatility, normalised by the sample-average inverse volatility
  and capped at 2x (normalised by the expanding mean, no future data), so the average deployed size is ~1x notional.
* Signals are executed at the next day's close (positions are shifted two
  days relative to the signal close, i.e. the signal earns the t+2 return) to avoid
  look-ahead bias.

Reported statistics per ETF, against a buy-and-hold benchmark: annualised
return, annualised Sharpe ratio, maximum drawdown, per-trade hit rate and
average holding period.

Outputs:

* ``data/processed/backtest_summary.csv``  summary statistics table
* ``data/processed/backtest_returns.csv``  daily strategy returns per ETF
* ``outputs/backtest_cumulative.png``      cumulative return chart vs benchmark

Usage::

    python backtest/strategy.py
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

ENTRY_Z = -2.0
EXIT_Z = 0.0
VOL_WINDOW = 21
MAX_LEVERAGE = 2.0
EXEC_LAG = 2  # days from the close that fixes the signal to the first day of P&L
COST_BPS = 5.0  # one-way cost per unit of turnover, basis points (same as etf-nav-arbitrage-study)
TRADING_DAYS = 252


def signal_from_zscore(
    z: pd.Series, entry: float = ENTRY_Z, exit_: float = EXIT_Z
) -> pd.Series:
    """Generate the long/flat state series from the z-score.

    Args:
        z: Rolling z-score of the premium/discount.
        entry: Enter long when z falls below this level (default -2).
        exit_: Exit when z rises to/above this level (default 0).

    Returns:
        Series of 0/1 position states (1 = long), same index as ``z``.
    """
    state = 0
    states = np.zeros(len(z))
    for i, value in enumerate(z.to_numpy()):
        if np.isnan(value):
            states[i] = state
            continue
        if state == 0 and value < entry:
            state = 1
        elif state == 1 and value >= exit_:
            state = 0
        states[i] = state
    return pd.Series(states, index=z.index, name="signal")


def inverse_vol_weights(
    returns: pd.Series, window: int = VOL_WINDOW, cap: float = MAX_LEVERAGE
) -> pd.Series:
    """Compute normalised inverse-volatility position weights.

    Weights are ``1 / rolling_vol`` rescaled by the *expanding* mean of the
    inverse volatility known at each date (so a typical position is ~1x and
    no future volatility leaks into today's size) and capped.

    Args:
        returns: Daily simple returns of the ETF.
        window: Rolling volatility window in days (default 21).
        cap: Maximum weight (default 2.0).

    Returns:
        Series of position weights aligned to ``returns``.
    """
    vol = returns.rolling(window).std()
    inv = 1.0 / vol.replace(0, np.nan)
    weights = inv / inv.expanding(min_periods=window).mean()
    return weights.clip(upper=cap).fillna(0.0)


def max_drawdown(cum: pd.Series) -> float:
    """Compute the maximum drawdown of a cumulative wealth series.

    Args:
        cum: Cumulative wealth series (starting near 1.0).

    Returns:
        Maximum peak-to-trough drawdown as a negative fraction.
    """
    return float((cum / cum.cummax() - 1).min())


def trade_stats(position: pd.Series, strat_ret: pd.Series) -> tuple[float, float, int]:
    """Compute per-trade hit rate, average holding period and trade count.

    Args:
        position: Daily position series (0 when flat, >0 when long).
        strat_ret: Daily strategy returns.

    Returns:
        Tuple of (hit_rate, avg_holding_days, n_trades). Hit rate is the
        fraction of round-trip trades with positive cumulative PnL.
    """
    in_pos = position > 0
    trade_id = (in_pos & ~in_pos.shift(1, fill_value=False)).cumsum()
    trade_id = trade_id.where(in_pos)
    pnls, lengths = [], []
    for _, chunk in strat_ret.groupby(trade_id):
        pnls.append((1 + chunk).prod() - 1)
        lengths.append(len(chunk))
    if not pnls:
        return np.nan, np.nan, 0
    hit_rate = float(np.mean([p > 0 for p in pnls]))
    return hit_rate, float(np.mean(lengths)), len(pnls)


def load_daily_rf(path: Path | None = None) -> pd.Series | None:
    """Daily cash return from the 3-month T-bill yield in ``data/raw/tbill.csv``.

    Returns None when the file is missing (FRED was unreachable at fetch time).
    The rate applied on day t is the one published at t-1.
    """
    path = path or (PROJECT_ROOT / "data" / "raw" / "tbill.csv")
    if not path.exists():
        return None
    s = pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0].astype(float) / 100.0
    return s


def _align_rf(rate: pd.Series | None, index: pd.DatetimeIndex) -> pd.Series:
    if rate is None:
        return pd.Series(0.0, index=index)
    r = rate.reindex(rate.index.union(index)).ffill().reindex(index).shift(1)
    return ((1.0 + r) ** (1.0 / TRADING_DAYS) - 1.0).fillna(0.0)


def _excess_sharpe(ret: pd.Series, rf: pd.Series) -> float:
    ex = ret - rf
    sd = ex.std()
    return float(ex.mean() / sd * np.sqrt(TRADING_DAYS)) if sd > 0 else np.nan


def backtest_etf(
    close: pd.Series,
    z: pd.Series,
    exec_lag: int = EXEC_LAG,
    cost_bps: float = COST_BPS,
    rf_rate: pd.Series | None = None,
    entry: float = ENTRY_Z,
    exit_: float = EXIT_Z,
) -> dict:
    """Backtest the discount-reversion strategy for one ETF.

    Args:
        close: Closing price series.
        z: Rolling z-score of the premium/discount, aligned to ``close``.
        exec_lag: Days between the close that produces the signal and the
            first return earned.  The z-score uses day-t price and NAV, so
            the earliest realistic fill is the *next* close: a signal at the
            close of t earns the return of t+2 (``exec_lag=2``).  ``1`` earns
            the t+1 return, i.e. assumes a fill at the very close that
            produced the signal.
        cost_bps: One-way transaction cost in basis points, charged on the
            absolute daily change in position (entries, exits and the daily
            inverse-volatility resizing while in a trade).
        rf_rate: Annual T-bill yield (decimal) indexed by date, or None. The
            share of capital not in the ETF (1 - position) earns the daily
            T-bill return; above 1x the excess is financed at the same rate.
        entry, exit_: z-score entry and exit levels (defaults -2 and 0).

    Returns:
        Dict with keys ``stats`` (summary statistics for strategy and
        benchmark), ``strategy_returns``, ``benchmark_returns`` and
        ``position`` (daily series).
    """
    close, z = close.align(z, join="inner")
    returns = close.pct_change().fillna(0.0)
    # A signal observed at the close of day t is filled at the close of t+1,
    # so it first earns the return of day t+2.
    position = (signal_from_zscore(z, entry, exit_) * inverse_vol_weights(returns)).shift(exec_lag)
    position = position.fillna(0.0)
    turnover = position.diff().abs().fillna(position.abs())
    cost = turnover * cost_bps / 1e4
    rf = _align_rf(rf_rate, returns.index)
    strat_ret = position * returns + (1.0 - position) * rf - cost

    n = len(strat_ret)
    cum = (1 + strat_ret).cumprod()
    bench_cum = (1 + returns).cumprod()
    ann_ret = float(cum.iloc[-1] ** (TRADING_DAYS / n) - 1)
    bench_ann = float(bench_cum.iloc[-1] ** (TRADING_DAYS / n) - 1)
    vol = strat_ret.std()
    sharpe = float(strat_ret.mean() / vol * np.sqrt(TRADING_DAYS)) if vol > 0 else np.nan
    bench_vol = returns.std()
    bench_sharpe = (
        float(returns.mean() / bench_vol * np.sqrt(TRADING_DAYS))
        if bench_vol > 0
        else np.nan
    )
    hit_rate, avg_hold, n_trades = trade_stats(position, strat_ret)

    stats = {
        "ann_return": ann_ret,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown(cum),
        "hit_rate": hit_rate,
        "avg_holding_days": avg_hold,
        "n_trades": n_trades,
        "ann_turnover": float(turnover.sum() * TRADING_DAYS / n),
        "cost_drag_ann": float(cost.sum() * TRADING_DAYS / n),
        "sharpe_excess": _excess_sharpe(strat_ret, rf),
        "bench_ann_return": bench_ann,
        "bench_sharpe_excess": _excess_sharpe(returns, rf),
        "bench_sharpe": bench_sharpe,
        "bench_max_drawdown": max_drawdown(bench_cum),
    }
    return {
        "stats": stats,
        "strategy_returns": strat_ret,
        "benchmark_returns": returns,
        "position": position,
    }


def run_backtests() -> tuple[pd.DataFrame, dict[str, dict]]:
    """Run the backtest for every ETF in the universe.

    Returns:
        Tuple of (summary, results): a summary statistics frame indexed by
        ticker and the per-ticker result dicts from :func:`backtest_etf`.
    """
    _, zscores, _ = build_spread_panel()
    rf_rate = load_daily_rf()
    if rf_rate is None:
        logger.warning("No data/raw/tbill.csv; cash earns zero in this run")
    results = {}
    for ticker in ETF_TICKERS:
        results[ticker] = backtest_etf(load_adj_close(ticker), zscores[ticker], rf_rate=rf_rate)
        logger.info(
            "%s: sharpe=%.2f, ann_return=%.2f%%",
            ticker,
            results[ticker]["stats"]["sharpe"],
            results[ticker]["stats"]["ann_return"] * 100,
        )
    summary = pd.DataFrame({t: r["stats"] for t, r in results.items()}).T
    return summary.round(3), results


def plot_cumulative_returns(
    results: dict[str, dict], save_path: Path | None = None
) -> plt.Figure:
    """Plot strategy vs buy-and-hold cumulative returns per ETF.

    Args:
        results: Per-ticker result dicts from :func:`backtest_etf`.
        save_path: Optional PNG destination.

    Returns:
        The matplotlib figure.
    """
    fig, axes = plt.subplots(
        len(results), 1, figsize=(11, 2.2 * len(results)), sharex=True
    )
    axes = np.atleast_1d(axes)
    for ax, (ticker, res) in zip(axes, results.items()):
        strat_cum = (1 + res["strategy_returns"]).cumprod()
        bench_cum = (1 + res["benchmark_returns"]).cumprod()
        ax.plot(strat_cum.index, strat_cum, label="strategy", color="tab:blue")
        ax.plot(bench_cum.index, bench_cum, label="buy & hold",
                color="tab:grey", alpha=0.8)
        ax.set_ylabel(ticker, fontsize=9)
        ax.legend(fontsize=7, loc="upper left")
    fig.suptitle("Discount mean-reversion strategy vs buy-and-hold")
    fig.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
        logger.info("Saved %s", save_path)
    return fig


ENTRY_GRID = (-1.5, -2.0, -2.5, -3.0)
EXIT_GRID = (-0.5, 0.0, 0.5)


def sensitivity_grid(entries=ENTRY_GRID, exits=EXIT_GRID) -> pd.DataFrame:
    """Mean excess Sharpe across ETFs for each entry/exit pair.

    Shown to judge whether the default (-2, 0) is a lucky setting. Nothing
    is chosen from this grid.
    """
    _, zscores, _ = build_spread_panel()
    rf_rate = load_daily_rf()
    rows = []
    for e in entries:
        for x in exits:
            sh = [
                backtest_etf(load_adj_close(t), zscores[t], rf_rate=rf_rate, entry=e, exit_=x)
                ["stats"]["sharpe_excess"]
                for t in ETF_TICKERS
            ]
            rows.append({"entry_z": e, "exit_z": x, "mean_sharpe_excess": float(np.nanmean(sh)),
                         "min_sharpe_excess": float(np.nanmin(sh)), "max_sharpe_excess": float(np.nanmax(sh))})
    return pd.DataFrame(rows)


def main() -> None:
    """Run all backtests, save outputs and log the summary table."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    summary, results = run_backtests()
    summary.to_csv(PROCESSED_DIR / "backtest_summary.csv")
    daily = pd.DataFrame({t: r["strategy_returns"] for t, r in results.items()})
    daily.to_csv(PROCESSED_DIR / "backtest_returns.csv")
    logger.info("Backtest summary:\n%s", summary.to_string())
    plot_cumulative_returns(results, OUTPUTS_DIR / "backtest_cumulative.png")
    grid = sensitivity_grid()
    grid.to_csv(PROCESSED_DIR / "sensitivity_grid.csv", index=False)
    logger.info("Sensitivity grid (mean excess Sharpe across ETFs):\n%s",
                grid.pivot(index="entry_z", columns="exit_z", values="mean_sharpe_excess").round(2))


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    main()
