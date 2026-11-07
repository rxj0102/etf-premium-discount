import numpy as np
import pandas as pd
import pytest

from analysis.spread import compute_premium_discount, flag_events, rolling_zscore
from backtest.strategy import (
    backtest_etf,
    inverse_vol_weights,
    max_drawdown,
    signal_from_zscore,
    trade_stats,
)

IDX = pd.bdate_range("2021-01-01", periods=400)


def test_signal_enters_below_entry_and_exits_at_zero():
    z = pd.Series([0, -1, -2.5, -1.5, -0.5, 0.1, -3, 0.2], dtype=float)
    assert signal_from_zscore(z).tolist() == [0, 0, 1, 1, 1, 0, 1, 0]


def test_signal_holds_through_nan():
    z = pd.Series([0, -3, np.nan, -1, 0.5], dtype=float)
    assert signal_from_zscore(z).tolist() == [0, 1, 1, 1, 0]


def test_inverse_vol_weights_do_not_use_future_volatility():
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0, 0.01, 400), index=IDX)
    base = inverse_vol_weights(r)
    shocked = r.copy()
    shocked.iloc[-1] = 0.5  # future shock must not change earlier weights
    alt = inverse_vol_weights(shocked)
    assert np.allclose(base.iloc[:-1], alt.iloc[:-1])
    assert base.max() <= 2.0 + 1e-12 and (base >= 0).all()


def test_zscore_is_point_in_time():
    rng = np.random.default_rng(1)
    s = pd.Series(rng.normal(size=300), index=IDX[:300])
    z = rolling_zscore(s, 63)
    z_trunc = rolling_zscore(s.iloc[:200], 63)
    assert np.allclose(z.iloc[:200].dropna(), z_trunc.dropna())


def test_flag_events_counts_crossings_not_levels():
    z = pd.DataFrame({"X": [0, -1, -2.1, -2.5, -2.2, -1, 0, 2.2, 2.5, 1]}, dtype=float)
    spreads = z.copy()
    ev = flag_events(z, spreads)
    assert (ev["side"] == "discount").sum() == 1 and (ev["side"] == "premium").sum() == 1


def test_premium_discount_sign():
    close = pd.Series([99.0, 101.0], index=IDX[:2], name="X")
    nav = pd.Series([100.0, 100.0], index=IDX[:2])
    assert compute_premium_discount(close, nav).round(6).tolist() == [-1.0, 1.0]


def test_signal_cannot_earn_the_return_it_is_formed_on():
    """A z-score that dips on a +10% day must not capture that day's return."""
    n = 300
    close = pd.Series(100.0, index=IDX[:n])
    jump_day = 150
    close.iloc[jump_day:] = 110.0                 # +10% on jump_day
    z = pd.Series(0.0, index=close.index)
    z.iloc[jump_day] = -3.0                       # signal observed at that close
    z.iloc[jump_day + 1:] = 1.0                   # exits next day
    out = backtest_etf(close, z)
    assert out["strategy_returns"].abs().max() < 1e-12  # never captured the jump
    # the old convention (exec_lag=1) would credit the t+1 return, still not the jump
    assert backtest_etf(close, z, exec_lag=1)["strategy_returns"].abs().max() < 1e-12


def test_executed_one_close_after_signal():
    n = 300
    rng = np.random.default_rng(2)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, n))), index=IDX[:n])
    z = pd.Series(0.0, index=close.index)
    z.iloc[200] = -3.0
    z.iloc[205:] = 1.0
    out = backtest_etf(close, z)
    pos = out["position"]
    assert pos.iloc[:202].eq(0).all() and pos.iloc[202] > 0     # first held day is t+2
    assert pos.iloc[206:].eq(0).all()


def test_trade_stats_and_drawdown():
    pos = pd.Series([0, 1, 1, 0, 1, 0], dtype=float)
    ret = pd.Series([0, 0.1, 0.1, 0, -0.05, 0])
    hit, avg_hold, n = trade_stats(pos, ret)
    assert n == 2 and hit == 0.5 and avg_hold == 1.5
    assert max_drawdown((1 + pd.Series([0.1, -0.2, 0.1])).cumprod()) == pytest.approx(-0.2)
