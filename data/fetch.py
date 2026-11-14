"""Download and persist all raw market data for the ETF premium/discount study.

Fetches, for the past five years:

* Daily OHLCV prices for the fixed income ETFs HYG, LQD, JNK, TLT and AGG
  (via yfinance).
* Daily VIX (^VIX) closes (via yfinance).
* End-of-day NAV per ETF. FRED does not publish ETF NAV series, so NAV is
  simulated as price adjusted by a mean-reverting (Ornstein-Uhlenbeck) noise
  term bounded at +/-0.5%, amplified during known stress windows (Mar 2020,
  Oct 2022) so that the premium/discount series exhibits realistic stress
  behaviour. If a file ``data/raw/<TICKER>_nav_external.csv`` exists (a
  user-supplied NAV series with columns ``Date,NAV``), it is used instead.
* Credit OAS spreads (BAMLH0A0HYM2 high yield, BAMLC0A0CM investment grade)
  from FRED, via fredapi when ``FRED_API_KEY`` is set, otherwise via FRED's
  public ``fredgraph.csv`` endpoint which requires no key.

If the network is unavailable entirely, every series falls back to a clearly
logged, reproducible simulation so the rest of the pipeline still runs
end-to-end (useful for CI and offline development).

All outputs are written to ``data/raw/`` as CSV files indexed by date:

* ``<TICKER>.csv``       OHLCV prices
* ``<TICKER>_nav.csv``   end-of-day NAV
* ``vix.csv``            VIX close
* ``oas.csv``            credit OAS spreads (percent)

Usage::

    python data/fetch.py
"""

from __future__ import annotations

import logging
import os
import zlib
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"

ETF_TICKERS = ["HYG", "LQD", "JNK", "TLT", "AGG"]
VIX_TICKER = "^VIX"
FRED_OAS_SERIES = {
    "BAMLH0A0HYM2": "hy_oas",  # ICE BofA US High Yield OAS
    "BAMLC0A0CM": "ig_oas",    # ICE BofA US Corporate (IG) OAS
}
LOOKBACK_YEARS = 5

# Stress windows where simulated NAV noise is amplified so the synthetic
# premium/discount series shows dislocations where the real ones occurred.
STRESS_WINDOWS = [
    ("2020-02-20", "2020-04-15"),  # COVID-19 liquidity crunch
    ("2022-09-15", "2022-10-31"),  # 2022 rate shock / gilt crisis spillover
]

# Annualised volatility / drift used only by the offline price simulator.
SIM_PARAMS = {
    "HYG": {"vol": 0.08, "drift": 0.03, "s0": 78.0},
    "LQD": {"vol": 0.09, "drift": 0.02, "s0": 110.0},
    "JNK": {"vol": 0.08, "drift": 0.03, "s0": 95.0},
    "TLT": {"vol": 0.15, "drift": 0.00, "s0": 100.0},
    "AGG": {"vol": 0.05, "drift": 0.01, "s0": 100.0},
}


def _rng_for(name: str) -> np.random.Generator:
    """Return a reproducible random generator seeded from a string.

    Args:
        name: Any string (e.g. a ticker symbol).

    Returns:
        A ``numpy.random.Generator`` whose seed is a stable CRC32 hash of
        ``name``, so simulations are reproducible across runs and machines.
    """
    return np.random.default_rng(zlib.crc32(name.encode()))


def _date_range() -> tuple[str, str]:
    """Return the (start, end) ISO date strings for the lookback window.

    Returns:
        Tuple of start and end dates spanning ``LOOKBACK_YEARS`` years up to
        today.
    """
    end = date.today()
    start = end - timedelta(days=int(365.25 * LOOKBACK_YEARS))
    return start.isoformat(), end.isoformat()


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse the MultiIndex columns yfinance returns for single tickers.

    Args:
        df: Raw frame returned by ``yfinance.download``.

    Returns:
        The same frame with single-level column labels.
    """
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)
    return df


def simulate_ohlcv(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Simulate a daily OHLCV history for a ticker (offline fallback).

    A geometric Brownian motion close series is generated with per-ticker
    volatility/drift from ``SIM_PARAMS``; open/high/low are derived from the
    close with small intraday noise. Used only when yfinance is unreachable.

    Args:
        ticker: ETF ticker symbol.
        start: ISO start date.
        end: ISO end date.

    Returns:
        DataFrame with columns Open, High, Low, Close, Adj Close, Volume
        indexed by business day.
    """
    params = SIM_PARAMS.get(ticker, {"vol": 0.10, "drift": 0.0, "s0": 100.0})
    rng = _rng_for(ticker)
    idx = pd.bdate_range(start, end, name="Date")
    n = len(idx)
    daily_vol = params["vol"] / np.sqrt(252)
    rets = rng.normal(params["drift"] / 252, daily_vol, n)
    close = params["s0"] * np.exp(np.cumsum(rets))
    intraday = np.abs(rng.normal(0, daily_vol / 2, n))
    open_ = close * (1 + rng.normal(0, daily_vol / 3, n))
    high = np.maximum(open_, close) * (1 + intraday)
    low = np.minimum(open_, close) * (1 - intraday)
    volume = rng.integers(2_000_000, 30_000_000, n)
    return pd.DataFrame(
        {
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": close,
            "Adj Close": close,
            "Volume": volume,
        },
        index=idx,
    )


def fetch_ohlcv(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Fetch daily OHLCV data for one ticker via yfinance.

    Falls back to :func:`simulate_ohlcv` (with a prominent warning) if the
    download fails or comes back empty, so the pipeline still runs offline.

    Args:
        ticker: Ticker symbol understood by Yahoo Finance.
        start: ISO start date.
        end: ISO end date.

    Returns:
        DataFrame of OHLCV data indexed by date.
    """
    try:
        import yfinance as yf

        df = yf.download(
            ticker, start=start, end=end, auto_adjust=False, progress=False
        )
        df = _flatten_columns(df)
        if df.empty:
            raise ValueError(f"yfinance returned no rows for {ticker}")
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df.index.name = "Date"
        logger.info("Fetched %d rows of %s from yfinance", len(df), ticker)
        return df
    except Exception as exc:  # network down, rate limit, delisted, ...
        logger.warning(
            "yfinance fetch failed for %s (%s); using SIMULATED prices. "
            "Re-run with network access for real data.",
            ticker,
            exc,
        )
        return simulate_ohlcv(ticker, start, end)


def fetch_vix(start: str, end: str) -> pd.Series:
    """Fetch the VIX daily close series.

    Falls back to a reproducible lognormal simulation (mean ~18, with spikes
    in the configured stress windows) when yfinance is unreachable.

    Args:
        start: ISO start date.
        end: ISO end date.

    Returns:
        Series of VIX closes named ``vix`` indexed by date.
    """
    df = fetch_ohlcv(VIX_TICKER, start, end)
    vix = df["Close"].rename("vix")
    if VIX_TICKER not in SIM_PARAMS and float(vix.iloc[0]) < 5:
        # The generic simulator produced a price-like path, not a vol index;
        # replace with a level series centred on typical VIX values.
        rng = _rng_for("VIX")
        level = 18 * np.exp(rng.normal(0, 0.05, len(vix)).cumsum() * 0.3)
        vix = pd.Series(np.clip(level, 10, 80), index=vix.index, name="vix")
    for start_w, end_w in STRESS_WINDOWS:
        mask = (vix.index >= start_w) & (vix.index <= end_w)
        if mask.any() and vix[mask].max() < 25:
            # Only applies to simulated data: inject stress-level vol so the
            # downstream regime split has a populated high-VIX bucket.
            vix.loc[mask] = vix.loc[mask] * 2.2
    return vix


def simulate_nav(close: pd.Series, ticker: str) -> pd.Series:
    """Simulate end-of-day NAV from the price series.

    FRED does not publish ETF NAV series, so NAV is modelled as

        NAV_t = Price_t * (1 - s_t)

    where ``s_t`` (the premium/discount fraction) follows a discrete
    Ornstein-Uhlenbeck process with a ~5 day half-life, clipped to +/-0.5%.
    Inside ``STRESS_WINDOWS`` the noise volatility is amplified 4x so the
    synthetic series exhibits the dislocation/reversion dynamics this project
    studies. The process is seeded per ticker for reproducibility.

    Args:
        close: ETF closing price series indexed by date.
        ticker: Ticker symbol (used only for the random seed).

    Returns:
        Simulated NAV series named ``nav`` aligned to ``close``.
    """
    rng = _rng_for(ticker + "_nav")
    theta = 0.15          # mean-reversion speed (half-life ~ ln2/theta ~ 4.6d)
    sigma = 0.0008        # shock vol -> stationary std ~ 0.15%
    stress = pd.Series(False, index=close.index)
    for start_w, end_w in STRESS_WINDOWS:
        stress |= (close.index >= start_w) & (close.index <= end_w)

    s = np.zeros(len(close))
    for t in range(1, len(close)):
        shock_vol = sigma * (4.0 if stress.iloc[t] else 1.0)
        s[t] = (1 - theta) * s[t - 1] + rng.normal(0, shock_vol)
    s = np.clip(s, -0.005, 0.005)  # bound the premium/discount at +/-0.5%
    return pd.Series(close.values * (1 - s), index=close.index, name="nav")


def fetch_nav(ticker: str, close: pd.Series) -> pd.Series:
    """Return the NAV series for an ETF, preferring user-supplied data.

    Lookup order:

    1. ``data/raw/<TICKER>_nav_external.csv`` (columns ``Date,NAV``) — drop a
       real NAV export here (e.g. from the issuer's website) to use it.
    2. Simulated NAV via :func:`simulate_nav` (FRED has no ETF NAV series).

    Args:
        ticker: ETF ticker symbol.
        close: Closing price series used as the simulation anchor.

    Returns:
        NAV series named ``nav`` aligned to ``close``.
    """
    external = RAW_DIR / f"{ticker}_nav_external.csv"
    if external.exists():
        logger.info("Using external NAV file for %s: %s", ticker, external)
        nav = pd.read_csv(external, index_col=0, parse_dates=True).iloc[:, 0]
        return nav.reindex(close.index).ffill().rename("nav")
    logger.info(
        "No external NAV for %s; simulating NAV = price * (1 - OU noise, "
        "clipped to +/-0.5%%). See simulate_nav docstring.",
        ticker,
    )
    return simulate_nav(close, ticker)


def fetch_fred_series(series_id: str, start: str) -> pd.Series | None:
    """Fetch one FRED series, with key-less fallback.

    Tries ``fredapi`` when the ``FRED_API_KEY`` environment variable is set;
    otherwise (or on failure) falls back to FRED's public ``fredgraph.csv``
    endpoint which needs no API key.

    Args:
        series_id: FRED series identifier (e.g. ``BAMLH0A0HYM2``).
        start: ISO start date used to trim the series.

    Returns:
        The series indexed by date, or ``None`` if both methods fail.
    """
    api_key = os.environ.get("FRED_API_KEY")
    if api_key:
        try:
            from fredapi import Fred

            series = Fred(api_key=api_key).get_series(series_id)
            series.index = pd.to_datetime(series.index)
            logger.info("Fetched %s via fredapi", series_id)
            return series.loc[series.index >= start]
        except Exception as exc:
            logger.warning("fredapi failed for %s (%s); trying CSV", series_id, exc)
    try:
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
        df = pd.read_csv(url, index_col=0, parse_dates=True, na_values=".")
        series = df.iloc[:, 0].astype(float)
        logger.info("Fetched %s via fredgraph.csv (no API key)", series_id)
        return series.loc[series.index >= start]
    except Exception as exc:
        logger.warning("FRED CSV fallback failed for %s: %s", series_id, exc)
        return None


def fetch_oas(start: str, end: str) -> pd.DataFrame:
    """Fetch HY and IG credit OAS spreads from FRED.

    Series that cannot be fetched are simulated with a reproducible
    mean-reverting process around typical levels (HY ~4%, IG ~1.3%) so the
    research notebook can still render offline; this is clearly logged.

    Args:
        start: ISO start date.
        end: ISO end date.

    Returns:
        DataFrame with columns ``hy_oas`` and ``ig_oas`` (percent) indexed by
        date.
    """
    idx = pd.bdate_range(start, end, name="Date")
    out = {}
    defaults = {"hy_oas": 4.0, "ig_oas": 1.3}
    for series_id, name in FRED_OAS_SERIES.items():
        series = fetch_fred_series(series_id, start)
        if series is None:
            logger.warning(
                "Simulating %s (%s): FRED unreachable. Set FRED_API_KEY or "
                "enable network access for real data.",
                name,
                series_id,
            )
            rng = _rng_for(name)
            level = defaults[name]
            shocks = rng.normal(0, 0.02, len(idx))
            path = np.zeros(len(idx))
            path[0] = level
            for t in range(1, len(idx)):
                path[t] = path[t - 1] + 0.05 * (level - path[t - 1]) + shocks[t]
            series = pd.Series(np.clip(path, 0.3, 12), index=idx)
        out[name] = series
    df = pd.DataFrame(out)
    df.index.name = "Date"
    return df.loc[(df.index >= start) & (df.index <= end)]


def main() -> None:
    """Fetch all raw datasets and write them to ``data/raw/`` as CSVs."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    start, end = _date_range()
    logger.info("Fetching data from %s to %s into %s", start, end, RAW_DIR)

    for ticker in ETF_TICKERS:
        ohlcv = fetch_ohlcv(ticker, start, end)
        ohlcv.to_csv(RAW_DIR / f"{ticker}.csv")
        nav = fetch_nav(ticker, ohlcv["Close"])
        nav.to_frame().to_csv(RAW_DIR / f"{ticker}_nav.csv")
        logger.info("Saved %s prices and NAV (%d rows)", ticker, len(ohlcv))

    vix = fetch_vix(start, end)
    vix.to_frame().to_csv(RAW_DIR / "vix.csv")
    logger.info("Saved VIX (%d rows)", len(vix))

    oas = fetch_oas(start, end)
    oas.to_csv(RAW_DIR / "oas.csv")
    logger.info("Saved OAS spreads (%d rows)", len(oas))

    # 3-month T-bill yield (percent) for interest on cash in the backtest.
    # Not simulated if FRED is unreachable: the backtest then credits no
    # interest and says so.
    tbill = fetch_fred_series("DGS3MO", start)
    if tbill is not None:
        tbill = tbill.loc[(tbill.index >= start) & (tbill.index <= end)]
        tbill.rename("tbill_pct").to_frame().rename_axis("Date").to_csv(RAW_DIR / "tbill.csv")
        logger.info("Saved T-bill yield (%d rows)", len(tbill))
    else:
        logger.warning("T-bill yield unavailable; backtest will credit no interest on cash")
    logger.info("Done. Raw data written to %s", RAW_DIR)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    main()
