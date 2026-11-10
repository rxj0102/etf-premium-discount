# ETF Premium/Discount Mean Reversion

[![CI](https://github.com/rxj0102/etf-premium-discount/actions/workflows/ci.yml/badge.svg)](https://github.com/rxj0102/etf-premium-discount/actions/workflows/ci.yml)

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

A research project analyzing **premium/discount mean reversion in fixed
income ETFs** (HYG, LQD, JNK, TLT, AGG). Fixed income ETFs trade
continuously while their underlying bonds trade over the counter and often
stale-price, so the ETF price can deviate materially from its end-of-day NAV
during stress. This project measures those dislocations as a 63-day rolling
z-score of the premium/discount, runs an event study on ±2σ episodes,
conditions the results on the VIX regime, and backtests a simple long-only
discount-reversion strategy with a Streamlit monitoring dashboard.

## Hypothesis

> When a fixed income ETF's premium/discount to NAV exceeds two standard
> deviations of its recent (63-day) history, the dislocation reflects
> temporary liquidity pressure rather than a change in fundamental value, and
> therefore mean-reverts over the following 1–20 trading days — faster and
> more profitably after deep **discounts** and during **high-VIX** regimes.

## Folder structure

```
etf-premium-discount/
├── data/
│   ├── fetch.py            # Download prices, VIX, NAV, OAS → data/raw/
│   ├── raw/                # Raw CSVs (git-ignored, reproducible)
│   └── processed/          # Derived tables (git-ignored)
├── analysis/
│   ├── spread.py           # Premium/discount, 63d z-score, ±2σ events
│   ├── reversion.py        # Event study: forward returns & reversion paths
│   └── regime.py           # High-VIX vs low-VIX regime comparison
├── backtest/
│   └── strategy.py         # Long-discount strategy backtest vs buy & hold
├── dashboard/
│   └── app.py              # Streamlit monitoring dashboard
├── notebooks/
│   └── research.ipynb      # Full narrative research notebook
├── outputs/                # Saved PNG charts
├── requirements.txt
└── README.md
```

## Setup

```bash
# 1. Install dependencies (Python 3.10+)
pip install -r requirements.txt

# 2. (Optional) set a FRED API key for credit OAS data
export FRED_API_KEY=your_key_here   # falls back to FRED's public CSV endpoint

# 3. Fetch raw data (≈5 years of daily data)
python data/fetch.py

# 4. Run the analysis pipeline (each step saves CSVs + PNG charts)
python analysis/spread.py
python analysis/reversion.py
python analysis/regime.py
python backtest/strategy.py

# 5. Launch the dashboard
streamlit run dashboard/app.py

# 6. Or explore the narrative notebook
jupyter notebook notebooks/research.ipynb
```

## Key findings

*(Placeholder — regenerate after running the pipeline on live data.)*

- Spread history with stress episodes:

  ![Spread history](outputs/spread_history.png)

- Average post-event reversion paths (95% CI):

  ![Reversion paths](outputs/reversion_paths.png)

- High-VIX vs low-VIX regime comparison:

  ![Regime comparison](outputs/regime_comparison.png)

- Strategy vs buy-and-hold cumulative returns:

  ![Backtest](outputs/backtest_cumulative.png)

## Data sources

| Dataset | Source | Notes |
|---|---|---|
| ETF OHLCV prices (HYG, LQD, JNK, TLT, AGG) | Yahoo Finance via `yfinance` | 5 years daily |
| VIX (^VIX) | Yahoo Finance via `yfinance` | Daily close |
| ETF NAV | Simulated (or user-supplied) | FRED publishes no ETF NAV series; NAV is modelled as price adjusted by a mean-reverting noise term bounded at ±0.5% (amplified in Mar 2020 / Oct 2022 stress windows). Drop a real NAV export at `data/raw/<TICKER>_nav_external.csv` to override. |
| Credit OAS (BAMLH0A0HYM2, BAMLC0A0CM) | FRED via `fredapi` | Key-less fallback to FRED's public `fredgraph.csv` endpoint |

If the network is unavailable, `data/fetch.py` falls back to clearly logged,
reproducible simulations so the full pipeline still runs end to end.

## Reproducibility

### Data availability

All inputs come from free, public sources. No paid data or API keys are required unless noted.

- Yahoo Finance via the yfinance package (free, no key)
- FRED, Federal Reserve Bank of St. Louis (free)
- Files shipped in the `data/` folder of this repository

### Computational requirements

- Python 3.12.3. Every dependency version is pinned in `uv.lock`.
- Checked on Linux x86_64 with 2 CPU cores and 8 GB RAM. No GPU is needed.

```bash
pip install uv     # once
uv sync            # installs the exact pinned environment
uv run pytest -q   # runs the test suite
```

### Runtime

The test suite (9 tests) finishes in about 9 seconds on the machine above, including any data downloads it triggers.

### Citation

Citation metadata is in `CITATION.cff`. GitHub turns it into a Cite this repository button on the repo page.


## License

MIT — see [LICENSE](LICENSE).

## Corrections and caveats

- **Backtest timing.** Despite the documented "next-day close" execution, positions were shifted by one day, so a signal formed from day-t price and NAV earned the day-t+1 return (a fill at the very close that produced the signal). Execution is now lagged by two days (`EXEC_LAG`); pass `exec_lag=1` to reproduce the old convention.
- **Position sizing look-ahead.** Inverse-volatility weights were rescaled by the full-sample mean of inverse volatility; they now use the expanding mean.
- **The NAV series is simulated** unless you supply `data/raw/<TICKER>_nav_external.csv`. With simulated NAV the premium/discount is noise independent of subsequent price moves, so the backtest and event study demonstrate the pipeline, not a tradable edge. Add real issuer NAV data before drawing any conclusion from `backtest_summary.csv` or `event_study.csv`; regenerate them after the timing fix.
- Event-study confidence bands treat events as independent although 20-day windows overlap, so they are too narrow.
- `pytest` and CI added.
