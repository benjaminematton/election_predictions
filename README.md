# oath_score · Oath-style Impact Score for U.S. House 2024

A 1–10 candidate-impact algorithm modeled after [Oath](https://oath.vote)'s donor-allocation product, combining three sub-scores into a single Impact Score per Democratic House candidate:

* **Competitiveness** — direct multi-quantile regression on signed margin (D% − R%); empirical predictive CDF; P(\|margin\| < 5%).
* **Chamber-control stakes** — correlated Monte Carlo (10k iterations, snapshot-dependent σ); per-seat partisan-pivot probability via single-MC-run decomposition.
* **Financial need** — 25th-percentile-of-winners viable-floor quantile regression; need = (floor − own snapshot spend) / floor.

**Live demo:** _add Streamlit Community Cloud URL here after deploy._ First request may take ~30 seconds while Streamlit Cloud spins up the container.

## Resume bullet (held-out 2024 backtest)

> Built and back-tested an Oath-style political-impact-scoring algorithm for U.S. House races. At T-60 days before Election 2024, top-decile recommendations targeted contests decided by <5% in **89.8% of cases** — **beating Cook Political Report's final ratings by 60pp** under identical allocation logic, using only public, automatable inputs (Census ACS, FEC bulk filings, MIT Election Lab, Wikipedia revision-history ratings).

Full numbers across snapshots (final calibrated config: train on 2016 + 2022, full feature set, multi-quantile, combine=impact, α = 0.3, top-N = 10):

| Snapshot | Model close-race | Fundraising baseline | Cook-final benchmark | Δ vs Cook |
|---|---:|---:|---:|---:|
| T-110 | 1.000 | 0.310 | 0.299 | **+0.701** |
| T-60 | 0.898 | 0.267 | 0.299 | **+0.600** |
| T-20 | 0.808 | 0.464 | 0.299 | **+0.510** |

## Methodology

* **Snapshot-aware everything.** All time-varying inputs (FEC contributions, Schedule E IE, Cook ratings) filtered to a fixed pre-election date so the backtest reflects what was knowable mid-cycle. Three snapshots per cycle: T-110, T-60, T-20.
* **District-agnostic features** survive 2022 redistricting (Cook rating, CPVI of *current* district, ACS demographics, snapshot fundraising, opponent / IE features).
* **Wikipedia ratings via revision API** for snapshot-aware Cook/Sabato/Inside ordinals — the back-fill caveat for 2014/2016 is documented in code.
* **Hindsight oracle and Cook-final benchmark** as reference lines, both run through the same allocation function as the model — apples-to-apples comparison.
* **Bootstrap 95% CIs** on every backtest row's headline metric.

Notebooks (browse without running):

* [`notebooks/03_backtest_curves.ipynb`](notebooks/03_backtest_curves.ipynb) — improvement curves across feature sets × snapshots × model classes; combined-score panels.
* [`notebooks/02_competitiveness_diagnostics.ipynb`](notebooks/02_competitiveness_diagnostics.ipynb) — Q-Q, calibration, residuals.
* [`notebooks/04_stakes_diagnostics.ipynb`](notebooks/04_stakes_diagnostics.ipynb) — chamber-composition distribution, top-stakes seats.
* [`notebooks/05_calibration_results.ipynb`](notebooks/05_calibration_results.ipynb) — α grid, N sensitivity, 2014 ablation, decile cutpoints.

## Local development

```bash
python3.13 -m venv .venv
.venv/bin/pip install -r requirements.txt
echo "CENSUS_API_KEY=<your-key>" > .env

# Run the deployed-app form against the precomputed parquet
PYTHONPATH=src .venv/bin/streamlit run streamlit_app.py
```

To regenerate the deployed-app parquet from raw data (requires Census API key + ~10 GB of FEC bulk downloads):

```bash
set -a; source .env; set +a
./scripts/build_all_features.sh        # ingest, features (per cycle × snapshot)
./scripts/run_improvement_curve.sh logistic
./scripts/run_improvement_curve.sh multi-quantile
PYTHONPATH=src .venv/bin/python -m oath_score.calibration  # picks α*, writes deciles
PYTHONPATH=src .venv/bin/python scripts/bake_app_data.py   # writes app parquet
```

## Repo layout

```
streamlit_app.py            # Streamlit Cloud entry point (auto-detected)
src/oath_score/
├── ingest/                # FEC, Census, Schedule E, MIT, Wikipedia ratings, Daily Kos
├── features.py            # join + contested-race filter + signed margin
├── feature_sets.py        # registry mapping flag → feature columns
├── scores/
│   ├── competitiveness.py # multi-quantile + logistic baselines
│   ├── stakes.py          # correlated-MC simulator
│   ├── chamber.py         # 435-seat House view
│   ├── financial_need.py  # viable-floor quantile regression
│   ├── impact.py          # combine sub-scores
│   └── deciling.py        # frozen 1-10 thresholds
├── allocation.py          # top-N score-weighted with optional need-cap
├── backtest.py            # train/test loop, bootstrap CI, Cook-final benchmark
├── calibration.py         # α grid, N sweep, 2014 ablation
└── app.py                 # Streamlit page
```

## Honest caveats

1. **This is a simulation, not a solicitation.** No donations are collected or routed.
2. The 2024 percentages above are *backtest validation* using actual outcomes — not a forecast. For 2026 use, the model would need refitting on 2026 snapshot data.
3. **2014 was dropped** from the final training set per the LOO ablation — pre-Trump-era patterns don't transfer to 2022/2024.
4. Bootstrap CIs are wide at top-N=10 because of small-sample noise (33 close races among 320 contested Dems in 2024).

## License

MIT (see [`LICENSE`](LICENSE)).
