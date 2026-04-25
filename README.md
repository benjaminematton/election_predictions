# oath_score

Oath-style 1–10 **Impact Score** for U.S. House candidates, combining
**competitiveness** (P[margin < 5%] from multi-quantile regression on
predicted vote margin), **stakes** (chamber-control swing from a correlated
Monte Carlo), and a **financial-need** adjustment (quantile-regression
viable-spend floor). The score is back-tested at three pre-election snapshots
(T-110, T-60, T-20) per cycle, then served via a Streamlit UI.

**Status: Phase 2 — ingestion + features complete.**
Snapshot-aware ingestion for ACS demographics, FEC contributions and Schedule E
independent expenditures, MIT Election Lab district results, Cook/Sabato/Inside
race ratings (via MediaWiki revision API), and Daily Kos CPVI (both pre- and
post-2020 maps). End-to-end `features.build_features(cycle, snapshot)` joins
all six sources and applies the contested-race filter. Model code (Phases 4–7)
and Streamlit UI (Phase 8) not yet built.

**Cycles covered:** 2014, 2016, 2022, 2024.

**Snapshots per cycle:** T-110 (~July 15), T-60 (~Sept 1), T-20 (~mid-Oct).

**Live demo (Phase 8 deliverable):** _coming soon_ — Streamlit Community Cloud URL TBD.

## Layout

```
src/oath_score/
  constants.py        # cycles, snapshot offsets, election dates
  feature_sets.py     # parent-chain registry driving the improvement curve
  features.py         # join + contested-race filter → per-(cycle, snapshot) matrix
  ingest/             # _download, census, fec, fec_ie, opensecrets, pvi, ratings, results
  scores/             # Phase 4-6: competitiveness, stakes, financial_need, impact (TBD)
tests/                # 78 tests, ~250 LOC, no network
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[ml,ui,dev]"
export CENSUS_API_KEY=...   # https://api.census.gov/data/key_signup.html
```

## Run the tests

```bash
pytest -q
```

## Run the ingestion pipeline (downloads ~1GB FEC bulk files)

```bash
python -m oath_score.ingest.fec --cycle 2024 --snapshot 2024-09-06
python -m oath_score.features --cycle 2024 --snapshot T-60
```

## Full plan

See `~/.claude/plans/review-this-project-i-imperative-comet.md` (v3) for the
complete 9-phase methodology, the multi-snapshot backtest design, and the
Cook-final-ratings benchmark used as the success ceiling.
