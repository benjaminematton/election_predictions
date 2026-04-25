# oath_score

Oath-style 1–10 **Impact Score** for U.S. House candidates, combining
**competitiveness** (P[margin < 5%] from multi-quantile regression on
predicted vote margin), **stakes** (chamber-control swing from a correlated
Monte Carlo), and a **financial-need** adjustment (quantile-regression
viable-spend floor). The score is back-tested at three pre-election snapshots
(T-110, T-60, T-20) per cycle, then served via a Streamlit UI.

**Status: Phase 1 — skeleton complete.** Package layout, ingestion module
refactor (Census ACS, FEC bulk contributions, OpenSecrets dark-money), feature-set
registry for the improvement curve, and snapshot-date arithmetic are in place.
Smoke-tested via `tests/`. No model code yet.

**Cycles covered:** 2014, 2016, 2022, 2024.

**Snapshots per cycle:** T-110 (~July 15), T-60 (~Sept 1), T-20 (~mid-Oct).

**Live demo (Phase 8 deliverable):** _coming soon_ — Streamlit Community Cloud URL TBD.

## Layout

```
src/oath_score/
  constants.py        # cycles, snapshot offsets, election dates
  feature_sets.py     # parent-chain registry driving the improvement curve
  ingest/             # census, fec, opensecrets (Phase 2 will add fec_ie, results, ratings, pvi)
  scores/             # Phase 4-6: competitiveness, stakes, financial_need, impact
tests/                # snapshot-date and feature-ladder pins
```

## Run the smoke tests

```bash
pytest -q
```

## Full plan

See `~/.claude/plans/review-this-project-i-imperative-comet.md` (v3) for the
complete 9-phase methodology, the multi-snapshot backtest design, and the
Cook-final-ratings benchmark used as the success ceiling.
