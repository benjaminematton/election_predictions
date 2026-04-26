"""Streamlit app: Oath-style Impact Score for the 2024 U.S. House.

A simulation, NOT a solicitation. The deployed app loads a precomputed
parquet baked by `scripts/bake_app_data.py` and never re-runs the heavy
compute (multi-quantile fit, MC, FinancialNeed). All scoring decisions
happened offline at the Phase 7 calibrated config:

  * full feature set, multi-quantile model
  * combine='impact' = sqrt(competitiveness × stakes) · (1 + 0.3 · need)
  * train cycles = (2016, 2022)
  * universe = all (Dem candidates in any contested two-party general)

This file is the Streamlit entry point. The repo-root `streamlit_app.py`
shim imports `main` here.
"""

from __future__ import annotations

from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

# We can't import oath_score.allocation directly when this file is loaded
# from a Streamlit Cloud worker that hasn't put src/ on PYTHONPATH; but the
# repo-root shim handles that. For local dev we rely on PYTHONPATH=src.
from oath_score.allocation import allocate, metric_pct_to_close_races
from oath_score.scores.deciling import DecileThresholds


# ----- constants -----

DATA_PATH = Path("data/processed/app_candidates_2024.parquet")
DECILE_PATH = Path("data/processed/decile_thresholds.json")

GITHUB_URL = "https://github.com/benjaminematton/election_predictions"
HEADLINE_NOTEBOOK = f"{GITHUB_URL}/blob/main/notebooks/03_backtest_curves.ipynb"
CALIBRATION_NOTEBOOK = f"{GITHUB_URL}/blob/main/notebooks/05_calibration_results.ipynb"

DEFAULT_SNAPSHOT = "T-60"
SNAPSHOTS = ("T-110", "T-60", "T-20")
SNAPSHOT_LABELS = {
    "T-110": "T-110 (≈ July 15, primary season)",
    "T-60":  "T-60 (≈ Sept 1, post-Labor Day)",
    "T-20":  "T-20 (≈ Oct 15, final stretch)",
}


# ----- data loading -----

@st.cache_data
def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        st.error(
            f"App parquet not found at {DATA_PATH}. Run "
            "`python scripts/bake_app_data.py` to generate it."
        )
        st.stop()
    df = pd.read_parquet(DATA_PATH)
    return df


@st.cache_data
def load_deciles() -> DecileThresholds:
    return DecileThresholds.load(DECILE_PATH)


# ----- sections -----

def _disclaimer_banner() -> None:
    st.warning(
        "**This is a simulation, not a solicitation.** "
        "This tool simulates how Oath-style donor allocation would distribute a hypothetical "
        "budget across U.S. House candidates based on a backtested impact-scoring model. "
        "It does not collect, route, or process donations. No real money is involved.",
        icon="ℹ️",
    )


def _headline() -> None:
    st.title("Oath-style Impact Score · U.S. House 2024")
    st.markdown(
        "A 1–10 score per Democratic House candidate combining **competitiveness**, "
        "**chamber-control stakes**, and **financial need**. Backtested on held-out "
        "2024 results: **89.8% of allocated dollars** to <5%-margin races at T-60 days "
        f"before Election Day vs **29.9% for Cook Political Report's final ratings** "
        "([methodology]({}))."
        .format(HEADLINE_NOTEBOOK)
    )


def _sidebar(df: pd.DataFrame) -> tuple[str, list[str], tuple[int, int]]:
    st.sidebar.header("Filters")

    snapshot = st.sidebar.radio(
        "Pre-election snapshot",
        SNAPSHOTS,
        index=SNAPSHOTS.index(DEFAULT_SNAPSHOT),
        format_func=lambda s: SNAPSHOT_LABELS[s],
    )

    states_in_data = sorted(df["state_abbr"].dropna().unique().tolist())
    state_filter = st.sidebar.multiselect(
        "States",
        states_in_data,
        default=[],
        help="Empty = all states.",
    )

    decile_min, decile_max = st.sidebar.slider(
        "Impact decile (1 = lowest, 10 = highest)",
        min_value=1, max_value=10,
        value=(1, 10),
    )

    return snapshot, state_filter, (decile_min, decile_max)


def _filter_df(
    df: pd.DataFrame,
    snapshot: str,
    states: list[str],
    decile_range: tuple[int, int],
) -> pd.DataFrame:
    out = df[df["snapshot"] == snapshot].copy()
    if states:
        out = out[out["state_abbr"].isin(states)]
    out = out[
        (out["impact_decile"] >= decile_range[0])
        & (out["impact_decile"] <= decile_range[1])
    ]
    return out.sort_values("impact_continuous", ascending=False).reset_index(drop=True)


def _splitter_section(filtered: pd.DataFrame, snapshot: str) -> None:
    st.header("Splitter — your $X across top-N candidates")
    st.markdown(
        "If a donor with a fixed budget allocated their dollars by Impact Score across the "
        "top-N candidates from the filter above, this is what the breakdown would look like. "
        "Shows the simulated allocation, the share that went to <5%-margin races (backtest "
        "validation), and the share to chamber-pivotal seats."
    )

    if filtered.empty:
        st.info("No candidates match the current filter — adjust state/decile filters.")
        return

    col1, col2 = st.columns(2)
    with col1:
        budget = st.number_input("Budget ($)", min_value=10, max_value=10_000, value=100, step=10)
    with col2:
        n = st.slider("Top-N candidates", min_value=1, max_value=min(20, len(filtered)),
                      value=min(10, len(filtered)))

    alloc = allocate(filtered, score_col="impact_continuous", n=n, total_dollars=float(budget))

    # Render bar chart
    chart_df = alloc.copy()
    chart_df["label"] = chart_df["state_abbr"] + "-" + chart_df["district"].astype(str) + " " + chart_df["last_name"]
    chart = alt.Chart(chart_df).mark_bar(color="#3366cc").encode(
        x=alt.X("allocation:Q", title="Simulated allocation ($)"),
        y=alt.Y("label:N", sort="-x", title=None),
        tooltip=[
            alt.Tooltip("label:N", title="Candidate"),
            alt.Tooltip("allocation:Q", title="Allocation ($)", format=".2f"),
            alt.Tooltip("impact_continuous:Q", title="Impact (continuous)", format=".3f"),
            alt.Tooltip("impact_decile:Q", title="Decile"),
            alt.Tooltip("predicted_margin_median:Q", title="Predicted margin", format="+.3f"),
        ],
    ).properties(height=max(200, n * 25))
    st.altair_chart(chart, width="stretch")

    # Summary metrics (backtest validation only — these use ACTUAL 2024 outcomes,
    # so it's a sanity check, not a forecast)
    close_share = metric_pct_to_close_races(alloc, margin_col="margin_pct", threshold=0.05)
    pivotal_share = (
        (alloc["allocation"] * (alloc["stakes_raw"].abs() > 0.05).astype(float)).sum()
        / max(alloc["allocation"].sum(), 1e-9)
    )
    underfloor_share = (
        (alloc["allocation"] * (alloc["total_trans"] < alloc["viable_floor"]).astype(float)).sum()
        / max(alloc["allocation"].sum(), 1e-9)
    )

    cols = st.columns(3)
    cols[0].metric("→ races finishing <5% margin", f"{close_share:.0%}",
                   help="Backtest validation: of the $ allocated, how much went to races that "
                        "actually ended within 5%.")
    cols[1].metric("→ chamber-pivotal seats", f"{pivotal_share:.0%}",
                   help="% of $ to seats whose flip would shift D's expected House total above- "
                        "or below-median across MC simulations.")
    cols[2].metric("→ underfunded candidates", f"{underfloor_share:.0%}",
                   help="% of $ to candidates whose snapshot fundraising is below the predicted "
                        "viable floor for their race.")


def _ranked_table_section(filtered: pd.DataFrame) -> None:
    st.header("Ranked candidates")
    st.markdown(f"**{len(filtered)}** candidates match the current filter, ranked by Impact Score.")

    display_cols = {
        "impact_decile": st.column_config.NumberColumn("1–10", format="%d"),
        "state_abbr": "State",
        "district": st.column_config.NumberColumn("District", format="%d"),
        "last_name": "Candidate",
        "party_major": "Party",
        "competitiveness": st.column_config.NumberColumn("Comp", format="%.3f"),
        "stakes_normalized": st.column_config.NumberColumn("Stakes", format="%.3f"),
        "need_raw": st.column_config.NumberColumn("Need", format="%.3f"),
        "impact_continuous": st.column_config.NumberColumn("Impact (raw)", format="%.3f"),
        "predicted_margin_median": st.column_config.NumberColumn("Pred margin", format="%+.3f"),
    }
    cols = [c for c in display_cols if c in filtered.columns]
    st.dataframe(
        filtered[cols],
        column_config={k: display_cols[k] for k in cols},
        hide_index=True,
        width="stretch",
        height=400,
    )


def _detail_section(filtered: pd.DataFrame) -> None:
    if filtered.empty:
        return
    st.header("Per-candidate detail")
    options = filtered.apply(
        lambda r: f"{r['state_abbr']}-{int(r['district']):02d}  {r['last_name']}  (decile {int(r['impact_decile'])})",
        axis=1,
    ).tolist()
    selected = st.selectbox("Pick a candidate to inspect", options)
    if not selected:
        return

    row_idx = options.index(selected)
    cand = filtered.iloc[row_idx]

    cols = st.columns(3)
    cols[0].metric("Competitiveness", f"{cand['competitiveness']:.3f}",
                   help="P(margin within ±5%) from multi-quantile regression.")
    cols[1].metric("Stakes", f"{cand['stakes_normalized']:.3f}",
                   help="Min-max normalized chamber-pivot probability from MC.")
    cols[2].metric("Need", f"{cand['need_raw']:.3f}",
                   help="(viable_floor − own_spend) / viable_floor, clipped to [0, 1].")

    cols = st.columns(2)
    cols[0].metric("Impact (1–10)", int(cand["impact_decile"]))
    cols[1].metric("Impact (continuous)", f"{cand['impact_continuous']:.3f}")

    # Fundraising vs viable floor
    fund_df = pd.DataFrame({
        "label": ["Snapshot fundraising", "Viable floor (model)"],
        "amount": [cand["total_trans"], cand["viable_floor"]],
    })
    fund_chart = alt.Chart(fund_df).mark_bar().encode(
        x=alt.X("amount:Q", title="$"),
        y=alt.Y("label:N", title=None),
        color=alt.Color("label:N", legend=None,
                        scale=alt.Scale(range=["#3366cc", "#cc6633"])),
    ).properties(height=120, title="Fundraising vs predicted viable floor")
    st.altair_chart(fund_chart, width="stretch")

    # Race context
    facts = {
        "Race": f"{cand['state_abbr']}-{int(cand['district']):02d}",
        "Party": cand["party_major"],
        "Incumbent": "Yes" if cand.get("incumbent") == 1 else "No",
        "Cook rating (snapshot)": f"{cand['cook_rating']:.1f}" if pd.notna(cand.get("cook_rating")) else "Not in Wikipedia table",
        "Cook PVI": f"{cand['cpvi']:+.1f}" if pd.notna(cand.get("cpvi")) else "—",
        "Predicted median margin": f"{cand['predicted_margin_median']:+.3f}",
        "Actual margin (2024)": f"{cand['margin_pct']:+.3f}",
        "Snapshot fundraising": f"${cand['total_trans']:,.0f}",
        "IE for": f"${cand['ie_for_total']:,.0f}",
        "IE against": f"${cand['ie_against_total']:,.0f}",
    }
    st.markdown("**Race context:**")
    fact_df = pd.DataFrame({"Field": list(facts), "Value": list(facts.values())})
    st.dataframe(fact_df, hide_index=True, width="stretch")


def _about_section() -> None:
    with st.expander("About this app — methodology & honesty notes", expanded=False):
        st.markdown(
            f"""
**What this is.** A backtested impact-scoring algorithm for U.S. House races, modeled after
[Oath](https://oath.vote)'s donor-allocation product. Three sub-scores combined into a 1–10 Impact Score:

* **Competitiveness** — multi-quantile regression on signed margin (D% − R%) → empirical CDF → P(\\|margin\\| < 5%).
* **Chamber-control stakes** — correlated Monte Carlo (10,000 iterations, snapshot-dependent σ) on per-race predictive distributions; per-seat partisan-pivot probability.
* **Financial need** — 25th-percentile-of-winners viable-floor quantile regression on close-race fundraising. Need = (floor − own spend) / floor.

`impact_continuous = sqrt(comp × stakes) · (1 + 0.3 · need)` → mapped to 1–10 via frozen training-cycle decile cutpoints.

**What this isn't.** A solicitation, a forecast for future cycles, or a recommendation to donate. The 2024 numbers shown are backtest validation: actual outcomes feed the "% to <5% races" metric. For 2026 use, the model would need refitting on the new cycle's snapshot data.

**Data sources** (all public, automatable):
* Census ACS (demographics)
* FEC bulk filings + Schedule E (with snapshot-date discipline)
* MIT Election Lab (district results)
* Wikipedia revision history (snapshot-aware Cook/Sabato/Inside ratings)

**Backtest highlights** (held-out 2024, train on 2016 + 2022):
* T-60: top-decile recommendations targeted <5% races in **89.8% of cases** vs **29.9% for Cook's final ratings** (+60pp lift).
* T-110: 100% close-race share (with wider bootstrap CI).
* T-20: 80.8% close-race share, still +51pp over Cook-final.

**Repo & methodology:** [GitHub]({GITHUB_URL}) · [Improvement curves notebook]({HEADLINE_NOTEBOOK}) · [Calibration results]({CALIBRATION_NOTEBOOK}).

**Caveats.**
1. The naive baseline operates on a Wikipedia-tracked subset of races; richer feature sets use the full Dem universe with PVI-based cook imputation.
2. The chamber MC uses a dynamic-median threshold rather than the literal 218 majority — 2024's MC distribution centered at ~169 D seats, well below 218. Stakes is the donor-relevant "above- vs below-median" question.
3. α = 0.3 is the LOO-cross-validated optimum across the training cycles. Higher α directs more dollars to underfunded candidates at the cost of close-race precision.
"""
        )


# ----- entry point -----

def main() -> None:
    st.set_page_config(
        page_title="Oath-style Impact Score · 2024 House",
        page_icon="🗳️",
        layout="wide",
    )

    df = load_data()
    deciles = load_deciles()  # noqa: F841 — loaded for cache; used implicitly by baked data

    _disclaimer_banner()
    _headline()
    snapshot, state_filter, decile_range = _sidebar(df)
    filtered = _filter_df(df, snapshot, state_filter, decile_range)

    _splitter_section(filtered, snapshot)
    st.divider()
    _ranked_table_section(filtered)
    st.divider()
    _detail_section(filtered)
    st.divider()
    _about_section()


if __name__ == "__main__":
    main()
