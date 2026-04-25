"""Feature-set registry for the improvement-curve backtest.

Each named feature set selects a subset of columns from the per-candidate
matrix produced by `features.py`. The point: each row of `notebooks/
03_backtest_curves.ipynb` corresponds to one entry here. Adding a feature is
adding a row to the registry, not changing model code.

Order matters: feature sets are listed in the order they appear on the
improvement curve (naive → richer → richest).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class FeatureSet:
    name: str
    columns: tuple[str, ...]
    description: str = ""
    requires: tuple[str, ...] = field(default_factory=tuple)  # other feature_set names included


# Column names below are the canonical names produced by features.py.
# They correspond to:
#   cook_rating          ordinal 1-7 from "Solid R" to "Solid D"
#   incumbent            int (0/1)
#   cpvi                 cook partisan voting index, signed (D positive)
#   acs_*                ~33 demographic columns from census.py
#   self_raised_pct      candidate's snapshot-date receipts / (own + opponent)
#   self_raised_log      log1p of own snapshot-date total receipts
#   opp_raised_log       log1p of opponent's snapshot-date total receipts
#   ie_for_log           log1p of independent expenditures FOR (snapshot)
#   ie_against_log       log1p of independent expenditures AGAINST (snapshot)


_NAIVE = FeatureSet(
    name="naive",
    columns=("cook_rating", "incumbent"),
    description="Logistic baseline. Cook rating + incumbency only — what an analyst would predict in 30 seconds.",
)

_PVI = FeatureSet(
    name="naive+pvi",
    columns=_NAIVE.columns + ("cpvi",),
    description="Adds Cook PVI (district partisan lean).",
    requires=("naive",),
)

_DEMO = FeatureSet(
    name="naive+pvi+demo",
    columns=_PVI.columns + (
        "acs_median_age", "acs_race_white", "acs_race_black", "acs_race_asian",
        "acs_edu_bachelors", "acs_edu_grad_degree", "acs_median_income",
        "acs_below_100pct_fpl",
    ),
    description="Adds ACS demographics: race, education, income, poverty.",
    requires=("naive+pvi",),
)

_FUNDRAISING = FeatureSet(
    name="naive+pvi+demo+fund",
    columns=_DEMO.columns + ("self_raised_log", "self_raised_pct"),
    description="Adds snapshot-date candidate fundraising. SNAPSHOT-CRITICAL.",
    requires=("naive+pvi+demo",),
)

_OPPONENT = FeatureSet(
    name="naive+pvi+demo+fund+opp",
    columns=_FUNDRAISING.columns + ("opp_raised_log",),
    description="Adds opponent fundraising at snapshot.",
    requires=("naive+pvi+demo+fund",),
)

_IE = FeatureSet(
    name="full",
    columns=_OPPONENT.columns + ("ie_for_log", "ie_against_log"),
    description="Full feature set: adds independent-expenditure totals (FEC Schedule E).",
    requires=("naive+pvi+demo+fund+opp",),
)


REGISTRY: dict[str, FeatureSet] = {
    fs.name: fs for fs in (_NAIVE, _PVI, _DEMO, _FUNDRAISING, _OPPONENT, _IE)
}

# Order to plot on the improvement curve, left → right.
CURVE_ORDER: tuple[str, ...] = (
    "naive",
    "naive+pvi",
    "naive+pvi+demo",
    "naive+pvi+demo+fund",
    "naive+pvi+demo+fund+opp",
    "full",
)


def get(name: str) -> FeatureSet:
    if name not in REGISTRY:
        raise KeyError(f"Unknown feature set {name!r}. Known: {sorted(REGISTRY)}")
    return REGISTRY[name]
