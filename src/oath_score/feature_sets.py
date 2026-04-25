"""Feature-set registry for the improvement-curve backtest.

Each named feature set selects a subset of columns from the per-candidate
matrix produced by `features.py`. Each row of `notebooks/03_backtest_curves.ipynb`
corresponds to one entry here. Adding a feature is adding a row to the registry,
not changing model code.

Single source of truth: each FeatureSet declares only its NEW columns plus the
name of its parent. The cumulative `columns` tuple is derived by walking the
parent chain. To add a step in the ladder, add one short entry.
"""

from __future__ import annotations

from dataclasses import dataclass


# Column names below are the canonical names produced by features.py:
#   cook_rating          ordinal 1-7 from "Solid R" to "Solid D"
#   incumbent            int (0/1)
#   cpvi                 cook partisan voting index, signed (D positive)
#   acs_*                ~33 demographic columns from census.py
#   self_raised_pct      candidate's snapshot-date receipts / (own + opponent)
#   self_raised_log      log1p of own snapshot-date total receipts
#   opp_raised_log       log1p of opponent's snapshot-date total receipts
#   ie_for_log           log1p of independent expenditures FOR (snapshot)
#   ie_against_log       log1p of independent expenditures AGAINST (snapshot)


@dataclass(frozen=True)
class FeatureSet:
    name: str
    new_columns: tuple[str, ...]
    parent: str | None = None
    description: str = ""

    @property
    def columns(self) -> tuple[str, ...]:
        if self.parent is None:
            return self.new_columns
        return REGISTRY[self.parent].columns + self.new_columns


_ENTRIES: tuple[FeatureSet, ...] = (
    FeatureSet(
        name="naive",
        new_columns=("cook_rating", "incumbent"),
        parent=None,
        description="Logistic baseline. Cook rating + incumbency only — what an analyst would predict in 30 seconds.",
    ),
    FeatureSet(
        name="naive+pvi",
        new_columns=("cpvi",),
        parent="naive",
        description="Adds Cook PVI (district partisan lean).",
    ),
    FeatureSet(
        name="naive+pvi+demo",
        new_columns=(
            "acs_median_age", "acs_race_white", "acs_race_black", "acs_race_asian",
            "acs_edu_bachelors", "acs_edu_grad_degree", "acs_median_income",
            "acs_below_100pct_fpl",
        ),
        parent="naive+pvi",
        description="Adds ACS demographics: race, education, income, poverty.",
    ),
    FeatureSet(
        name="naive+pvi+demo+fund",
        new_columns=("self_raised_log", "self_raised_pct"),
        parent="naive+pvi+demo",
        description="Adds snapshot-date candidate fundraising. SNAPSHOT-CRITICAL.",
    ),
    FeatureSet(
        name="naive+pvi+demo+fund+opp",
        new_columns=("opp_raised_log",),
        parent="naive+pvi+demo+fund",
        description="Adds opponent fundraising at snapshot.",
    ),
    FeatureSet(
        name="full",
        new_columns=("ie_for_log", "ie_against_log"),
        parent="naive+pvi+demo+fund+opp",
        description="Full feature set: adds independent-expenditure totals (FEC Schedule E).",
    ),
)


REGISTRY: dict[str, FeatureSet] = {fs.name: fs for fs in _ENTRIES}

# Order to plot on the improvement curve, left → right.
CURVE_ORDER: tuple[str, ...] = tuple(fs.name for fs in _ENTRIES)


def get(name: str) -> FeatureSet:
    if name not in REGISTRY:
        raise KeyError(f"Unknown feature set {name!r}. Known: {sorted(REGISTRY)}")
    return REGISTRY[name]
