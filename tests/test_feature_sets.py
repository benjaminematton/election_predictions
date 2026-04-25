"""Pin the feature-set registry: ladder is monotonic, parents are real, columns chain correctly."""

from __future__ import annotations

import pytest

from oath_score.feature_sets import CURVE_ORDER, REGISTRY, get


# Locks the cumulative column count at each rung. If you intentionally add or
# remove a feature, update this list together with the registry.
EXPECTED_LADDER: list[int] = [2, 3, 11, 13, 14, 16]


def test_curve_order_matches_registry() -> None:
    assert set(CURVE_ORDER) == set(REGISTRY.keys())


def test_curve_order_no_duplicates() -> None:
    assert len(CURVE_ORDER) == len(set(CURVE_ORDER))


def test_ladder_column_counts() -> None:
    counts = [len(get(name).columns) for name in CURVE_ORDER]
    assert counts == EXPECTED_LADDER


def test_ladder_is_monotonic_non_decreasing() -> None:
    counts = [len(get(name).columns) for name in CURVE_ORDER]
    assert all(b >= a for a, b in zip(counts, counts[1:])), counts


def test_each_set_is_superset_of_parent() -> None:
    for name in CURVE_ORDER:
        fs = get(name)
        if fs.parent is None:
            continue
        parent_cols = set(get(fs.parent).columns)
        cols = set(fs.columns)
        assert parent_cols.issubset(cols), (
            f"{name} columns missing parent ({fs.parent}) columns: "
            f"{parent_cols - cols}"
        )


def test_parents_exist() -> None:
    for fs in REGISTRY.values():
        if fs.parent is not None:
            assert fs.parent in REGISTRY, f"{fs.name} parent {fs.parent!r} not in registry"


def test_exactly_one_root() -> None:
    roots = [fs.name for fs in REGISTRY.values() if fs.parent is None]
    assert roots == ["naive"]


def test_new_columns_are_unique_within_set() -> None:
    for fs in REGISTRY.values():
        assert len(fs.new_columns) == len(set(fs.new_columns)), fs.name


def test_no_column_introduced_twice_in_chain() -> None:
    # Walking the chain shouldn't re-add a column already inherited.
    for name in CURVE_ORDER:
        fs = get(name)
        if fs.parent is None:
            continue
        inherited = set(get(fs.parent).columns)
        new = set(fs.new_columns)
        assert inherited.isdisjoint(new), (
            f"{name} re-introduces inherited columns: {inherited & new}"
        )


def test_get_unknown_raises() -> None:
    with pytest.raises(KeyError):
        get("does-not-exist")
