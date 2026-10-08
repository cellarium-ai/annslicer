"""
Tests for how annslicer merge combines the obs tables of shards that need not agree.

Categorical columns must come out categorical, with the union of the shards' categories, even
when the shards were not harmonised beforehand (so not just for shards made by ``annslicer slice``).
"""

from __future__ import annotations

import logging

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from annslicer import _obs
from annslicer._obs import merge_obs
from annslicer.merge import merge_out_of_core
from annslicer.slice import shard_h5ad


def _frame(start: int, **columns) -> pd.DataFrame:
    n = len(next(iter(columns.values())))
    return pd.DataFrame(columns, index=[f"c{start + i}" for i in range(n)])


def _cat(values, categories=None, ordered=False) -> pd.Categorical:
    return pd.Categorical(values, categories=categories, ordered=ordered)


# ---------------------------------------------------------------------------
# merge_obs on in-memory tables
# ---------------------------------------------------------------------------


def test_union_of_different_category_sets_in_first_appearance_order():
    a = _frame(0, ct=_cat(["b", "a", "b"]))
    b = _frame(3, ct=_cat(["c", "a"]))
    out = merge_obs([a, b])["ct"]
    assert out.dtype == "category"
    assert out.cat.categories.tolist() == ["a", "b", "c"]  # a's own order is a, b (sorted)
    assert out.tolist() == ["b", "a", "b", "c", "a"]


def test_categories_unused_in_every_shard_are_kept():
    a = _frame(0, ct=_cat(["x", "x"], categories=["x", "unused"]))
    b = _frame(2, ct=_cat(["y"], categories=["y", "also_unused"]))
    out = merge_obs([a, b])["ct"]
    assert out.cat.categories.tolist() == ["x", "unused", "y", "also_unused"]


def test_same_categories_in_a_different_order_keep_the_first_shards_order():
    a = _frame(0, ct=_cat(["p", "q"], categories=["q", "p"]))
    b = _frame(2, ct=_cat(["q", "p"], categories=["p", "q"]))
    out = merge_obs([a, b])["ct"]
    assert out.cat.categories.tolist() == ["q", "p"]
    assert out.tolist() == ["p", "q", "q", "p"]


def test_missing_values_stay_missing_and_a_column_absent_from_a_shard_is_missing_there():
    a = _frame(0, ct=_cat(["a", None]))
    b = _frame(2, other=[1, 2])
    c = _frame(4, ct=_cat(["b"]))
    out = merge_obs([a, b, c])
    assert out["ct"].dtype == "category"
    assert out["ct"].tolist()[0] == "a" and out["ct"].isna().tolist() == [
        False,
        True,
        True,
        True,
        False,
    ]
    assert out["ct"].iloc[4] == "b"
    assert out["other"].isna().tolist() == [True, True, False, False, True]


def test_categorical_in_one_shard_and_plain_strings_in_another_is_categorical():
    a = _frame(0, ct=_cat(["a", "b"]))
    b = _frame(2, ct=np.array(["c", "a"], dtype=object))
    out = merge_obs([a, b])["ct"]
    assert out.dtype == "category"
    assert out.cat.categories.tolist() == ["a", "b", "c"]
    assert out.tolist() == ["a", "b", "c", "a"]


def test_integer_and_float_categories_are_united_as_numbers():
    a = _frame(0, k=_cat([1, 2]))
    b = _frame(2, k=_cat([2.5, 1.0]))
    out = merge_obs([a, b])["k"]
    assert out.dtype == "category"
    assert sorted(out.cat.categories.tolist()) == [1.0, 2.0, 2.5]
    assert out.tolist() == [1, 2, 2.5, 1.0]


def test_mixed_category_types_are_cast_to_strings_and_logged(caplog):
    a = _frame(0, k=_cat([1, 2]))
    b = _frame(2, k=_cat(["x", "1"]))
    with caplog.at_level(logging.INFO, logger=_obs.logger.name):
        out = merge_obs([a, b])["k"]
    assert out.dtype == "category"
    assert out.cat.categories.tolist() == ["1", "2", "x"]
    assert out.tolist() == ["1", "2", "x", "1"]
    assert "obs column 'k' has mixed value types" in caplog.text


def test_ordered_categoricals_are_kept_only_when_every_shard_agrees():
    same = merge_obs(
        [
            _frame(0, s=_cat(["lo", "hi"], categories=["lo", "hi"], ordered=True)),
            _frame(2, s=_cat(["hi"], categories=["lo", "hi"], ordered=True)),
        ]
    )["s"]
    assert same.cat.ordered and same.cat.categories.tolist() == ["lo", "hi"]

    differ = merge_obs(
        [
            _frame(0, s=_cat(["lo"], categories=["lo", "hi"], ordered=True)),
            _frame(1, s=_cat(["mid"], categories=["lo", "mid"], ordered=True)),
        ]
    )["s"]
    assert not differ.cat.ordered
    assert differ.cat.categories.tolist() == ["lo", "hi", "mid"]


def test_union_over_the_category_limit_becomes_numeric_or_string(monkeypatch):
    monkeypatch.setattr(_obs, "MAX_CATEGORIES", 5)
    a = _frame(0, n=_cat(["1", "2", "3"]), s=_cat(["a", "b", "c"]))
    b = _frame(3, n=_cat(["4", "5", "6"]), s=_cat(["d", "e", "f"]))
    out = merge_obs([a, b])
    assert out["n"].tolist() == [1, 2, 3, 4, 5, 6] and out["n"].dtype == np.int32
    assert out["s"].tolist() == list("abcdef") and not isinstance(
        out["s"].dtype, pd.CategoricalDtype
    )


def test_union_within_the_limit_stays_categorical(monkeypatch):
    monkeypatch.setattr(_obs, "MAX_CATEGORIES", 6)
    a = _frame(0, n=_cat(["1", "2", "3"]))
    b = _frame(3, n=_cat(["4", "5", "6"]))
    out = merge_obs([a, b])["n"]
    assert out.dtype == "category" and out.cat.categories.tolist() == list("123456")


def test_plain_columns_are_not_categorized_and_mixed_ones_become_strings():
    a = _frame(0, s=np.array(["a", "b"], dtype=object), v=[1, 2], m=[1, 2])
    b = _frame(
        2,
        s=np.array(["c", np.nan], dtype=object),
        v=[3.5, 4.5],
        m=np.array(["x", "y"], dtype=object),
    )
    out = merge_obs([a, b])
    assert out["s"].tolist() == ["a", "b", "c", ""]
    assert out["v"].dtype == np.float64 and out["v"].tolist() == [1, 2, 3.5, 4.5]
    assert out["m"].tolist() == ["1", "2", "x", "y"]


# ---------------------------------------------------------------------------
# Through merge_out_of_core, with shards written directly (not by annslicer slice)
# ---------------------------------------------------------------------------


def _write_shard(path: str, obs: pd.DataFrame) -> str:
    n = len(obs)
    adata = ad.AnnData(
        X=sp.random(n, 4, density=0.5, format="csr", dtype=np.float32, random_state=0),
        obs=obs,
        var=pd.DataFrame(index=[f"g{j}" for j in range(4)]),
    )
    adata.write_h5ad(path, convert_strings_to_categoricals=False)
    return path


@pytest.fixture()
def unharmonised_shards(tmp_path) -> list[str]:
    a = _frame(0, ct=_cat(["T", "B", "T"]), donor=_cat(["001", "002", "001"]), tag=["u", "v", "u"])
    b = _frame(3, ct=_cat(["NK", "T"]), donor=_cat(["003", "001"]), tag=["w", "u"])
    return [_write_shard(str(tmp_path / f"s{i}.h5ad"), df) for i, df in enumerate([a, b])]


@pytest.mark.parametrize("suffix", ["h5ad", "zarr"])
def test_merge_out_of_core_unions_categories_of_unharmonised_shards(
    unharmonised_shards, tmp_path, suffix
):
    if suffix == "zarr":
        pytest.importorskip("zarr")
    out_path = str(tmp_path / f"merged.{suffix}")
    merge_out_of_core(unharmonised_shards, out_path)
    merged = (ad.read_zarr if suffix == "zarr" else ad.read_h5ad)(out_path).obs
    assert merged["ct"].dtype == "category"
    assert merged["ct"].cat.categories.tolist() == ["B", "T", "NK"]
    assert merged["ct"].tolist() == ["T", "B", "T", "NK", "T"]
    assert merged["donor"].dtype == "category"
    assert merged["donor"].cat.categories.tolist() == ["001", "002", "003"]
    assert not isinstance(merged["tag"].dtype, pd.CategoricalDtype)  # plain strings stay plain


def test_slice_then_merge_round_trip_keeps_categoricals_with_all_categories(tmp_path):
    n = 90
    obs = pd.DataFrame(
        {
            "ct": _cat(np.tile(["T", "B"], n // 2), categories=["T", "B", "never_used"]),
            "donor": _cat(np.where(np.arange(n) < 45, "001", "002")),
        },
        index=[f"cell_{i}" for i in range(n)],
    )
    adata = ad.AnnData(
        X=sp.random(n, 6, density=0.4, format="csr", dtype=np.float32, random_state=3), obs=obs
    )
    src = str(tmp_path / "in.h5ad")
    adata.write_h5ad(src)

    shard_h5ad(src, str(tmp_path / "s"), shard_size=30, shuffle=True, seed=2, n_jobs=1)
    shards = sorted(str(p) for p in tmp_path.glob("s_shard_*.h5ad"))
    for shard in shards:  # every shard carries the full, identical category lists
        o = ad.read_h5ad(shard).obs
        assert o["ct"].cat.categories.tolist() == ["T", "B", "never_used"]
        assert o["donor"].cat.categories.tolist() == ["001", "002"]

    out_path = str(tmp_path / "merged.h5ad")
    merge_out_of_core(shards, out_path)
    merged = ad.read_h5ad(out_path).obs
    assert merged["ct"].cat.categories.tolist() == ["T", "B", "never_used"]
    assert merged["donor"].cat.categories.tolist() == ["001", "002"]
    assert merged.loc[obs.index, "ct"].tolist() == obs["ct"].tolist()
    assert merged.loc[obs.index, "donor"].tolist() == obs["donor"].tolist()
