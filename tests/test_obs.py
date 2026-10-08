"""
Tests for annslicer._obs: the category limit, string columns, and what shards carry over.
"""

from __future__ import annotations

import logging

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from annslicer import _obs
from annslicer._obs import normalize_obs
from annslicer.slice import shard_h5ad


@pytest.fixture()
def small_limit(monkeypatch):
    monkeypatch.setattr(_obs, "MAX_CATEGORIES", 4)


def _frame(**columns) -> pd.DataFrame:
    n = len(next(iter(columns.values())))
    return pd.DataFrame(columns, index=[f"c{i}" for i in range(n)])


# ---------------------------------------------------------------------------
# normalize_obs: columns over the category limit
# ---------------------------------------------------------------------------


def test_over_limit_integer_strings_become_int32(small_limit):
    obs = _frame(x=pd.Categorical([str(i) for i in range(10)]))
    out = normalize_obs(obs)
    assert out["x"].dtype == np.int32
    assert out["x"].tolist() == list(range(10))


def test_over_limit_float_strings_keep_precision(small_limit):
    values = [f"0.70218101162916{i}" for i in range(10)]
    out = normalize_obs(_frame(x=pd.Categorical(values)))
    assert out["x"].dtype == np.float64
    assert out["x"].tolist() == [float(v) for v in values]


def test_over_limit_floats_use_float32_only_when_exact(small_limit):
    exact = [f"{i}.5" for i in range(10)]
    assert normalize_obs(_frame(x=pd.Categorical(exact)))["x"].dtype == np.float32


def test_over_limit_large_integers_stay_64_bit(small_limit):
    values = [str(3_000_000_000 + i) for i in range(10)]
    out = normalize_obs(_frame(x=pd.Categorical(values)))
    assert out["x"].dtype == np.int64


def test_over_limit_numeric_categories_and_missing_values(small_limit):
    cat = pd.Categorical([float(i) for i in range(9)] + [None])
    out = normalize_obs(_frame(x=cat))["x"]
    assert out.dtype == np.float32
    assert np.isnan(out.iloc[-1]) and out.iloc[:-1].tolist() == [float(i) for i in range(9)]


def test_over_limit_non_numeric_becomes_plain_strings(small_limit):
    cat = pd.Categorical([f"id_{i}" for i in range(9)] + [None])
    out = normalize_obs(_frame(x=cat))["x"]
    assert out.dtype == object and not isinstance(out.dtype, pd.CategoricalDtype)
    assert out.tolist() == [f"id_{i}" for i in range(9)] + [""]


def test_over_limit_logs_the_conversion(small_limit, caplog):
    with caplog.at_level(logging.INFO, logger=_obs.logger.name):
        normalize_obs(_frame(entropy=pd.Categorical([str(i) for i in range(10)])))
    assert "obs column 'entropy' has 10 categories (more than 4)" in caplog.text


# ---------------------------------------------------------------------------
# normalize_obs: columns that must be left alone
# ---------------------------------------------------------------------------


def test_categorical_within_limit_is_untouched_whatever_it_looks_like(small_limit):
    donor = pd.Categorical(["001", "002", "001"], categories=["001", "002", "003", "004"])
    leiden = pd.Categorical([0, 1, 2, 1])
    out = normalize_obs(_frame(donor=donor[[0, 1, 0, 1]], leiden=leiden))
    assert out["donor"].cat.categories.tolist() == ["001", "002", "003", "004"]  # unused kept
    assert out["leiden"].cat.categories.tolist() == [0, 1, 2]


def test_numeric_and_bool_columns_are_untouched():
    obs = _frame(a=[1, 2, 3], b=[0.5, np.nan, 1.5], c=[True, False, True])
    out = normalize_obs(obs)
    assert out.dtypes.to_dict() == obs.dtypes.to_dict()


def test_plain_string_column_stays_a_string_and_missing_becomes_blank():
    out = normalize_obs(_frame(s=np.array(["a", np.nan, "b"], dtype=object)))["s"]
    assert out.tolist() == ["a", "", "b"]


# ---------------------------------------------------------------------------
# End to end: what shards contain
# ---------------------------------------------------------------------------

N = 60


@pytest.fixture(scope="module")
def rich_h5ad(tmp_path_factory) -> str:
    rng = np.random.default_rng(0)
    obs = pd.DataFrame(
        {
            "entropy": pd.Categorical([f"{0.1 * i + 0.0123456789:.15f}" for i in range(N)]),
            "donor": pd.Categorical(
                np.where(np.arange(N) < 30, "001", "002"),
                categories=["001", "002", "003", "unused"],
            ),
            "label": pd.Categorical([f"id_{i}" for i in range(N - 1)] + [None]),
        },
        index=[f"cell_{i}" for i in range(N)],
    )
    adata = ad.AnnData(
        X=sp.random(N, 8, density=0.5, format="csr", dtype=np.float32, random_state=rng),
        obs=obs,
        obsp={"connectivities": sp.random(N, N, density=0.1, format="csr", random_state=rng)},
        varm={"pcs": rng.random((8, 3))},
        varp={"corr": rng.random((8, 8))},
    )
    adata.raw = adata.copy()
    path = str(tmp_path_factory.mktemp("rich") / "rich.h5ad")
    adata.write_h5ad(path, convert_strings_to_categoricals=False)
    return path


@pytest.mark.parametrize("shuffle", [False, True])
def test_shards_apply_the_category_limit_and_keep_real_categoricals(
    rich_h5ad, tmp_path, monkeypatch, shuffle
):
    monkeypatch.setattr(_obs, "MAX_CATEGORIES", 50)
    prefix = str(tmp_path / "s")
    shard_h5ad(rich_h5ad, prefix, shard_size=20, shuffle=shuffle, seed=1, n_jobs=1)
    original = ad.read_h5ad(rich_h5ad).obs
    for j in range(3):
        shard = ad.read_h5ad(f"{prefix}_shard_{j}.h5ad").obs
        # 60 categories > 50: numeric, with the original values
        assert shard["entropy"].dtype == np.float64
        want = original.loc[shard.index, "entropy"].astype(float)
        np.testing.assert_array_equal(shard["entropy"].to_numpy(), want.to_numpy())
        # 4 categories <= 50: categorical in every shard, including the unused category
        assert shard["donor"].dtype == "category"
        assert shard["donor"].cat.categories.tolist() == ["001", "002", "003", "unused"]
        # 59 non-numeric categories > 50: a plain string column, missing as ""
        assert not isinstance(shard["label"].dtype, pd.CategoricalDtype)
        assert shard["label"].tolist() == [
            original.loc[i, "label"] if isinstance(original.loc[i, "label"], str) else ""
            for i in shard.index
        ]


def test_plain_string_columns_are_not_auto_categorized(tmp_path):
    obs = pd.DataFrame({"s": ["a", "b", "a", "b", "c", "c"]}, index=[f"c{i}" for i in range(6)])
    adata = ad.AnnData(X=sp.csr_matrix(np.ones((6, 2), dtype=np.float32)), obs=obs)
    path = str(tmp_path / "plain.h5ad")
    adata.write_h5ad(path, convert_strings_to_categoricals=False)

    shard_h5ad(path, str(tmp_path / "out"), shard_size=3, n_jobs=1)
    for j in range(2):
        s = ad.read_h5ad(tmp_path / f"out_shard_{j}.h5ad").obs["s"]
        assert not isinstance(s.dtype, pd.CategoricalDtype)


def test_no_warning_when_the_dropped_groups_are_absent_or_empty(synthetic_h5ad, tmp_path, caplog):
    with caplog.at_level(logging.WARNING, logger="annslicer"):
        shard_h5ad(synthetic_h5ad, str(tmp_path / "s"), shard_size=75, n_jobs=1)
    assert "does not carry over" not in caplog.text


def test_groups_annslicer_does_not_carry_are_dropped_with_a_warning(rich_h5ad, tmp_path, caplog):
    with caplog.at_level(logging.WARNING, logger="annslicer"):
        shard_h5ad(rich_h5ad, str(tmp_path / "s"), shard_size=30, n_jobs=1)
    assert "obsp, varm, varp, raw" in caplog.text
    shard = ad.read_h5ad(tmp_path / "s_shard_0.h5ad")
    assert len(shard.obsp) == 0 and len(shard.varm) == 0 and len(shard.varp) == 0
    assert shard.raw is None
