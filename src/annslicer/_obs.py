"""
Handling of ``obs`` columns: which columns count as categorical, and how shards are merged.

A column stays categorical, with *all* of its categories, in every output shard only if it has
at most ``MAX_CATEGORIES`` of them.  Anything above that is not treated as categorical: it
becomes a numeric column if all its categories are numbers, and a plain string column
otherwise.  (Writing one copy of a multi-million-entry category list into every shard would
dwarf the data itself.)  Values are never inspected below the limit, so a categorical that
merely looks numeric (cluster ``0..30``, donor ``"001"``) stays categorical.

Plain string columns are written as strings, never auto-categorized: anndata would give each
shard its own category list, and shards must agree.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

MAX_CATEGORIES = 50_000


def _downcast(values: np.ndarray) -> np.ndarray:
    """Narrow integers to int32 when they fit and floats to float32 when that loses nothing."""
    if values.dtype.kind in "iu":
        info = np.iinfo(np.int32)
        fits = values.size == 0 or (values.min() >= info.min and values.max() <= info.max)
        return values.astype(np.int32) if fits else values
    if values.dtype.kind == "f":
        with np.errstate(over="ignore"):
            narrow = values.astype(np.float32)
        if np.array_equal(narrow.astype(values.dtype), values, equal_nan=True):
            return narrow
    return values


def _to_string(name: str, col: pd.Series) -> pd.Series:
    """Cast a column to plain strings; missing values become ``""`` (anndata can't write NaN)."""
    missing = col.isna().to_numpy()
    values = np.array(col.astype(str), dtype=object)
    values[missing] = ""
    if missing.any():
        logger.info(
            "obs column %r: %d missing values written as empty strings.", name, missing.sum()
        )
    return pd.Series(values, index=col.index, name=col.name)


def _demote(name: str, col: pd.Series) -> pd.Series:
    """Replace a categorical column that has too many categories by a numeric or string one."""
    cats = col.cat.categories
    codes = col.cat.codes.to_numpy()
    missing = codes < 0
    try:
        numbers = (
            cats.to_numpy()
            if cats.dtype.kind in "iuf"
            else pd.to_numeric(cats.astype(str)).to_numpy()
        )
    except (ValueError, TypeError):
        numbers = None

    if numbers is not None:
        values = numbers[codes]
        if missing.any():
            values = values.astype(np.float64)
            values[missing] = np.nan
        values = _downcast(values)
        kind = f"numeric ({values.dtype})"
        result = pd.Series(values, index=col.index, name=col.name)
    else:
        values = np.array(cats.astype(str), dtype=object)[codes]
        values[missing] = ""
        kind = "string"
        result = pd.Series(values, index=col.index, name=col.name)
    logger.info(
        "obs column %r has %d categories (more than %d): treating it as %s, not categorical.",
        name,
        len(cats),
        MAX_CATEGORIES,
        kind,
    )
    return result


def normalize_obs(obs: pd.DataFrame) -> pd.DataFrame:
    """Apply the category limit and make every plain string column writable as strings."""
    obs = obs.copy(deep=False)
    for name in obs.columns:
        col = obs[name]
        if isinstance(col.dtype, pd.CategoricalDtype):
            if len(col.cat.categories) > MAX_CATEGORIES:
                obs[name] = _demote(name, col)
        elif _family(col.dtype) == "string":
            obs[name] = _to_string(name, col)
    return obs


# ---------------------------------------------------------------------------
# Merging the obs tables of several shards
# ---------------------------------------------------------------------------


def _is_numeric(dtype: object) -> bool:
    return pd.api.types.is_numeric_dtype(dtype) and not pd.api.types.is_bool_dtype(dtype)


def _family(dtype: object) -> str:
    if isinstance(dtype, pd.CategoricalDtype):
        return "categorical"
    if _is_numeric(dtype):
        return "numeric"
    if pd.api.types.is_object_dtype(dtype) or isinstance(dtype, pd.StringDtype):
        return "string"
    return str(dtype)


def _union_categorical(
    name: str, parts: list[pd.Series | None], lengths: list[int]
) -> pd.Categorical:
    """
    Combine one column, categorical in at least one shard, into a single categorical.

    The categories are the union over all shards in order of first appearance.  Plain values
    from shards where the column is not categorical become categories too, and shards without
    the column get missing values.  If the shards' category dtypes are not all numbers or all
    the same, everything is cast to strings.
    """
    present = [p for p in parts if p is not None]
    dtypes = [
        p.cat.categories.dtype if _family(p.dtype) == "categorical" else p.dtype for p in present
    ]
    cast = len({str(d) for d in dtypes}) > 1 and not all(_is_numeric(d) for d in dtypes)
    if cast:
        logger.info("obs column %r has mixed value types across shards: casting to strings.", name)

    def values_of(p: pd.Series) -> pd.Series:
        return p.astype(str).where(p.notna()) if cast else p

    part_cats = []
    for p in present:
        if _family(p.dtype) == "categorical":
            cats = p.cat.categories
            part_cats.append(cats.astype(str) if cast else cats)
        else:
            part_cats.append(pd.Index(pd.unique(values_of(p).dropna())))
    cats = pd.Index(np.concatenate([c.to_numpy() for c in part_cats])).unique()

    codes: list[np.ndarray] = []
    for p, n in zip(parts, lengths, strict=True):
        if p is None:
            codes.append(np.full(n, -1, dtype=np.int64))
        elif _family(p.dtype) == "categorical":
            remap = cats.get_indexer(p.cat.categories.astype(str) if cast else p.cat.categories)
            own = p.cat.codes.to_numpy()
            codes.append(np.where(own >= 0, remap[own], -1))
        else:
            codes.append(cats.get_indexer(values_of(p)))

    same = all(
        _family(p.dtype) == "categorical" and p.cat.ordered and p.cat.categories.equals(cats)
        for p in present
    )
    return pd.Categorical.from_codes(np.concatenate(codes), categories=cats, ordered=same)


def merge_obs(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """
    Concatenate the ``obs`` tables of shards that need not agree with each other.

    A column that is categorical in any shard becomes one categorical whose categories are the
    union over all shards; a column with mixed value types is cast to strings.  The category
    limit then applies to the merged result.
    """
    merged = pd.concat(frames, axis=0)
    lengths = [len(f) for f in frames]
    for name in merged.columns:
        parts = [f[name] if name in f.columns else None for f in frames]
        present = [p for p in parts if p is not None]
        if any(_family(p.dtype) == "categorical" for p in present):
            merged[name] = pd.Series(
                _union_categorical(name, parts, lengths), index=merged.index, name=name
            )
        elif len({_family(p.dtype) for p in present}) > 1:
            logger.info(
                "obs column %r has mixed value types across shards: casting to strings.", name
            )
            merged[name] = _to_string(name, merged[name])
    return normalize_obs(merged)
