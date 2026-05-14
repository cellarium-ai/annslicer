"""
annslicer — out-of-core sharding of large .h5ad AnnData files.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("annslicer")
except PackageNotFoundError:
    # Package is not installed (e.g. running from source without install)
    __version__ = "unknown"

from annslicer.filter import filter_h5ad
from annslicer.merge import merge_out_of_core
from annslicer.slice import shard_by_obs_column, shard_h5ad

__all__ = ["shard_h5ad", "shard_by_obs_column", "filter_h5ad", "merge_out_of_core"]
