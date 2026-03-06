"""
annslicer — out-of-core sharding of large .h5ad AnnData files.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("annslicer")
except PackageNotFoundError:
    # Package is not installed (e.g. running from source without install)
    __version__ = "unknown"

from annslicer.merge import merge_out_of_core
from annslicer.slice import shard_h5ad

__all__ = ["shard_h5ad", "merge_out_of_core"]
