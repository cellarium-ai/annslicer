"""
annslicer — out-of-core sharding of large .h5ad AnnData files.
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("annslicer")
except PackageNotFoundError:
    # Package is not installed (e.g. running from source without install)
    __version__ = "unknown"

from annslicer.core import shard_h5ad

__all__ = ["shard_h5ad"]
