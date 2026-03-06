"""
Unified command-line entry point for annslicer.

Subcommands
-----------
annslicer slice   -- shard a large .h5ad file into smaller files
annslicer merge   -- merge sharded files back into one large file
"""

from __future__ import annotations

import argparse
import logging

from annslicer import merge, slice


def main() -> None:
    """Entry point for the ``annslicer`` command."""
    parser = argparse.ArgumentParser(
        prog="annslicer",
        description="Out-of-core sharding and merging of large .h5ad AnnData files.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable verbose debug-level logging.",
    )

    subparsers = parser.add_subparsers(
        title="subcommands",
        dest="subcommand",
        metavar="<command>",
    )
    subparsers.required = True

    slice.register_subcommand(subparsers)
    merge.register_subcommand(subparsers)

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    args.func(args)
