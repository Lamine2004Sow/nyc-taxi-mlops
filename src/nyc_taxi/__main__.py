"""Command line entry point: python -m nyc_taxi <command>."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from nyc_taxi.config import Settings, get_settings
from nyc_taxi.data import download_month
from nyc_taxi.zones import build_centroids


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="nyc_taxi", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("download", help="download raw TLC parquet months")
    p.add_argument("--months", nargs="+", required=True)

    sub.add_parser("zones", help="build zone centroids CSV from the TLC shapefile")

    p = sub.add_parser("train", help="train and register the champion")
    p.add_argument("--month", default=None)

    p = sub.add_parser("replay", help="replay production months with monitoring and retraining")
    p.add_argument("--api-url", default=None)
    p.add_argument("--monitor-url", default=None)

    p = sub.add_parser("retrain", help="retrain a challenger on a month and gate promotion")
    p.add_argument("--month", required=True)

    p = sub.add_parser("evaluate-ci", help="train on the fixture and gate against the baseline")
    p.add_argument("--fixture", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run a CLI command; returns the process exit code."""
    args = _build_parser().parse_args(argv)
    settings = get_settings()

    if args.command == "download":
        for month in args.months:
            print(download_month(month, settings.data_dir))
        return 0
    if args.command == "zones":
        print(build_centroids(settings.data_dir))
        return 0

    # Heavy imports (ML stack) only when a model command is requested.
    from nyc_taxi import pipeline

    if args.command == "train":
        if args.month:
            settings = _with(settings, train_month=args.month)
        print(json.dumps(pipeline.train_champion(settings), indent=2))
        return 0
    if args.command == "replay":
        history = pipeline.run_replay(settings, api_url=args.api_url, monitor_url=args.monitor_url)
        print(f"replayed {len(history)} months; reports in {Path(settings.reports_dir).resolve()}")
        return 0
    if args.command == "retrain":
        print(json.dumps(pipeline.retrain(args.month, settings), indent=2, default=str))
        return 0
    if args.command == "evaluate-ci":
        return pipeline.evaluate_ci(args.fixture, settings)
    return 2


def _with(settings: Settings, **changes: str) -> Settings:
    return replace(settings, **changes)


if __name__ == "__main__":
    sys.exit(main())
