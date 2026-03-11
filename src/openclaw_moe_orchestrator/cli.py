from __future__ import annotations

import argparse
import json
import logging

from .environment import collect_environment_report, format_environment_report
from .logging_utils import configure_logging
from .paths import RepoPaths
from .pipelines import (
    run_integrated_orchestrator,
    run_multi_asset_detection,
    run_multi_asset_report,
    run_multi_asset_visualization,
    run_single_asset_detection,
    run_single_asset_mission,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="openclaw-moe")
    parser.add_argument("--repo-root", default=None)
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("run-mission")
    subparsers.add_parser("detect-single")
    subparsers.add_parser("detect-multi")
    subparsers.add_parser("run-multi-report")
    subparsers.add_parser("visualize-multi")
    subparsers.add_parser("run-integrated")
    subparsers.add_parser("doctor")
    return parser


def main() -> int:
    configure_logging()
    parser = build_parser()
    args = parser.parse_args()
    paths = RepoPaths.discover(args.repo_root)
    paths.ensure_directories()

    if args.command == "run-mission":
        outputs = run_single_asset_mission(paths)
        logging.getLogger(__name__).info("mission_outputs=%s", outputs)
        return 0
    if args.command == "detect-single":
        run_single_asset_detection(paths)
        return 0
    if args.command == "detect-multi":
        run_multi_asset_detection(paths)
        return 0
    if args.command == "run-multi-report":
        outputs = run_multi_asset_report(paths)
        logging.getLogger(__name__).info("multi_asset_outputs=%s", outputs)
        return 0
    if args.command == "visualize-multi":
        output = run_multi_asset_visualization(paths)
        logging.getLogger(__name__).info("multi_asset_chart=%s", output)
        return 0
    if args.command == "run-integrated":
        result = run_integrated_orchestrator(paths)
        print(json.dumps(result, ensure_ascii=True))
        return 0
    if args.command == "doctor":
        print(format_environment_report(collect_environment_report(paths)))
        return 0
    parser.error(f"Unknown command {args.command}")
    return 2
