from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from .environment import collect_environment_report, format_environment_report
from .logging_utils import configure_logging
from .openclaw_local import collect_openclaw_local_status, install_openclaw_local_bundle
from .ollama_sync import sync_models
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
    install_parser = subparsers.add_parser(
        "install-openclaw-cloud",
        aliases=("install-openclaw-local",),
    )
    install_parser.add_argument("--state-dir", default=None)
    install_parser.add_argument("--skill-mode", choices=("symlink", "copy"), default="copy")
    install_parser.add_argument("--manifest-path", default=None)
    doctor_parser = subparsers.add_parser(
        "doctor-openclaw-cloud",
        aliases=("doctor-openclaw-local",),
    )
    doctor_parser.add_argument("--state-dir", default=None)
    subparsers.add_parser("list-ollama-models")
    sync_parser = subparsers.add_parser("sync-ollama-models")
    sync_parser.add_argument("--manifest-path", default=None)
    sync_parser.add_argument(
        "--role",
        action="append",
        choices=("reasoning", "coding", "general", "vision", "embedding", "safety"),
        default=None,
    )
    sync_parser.add_argument("--max-models", type=int, default=None)
    sync_parser.add_argument("--dry-run", action="store_true")
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
    if args.command in {"install-openclaw-cloud", "install-openclaw-local"}:
        result = install_openclaw_local_bundle(
            paths,
            state_dir=Path(args.state_dir) if args.state_dir else None,
            skill_mode=args.skill_mode,
            manifest_path=Path(args.manifest_path) if args.manifest_path else None,
        )
        print(json.dumps(result, ensure_ascii=True, indent=2))
        return 0
    if args.command in {"doctor-openclaw-cloud", "doctor-openclaw-local"}:
        print(
            json.dumps(
                collect_openclaw_local_status(
                    paths,
                    state_dir=Path(args.state_dir) if args.state_dir else None,
                ),
                ensure_ascii=True,
                indent=2,
            )
        )
        return 0
    if args.command == "list-ollama-models":
        from .llm import OllamaClient

        print(json.dumps({"models": OllamaClient().list_models()}, ensure_ascii=True, indent=2))
        return 0
    if args.command == "sync-ollama-models":
        from .llm import ModelRole

        result = sync_models(
            args.manifest_path or (paths.config_dir / "ollama_model_manifest.json"),
            roles=tuple(ModelRole(role) for role in args.role) if args.role else None,
            max_models=args.max_models,
            dry_run=args.dry_run,
        )
        print(json.dumps(result, ensure_ascii=True, indent=2))
        return 0
    parser.error(f"Unknown command {args.command}")
    return 2
