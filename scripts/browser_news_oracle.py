import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from openclaw_moe_orchestrator.logging_utils import configure_logging
from openclaw_moe_orchestrator.news import get_live_news


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("asset", nargs="?", default="BTC")
    return parser


if __name__ == "__main__":
    configure_logging()
    args = build_parser().parse_args()
    print(get_live_news(args.asset))
