import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from openclaw_moe_orchestrator.logging_utils import configure_logging
from openclaw_moe_orchestrator.paths import RepoPaths
from openclaw_moe_orchestrator.pipelines import run_single_asset_mission


if __name__ == "__main__":
    configure_logging()
    run_single_asset_mission(RepoPaths.discover())
