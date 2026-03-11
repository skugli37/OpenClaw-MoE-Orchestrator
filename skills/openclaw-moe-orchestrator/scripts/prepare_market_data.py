import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from openclaw_moe_orchestrator.data_pipeline import prepare_market_data
from openclaw_moe_orchestrator.logging_utils import configure_logging
from openclaw_moe_orchestrator.paths import RepoPaths


if __name__ == "__main__":
    configure_logging()
    prepare_market_data(RepoPaths.discover())
