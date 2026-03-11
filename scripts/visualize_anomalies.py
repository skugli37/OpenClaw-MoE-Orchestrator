import os
import sys
from pathlib import Path

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from openclaw_moe_orchestrator.logging_utils import configure_logging
from openclaw_moe_orchestrator.paths import RepoPaths
from openclaw_moe_orchestrator.visualization import visualize_single_asset


if __name__ == "__main__":
    configure_logging()
    paths = RepoPaths.discover()
    visualize_single_asset(
        paths.artifacts_dir / "anomaly_results.csv",
        paths.artifacts_dir / "anomaly_chart.png",
    )
