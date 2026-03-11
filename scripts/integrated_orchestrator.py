import json
import os
import sys
import warnings
from pathlib import Path

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
warnings.filterwarnings("ignore", message="Backend compiler failed with a fake tensor exception.*")
warnings.filterwarnings("ignore", message="Adding a graph break\\.")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from openclaw_moe_orchestrator.logging_utils import configure_logging
from openclaw_moe_orchestrator.paths import RepoPaths
from openclaw_moe_orchestrator.pipelines import run_integrated_orchestrator


if __name__ == "__main__":
    configure_logging()
    print(json.dumps(run_integrated_orchestrator(RepoPaths.discover()), ensure_ascii=True))
