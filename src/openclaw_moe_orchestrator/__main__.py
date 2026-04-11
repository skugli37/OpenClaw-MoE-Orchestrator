import os
import warnings

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
warnings.filterwarnings("ignore", message="Backend compiler failed with a fake tensor exception.*")
warnings.filterwarnings("ignore", message="Adding a graph break\\.")

from .cli import main


if __name__ == "__main__":
    raise SystemExit(main())
