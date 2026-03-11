from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path


def git_revision(repo_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return result.stdout.strip()


def write_run_metadata(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        **payload,
    }
    path.write_text(json.dumps(data, ensure_ascii=True, indent=2))
    return path
