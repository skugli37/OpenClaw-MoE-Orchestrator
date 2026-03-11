from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RepoPaths:
    repo_root: Path
    config_dir: Path
    docs_dir: Path
    data_dir: Path
    raw_data_dir: Path
    processed_data_dir: Path
    artifacts_dir: Path
    logs_dir: Path

    @classmethod
    def discover(cls, repo_root: Path | None = None) -> "RepoPaths":
        root = (repo_root or Path(__file__).resolve().parents[2]).resolve()
        data_dir = root / "data"
        return cls(
            repo_root=root,
            config_dir=root / "configs",
            docs_dir=root / "docs",
            data_dir=data_dir,
            raw_data_dir=data_dir / "raw",
            processed_data_dir=data_dir / "processed",
            artifacts_dir=root / "artifacts",
            logs_dir=root / "artifacts" / "logs",
        )

    def ensure_directories(self) -> None:
        for directory in (
            self.docs_dir,
            self.raw_data_dir,
            self.processed_data_dir,
            self.artifacts_dir,
            self.logs_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)
