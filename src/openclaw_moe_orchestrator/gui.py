from __future__ import annotations

import json
import logging
import threading
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from functools import partial
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from .environment import collect_environment_report
from .exceptions import ConfigurationError
from .llm import ModelRole, OllamaClient, load_manifest
from .openclaw_local import collect_openclaw_local_status, install_openclaw_local_bundle
from .paths import RepoPaths
from .pipelines import run_integrated_orchestrator, run_multi_asset_report, run_single_asset_mission

LOGGER = logging.getLogger(__name__)


@dataclass
class GuiJob:
    job_id: str
    workflow: str
    status: str = "queued"
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    started_at: str | None = None
    finished_at: str | None = None
    error: str | None = None
    result: dict[str, Any] | None = None


class GuiJobStore:
    def __init__(self) -> None:
        self._jobs: dict[str, GuiJob] = {}
        self._lock = threading.Lock()

    def create(self, workflow: str) -> GuiJob:
        job_id = f"{workflow}-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{threading.get_ident()}"
        job = GuiJob(job_id=job_id, workflow=workflow)
        with self._lock:
            self._jobs[job_id] = job
        return job

    def update(self, job_id: str, **fields: Any) -> GuiJob:
        with self._lock:
            job = self._jobs[job_id]
            for key, value in fields.items():
                setattr(job, key, value)
            return job

    def get(self, job_id: str) -> GuiJob | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self) -> list[GuiJob]:
        with self._lock:
            return sorted(self._jobs.values(), key=lambda item: item.created_at, reverse=True)


def build_recent_runs(paths: RepoPaths, limit: int = 12) -> list[dict[str, Any]]:
    runs_dir = paths.artifacts_dir / "runs"
    if not runs_dir.exists():
        return []
    runs: list[dict[str, Any]] = []
    for run_dir in sorted(runs_dir.iterdir(), key=lambda item: item.stat().st_mtime, reverse=True):
        if not run_dir.is_dir():
            continue
        outputs_dir = run_dir / "outputs"
        metadata_files = sorted(outputs_dir.glob("*metadata*.json"))
        metadata: dict[str, Any] = {}
        if metadata_files:
            try:
                metadata = json.loads(metadata_files[0].read_text())
            except json.JSONDecodeError:
                metadata = {}
        outputs = []
        if outputs_dir.exists():
            for file_path in sorted(outputs_dir.iterdir()):
                if file_path.is_file():
                    outputs.append(
                        {
                            "name": file_path.name,
                            "path": str(file_path.relative_to(paths.repo_root)),
                            "size_bytes": file_path.stat().st_size,
                        }
                    )
        runs.append(
            {
                "run_id": run_dir.name,
                "workflow": metadata.get("workflow"),
                "created_at": datetime.fromtimestamp(run_dir.stat().st_mtime, UTC).isoformat(),
                "bundle_dir": str(run_dir.relative_to(paths.repo_root)),
                "outputs": outputs,
                "metadata": metadata,
            }
        )
        if len(runs) >= limit:
            break
    return runs


def active_openclaw_profile() -> dict[str, Any]:
    config_path = Path.home() / ".openclaw" / "openclaw.json"
    if not config_path.exists():
        return {"exists": False}
    payload = json.loads(config_path.read_text())
    model = payload.get("agents", {}).get("defaults", {}).get("model", {})
    provider = payload.get("models", {}).get("providers", {}).get("ollama", {})
    return {
        "exists": True,
        "config_path": str(config_path),
        "primary": model.get("primary"),
        "fallbacks": model.get("fallbacks", []),
        "base_url": provider.get("baseUrl"),
        "api": provider.get("api"),
        "provider_models": [
            spec.get("id")
            for spec in provider.get("models", [])
            if isinstance(spec, dict) and spec.get("id")
        ],
    }


def build_model_catalog(paths: RepoPaths) -> dict[str, Any]:
    manifest = load_manifest(paths.config_dir / "ollama_model_manifest.json")
    try:
        live_models = set(OllamaClient(base_url=manifest.base_url).list_models())
        live_error = None
    except Exception as error:  # pragma: no cover - depends on local Ollama daemon
        live_models = set()
        live_error = str(error)

    active_profile = active_openclaw_profile()
    active_models = {
        str(active_profile.get("primary") or "").removeprefix("ollama/"),
        *[
            str(item).removeprefix("ollama/")
            for item in active_profile.get("fallbacks", [])
            if str(item).strip()
        ],
    }
    roles: dict[str, list[dict[str, Any]]] = {}
    defaults: dict[str, list[str]] = {}
    for role in ModelRole:
        ordered_models = sorted(manifest.models_for_role(role), key=lambda item: item.priority, reverse=True)
        defaults[role.value] = [item.model for item in ordered_models]
        roles[role.value] = [
            {
                "model": item.model,
                "priority": item.priority,
                "warm": item.warm,
                "max_concurrency": item.max_concurrency,
                "min_context_window": item.min_context_window,
                "capabilities": list(item.capabilities),
                "available": item.model in live_models,
                "active": item.model in active_models,
            }
            for item in ordered_models
        ]
    return {
        "base_url": manifest.base_url,
        "timeout_seconds": manifest.timeout_seconds,
        "cooldown_seconds": manifest.cooldown_seconds,
        "live_models": sorted(live_models),
        "live_error": live_error,
        "defaults": defaults,
        "roles": roles,
    }


def gui_snapshot(paths: RepoPaths, jobs: GuiJobStore) -> dict[str, Any]:
    return {
        "environment": collect_environment_report(paths),
        "openclaw": collect_openclaw_local_status(paths),
        "active_profile": active_openclaw_profile(),
        "model_catalog": build_model_catalog(paths),
        "jobs": [asdict(job) for job in jobs.list()[:10]],
        "recent_runs": build_recent_runs(paths),
    }


def _role_model_overrides_from_payload(payload: dict[str, Any]) -> dict[ModelRole, tuple[str, ...]] | None:
    role_model_overrides: dict[ModelRole, tuple[str, ...]] = {}
    for role in ModelRole:
        raw_models = payload.get(f"{role.value}_models")
        if raw_models is None:
            continue
        if not isinstance(raw_models, list):
            raise ConfigurationError(f"{role.value}_models must be a list")
        cleaned_models = tuple(str(item).strip() for item in raw_models if str(item).strip())
        if len(cleaned_models) != len(set(cleaned_models)):
            raise ConfigurationError(f"{role.value}_models contains duplicates")
        if cleaned_models:
            role_model_overrides[role] = cleaned_models
    return role_model_overrides or None


def _run_workflow(paths: RepoPaths, workflow: str, payload: dict[str, Any]) -> dict[str, Any]:
    if workflow == "run-mission":
        result = run_single_asset_mission(paths)
        return {"outputs": {key: str(value) for key, value in result.items()}}
    if workflow == "run-multi-report":
        result = run_multi_asset_report(paths)
        return {"outputs": {key: str(value) for key, value in result.items()}}
    if workflow == "run-integrated":
        result = run_integrated_orchestrator(paths)
        return {"outputs": result}
    if workflow == "install-openclaw-cloud":
        result = install_openclaw_local_bundle(
            paths,
            role_model_overrides=_role_model_overrides_from_payload(payload),
        )
        return {"outputs": result}
    raise ConfigurationError(f"Unsupported GUI workflow: {workflow}")


def _start_job(paths: RepoPaths, jobs: GuiJobStore, workflow: str, payload: dict[str, Any]) -> GuiJob:
    job = jobs.create(workflow)

    def runner() -> None:
        jobs.update(job.job_id, status="running", started_at=datetime.now(UTC).isoformat())
        try:
            result = _run_workflow(paths, workflow, payload)
        except Exception as error:  # pragma: no cover - guarded via API
            LOGGER.exception("GUI workflow failed: %s", workflow)
            jobs.update(
                job.job_id,
                status="failed",
                finished_at=datetime.now(UTC).isoformat(),
                error=str(error),
            )
            return
        jobs.update(
            job.job_id,
            status="completed",
            finished_at=datetime.now(UTC).isoformat(),
            result=result,
        )

    thread = threading.Thread(target=runner, name=f"gui-job-{workflow}", daemon=True)
    thread.start()
    return job


class GuiRequestHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args: Any, paths: RepoPaths, jobs: GuiJobStore, **kwargs: Any) -> None:
        self.paths = paths
        self.jobs = jobs
        self.ui_root = resources.files("openclaw_moe_orchestrator").joinpath("ui")
        super().__init__(*args, directory=str(self.ui_root), **kwargs)

    def log_message(self, format: str, *args: Any) -> None:
        LOGGER.info("gui " + format, *args)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/api/health":
            return self._write_json({"ok": True, "timestamp": datetime.now(UTC).isoformat()})
        if parsed.path == "/api/dashboard":
            return self._write_json(gui_snapshot(self.paths, self.jobs))
        if parsed.path == "/api/jobs":
            return self._write_json({"jobs": [asdict(job) for job in self.jobs.list()]})
        if parsed.path.startswith("/api/jobs/"):
            job_id = parsed.path.rsplit("/", 1)[-1]
            job = self.jobs.get(job_id)
            if job is None:
                return self._write_json({"error": "job not found"}, status=HTTPStatus.NOT_FOUND)
            return self._write_json(asdict(job))
        if parsed.path == "/api/openclaw/config":
            return self._write_json(active_openclaw_profile())
        if parsed.path == "/api/models/catalog":
            return self._write_json(build_model_catalog(self.paths))
        if parsed.path == "/api/runs":
            query = parse_qs(parsed.query)
            limit = int(query.get("limit", ["12"])[0])
            return self._write_json({"runs": build_recent_runs(self.paths, limit=limit)})
        if parsed.path.startswith("/repo/"):
            relative_path = parsed.path.removeprefix("/repo/")
            return self._serve_repo_file(relative_path)
        return super().do_GET()

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path != "/api/jobs":
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")
            return
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(content_length) or b"{}")
        except json.JSONDecodeError:
            return self._write_json({"error": "invalid json payload"}, status=HTTPStatus.BAD_REQUEST)
        workflow = str(payload.get("workflow") or "").strip()
        if workflow not in {"run-mission", "run-multi-report", "run-integrated", "install-openclaw-cloud"}:
            return self._write_json({"error": "unsupported workflow"}, status=HTTPStatus.BAD_REQUEST)
        try:
            job = _start_job(self.paths, self.jobs, workflow, payload)
        except ConfigurationError as error:
            return self._write_json({"error": str(error)}, status=HTTPStatus.BAD_REQUEST)
        return self._write_json(asdict(job), status=HTTPStatus.ACCEPTED)

    def _serve_repo_file(self, relative_path: str) -> None:
        candidate = (self.paths.repo_root / relative_path).resolve()
        try:
            candidate.relative_to(self.paths.repo_root)
        except ValueError:
            self.send_error(HTTPStatus.FORBIDDEN, "Forbidden")
            return
        if not candidate.exists() or not candidate.is_file():
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")
            return
        self.path = "/" + candidate.name
        self.directory = str(candidate.parent)
        return super().do_GET()

    def _write_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=True).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def serve_gui(paths: RepoPaths, host: str = "127.0.0.1", port: int = 8765) -> None:
    jobs = GuiJobStore()
    handler = partial(GuiRequestHandler, paths=paths, jobs=jobs)
    server = ThreadingHTTPServer((host, port), handler)
    LOGGER.info("GUI listening on http://%s:%s", host, port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        LOGGER.info("GUI shutdown requested")
    finally:
        server.server_close()
