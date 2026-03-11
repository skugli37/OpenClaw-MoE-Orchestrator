from __future__ import annotations

import types
from contextlib import nullcontext


def _pipeline_import_modules():
    torch_module = types.ModuleType("torch")
    nn_module = types.ModuleType("torch.nn")
    nn_module.Module = object
    nn_module.MSELoss = object
    torch_module.nn = nn_module

    deepspeed_module = types.ModuleType("deepspeed")
    deepspeed_module.initialize = lambda **kwargs: None

    data_pipeline_module = types.ModuleType("openclaw_moe_orchestrator.data_pipeline")
    data_pipeline_module.load_multi_asset_dataset = lambda *args, **kwargs: None
    data_pipeline_module.load_single_asset_dataset = lambda *args, **kwargs: None
    data_pipeline_module.prepare_market_data = lambda *args, **kwargs: None
    data_pipeline_module.prepare_multi_asset_data = lambda *args, **kwargs: None

    models_module = types.ModuleType("openclaw_moe_orchestrator.models")
    models_module.IntelligenceMoE = object
    models_module.MoEAutoencoder = object
    models_module.MultiAssetMoE = object

    news_module = types.ModuleType("openclaw_moe_orchestrator.news")
    news_module.get_live_news = lambda asset_name: f"{asset_name} news"

    reports_module = types.ModuleType("openclaw_moe_orchestrator.reports")
    reports_module.build_single_asset_report = lambda path: "report"
    reports_module.build_multi_asset_report = lambda path: "multi-report"

    runtime_module = types.ModuleType("openclaw_moe_orchestrator.runtime")
    runtime_module.gpu_execution_lock = lambda *args, **kwargs: nullcontext()
    runtime_module.load_runtime_config = lambda *args, **kwargs: {}
    runtime_module.prepare_distributed_env = lambda: None
    runtime_module.prepare_model_and_optimizer = lambda *args, **kwargs: (None, None, None)
    runtime_module.shutdown_distributed = lambda: None
    runtime_module.tensor_dtype = lambda: "float32"

    visualization_module = types.ModuleType("openclaw_moe_orchestrator.visualization")
    visualization_module.visualize_multi_asset = lambda *args, **kwargs: None
    visualization_module.visualize_single_asset = lambda *args, **kwargs: None

    return {
        "torch": torch_module,
        "torch.nn": nn_module,
        "deepspeed": deepspeed_module,
        "openclaw_moe_orchestrator.data_pipeline": data_pipeline_module,
        "openclaw_moe_orchestrator.models": models_module,
        "openclaw_moe_orchestrator.news": news_module,
        "openclaw_moe_orchestrator.reports": reports_module,
        "openclaw_moe_orchestrator.runtime": runtime_module,
        "openclaw_moe_orchestrator.visualization": visualization_module,
    }


def test_build_single_asset_report_summarizes_fixture_results(load_module, fixtures_dir):
    module = load_module(
        "src/openclaw_moe_orchestrator/reports.py",
        module_name="openclaw_moe_orchestrator.reports",
    )

    report = module.build_single_asset_report(fixtures_dir / "anomaly_results_sample.csv")

    assert "OpenClaw Intelligence Report" in report
    assert "Analyzed 5 BTC market observations from 2025-02-01 to 2025-02-05." in report
    assert "The MoE model marked 2 observations as anomalous" in report
    assert "Highest anomaly date: 2025-02-05" in report
    assert "BTC close at that point: $102,400.00" in report
    assert "Maximum reconstruction error: 0.137500" in report


def test_build_multi_asset_report_summarizes_fixture_results(load_module, fixtures_dir):
    module = load_module(
        "src/openclaw_moe_orchestrator/reports.py",
        module_name="openclaw_moe_orchestrator.reports",
    )

    report = module.build_multi_asset_report(fixtures_dir / "multi_asset_anomalies_sample.csv")

    assert "OpenClaw Multi-Asset Correlation Report" in report
    assert "Analyzed 4 multi-asset return observations from 2025-02-01 to 2025-02-04." in report
    assert "The MoE model marked 2 observations as correlation anomalies" in report
    assert "Highest anomaly date: 2025-02-04" in report
    assert "Maximum reconstruction error: 0.137500" in report


def test_run_single_asset_mission_writes_report_and_returns_artifacts(load_module, tmp_path, monkeypatch):
    module = load_module(
        "src/openclaw_moe_orchestrator/pipelines.py",
        module_name="openclaw_moe_orchestrator.pipelines",
        injected_modules=_pipeline_import_modules(),
    )
    paths = module.RepoPaths.discover(tmp_path)
    paths.ensure_directories()

    run_paths = paths.create_run_paths("mission")
    results_path = run_paths.outputs_dir / "anomaly_results.csv"
    chart_path = run_paths.outputs_dir / "anomaly_chart.png"
    metadata_path = run_paths.outputs_dir / "mission_run_metadata.json"
    bundled_dataset_path = run_paths.inputs_dir / "market_data_norm.csv"
    (paths.processed_data_dir / "market_data_norm.csv").write_text("date,close\n")

    monkeypatch.setattr(module.RepoPaths, "create_run_paths", lambda self, workflow_name: run_paths)
    monkeypatch.setattr(module, "run_single_asset_detection", lambda paths, config=None, output_path=None: results_path)
    monkeypatch.setattr(module, "visualize_single_asset", lambda results, output: chart_path)
    monkeypatch.setattr(module, "build_single_asset_report", lambda results: "# report")
    monkeypatch.setattr(module, "write_run_metadata", lambda path, payload: metadata_path)
    monkeypatch.setattr(module, "git_revision", lambda repo_root: "deadbeef")
    monkeypatch.setattr(module, "_copy_dataset_to_bundle", lambda source_path, run_paths: bundled_dataset_path)
    monkeypatch.setattr(module, "_single_asset_runtime_snapshot", lambda paths, config, dataset_path: {"runtime_config": {"train_batch_size": 1}})
    monkeypatch.setattr(module, "file_sha256", lambda path: "sha256")

    outputs = module.run_single_asset_mission(paths)

    assert outputs == {
        "results": results_path,
        "chart": chart_path,
        "report": run_paths.outputs_dir / "mission_report.md",
        "metadata": metadata_path,
    }
    assert outputs["results"].parent.parent == run_paths.root
    assert outputs["report"].parent.parent == run_paths.root
    assert outputs["metadata"].parent.parent == run_paths.root
    assert bundled_dataset_path.parent.name == "inputs"
    assert (run_paths.outputs_dir / "mission_report.md").read_text() == "# report"
    assert (paths.docs_dir / "mission_report.md").read_text() == "# report"


def test_run_multi_asset_report_writes_report_and_returns_artifacts(load_module, tmp_path, monkeypatch):
    module = load_module(
        "src/openclaw_moe_orchestrator/pipelines.py",
        module_name="openclaw_moe_orchestrator.pipelines",
        injected_modules=_pipeline_import_modules(),
    )
    paths = module.RepoPaths.discover(tmp_path)
    paths.ensure_directories()

    run_paths = paths.create_run_paths("multi-asset-report")
    results_path = run_paths.outputs_dir / "multi_asset_anomalies.csv"
    chart_path = run_paths.outputs_dir / "multi_asset_anomaly_chart.png"
    metadata_path = run_paths.outputs_dir / "multi_asset_report_metadata.json"
    bundled_dataset_path = run_paths.inputs_dir / "multi_asset_returns.csv"
    (paths.processed_data_dir / "multi_asset_returns.csv").write_text("date,btc\n")

    monkeypatch.setattr(module.RepoPaths, "create_run_paths", lambda self, workflow_name: run_paths)
    monkeypatch.setattr(module, "run_multi_asset_detection", lambda paths, config=None, output_path=None: results_path)
    monkeypatch.setattr(module, "visualize_multi_asset", lambda results, output: chart_path)
    monkeypatch.setattr(module, "build_multi_asset_report", lambda results: "# multi report")
    monkeypatch.setattr(module, "write_run_metadata", lambda path, payload: metadata_path)
    monkeypatch.setattr(module, "git_revision", lambda repo_root: "deadbeef")
    monkeypatch.setattr(module, "_copy_dataset_to_bundle", lambda source_path, run_paths: bundled_dataset_path)
    monkeypatch.setattr(module, "_multi_asset_runtime_snapshot", lambda paths, config, dataset_path: {"runtime_config": {"train_batch_size": 1}})
    monkeypatch.setattr(module, "file_sha256", lambda path: "sha256")

    outputs = module.run_multi_asset_report(paths)

    assert outputs == {
        "results": results_path,
        "chart": chart_path,
        "report": run_paths.outputs_dir / "multi_asset_report.md",
        "metadata": metadata_path,
    }
    assert outputs["results"].parent.parent == run_paths.root
    assert outputs["report"].parent.parent == run_paths.root
    assert outputs["metadata"].parent.parent == run_paths.root
    assert bundled_dataset_path.parent.name == "inputs"
    assert (run_paths.outputs_dir / "multi_asset_report.md").read_text() == "# multi report"
    assert (paths.docs_dir / "multi_asset_report.md").read_text() == "# multi report"
