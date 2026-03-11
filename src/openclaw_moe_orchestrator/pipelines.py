from __future__ import annotations

import logging
from pathlib import Path

import deepspeed
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .data_pipeline import (
    load_multi_asset_dataset,
    load_single_asset_dataset,
    prepare_market_data,
    prepare_multi_asset_data,
)
from .exceptions import ExternalSignalError
from .metadata import git_revision, write_run_metadata
from .models import IntelligenceMoE, MoEAutoencoder, MultiAssetMoE
from .news import get_live_news
from .paths import RepoPaths
from .reports import build_multi_asset_report, build_single_asset_report
from .runtime import (
    gpu_execution_lock,
    load_runtime_config,
    prepare_distributed_env,
    prepare_model_and_optimizer,
    shutdown_distributed,
    tensor_dtype,
)
from .settings import MULTI_ASSET_COLUMNS, SINGLE_ASSET_FEATURE_COLUMNS, MultiAssetConfig, SingleAssetConfig
from .visualization import visualize_multi_asset, visualize_single_asset

LOGGER = logging.getLogger(__name__)


def _config_path(paths: RepoPaths) -> Path:
    return paths.config_dir / "ds_config_zero2.json"


def run_single_asset_detection(paths: RepoPaths, config: SingleAssetConfig | None = None) -> Path:
    paths.ensure_directories()
    config = config or SingleAssetConfig()
    dataset_path = prepare_market_data(paths, config=config)
    dataset = load_single_asset_dataset(dataset_path)

    features = dataset[SINGLE_ASSET_FEATURE_COLUMNS].values.astype(np.float32)
    output_path = paths.artifacts_dir / "anomaly_results.csv"
    with gpu_execution_lock(paths.logs_dir / "gpu-runtime.lock"):
        runtime_config = load_runtime_config(_config_path(paths), batch_size=len(features))
        prepare_distributed_env()

        model = MoEAutoencoder(input_dim=len(SINGLE_ASSET_FEATURE_COLUMNS), hidden_dim=16, num_experts=4)
        model, optimizer, model_parameters = prepare_model_and_optimizer(model, runtime_config)
        device = next(model.parameters()).device
        inputs = torch.tensor(features, dtype=tensor_dtype(), device=device)

        try:
            model_engine, _, _, _ = deepspeed.initialize(
                args=None,
                model=model,
                model_parameters=model_parameters,
                optimizer=optimizer,
                config=runtime_config,
            )

            for epoch in range(config.epochs):
                outputs = model_engine(inputs)
                loss = nn.MSELoss()(outputs, inputs)
                model_engine.backward(loss)
                model_engine.step()
                if epoch % 10 == 0:
                    LOGGER.info("single_asset epoch=%s loss=%.6f", epoch, loss.item())

            with torch.no_grad():
                reconstructions = model_engine.module(inputs)
                mse = torch.mean((reconstructions.float() - inputs.float()) ** 2, dim=1)
                threshold = torch.quantile(mse, config.anomaly_quantile)
                results = pd.DataFrame(
                    {
                        "Date": dataset["Date"],
                        "Close": dataset["Close"],
                        "Is_Anomaly": (mse > threshold).cpu().numpy(),
                        "Reconstruction_Error": mse.cpu().numpy(),
                    }
                )
                results.to_csv(output_path, index=False)
                LOGGER.info("Wrote single-asset detection results to %s", output_path)
        finally:
            shutdown_distributed()
    return output_path


def run_multi_asset_detection(paths: RepoPaths, config: MultiAssetConfig | None = None) -> Path:
    paths.ensure_directories()
    config = config or MultiAssetConfig()
    dataset_path = prepare_multi_asset_data(paths, config=config)
    dataset = load_multi_asset_dataset(dataset_path)

    features = dataset[MULTI_ASSET_COLUMNS].values.astype(np.float32)
    output_path = paths.artifacts_dir / "multi_asset_anomalies.csv"
    with gpu_execution_lock(paths.logs_dir / "gpu-runtime.lock"):
        runtime_config = load_runtime_config(_config_path(paths), batch_size=len(features))
        prepare_distributed_env()

        model = MultiAssetMoE(input_dim=len(MULTI_ASSET_COLUMNS))
        model, optimizer, model_parameters = prepare_model_and_optimizer(model, runtime_config)
        device = next(model.parameters()).device
        inputs = torch.tensor(features, dtype=tensor_dtype(), device=device)

        try:
            model_engine, _, _, _ = deepspeed.initialize(
                args=None,
                model=model,
                model_parameters=model_parameters,
                optimizer=optimizer,
                config=runtime_config,
            )

            for epoch in range(config.epochs):
                outputs = model_engine(inputs)
                loss = nn.MSELoss()(outputs, inputs)
                model_engine.backward(loss)
                model_engine.step()
                if epoch % 20 == 0:
                    LOGGER.info("multi_asset epoch=%s loss=%.6f", epoch, loss.item())

            with torch.no_grad():
                reconstructions = model_engine.module(inputs)
                mse = torch.mean((reconstructions.float() - inputs.float()) ** 2, dim=1)
                threshold = torch.quantile(mse, config.anomaly_quantile)
                dataset = dataset.copy()
                dataset["is_anomaly"] = (mse > threshold).cpu().numpy()
                dataset["reconstruction_error"] = mse.cpu().numpy()
                dataset.to_csv(output_path, index=False)
                LOGGER.info("Wrote multi-asset detection results to %s", output_path)
        finally:
            shutdown_distributed()
    return output_path


def run_single_asset_mission(paths: RepoPaths, config: SingleAssetConfig | None = None) -> dict[str, Path]:
    results_path = run_single_asset_detection(paths, config=config)
    chart_path = visualize_single_asset(results_path, paths.artifacts_dir / "anomaly_chart.png")
    report_path = paths.docs_dir / "mission_report.md"
    report_path.write_text(build_single_asset_report(results_path))
    metadata_path = write_run_metadata(
        paths.artifacts_dir / "mission_run_metadata.json",
        {
            "git_revision": git_revision(paths.repo_root),
            "results_path": str(results_path),
            "chart_path": str(chart_path),
            "report_path": str(report_path),
        },
    )
    LOGGER.info("Wrote mission report to %s", report_path)
    LOGGER.info("Wrote mission metadata to %s", metadata_path)
    return {
        "results": results_path,
        "chart": chart_path,
        "report": report_path,
        "metadata": metadata_path,
    }


def run_multi_asset_report(paths: RepoPaths, config: MultiAssetConfig | None = None) -> dict[str, Path]:
    results_path = run_multi_asset_detection(paths, config=config)
    chart_path = visualize_multi_asset(results_path, paths.artifacts_dir / "multi_asset_anomaly_chart.png")
    report_path = paths.docs_dir / "multi_asset_report.md"
    report_path.write_text(build_multi_asset_report(results_path))
    metadata_path = write_run_metadata(
        paths.artifacts_dir / "multi_asset_report_metadata.json",
        {
            "git_revision": git_revision(paths.repo_root),
            "results_path": str(results_path),
            "chart_path": str(chart_path),
            "report_path": str(report_path),
        },
    )
    LOGGER.info("Wrote multi-asset report to %s", report_path)
    LOGGER.info("Wrote multi-asset report metadata to %s", metadata_path)
    return {
        "results": results_path,
        "chart": chart_path,
        "report": report_path,
        "metadata": metadata_path,
    }


def run_integrated_orchestrator(paths: RepoPaths, config: MultiAssetConfig | None = None) -> dict[str, str | float]:
    paths.ensure_directories()
    config = config or MultiAssetConfig()
    dataset_path = prepare_multi_asset_data(paths, config=config)
    dataset = load_multi_asset_dataset(dataset_path)

    features = dataset[MULTI_ASSET_COLUMNS].values.astype(np.float32)
    with gpu_execution_lock(paths.logs_dir / "gpu-runtime.lock"):
        runtime_config = load_runtime_config(_config_path(paths), batch_size=len(features))
        prepare_distributed_env()

        model = IntelligenceMoE(input_dim=len(MULTI_ASSET_COLUMNS))
        model, optimizer, model_parameters = prepare_model_and_optimizer(model, runtime_config)
        device = next(model.parameters()).device
        inputs = torch.tensor(features, dtype=tensor_dtype(), device=device)

        try:
            model_engine, _, _, _ = deepspeed.initialize(
                args=None,
                model=model,
                model_parameters=model_parameters,
                optimizer=optimizer,
                config=runtime_config,
            )

            for epoch in range(config.epochs):
                outputs = model_engine(inputs)
                loss = nn.MSELoss()(outputs, inputs)
                model_engine.backward(loss)
                model_engine.step()
                if epoch % 20 == 0:
                    LOGGER.info("integrated epoch=%s loss=%.6f", epoch, loss.item())

            with torch.no_grad():
                reconstructions = model_engine.module(inputs)
                mse = torch.mean((reconstructions.float() - inputs.float()) ** 2, dim=1)
                threshold = torch.quantile(mse, config.integrated_quantile)
                anomaly_indices = torch.nonzero(mse > threshold, as_tuple=False).flatten()
                if len(anomaly_indices) == 0:
                    raise ExternalSignalError("Model completed successfully but found no anomalies above threshold")

                top_idx = anomaly_indices[torch.argmax(mse[anomaly_indices])].item()
                asset_idx = torch.argmax(torch.abs(inputs[top_idx])).item()
                trigger_asset = ["BTC", "ETH", "SOL"][asset_idx]
                news_summary = get_live_news(trigger_asset)
                result = {
                    "asset": trigger_asset,
                    "date": str(dataset.loc[top_idx, "Date"].date()),
                    "score": float(mse[top_idx].item()),
                    "news": news_summary,
                }
                metadata_path = write_run_metadata(
                    paths.artifacts_dir / "integrated_run_metadata.json",
                    {
                        "git_revision": git_revision(paths.repo_root),
                        "result": result,
                        "dataset_path": str(dataset_path),
                    },
                )
                result["metadata_path"] = str(metadata_path)
                LOGGER.info("Integrated orchestrator result: %s", result)
                return result
        finally:
            shutdown_distributed()


def run_multi_asset_visualization(paths: RepoPaths) -> Path:
    results_path = paths.artifacts_dir / "multi_asset_anomalies.csv"
    return visualize_multi_asset(results_path, paths.artifacts_dir / "multi_asset_anomaly_chart.png")
