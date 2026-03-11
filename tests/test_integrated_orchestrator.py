from __future__ import annotations

import types
from contextlib import nullcontext

import numpy as np
import pandas as pd
import pytest


class FakeTensor:
    def __init__(self, data, *, device="cpu"):
        self.data = np.array(data)
        self.device = device

    def float(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.data

    def flatten(self):
        return FakeTensor(self.data.flatten(), device=self.device)

    def item(self):
        return np.array(self.data).item()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        if isinstance(item, FakeTensor):
            item = item.data
        return FakeTensor(self.data[item], device=self.device)

    def __sub__(self, other):
        other_data = other.data if isinstance(other, FakeTensor) else other
        return FakeTensor(self.data - other_data, device=self.device)

    def __pow__(self, power):
        return FakeTensor(self.data ** power, device=self.device)

    def __gt__(self, other):
        other_data = other.data if isinstance(other, FakeTensor) else other
        return FakeTensor(self.data > other_data, device=self.device)


class FakeNoGrad:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return False


class FakeTorchRuntime:
    cuda = types.SimpleNamespace(is_available=lambda: False)
    float16 = "float16"
    float32 = "float32"

    @staticmethod
    def tensor(value, dtype=None, device=None):
        return FakeTensor(value, device=device or "cpu")

    @staticmethod
    def mean(value, dim=None):
        return FakeTensor(np.mean(value.data, axis=dim), device=value.device)

    @staticmethod
    def quantile(value, q):
        return FakeTensor(np.quantile(value.data, q), device=value.device)

    @staticmethod
    def nonzero(value, as_tuple=False):
        return FakeTensor(np.argwhere(value.data), device=value.device)

    @staticmethod
    def argmax(value):
        return FakeTensor(np.argmax(value.data))

    @staticmethod
    def abs(value):
        return FakeTensor(np.abs(value.data), device=value.device)

    @staticmethod
    def no_grad():
        return FakeNoGrad()


class FakeLoss:
    def item(self):
        return 0.0


class FakeMSELoss:
    def __call__(self, outputs, inputs):
        return FakeLoss()


class FakeModel:
    def __init__(self, *args, **kwargs):
        self._params = [types.SimpleNamespace(device="cpu")]

    def parameters(self):
        return iter(self._params)


class FakeEngine:
    def __init__(self):
        self.module = lambda data: data
        self.step_calls = 0
        self.backward_calls = 0

    def __call__(self, data):
        return data

    def backward(self, loss):
        self.backward_calls += 1

    def step(self):
        self.step_calls += 1


def _pipeline_import_modules():
    torch_module = types.ModuleType("torch")
    torch_module.cuda = types.SimpleNamespace(is_available=lambda: False)
    torch_module.float16 = "float16"
    torch_module.float32 = "float32"
    nn_module = types.ModuleType("torch.nn")
    nn_module.Module = object
    nn_module.MSELoss = FakeMSELoss
    torch_module.nn = nn_module

    deepspeed_module = types.ModuleType("deepspeed")
    deepspeed_module.initialize = lambda **kwargs: (FakeEngine(), None, None, None)

    data_pipeline_module = types.ModuleType("openclaw_moe_orchestrator.data_pipeline")
    data_pipeline_module.load_multi_asset_dataset = lambda *args, **kwargs: None
    data_pipeline_module.load_single_asset_dataset = lambda *args, **kwargs: None
    data_pipeline_module.prepare_market_data = lambda *args, **kwargs: None
    data_pipeline_module.prepare_multi_asset_data = lambda *args, **kwargs: None

    models_module = types.ModuleType("openclaw_moe_orchestrator.models")
    models_module.IntelligenceMoE = FakeModel
    models_module.MoEAutoencoder = FakeModel
    models_module.MultiAssetMoE = FakeModel

    news_module = types.ModuleType("openclaw_moe_orchestrator.news")
    news_module.get_live_news = lambda asset_name: f"{asset_name} headline"

    reports_module = types.ModuleType("openclaw_moe_orchestrator.reports")
    reports_module.build_single_asset_report = lambda path: "report"
    reports_module.build_multi_asset_report = lambda path: "multi-report"

    runtime_module = types.ModuleType("openclaw_moe_orchestrator.runtime")
    runtime_module.gpu_execution_lock = lambda *args, **kwargs: nullcontext()
    runtime_module.load_runtime_config = lambda *args, **kwargs: {}
    runtime_module.prepare_distributed_env = lambda: None
    runtime_module.prepare_model_and_optimizer = lambda model, runtime_config: (model, object(), ["params"])
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


def test_run_integrated_orchestrator_raises_when_no_anomalies_are_found(load_module, tmp_path, monkeypatch):
    module = load_module(
        "src/openclaw_moe_orchestrator/pipelines.py",
        module_name="openclaw_moe_orchestrator.pipelines",
        injected_modules=_pipeline_import_modules(),
    )
    paths = module.RepoPaths.discover(tmp_path)
    data = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2025-03-01", "2025-03-02", "2025-03-03"]),
            "BTC-USD": [0.2, 0.1, -0.1],
            "ETH-USD": [0.0, 0.2, -0.2],
            "SOL-USD": [0.1, -0.1, 0.0],
        }
    )
    engine = FakeEngine()
    shutdown = {"called": False}

    monkeypatch.setattr(module, "prepare_multi_asset_data", lambda paths, config=None: paths.processed_data_dir / "multi_asset_returns.csv")
    monkeypatch.setattr(module, "load_multi_asset_dataset", lambda dataset_path: data.copy())
    monkeypatch.setattr(module, "load_runtime_config", lambda config_path, batch_size: {"batch_size": batch_size})
    monkeypatch.setattr(module, "prepare_distributed_env", lambda: None)
    monkeypatch.setattr(module, "prepare_model_and_optimizer", lambda model, runtime_config: (model, object(), ["params"]))
    monkeypatch.setattr(module, "IntelligenceMoE", FakeModel)
    monkeypatch.setattr(module, "torch", FakeTorchRuntime)
    monkeypatch.setattr(module.nn, "MSELoss", FakeMSELoss)
    monkeypatch.setattr(module.deepspeed, "initialize", lambda **kwargs: (engine, None, None, None))
    monkeypatch.setattr(module, "shutdown_distributed", lambda: shutdown.__setitem__("called", True))

    with pytest.raises(module.ExternalSignalError, match="found no anomalies"):
        module.run_integrated_orchestrator(paths)

    assert engine.backward_calls == 100
    assert engine.step_calls == 100
    assert shutdown["called"] is True


def test_run_integrated_orchestrator_returns_top_anomaly_with_news(load_module, tmp_path, monkeypatch):
    module = load_module(
        "src/openclaw_moe_orchestrator/pipelines.py",
        module_name="openclaw_moe_orchestrator.pipelines",
        injected_modules=_pipeline_import_modules(),
    )
    paths = module.RepoPaths.discover(tmp_path)
    data = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2025-03-01", "2025-03-02", "2025-03-03", "2025-03-04"]),
            "BTC-USD": [0.0, 0.1, 0.2, 4.0],
            "ETH-USD": [0.0, 0.1, 0.2, 0.1],
            "SOL-USD": [0.0, 0.1, 0.2, 0.1],
        }
    )
    engine = FakeEngine()
    shutdown = {"called": False}

    def anomalous_reconstruction(inputs):
        adjusted = inputs.data.copy()
        adjusted[-1, 0] = adjusted[-1, 0] - 3.5
        return FakeTensor(adjusted, device=inputs.device)

    engine.module = anomalous_reconstruction

    monkeypatch.setattr(module, "prepare_multi_asset_data", lambda paths, config=None: paths.processed_data_dir / "multi_asset_returns.csv")
    monkeypatch.setattr(module, "load_multi_asset_dataset", lambda dataset_path: data.copy())
    monkeypatch.setattr(module, "load_runtime_config", lambda config_path, batch_size: {"batch_size": batch_size})
    monkeypatch.setattr(module, "prepare_distributed_env", lambda: None)
    monkeypatch.setattr(module, "prepare_model_and_optimizer", lambda model, runtime_config: (model, object(), ["params"]))
    monkeypatch.setattr(module, "IntelligenceMoE", FakeModel)
    monkeypatch.setattr(module, "torch", FakeTorchRuntime)
    monkeypatch.setattr(module.nn, "MSELoss", FakeMSELoss)
    monkeypatch.setattr(module.deepspeed, "initialize", lambda **kwargs: (engine, None, None, None))
    monkeypatch.setattr(module, "get_live_news", lambda asset_name: f"{asset_name} breakout headline")
    monkeypatch.setattr(module, "shutdown_distributed", lambda: shutdown.__setitem__("called", True))

    result = module.run_integrated_orchestrator(paths)

    assert result["asset"] == "BTC"
    assert result["date"] == "2025-03-04"
    assert result["news"] == "BTC breakout headline"
    assert result["score"] == pytest.approx(4.083333333333333)
    assert shutdown["called"] is True
