from __future__ import annotations

import json
import types

import pytest


class FakeModel:
    def __init__(self):
        self.operations = []
        self._params = [types.SimpleNamespace(device="cpu"), types.SimpleNamespace(device="cpu")]

    def parameters(self):
        return iter(self._params)

    def cuda(self):
        self.operations.append("cuda")
        return self

    def half(self):
        self.operations.append("half")
        return self

    def float(self):
        self.operations.append("float")
        return self


def _fake_runtime_modules(cuda_available=False, dist_initialized=False):
    torch_module = types.ModuleType("torch")
    empty_cache_state = {"called": False}
    torch_module.cuda = types.SimpleNamespace(
        is_available=lambda: cuda_available,
        get_device_properties=lambda idx: types.SimpleNamespace(total_memory=8 * 1024 * 1024 * 1024),
        empty_cache=lambda: empty_cache_state.__setitem__("called", True),
    )
    torch_module.float16 = "float16"
    torch_module.float32 = "float32"
    torch_module.nn = types.SimpleNamespace(Module=object)
    dist_module = types.ModuleType("torch.distributed")
    dist_state = {"destroyed": False}

    class FakeAdam:
        def __init__(self, param_groups, lr):
            self.param_groups = param_groups
            self.lr = lr

    torch_module.optim = types.SimpleNamespace(Adam=FakeAdam)
    dist_module.is_available = lambda: True
    dist_module.is_initialized = lambda: dist_initialized
    dist_module.destroy_process_group = lambda: dist_state.__setitem__("destroyed", True)
    torch_module.distributed = dist_module

    deepspeed_module = types.ModuleType("deepspeed")
    moe_module = types.ModuleType("deepspeed.moe")
    utils_module = types.ModuleType("deepspeed.moe.utils")
    utils_module.split_params_into_different_moe_groups_for_optimizer = lambda group: [group]
    deepspeed_module.moe = moe_module
    moe_module.utils = utils_module

    return {
        "torch": torch_module,
        "torch.distributed": dist_module,
        "deepspeed": deepspeed_module,
        "deepspeed.moe": moe_module,
        "deepspeed.moe.utils": utils_module,
    }, dist_state, empty_cache_state


def test_load_runtime_config_strips_offload_and_sets_batch_size(load_module, tmp_path):
    injected_modules, _, _ = _fake_runtime_modules(cuda_available=False)
    module = load_module(
        "src/openclaw_moe_orchestrator/runtime.py",
        module_name="openclaw_moe_orchestrator.runtime",
        injected_modules=injected_modules,
    )

    config_path = tmp_path / "ds_config.json"
    config_path.write_text(
        json.dumps(
            {
                "optimizer": {"params": {"lr": 0.005}},
                "fp16": {"enabled": True},
                "zero_optimization": {
                    "stage": 2,
                    "offload_optimizer": {"device": "cpu"},
                    "offload_param": {"device": "cpu"},
                    "cpu_offload": True,
                },
            }
        )
    )

    runtime_config = module.load_runtime_config(config_path, batch_size=8)

    assert runtime_config["train_batch_size"] == 8
    assert runtime_config["train_micro_batch_size_per_gpu"] == 8
    assert runtime_config["gradient_accumulation_steps"] == 1
    assert runtime_config["zero_optimization"] == {"stage": 2}
    assert runtime_config["fp16"]["enabled"] is False


def test_prepare_model_and_optimizer_uses_runtime_lr_and_param_groups(load_module):
    injected_modules, _, _ = _fake_runtime_modules(cuda_available=False)
    module = load_module(
        "src/openclaw_moe_orchestrator/runtime.py",
        module_name="openclaw_moe_orchestrator.runtime",
        injected_modules=injected_modules,
    )

    model = FakeModel()

    prepared_model, optimizer, param_groups = module.prepare_model_and_optimizer(
        model,
        {"optimizer": {"params": {"lr": 0.123}}},
    )

    assert prepared_model is model
    assert model.operations == ["float"]
    assert optimizer.lr == 0.123
    assert len(param_groups) == 1
    assert param_groups[0]["name"] == "main"
    assert list(param_groups[0]["params"]) == list(model.parameters())


def test_shutdown_distributed_destroys_initialized_process_group(load_module):
    injected_modules, dist_state, _ = _fake_runtime_modules(dist_initialized=True)
    module = load_module(
        "src/openclaw_moe_orchestrator/runtime.py",
        module_name="openclaw_moe_orchestrator.runtime",
        injected_modules=injected_modules,
    )

    module.shutdown_distributed()

    assert dist_state["destroyed"] is True


def test_load_runtime_config_caps_zero_bucket_sizes_for_smaller_gpus(load_module, tmp_path):
    injected_modules, _, _ = _fake_runtime_modules(cuda_available=True)
    module = load_module(
        "src/openclaw_moe_orchestrator/runtime.py",
        module_name="openclaw_moe_orchestrator.runtime",
        injected_modules=injected_modules,
    )

    config_path = tmp_path / "ds_config.json"
    config_path.write_text(
        json.dumps(
            {
                "fp16": {"enabled": True},
                "zero_optimization": {
                    "stage": 2,
                    "allgather_bucket_size": int(5e8),
                    "reduce_bucket_size": int(5e8),
                },
            }
        )
    )

    runtime_config = module.load_runtime_config(config_path, batch_size=8)

    expected_bucket_size = 67_108_864
    assert runtime_config["zero_optimization"]["allgather_bucket_size"] == expected_bucket_size
    assert runtime_config["zero_optimization"]["reduce_bucket_size"] == expected_bucket_size


def test_gpu_execution_lock_raises_when_timeout_is_exceeded(load_module, tmp_path, monkeypatch):
    injected_modules, _, _ = _fake_runtime_modules(cuda_available=False)
    module = load_module(
        "src/openclaw_moe_orchestrator/runtime.py",
        module_name="openclaw_moe_orchestrator.runtime",
        injected_modules=injected_modules,
    )

    def always_block(*args, **kwargs):
        raise BlockingIOError("busy")

    monkeypatch.setattr(module.fcntl, "flock", always_block)

    with pytest.raises(module.ResourceContentionError, match="exclusive GPU lock"):
        with module.gpu_execution_lock(tmp_path / "gpu-runtime.lock", timeout_seconds=0, poll_interval_seconds=0):
            pass
