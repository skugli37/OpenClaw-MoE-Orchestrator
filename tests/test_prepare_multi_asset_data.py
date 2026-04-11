from __future__ import annotations

import types

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest


ASSETS = ["BTC-USD", "ETH-USD", "SOL-USD"]


def _multi_asset_download_frame(periods: int = 40) -> pd.DataFrame:
    dates = pd.date_range("2025-01-01", periods=periods, freq="D", name="Date")
    close = pd.DataFrame(
        {
            "BTC-USD": 100.0 + np.arange(periods) * 2.0 + (np.arange(periods) % 4) * 0.3,
            "ETH-USD": 10.0 + np.arange(periods) * 0.25 + (np.arange(periods) % 3) * 0.05,
            "SOL-USD": 1.0 + np.arange(periods) * 0.04 + (np.arange(periods) % 5) * 0.01,
        },
        index=dates,
    )
    volume = pd.DataFrame(
        {
            "BTC-USD": 5_000.0 + np.arange(periods) * 25.0,
            "ETH-USD": 8_000.0 + np.arange(periods) * 30.0,
            "SOL-USD": 12_000.0 + np.arange(periods) * 35.0,
        },
        index=dates,
    )
    joined = pd.concat({"Close": close, "Volume": volume}, axis=1)
    joined.columns = pd.MultiIndex.from_tuples(joined.columns.to_flat_index())
    return joined


def test_flatten_columns_joins_multiindex_levels(load_module):
    yfinance = types.SimpleNamespace(download=lambda *args, **kwargs: None)
    module = load_module(
        "src/openclaw_moe_orchestrator/data_pipeline.py",
        module_name="openclaw_moe_orchestrator.data_pipeline",
        injected_modules={"yfinance": yfinance},
    )

    df = pd.DataFrame(
        [[100.0, 5000.0]],
        columns=pd.MultiIndex.from_tuples([("Close", "BTC-USD"), ("Volume", "BTC-USD")]),
    )

    flattened = module._flatten_columns(df.copy())

    assert list(flattened.columns) == ["Close_BTC-USD", "Volume_BTC-USD"]


def test_prepare_multi_asset_data_writes_expected_normalized_returns(load_module, tmp_path):
    downloaded = _multi_asset_download_frame()
    yfinance = types.SimpleNamespace(download=lambda *args, **kwargs: downloaded)
    module = load_module(
        "src/openclaw_moe_orchestrator/data_pipeline.py",
        module_name="openclaw_moe_orchestrator.data_pipeline",
        injected_modules={"yfinance": yfinance},
    )
    paths = module.RepoPaths.discover(tmp_path)

    output_path = module.prepare_multi_asset_data(paths)
    written = pd.read_csv(output_path, parse_dates=["Date"])

    expected = downloaded.copy()
    expected = expected[[("Close", asset) for asset in ASSETS]].copy()
    expected.columns = ASSETS
    expected = expected.pct_change().dropna().reset_index()
    expected[ASSETS] = (expected[ASSETS] - expected[ASSETS].mean()) / expected[ASSETS].std()

    assert output_path.name == "multi_asset_returns.csv"
    assert output_path.parent == paths.processed_data_dir
    pdt.assert_frame_equal(written, expected, check_dtype=False)


def test_prepare_multi_asset_data_rejects_missing_asset_close_columns(load_module, tmp_path):
    downloaded = _multi_asset_download_frame().drop(columns=[("Close", "SOL-USD")])
    yfinance = types.SimpleNamespace(download=lambda *args, **kwargs: downloaded)
    module = load_module(
        "src/openclaw_moe_orchestrator/data_pipeline.py",
        module_name="openclaw_moe_orchestrator.data_pipeline",
        injected_modules={"yfinance": yfinance},
    )
    paths = module.RepoPaths.discover(tmp_path)

    with pytest.raises(module.DataValidationError, match="Missing close columns"):
        module.prepare_multi_asset_data(paths)
