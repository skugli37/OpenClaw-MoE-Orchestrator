from __future__ import annotations

import types

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest


def _single_asset_download_frame(periods: int = 45) -> pd.DataFrame:
    dates = pd.date_range("2025-01-01", periods=periods, freq="D", name="Date")
    base = pd.DataFrame(
        {
            "Close": 100.0 + np.arange(periods) * 1.5 + (np.arange(periods) % 5) * 0.2,
            "Volume": 1_000.0 + np.arange(periods) * 15.0 + (np.arange(periods) % 3) * 7.0,
        },
        index=dates,
    )
    base.columns = pd.MultiIndex.from_product([base.columns, ["BTC-USD"]])
    return base


def test_flatten_columns_joins_ohlcv_multiindex_levels(load_module):
    yfinance = types.SimpleNamespace(download=lambda *args, **kwargs: None)
    module = load_module(
        "src/openclaw_moe_orchestrator/data_pipeline.py",
        module_name="openclaw_moe_orchestrator.data_pipeline",
        injected_modules={"yfinance": yfinance},
    )

    df = pd.DataFrame(
        [[100.0, 1200.0]],
        columns=pd.MultiIndex.from_tuples([("Close", "BTC-USD"), ("Volume", "BTC-USD")]),
    )

    flattened = module._flatten_columns(df.copy())

    assert list(flattened.columns) == ["Close_BTC-USD", "Volume_BTC-USD"]


def test_prepare_market_data_writes_expected_normalized_dataset(load_module, tmp_path):
    downloaded = _single_asset_download_frame()
    yfinance = types.SimpleNamespace(download=lambda *args, **kwargs: downloaded)
    module = load_module(
        "src/openclaw_moe_orchestrator/data_pipeline.py",
        module_name="openclaw_moe_orchestrator.data_pipeline",
        injected_modules={"yfinance": yfinance},
    )
    paths = module.RepoPaths.discover(tmp_path)

    output_path = module.prepare_market_data(paths)
    written = pd.read_csv(output_path, parse_dates=["Date"])

    expected = downloaded.copy()
    expected.columns = ["Close", "Volume"]
    expected["Returns"] = expected["Close"].pct_change()
    expected["Vol"] = expected["Returns"].rolling(window=10).std()
    expected = expected.dropna().reset_index()
    normalized = (expected[["Close", "Volume", "Returns", "Vol"]] - expected[["Close", "Volume", "Returns", "Vol"]].mean()) / expected[["Close", "Volume", "Returns", "Vol"]].std()
    normalized.columns = module.SINGLE_ASSET_FEATURE_COLUMNS
    expected = pd.concat([expected[["Date", "Close"]], normalized], axis=1)

    assert output_path.name == "market_data_norm.csv"
    assert output_path.parent == paths.processed_data_dir
    pdt.assert_frame_equal(written, expected, check_dtype=False)


def test_prepare_market_data_rejects_missing_volume_column(load_module, tmp_path):
    downloaded = _single_asset_download_frame().drop(columns=[("Volume", "BTC-USD")])
    yfinance = types.SimpleNamespace(download=lambda *args, **kwargs: downloaded)
    module = load_module(
        "src/openclaw_moe_orchestrator/data_pipeline.py",
        module_name="openclaw_moe_orchestrator.data_pipeline",
        injected_modules={"yfinance": yfinance},
    )
    paths = module.RepoPaths.discover(tmp_path)

    with pytest.raises(module.DataValidationError, match="Missing required columns"):
        module.prepare_market_data(paths)
