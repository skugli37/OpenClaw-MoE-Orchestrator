from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import yfinance as yf

from .exceptions import DataValidationError
from .paths import RepoPaths
from .settings import MULTI_ASSET_COLUMNS, SINGLE_ASSET_FEATURE_COLUMNS, MultiAssetConfig, SingleAssetConfig

LOGGER = logging.getLogger(__name__)


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        if len(df.columns.levels) == 2 and set(df.columns.get_level_values(0)).issuperset({"Close", "Volume"}):
            df.columns = ["_".join(str(part) for part in col if part) for col in df.columns.to_flat_index()]
        else:
            df.columns = [str(col[-1] or col[0]) for col in df.columns.to_flat_index()]
    return df


def _validate_dataframe(df: pd.DataFrame, context: str, minimum_rows: int = 30) -> None:
    if df.empty:
        raise DataValidationError(f"{context}: downloaded dataset is empty")
    if len(df) < minimum_rows:
        raise DataValidationError(f"{context}: expected at least {minimum_rows} rows, got {len(df)}")
    if df.isna().any().any():
        raise DataValidationError(f"{context}: dataset contains NaN values after preprocessing")


def prepare_market_data(paths: RepoPaths, config: SingleAssetConfig | None = None) -> Path:
    paths.ensure_directories()
    config = config or SingleAssetConfig()

    LOGGER.info("Downloading single-asset market data for %s", config.ticker)
    df = yf.download(
        config.ticker,
        start=config.start_date,
        end=config.end_date,
        auto_adjust=False,
        progress=False,
    )
    if df.empty:
        raise DataValidationError(f"No data returned for {config.ticker}")

    df = _flatten_columns(df)
    if f"Close_{config.ticker}" in df.columns and f"Volume_{config.ticker}" in df.columns:
        df = df.rename(
            columns={
                f"Close_{config.ticker}": "Close",
                f"Volume_{config.ticker}": "Volume",
            }
        )

    required_columns = {"Close", "Volume"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise DataValidationError(f"Missing required columns for {config.ticker}: {sorted(missing_columns)}")

    dataset = df[["Close", "Volume"]].copy()
    dataset["Returns"] = dataset["Close"].pct_change()
    dataset["Vol"] = dataset["Returns"].rolling(window=config.volatility_window).std()
    dataset = dataset.dropna().reset_index()
    if "Date" not in dataset.columns:
        raise DataValidationError("Expected `Date` column after data reset_index()")

    normalized = (dataset[["Close", "Volume", "Returns", "Vol"]] - dataset[["Close", "Volume", "Returns", "Vol"]].mean()) / dataset[["Close", "Volume", "Returns", "Vol"]].std()
    normalized.columns = SINGLE_ASSET_FEATURE_COLUMNS
    prepared = pd.concat([dataset[["Date", "Close"]], normalized], axis=1)
    _validate_dataframe(prepared, "single_asset_prepared_data")

    output_path = paths.processed_data_dir / "market_data_norm.csv"
    prepared.to_csv(output_path, index=False)
    LOGGER.info("Wrote processed single-asset dataset to %s", output_path)
    return output_path


def prepare_multi_asset_data(paths: RepoPaths, config: MultiAssetConfig | None = None) -> Path:
    paths.ensure_directories()
    config = config or MultiAssetConfig()

    LOGGER.info("Downloading multi-asset market data for %s", ",".join(config.tickers))
    df = yf.download(
        list(config.tickers),
        start=config.start_date,
        end=config.end_date,
        auto_adjust=False,
        progress=False,
    )
    if df.empty:
        raise DataValidationError("No multi-asset data returned")

    df = _flatten_columns(df)
    close_columns = [f"Close_{ticker}" for ticker in config.tickers]
    missing_columns = [column for column in close_columns if column not in df.columns]
    if missing_columns:
        raise DataValidationError(f"Missing close columns in multi-asset dataset: {missing_columns}")

    dataset = df[close_columns].copy()
    dataset.columns = list(config.tickers)
    returns = dataset.pct_change().dropna().reset_index()
    if "Date" not in returns.columns:
        raise DataValidationError("Expected `Date` column in multi-asset returns dataset")

    normalized = returns.copy()
    normalized[list(config.tickers)] = (returns[list(config.tickers)] - returns[list(config.tickers)].mean()) / returns[list(config.tickers)].std()
    _validate_dataframe(normalized, "multi_asset_prepared_data")

    output_path = paths.processed_data_dir / "multi_asset_returns.csv"
    normalized.to_csv(output_path, index=False)
    LOGGER.info("Wrote processed multi-asset dataset to %s", output_path)
    return output_path


def load_single_asset_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Date"])
    missing_columns = set(["Date", "Close", *SINGLE_ASSET_FEATURE_COLUMNS]) - set(df.columns)
    if missing_columns:
        raise DataValidationError(f"Single-asset dataset missing columns: {sorted(missing_columns)}")
    _validate_dataframe(df, "single_asset_loaded_data")
    return df


def load_multi_asset_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Date"])
    missing_columns = set(["Date", *MULTI_ASSET_COLUMNS]) - set(df.columns)
    if missing_columns:
        raise DataValidationError(f"Multi-asset dataset missing columns: {sorted(missing_columns)}")
    _validate_dataframe(df, "multi_asset_loaded_data")
    return df
