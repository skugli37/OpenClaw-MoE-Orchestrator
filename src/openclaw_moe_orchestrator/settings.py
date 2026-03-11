from dataclasses import dataclass


SINGLE_ASSET_FEATURE_COLUMNS = [
    "feature_close",
    "feature_volume",
    "feature_returns",
    "feature_vol",
]
MULTI_ASSET_COLUMNS = ["BTC-USD", "ETH-USD", "SOL-USD"]


@dataclass(frozen=True)
class SingleAssetConfig:
    ticker: str = "BTC-USD"
    start_date: str = "2025-01-01"
    end_date: str = "2026-02-15"
    volatility_window: int = 10
    epochs: int = 50
    anomaly_quantile: float = 0.95


@dataclass(frozen=True)
class MultiAssetConfig:
    tickers: tuple[str, ...] = ("BTC-USD", "ETH-USD", "SOL-USD")
    start_date: str = "2025-01-01"
    end_date: str = "2026-03-01"
    epochs: int = 100
    anomaly_quantile: float = 0.95
    integrated_quantile: float = 0.98
