from pydantic import BaseModel  # <-- CRITICAL: This was missing
import yaml
from pathlib import Path
from typing import Optional


class DataConfig(BaseModel):
    source: str = "yfinance"
    symbol: str = "SPY"
    start: str = "2010-01-01"
    end: str = "2026-06-01"


class ModelConfig(BaseModel):
    lookback: int = 60
    horizon: int = 5
    d_model: int = 64
    nhead: int = 4
    num_layers: int = 2
    dropout: float = 0.1
    optimizer: str = "adamw"
    loss: str = "mse"


class TrainingConfig(BaseModel):
    batch_size: int = 64
    lr: float = 0.001
    weight_decay: float = 1e-5
    epochs: int = 200
    early_stopping_patience: int = 15
    gradient_clip_norm: float = 1.0
    scheduler_factor: float = 0.5
    scheduler_patience: int = 10
    optimizer: str = "adamw"  # Added to match create_optimizer


class BacktestConfig(BaseModel):
    initial_capital: float = 100000.0
    position_size: float = 0.1
    commission: float = 0.001
    slippage: float = 0.0005


class AppConfig(BaseModel):
    data: DataConfig = DataConfig()
    model: ModelConfig = ModelConfig()
    training: TrainingConfig = TrainingConfig()
    backtest: BacktestConfig = BacktestConfig()

    @classmethod
    def from_yaml(cls, path: Path) -> "AppConfig":
        with open(path, "r") as f:
            raw = yaml.safe_load(f)
        return cls(**raw)


def load_config(path: Optional[Path] = None) -> AppConfig:
    """Load configuration from a YAML file, falling back to defaults."""
    if path is None:
        path = Path("configs/base.yaml")
    if not path.exists():
        return AppConfig()  # fallback to defaults
    return AppConfig.from_yaml(path)