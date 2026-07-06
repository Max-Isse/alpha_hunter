"""
Optuna hyperparameter optimisation for the Hybrid Transformer-GRU.
Runs 50 trials to find the best d_model, nhead, dropout, and lr.
"""

import torch
import optuna
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

from src.utils.config import AppConfig, ModelConfig, TrainingConfig
from src.utils.logging import setup_logging
from src.data.loader import load_data
from src.data.pipeline import FeaturePipeline
from src.models.transformer_gru import HybridTransformerGRU
from src.training.trainer import Trainer
from src.training.callbacks import EarlyStopping, ModelCheckpoint


def objective(trial, X_train, y_train, X_val, y_val, input_features, device):
    """
    Optuna objective function.
    Suggests hyperparameters, trains a model, and returns validation loss.
    """
    # --- 1. Suggest hyperparameters ---
    d_model = trial.suggest_int("d_model", 32, 256, step=32)
    nhead = trial.suggest_int("nhead", 2, 8, step=2)
    num_layers = trial.suggest_int("num_layers", 1, 3)
    dropout = trial.suggest_float("dropout", 0.05, 0.4)
    lr = trial.suggest_float("lr", 1e-4, 1e-3, log=True)
    batch_size = trial.suggest_int("batch_size", 32, 128, step=32)

    # Ensure d_model is divisible by nhead (requirement for multi-head attention)
    if d_model % nhead != 0:
        return float("inf")

    # --- 2. Build model ---
    model = HybridTransformerGRU(
        n_features=input_features,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)

    # --- 3. Create DataLoaders with the suggested batch size ---
    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train), torch.tensor(y_train)),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(X_val), torch.tensor(y_val)),
        batch_size=batch_size,
        shuffle=False,
    )

    # --- 4. Create a dynamic TrainingConfig with the suggested lr ---
    train_config = TrainingConfig(
        batch_size=batch_size,
        lr=lr,
        weight_decay=1e-5,
        epochs=100,  # Reduced for speed; early stopping will cut it short
        early_stopping_patience=10,
        gradient_clip_norm=1.0,
        scheduler_factor=0.5,
        scheduler_patience=10,
        optimizer="adamw",
    )

    # --- 5. Trainer with callbacks ---
    callbacks = [
        EarlyStopping(patience=train_config.early_stopping_patience),
    ]
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=train_config,
        device=device,
        callbacks=callbacks,
    )

    # --- 6. Train and return the best validation loss ---
    trainer.fit()
    best_val_loss = min(trainer.val_losses) if trainer.val_losses else float("inf")

    return best_val_loss


def main():
    # Setup
    setup_logging(level="INFO")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔧 Using device: {device}")

    # --- Load data ONCE (outside the optimisation loop) ---
    print("📥 Loading data...")
    config = AppConfig()
    df = load_data(config.data.source, config.data.symbol, config.data.start, config.data.end)
    pipeline = FeaturePipeline(lookback=config.model.lookback, horizon=config.model.horizon)
    X, y = pipeline.fit_transform(df)

    # Split chronologically
    n = len(X)
    train_end = int(0.7 * n)
    val_end = int(0.85 * n)
    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]

    print(f"✅ Data loaded: Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}")

    # --- Run Optuna study ---
    print("\n🔍 Starting Optuna hyperparameter optimisation...")
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10),
    )

    study.optimize(
        lambda trial: objective(
            trial,
            X_train,
            y_train,
            X_val,
            y_val,
            X.shape[2],
            device,
        ),
        n_trials=5,
        show_progress_bar=True,
    )

    # --- Results ---
    print("\n🏆 Best trial:")
    best_trial = study.best_trial
    print(f"  Validation Loss: {best_trial.value:.6f}")
    print("  Hyperparameters:")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")

    # Save the best parameters to a YAML file for later use
    import yaml
    best_config = {
        "model": {
            "d_model": best_trial.params["d_model"],
            "nhead": best_trial.params["nhead"],
            "num_layers": best_trial.params["num_layers"],
            "dropout": best_trial.params["dropout"],
        },
        "training": {
            "lr": best_trial.params["lr"],
            "batch_size": best_trial.params["batch_size"],
        },
    }
    with open("configs/best_optuna_config.yaml", "w") as f:
        yaml.dump(best_config, f, default_flow_style=False)
    print("\n✅ Best hyperparameters saved to configs/best_optuna_config.yaml")


if __name__ == "__main__":
    main()