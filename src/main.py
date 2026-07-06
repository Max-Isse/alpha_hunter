import click
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset

from src.utils.config import load_config
from src.utils.logging import setup_logging, logger  # <-- FIXED: setup_logging, not setup_logger
from src.data.loader import load_data
from src.data.pipeline import FeaturePipeline
from src.models import HybridTransformerGRU
from src.training import Trainer, EarlyStopping, ModelCheckpoint
from src.backtest import run_backtest
from src.utils.visualization import plot_training_history, plot_predictions


@click.command()
@click.option("--config", "config_path", default="configs/base.yaml", type=click.Path(exists=True))
@click.option("--mode", default="full", type=click.Choice(["train", "backtest", "full"]))
def cli(config_path: str, mode: str):
    # --- Setup ---
    setup_logging(level="INFO")  # <-- FIXED: setup_logging
    config = load_config(Path(config_path))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # --- Data ---
    logger.info(f"Loading data: {config.data.symbol} from {config.data.start} to {config.data.end}")
    df = load_data(config.data.source, config.data.symbol, config.data.start, config.data.end)
    pipeline = FeaturePipeline(lookback=config.model.lookback, horizon=config.model.horizon)
    X, y = pipeline.fit_transform(df)
    logger.info(f"Created sequences: X={X.shape}, y={y.shape}")

    # --- Split chronologically ---
    n = len(X)
    train_end = int(0.7 * n)
    val_end = int(0.85 * n)
    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]
    logger.info(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    # --- DataLoaders ---
    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32)),
        batch_size=config.training.batch_size,
        shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32)),
        batch_size=config.training.batch_size,
        shuffle=False
    )
    test_loader = DataLoader(
        TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32)),
        batch_size=config.training.batch_size,
        shuffle=False
    )

    # --- Model ---
    model = HybridTransformerGRU(
        n_features=X.shape[2],
        d_model=config.model.d_model,
        nhead=config.model.nhead,
        num_layers=config.model.num_layers,
        dropout=config.model.dropout
    )
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # --- Callbacks ---
    callbacks = [
        EarlyStopping(patience=config.training.early_stopping_patience),
        ModelCheckpoint("results/models/best_model.pth")
    ]

    # --- Trainer ---
    trainer = Trainer(model, train_loader, val_loader, config.training, device, callbacks)

    # --- Execute ---
    if mode in ["train", "full"]:
        logger.info("Starting training...")
        train_losses, val_losses = trainer.fit()
        plot_training_history(train_losses, val_losses, "results/plots/training_history.png")
        logger.info(f"Training complete. Best validation loss: {min(val_losses):.6f}")

    if mode in ["backtest", "full"]:
        # Load best checkpoint
        checkpoint_path = Path("results/models/best_model.pth")
        if not checkpoint_path.exists():
            logger.error("No trained model found. Please run training first.")
            return
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.to(device)
        model.eval()

        # Predict on test set
        y_pred_list = []
        with torch.no_grad():
            for X_batch, _ in test_loader:
                X_batch = X_batch.to(device)
                preds = model(X_batch)
                y_pred_list.append(preds.cpu().numpy())
        y_pred = np.concatenate(y_pred_list)
        y_test_np = y_test

        # --- Diagnostic logging ---
        logger.info(f"Predictions: min={y_pred.min():.6f}, mean={y_pred.mean():.6f}, max={y_pred.max():.6f}, std={y_pred.std():.6f}")
        logger.info(f"Actuals:     min={y_test_np.min():.6f}, mean={y_test_np.mean():.6f}, max={y_test_np.max():.6f}, std={y_test_np.std():.6f}")
        logger.info(f"Positive predictions: {np.mean(y_pred > 0):.1%}, Negative: {np.mean(y_pred < 0):.1%}")

        # Plot predictions vs actuals
        plot_predictions(y_test_np, y_pred, "results/plots/predictions.png")

        # --- Backtest ---
        logger.info("Running backtest...")
        results = run_backtest(y_pred, y_test_np, config.backtest, horizon=config.model.horizon)

        # Save equity curve CSV
        results.to_csv("results/backtest_equity.csv", index=False)

    logger.info("Pipeline complete.")


if __name__ == "__main__":
    cli()