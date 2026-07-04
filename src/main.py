import click
import torch
from pathlib import Path
from src.utils.config import load_config
from src.utils.logging import setup_logging
from src.data.loader import load_data
from src.data.pipeline import FeaturePipeline
from src.models import HybridTransformerGRU
from src.training import Trainer, EarlyStopping, ModelCheckpoint
from src.backtest import run_backtest
from src.utils.visualization import plot_training_history, plot_predictions
from torch.utils.data import DataLoader, TensorDataset
import numpy as np


@click.command()
@click.option("--config", "config_path", default="configs/base.yaml", type=click.Path(exists=True))
@click.option("--mode", default="full", type=click.Choice(["train", "backtest", "full"]))
def cli(config_path: str, mode: str):
    # Setup
    setup_logging()
    config = load_config(Path(config_path))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data
    df = load_data(config.data.source, config.data.symbol, config.data.start, config.data.end)
    pipeline = FeaturePipeline(lookback=config.model.lookback, horizon=config.model.horizon)
    X, y = pipeline.fit_transform(df)

    # Split
    n = len(X)
    train_end = int(0.7 * n)
    val_end = int(0.85 * n)
    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]

    # Dataloaders
    train_loader = DataLoader(TensorDataset(torch.tensor(X_train), torch.tensor(y_train)), batch_size=config.training.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.tensor(X_val), torch.tensor(y_val)), batch_size=config.training.batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(torch.tensor(X_test), torch.tensor(y_test)), batch_size=config.training.batch_size, shuffle=False)

    # Model
    model = HybridTransformerGRU(
        n_features=X.shape[2],
        d_model=config.model.d_model,
        nhead=config.model.nhead,
        num_layers=config.model.num_layers,
        dropout=config.model.dropout
    )

    # Callbacks
    callbacks = [
        EarlyStopping(patience=config.training.early_stopping_patience),
        ModelCheckpoint("results/models/best_model.pth")
    ]

    trainer = Trainer(model, train_loader, val_loader, config.training, device, callbacks)

    if mode in ["train", "full"]:
        train_losses, val_losses = trainer.fit()
        plot_training_history(train_losses, val_losses, "results/plots/training_history.png")

    if mode in ["backtest", "full"]:
        # Load best checkpoint
        model.load_state_dict(torch.load("results/models/best_model.pth", map_location=device))
        model.eval()
        # Predict on test set
        y_pred_list = []
        with torch.no_grad():
            for X_batch, _ in test_loader:
                X_batch = X_batch.to(device)
                y_pred_list.append(model(X_batch).cpu().numpy())
        y_pred = np.concatenate(y_pred_list)
        # Backtest
        run_backtest(y_pred, y_test, config.backtest)
        plot_predictions(y_test, y_pred, "results/plots/predictions.png")

    print("Pipeline complete.")


if __name__ == "__main__":
    cli()