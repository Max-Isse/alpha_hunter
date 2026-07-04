import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Optional


def plot_training_history(train_losses: list, val_losses: list, save_path: Optional[str] = None):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.title("Training History")
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close()


def plot_predictions(y_true: np.ndarray, y_pred: np.ndarray, save_path: Optional[str] = None):
    plt.figure(figsize=(8, 8))
    plt.scatter(y_true, y_pred, alpha=0.5, s=10)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    plt.plot(lims, lims, "r--", alpha=0.7)
    plt.xlabel("Actual Return")
    plt.ylabel("Predicted Return")
    plt.title("Predictions vs. Actual")
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close()