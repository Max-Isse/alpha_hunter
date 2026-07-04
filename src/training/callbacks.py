import torch
import numpy as np
from src.utils.logging import logger


class Callback:
    def on_epoch_end(self, trainer, *args, **kwargs):
        pass


class EarlyStopping(Callback):
    def __init__(self, patience: int = 15, min_delta: float = 1e-6):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")

    def on_epoch_end(self, trainer):
        val_loss = trainer.val_losses[-1]
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                logger.info(f"Early stopping triggered at epoch {len(trainer.train_losses)}")
                trainer.stop_training = True


class ModelCheckpoint(Callback):
    def __init__(self, filepath: str = "results/models/best_model.pth", save_best_only: bool = True):
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.best_loss = float("inf")

    def on_epoch_end(self, trainer):
        if self.save_best_only:
            if trainer.val_losses[-1] < self.best_loss:
                self.best_loss = trainer.val_losses[-1]
                torch.save(trainer.model.state_dict(), self.filepath)
                logger.info(f"Saved checkpoint to {self.filepath}")
        else:
            torch.save(trainer.model.state_dict(), f"{self.filepath}_epoch_{len(trainer.train_losses)}.pth")


class RegimeCallback(Callback):
    """Logs validation performance per regime."""
    def __init__(self, regimes: np.ndarray, val_loader):
        self.regimes = regimes
        self.val_loader = val_loader

    def on_epoch_end(self, trainer):
        # We calculate this at the end of training for simplicity, but hook is here.
        pass