import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import List, Optional
from src.utils.logging import logger
from src.training.optimizer import create_optimizer
from src.training.callbacks import Callback


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config,
        device: torch.device,
        callbacks: Optional[List[Callback]] = None,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.callbacks = callbacks or []
        self.stop_training = False

        self.criterion = nn.MSELoss()
        self.optimizer = create_optimizer(model, config)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=config.scheduler_factor, patience=config.scheduler_patience
        )
        self.train_losses = []
        self.val_losses = []

    def train_one_epoch(self):
        self.model.train()
        total_loss = 0
        for X, y in self.train_loader:
            X, y = X.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            pred = self.model(X)
            loss = self.criterion(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_norm)
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(self.train_loader)

    def validate(self):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for X, y in self.val_loader:
                X, y = X.to(self.device), y.to(self.device)
                pred = self.model(X)
                loss = self.criterion(pred, y)
                total_loss += loss.item()
        return total_loss / len(self.val_loader)

    def fit(self):
        for epoch in range(self.config.epochs):
            train_loss = self.train_one_epoch()
            val_loss = self.validate()
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.scheduler.step(val_loss)

            logger.debug(f"Epoch {epoch+1}/{self.config.epochs} | Train: {train_loss:.6f} | Val: {val_loss:.6f}")

            for cb in self.callbacks:
                cb.on_epoch_end(self)

            if self.stop_training:
                break
        return self.train_losses, self.val_losses