import torch.optim as optim
from src.utils.config import TrainingConfig


def create_optimizer(model, config: TrainingConfig):
    if config.optimizer.lower() == "adamw":
        return optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    elif config.optimizer.lower() == "sgd":
        return optim.SGD(model.parameters(), lr=config.lr, momentum=0.9, weight_decay=config.weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {config.optimizer}")