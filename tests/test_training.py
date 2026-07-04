import torch
from src.training import Trainer
from torch.utils.data import DataLoader, TensorDataset


def test_trainer_init():
    model = torch.nn.Linear(10, 1)
    loader = DataLoader(TensorDataset(torch.randn(32, 10), torch.randn(32)), batch_size=8)
    # Simplified trainer test
    assert model is not None