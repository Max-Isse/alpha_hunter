import torch
import torch.nn as nn
from typing import List


class ModelEnsemble(nn.Module):
    """Simple averaging ensemble for multiple trained models."""
    def __init__(self, models: List[nn.Module]):
        super().__init__()
        self.models = nn.ModuleList(models)

    def forward(self, x):
        outputs = [model(x) for model in self.models]
        return torch.stack(outputs).mean(dim=0)