import torch
from src.models import HybridTransformerGRU


def test_model_forward():
    model = HybridTransformerGRU(n_features=10, d_model=32)
    x = torch.randn(4, 60, 10)  # batch, seq, features
    out = model(x)
    assert out.shape == (4,)