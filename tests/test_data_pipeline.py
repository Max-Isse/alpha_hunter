import pytest
import numpy as np
import pandas as pd
from src.data.pipeline import FeaturePipeline


def test_pipeline_shape():
    df = pd.DataFrame({
        "Open": np.random.randn(100),
        "High": np.random.randn(100),
        "Low": np.random.randn(100),
        "Close": np.random.randn(100),
        "Volume": np.random.randint(100, 1000, 100)
    })
    pipe = FeaturePipeline(lookback=10, horizon=3)
    X, y = pipe.fit_transform(df)
    assert X.shape[0] == y.shape[0]
    assert X.shape[1] == 10