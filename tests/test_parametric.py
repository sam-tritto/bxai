import numpy as np
import pandas as pd
import pytest

from bxai.parametric.shrinkage_pip import ShrinkagePIP
from bxai.parametric.bart_importance import BARTImportance


def test_shrinkage_pip():
    # Make small dummy classification data
    np.random.seed(42)
    X = np.random.randn(20, 3)
    y = np.random.randint(0, 2, size=20)
    
    # Run with small sample/tune counts for fast unit test
    selector = ShrinkagePIP(
        model_type="logistic",
        prior="horseshoe",
        n_samples=50,
        tune=50,
        chains=1,
        progressbar=False,
        random_state=42
    )
    
    selector.fit(X, y)
    
    assert hasattr(selector, "pip_")
    assert len(selector.pip_) == 3
    assert hasattr(selector, "confirmed_")
    
    df_sum = selector.summary()
    assert isinstance(df_sum, pd.DataFrame)
    assert len(df_sum) == 3
    assert "pip" in df_sum.columns
    assert "interval_type" in df_sum.columns
    assert (df_sum["interval_type"] == "hdi").all()


def test_bart_importance():
    # Make small dummy regression data
    np.random.seed(42)
    X = np.random.randn(20, 3)
    y = X[:, 0] + np.random.randn(20)
    
    selector = BARTImportance(
        n_trees=3,
        n_samples=50,
        tune=50,
        chains=1,
        progressbar=False,
        random_state=42
    )
    
    selector.fit(X, y)
    
    assert hasattr(selector, "vif_mean_")
    assert len(selector.vif_mean_) == 3
    assert hasattr(selector, "confirmed_")
    
    df_sum = selector.summary()
    assert isinstance(df_sum, pd.DataFrame)
    assert len(df_sum) == 3
    assert "vif_mean" in df_sum.columns
    assert "interval_type" in df_sum.columns
    assert (df_sum["interval_type"] == "hdi").all()
