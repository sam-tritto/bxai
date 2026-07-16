import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

from bxai.selection.boruta_shap import BayesianBorutaSHAP
from bxai.selection.permutation import BayesianPermutation
from bxai._utils.types import FeatureStatus


def test_bayesian_boruta_shap_discrete():
    # Make small dataset
    X, y = make_classification(n_samples=60, n_features=6, n_informative=2, random_state=42)
    
    # We will fit BorutaSHAP using discrete mode
    # For speed, use very few iterations
    selector = BayesianBorutaSHAP(
        mode="discrete",
        max_iter=3,
        random_state=42
    )
    
    selector.fit(X, y)
    
    # Check attributes
    assert hasattr(selector, "confirmed_")
    assert hasattr(selector, "rejected_")
    assert hasattr(selector, "tentative_")
    assert isinstance(selector.confirmed_, list)
    
    # Summary
    df_sum = selector.summary()
    assert isinstance(df_sum, pd.DataFrame)
    assert len(df_sum) == 6
    assert "status" in df_sum.columns
    assert "mean" in df_sum.columns


def test_bayesian_boruta_shap_continuous():
    X, y = make_classification(n_samples=50, n_features=5, n_informative=2, random_state=42)
    
    selector = BayesianBorutaSHAP(
        mode="continuous",
        max_iter=3,
        random_state=42
    )
    selector.fit(X, y)
    
    df_sum = selector.summary()
    assert "mu" in df_sum.columns


def test_bayesian_permutation():
    X, y = make_classification(n_samples=50, n_features=4, n_informative=2, random_state=42)
    
    # Pre-train a simple random forest model
    model = RandomForestClassifier(n_estimators=5, random_state=42)
    model.fit(X, y)
    
    # Fit permutation selector
    selector = BayesianPermutation(
        model=model,
        scoring="accuracy",
        n_repeats=5,
        random_state=42
    )
    selector.fit(X, y)
    
    assert hasattr(selector, "confirmed_")
    assert hasattr(selector, "rejected_")
    
    df_sum = selector.summary()
    assert isinstance(df_sum, pd.DataFrame)
    assert len(df_sum) == 4
    assert "hdi_lower" in df_sum.columns
