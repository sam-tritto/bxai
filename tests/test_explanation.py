import numpy as np
import pandas as pd
import pytest
from bxai.explanation.baylime import BayLIME, BayLIMEExplanation


def test_baylime_explainer():
    # Make small dummy training data
    np.random.seed(42)
    training_data = np.random.randn(100, 4)
    
    # explainer
    explainer = BayLIME(
        training_data=training_data,
        feature_names=["f1", "f2", "f3", "f4"],
        num_samples=100,
        random_state=42
    )
    
    # instance to explain
    instance = np.array([0.5, -0.5, 1.0, -1.0])
    
    # Mock prediction function: simply sum the inputs
    def predict_fn(Z):
        return Z.sum(axis=1)
        
    explanation = explainer.explain_instance(
        instance=instance,
        predict_fn=predict_fn,
        label=0
    )
    
    assert isinstance(explanation, BayLIMEExplanation)
    assert len(explanation.coef_mean) == 4
    assert explanation.coef_cov.shape == (4, 4)
    
    # as_list
    l = explanation.as_list()
    assert len(l) == 4
    assert isinstance(l[0][0], str)
    assert isinstance(l[0][1], float)
    
    # as_dataframe
    df = explanation.as_dataframe()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 4
    assert "mean" in df.columns
    assert "std" in df.columns
    assert "hdi_lower" in df.columns
    assert "hdi_upper" in df.columns
    
    # credible intervals
    ci = explanation.credible_intervals()
    assert len(ci) == 4
    assert "f1" in ci
    assert len(ci["f1"]) == 2
