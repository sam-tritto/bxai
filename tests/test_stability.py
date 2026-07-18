import numpy as np
import pytest
from sklearn.datasets import make_classification

from bxai.selection.boruta_shap import BayesianBorutaSHAP
from bxai.selection.stability import (
    CVStabilityResult,
    calculate_jaccard_stability,
    calculate_nogueira_stability,
    cross_val_feature_stability,
)


def test_calculate_nogueira_stability_mathematical_correctness():
    # 1. Perfect stability (all ones or all zeros)
    all_ones = np.ones((5, 10))
    all_zeros = np.zeros((5, 10))
    assert calculate_nogueira_stability(all_ones) == 1.0
    assert calculate_nogueira_stability(all_zeros) == 1.0

    # 2. Perfect stability: mix of columns with zero variance
    mix_zeros_ones = np.zeros((5, 4))
    mix_zeros_ones[:, 0] = 1.0
    mix_zeros_ones[:, 2] = 1.0
    assert calculate_nogueira_stability(mix_zeros_ones) == 1.0

    # 3. Known manually calculated matrix
    # Z = [[1, 0, 0],
    #      [1, 1, 0],
    #      [1, 0, 1]]
    # M = 3 runs, d = 3 features
    # p = [1.0, 1/3, 1/3]
    # k_bar = 5/3
    # s_1^2 = 0
    # s_2^2 = 1/3
    # s_3^2 = 1/3
    # sum(s_j^2) = 2/3, avg(s_j^2) = 2/9
    # k_bar/d * (1 - k_bar/d) = 5/9 * 4/9 = 20/81
    # stability = 1 - (2/9) / (20/81) = 1 - 0.9 = 0.1
    Z = np.array([[1, 0, 0], [1, 1, 0], [1, 0, 1]])
    assert pytest.approx(calculate_nogueira_stability(Z)) == 0.1

    # 4. Check error handling
    with pytest.raises(ValueError, match="must be a 2D array"):
        calculate_nogueira_stability(np.array([1, 0, 1]))
    with pytest.raises(ValueError, match="At least 2 runs"):
        calculate_nogueira_stability(np.array([[1, 0, 1]]))


def test_calculate_jaccard_stability_mathematical_correctness():
    # Z = [[1, 0, 0],
    #      [1, 1, 0],
    #      [1, 0, 1]]
    # Pairwise Jaccards:
    # 0 vs 1: intersection [1,0,0] (1), union [1,1,0] (2) => 0.5
    # 0 vs 2: intersection [1,0,0] (1), union [1,0,1] (2) => 0.5
    # 1 vs 2: intersection [1,0,0] (1), union [1,1,1] (3) => 1/3
    # average: (0.5 + 0.5 + 1/3) / 3 = 4/9
    Z = np.array([[1, 0, 0], [1, 1, 0], [1, 0, 1]])
    assert pytest.approx(calculate_jaccard_stability(Z)) == 4 / 9

    # Check zero features Jaccard edge case
    all_zeros = np.zeros((3, 5))
    assert calculate_jaccard_stability(all_zeros) == 1.0


def test_cross_val_feature_stability_validation():
    # Estimator must implement get_support
    class DummyEstimator:
        pass

    X = np.random.randn(10, 5)
    y = np.random.randint(0, 2, size=10)

    with pytest.raises(TypeError, match="must be a feature selector"):
        cross_val_feature_stability(DummyEstimator(), X, y)


def test_cross_val_feature_stability_integration():
    X, y = make_classification(
        n_samples=60, n_features=6, n_informative=3, n_redundant=0, random_state=42
    )

    # Use BayesianBorutaSHAP as our estimator
    selector = BayesianBorutaSHAP(mode="discrete", max_iter=3, random_state=42)

    # Run with 3-fold CV
    result = cross_val_feature_stability(selector, X, y, cv=3)

    assert isinstance(result, CVStabilityResult)
    assert isinstance(result.stability_index, float)
    assert -1.0 <= result.stability_index <= 1.0
    assert 0.0 <= result.jaccard_stability_index <= 1.0
    assert result.support_matrix.shape == (3, 6)
    assert result.selection_frequencies.shape == (6,)
    assert 0.0 <= result.mean_selected_features <= 6.0
    assert result.feature_names is None

    # Test with Pandas DataFrame input to verify feature_names
    import pandas as pd

    df_X = pd.DataFrame(X, columns=[f"col_{i}" for i in range(6)])
    result_df = cross_val_feature_stability(selector, df_X, y, cv=3)
    assert result_df.feature_names == [f"col_{i}" for i in range(6)]
