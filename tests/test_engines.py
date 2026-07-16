import numpy as np
import pytest
from bxai._engines.beta_binomial import BetaBinomialTracker
from bxai._engines.normal_ig import NormalIGTracker
from bxai._utils.types import FeatureStatus


def test_beta_binomial_tracker():
    tracker = BetaBinomialTracker(n_features=3, prior_alpha=1.0, prior_beta=1.0)
    assert np.all(tracker.alpha == 1.0)
    assert np.all(tracker.beta == 1.0)

    # First update
    tracker.update(np.array([1, 0, 1]))
    # alpha: [2, 1, 2], beta: [1, 2, 1]
    assert np.all(tracker.alpha == [2.0, 1.0, 2.0])
    assert np.all(tracker.beta == [1.0, 2.0, 1.0])

    # Second update on subset indices [0, 2]
    tracker.update(np.array([1, 0]), indices=np.array([0, 2]))
    # alpha: [3, 1, 2], beta: [1, 2, 2]
    assert np.all(tracker.alpha == [3.0, 1.0, 2.0])
    assert np.all(tracker.beta == [1.0, 2.0, 2.0])

    # Exceedance probabilities (theta > 0.5)
    # Feature 0: Beta(3, 1) -> mean = 0.75, P(theta > 0.5) = 1 - 0.5**3 = 0.875
    probs = tracker.exceedance_probability(0.5)
    assert probs.shape == (3,)
    assert 0.0 <= probs[0] <= 1.0

    # Decision checks
    decisions = tracker.decide(confirm_threshold=0.85, reject_threshold=0.15)
    assert decisions[0] == FeatureStatus.CONFIRMED

    # Credible interval
    lower, upper = tracker.credible_interval(0.95)
    assert len(lower) == 3
    assert len(upper) == 3
    assert np.all(lower < upper)


def test_normal_ig_tracker():
    tracker = NormalIGTracker(n_features=2, prior_mu=0.0, prior_nu=1e-2, prior_alpha=1e-2, prior_beta=1e-2)
    assert np.all(tracker.mu == 0.0)

    # Update with some values
    # Feature 0: values [1.0, 2.0, 3.0] (mean 2.0)
    # Feature 1: values [-1.0, -2.0, -3.0] (mean -2.0)
    data = np.array([
        [1.0, -1.0],
        [2.0, -2.0],
        [3.0, -3.0]
    ])
    tracker.update(data)
    
    # Check that posteriors have updated appropriately
    assert np.all(tracker.nu > 1e-2)
    assert tracker.mu[0] > 0.0
    assert tracker.mu[1] < 0.0

    # Test decide
    decisions = tracker.decide(credible_mass=0.90, threshold=0.0)
    assert decisions.shape == (2,)

    # Test update with indices
    tracker.update(np.array([2.5]), indices=np.array([0]))
