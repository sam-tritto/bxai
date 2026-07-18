"""Tests for input validation on hyperparameters across all estimators and engines."""

import warnings

import numpy as np
import pytest

from bxai._engines.beta_binomial import BetaBinomialTracker
from bxai._engines.normal_ig import NormalIGTracker
from bxai.parametric.bart_importance import BARTImportance
from bxai.parametric.shrinkage_pip import ShrinkagePIP
from bxai.selection.boruta_shap import BayesianBorutaSHAP
from bxai.selection.permutation import BayesianPermutation

# Shared fixtures (tiny_Xy, tiny_rf) are provided by tests/conftest.py.


# ---------------------------------------------------------------------------
# BetaBinomialTracker — prior constraints
# ---------------------------------------------------------------------------


class TestBetaBinomialTrackerInit:
    def test_prior_alpha_zero_raises(self):
        with pytest.raises(ValueError, match="prior_alpha must be > 0"):
            BetaBinomialTracker(n_features=3, prior_alpha=0.0)

    def test_prior_alpha_negative_raises(self):
        with pytest.raises(ValueError, match="prior_alpha must be > 0"):
            BetaBinomialTracker(n_features=3, prior_alpha=-1.0)

    def test_prior_beta_zero_raises(self):
        with pytest.raises(ValueError, match="prior_beta must be > 0"):
            BetaBinomialTracker(n_features=3, prior_beta=0.0)

    def test_prior_beta_negative_raises(self):
        with pytest.raises(ValueError, match="prior_beta must be > 0"):
            BetaBinomialTracker(n_features=3, prior_beta=-0.5)

    def test_valid_priors_ok(self):
        tracker = BetaBinomialTracker(n_features=2, prior_alpha=0.5, prior_beta=2.0)
        assert tracker.n_features == 2


class TestBetaBinomialTrackerCredibleInterval:
    def setup_method(self):
        self.tracker = BetaBinomialTracker(n_features=2)

    def test_credible_mass_zero_raises(self):
        with pytest.raises(ValueError, match="credible_mass must be in"):
            self.tracker.credible_interval(credible_mass=0.0)

    def test_credible_mass_one_raises(self):
        with pytest.raises(ValueError, match="credible_mass must be in"):
            self.tracker.credible_interval(credible_mass=1.0)

    def test_credible_mass_negative_raises(self):
        with pytest.raises(ValueError, match="credible_mass must be in"):
            self.tracker.credible_interval(credible_mass=-0.1)

    def test_credible_mass_above_one_raises(self):
        with pytest.raises(ValueError, match="credible_mass must be in"):
            self.tracker.credible_interval(credible_mass=1.5)

    def test_valid_credible_mass_ok(self):
        lower, upper = self.tracker.credible_interval(credible_mass=0.90)
        assert lower.shape == (2,)


class TestBetaBinomialTrackerDecide:
    def setup_method(self):
        self.tracker = BetaBinomialTracker(n_features=3)

    def test_confirm_below_reject_raises(self):
        with pytest.raises(
            ValueError, match="confirm_threshold.*must be strictly greater"
        ):
            self.tracker.decide(confirm_threshold=0.3, reject_threshold=0.7)

    def test_confirm_equal_reject_raises(self):
        with pytest.raises(
            ValueError, match="confirm_threshold.*must be strictly greater"
        ):
            self.tracker.decide(confirm_threshold=0.5, reject_threshold=0.5)

    def test_confirm_threshold_zero_raises(self):
        with pytest.raises(ValueError, match="confirm_threshold must be in"):
            self.tracker.decide(confirm_threshold=0.0, reject_threshold=0.05)

    def test_confirm_threshold_one_raises(self):
        with pytest.raises(ValueError, match="confirm_threshold must be in"):
            self.tracker.decide(confirm_threshold=1.0, reject_threshold=0.05)

    def test_reject_threshold_zero_raises(self):
        with pytest.raises(ValueError, match="reject_threshold must be in"):
            self.tracker.decide(confirm_threshold=0.95, reject_threshold=0.0)

    def test_reject_threshold_one_raises(self):
        with pytest.raises(ValueError, match="reject_threshold must be in"):
            self.tracker.decide(confirm_threshold=0.95, reject_threshold=1.0)

    def test_valid_thresholds_ok(self):
        decisions = self.tracker.decide(confirm_threshold=0.95, reject_threshold=0.05)
        assert decisions.shape == (3,)


# ---------------------------------------------------------------------------
# NormalIGTracker — prior constraints
# ---------------------------------------------------------------------------


class TestNormalIGTrackerInit:
    def test_prior_nu_zero_raises(self):
        with pytest.raises(ValueError, match="prior_nu must be > 0"):
            NormalIGTracker(n_features=2, prior_nu=0.0)

    def test_prior_nu_negative_raises(self):
        with pytest.raises(ValueError, match="prior_nu must be > 0"):
            NormalIGTracker(n_features=2, prior_nu=-1e-4)

    def test_prior_alpha_zero_raises(self):
        with pytest.raises(ValueError, match="prior_alpha must be > 0"):
            NormalIGTracker(n_features=2, prior_alpha=0.0)

    def test_prior_alpha_negative_raises(self):
        with pytest.raises(ValueError, match="prior_alpha must be > 0"):
            NormalIGTracker(n_features=2, prior_alpha=-1.0)

    def test_prior_beta_zero_raises(self):
        with pytest.raises(ValueError, match="prior_beta must be > 0"):
            NormalIGTracker(n_features=2, prior_beta=0.0)

    def test_prior_beta_negative_raises(self):
        with pytest.raises(ValueError, match="prior_beta must be > 0"):
            NormalIGTracker(n_features=2, prior_beta=-1.0)

    def test_valid_priors_ok(self):
        tracker = NormalIGTracker(
            n_features=3, prior_nu=1e-2, prior_alpha=1e-2, prior_beta=1e-2
        )
        assert tracker.n_features == 3


class TestNormalIGTrackerCredibleInterval:
    def setup_method(self):
        self.tracker = NormalIGTracker(n_features=2)
        data = np.array([[1.0, -1.0], [2.0, -2.0], [3.0, -3.0]])
        self.tracker.update(data)

    def test_credible_mass_zero_raises(self):
        with pytest.raises(ValueError, match="credible_mass must be in"):
            self.tracker.credible_interval(credible_mass=0.0)

    def test_credible_mass_one_raises(self):
        with pytest.raises(ValueError, match="credible_mass must be in"):
            self.tracker.credible_interval(credible_mass=1.0)

    def test_credible_mass_negative_raises(self):
        with pytest.raises(ValueError, match="credible_mass must be in"):
            self.tracker.credible_interval(credible_mass=-0.5)

    def test_valid_credible_mass_ok(self):
        lower, upper = self.tracker.credible_interval(credible_mass=0.95)
        assert lower.shape == (2,)


class TestNormalIGTrackerDecide:
    def setup_method(self):
        self.tracker = NormalIGTracker(n_features=2)
        data = np.array([[1.0, -1.0], [2.0, -2.0], [3.0, -3.0]])
        self.tracker.update(data)

    def test_credible_mass_zero_raises(self):
        with pytest.raises(ValueError, match="credible_mass must be in"):
            self.tracker.decide(credible_mass=0.0)

    def test_credible_mass_one_raises(self):
        with pytest.raises(ValueError, match="credible_mass must be in"):
            self.tracker.decide(credible_mass=1.0)

    def test_valid_decide_ok(self):
        decisions = self.tracker.decide(credible_mass=0.90)
        assert decisions.shape == (2,)


# ---------------------------------------------------------------------------
# BayesianBorutaSHAP — estimator-level guards
# ---------------------------------------------------------------------------


class TestBayesianBorutaSHAPValidation:
    def test_credible_mass_zero_raises(self, tiny_Xy):
        X, y = tiny_Xy
        selector = BayesianBorutaSHAP(credible_mass=0.0, max_iter=1)
        with pytest.raises(ValueError, match="credible_mass must be in"):
            selector.fit(X, y)

    def test_credible_mass_one_raises(self, tiny_Xy):
        X, y = tiny_Xy
        selector = BayesianBorutaSHAP(credible_mass=1.0, max_iter=1)
        with pytest.raises(ValueError, match="credible_mass must be in"):
            selector.fit(X, y)

    def test_confirm_below_reject_raises(self, tiny_Xy):
        X, y = tiny_Xy
        selector = BayesianBorutaSHAP(
            confirm_threshold=0.2, reject_threshold=0.8, max_iter=1
        )
        with pytest.raises(
            ValueError, match="confirm_threshold.*must be strictly greater"
        ):
            selector.fit(X, y)

    def test_confirm_equal_reject_raises(self, tiny_Xy):
        X, y = tiny_Xy
        selector = BayesianBorutaSHAP(
            confirm_threshold=0.5, reject_threshold=0.5, max_iter=1
        )
        with pytest.raises(
            ValueError, match="confirm_threshold.*must be strictly greater"
        ):
            selector.fit(X, y)

    def test_prior_alpha_nonpositive_raises(self, tiny_Xy):
        X, y = tiny_Xy
        selector = BayesianBorutaSHAP(prior_alpha=0.0, max_iter=1)
        with pytest.raises(ValueError, match="prior_alpha must be > 0"):
            selector.fit(X, y)

    def test_prior_beta_nonpositive_raises(self, tiny_Xy):
        X, y = tiny_Xy
        selector = BayesianBorutaSHAP(prior_beta=-1.0, max_iter=1)
        with pytest.raises(ValueError, match="prior_beta must be > 0"):
            selector.fit(X, y)

    def test_prior_alpha_continuous_nonpositive_raises(self, tiny_Xy):
        X, y = tiny_Xy
        selector = BayesianBorutaSHAP(
            prior_alpha_continuous=0.0, mode="continuous", max_iter=1
        )
        with pytest.raises(ValueError, match="prior_alpha_continuous must be > 0"):
            selector.fit(X, y)

    def test_prior_beta_continuous_nonpositive_raises(self, tiny_Xy):
        X, y = tiny_Xy
        selector = BayesianBorutaSHAP(
            prior_beta_continuous=-0.5, mode="continuous", max_iter=1
        )
        with pytest.raises(ValueError, match="prior_beta_continuous must be > 0"):
            selector.fit(X, y)

    def test_prior_nu_zero_raises(self, tiny_Xy):
        X, y = tiny_Xy
        selector = BayesianBorutaSHAP(prior_nu=0.0, mode="continuous", max_iter=1)
        with pytest.raises(ValueError, match="prior_nu must be > 0"):
            selector.fit(X, y)

    def test_prior_nu_negative_raises(self, tiny_Xy):
        X, y = tiny_Xy
        selector = BayesianBorutaSHAP(prior_nu=-1e-4, mode="continuous", max_iter=1)
        with pytest.raises(ValueError, match="prior_nu must be > 0"):
            selector.fit(X, y)


# ---------------------------------------------------------------------------
# BayesianBorutaSHAP — cross-mode unused-parameter warnings
# ---------------------------------------------------------------------------


class TestBayesianBorutaSHAPCrossModeWarnings:
    """Verify that mode-irrelevant prior parameters trigger UserWarnings."""

    # -- continuous mode warned about discrete priors ----------------------

    def test_continuous_mode_warns_on_prior_alpha(self, tiny_Xy):
        """prior_alpha has no effect in continuous mode → UserWarning."""
        X, y = tiny_Xy
        selector = BayesianBorutaSHAP(mode="continuous", prior_alpha=2.0, max_iter=1)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            selector.fit(X, y)
        messages = [
            str(w.message) for w in caught if issubclass(w.category, UserWarning)
        ]
        assert any("prior_alpha" in m for m in messages), (
            f"Expected a UserWarning mentioning 'prior_alpha'; got: {messages}"
        )

    def test_continuous_mode_warns_on_prior_beta(self, tiny_Xy):
        """prior_beta has no effect in continuous mode → UserWarning."""
        X, y = tiny_Xy
        selector = BayesianBorutaSHAP(mode="continuous", prior_beta=3.0, max_iter=1)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            selector.fit(X, y)
        messages = [
            str(w.message) for w in caught if issubclass(w.category, UserWarning)
        ]
        assert any("prior_beta" in m for m in messages), (
            f"Expected a UserWarning mentioning 'prior_beta'; got: {messages}"
        )

    def test_continuous_mode_warns_on_multiple_discrete_priors(self, tiny_Xy):
        """Both prior_alpha and prior_beta non-default → single warning listing both."""
        X, y = tiny_Xy
        selector = BayesianBorutaSHAP(
            mode="continuous", prior_alpha=2.0, prior_beta=0.5, max_iter=1
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            selector.fit(X, y)
        messages = [
            str(w.message) for w in caught if issubclass(w.category, UserWarning)
        ]
        assert any("prior_alpha" in m and "prior_beta" in m for m in messages), (
            f"Expected a single UserWarning mentioning both priors; got: {messages}"
        )

    # -- discrete mode warned about NIG priors -----------------------------

    def test_discrete_mode_warns_on_prior_nu(self, tiny_Xy):
        """prior_nu has no effect in discrete mode → UserWarning."""
        X, y = tiny_Xy
        selector = BayesianBorutaSHAP(mode="discrete", prior_nu=1.0, max_iter=1)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            selector.fit(X, y)
        messages = [
            str(w.message) for w in caught if issubclass(w.category, UserWarning)
        ]
        assert any("prior_nu" in m for m in messages), (
            f"Expected a UserWarning mentioning 'prior_nu'; got: {messages}"
        )

    def test_discrete_mode_warns_on_prior_alpha_continuous(self, tiny_Xy):
        """prior_alpha_continuous has no effect in discrete mode → UserWarning."""
        X, y = tiny_Xy
        selector = BayesianBorutaSHAP(
            mode="discrete", prior_alpha_continuous=1.0, max_iter=1
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            selector.fit(X, y)
        messages = [
            str(w.message) for w in caught if issubclass(w.category, UserWarning)
        ]
        assert any("prior_alpha_continuous" in m for m in messages), (
            f"Expected a UserWarning mentioning 'prior_alpha_continuous'; got: {messages}"
        )

    # -- all defaults → no cross-mode warning ------------------------------

    def test_no_warning_with_defaults_discrete(self, tiny_Xy):
        """Default parameters in discrete mode must not produce any UserWarning."""
        X, y = tiny_Xy
        selector = BayesianBorutaSHAP(mode="discrete", max_iter=1)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            selector.fit(X, y)
        cross_mode = [
            w
            for w in caught
            if issubclass(w.category, UserWarning)
            and ("NIG" in str(w.message) or "Beta-Binomial" in str(w.message))
        ]
        assert not cross_mode, (
            f"Unexpected cross-mode UserWarning with default parameters: {cross_mode}"
        )

    def test_no_warning_with_defaults_continuous(self, tiny_Xy):
        """Default parameters in continuous mode must not produce any UserWarning."""
        X, y = tiny_Xy
        selector = BayesianBorutaSHAP(mode="continuous", max_iter=1)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            selector.fit(X, y)
        cross_mode = [
            w
            for w in caught
            if issubclass(w.category, UserWarning)
            and ("NIG" in str(w.message) or "Beta-Binomial" in str(w.message))
        ]
        assert not cross_mode, (
            f"Unexpected cross-mode UserWarning with default parameters: {cross_mode}"
        )


# ---------------------------------------------------------------------------
# BayesianPermutation — estimator-level guards
# ---------------------------------------------------------------------------


class TestBayesianPermutationValidation:
    def test_n_repeats_one_raises(self, tiny_rf, tiny_Xy):
        X, y = tiny_Xy
        selector = BayesianPermutation(model=tiny_rf, scoring="accuracy", n_repeats=1)
        with pytest.raises(ValueError, match="n_repeats must be >= 2"):
            selector.fit(X, y)

    def test_n_repeats_zero_raises(self, tiny_rf, tiny_Xy):
        X, y = tiny_Xy
        selector = BayesianPermutation(model=tiny_rf, scoring="accuracy", n_repeats=0)
        with pytest.raises(ValueError, match="n_repeats must be >= 2"):
            selector.fit(X, y)

    def test_credible_mass_zero_raises(self, tiny_rf, tiny_Xy):
        X, y = tiny_Xy
        selector = BayesianPermutation(
            model=tiny_rf, scoring="accuracy", credible_mass=0.0
        )
        with pytest.raises(ValueError, match="credible_mass must be in"):
            selector.fit(X, y)

    def test_credible_mass_one_raises(self, tiny_rf, tiny_Xy):
        X, y = tiny_Xy
        selector = BayesianPermutation(
            model=tiny_rf, scoring="accuracy", credible_mass=1.0
        )
        with pytest.raises(ValueError, match="credible_mass must be in"):
            selector.fit(X, y)

    def test_prior_alpha_nonpositive_raises(self, tiny_rf, tiny_Xy):
        X, y = tiny_Xy
        selector = BayesianPermutation(
            model=tiny_rf, scoring="accuracy", prior_alpha=0.0
        )
        with pytest.raises(ValueError, match="prior_alpha must be > 0"):
            selector.fit(X, y)

    def test_prior_beta_nonpositive_raises(self, tiny_rf, tiny_Xy):
        X, y = tiny_Xy
        selector = BayesianPermutation(
            model=tiny_rf, scoring="accuracy", prior_beta=-1.0
        )
        with pytest.raises(ValueError, match="prior_beta must be > 0"):
            selector.fit(X, y)

    def test_prior_nu_zero_raises(self, tiny_rf, tiny_Xy):
        X, y = tiny_Xy
        selector = BayesianPermutation(model=tiny_rf, scoring="accuracy", prior_nu=0.0)
        with pytest.raises(ValueError, match="prior_nu must be > 0"):
            selector.fit(X, y)

    def test_prior_nu_negative_raises(self, tiny_rf, tiny_Xy):
        X, y = tiny_Xy
        selector = BayesianPermutation(
            model=tiny_rf, scoring="accuracy", prior_nu=-1e-4
        )
        with pytest.raises(ValueError, match="prior_nu must be > 0"):
            selector.fit(X, y)


# ---------------------------------------------------------------------------
# ShrinkagePIP — estimator-level guards
# ---------------------------------------------------------------------------


class TestShrinkagePIPValidation:
    def _make_data(self):
        return np.random.randn(10, 3), np.random.randn(10)

    def test_kappa_threshold_zero_raises(self):
        X, y = self._make_data()
        selector = ShrinkagePIP(kappa_threshold=0.0)
        with pytest.raises(ValueError, match="kappa_threshold must be in"):
            selector.fit(X, y)

    def test_kappa_threshold_one_raises(self):
        X, y = self._make_data()
        selector = ShrinkagePIP(kappa_threshold=1.0)
        with pytest.raises(ValueError, match="kappa_threshold must be in"):
            selector.fit(X, y)

    def test_kappa_threshold_above_one_raises(self):
        X, y = self._make_data()
        selector = ShrinkagePIP(kappa_threshold=1.5)
        with pytest.raises(ValueError, match="kappa_threshold must be in"):
            selector.fit(X, y)

    def test_pip_threshold_zero_raises(self):
        X, y = self._make_data()
        selector = ShrinkagePIP(pip_threshold=0.0)
        with pytest.raises(ValueError, match="pip_threshold must be in"):
            selector.fit(X, y)

    def test_pip_threshold_one_raises(self):
        X, y = self._make_data()
        selector = ShrinkagePIP(pip_threshold=1.0)
        with pytest.raises(ValueError, match="pip_threshold must be in"):
            selector.fit(X, y)

    def test_cores_invalid_raises(self):
        X, y = self._make_data()
        selector = ShrinkagePIP(cores=0)
        with pytest.raises(ValueError, match="cores must be"):
            selector.fit(X, y)

        selector = ShrinkagePIP(cores="invalid")
        with pytest.raises(ValueError, match="cores must be"):
            selector.fit(X, y)


# ---------------------------------------------------------------------------
# BARTImportance — estimator-level guards
# ---------------------------------------------------------------------------


class TestBARTImportanceValidation:
    def _make_data(self):
        return np.random.randn(10, 3), np.random.randn(10)

    def test_credible_mass_zero_raises(self):
        X, y = self._make_data()
        selector = BARTImportance(credible_mass=0.0)
        with pytest.raises(ValueError, match="credible_mass must be in"):
            selector.fit(X, y)

    def test_credible_mass_one_raises(self):
        X, y = self._make_data()
        selector = BARTImportance(credible_mass=1.0)
        with pytest.raises(ValueError, match="credible_mass must be in"):
            selector.fit(X, y)

    def test_credible_mass_negative_raises(self):
        X, y = self._make_data()
        selector = BARTImportance(credible_mass=-0.5)
        with pytest.raises(ValueError, match="credible_mass must be in"):
            selector.fit(X, y)

    def test_model_type_invalid_raises(self):
        X, y = self._make_data()
        selector = BARTImportance(model_type="invalid")
        with pytest.raises(ValueError, match="model_type must be either"):
            selector.fit(X, y)

    def test_model_type_classification_with_continuous_y_raises(self):
        X, y = self._make_data()
        selector = BARTImportance(model_type="classification")
        with pytest.raises(ValueError, match="requires a binary target"):
            selector.fit(X, y)

    def test_cores_invalid_raises(self):
        X, y = self._make_data()
        selector = BARTImportance(cores=0)
        with pytest.raises(ValueError, match="cores must be"):
            selector.fit(X, y)

        selector = BARTImportance(cores="invalid")
        with pytest.raises(ValueError, match="cores must be"):
            selector.fit(X, y)
