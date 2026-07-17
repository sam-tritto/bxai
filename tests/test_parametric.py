"""Tests for ShrinkagePIP and BARTImportance.

Test organisation
-----------------
  Unit tests   — pure-Python helpers (_decode_vi, _resolve_pip_method,
                 _resolve_epsilon, VIF normalisation) that need no MCMC.
  Integration  — full MCMC fits; marked @pytest.mark.slow @pytest.mark.mcmc
                 so they can be skipped in fast CI via ``pytest --skip-slow``.

Each integration test uses the minimal settings that still exercise the full
statistical path:  n_samples=50, tune=50, chains=1.
"""
from __future__ import annotations

import base64
import struct

import numpy as np
import pandas as pd
import pytest

from bxai.parametric.bart_importance import BARTImportance, _decode_vi
from bxai.parametric.shrinkage_pip import ShrinkagePIP

# Convenience marker aliases
_slow = pytest.mark.slow
_mcmc = pytest.mark.mcmc


# ===========================================================================
# _decode_vi  (pure-Python unit tests — zero MCMC overhead)
# ===========================================================================

class TestDecodeVI:
    """Unit tests for the base64 VIF decoder used by BARTImportance."""

    @staticmethod
    def _encode_varint(n: int) -> bytes:
        """Encode a non-negative integer as a variable-length (LEB128-style) byte."""
        buf = bytearray()
        while True:
            byte = n & 0x7F
            n >>= 7
            if n:
                buf.append(byte | 0x80)
            else:
                buf.append(byte)
                break
        return bytes(buf)

    @staticmethod
    def _make_vi_string(counts: list[int]) -> str:
        """Build a valid base64-encoded VIF string from a list of split counts."""
        raw = b"".join(TestDecodeVI._encode_varint(c) for c in counts)
        return base64.b64encode(raw).decode()

    # -----------------------------------------------------------------------
    # Happy-path decoding
    # -----------------------------------------------------------------------

    def test_decode_single_zero(self):
        """A single feature with zero splits decodes to [0]."""
        s = self._make_vi_string([0])
        assert _decode_vi(s, 1) == [0]

    def test_decode_known_counts(self):
        """Arbitrary counts round-trip through encode → decode."""
        counts = [3, 0, 7, 1]
        s = self._make_vi_string(counts)
        assert _decode_vi(s, 4) == counts

    def test_decode_large_value(self):
        """Values > 127 require multi-byte varint; must decode correctly."""
        counts = [200, 1]
        s = self._make_vi_string(counts)
        assert _decode_vi(s, 2) == counts

    # -----------------------------------------------------------------------
    # Zero-padding when byte stream is short
    # -----------------------------------------------------------------------

    def test_short_stream_is_zero_padded(self):
        """A stream encoding 1 value is zero-padded to reach length=3."""
        s = self._make_vi_string([5])
        result = _decode_vi(s, 3)
        assert result[0] == 5
        assert result[1] == 0
        assert result[2] == 0
        assert len(result) == 3

    def test_length_always_matches_request(self):
        """Output length equals the requested length regardless of stream length."""
        s = self._make_vi_string([1, 2, 3, 4, 5])
        for n in (2, 3, 5, 7):
            assert len(_decode_vi(s, n)) == n

    # -----------------------------------------------------------------------
    # Error handling
    # -----------------------------------------------------------------------

    def test_invalid_base64_raises_value_error(self):
        """Corrupt base64 must raise ValueError (not silently return zeros)."""
        with pytest.raises(ValueError, match="failed to decode base64"):
            _decode_vi("!!!not_base64!!!", 3)

    def test_empty_string_raises_or_returns_zeros(self):
        """An empty payload string is valid base64 but decodes to all zeros."""
        s = base64.b64encode(b"").decode()
        result = _decode_vi(s, 4)
        assert result == [0, 0, 0, 0]


# ===========================================================================
# ShrinkagePIP — pure-logic helpers (no MCMC)
# ===========================================================================

class TestShrinkagePIPHelpers:
    """Tests for internal helper methods that do not require MCMC."""

    # -----------------------------------------------------------------------
    # _resolve_pip_method
    # -----------------------------------------------------------------------

    def test_auto_horseshoe_resolves_to_kappa(self):
        sel = ShrinkagePIP(prior="horseshoe", pip_method="auto")
        assert sel._resolve_pip_method() == "kappa"

    def test_auto_lasso_resolves_to_threshold(self):
        sel = ShrinkagePIP(prior="lasso", pip_method="auto")
        assert sel._resolve_pip_method() == "threshold"

    def test_explicit_kappa_returned_unchanged(self):
        sel = ShrinkagePIP(pip_method="kappa")
        assert sel._resolve_pip_method() == "kappa"

    def test_explicit_threshold_returned_unchanged(self):
        sel = ShrinkagePIP(pip_method="threshold")
        assert sel._resolve_pip_method() == "threshold"

    def test_invalid_pip_method_raises(self):
        sel = ShrinkagePIP(pip_method="nonsense")
        with pytest.raises(ValueError, match="pip_method must be"):
            sel._resolve_pip_method()

    # -----------------------------------------------------------------------
    # _resolve_epsilon
    # -----------------------------------------------------------------------

    def test_explicit_epsilon_returned_unchanged(self):
        sel = ShrinkagePIP(epsilon=0.25)
        y = np.random.default_rng(0).standard_normal(20)
        assert sel._resolve_epsilon(y) == pytest.approx(0.25)

    def test_auto_epsilon_linear_equals_std_over_ten(self):
        sel = ShrinkagePIP(model_type="linear", epsilon=None)
        y = np.random.default_rng(1).standard_normal(50) * 3.0
        expected = np.std(y) / 10.0
        assert sel._resolve_epsilon(y) == pytest.approx(expected)

    def test_auto_epsilon_logistic_is_point_one(self):
        sel = ShrinkagePIP(model_type="logistic", epsilon=None)
        y = np.array([0, 1, 0, 1, 1])
        assert sel._resolve_epsilon(y) == pytest.approx(0.1)

    # -----------------------------------------------------------------------
    # _validate_hyperparams
    # -----------------------------------------------------------------------

    def test_kappa_threshold_zero_raises(self):
        sel = ShrinkagePIP(kappa_threshold=0.0)
        with pytest.raises(ValueError, match="kappa_threshold"):
            sel._validate_hyperparams()

    def test_kappa_threshold_one_raises(self):
        sel = ShrinkagePIP(kappa_threshold=1.0)
        with pytest.raises(ValueError, match="kappa_threshold"):
            sel._validate_hyperparams()

    def test_pip_threshold_zero_raises(self):
        sel = ShrinkagePIP(pip_threshold=0.0)
        with pytest.raises(ValueError, match="pip_threshold"):
            sel._validate_hyperparams()

    def test_pip_threshold_one_raises(self):
        sel = ShrinkagePIP(pip_threshold=1.0)
        with pytest.raises(ValueError, match="pip_threshold"):
            sel._validate_hyperparams()

    def test_valid_hyperparams_do_not_raise(self):
        sel = ShrinkagePIP(kappa_threshold=0.5, pip_threshold=0.8)
        sel._validate_hyperparams()  # must not raise


# ===========================================================================
# ShrinkagePIP — integration (MCMC)
# ===========================================================================

@_slow
@_mcmc
class TestShrinkagePIPIntegration:
    """End-to-end MCMC tests for ShrinkagePIP (marked slow + mcmc)."""

    _RNG = np.random.default_rng(42)
    _N, _P = 60, 4
    # First 2 features are signal; last 2 are noise
    _X = _RNG.standard_normal((_N, _P))
    _y_linear = _X[:, 0] * 2.0 + _X[:, 1] * -1.5 + _RNG.standard_normal(_N) * 0.3
    _y_binary = (_y_linear > 0.0).astype(int)

    # -----------------------------------------------------------------------
    # Smoke / attribute structure
    # -----------------------------------------------------------------------

    def test_fit_populates_pip(self):
        """fit() must set pip_ with one value per feature."""
        sel = ShrinkagePIP(
            model_type="linear", prior="horseshoe",
            n_samples=50, tune=50, chains=1, random_state=42,
        )
        sel.fit(self._X, self._y_linear)
        assert hasattr(sel, "pip_")
        assert sel.pip_.shape == (self._P,)

    def test_fit_populates_confirmed_rejected_tentative(self):
        """All three selection lists must be present after fit()."""
        sel = ShrinkagePIP(
            model_type="linear", prior="horseshoe",
            n_samples=50, tune=50, chains=1, random_state=42,
        )
        sel.fit(self._X, self._y_linear)
        assert hasattr(sel, "confirmed_")
        assert hasattr(sel, "rejected_")
        assert hasattr(sel, "tentative_")
        # Parametric models have no tentative state
        assert sel.tentative_ == []

    def test_fit_populates_coef_mean_and_std(self):
        """coef_mean_ and coef_std_ must have shape (n_features,)."""
        sel = ShrinkagePIP(
            model_type="linear", prior="horseshoe",
            n_samples=50, tune=50, chains=1, random_state=42,
        )
        sel.fit(self._X, self._y_linear)
        assert sel.coef_mean_.shape == (self._P,)
        assert sel.coef_std_.shape == (self._P,)

    def test_fit_with_dataframe_uses_column_names(self):
        """When X is a DataFrame, feature_names_ must equal its columns."""
        cols = [f"col_{i}" for i in range(self._P)]
        df = pd.DataFrame(self._X, columns=cols)
        sel = ShrinkagePIP(
            model_type="linear", prior="horseshoe",
            n_samples=50, tune=50, chains=1, random_state=0,
        )
        sel.fit(df, self._y_linear)
        assert sel.feature_names_ == cols

    def test_fit_with_array_uses_feature_i_names(self):
        """When X is a plain ndarray, feature_names_ must be ['feature_0', ...]."""
        sel = ShrinkagePIP(
            model_type="linear", prior="horseshoe",
            n_samples=50, tune=50, chains=1, random_state=0,
        )
        sel.fit(self._X, self._y_linear)
        assert sel.feature_names_ == [f"feature_{i}" for i in range(self._P)]

    # -----------------------------------------------------------------------
    # PIP range
    # -----------------------------------------------------------------------

    def test_pip_values_in_unit_interval(self):
        """PIP must lie in [0, 1] for every feature."""
        sel = ShrinkagePIP(
            model_type="linear", prior="horseshoe",
            n_samples=50, tune=50, chains=1, random_state=42,
        )
        sel.fit(self._X, self._y_linear)
        assert np.all(sel.pip_ >= 0.0)
        assert np.all(sel.pip_ <= 1.0)

    def test_kappa_mean_in_unit_interval(self):
        """kappa_mean_ (Horseshoe shrinkage factor) must lie in (0, 1)."""
        sel = ShrinkagePIP(
            model_type="linear", prior="horseshoe", pip_method="kappa",
            n_samples=50, tune=50, chains=1, random_state=42,
        )
        sel.fit(self._X, self._y_linear)
        assert np.all(sel.kappa_mean_ > 0.0)
        assert np.all(sel.kappa_mean_ < 1.0)

    # -----------------------------------------------------------------------
    # Selection consistency
    # -----------------------------------------------------------------------

    def test_confirmed_plus_rejected_equals_all_features(self):
        """len(confirmed_) + len(rejected_) must equal n_features."""
        sel = ShrinkagePIP(
            model_type="linear", prior="horseshoe",
            n_samples=50, tune=50, chains=1, random_state=42,
        )
        sel.fit(self._X, self._y_linear)
        assert len(sel.confirmed_) + len(sel.rejected_) == self._P

    def test_support_matches_confirmed_count(self):
        """support_.sum() must equal len(confirmed_)."""
        sel = ShrinkagePIP(
            model_type="linear", prior="horseshoe",
            n_samples=50, tune=50, chains=1, random_state=42,
        )
        sel.fit(self._X, self._y_linear)
        assert int(sel.support_.sum()) == len(sel.confirmed_)

    # -----------------------------------------------------------------------
    # Summary schema
    # -----------------------------------------------------------------------

    def test_summary_has_required_columns_horseshoe(self):
        """Horseshoe summary() must contain pip, pip_method, mean, std, hdi_*, kappa_mean."""
        sel = ShrinkagePIP(
            model_type="linear", prior="horseshoe",
            n_samples=50, tune=50, chains=1, random_state=42,
        )
        sel.fit(self._X, self._y_linear)
        df = sel.summary()
        required = {"feature", "pip", "pip_method", "selected", "mean", "std",
                    "hdi_lower", "hdi_upper", "interval_type", "kappa_mean"}
        assert required.issubset(df.columns)

    def test_summary_has_required_columns_lasso(self):
        """Lasso summary() must contain pip, pip_method, mean, std, hdi_*, epsilon."""
        sel = ShrinkagePIP(
            model_type="linear", prior="lasso",
            n_samples=50, tune=50, chains=1, random_state=42,
        )
        sel.fit(self._X, self._y_linear)
        df = sel.summary()
        required = {"feature", "pip", "pip_method", "selected", "mean", "std",
                    "hdi_lower", "hdi_upper", "interval_type", "epsilon"}
        assert required.issubset(df.columns)

    def test_summary_row_count_equals_n_features(self):
        """summary() must have exactly one row per feature."""
        sel = ShrinkagePIP(
            model_type="linear", prior="horseshoe",
            n_samples=50, tune=50, chains=1, random_state=0,
        )
        sel.fit(self._X, self._y_linear)
        assert len(sel.summary()) == self._P

    def test_summary_hdi_lower_less_than_upper(self):
        """hdi_lower must be strictly less than hdi_upper for every feature."""
        sel = ShrinkagePIP(
            model_type="linear", prior="horseshoe",
            n_samples=50, tune=50, chains=1, random_state=42,
        )
        sel.fit(self._X, self._y_linear)
        df = sel.summary()
        assert (df["hdi_lower"] < df["hdi_upper"]).all(), (
            f"hdi_lower ≥ hdi_upper for some feature:\n{df[['feature', 'hdi_lower', 'hdi_upper']]}"
        )

    def test_summary_pip_method_column_matches_resolved_method(self):
        """The pip_method column in summary() must match pip_method_ on the fitted selector."""
        sel = ShrinkagePIP(
            model_type="linear", prior="horseshoe", pip_method="auto",
            n_samples=50, tune=50, chains=1, random_state=42,
        )
        sel.fit(self._X, self._y_linear)
        df = sel.summary()
        assert (df["pip_method"] == sel.pip_method_).all()

    # -----------------------------------------------------------------------
    # Prior variants
    # -----------------------------------------------------------------------

    def test_lasso_prior_linear(self):
        """Lasso prior + linear model must complete without error."""
        sel = ShrinkagePIP(
            model_type="linear", prior="lasso", pip_method="threshold",
            n_samples=50, tune=50, chains=1, random_state=7,
        )
        sel.fit(self._X, self._y_linear)
        assert sel.pip_.shape == (self._P,)
        assert sel.epsilon_ is not None

    def test_horseshoe_logistic(self):
        """Horseshoe prior + logistic model must complete without error."""
        sel = ShrinkagePIP(
            model_type="logistic", prior="horseshoe",
            n_samples=50, tune=50, chains=1, random_state=99,
        )
        sel.fit(self._X, self._y_binary)
        assert sel.pip_.shape == (self._P,)

    def test_kappa_method_with_lasso_raises(self):
        """pip_method='kappa' is invalid for Lasso — must raise ValueError after fit."""
        sel = ShrinkagePIP(
            model_type="linear", prior="lasso", pip_method="kappa",
            n_samples=50, tune=50, chains=1, random_state=0,
        )
        with pytest.raises(ValueError, match="pip_method='kappa' is only valid"):
            sel.fit(self._X, self._y_linear)

    # -----------------------------------------------------------------------
    # Statistical direction (weak signal with 50 draws, use generous threshold)
    # -----------------------------------------------------------------------

    def test_signal_features_have_higher_pip_than_noise(self):
        """Signal features (idx 0, 1) must have higher mean PIP than noise features (idx 2, 3).

        We use a very small sample count (50 draws) so this is a weak test;
        the assertion uses the *group mean* rather than individual features to
        reduce variance.
        """
        sel = ShrinkagePIP(
            model_type="linear", prior="horseshoe",
            n_samples=200, tune=100, chains=1, random_state=42,
        )
        sel.fit(self._X, self._y_linear)
        mean_signal_pip = sel.pip_[:2].mean()
        mean_noise_pip = sel.pip_[2:].mean()
        assert mean_signal_pip > mean_noise_pip, (
            f"Signal PIP ({mean_signal_pip:.3f}) must exceed noise PIP "
            f"({mean_noise_pip:.3f})"
        )

    def test_signal_features_have_lower_kappa_than_noise(self):
        """Signal features must have lower mean κ (less shrinkage) than noise features."""
        sel = ShrinkagePIP(
            model_type="linear", prior="horseshoe", pip_method="kappa",
            n_samples=200, tune=100, chains=1, random_state=42,
        )
        sel.fit(self._X, self._y_linear)
        mean_signal_kappa = sel.kappa_mean_[:2].mean()
        mean_noise_kappa = sel.kappa_mean_[2:].mean()
        assert mean_signal_kappa < mean_noise_kappa, (
            f"Signal κ ({mean_signal_kappa:.3f}) must be below noise κ "
            f"({mean_noise_kappa:.3f})"
        )


# ===========================================================================
# BARTImportance — pure-logic helpers (no MCMC)
# ===========================================================================

class TestBARTImportanceHelpers:
    """Tests for BARTImportance logic that does not require MCMC sampling."""

    def test_validate_credible_mass_zero_raises(self):
        bi = BARTImportance(credible_mass=0.0)
        with pytest.raises(ValueError, match="credible_mass"):
            bi._validate_hyperparams()

    def test_validate_credible_mass_one_raises(self):
        bi = BARTImportance(credible_mass=1.0)
        with pytest.raises(ValueError, match="credible_mass"):
            bi._validate_hyperparams()

    def test_validate_valid_credible_mass_ok(self):
        bi = BARTImportance(credible_mass=0.95)
        bi._validate_hyperparams()  # must not raise

    def test_vif_normalisation_sums_to_one_per_row(self):
        """Manually exercise the row-normalisation logic used by fit()."""
        rng = np.random.default_rng(0)
        raw = rng.integers(0, 10, size=(20, 4)).astype(float)
        row_sums = raw.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0.0] = 1.0
        normed = raw / row_sums
        # Every row must sum to ≤ 1.0 (exactly 1 unless all-zero row)
        assert np.allclose(normed.sum(axis=1), 1.0)

    def test_vif_normalisation_all_zero_row_stays_zero(self):
        """An all-zero draw (no splits at all) must not cause division by zero."""
        raw = np.zeros((3, 4))
        row_sums = raw.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0.0] = 1.0
        normed = raw / row_sums
        assert np.all(normed == 0.0)


# ===========================================================================
# BARTImportance — integration (MCMC)
# ===========================================================================

@_slow
@_mcmc
class TestBARTImportanceIntegration:
    """End-to-end MCMC tests for BARTImportance (marked slow + mcmc)."""

    _RNG = np.random.default_rng(7)
    _N, _P = 80, 4
    # Feature 0 is a strong linear signal; others are noise
    _X = _RNG.standard_normal((_N, _P))
    _y = _X[:, 0] * 3.0 + _RNG.standard_normal(_N) * 0.2

    # -----------------------------------------------------------------------
    # Smoke / attribute structure
    # -----------------------------------------------------------------------

    def test_fit_populates_vif_mean(self):
        """fit() must set vif_mean_ with one value per feature."""
        bi = BARTImportance(
            n_trees=3, n_samples=50, tune=50, chains=1, random_state=7,
        )
        bi.fit(self._X, self._y)
        assert hasattr(bi, "vif_mean_")
        assert bi.vif_mean_.shape == (self._P,)

    def test_fit_populates_vif_std(self):
        """vif_std_ must have shape (n_features,)."""
        bi = BARTImportance(
            n_trees=3, n_samples=50, tune=50, chains=1, random_state=7,
        )
        bi.fit(self._X, self._y)
        assert bi.vif_std_.shape == (self._P,)

    def test_fit_populates_hdi_bounds(self):
        """hdi_lower_ and hdi_upper_ must have shape (n_features,)."""
        bi = BARTImportance(
            n_trees=3, n_samples=50, tune=50, chains=1, random_state=7,
        )
        bi.fit(self._X, self._y)
        assert bi.hdi_lower_.shape == (self._P,)
        assert bi.hdi_upper_.shape == (self._P,)

    def test_fit_populates_confirmed_rejected_tentative(self):
        """All three selection lists must be present; tentative must be empty."""
        bi = BARTImportance(
            n_trees=3, n_samples=50, tune=50, chains=1, random_state=7,
        )
        bi.fit(self._X, self._y)
        assert hasattr(bi, "confirmed_")
        assert hasattr(bi, "rejected_")
        assert bi.tentative_ == []

    def test_fit_with_dataframe_uses_column_names(self):
        """When X is a DataFrame, feature_names_ must equal its columns."""
        cols = [f"f{i}" for i in range(self._P)]
        df = pd.DataFrame(self._X, columns=cols)
        bi = BARTImportance(
            n_trees=3, n_samples=50, tune=50, chains=1, random_state=0,
        )
        bi.fit(df, self._y)
        assert bi.feature_names_ == cols

    # -----------------------------------------------------------------------
    # VIF statistical invariants
    # -----------------------------------------------------------------------

    def test_vif_mean_sums_to_approx_one(self):
        """Mean VIF across features must sum to ≈ 1 (row-normalised distribution)."""
        bi = BARTImportance(
            n_trees=3, n_samples=50, tune=50, chains=1, random_state=7,
        )
        bi.fit(self._X, self._y)
        assert bi.vif_mean_.sum() == pytest.approx(1.0, abs=1e-6)

    def test_vif_mean_non_negative(self):
        """VIF values are frequencies — must all be ≥ 0."""
        bi = BARTImportance(
            n_trees=3, n_samples=50, tune=50, chains=1, random_state=7,
        )
        bi.fit(self._X, self._y)
        assert np.all(bi.vif_mean_ >= 0.0)

    def test_hdi_lower_less_than_upper(self):
        """hdi_lower_ must be less than or equal to hdi_upper_ for every feature."""
        bi = BARTImportance(
            n_trees=3, n_samples=50, tune=50, chains=1, random_state=7,
        )
        bi.fit(self._X, self._y)
        assert np.all(bi.hdi_lower_ <= bi.hdi_upper_), (
            f"hdi_lower > hdi_upper for some feature.\n"
            f"lower={bi.hdi_lower_}, upper={bi.hdi_upper_}"
        )

    def test_baseline_threshold_uses_uniform_default(self):
        """Default baseline_frequency=None must set baseline_threshold_ = 1/n_features."""
        bi = BARTImportance(
            baseline_frequency=None,
            n_trees=3, n_samples=50, tune=50, chains=1, random_state=7,
        )
        bi.fit(self._X, self._y)
        assert bi.baseline_threshold_ == pytest.approx(1.0 / self._P)

    def test_custom_baseline_threshold_stored(self):
        """An explicit baseline_frequency must be stored as baseline_threshold_."""
        bi = BARTImportance(
            baseline_frequency=0.10,
            n_trees=3, n_samples=50, tune=50, chains=1, random_state=7,
        )
        bi.fit(self._X, self._y)
        assert bi.baseline_threshold_ == pytest.approx(0.10)

    # -----------------------------------------------------------------------
    # Selection consistency
    # -----------------------------------------------------------------------

    def test_confirmed_plus_rejected_equals_all_features(self):
        """len(confirmed_) + len(rejected_) must equal n_features."""
        bi = BARTImportance(
            n_trees=3, n_samples=50, tune=50, chains=1, random_state=7,
        )
        bi.fit(self._X, self._y)
        assert len(bi.confirmed_) + len(bi.rejected_) == self._P

    def test_support_matches_confirmed_count(self):
        """support_.sum() must equal len(confirmed_)."""
        bi = BARTImportance(
            n_trees=3, n_samples=50, tune=50, chains=1, random_state=7,
        )
        bi.fit(self._X, self._y)
        assert int(bi.support_.sum()) == len(bi.confirmed_)

    def test_selection_rule_matches_hdi_lower_vs_baseline(self):
        """A feature is selected iff hdi_lower_ > baseline_threshold_ (by construction)."""
        bi = BARTImportance(
            n_trees=3, n_samples=50, tune=50, chains=1, random_state=7,
        )
        bi.fit(self._X, self._y)
        expected = bi.hdi_lower_ > bi.baseline_threshold_
        np.testing.assert_array_equal(bi.support_, expected)

    # -----------------------------------------------------------------------
    # Summary schema
    # -----------------------------------------------------------------------

    def test_summary_has_required_columns(self):
        """summary() must contain all documented columns."""
        bi = BARTImportance(
            n_trees=3, n_samples=50, tune=50, chains=1, random_state=7,
        )
        bi.fit(self._X, self._y)
        df = bi.summary()
        required = {
            "feature", "selected", "vif_mean", "vif_std",
            "hdi_lower", "hdi_upper", "baseline_threshold", "interval_type",
        }
        assert required.issubset(df.columns), (
            f"Missing columns: {required - set(df.columns)}"
        )

    def test_summary_row_count_equals_n_features(self):
        """summary() must have exactly one row per feature."""
        bi = BARTImportance(
            n_trees=3, n_samples=50, tune=50, chains=1, random_state=7,
        )
        bi.fit(self._X, self._y)
        assert len(bi.summary()) == self._P

    def test_summary_hdi_lower_less_than_upper(self):
        """summary() hdi_lower <= hdi_upper for every feature."""
        bi = BARTImportance(
            n_trees=3, n_samples=50, tune=50, chains=1, random_state=7,
        )
        bi.fit(self._X, self._y)
        df = bi.summary()
        assert (df["hdi_lower"] <= df["hdi_upper"]).all()

    def test_summary_selected_matches_support(self):
        """summary()['selected'] must match support_ element-wise."""
        bi = BARTImportance(
            n_trees=3, n_samples=50, tune=50, chains=1, random_state=7,
        )
        bi.fit(self._X, self._y)
        df = bi.summary()
        expected = [bool(s) for s in bi.support_]
        assert list(df["selected"]) == expected

    def test_summary_interval_type_is_hdi_or_ci(self):
        """interval_type must be either 'hdi' or 'ci' (ArviZ fallback)."""
        bi = BARTImportance(
            n_trees=3, n_samples=50, tune=50, chains=1, random_state=7,
        )
        bi.fit(self._X, self._y)
        df = bi.summary()
        assert df["interval_type"].isin({"hdi", "ci"}).all()

    # -----------------------------------------------------------------------
    # Statistical direction
    # -----------------------------------------------------------------------

    def test_signal_feature_has_highest_vif(self):
        """Feature 0 (strong signal) must have the highest mean VIF.

        Uses n_trees=10 and more draws for a better VIF estimate.
        """
        bi = BARTImportance(
            n_trees=10, n_samples=200, tune=100, chains=1, random_state=7,
        )
        bi.fit(self._X, self._y)
        assert np.argmax(bi.vif_mean_) == 0, (
            f"Expected feature 0 to have the highest VIF; got argmax={np.argmax(bi.vif_mean_)}.\n"
            f"vif_mean_={bi.vif_mean_}"
        )
