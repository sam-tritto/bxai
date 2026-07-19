import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

from bxai._utils.types import FeatureStatus
from bxai.selection.boruta_shap import BayesianBorutaSHAP
from bxai.selection.permutation import BayesianPermutation

# Shared fixtures (tiny_Xy, small_Xy, large_Xy, small_rf, large_rf)
# are provided by tests/conftest.py and injected automatically by pytest.


# ===========================================================================
# Shared helpers
# ===========================================================================


def _informative_indices(n_features: int, n_informative: int) -> np.ndarray:
    """Return the indices that make_classification places the informative features at."""
    return np.arange(n_informative)


def _noise_indices(n_features: int, n_informative: int) -> np.ndarray:
    """Return the indices of pure-noise features (non-informative, non-redundant)."""
    return np.arange(n_informative, n_features)


# ===========================================================================
# BayesianBorutaSHAP — discrete (Beta-Binomial) mode
# ===========================================================================


class TestBayesianBorutaSHAPDiscrete:
    """Tests for BayesianBorutaSHAP in 'discrete' (Beta-Binomial) mode."""

    # -----------------------------------------------------------------------
    # Smoke / structure
    # -----------------------------------------------------------------------

    def test_attributes_exist_after_fit(self, small_Xy):
        """Fit must populate confirmed_, rejected_, tentative_, support_, feature_importances_."""
        X, y = small_Xy
        selector = BayesianBorutaSHAP(mode="discrete", max_iter=3, random_state=42)
        selector.fit(X, y)

        assert hasattr(selector, "confirmed_")
        assert hasattr(selector, "rejected_")
        assert hasattr(selector, "tentative_")
        assert isinstance(selector.confirmed_, list)
        assert hasattr(selector, "feature_importances_")
        assert len(selector.feature_importances_) == X.shape[1]

    def test_iteration_history_populated(self, small_Xy):
        """Fit must collect iteration history iteration-by-iteration."""
        X, y = small_Xy
        selector = BayesianBorutaSHAP(mode="discrete", max_iter=3, random_state=42)
        selector.fit(X, y)

        assert hasattr(selector, "iteration_history_")
        assert len(selector.iteration_history_) == selector.n_iterations_

        # Check first history entry structure
        first_entry = selector.iteration_history_[0]
        assert "iteration" in first_entry
        assert first_entry["iteration"] == 1
        assert "status" in first_entry
        assert len(first_entry["status"]) == X.shape[1]
        assert "alpha" in first_entry
        assert "beta" in first_entry
        assert len(first_entry["alpha"]) == X.shape[1]

    def test_summary_structure(self, small_Xy):
        """summary() must return a DataFrame with the expected columns and shape."""
        X, y = small_Xy
        selector = BayesianBorutaSHAP(mode="discrete", max_iter=3, random_state=42)
        selector.fit(X, y)

        df = selector.summary()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 6
        assert "status" in df.columns
        assert "mean" in df.columns
        assert "ci_lower" in df.columns
        assert "ci_upper" in df.columns

    def test_plot_raises_if_not_fitted(self):
        """plot() must raise NotFittedError if estimator is not yet fitted."""
        from sklearn.exceptions import NotFittedError

        selector = BayesianBorutaSHAP(mode="discrete")
        with pytest.raises(NotFittedError):
            selector.plot()

    def test_plot_runs(self, small_Xy):
        """plot() must run and return a matplotlib figure without raising exceptions."""
        X, y = small_Xy
        selector = BayesianBorutaSHAP(mode="discrete", max_iter=3, random_state=42)
        selector.fit(X, y)

        fig = selector.plot()
        import matplotlib.pyplot as plt

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    # -----------------------------------------------------------------------
    # credible_mass override
    # -----------------------------------------------------------------------

    def test_summary_credible_mass_override_produces_narrower_interval(self, small_Xy):
        """Passing a smaller credible_mass to summary() must yield narrower intervals."""
        X, y = small_Xy
        selector = BayesianBorutaSHAP(mode="discrete", max_iter=5, random_state=42)
        selector.fit(X, y)

        df_wide = selector.summary(credible_mass=0.95)
        df_narrow = selector.summary(credible_mass=0.50)

        widths_wide = df_wide["ci_upper"].values - df_wide["ci_lower"].values
        widths_narrow = df_narrow["ci_upper"].values - df_narrow["ci_lower"].values
        assert (widths_narrow < widths_wide).all(), (
            "A smaller credible_mass must yield strictly narrower credible intervals."
        )

    def test_summary_default_equals_constructor_credible_mass(self, small_Xy):
        """summary() with no argument must equal summary(credible_mass=self.credible_mass)."""
        X, y = small_Xy
        selector = BayesianBorutaSHAP(
            mode="discrete", max_iter=5, credible_mass=0.90, random_state=42
        )
        selector.fit(X, y)

        df_default = selector.summary()
        df_explicit = selector.summary(credible_mass=0.90)

        pd.testing.assert_frame_equal(df_default, df_explicit)

    # -----------------------------------------------------------------------
    # Credible-interval ordering
    # -----------------------------------------------------------------------

    def test_hdi_lower_less_than_upper(self, small_Xy):
        """summary() must satisfy hdi_lower < hdi_upper for every feature."""
        X, y = small_Xy
        selector = BayesianBorutaSHAP(mode="discrete", max_iter=5, random_state=42)
        selector.fit(X, y)

        df = selector.summary()
        assert (df["ci_lower"] < df["ci_upper"]).all(), (
            "ci_lower must be strictly less than ci_upper for all features.\n"
            f"Got:\n{df[['feature', 'ci_lower', 'ci_upper']]}"
        )

    def test_posterior_mean_in_unit_interval(self, small_Xy):
        """Discrete-mode posterior means are Beta distribution means: must lie in (0, 1)."""
        X, y = small_Xy
        selector = BayesianBorutaSHAP(mode="discrete", max_iter=5, random_state=42)
        selector.fit(X, y)

        df = selector.summary()
        assert (df["mean"] > 0.0).all()
        assert (df["mean"] < 1.0).all()

    # -----------------------------------------------------------------------
    # Informative vs noise ranking
    # -----------------------------------------------------------------------

    def test_confirmed_features_rank_higher_than_rejected(self):
        """Confirmed features must have a higher posterior mean than rejected features.

        We use the selector's own CONFIRMED / REJECTED decisions as ground truth for
        which features the Bayesian evidence supports.  Confirmed features must have
        a strictly higher Beta-Binomial posterior mean (α / (α+β)) than rejected ones.
        """
        X, y = make_classification(
            n_samples=200,
            n_features=8,
            n_informative=2,
            n_redundant=0,
            n_repeated=0,
            n_clusters_per_class=1,
            random_state=7,
        )
        selector = BayesianBorutaSHAP(
            mode="discrete",
            max_iter=30,
            random_state=7,
        )
        selector.fit(X, y)

        df = selector.summary()
        confirmed = df[df["status"] == FeatureStatus.CONFIRMED.value]
        rejected = df[df["status"] == FeatureStatus.REJECTED.value]

        # Only assert when the selector produced both groups
        if confirmed.empty or rejected.empty:
            pytest.skip(
                "Selector did not produce both CONFIRMED and REJECTED features — "
                "increase max_iter or data size."
            )

        mean_confirmed = confirmed["mean"].mean()
        mean_rejected = rejected["mean"].mean()

        assert mean_confirmed > mean_rejected, (
            f"CONFIRMED posterior mean ({mean_confirmed:.4f}) must exceed "
            f"REJECTED posterior mean ({mean_rejected:.4f}).\n"
            f"Summary:\n{df[['feature', 'mean', 'status']]}"
        )

    def test_posterior_updates_accumulate(self, small_Xy):
        """Running more iterations must raise posterior evidence (alpha + beta grows)."""
        X, y = small_Xy

        sel_few = BayesianBorutaSHAP(mode="discrete", max_iter=2, random_state=42)
        sel_few.fit(X, y)
        total_few = sel_few.tracker_.alpha.sum() + sel_few.tracker_.beta.sum()

        sel_more = BayesianBorutaSHAP(mode="discrete", max_iter=8, random_state=42)
        sel_more.fit(X, y)
        total_more = sel_more.tracker_.alpha.sum() + sel_more.tracker_.beta.sum()

        assert total_more > total_few, (
            "More iterations must accumulate more evidence in the posterior."
        )

    def test_early_stopping_disabled(self, small_Xy):
        """Fit must run the full max_iter iterations when early_stopping is False."""
        X, y = small_Xy
        selector = BayesianBorutaSHAP(
            mode="discrete", max_iter=6, early_stopping=False, random_state=42
        )
        selector.fit(X, y)
        assert selector.n_iterations_ == 6


# ===========================================================================
# BayesianBorutaSHAP — continuous (Normal-IG) mode
# ===========================================================================


class TestBayesianBorutaSHAPContinuous:
    """Tests for BayesianBorutaSHAP in 'continuous' (Normal-IG) mode."""

    # -----------------------------------------------------------------------
    # Smoke / structure
    # -----------------------------------------------------------------------

    def test_summary_has_mu_column(self, small_Xy):
        """Continuous-mode summary must contain a 'mu' column."""
        X, y = small_Xy
        selector = BayesianBorutaSHAP(mode="continuous", max_iter=3, random_state=42)
        selector.fit(X, y)

        df = selector.summary()
        assert "mu" in df.columns

    def test_plot_runs(self, small_Xy):
        """plot() must run in continuous mode and return a matplotlib figure."""
        X, y = small_Xy
        selector = BayesianBorutaSHAP(mode="continuous", max_iter=3, random_state=42)
        selector.fit(X, y)

        fig = selector.plot()
        import matplotlib.pyplot as plt

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_iteration_history_populated(self, small_Xy):
        """Fit must collect iteration history iteration-by-iteration in continuous mode."""
        X, y = small_Xy
        selector = BayesianBorutaSHAP(mode="continuous", max_iter=3, random_state=42)
        selector.fit(X, y)

        assert hasattr(selector, "iteration_history_")
        assert len(selector.iteration_history_) == selector.n_iterations_

        # Check first history entry structure
        first_entry = selector.iteration_history_[0]
        assert "iteration" in first_entry
        assert first_entry["iteration"] == 1
        assert "status" in first_entry
        assert len(first_entry["status"]) == X.shape[1]
        assert "mu" in first_entry
        assert "nu" in first_entry
        assert "alpha" in first_entry
        assert "beta" in first_entry
        assert len(first_entry["mu"]) == X.shape[1]

    # -----------------------------------------------------------------------
    # Credible-interval ordering
    # -----------------------------------------------------------------------

    def test_hdi_lower_less_than_upper(self, small_Xy):
        """hdi_lower < hdi_upper must hold for every feature in continuous mode."""
        X, y = small_Xy
        selector = BayesianBorutaSHAP(mode="continuous", max_iter=5, random_state=42)
        selector.fit(X, y)

        df = selector.summary()
        assert (df["hdi_lower"] < df["hdi_upper"]).all(), (
            f"hdi_lower must be strictly less than hdi_upper.\n"
            f"Got:\n{df[['feature', 'hdi_lower', 'hdi_upper']]}"
        )

    # -----------------------------------------------------------------------
    # Posterior update direction
    # -----------------------------------------------------------------------

    def test_confirmed_features_rank_higher_than_rejected(self):
        """CONFIRMED features must have a higher posterior mu than REJECTED features.

        In continuous mode the tracker accumulates (active_importance - max_shadow).
        Features the selector CONFIRMS must therefore have a strictly higher
        posterior mu than features it REJECTS.
        """
        X, y = make_classification(
            n_samples=200,
            n_features=8,
            n_informative=2,
            n_redundant=0,
            n_repeated=0,
            n_clusters_per_class=1,
            random_state=13,
        )
        selector = BayesianBorutaSHAP(
            mode="continuous",
            max_iter=30,
            random_state=13,
        )
        selector.fit(X, y)

        df = selector.summary()
        confirmed = df[df["status"] == FeatureStatus.CONFIRMED.value]
        rejected = df[df["status"] == FeatureStatus.REJECTED.value]

        if confirmed.empty or rejected.empty:
            pytest.skip(
                "Selector did not produce both CONFIRMED and REJECTED features."
            )

        mu_confirmed = confirmed["mu"].mean()
        mu_rejected = rejected["mu"].mean()

        assert mu_confirmed > mu_rejected, (
            f"CONFIRMED posterior mu ({mu_confirmed:.4f}) must exceed "
            f"REJECTED posterior mu ({mu_rejected:.4f}).\n"
            f"Summary:\n{df[['feature', 'mu', 'status']]}"
        )

    def test_confirmed_feature_mu_positive(self):
        """Every CONFIRMED feature must have a positive posterior mu.

        A feature is CONFIRMED because its CI for (importance − max_shadow) lies
        entirely above zero, so its posterior mean *must* be positive by construction.
        """
        X, y = make_classification(
            n_samples=200,
            n_features=8,
            n_informative=2,
            n_redundant=0,
            n_repeated=0,
            n_clusters_per_class=1,
            random_state=13,
        )
        selector = BayesianBorutaSHAP(mode="continuous", max_iter=30, random_state=13)
        selector.fit(X, y)

        df = selector.summary()
        confirmed = df[df["status"] == FeatureStatus.CONFIRMED.value]

        if confirmed.empty:
            pytest.skip("No features confirmed — increase max_iter or data size.")

        assert (confirmed["mu"] > 0.0).all(), (
            f"All CONFIRMED features must have positive posterior mu.\n"
            f"Got:\n{confirmed[['feature', 'mu']]}"
        )


# ===========================================================================
# BayesianPermutation
# ===========================================================================


class TestBayesianPermutation:
    """Tests for BayesianPermutation feature selection."""

    # -----------------------------------------------------------------------
    # Smoke / structure
    # -----------------------------------------------------------------------

    def test_attributes_exist_after_fit(self, small_Xy, small_rf):
        """Fit must populate confirmed_, rejected_, tentative_, feature_importances_."""
        X, y = small_Xy
        selector = BayesianPermutation(
            model=small_rf, scoring="accuracy", n_repeats=5, random_state=42
        )
        selector.fit(X, y)

        assert hasattr(selector, "confirmed_")
        assert hasattr(selector, "rejected_")
        assert hasattr(selector, "feature_importances_")
        assert len(selector.feature_importances_) == X.shape[1]

    def test_bayesian_permutation_parallel(self, small_Xy, small_rf):
        """Fit with n_jobs=2 must produce identical results to sequential fit."""
        X, y = small_Xy
        sel_seq = BayesianPermutation(
            model=small_rf, scoring="accuracy", n_repeats=5, n_jobs=1, random_state=42
        )
        sel_seq.fit(X, y)

        sel_par = BayesianPermutation(
            model=small_rf, scoring="accuracy", n_repeats=5, n_jobs=2, random_state=42
        )
        sel_par.fit(X, y)

        np.testing.assert_allclose(
            sel_seq.feature_importances_, sel_par.feature_importances_
        )
        assert sel_seq.confirmed_ == sel_par.confirmed_
        assert sel_seq.rejected_ == sel_par.rejected_

    def test_summary_structure(self, small_Xy, small_rf):
        """summary() must return a 6-row DataFrame with the expected columns."""
        X, y = small_Xy
        selector = BayesianPermutation(
            model=small_rf, scoring="accuracy", n_repeats=5, random_state=42
        )
        selector.fit(X, y)

        df = selector.summary()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 6
        assert "hdi_lower" in df.columns

    def test_plot_raises_if_not_fitted(self, small_rf):
        """plot() must raise NotFittedError if estimator is not yet fitted."""
        from sklearn.exceptions import NotFittedError

        selector = BayesianPermutation(model=small_rf, scoring="accuracy")
        with pytest.raises(NotFittedError):
            selector.plot()

    def test_plot_runs(self, small_Xy, small_rf):
        """plot() must run and return a matplotlib figure without raising exceptions."""
        X, y = small_Xy
        selector = BayesianPermutation(
            model=small_rf, scoring="accuracy", n_repeats=5, random_state=42
        )
        selector.fit(X, y)

        fig = selector.plot()
        import matplotlib.pyplot as plt

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    # -----------------------------------------------------------------------
    # credible_mass override
    # -----------------------------------------------------------------------

    def test_summary_credible_mass_override_produces_narrower_interval(
        self, small_Xy, small_rf
    ):
        """Passing a smaller credible_mass to summary() must yield narrower intervals."""
        X, y = small_Xy
        selector = BayesianPermutation(
            model=small_rf, scoring="accuracy", n_repeats=10, random_state=42
        )
        selector.fit(X, y)

        df_wide = selector.summary(credible_mass=0.95)
        df_narrow = selector.summary(credible_mass=0.50)

        widths_wide = df_wide["hdi_upper"].values - df_wide["hdi_lower"].values
        widths_narrow = df_narrow["hdi_upper"].values - df_narrow["hdi_lower"].values
        assert (widths_narrow < widths_wide).all(), (
            "A smaller credible_mass must yield strictly narrower credible intervals."
        )

    def test_summary_default_equals_constructor_credible_mass(self, small_Xy, small_rf):
        """summary() with no argument must equal summary(credible_mass=self.credible_mass)."""
        X, y = small_Xy
        selector = BayesianPermutation(
            model=small_rf,
            scoring="accuracy",
            n_repeats=5,
            credible_mass=0.90,
            random_state=42,
        )
        selector.fit(X, y)

        df_default = selector.summary()
        df_explicit = selector.summary(credible_mass=0.90)

        pd.testing.assert_frame_equal(df_default, df_explicit)

    def test_hdi_lower_less_than_upper(self, small_Xy, small_rf):
        """hdi_lower < hdi_upper must hold for every feature."""
        X, y = small_Xy
        selector = BayesianPermutation(
            model=small_rf, scoring="accuracy", n_repeats=10, random_state=42
        )
        selector.fit(X, y)

        df = selector.summary()
        assert (df["hdi_lower"] < df["hdi_upper"]).all(), (
            f"hdi_lower must be strictly less than hdi_upper.\n"
            f"Got:\n{df[['feature', 'hdi_lower', 'hdi_upper']]}"
        )

    # -----------------------------------------------------------------------
    # Posterior update direction
    # -----------------------------------------------------------------------

    def test_confirmed_features_rank_higher_than_rejected(self, large_Xy, large_rf):
        """CONFIRMED features must have a higher posterior mean score-drop than REJECTED ones.

        The selector's own decisions are the ground truth: by construction, a feature is
        CONFIRMED only when its CI lies entirely above zero, so CONFIRMED mean > REJECTED
        mean must always hold when both groups exist.
        """
        X, y = large_Xy
        selector = BayesianPermutation(
            model=large_rf,
            scoring="accuracy",
            n_repeats=20,
            random_state=0,
        )
        selector.fit(X, y)

        df = selector.summary()
        confirmed = df[df["status"] == FeatureStatus.CONFIRMED.value]
        rejected = df[df["status"] == FeatureStatus.REJECTED.value]

        if confirmed.empty or rejected.empty:
            pytest.skip(
                "Selector did not produce both CONFIRMED and REJECTED features."
            )

        mean_confirmed = confirmed["mean"].mean()
        mean_rejected = rejected["mean"].mean()

        assert mean_confirmed > mean_rejected, (
            f"CONFIRMED posterior mean ({mean_confirmed:.4f}) must exceed "
            f"REJECTED posterior mean ({mean_rejected:.4f}).\n"
            f"Summary:\n{df[['feature', 'mean', 'status']]}"
        )

    def test_confirmed_features_have_positive_posterior_mean(self, large_Xy, large_rf):
        """Any CONFIRMED feature must have a positive posterior mean (score drop > 0)."""
        X, y = large_Xy
        selector = BayesianPermutation(
            model=large_rf,
            scoring="accuracy",
            n_repeats=20,
            random_state=0,
        )
        selector.fit(X, y)

        df = selector.summary()
        confirmed = df[df["status"] == FeatureStatus.CONFIRMED.value]

        if not confirmed.empty:
            assert (confirmed["mean"] > 0.0).all(), (
                f"CONFIRMED features must have positive posterior means.\n"
                f"Got:\n{confirmed[['feature', 'mean']]}"
            )

    def test_posterior_mean_direction_matches_score_drop_sign(self, small_Xy, small_rf):
        """The sign of the posterior mean must reflect the actual mean score drop.

        For a good model and n_repeats that is large enough, posterior mu
        should converge close to the sample mean of the score drops.
        """
        X, y = small_Xy
        selector = BayesianPermutation(
            model=small_rf, scoring="accuracy", n_repeats=15, random_state=5
        )
        selector.fit(X, y)

        # Posterior mu must equal tracker.mu which is the conjugate posterior mean,
        # i.e. it must be in the same direction as the empirical signal.
        df = selector.summary()
        # For informative features (indices 0, 1), the average drop must be positive.
        mean_informative = df.iloc[:2]["mean"].mean()
        assert mean_informative > 0.0, (
            f"Informative features (idx 0,1) must have positive mean score drop; "
            f"got {mean_informative:.4f}"
        )

    def test_bayesian_permutation_rope_validation(self, small_Xy, small_rf):
        """Test rope validation on BayesianPermutation."""
        bp1 = BayesianPermutation(model=small_rf, scoring="accuracy", rope=0.001)
        bp1._validate_hyperparams()

        bp2 = BayesianPermutation(
            model=small_rf, scoring="accuracy", rope=(-0.001, 0.001)
        )
        bp2._validate_hyperparams()

        bp3 = BayesianPermutation(model=small_rf, scoring="accuracy", rope=-0.001)
        with pytest.raises(ValueError, match="rope must be non-negative"):
            bp3._validate_hyperparams()

        bp4 = BayesianPermutation(
            model=small_rf, scoring="accuracy", rope=(0.001, -0.001)
        )
        with pytest.raises(ValueError, match="rope lower bound must be <= upper bound"):
            bp4._validate_hyperparams()

        bp5 = BayesianPermutation(
            model=small_rf, scoring="accuracy", rope=(0.001, 0.002, 0.003)
        )
        with pytest.raises(TypeError):
            bp5._validate_hyperparams()

    def test_bayesian_permutation_rope_fit(self, small_Xy, small_rf):
        """Test that rope parameter changes features classification as expected."""
        X, y = small_Xy
        bp_large = BayesianPermutation(
            model=small_rf, scoring="accuracy", rope=10.0, n_repeats=5, random_state=42
        )
        bp_large.fit(X, y)

        # All features must be rejected due to huge ROPE
        assert len(bp_large.confirmed_) == 0
        assert len(bp_large.rejected_) == X.shape[1]


def test_pipeline_integration(small_Xy, small_rf):
    from sklearn.pipeline import Pipeline

    X, y = small_Xy

    # Test BayesianBorutaSHAP pipeline integration
    pipe_boruta = Pipeline(
        [("selector", BayesianBorutaSHAP(mode="discrete", max_iter=3, random_state=42))]
    )
    pipe_boruta.fit(X, y)
    X_trans = pipe_boruta.transform(X)
    assert X_trans.shape[0] == X.shape[0]

    # Force all features to be selected to verify non-empty transform
    selector_boruta = pipe_boruta.named_steps["selector"]
    selector_boruta.support_ = np.ones(X.shape[1], dtype=bool)
    X_trans_all = selector_boruta.transform(X)
    assert X_trans_all.shape == X.shape

    # Test BayesianPermutation pipeline integration
    pipe_perm = Pipeline(
        [
            (
                "selector",
                BayesianPermutation(
                    model=small_rf, scoring="accuracy", n_repeats=3, random_state=42
                ),
            )
        ]
    )
    pipe_perm.fit(X, y)
    X_trans_perm = pipe_perm.transform(X)
    assert X_trans_perm.shape[0] == X.shape[0]

    # Force all features to be selected to verify non-empty transform
    selector_perm = pipe_perm.named_steps["selector"]
    selector_perm.support_ = np.ones(X.shape[1], dtype=bool)
    X_trans_all_perm = selector_perm.transform(X)
    assert X_trans_all_perm.shape == X.shape
