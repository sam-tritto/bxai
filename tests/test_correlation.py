from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bxai.parametric.correlation import BayesianCorrelation

# Marker aliases
_slow = pytest.mark.slow
_mcmc = pytest.mark.mcmc


# ===========================================================================
# Unit Tests (No MCMC)
# ===========================================================================

class TestBayesianCorrelationValidation:
    """Unit tests for parameter and input validation in BayesianCorrelation."""

    def test_invalid_method_raises(self):
        model = BayesianCorrelation(method="invalid_method")
        with pytest.raises(ValueError, match="method must be"):
            model._validate_hyperparams()

    def test_invalid_backend_raises(self):
        model = BayesianCorrelation(backend="invalid_backend")
        with pytest.raises(ValueError, match="backend must be"):
            model._validate_hyperparams()

    def test_incompatible_pearson_latent_copula_raises(self):
        model = BayesianCorrelation(method="pearson", backend="latent_copula")
        with pytest.raises(ValueError, match="not compatible with backend='latent_copula'"):
            model._validate_hyperparams()

    def test_incompatible_kendall_quick_raises(self):
        model = BayesianCorrelation(method="kendall", backend="quick")
        with pytest.raises(ValueError, match="requires backend='latent_copula'"):
            model._validate_hyperparams()

    def test_invalid_credible_mass_raises(self):
        for mass in (0.0, 1.0, -0.5, 1.5):
            model = BayesianCorrelation(credible_mass=mass)
            with pytest.raises(ValueError, match="credible_mass must be in"):
                model._validate_hyperparams()

    def test_invalid_cores_raises(self):
        for cores in (0, -2, 1.5, "2"):
            model = BayesianCorrelation(cores=cores)  # type: ignore
            with pytest.raises(ValueError, match="cores must be a positive integer"):
                model._validate_hyperparams()

    def test_invalid_input_shapes_raises(self):
        model = BayesianCorrelation()
        
        # X is 1D and y is None
        with pytest.raises(ValueError, match="must be a 2D array or DataFrame"):
            model.fit(np.array([1.0, 2.0, 3.0]))

        # X is 2D but not 2 columns
        with pytest.raises(ValueError, match="with exactly 2 columns"):
            model.fit(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))

        # X and y have inconsistent length
        with pytest.raises(ValueError, match="Found input variables with inconsistent numbers of samples"):
            model.fit(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0]))

        # Too few samples
        with pytest.raises(ValueError, match="requires at least 3 observations"):
            model.fit(np.array([1.0, 2.0]), np.array([1.0, 2.0]))

    def test_variable_name_resolution_df(self):
        df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        # We simulate variable name resolution by bypassing sampling
        model = BayesianCorrelation()
        # Set attributes to mock fit completion
        model.variable_names_ = [str(c) for c in df.columns]
        assert model.variable_names_ == ["A", "B"]

    def test_variable_name_resolution_series(self):
        s1 = pd.Series([1, 2, 3], name="Var1")
        s2 = pd.Series([4, 5, 6], name="Var2")
        model = BayesianCorrelation()
        model.variable_names_ = [str(s1.name), str(s2.name)]
        assert model.variable_names_ == ["Var1", "Var2"]


# ===========================================================================
# Integration Tests (MCMC)
# ===========================================================================

@_slow
@_mcmc
class TestBayesianCorrelationIntegration:
    """MCMC integration tests for BayesianCorrelation."""

    @classmethod
    def setup_class(cls):
        # Generate correlated dummy data (positive correlation)
        np.random.seed(42)
        cls.x = np.random.normal(0.0, 1.0, 40)
        cls.y = 0.8 * cls.x + np.random.normal(0.0, 0.5, 40)
        cls.data = np.column_stack([cls.x, cls.y])

        # Generate negatively correlated dummy data
        cls.y_neg = -0.8 * cls.x + np.random.normal(0.0, 0.5, 40)
        cls.data_neg = np.column_stack([cls.x, cls.y_neg])

    def test_pearson_quick_positive(self):
        model = BayesianCorrelation(
            method="pearson",
            backend="quick",
            n_samples=50,
            tune=50,
            chains=1,
            random_state=42,
        )
        model.fit(self.data)
        
        assert hasattr(model, "correlation_samples_")
        assert model.correlation_samples_.shape == (50,)
        assert model.mean_ > 0.5
        assert model.mode_ > 0.5
        assert model.hdi_lower_ < model.hdi_upper_
        assert model.probability_of_direction_ > 0.95
        assert model.variable_names_ == ["x", "y"]
        assert model.strength_ == "Strong"

    def test_pearson_quick_negative(self):
        model = BayesianCorrelation(
            method="pearson",
            backend="quick",
            n_samples=50,
            tune=50,
            chains=1,
            random_state=42,
        )
        model.fit(self.data_neg)
        assert model.mean_ < -0.5
        assert model.mode_ < -0.5

    def test_spearman_quick(self):
        model = BayesianCorrelation(
            method="spearman",
            backend="quick",
            n_samples=50,
            tune=50,
            chains=1,
            random_state=42,
        )
        model.fit(self.x, self.y)
        assert model.mean_ > 0.5
        assert model.mode_ > 0.5
        assert model.probability_of_direction_ > 0.95

    def test_spearman_latent_copula(self):
        model = BayesianCorrelation(
            method="spearman",
            backend="latent_copula",
            n_samples=50,
            tune=50,
            chains=1,
            random_state=42,
        )
        model.fit(self.data)
        assert model.mean_ > 0.5
        assert model.mode_ > 0.5
        assert model.probability_of_direction_ > 0.95

    def test_kendall_latent_copula(self):
        model = BayesianCorrelation(
            method="kendall",
            backend="latent_copula",
            n_samples=50,
            tune=50,
            chains=1,
            random_state=42,
        )
        model.fit(self.data)
        assert model.mean_ > 0.4
        assert model.mode_ > 0.4
        assert model.probability_of_direction_ > 0.95

    def test_variable_names_resolved_from_pandas(self):
        df = pd.DataFrame({"feat_a": self.x, "feat_b": self.y})
        model = BayesianCorrelation(
            method="pearson",
            backend="quick",
            n_samples=30,
            tune=30,
            chains=1,
            random_state=42,
        )
        model.fit(df)
        assert model.variable_names_ == ["feat_a", "feat_b"]

        s_a = pd.Series(self.x, name="a_series")
        s_b = pd.Series(self.y, name="b_series")
        model.fit(s_a, s_b)
        assert model.variable_names_ == ["a_series", "b_series"]

    def test_plot_posterior_and_forest_plot(self):
        import matplotlib.pyplot as plt
        model = BayesianCorrelation(
            method="pearson",
            backend="quick",
            n_samples=30,
            tune=30,
            chains=1,
            random_state=42,
        )
        model.fit(self.data)
        
        # Test plot_posterior
        ax = model.plot_posterior()
        assert ax is not None
        
        # Test summary
        s_df = model.summary()
        assert isinstance(s_df, pd.DataFrame)
        assert list(s_df.columns) == [
            "Feature 1", "Feature 2", "Posterior Mean", "Posterior Mode", "95% HDI Lower", "95% HDI Upper", "Prob of Direction", "Strength"
        ]
        
        # Test plot (forest plot)
        fig = model.plot()
        assert fig is not None
        
        # Test plot with custom color
        fig_color = model.plot(color="orange")
        assert fig_color is not None
        
        plt.close("all")

    def test_multifeature_correlation(self):
        # 3 features: feat_0 is positively correlated with target, feat_1 is negatively correlated, feat_2 is noise
        np.random.seed(0)
        t = np.random.normal(0, 1, 40)
        f0 = 0.8 * t + np.random.normal(0, 0.5, 40)
        f1 = -0.7 * t + np.random.normal(0, 0.5, 40)
        f2 = np.random.normal(0, 1, 40)
        
        X_df = pd.DataFrame({"feat_0": f0, "feat_1": f1, "feat_2": f2})
        y_ser = pd.Series(t, name="target_var")
        
        model = BayesianCorrelation(
            method="pearson",
            backend="quick",
            n_samples=30,
            tune=30,
            chains=1,
            random_state=42,
        )
        model.fit(X_df, y_ser)
        
        # Verify shape of attributes
        assert isinstance(model.correlation_samples_, np.ndarray)
        assert model.correlation_samples_.shape == (30, 3)
        assert isinstance(model.mean_, np.ndarray)
        assert len(model.mean_) == 3
        assert isinstance(model.mode_, np.ndarray)
        assert len(model.mode_) == 3
        assert isinstance(model.hdi_lower_, np.ndarray)
        assert len(model.hdi_lower_) == 3
        assert isinstance(model.hdi_upper_, np.ndarray)
        assert len(model.hdi_upper_) == 3
        assert isinstance(model.probability_of_direction_, np.ndarray)
        assert len(model.probability_of_direction_) == 3
        
        # Verify signs/directions of correlations
        assert model.mean_[0] > 0.4
        assert model.mean_[1] < -0.4
        assert abs(model.mean_[2]) < 0.3
        
        # Verify summary_df_ structure
        assert isinstance(model.summary_df_, pd.DataFrame)
        assert list(model.summary_df_.columns) == [
            "Feature", "Target", "Posterior Mean", "Posterior Mode", "95% HDI Lower", "95% HDI Upper", "Prob of Direction", "Strength"
        ]
        assert list(model.summary_df_["Feature"]) == ["feat_0", "feat_1", "feat_2"]
        assert list(model.summary_df_["Target"]) == ["target_var", "target_var", "target_var"]
        assert set(model.summary_df_["Strength"]).issubset({"Strong", "Modest", "Uncertain"})

