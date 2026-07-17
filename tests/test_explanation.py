import numpy as np
import pandas as pd
import pytest
from bxai.explanation.baylime import BayLIME, BayLIMEExplanation


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def make_data(n=100, p=4, seed=42):
    rng = np.random.default_rng(seed)
    training_data = rng.standard_normal((n, p))
    return training_data


def linear_predict(Z):
    """Simple deterministic predict function: returns column sum."""
    return Z.sum(axis=1)


# ---------------------------------------------------------------------------
# Analytical backend tests
# ---------------------------------------------------------------------------

def test_baylime_analytical_basic():
    training_data = make_data()
    explainer = BayLIME(
        training_data=training_data,
        feature_names=["f1", "f2", "f3", "f4"],
        num_samples=200,
        backend="analytical",
        random_state=42,
    )
    instance = np.array([0.5, -0.5, 1.0, -1.0])
    explanation = explainer.explain_instance(
        instance=instance,
        predict_fn=linear_predict,
        label=0,
    )

    assert isinstance(explanation, BayLIMEExplanation)
    assert explanation.backend == "analytical"
    assert len(explanation.coef_mean) == 4
    assert explanation.coef_cov.shape == (4, 4)
    assert explanation.posterior_draws_ is None


def test_baylime_analytical_as_list():
    training_data = make_data()
    explainer = BayLIME(
        training_data=training_data,
        num_samples=200,
        backend="analytical",
        random_state=42,
    )
    instance = np.zeros(4)
    explanation = explainer.explain_instance(instance, linear_predict, label=0)

    lst = explanation.as_list()
    assert len(lst) == 4
    assert all(isinstance(name, str) for name, _ in lst)
    assert all(isinstance(w, float) for _, w in lst)
    # Should be sorted by absolute magnitude descending
    abs_vals = [abs(w) for _, w in lst]
    assert abs_vals == sorted(abs_vals, reverse=True)


def test_baylime_analytical_as_dataframe():
    training_data = make_data()
    explainer = BayLIME(
        training_data=training_data,
        num_samples=200,
        backend="analytical",
        random_state=42,
    )
    instance = np.zeros(4)
    explanation = explainer.explain_instance(instance, linear_predict, label=0)

    df = explanation.as_dataframe()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 4
    assert set(["feature", "mean", "std", "hdi_lower", "hdi_upper", "value", "backend"]).issubset(df.columns)
    assert (df["hdi_lower"] < df["hdi_upper"]).all()
    assert (df["backend"] == "analytical").all()


def test_baylime_analytical_credible_intervals():
    training_data = make_data()
    explainer = BayLIME(
        training_data=training_data,
        num_samples=200,
        backend="analytical",
        random_state=42,
    )
    instance = np.zeros(4)
    explanation = explainer.explain_instance(instance, linear_predict, label=0)

    ci = explanation.credible_intervals(credible_mass=0.90)
    assert len(ci) == 4
    assert "feature_0" in ci
    lower, upper = ci["feature_0"]
    assert lower < upper


def test_baylime_analytical_with_prior_mean():
    training_data = make_data()
    # Inject a prior mean (simulating Phase 1 global SHAP weights)
    prior_mean = np.array([1.0, 0.0, -1.0, 0.5])
    explainer = BayLIME(
        training_data=training_data,
        num_samples=200,
        backend="analytical",
        prior_mean=prior_mean,
        random_state=42,
    )
    instance = np.zeros(4)
    explanation = explainer.explain_instance(instance, linear_predict, label=0)
    assert len(explanation.coef_mean) == 4


def test_baylime_analytical_bad_backend():
    training_data = make_data()
    explainer = BayLIME(training_data=training_data, backend="invalid")
    with pytest.raises(ValueError, match="backend must be"):
        explainer.explain_instance(np.zeros(4), linear_predict, label=0)


def test_baylime_analytical_bad_mcmc_prior():
    training_data = make_data()
    explainer = BayLIME(training_data=training_data, mcmc_prior="unsupported")
    with pytest.raises(ValueError, match="mcmc_prior must be"):
        explainer.explain_instance(np.zeros(4), linear_predict, label=0)


# ---------------------------------------------------------------------------
# MCMC backend tests
# ---------------------------------------------------------------------------

def test_baylime_mcmc_normal_prior():
    training_data = make_data(n=50, p=3, seed=0)
    explainer = BayLIME(
        training_data=training_data,
        feature_names=["a", "b", "c"],
        num_samples=100,
        backend="mcmc",
        mcmc_prior="normal",
        mcmc_samples=50,
        mcmc_tune=50,
        mcmc_chains=1,
        progressbar=False,
        random_state=42,
    )
    instance = np.array([0.1, -0.2, 0.3])
    explanation = explainer.explain_instance(instance, linear_predict, label=0)

    assert isinstance(explanation, BayLIMEExplanation)
    assert explanation.backend == "mcmc"
    assert explanation.posterior_draws_ is not None
    # Draws shape: (n_draws * n_chains, n_features)
    assert explanation.posterior_draws_.shape[1] == 3
    assert len(explanation.coef_mean) == 3


def test_baylime_mcmc_horseshoe_prior():
    training_data = make_data(n=50, p=3, seed=1)
    explainer = BayLIME(
        training_data=training_data,
        num_samples=100,
        backend="mcmc",
        mcmc_prior="horseshoe",
        mcmc_samples=50,
        mcmc_tune=50,
        mcmc_chains=1,
        progressbar=False,
        random_state=42,
    )
    instance = np.zeros(3)
    explanation = explainer.explain_instance(instance, linear_predict, label=0)

    assert explanation.backend == "mcmc"
    assert explanation.posterior_draws_ is not None


def test_baylime_mcmc_credible_intervals():
    training_data = make_data(n=50, p=3, seed=2)
    explainer = BayLIME(
        training_data=training_data,
        feature_names=["x", "y", "z"],
        num_samples=100,
        backend="mcmc",
        mcmc_prior="normal",
        mcmc_samples=50,
        mcmc_tune=50,
        mcmc_chains=1,
        progressbar=False,
        random_state=42,
    )
    instance = np.array([0.5, -0.5, 0.0])
    explanation = explainer.explain_instance(instance, linear_predict, label=0)

    ci = explanation.credible_intervals(credible_mass=0.90)
    assert len(ci) == 3
    assert all(lo < hi for lo, hi in ci.values())


def test_baylime_mcmc_as_dataframe():
    training_data = make_data(n=50, p=3, seed=3)
    explainer = BayLIME(
        training_data=training_data,
        num_samples=100,
        backend="mcmc",
        mcmc_prior="normal",
        mcmc_samples=50,
        mcmc_tune=50,
        mcmc_chains=1,
        progressbar=False,
        random_state=42,
    )
    instance = np.zeros(3)
    explanation = explainer.explain_instance(instance, linear_predict, label=0)

    df = explanation.as_dataframe()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3
    assert (df["backend"] == "mcmc").all()
    assert (df["hdi_lower"] < df["hdi_upper"]).all()


def test_baylime_mcmc_convergence_on_linear_truth():
    """MCMC posterior means should broadly agree with analytical means on a linear function."""
    np.random.seed(99)
    n, p = 80, 3
    training_data = np.random.randn(n, p)

    analytical = BayLIME(
        training_data=training_data,
        num_samples=300,
        backend="analytical",
        random_state=0,
    )
    mcmc = BayLIME(
        training_data=training_data,
        num_samples=300,
        backend="mcmc",
        mcmc_prior="normal",
        mcmc_samples=100,
        mcmc_tune=100,
        mcmc_chains=1,
        progressbar=False,
        random_state=0,
    )
    instance = np.zeros(p)

    exp_a = analytical.explain_instance(instance, linear_predict, label=0)
    exp_m = mcmc.explain_instance(instance, linear_predict, label=0)

    # Both backends should produce coefficient means of the same sign for each feature
    assert np.all(np.sign(exp_a.coef_mean) == np.sign(exp_m.coef_mean)), (
        f"Sign mismatch: analytical={exp_a.coef_mean}, mcmc={exp_m.coef_mean}"
    )


# ---------------------------------------------------------------------------
# Memory / summary-statistics tests
# ---------------------------------------------------------------------------

def test_baylime_does_not_store_training_array_after_setup():
    """_setup() must not retain the full training array — only means_ and stds_."""
    training_data = make_data(n=200, p=5)
    explainer = BayLIME(
        training_data=training_data,
        num_samples=50,
        backend="analytical",
        random_state=0,
    )
    instance = np.zeros(5)
    explainer.explain_instance(instance, linear_predict, label=0)

    assert not hasattr(explainer, "_training_data"), (
        "_training_data must not be stored after _setup(); it holds the full "
        "O(n_samples × n_features) copy and is unused after means_/stds_ are computed."
    )


def test_baylime_means_and_stds_match_training_data():
    """means_ and stds_ stored by _setup() must equal the actual column statistics."""
    rng = np.random.default_rng(7)
    training_data = rng.standard_normal((150, 4))

    explainer = BayLIME(
        training_data=training_data,
        num_samples=50,
        backend="analytical",
        random_state=7,
    )
    instance = np.zeros(4)
    explainer.explain_instance(instance, linear_predict, label=0)

    np.testing.assert_allclose(explainer.means_, training_data.mean(axis=0))
    expected_stds = training_data.std(axis=0)
    expected_stds[expected_stds == 0.0] = 1.0
    np.testing.assert_allclose(explainer.stds_, expected_stds)


def test_baylime_stds_zero_clamped():
    """Constant-value columns must produce stds_ == 1.0 (not 0.0) to prevent division by zero."""
    # Column 0 is constant; columns 1 and 2 vary
    training_data = np.ones((50, 3))
    training_data[:, 1] = np.linspace(-1, 1, 50)
    training_data[:, 2] = np.linspace(0, 2, 50)

    explainer = BayLIME(
        training_data=training_data,
        num_samples=50,
        backend="analytical",
        random_state=0,
    )
    instance = np.ones(3)
    explainer.explain_instance(instance, linear_predict, label=0)

    assert explainer.stds_[0] == 1.0, (
        f"Constant column std must be clamped to 1.0; got {explainer.stds_[0]}"
    )
    assert explainer.stds_[1] > 0.0
    assert explainer.stds_[2] > 0.0
