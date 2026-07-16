import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from typing import Optional, List, Dict, Tuple, Union, Any

from bxai._utils.types import FeatureStatus
from bxai._utils.validation import check_consistent_length


class ShrinkagePIP(BaseEstimator):
    """Shrinkage Prior (Horseshoe or Lasso) Feature Selection for GLMs.

    Fits a linear or logistic regression model regularized with a Horseshoe or
    Lasso shrinkage prior using PyMC (MCMC), and computes the Posterior Inclusion
    Probability (PIP) for each feature.  Features are selected if their PIP
    crosses a user-defined threshold.

    PIP computation strategy
    ------------------------
    ``pip_method='kappa'`` (default for Horseshoe prior)
        Uses the posterior shrinkage factor

            κ_j = 1 / (1 + λ_j² τ²)

        A feature is "included" in a given draw when κ_j < ``kappa_threshold``
        (i.e. the local scale dominates and the coefficient is *not* shrunk to
        zero).  This is the statistically correct quantity for continuous
        shrinkage priors such as the Horseshoe because the posterior of β_j
        has full support on ℝ — P(|β_j| > ε) is trivially close to 1 for any
        small ε, making a direct threshold useless.

    ``pip_method='threshold'`` (default for Lasso prior)
        Uses P(|β_j| > ε | data) in the classical sense.  ``epsilon`` may be
        supplied explicitly; when it is ``None`` the value is inferred from the
        data scale:

        * linear model   → ε = std(y) / 10  (one-tenth of the response SD)
        * logistic model → ε = 0.1          (on the log-odds scale)

    Parameters
    ----------
    model_type : {'linear', 'logistic'}, default 'linear'
        Type of GLM likelihood.
    prior : {'horseshoe', 'lasso'}, default 'horseshoe'
        Shrinkage prior to place on the coefficients.
    pip_method : {'kappa', 'threshold', 'auto'}, default 'auto'
        PIP computation strategy (see above).  ``'auto'`` selects ``'kappa'``
        for ``prior='horseshoe'`` and ``'threshold'`` for ``prior='lasso'``.
    kappa_threshold : float, default 0.5
        Decision boundary for the shrinkage-factor PIP.  A draw contributes to
        inclusion when κ_j < ``kappa_threshold``.  Only used when
        ``pip_method='kappa'``.
    epsilon : float or None, default None
        Coefficient-magnitude threshold for the threshold-based PIP.  When
        ``None`` the value is chosen automatically from the data scale.  Only
        used when ``pip_method='threshold'``.
    pip_threshold : float, default 0.80
        Minimum PIP to classify a feature as selected.
    n_samples : int, default 1000
        Number of posterior draws per chain.
    tune : int, default 1000
        Number of tuning steps per chain.
    chains : int, default 2
        Number of MCMC chains.
    progressbar : bool, default False
        Whether to display a PyMC progress bar during sampling.
    random_state : int or None, default None
        Seed for reproducibility.
    """

    def __init__(
        self,
        model_type: str = "linear",
        prior: str = "horseshoe",
        pip_method: str = "auto",
        kappa_threshold: float = 0.5,
        epsilon: Optional[float] = None,
        pip_threshold: float = 0.80,
        n_samples: int = 1000,
        tune: int = 1000,
        chains: int = 2,
        progressbar: bool = False,
        random_state: Optional[int] = None,
    ):
        self.model_type = model_type
        self.prior = prior
        self.pip_method = pip_method
        self.kappa_threshold = kappa_threshold
        self.epsilon = epsilon
        self.pip_threshold = pip_threshold
        self.n_samples = n_samples
        self.tune = tune
        self.chains = chains
        self.progressbar = progressbar
        self.random_state = random_state

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate_hyperparams(self) -> None:
        """Raise ValueError for any hyperparameter combination that is statistically invalid."""
        if not (0.0 < self.kappa_threshold < 1.0):
            raise ValueError(
                f"kappa_threshold must be in (0, 1) because the shrinkage factor \u03ba_j is "
                f"bounded in (0, 1); got {self.kappa_threshold!r}"
            )
        if not (0.0 < self.pip_threshold < 1.0):
            raise ValueError(
                f"pip_threshold must be in (0, 1) because PIP is a probability; "
                f"got {self.pip_threshold!r}"
            )

    def _resolve_pip_method(self) -> str:
        """Return the effective PIP method given ``prior`` and ``pip_method``."""
        if self.pip_method == "auto":
            return "kappa" if self.prior == "horseshoe" else "threshold"
        if self.pip_method not in ("kappa", "threshold"):
            raise ValueError(
                f"pip_method must be 'kappa', 'threshold', or 'auto'; got {self.pip_method!r}"
            )
        return self.pip_method

    def _resolve_epsilon(self, y_arr: np.ndarray) -> float:
        """Return the effective epsilon for threshold-based PIP."""
        if self.epsilon is not None:
            return float(self.epsilon)
        # Auto-scale epsilon to the response
        if self.model_type == "linear":
            return float(np.std(y_arr)) / 10.0
        else:  # logistic — work on log-odds scale
            return 0.1

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X: Any, y: Any) -> "ShrinkagePIP":
        """Fit the Bayesian GLM using PyMC MCMC sampling.

        Parameters
        ----------
        X : array-like or pd.DataFrame
            Feature matrix.
        y : array-like
            Target vector (binary for logistic, continuous for linear).
        """
        self._validate_hyperparams()
        try:
            import pymc as pm
        except ImportError:
            raise ImportError(
                "pymc is required to use ShrinkagePIP. "
                "Install it using `pip install 'bxai[mcmc]'` or `uv sync --extra mcmc`."
            )

        X_arr, y_arr = check_consistent_length(X, y)
        n_samples, n_features = X_arr.shape

        if hasattr(X, "columns"):
            self.feature_names_ = list(X.columns)
        else:
            self.feature_names_ = [f"feature_{i}" for i in range(n_features)]

        effective_pip_method = self._resolve_pip_method()

        with pm.Model() as model:
            # Inputs
            X_shared = pm.Data("X", X_arr)
            
            # Intercept prior
            intercept = pm.Normal("intercept", mu=0.0, sigma=10.0)
            
            # Feature coefficient priors
            if self.prior == "horseshoe":
                # Global shrinkage
                tau = pm.HalfCauchy("tau", beta=1.0)
                # Local shrinkage
                lambdas = pm.HalfCauchy("lambdas", beta=1.0, shape=n_features)
                # Coefficients
                beta = pm.Normal("beta", mu=0.0, sigma=tau * lambdas, shape=n_features)
            elif self.prior == "lasso":
                # Lasso corresponds to a Laplace prior
                b = pm.HalfCauchy("b", beta=1.0)
                beta = pm.Laplace("beta", mu=0.0, b=b, shape=n_features)
            else:
                raise ValueError("prior must be 'horseshoe' or 'lasso'")

            # Linear prediction
            mu_pred = intercept + pm.math.dot(X_shared, beta)

            # Likelihood definition
            if self.model_type == "linear":
                sigma_y = pm.HalfNormal("sigma_y", sigma=5.0)
                pm.Normal("y_obs", mu=mu_pred, sigma=sigma_y, observed=y_arr)
            elif self.model_type == "logistic":
                p = pm.math.invlogit(mu_pred)
                pm.Bernoulli("y_obs", p=p, observed=y_arr)
            else:
                raise ValueError("model_type must be 'linear' or 'logistic'")

            # Draw samples
            self.trace_ = pm.sample(
                draws=self.n_samples,
                tune=self.tune,
                chains=self.chains,
                random_seed=self.random_state,
                progressbar=self.progressbar,
                return_inferencedata=True,
            )

        # ---- Extract posterior draws ----
        beta_draws = self.trace_.posterior["beta"].values
        # shape: (chains, draws, n_features)
        self.beta_flat_ = beta_draws.reshape(-1, n_features)

        # ---- Compute PIP using the chosen strategy ----
        if effective_pip_method == "kappa":
            if self.prior != "horseshoe":
                raise ValueError(
                    "pip_method='kappa' is only valid with prior='horseshoe'; "
                    "the shrinkage factor \u03ba is not defined for the Lasso prior."
                )
            tau_draws = self.trace_.posterior["tau"].values.reshape(-1)  # (total_draws,)
            lambda_draws = self.trace_.posterior["lambdas"].values.reshape(-1, n_features)  # (total_draws, p)

            # \u03ba_j = 1 / (1 + \u03bb_j^2 * \u03c4^2)  in (0, 1)
            # \u03ba_j -> 0: local scale dominates -> signal (not shrunk)
            # \u03ba_j -> 1: global shrinkage dominates -> noise (shrunk to zero)
            kappa = 1.0 / (
                1.0 + lambda_draws ** 2 * tau_draws[:, np.newaxis] ** 2
            )
            self.kappa_mean_ = np.mean(kappa, axis=0)
            # PIP = P(\u03ba_j < kappa_threshold | data)
            self.pip_ = np.mean(kappa < self.kappa_threshold, axis=0)
            self.epsilon_ = None  # not applicable

        else:  # threshold
            self.epsilon_ = self._resolve_epsilon(y_arr)
            # P(|\u03b2_j| > \u03b5 | data)
            self.pip_ = np.mean(np.abs(self.beta_flat_) > self.epsilon_, axis=0)
            self.kappa_mean_ = None  # not applicable

        self.coef_mean_ = np.mean(self.beta_flat_, axis=0)
        self.coef_std_ = np.std(self.beta_flat_, axis=0)
        self.pip_method_ = effective_pip_method

        # ---- Selection ----
        self.support_ = self.pip_ >= self.pip_threshold

        self.confirmed_ = [
            self.feature_names_[i] for i, s in enumerate(self.support_) if s
        ]
        self.rejected_ = [
            self.feature_names_[i] for i, s in enumerate(self.support_) if not s
        ]
        self.tentative_ = []  # Parametric shrinkage models have no tentative state

        return self

    def summary(self) -> pd.DataFrame:
        """Return a summary of feature coefficients, standard deviations, and PIPs."""
        lower = np.percentile(self.beta_flat_, 2.5, axis=0)
        upper = np.percentile(self.beta_flat_, 97.5, axis=0)

        data = []
        for i, name in enumerate(self.feature_names_):
            row: Dict[str, Any] = {
                "feature": name,
                "pip": self.pip_[i],
                "pip_method": self.pip_method_,
                "selected": bool(self.support_[i]),
                "mean": self.coef_mean_[i],
                "std": self.coef_std_[i],
                "hdi_lower": lower[i],
                "hdi_upper": upper[i],
            }
            if self.pip_method_ == "kappa":
                row["kappa_mean"] = self.kappa_mean_[i]
            else:
                row["epsilon"] = self.epsilon_
            data.append(row)
        return pd.DataFrame(data)
