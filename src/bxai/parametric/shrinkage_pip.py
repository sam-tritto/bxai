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
    Probability (PIP) for each feature. Features are selected if their PIP
    crosses a user-defined threshold.
    """

    def __init__(
        self,
        model_type: str = "linear",
        prior: str = "horseshoe",
        n_samples: int = 1000,
        tune: int = 1000,
        chains: int = 2,
        pip_threshold: float = 0.80,
        epsilon: float = 1e-3,
        progressbar: bool = False,
        random_state: Optional[int] = None,
    ):
        self.model_type = model_type
        self.prior = prior
        self.n_samples = n_samples
        self.tune = tune
        self.chains = chains
        self.pip_threshold = pip_threshold
        self.epsilon = epsilon
        self.progressbar = progressbar
        self.random_state = random_state

    def fit(self, X: Any, y: Any) -> "ShrinkagePIP":
        """Fit the Bayesian GLM using PyMC MCMC sampling.
        
        Parameters
        ----------
        X : array-like or pd.DataFrame
            Feature matrix.
        y : array-like
            Target vector (binary targets for logistic, continuous for linear).
        """
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

        # Extract posterior values
        beta_draws = self.trace_.posterior["beta"].values
        # shape: (chains, draws, n_features)
        self.beta_flat_ = beta_draws.reshape(-1, n_features)

        # Calculate Posterior Inclusion Probabilities (PIP)
        # P(|beta_j| > epsilon | data)
        self.pip_ = np.mean(np.abs(self.beta_flat_) > self.epsilon, axis=0)
        self.coef_mean_ = np.mean(self.beta_flat_, axis=0)
        self.coef_std_ = np.std(self.beta_flat_, axis=0)

        # Perform selection based on PIP threshold
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
        # Calculate HDI for coefficients
        lower = np.percentile(self.beta_flat_, 2.5, axis=0)
        upper = np.percentile(self.beta_flat_, 97.5, axis=0)

        data = []
        for i, name in enumerate(self.feature_names_):
            data.append({
                "feature": name,
                "pip": self.pip_[i],
                "selected": bool(self.support_[i]),
                "mean": self.coef_mean_[i],
                "std": self.coef_std_[i],
                "hdi_lower": lower[i],
                "hdi_upper": upper[i],
            })
        return pd.DataFrame(data)
