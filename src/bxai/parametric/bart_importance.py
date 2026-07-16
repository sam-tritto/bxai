import base64
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from typing import Optional, List, Dict, Tuple, Union, Any

from bxai._utils.validation import check_consistent_length


def _decode_vi(s: str, length: int) -> List[int]:
    """Decode base64 string variable inclusion data back to split counts vector."""
    try:
        data = base64.b64decode(s)
    except Exception:
        return [0] * length
        
    result: List[int] = []
    i = 0
    while len(result) < length and i < len(data):
        num = 0
        shift = 0
        while i < len(data):
            byte = data[i]
            i += 1
            num |= (byte & 0x7F) << shift
            if not (byte & 0x80):
                break
            shift += 7
        result.append(num)
    if len(result) < length:
        result.extend([0] * (length - len(result)))
    return result[:length]


class BARTImportance(BaseEstimator):
    """BART-based feature importance and selection.
    
    Fits a Bayesian Additive Regression Trees (BART) model using PyMC and
    pymc-bart, extracts the posterior distribution of Variable Inclusion Frequencies
    (VIF) by decoding tree split counts, and selects features whose lower HDI
    boundary exceeds a baseline random-chance frequency.
    """

    def __init__(
        self,
        n_trees: int = 50,
        n_samples: int = 1000,
        tune: int = 1000,
        chains: int = 2,
        credible_mass: float = 0.95,
        baseline_frequency: Optional[float] = None,
        progressbar: bool = False,
        random_state: Optional[int] = None,
    ):
        self.n_trees = n_trees
        self.n_samples = n_samples
        self.tune = tune
        self.chains = chains
        self.credible_mass = credible_mass
        self.baseline_frequency = baseline_frequency
        self.progressbar = progressbar
        self.random_state = random_state

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate_hyperparams(self) -> None:
        """Raise ValueError for any hyperparameter combination that is statistically invalid."""
        if not (0.0 < self.credible_mass < 1.0):
            raise ValueError(
                f"credible_mass must be in (0, 1); got {self.credible_mass!r}"
            )

    def fit(self, X: Any, y: Any) -> "BARTImportance":
        """Fit the BART model using PyMC.
        
        Parameters
        ----------
        X : array-like or pd.DataFrame
            Feature matrix.
        y : array-like
            Continuous target vector.
        """
        self._validate_hyperparams()
        try:
            import pymc as pm
            import pymc_bart as pmb
        except ImportError:
            raise ImportError(
                "pymc and pymc-bart are required to use BARTImportance. "
                "Install them using `pip install 'bxai[mcmc]'` or `uv sync --extra mcmc`."
            )

        X_arr, y_arr = check_consistent_length(X, y)
        n_samples, n_features = X_arr.shape

        if hasattr(X, "columns"):
            self.feature_names_ = list(X.columns)
        else:
            self.feature_names_ = [f"feature_{i}" for i in range(n_features)]

        # Determine baseline frequency threshold (default is 1.0 / n_features)
        if self.baseline_frequency is None:
            self.baseline_threshold_ = 1.0 / n_features
        else:
            self.baseline_threshold_ = self.baseline_frequency

        with pm.Model() as model:
            # Fit BART model
            mu = pmb.BART("mu", X_arr, y_arr, m=self.n_trees)
            sigma = pm.HalfNormal("sigma", sigma=np.std(y_arr) if len(y_arr) > 0 else 1.0)
            pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y_arr)

            self.trace_ = pm.sample(
                draws=self.n_samples,
                tune=self.tune,
                chains=self.chains,
                random_seed=self.random_state,
                progressbar=self.progressbar,
                return_inferencedata=True,
            )

        # Extract variable inclusion from sample stats
        vi_xarray = self.trace_.sample_stats["variable_inclusion"]
        vi_vals = vi_xarray.values.ravel()

        # Decode variable inclusion values to split counts
        # shape: (n_draws * n_chains, n_features)
        self.vif_raw_ = np.array([_decode_vi(val, n_features) for val in vi_vals])

        # Normalize split frequencies per draw to compute VIF distribution
        row_sums = self.vif_raw_.sum(axis=1, keepdims=True)
        # Avoid division by zero
        row_sums[row_sums == 0.0] = 1.0
        self.vif_distribution_ = self.vif_raw_ / row_sums

        # Compute posterior statistics
        self.vif_mean_ = np.mean(self.vif_distribution_, axis=0)
        self.vif_std_ = np.std(self.vif_distribution_, axis=0)

        # Compute Highest Density Interval (HDI) boundaries per feature
        q_lower = (1.0 - self.credible_mass) / 2.0
        q_upper = 1.0 - q_lower
        self.hdi_lower_ = np.percentile(self.vif_distribution_, q_lower * 100, axis=0)
        self.hdi_upper_ = np.percentile(self.vif_distribution_, q_upper * 100, axis=0)

        # Select features whose lower HDI exceeds baseline random-chance frequency
        self.support_ = self.hdi_lower_ > self.baseline_threshold_

        self.confirmed_ = [
            self.feature_names_[i] for i, s in enumerate(self.support_) if s
        ]
        self.rejected_ = [
            self.feature_names_[i] for i, s in enumerate(self.support_) if not s
        ]
        self.tentative_ = []

        return self

    def summary(self) -> pd.DataFrame:
        """Return a summary of Variable Inclusion Frequencies (VIFs) and selection decisions."""
        data = []
        for i, name in enumerate(self.feature_names_):
            data.append({
                "feature": name,
                "selected": bool(self.support_[i]),
                "vif_mean": self.vif_mean_[i],
                "vif_std": self.vif_std_[i],
                "hdi_lower": self.hdi_lower_[i],
                "hdi_upper": self.hdi_upper_[i],
                "baseline_threshold": self.baseline_threshold_,
            })
        return pd.DataFrame(data)
