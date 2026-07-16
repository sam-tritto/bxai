import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import get_scorer
from typing import Optional, Union, Callable, Any

from bxai._utils.types import FeatureStatus, FeaturePosterior
from bxai._utils.validation import check_consistent_length, check_is_fitted
from bxai._engines.normal_ig import NormalIGTracker


class BayesianPermutation(BaseEstimator):
    """Bayesian Permutation Feature Importance.
    
    A model-agnostic feature selection tool that computes a feature's actual
    impact on validation performance by comparing baseline score/loss versus
    shuffled feature score/loss over multiple permutation trials, updating
    a Normal-Inverse-Gamma continuous tracker.
    """

    def __init__(
        self,
        model: Any,
        scoring: Union[str, Callable],
        n_repeats: int = 30,
        credible_mass: float = 0.95,
        prior_mu: float = 0.0,
        prior_nu: float = 1e-4,
        prior_alpha: float = 1e-4,
        prior_beta: float = 1e-4,
        random_state: Optional[int] = None,
    ):
        self.model = model
        self.scoring = scoring
        self.n_repeats = n_repeats
        self.credible_mass = credible_mass
        self.prior_mu = prior_mu
        self.prior_nu = prior_nu
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        self.random_state = random_state

    def fit(self, X: Any, y: Any) -> "BayesianPermutation":
        """Compute the Bayesian Permutation Importance of features in X.
        
        Parameters
        ----------
        X : array-like or pd.DataFrame
            Validation feature matrix.
        y : array-like
            Validation target vector.
        """
        check_is_fitted(self.model)
        X_arr, y_arr = check_consistent_length(X, y)
        n_samples, n_features = X_arr.shape

        if hasattr(X, "columns"):
            self.feature_names_ = list(X.columns)
        else:
            self.feature_names_ = [f"feature_{i}" for i in range(n_features)]

        # Set up RNG
        rng = np.random.default_rng(self.random_state)

        # Resolve scorer
        if isinstance(self.scoring, str):
            scorer = get_scorer(self.scoring)
        else:
            scorer = self.scoring

        # Compute baseline score (sklearn scorers compute higher values for better performance)
        baseline_score = scorer(self.model, X_arr, y_arr)

        # Initialize tracker
        self.tracker_ = NormalIGTracker(
            n_features=n_features,
            prior_mu=self.prior_mu,
            prior_nu=self.prior_nu,
            prior_alpha=self.prior_alpha,
            prior_beta=self.prior_beta,
        )

        # Record deltas
        deltas = np.zeros((self.n_repeats, n_features))

        # Perform permutation repeats
        for r in range(self.n_repeats):
            X_temp = X_arr.copy()
            for col_idx in range(n_features):
                # Shuffle column
                original_col = X_temp[:, col_idx].copy()
                X_temp[:, col_idx] = rng.permutation(original_col)
                
                # Compute shuffled score
                shuffled_score = scorer(self.model, X_temp, y_arr)
                
                # Drop in score is baseline - shuffled (higher positive means more important)
                deltas[r, col_idx] = baseline_score - shuffled_score
                
                # Restore column
                X_temp[:, col_idx] = original_col

        # Update tracker with all deltas at once
        self.tracker_.update(deltas)

        # Make decisions
        self.status_ = self.tracker_.decide(
            credible_mass=self.credible_mass,
            threshold=0.0,
        )

        # Compile results
        self.confirmed_ = [
            self.feature_names_[i] for i, s in enumerate(self.status_) if s == FeatureStatus.CONFIRMED
        ]
        self.rejected_ = [
            self.feature_names_[i] for i, s in enumerate(self.status_) if s == FeatureStatus.REJECTED
        ]
        self.tentative_ = [
            self.feature_names_[i] for i, s in enumerate(self.status_) if s == FeatureStatus.TENTATIVE
        ]
        self.support_ = self.status_ == FeatureStatus.CONFIRMED

        return self

    def summary(self) -> pd.DataFrame:
        """Return a summary of the features and their permutation decisions."""
        lower, upper = self.tracker_.credible_interval(self.credible_mass)
        
        data = []
        for i, name in enumerate(self.feature_names_):
            status = self.status_[i]
            data.append({
                "feature": name,
                "status": status.value,
                "mean": self.tracker_.mu[i],
                "hdi_lower": lower[i],
                "hdi_upper": upper[i],
                "nu": self.tracker_.nu[i],
                "alpha": self.tracker_.alpha[i],
                "beta": self.tracker_.beta[i],
            })
            
        return pd.DataFrame(data)
