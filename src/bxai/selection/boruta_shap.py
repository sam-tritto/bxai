import numpy as np
import pandas as pd
import shap
from sklearn.base import BaseEstimator, clone
from sklearn.utils.multiclass import type_of_target
import lightgbm as lgb
from typing import Optional, List, Union, Any

from bxai._utils.types import FeatureStatus, FeaturePosterior
from bxai._utils.validation import check_consistent_length
from bxai._engines.beta_binomial import BetaBinomialTracker
from bxai._engines.normal_ig import NormalIGTracker


def _default_model(y: np.ndarray) -> Any:
    """Instantiate a default LightGBM model based on target type."""
    target_type = type_of_target(y)
    if target_type in ("binary", "multiclass"):
        return lgb.LGBMClassifier(verbosity=-1)
    else:
        return lgb.LGBMRegressor(verbosity=-1)


def _extract_shap_importances(explainer: shap.Explainer, X: np.ndarray) -> np.ndarray:
    """Extract feature importances as mean absolute SHAP values."""
    # Try using explainer(X) first as recommended by modern SHAP
    try:
        explanation = explainer(X)
        shap_values = explanation.values
    except Exception:
        # Fallback to legacy shap_values call
        try:
            shap_values = explainer.shap_values(X)
        except Exception as e:
            # Last fallback
            legacy_explainer = shap.Explainer(explainer.model, X)
            shap_values = legacy_explainer(X).values

    # Handle various output shapes/types of SHAP
    if isinstance(shap_values, list):
        if len(shap_values) == 2:
            # Binary classification: take SHAP of the positive class
            vals = shap_values[1]
        else:
            # Multiclass: take mean of absolute SHAP values across all classes
            vals = np.mean([np.abs(v) for v in shap_values], axis=0)
    else:
        vals = shap_values

    # Check for 3D array (e.g. from shap.Explainer on multiclass models)
    if isinstance(vals, np.ndarray) and vals.ndim == 3:
        vals = np.mean(np.abs(vals), axis=2)

    # Compute mean absolute SHAP value per feature
    # shape of vals: (n_samples, n_features)
    return np.mean(np.abs(vals), axis=0)


class BayesianBorutaSHAP(BaseEstimator):
    """Bayesian Boruta SHAP Feature Selection.
    
    Wrapper-selects features using tree-based models and SHAP values,
    swapping frequentist p-values for Bayesian credible intervals.
    Supports discrete (Beta-Binomial) and continuous (Normal-Inverse-Gamma) modes
    with dynamic pruning of confirmed/rejected features for massive speedups.
    """

    def __init__(
        self,
        model: Optional[Any] = None,
        mode: str = "discrete",
        max_iter: int = 100,
        credible_mass: float = 0.95,
        confirm_threshold: float = 0.95,
        reject_threshold: float = 0.05,
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0,
        prior_mu: float = 0.0,
        prior_nu: float = 1e-4,
        prior_alpha_continuous: float = 1e-4,
        prior_beta_continuous: float = 1e-4,
        random_state: Optional[int] = None,
    ):
        self.model = model
        self.mode = mode
        self.max_iter = max_iter
        self.credible_mass = credible_mass
        self.confirm_threshold = confirm_threshold
        self.reject_threshold = reject_threshold
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        self.prior_mu = prior_mu
        self.prior_nu = prior_nu
        self.prior_alpha_continuous = prior_alpha_continuous
        self.prior_beta_continuous = prior_beta_continuous
        self.random_state = random_state

    def fit(self, X: Any, y: Any) -> "BayesianBorutaSHAP":
        """Run the Bayesian Boruta SHAP loop on X and y.
        
        Parameters
        ----------
        X : array-like or pd.DataFrame
            Feature matrix.
        y : array-like
            Target vector.
        """
        X_arr, y_arr = check_consistent_length(X, y)
        n_samples, n_features = X_arr.shape

        if hasattr(X, "columns"):
            self.feature_names_ = list(X.columns)
        else:
            self.feature_names_ = [f"feature_{i}" for i in range(n_features)]

        # Set up RNG
        rng = np.random.default_rng(self.random_state)

        # Clone or initialize the model
        if self.model is None:
            base_model = _default_model(y_arr)
        else:
            base_model = clone(self.model)

        if hasattr(base_model, "random_state"):
            base_model.random_state = self.random_state

        # Initialize engines
        if self.mode == "discrete":
            self.tracker_ = BetaBinomialTracker(
                n_features=n_features,
                prior_alpha=self.prior_alpha,
                prior_beta=self.prior_beta,
            )
        elif self.mode == "continuous":
            self.tracker_ = NormalIGTracker(
                n_features=n_features,
                prior_mu=self.prior_mu,
                prior_nu=self.prior_nu,
                prior_alpha=self.prior_alpha_continuous,
                prior_beta=self.prior_beta_continuous,
            )
        else:
            raise ValueError("mode must be 'discrete' or 'continuous'")

        self.status_ = np.full(n_features, FeatureStatus.TENTATIVE, dtype=object)
        self.n_iterations_ = 0

        # Loop until max iterations or all features decided
        for iteration in range(1, self.max_iter + 1):
            self.n_iterations_ = iteration
            
            # Check tentative mask
            tentative_mask = self.status_ == FeatureStatus.TENTATIVE
            if not np.any(tentative_mask):
                break

            tentative_indices = np.where(tentative_mask)[0]
            n_active = len(tentative_indices)

            # Subset to tentative features
            X_active = X_arr[:, tentative_indices]

            # Generate shadow features
            X_shadow = np.empty_like(X_active)
            for j in range(n_active):
                X_shadow[:, j] = rng.permutation(X_active[:, j])

            # Concatenate features
            X_combined = np.hstack([X_active, X_shadow])

            # Fit the model
            model_it = clone(base_model)
            model_it.fit(X_combined, y_arr)

            # Compute SHAP values
            explainer = shap.TreeExplainer(model_it)
            importances = _extract_shap_importances(explainer, X_combined)

            # Partition active and shadow importances
            active_importances = importances[:n_active]
            shadow_importances = importances[n_active:]
            max_shadow = np.max(shadow_importances) if len(shadow_importances) > 0 else 0.0

            if self.mode == "discrete":
                hits = (active_importances > max_shadow).astype(float)
                self.tracker_.update(hits, tentative_indices)
                self.status_ = self.tracker_.decide(
                    confirm_threshold=self.confirm_threshold,
                    reject_threshold=self.reject_threshold,
                )
            elif self.mode == "continuous":
                diffs = active_importances - max_shadow
                self.tracker_.update(diffs, tentative_indices)
                self.status_ = self.tracker_.decide(
                    credible_mass=self.credible_mass,
                    threshold=0.0,
                )

        # Compile final results
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
        """Return a summary of the features and their decisions."""
        lower, upper = self.tracker_.credible_interval(self.credible_mass)
        
        data = []
        for i, name in enumerate(self.feature_names_):
            status = self.status_[i]
            
            if self.mode == "discrete":
                alpha_val = self.tracker_.alpha[i]
                beta_val = self.tracker_.beta[i]
                # Beta distribution mean: alpha / (alpha + beta)
                mean_val = alpha_val / (alpha_val + beta_val)
                raw_params = {"alpha": alpha_val, "beta": beta_val}
            else:
                mean_val = self.tracker_.mu[i]
                raw_params = {
                    "mu": self.tracker_.mu[i],
                    "nu": self.tracker_.nu[i],
                    "alpha": self.tracker_.alpha[i],
                    "beta": self.tracker_.beta[i],
                }
                
            data.append({
                "feature": name,
                "status": status.value,
                "mean": mean_val,
                "hdi_lower": lower[i],
                "hdi_upper": upper[i],
                **raw_params
            })
            
        return pd.DataFrame(data)
