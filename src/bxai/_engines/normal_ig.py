import numpy as np
from scipy import stats
from typing import Tuple, Optional, Union
from bxai._utils.types import FeatureStatus


class NormalIGTracker:
    """Stateful conjugate Bayesian tracker using the Normal-Inverse-Gamma framework.
    
    Models continuous values (e.g. SHAP differences or loss differences):
    X ~ Normal(μ, σ^2)
    with prior (μ, σ^2) ~ NIG(μ_0, ν_0, α_0, β_0).
    
    The marginal posterior distribution of μ is a Student-t distribution:
    μ | D ~ t_{2α}(μ, β / (α * ν)).
    """

    def __init__(
        self,
        n_features: int,
        prior_mu: float = 0.0,
        prior_nu: float = 1e-4,
        prior_alpha: float = 1e-4,
        prior_beta: float = 1e-4,
    ):
        if prior_nu <= 0:
            raise ValueError(
                f"prior_nu must be > 0 for a valid Normal-Inverse-Gamma prior; got {prior_nu!r}"
            )
        if prior_alpha <= 0:
            raise ValueError(
                f"prior_alpha must be > 0 for a valid Normal-Inverse-Gamma prior; got {prior_alpha!r}"
            )
        if prior_beta <= 0:
            raise ValueError(
                f"prior_beta must be > 0 for a valid Normal-Inverse-Gamma prior; got {prior_beta!r}"
            )
        self.n_features = n_features

        # Initialize prior parameters
        self.mu = np.full(n_features, float(prior_mu))
        self.nu = np.full(n_features, float(prior_nu))
        self.alpha = np.full(n_features, float(prior_alpha))
        self.beta = np.full(n_features, float(prior_beta))

    def update(self, new_data: np.ndarray, indices: Optional[np.ndarray] = None) -> None:
        """Update posterior parameters with new data points.

        This method implements ``n_active`` *independent univariate* NIG updates,
        one per feature column.  Despite accepting a 2-D matrix, the model is
        **not** multivariate: each column j has its own ``(μ_j, σ²_j)`` posterior
        and is updated entirely from ``new_data[:, j]``.

        The i.i.d. assumption applies *within* each column: the ``n_samples``
        rows of ``new_data[:, j]`` are treated as i.i.d. draws from
        ``Normal(μ_j, σ²_j)``.  There is no assumption about the relationship
        between columns — each feature may have a completely different scale or
        distribution without affecting correctness.

        Parameters
        ----------
        new_data : np.ndarray
            1D array of shape ``(n_active,)`` or 2D array of shape
            ``(n_samples, n_active)``.  If 1D, it is treated as a single
            observation (n_samples=1) for each active feature.
        indices : Optional[np.ndarray], default=None
            Feature indices corresponding to the columns of ``new_data``.
            If ``None``, all ``n_features`` features are updated and
            ``n_active`` must equal ``n_features``.
        """
        data = np.asarray(new_data, dtype=float)
        if data.ndim == 1:
            data = data[np.newaxis, :]  # shape: (1, n_active)
            
        n_samples, n_active = data.shape
        
        if indices is None:
            if n_active != self.n_features:
                raise ValueError(f"Data columns ({n_active}) must match n_features ({self.n_features})")
            idx = np.arange(self.n_features)
        else:
            idx = np.asarray(indices, dtype=int)
            if n_active != len(idx):
                raise ValueError(f"Data columns ({n_active}) must match len(indices) ({len(idx)})")
                
        # Calculate statistics
        x_bar = np.mean(data, axis=0)
        ss = np.sum((data - x_bar) ** 2, axis=0)
        
        # In-place Bayesian conjugate updates
        nu_old = self.nu[idx]
        mu_old = self.mu[idx]
        
        nu_new = nu_old + n_samples
        mu_new = (nu_old * mu_old + n_samples * x_bar) / nu_new
        alpha_new = self.alpha[idx] + 0.5 * n_samples
        beta_new = (
            self.beta[idx]
            + 0.5 * ss
            + 0.5 * n_samples * nu_old * (x_bar - mu_old) ** 2 / nu_new
        )
        
        # Write back to state
        self.nu[idx] = nu_new
        self.mu[idx] = mu_new
        self.alpha[idx] = alpha_new
        self.beta[idx] = beta_new

    def credible_interval(self, credible_mass: float = 0.95) -> Tuple[np.ndarray, np.ndarray]:
        """Compute the Highest Density / Equal-Tailed Credible Interval bounds for the mean μ.

        Returns
        ----------
        lower, upper : np.ndarray
        """
        if not (0.0 < credible_mass < 1.0):
            raise ValueError(
                f"credible_mass must be in (0, 1); got {credible_mass!r}"
            )
        # Degrees of freedom: 2 * alpha
        df = 2.0 * self.alpha
        loc = self.mu
        # Scale: sqrt(beta / (alpha * nu))
        scale = np.sqrt(self.beta / (self.alpha * self.nu))
        
        lower, upper = stats.t.interval(credible_mass, df=df, loc=loc, scale=scale)
        return lower, upper

    def decide(self, credible_mass: float = 0.95, threshold: float = 0.0) -> np.ndarray:
        """Decide the status of each feature based on whether the HDI bounds zero.

        Parameters
        ----------
        credible_mass : float, default=0.95
            The credible mass (1 - alpha) of the interval.  Must be in (0, 1).
        threshold : float, default=0.0
            The reference value (typically 0.0). If the interval is completely above
            the threshold, the feature is CONFIRMED. If the interval is completely
            below the threshold, the feature is REJECTED. Otherwise, TENTATIVE.

        Returns
        ----------
        status : np.ndarray of FeatureStatus
        """
        if not (0.0 < credible_mass < 1.0):
            raise ValueError(
                f"credible_mass must be in (0, 1); got {credible_mass!r}"
            )
        lower, upper = self.credible_interval(credible_mass)
        status = np.full(self.n_features, FeatureStatus.TENTATIVE, dtype=object)
        
        # If the lower bound is above threshold, it's confirmed
        status[lower > threshold] = FeatureStatus.CONFIRMED
        # If the upper bound is below threshold, it's rejected
        status[upper < threshold] = FeatureStatus.REJECTED
        
        return status
