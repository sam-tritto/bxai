import numpy as np
from scipy import stats
from typing import Tuple, Optional
from bxai._utils.types import FeatureStatus


class BetaBinomialTracker:
    """Stateful conjugate Bayesian tracker using the Beta-Binomial framework.
    
    Models a binary outcome: θ_j ~ Beta(α_j, β_j) where θ_j is the probability
    that feature j outperforms the maximum shadow feature in a Boruta iteration.
    """

    def __init__(
        self,
        n_features: int,
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0,
    ):
        self.n_features = n_features
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        
        # Initialize posterior parameters
        self.alpha = np.full(n_features, float(prior_alpha))
        self.beta = np.full(n_features, float(prior_beta))

    def update(self, hits: np.ndarray, indices: Optional[np.ndarray] = None) -> None:
        """Update the posterior parameters for the specified feature indices.
        
        Parameters
        ----------
        hits : np.ndarray
            Binary indicators (1 for hit, 0 for miss) of shape (len(indices),) or (n_features,).
        indices : Optional[np.ndarray], default=None
            The feature indices that were active. If None, assumes all features are updated.
        """
        hits = np.asarray(hits, dtype=float)
        if indices is None:
            if len(hits) != self.n_features:
                raise ValueError(f"Expected hits of length {self.n_features}, got {len(hits)}")
            self.alpha += hits
            self.beta += (1.0 - hits)
        else:
            indices = np.asarray(indices, dtype=int)
            if len(hits) != len(indices):
                raise ValueError(f"Length of hits ({len(hits)}) must match length of indices ({len(indices)})")
            self.alpha[indices] += hits
            self.beta[indices] += (1.0 - hits)

    def exceedance_probability(self, threshold: float = 0.5) -> np.ndarray:
        """Compute P(θ_j > threshold | data) for each feature."""
        # Using the survival function (sf = 1 - cdf) for precision
        return stats.beta.sf(threshold, self.alpha, self.beta)

    def credible_interval(self, credible_mass: float = 0.95) -> Tuple[np.ndarray, np.ndarray]:
        """Compute the Highest Density / Equal-Tailed Credible Interval bounds."""
        lower, upper = stats.beta.interval(credible_mass, self.alpha, self.beta)
        return lower, upper

    def decide(
        self,
        confirm_threshold: float = 0.95,
        reject_threshold: float = 0.05,
        threshold: float = 0.5,
    ) -> np.ndarray:
        """Decide the status of each feature.
        
        Returns
        ----------
        status : np.ndarray of FeatureStatus
        """
        prob = self.exceedance_probability(threshold)
        status = np.full(self.n_features, FeatureStatus.TENTATIVE, dtype=object)
        
        status[prob >= confirm_threshold] = FeatureStatus.CONFIRMED
        status[prob <= reject_threshold] = FeatureStatus.REJECTED
        
        return status
