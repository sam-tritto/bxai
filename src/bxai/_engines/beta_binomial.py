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
        if prior_alpha <= 0:
            raise ValueError(
                f"prior_alpha must be > 0 for a valid Beta distribution; got {prior_alpha!r}"
            )
        if prior_beta <= 0:
            raise ValueError(
                f"prior_beta must be > 0 for a valid Beta distribution; got {prior_beta!r}"
            )
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
        """Compute the Equal-Tailed Credible Interval (ETI) bounds.

        .. note::
            ``scipy.stats.beta.interval`` computes an *equal-tailed* interval
            (i.e. ``(1-credible_mass)/2`` probability mass in each tail).  For
            a symmetric distribution this coincides with the HDI, but the Beta
            distribution is skewed whenever α ≠ β, so the ETI and HDI differ.
            The summary columns produced from this interval are therefore
            labelled ``ci_lower`` / ``ci_upper`` — **not** ``hdi_lower`` /
            ``hdi_upper`` — to reflect their true statistical nature.
        """
        if not (0.0 < credible_mass < 1.0):
            raise ValueError(
                f"credible_mass must be in (0, 1); got {credible_mass!r}"
            )
        lower, upper = stats.beta.interval(credible_mass, self.alpha, self.beta)
        return lower, upper

    def decide(
        self,
        confirm_threshold: float = 0.95,
        reject_threshold: float = 0.05,
        threshold: float = 0.5,
    ) -> np.ndarray:
        """Decide the status of each feature.

        Parameters
        ----------
        confirm_threshold : float
            Probability above which a feature is CONFIRMED.  Must be in (0, 1)
            and strictly greater than ``reject_threshold``.
        reject_threshold : float
            Probability below which a feature is REJECTED.  Must be in (0, 1)
            and strictly less than ``confirm_threshold``.
        threshold : float
            The exceedance target θ (default 0.5 for the median).

        Returns
        ----------
        status : np.ndarray of FeatureStatus
        """
        if not (0.0 < confirm_threshold < 1.0):
            raise ValueError(
                f"confirm_threshold must be in (0, 1); got {confirm_threshold!r}"
            )
        if not (0.0 < reject_threshold < 1.0):
            raise ValueError(
                f"reject_threshold must be in (0, 1); got {reject_threshold!r}"
            )
        if confirm_threshold <= reject_threshold:
            raise ValueError(
                f"confirm_threshold ({confirm_threshold!r}) must be strictly greater than "
                f"reject_threshold ({reject_threshold!r}); otherwise no decision is possible."
            )
        prob = self.exceedance_probability(threshold)
        status = np.full(self.n_features, FeatureStatus.TENTATIVE, dtype=object)

        status[prob >= confirm_threshold] = FeatureStatus.CONFIRMED
        status[prob <= reject_threshold] = FeatureStatus.REJECTED

        return status
