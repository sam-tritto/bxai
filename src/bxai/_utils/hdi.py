import numpy as np
from typing import Tuple

def compute_hdi(draws: np.ndarray, credible_mass: float) -> Tuple[np.ndarray, np.ndarray, str]:
    """Compute HDI from posterior draws.

    Tries to use ArviZ for proper HDI calculation. Falls back to
    equal-tailed percentile intervals if ArviZ is not available.

    Parameters
    ----------
    draws : np.ndarray
        2D array of shape (n_draws, n_features).
    credible_mass : float
        The probability mass to cover.

    Returns
    -------
    lower, upper : np.ndarray
        Lower and upper bounds of shape (n_features,).
    interval_type : str
        Either "hdi" (Highest Density Interval) or "ci" (credible interval).
    """
    try:
        import arviz as az
        # Pass as 3D (1 chain, n_draws, n_features) -- unambiguous in current and
        # future ArviZ versions, avoiding the (draw, shape) FutureWarning.
        draws_3d = draws[np.newaxis, ...]  # shape: (1, n_draws, n_features)
        hdi_result = az.hdi(draws_3d, hdi_prob=credible_mass)
        return hdi_result[:, 0], hdi_result[:, 1], "hdi"
    except ImportError:
        pass

    # Fallback: equal-tailed interval
    q_lower = (1.0 - credible_mass) / 2.0
    q_upper = 1.0 - q_lower
    lower = np.percentile(draws, q_lower * 100, axis=0)
    upper = np.percentile(draws, q_upper * 100, axis=0)
    return lower, upper, "ci"
