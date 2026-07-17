"""Highest Density Interval computation utility.

ArviZ availability is probed **once at import time** and stored in
``ARVIZ_AVAILABLE``.  All subsequent calls to :func:`compute_hdi` reuse the
cached result — no repeated ``import arviz`` inside a hot loop.

The public interface is::

    lower, upper = compute_hdi(draws, credible_mass)
    label = HDI_LABEL  # "hdi" if ArviZ is present, "ci" otherwise

This 2-tuple return shape is consistent with
:meth:`BetaBinomialTracker.credible_interval` and
:meth:`NormalIGTracker.credible_interval`, eliminating the need for callers to
unpack differently depending on whether they arrived via the draw-based or the
conjugate-formula path.

Callers that need to label a summary column (e.g. ``"hdi_lower"`` vs
``"ci_lower"``) should read :data:`HDI_LABEL` once at module load time rather
than inspecting a third return value on every call.
"""
from __future__ import annotations

import numpy as np
from typing import Tuple

# ---------------------------------------------------------------------------
# Module-level ArviZ availability probe (runs exactly once per interpreter)
# ---------------------------------------------------------------------------
try:
    import arviz as _az  # noqa: F401  — kept as a reference in this scope
    ARVIZ_AVAILABLE: bool = True
except ImportError:
    ARVIZ_AVAILABLE: bool = False

#: String label for the interval type produced by :func:`compute_hdi`.
#: ``"hdi"`` when ArviZ is installed (true Highest Density Interval),
#: ``"ci"`` when falling back to an equal-tailed percentile interval.
HDI_LABEL: str = "hdi" if ARVIZ_AVAILABLE else "ci"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_hdi(
    draws: np.ndarray,
    credible_mass: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute a credible interval from posterior draws.

    Uses ArviZ's true Highest Density Interval (HDI) when the package is
    available; falls back to an equal-tailed percentile interval otherwise.
    The interval type can be determined by inspecting :data:`HDI_LABEL`.

    Parameters
    ----------
    draws : np.ndarray
        2D array of shape ``(n_draws, n_features)``.
    credible_mass : float
        Probability mass to cover; must be in ``(0, 1)``.

    Returns
    -------
    lower, upper : np.ndarray
        Lower and upper bounds, each of shape ``(n_features,)``.

    Notes
    -----
    The return shape is intentionally a plain 2-tuple, matching the contract of
    :meth:`BetaBinomialTracker.credible_interval` and
    :meth:`NormalIGTracker.credible_interval`.  The interval label is
    deliberately *not* returned here; read :data:`HDI_LABEL` instead.
    """
    if ARVIZ_AVAILABLE:
        import arviz as az
        # Pass as 3D (1 chain, n_draws, n_features) to avoid the ArviZ
        # FutureWarning about ambiguous (draws, shape) input.
        draws_3d = draws[np.newaxis, ...]  # (1, n_draws, n_features)
        hdi_result = az.hdi(draws_3d, hdi_prob=credible_mass)
        return hdi_result[:, 0], hdi_result[:, 1]

    # Fallback: equal-tailed percentile interval
    q_lower = (1.0 - credible_mass) / 2.0
    q_upper = 1.0 - q_lower
    lower = np.percentile(draws, q_lower * 100, axis=0)
    upper = np.percentile(draws, q_upper * 100, axis=0)
    return lower, upper
