import base64
import binascii
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.feature_selection import SelectorMixin
from sklearn.utils.validation import check_is_fitted

from bxai._utils.hdi import HDI_LABEL, compute_hdi
from bxai._utils.validation import check_consistent_length


def _decode_vi(s: str, length: int) -> list[int]:
    """Decode a base64-encoded variable-inclusion string to a split-count vector.

    Parameters
    ----------
    s : str
        Base64-encoded payload produced by pymc-bart for one MCMC draw.
    length : int
        Expected number of features (= expected vector length).

    Returns
    -------
    List[int]
        Decoded split counts, one per feature.  If the decoded byte stream is
        shorter than *length* the remaining entries are zero-padded (legitimate
        for draws where some features had zero splits).

    Raises
    ------
    ValueError
        If *s* is not valid base64.  A corrupt or empty VIF entry must not
        silently return a zero vector — that would bias ``vif_distribution_``
        downward for the affected draw.  The caller is responsible for deciding
        whether to skip the draw or surface the error.
    """
    try:
        data = base64.b64decode(s)
    except binascii.Error as exc:
        raise ValueError(
            f"_decode_vi: failed to decode base64 VIF entry "
            f"(raw value={s!r}, length={length}): {exc}"
        ) from exc

    result: list[int] = []
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


class BARTImportance(SelectorMixin, BaseEstimator):
    """BART-based feature importance and selection.

    Fits a Bayesian Additive Regression Trees (BART) model using PyMC and
    pymc-bart, extracts the posterior distribution of Variable Inclusion
    Frequencies (VIF) by decoding tree split counts, and selects features
    whose lower HDI boundary exceeds a baseline random-chance frequency.

    Interval computation
    --------------------
    The VIF distribution for important features is typically right-skewed
    (many draws near ``1/n_features``, rare large values), so the equal-tailed
    percentile interval and the true Highest Density Interval (HDI) can differ
    meaningfully.  This class therefore uses :func:`arviz.hdi` — available as a
    transitive dependency of PyMC — to compute the *true* HDI.  The fitted
    attributes ``hdi_lower_`` and ``hdi_upper_`` store the genuine HDI
    boundaries; the :meth:`summary` columns are labelled ``hdi_lower`` /
    ``hdi_upper`` accordingly.
    """

    def __init__(
        self,
        model_type: str = "regression",
        n_trees: int = 50,
        n_samples: int = 1000,
        tune: int = 1000,
        chains: int = 2,
        credible_mass: float = 0.95,
        baseline_frequency: float | None = None,
        progressbar: bool = False,
        random_state: int | None = None,
    ):
        self.model_type = model_type
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
        if self.model_type not in ("regression", "classification"):
            raise ValueError(
                f"model_type must be either 'regression' or 'classification'; got {self.model_type!r}"
            )
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

        with pm.Model():
            # Fit BART model
            if self.model_type == "regression":
                mu = pmb.BART("mu", X_arr, y_arr, m=self.n_trees)
                sigma = pm.HalfNormal(
                    "sigma", sigma=np.std(y_arr) if len(y_arr) > 0 else 1.0
                )
                pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y_arr)
            elif self.model_type == "classification":
                from sklearn.utils.multiclass import type_of_target

                target_type = type_of_target(y_arr)
                if target_type != "binary":
                    raise ValueError(
                        f"BARTImportance in classification mode requires a binary target; "
                        f"got target type {target_type!r}"
                    )
                # Map binary targets to 0 and 1
                unique_y = np.unique(y_arr)
                if not set(unique_y).issubset({0, 1}):
                    y_arr = (y_arr == unique_y[1]).astype(int)

                mu = pmb.BART("mu", X_arr, y_arr, m=self.n_trees)
                p = pm.Deterministic("p", pm.math.invprobit(mu))
                pm.Bernoulli("y_obs", p=p, observed=y_arr)
            else:
                raise ValueError(
                    f"model_type must be either 'regression' or 'classification'; got {self.model_type!r}"
                )

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

        # Decode variable inclusion values to split counts.
        # shape: (n_draws * n_chains, n_features)
        #
        # _decode_vi raises ValueError for corrupt / non-base64 entries rather
        # than silently returning a zero vector (which would bias
        # vif_distribution_ downward for the affected draw).  We re-raise with
        # the draw index so the user knows exactly which MCMC sample is corrupt.
        decoded: list[list[int]] = []
        for draw_idx, val in enumerate(vi_vals):
            try:
                decoded.append(_decode_vi(val, n_features))
            except ValueError as exc:
                raise ValueError(
                    f"BARTImportance.fit: corrupt variable-inclusion entry at "
                    f"draw index {draw_idx} (of {len(vi_vals)} total draws). "
                    f"Original error: {exc}"
                ) from exc
        self.vif_raw_ = np.array(decoded)

        # Normalize split frequencies per draw to compute VIF distribution
        row_sums = self.vif_raw_.sum(axis=1, keepdims=True)
        # Avoid division by zero
        row_sums[row_sums == 0.0] = 1.0
        self.vif_distribution_ = self.vif_raw_ / row_sums

        # Compute posterior statistics
        self.vif_mean_ = np.mean(self.vif_distribution_, axis=0)
        self.vif_std_ = np.std(self.vif_distribution_, axis=0)

        # Compute the true Highest Density Interval (HDI) per feature.
        # VIF distributions are right-skewed, so the equal-tailed percentile
        # interval and the HDI can differ meaningfully.
        self.hdi_lower_, self.hdi_upper_ = compute_hdi(
            self.vif_distribution_, self.credible_mass
        )
        self._interval_label = HDI_LABEL

        # Select features whose lower HDI exceeds baseline random-chance frequency
        self.support_ = self.hdi_lower_ > self.baseline_threshold_

        self.confirmed_ = [
            self.feature_names_[i] for i, s in enumerate(self.support_) if s
        ]
        self.rejected_ = [
            self.feature_names_[i] for i, s in enumerate(self.support_) if not s
        ]
        self.tentative_: list[str] = []

        self.feature_importances_ = self.vif_mean_

        return self

    # ------------------------------------------------------------------
    # sklearn SelectorMixin interface
    # ------------------------------------------------------------------

    def get_support(self, indices: bool = False):
        """Return a boolean mask or integer indices of the selected features.

        Parameters
        ----------
        indices : bool, default False
            If True, return integer indices rather than a boolean mask.

        Returns
        -------
        support : np.ndarray of shape (n_features,)
            Boolean mask, or integer indices when *indices* is True.
        """
        check_is_fitted(self, "support_")
        if indices:
            return np.where(self.support_)[0]
        return self.support_

    def _get_support_mask(self) -> np.ndarray:
        """Required by SelectorMixin to power transform() / inverse_transform()."""
        return self.get_support()

    def summary(self) -> pd.DataFrame:
        """Return a summary of Variable Inclusion Frequencies (VIFs) and selection decisions.

        The ``hdi_lower`` / ``hdi_upper`` columns contain the true Highest
        Density Interval computed via :func:`arviz.hdi`.  In the rare case
        where ArviZ is unavailable they fall back to an equal-tailed credible
        interval, indicated by the ``interval_type`` column being ``'ci'``
        rather than ``'hdi'``.
        """
        label = getattr(self, "_interval_label", "hdi")
        data = []
        for i, name in enumerate(self.feature_names_):
            data.append(
                {
                    "feature": name,
                    "selected": bool(self.support_[i]),
                    "vif_mean": self.vif_mean_[i],
                    "vif_std": self.vif_std_[i],
                    "hdi_lower": self.hdi_lower_[i],
                    "hdi_upper": self.hdi_upper_[i],
                    "baseline_threshold": self.baseline_threshold_,
                    "interval_type": label,
                }
            )
        return pd.DataFrame(data)
