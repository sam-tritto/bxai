import warnings
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.feature_selection import SelectorMixin
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import check_is_fitted

from bxai._engines.beta_binomial import BetaBinomialTracker
from bxai._engines.normal_ig import NormalIGTracker
from bxai._utils.types import FeatureStatus
from bxai._utils.validation import check_consistent_length


def _default_model(y: np.ndarray) -> Any:
    """Instantiate a default LightGBM model based on target type."""
    try:
        import lightgbm as lgb
    except ImportError:
        raise ImportError(
            "lightgbm is required to use the default model in BayesianBorutaSHAP. "
            "Install it using `pip install 'bxai[boruta]'` or pass a custom fitted model."
        )
    target_type = type_of_target(y)
    if target_type in ("binary", "multiclass"):
        return lgb.LGBMClassifier(verbosity=-1)
    else:
        return lgb.LGBMRegressor(verbosity=-1)


def _extract_shap_importances(explainer: Any, X: np.ndarray) -> np.ndarray:
    """Extract feature importances as mean absolute SHAP values.

    Tries the modern ``explainer(X)`` API first, then falls back to the legacy
    ``explainer.shap_values(X)`` call.  Only ``NotImplementedError`` and
    ``TypeError`` are suppressed between the two attempts — these are the
    documented cases where a particular ``shap.Explainer`` subclass does not
    support one of the two call conventions.  All other exceptions (CUDA
    errors, memory errors, SHAP API changes, incompatible model types, …) are
    allowed to propagate so that the caller receives a meaningful traceback.

    Parameters
    ----------
    explainer : shap.Explainer
        A fitted SHAP explainer (e.g. ``shap.TreeExplainer``).
    X : np.ndarray
        The feature matrix to explain.

    Returns
    -------
    np.ndarray
        1-D array of mean absolute SHAP values, one entry per feature.

    Raises
    ------
    RuntimeError
        When both the modern and legacy SHAP call conventions fail, wrapping
        the original exception with a descriptive message.
    """
    shap_values: Any

    # --- Attempt 1: modern API -------------------------------------------
    try:
        explanation = explainer(X)
        shap_values = explanation.values
    except (NotImplementedError, TypeError) as _first_err:
        # These two exceptions are the only ones that legitimately indicate
        # "this explainer does not support the modern __call__ convention".
        # Everything else (MemoryError, AttributeError, RuntimeError from
        # CUDA, unexpected SHAP API changes, …) should propagate unchanged.
        first_err = _first_err

        # --- Attempt 2: legacy API ---------------------------------------
        try:
            shap_values = explainer.shap_values(X)
        except Exception as second_err:
            raise RuntimeError(
                "Both SHAP call conventions failed for explainer "
                f"{type(explainer).__name__!r}.\n"
                f"  Modern API error  : {type(first_err).__name__}: {first_err}\n"
                f"  Legacy API error  : {type(second_err).__name__}: {second_err}\n"
                "Check that the explainer and model types are compatible with "
                "the installed version of the shap library."
            ) from second_err

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


class BayesianBorutaSHAP(SelectorMixin, BaseEstimator):
    """Bayesian Boruta SHAP Feature Selection.

    Wrapper-selects features using tree-based models and SHAP values,
    swapping frequentist p-values for Bayesian credible intervals.
    Supports discrete (Beta-Binomial) and continuous (Normal-Inverse-Gamma) modes
    with dynamic pruning of confirmed/rejected features for massive speedups.

    .. rubric:: Design note — ``mode='continuous'`` convergence

    In continuous mode each Boruta iteration produces a single scalar
    ``diff = active_importance - max_shadow_importance`` per active feature.
    That scalar is passed to :class:`~bxai._engines.normal_ig.NormalIGTracker`
    as one observation, so the NIG posterior receives **exactly one data point
    per feature per iteration**.

    Consequences:

    * The posterior variance (σ²) is estimated from the sample variance of
      accumulated diffs.  With fewer than ~20–30 observations the sample
      variance is highly unstable, so the Student-t HDI will be very wide and
      decisions unreliable.
    * This is in contrast to ``mode='discrete'``, where a single Bernoulli
      hit/miss per iteration is sufficient for the Beta posterior to tighten
      relatively quickly.

    **Practical guidance**: set ``max_iter`` to at least 150–200 when using
    ``mode='continuous'``.  Fewer iterations may leave many features
    tentative because the HDI cannot exclude zero with sufficient
    ``credible_mass``.
    """

    def __init__(
        self,
        model: Any | None = None,
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
        random_state: int | None = None,
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

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate_hyperparams(self) -> None:
        """Raise ValueError for any hyperparameter combination that is statistically invalid."""
        if not (0.0 < self.credible_mass < 1.0):
            raise ValueError(
                f"credible_mass must be in (0, 1); got {self.credible_mass!r}"
            )
        if not (0.0 < self.confirm_threshold < 1.0):
            raise ValueError(
                f"confirm_threshold must be in (0, 1); got {self.confirm_threshold!r}"
            )
        if not (0.0 < self.reject_threshold < 1.0):
            raise ValueError(
                f"reject_threshold must be in (0, 1); got {self.reject_threshold!r}"
            )
        if self.confirm_threshold <= self.reject_threshold:
            raise ValueError(
                f"confirm_threshold ({self.confirm_threshold!r}) must be strictly greater than "
                f"reject_threshold ({self.reject_threshold!r}); otherwise no decision is possible."
            )
        if self.prior_alpha <= 0:
            raise ValueError(
                f"prior_alpha must be > 0 for a valid Beta distribution; got {self.prior_alpha!r}"
            )
        if self.prior_beta <= 0:
            raise ValueError(
                f"prior_beta must be > 0 for a valid Beta distribution; got {self.prior_beta!r}"
            )
        if self.prior_alpha_continuous <= 0:
            raise ValueError(
                f"prior_alpha_continuous must be > 0 for a valid NIG prior; "
                f"got {self.prior_alpha_continuous!r}"
            )
        if self.prior_beta_continuous <= 0:
            raise ValueError(
                f"prior_beta_continuous must be > 0 for a valid NIG prior; "
                f"got {self.prior_beta_continuous!r}"
            )
        if self.prior_nu <= 0:
            raise ValueError(
                f"prior_nu must be > 0 for a valid NIG prior; got {self.prior_nu!r}"
            )
        # ------------------------------------------------------------------
        # Cross-mode unused-parameter warnings
        # ------------------------------------------------------------------
        _DISCRETE_DEFAULTS = {"prior_alpha": 1.0, "prior_beta": 1.0}
        _CONTINUOUS_DEFAULTS = {
            "prior_mu": 0.0,
            "prior_nu": 1e-4,
            "prior_alpha_continuous": 1e-4,
            "prior_beta_continuous": 1e-4,
        }
        if self.mode == "continuous":
            ignored = [
                name
                for name, default in _DISCRETE_DEFAULTS.items()
                if getattr(self, name) != default
            ]
            if ignored:
                warnings.warn(
                    f"mode='continuous' uses the NIG tracker; the following discrete "
                    f"Beta-prior parameter(s) have no effect and will be ignored: "
                    f"{ignored!r}. "
                    f"Did you mean 'prior_alpha_continuous' / 'prior_beta_continuous'?",
                    UserWarning,
                    stacklevel=3,
                )
        elif self.mode == "discrete":
            ignored = [
                name
                for name, default in _CONTINUOUS_DEFAULTS.items()
                if getattr(self, name) != default
            ]
            if ignored:
                warnings.warn(
                    f"mode='discrete' uses the Beta-Binomial tracker; the following NIG "
                    f"prior parameter(s) have no effect and will be ignored: "
                    f"{ignored!r}. "
                    f"Did you mean 'prior_alpha' / 'prior_beta'?",
                    UserWarning,
                    stacklevel=3,
                )

    def fit(self, X: Any, y: Any) -> "BayesianBorutaSHAP":
        """Run the Bayesian Boruta SHAP loop on X and y.

        Parameters
        ----------
        X : array-like or pd.DataFrame
            Feature matrix.
        y : array-like
            Target vector.
        """
        try:
            import shap
        except ImportError:
            raise ImportError(
                "shap is required to use BayesianBorutaSHAP. "
                "Install it using `pip install 'bxai[boruta]'`."
            )
        self._validate_hyperparams()
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

        # Do NOT set random_state on base_model here.  Each iteration must
        # receive its own derived seed (see below); setting a fixed seed now
        # would cause every clone to use the same random state, neutralising
        # the stochasticity of the shadow-feature permutation step.

        # Initialize engines
        self.tracker_: BetaBinomialTracker | NormalIGTracker
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
        self.iteration_history_ = []

        # Loop until max iterations or all features decided
        for iteration in range(1, self.max_iter + 1):
            # Check tentative mask
            tentative_mask = self.status_ == FeatureStatus.TENTATIVE
            if not np.any(tentative_mask):
                break

            self.n_iterations_ = iteration

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

            # Fit the model — give each iteration its own derived seed so
            # the tree's internal randomness (feature sub-sampling, data
            # bootstrap) differs across iterations while still being
            # reproducible when self.random_state is set.
            model_it = clone(base_model)
            if hasattr(model_it, "random_state"):
                model_it.random_state = int(rng.integers(2**31 - 1))
            model_it.fit(X_combined, y_arr)

            # Compute SHAP values
            explainer = shap.TreeExplainer(model_it)
            importances = _extract_shap_importances(explainer, X_combined)

            # Partition active and shadow importances
            active_importances = importances[:n_active]
            shadow_importances = importances[n_active:]
            max_shadow = (
                np.max(shadow_importances) if len(shadow_importances) > 0 else 0.0
            )

            if self.mode == "discrete":
                hits = (active_importances > max_shadow).astype(float)
                assert isinstance(self.tracker_, BetaBinomialTracker)
                self.tracker_.update(hits, tentative_indices)
                self.status_ = self.tracker_.decide(
                    confirm_threshold=self.confirm_threshold,
                    reject_threshold=self.reject_threshold,
                )
            elif self.mode == "continuous":
                diffs = active_importances - max_shadow
                # NOTE: `diffs` is a 1-D array of length n_active.  The
                # tracker reshapes it to (1, n_active), so each iteration
                # contributes exactly ONE observation per feature to the NIG
                # posterior.  The NIG variance estimate (σ²) is derived from
                # the accumulated sample variance of these scalars; it is
                # highly unstable until ~20–30 observations have been seen.
                # As a result the Student-t HDI is very wide in early
                # iterations and reliable decisions require significantly more
                # iterations than discrete mode.  See the class docstring for
                # guidance on choosing `max_iter` in continuous mode.
                assert isinstance(self.tracker_, NormalIGTracker)
                self.tracker_.update(diffs, tentative_indices)
                self.status_ = self.tracker_.decide(
                    credible_mass=self.credible_mass,
                    threshold=0.0,
                )

            # Record history
            hist_entry = {
                "iteration": iteration,
                "status": self.status_.copy(),
            }
            if self.mode == "discrete":
                assert isinstance(self.tracker_, BetaBinomialTracker)
                hist_entry["alpha"] = self.tracker_.alpha.copy()
                hist_entry["beta"] = self.tracker_.beta.copy()
            else:
                assert isinstance(self.tracker_, NormalIGTracker)
                hist_entry["mu"] = self.tracker_.mu.copy()
                hist_entry["nu"] = self.tracker_.nu.copy()
                hist_entry["alpha"] = self.tracker_.alpha.copy()
                hist_entry["beta"] = self.tracker_.beta.copy()
            self.iteration_history_.append(hist_entry)

        # Compile final results
        self.confirmed_ = [
            self.feature_names_[i]
            for i, s in enumerate(self.status_)
            if s == FeatureStatus.CONFIRMED
        ]
        self.rejected_ = [
            self.feature_names_[i]
            for i, s in enumerate(self.status_)
            if s == FeatureStatus.REJECTED
        ]
        self.tentative_ = [
            self.feature_names_[i]
            for i, s in enumerate(self.status_)
            if s == FeatureStatus.TENTATIVE
        ]
        self.support_ = self.status_ == FeatureStatus.CONFIRMED

        if self.mode == "discrete":
            assert isinstance(self.tracker_, BetaBinomialTracker)
            self.feature_importances_ = self.tracker_.alpha / (
                self.tracker_.alpha + self.tracker_.beta
            )
        else:
            assert isinstance(self.tracker_, NormalIGTracker)
            self.feature_importances_ = self.tracker_.mu


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

    def summary(self, credible_mass: float | None = None) -> pd.DataFrame:
        """Return a summary of the features and their decisions.

        Parameters
        ----------
        credible_mass : float or None, optional
            The probability mass to include in the credible interval
            (must be in ``(0, 1)``).  When ``None`` (default), falls back to
            the ``credible_mass`` value supplied at construction time.

        Column notes
        ------------
        ``ci_lower`` / ``ci_upper`` (discrete mode)
            Equal-tailed interval from :meth:`BetaBinomialTracker.credible_interval`
            (``scipy.stats.beta.interval``).  The Beta posterior is skewed when
            α ≠ β, so the equal-tailed interval and the HDI differ — these
            columns are therefore **not** labelled as HDI.

        ``hdi_lower`` / ``hdi_upper`` (continuous mode)
            Equal-tailed interval from :meth:`NormalIGTracker.credible_interval`
            (``scipy.stats.t.interval``).  The marginal Student-t posterior
            for μ is symmetric, so the equal-tailed interval equals the HDI;
            the ``hdi_*`` label is therefore accurate here.
        """
        mass = credible_mass if credible_mass is not None else self.credible_mass
        lower, upper = self.tracker_.credible_interval(mass)

        data = []
        for i, name in enumerate(self.feature_names_):
            status = self.status_[i]

            if self.mode == "discrete":
                assert isinstance(self.tracker_, BetaBinomialTracker)
                alpha_val = self.tracker_.alpha[i]
                beta_val = self.tracker_.beta[i]
                # Beta distribution mean: alpha / (alpha + beta)
                mean_val = alpha_val / (alpha_val + beta_val)
                raw_params = {"alpha": alpha_val, "beta": beta_val}
                # Equal-tailed (not HDI) — Beta is skewed when α ≠ β
                interval_keys = {"ci_lower": lower[i], "ci_upper": upper[i]}
            else:
                assert isinstance(self.tracker_, NormalIGTracker)
                mean_val = self.tracker_.mu[i]
                raw_params = {
                    "mu": self.tracker_.mu[i],
                    "nu": self.tracker_.nu[i],
                    "alpha": self.tracker_.alpha[i],
                    "beta": self.tracker_.beta[i],
                }
                # Symmetric Student-t: equal-tailed == HDI
                interval_keys = {"hdi_lower": lower[i], "hdi_upper": upper[i]}

            data.append(
                {
                    "feature": name,
                    "status": status.value,
                    "mean": mean_val,
                    **interval_keys,
                    **raw_params,
                }
            )

        return pd.DataFrame(data)
