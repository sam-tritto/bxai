import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.feature_selection import SelectorMixin
from sklearn.metrics import get_scorer
from typing import Optional, Union, Callable, Any

from bxai._utils.types import FeatureStatus
from sklearn.utils.validation import check_is_fitted
from bxai._utils.validation import check_consistent_length
from bxai._engines.normal_ig import NormalIGTracker


class BayesianPermutation(SelectorMixin, BaseEstimator):
    """Bayesian Permutation Feature Importance.

    A model-agnostic feature selection tool that computes a feature's actual
    impact on validation performance by comparing baseline score/loss versus
    shuffled-feature score/loss over multiple permutation trials, updating
    a Normal-Inverse-Gamma (NIG) continuous tracker.

    Task-type support
    -----------------
    ``BayesianPermutation`` is task-agnostic: it works with any fitted
    scikit-learn estimator and any compatible scorer.  The table below lists
    typical combinations; the only hard requirement is that the *scorer*,
    the *model*, and the *target* ``y`` are mutually compatible.

    ==================  ============================================
    Task                Example ``scoring`` values
    ==================  ============================================
    Regression          ``'r2'``, ``'neg_mean_squared_error'``
    Binary classif.     ``'roc_auc'``, ``'f1'``, ``'accuracy'``
    Multiclass          ``'accuracy'``, ``'f1_macro'``,
                        ``'roc_auc_ovr'``
    ==================  ============================================

    Scorer/task mismatch
    --------------------
    If a scorer that requires a binary target (e.g. ``'roc_auc'``) is passed
    together with a multiclass ``y``, scikit-learn's scorer will raise a
    ``ValueError`` at the first scoring call.  ``BayesianPermutation`` does
    not duplicate that check — the error from scikit-learn is descriptive
    enough to diagnose the mismatch.  Use ``'roc_auc_ovr'`` or
    ``'roc_auc_ovo'`` for multiclass problems when an AUC-based metric is
    desired.

    See ``sklearn.metrics.get_scorer_names()`` for a full list of built-in
    scorer strings.
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
        """
        Parameters
        ----------
        model : fitted sklearn estimator
            A fitted estimator exposing a ``predict`` (regression) or
            ``predict_proba`` / ``decision_function`` (classification) method,
            as required by *scoring*.
        scoring : str or callable
            Scorer to use.  Either:

            * A string registered with scikit-learn (e.g. ``'r2'``,
              ``'accuracy'``, ``'roc_auc'``, ``'f1_macro'``).  Call
              ``sklearn.metrics.get_scorer_names()`` to list all valid names.
            * A callable with the signature
              ``scorer(estimator, X, y) -> float`` where higher values
              indicate better performance (negate losses if needed).

            The scorer must be compatible with *model* and with the ``y``
            passed to :meth:`fit`.  Passing a binary-only scorer (e.g.
            ``'roc_auc'``) with a multiclass ``y`` will raise a
            ``ValueError`` from scikit-learn at the first scoring call.
        n_repeats : int, default 30
            Number of permutation repeats per feature.  At least 2 are
            required so the NIG tracker can estimate a variance.
        credible_mass : float, default 0.95
            Posterior credible mass for the HDI used in
            :meth:`~bxai._engines.normal_ig.NormalIGTracker.decide`.
        prior_mu : float, default 0.0
            NIG prior mean for the importance of each feature.
        prior_nu : float, default 1e-4
            NIG prior pseudo-count (strength of prior on mean).
        prior_alpha : float, default 1e-4
            NIG prior shape parameter (must be > 0).
        prior_beta : float, default 1e-4
            NIG prior rate parameter (must be > 0).
        random_state : int or None, default None
            Seed for the column-permutation RNG.  Does not control model
            randomness (see the note on stochastic scorers in :meth:`fit`).
        """
        self.model = model
        self.scoring = scoring
        self.n_repeats = n_repeats
        self.credible_mass = credible_mass
        self.prior_mu = prior_mu
        self.prior_nu = prior_nu
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        self.random_state = random_state

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate_hyperparams(self) -> None:
        """Raise ValueError for any hyperparameter combination that is statistically invalid."""
        if self.n_repeats < 2:
            raise ValueError(
                f"n_repeats must be >= 2 so the NIG tracker can form a variance estimate "
                f"(requires at least 2 observations); got {self.n_repeats!r}"
            )
        if not (0.0 < self.credible_mass < 1.0):
            raise ValueError(
                f"credible_mass must be in (0, 1); got {self.credible_mass!r}"
            )
        if self.prior_alpha <= 0:
            raise ValueError(
                f"prior_alpha must be > 0 for a valid NIG prior; got {self.prior_alpha!r}"
            )
        if self.prior_beta <= 0:
            raise ValueError(
                f"prior_beta must be > 0 for a valid NIG prior; got {self.prior_beta!r}"
            )
        if self.prior_nu <= 0:
            raise ValueError(
                f"prior_nu must be > 0 for a valid NIG prior; got {self.prior_nu!r}"
            )

    def fit(self, X: Any, y: Any) -> "BayesianPermutation":
        """Compute the Bayesian Permutation Importance of features in X.

        Parameters
        ----------
        X : array-like or pd.DataFrame of shape (n_samples, n_features)
            Validation feature matrix.  Must be compatible with ``model``.
        y : array-like of shape (n_samples,)
            Validation target vector.  The dtype and value range must match
            what *scoring* expects:

            * **Regression**: numeric targets (float or int).
            * **Binary classification**: two distinct class labels.
            * **Multiclass**: three or more class labels.  Ensure *scoring*
              supports multiclass (e.g. ``'accuracy'``, ``'f1_macro'``,
              ``'roc_auc_ovr'``) — passing a binary-only scorer such as
              ``'roc_auc'`` with a multiclass ``y`` will raise a
              ``ValueError`` from scikit-learn.
        """
        self._validate_hyperparams()
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

        # NOTE: baseline_score is computed *inside* the repeat loop (see below) so that
        # stochastic scorers (e.g. subsampled ensembles) are evaluated under the same
        # noise regime as each shuffled score.  Hoisting the baseline outside would make
        # it a single noisy draw while each column drop gets its own draw, introducing
        # a systematic sign bias in the deltas.

        # Initialize tracker
        self.tracker_ = NormalIGTracker(
            n_features=n_features,
            prior_mu=self.prior_mu,
            prior_nu=self.prior_nu,
            prior_alpha=self.prior_alpha,
            prior_beta=self.prior_beta,
        )

        # deltas[r, j] = baseline_score − score_when_feature_j_is_shuffled.
        # Shape: (n_repeats, n_features).
        deltas = np.zeros((self.n_repeats, n_features))

        # Permutation protocol (mirrors sklearn permutation_importance):
        #   • Each repeat r starts from a fresh copy of X so that shuffling
        #     feature j does not bleed into the evaluation of feature j+1.
        #   • Within a repeat, only one feature is shuffled at a time; all
        #     others retain their original values.  After scoring, the column
        #     is restored before moving to the next feature.
        # This gives n_repeats independent, unbiased importance draws per feature.
        #
        # np.ascontiguousarray is used instead of .copy() to guarantee an
        # independent, C-contiguous buffer even when X_arr is a non-contiguous
        # view (e.g. a column-sliced DataFrame or Fortran-order array) — plain
        # .copy() with order='K' would preserve the non-contiguous layout and
        # could share memory with the original under certain NumPy versions.
        for r in range(self.n_repeats):
            # Re-compute baseline each repeat so stochastic scorers contribute
            # a fresh noise draw that is matched to the shuffled-score draws
            # within the same repeat, keeping deltas mean-zero under H0.
            baseline_score = scorer(self.model, X_arr, y_arr)

            X_temp = np.ascontiguousarray(X_arr)
            for col_idx in range(n_features):
                # Shuffle column
                original_col = X_temp[:, col_idx].copy()
                X_temp[:, col_idx] = rng.permutation(original_col)

                # Compute shuffled score
                shuffled_score = scorer(self.model, X_temp, y_arr)

                # Drop in score is baseline - shuffled (higher positive means more important)
                deltas[r, col_idx] = baseline_score - shuffled_score

                # Restore column before evaluating the next feature
                X_temp[:, col_idx] = original_col

        # Feed all repeats to the tracker in a single batch.
        #
        # Statistical semantics: deltas is treated as n_repeats i.i.d. draws
        # from the importance distribution of each feature.  The NIG conjugate
        # update decomposes *column-by-column* (each feature has its own
        # independent (μ_j, σ²_j) pair), so passing the full (n_repeats ×
        # n_features) matrix is equivalent to n_features independent univariate
        # NIG updates — one per feature — not a single multivariate update.
        # The rows are NOT assumed to be identically distributed across features
        # (each column has its own scale), only within a column.
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

    def summary(self, credible_mass: Optional[float] = None) -> pd.DataFrame:
        """Return a summary of the features and their permutation decisions.

        Parameters
        ----------
        credible_mass : float or None, optional
            The probability mass to include in the credible interval
            (must be in ``(0, 1)``).  When ``None`` (default), falls back to
            the ``credible_mass`` value supplied at construction time.
        """
        mass = credible_mass if credible_mass is not None else self.credible_mass
        lower, upper = self.tracker_.credible_interval(mass)
        
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
