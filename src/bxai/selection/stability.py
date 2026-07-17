from dataclasses import dataclass
from typing import List, Optional, Union, Any, Dict
import numpy as np
from sklearn.base import clone
from sklearn.model_selection import check_cv
from joblib import Parallel, delayed


@dataclass
class CVStabilityResult:
    """Result of cross-validation feature selection stability evaluation.

    Attributes
    ----------
    stability_index : float
        The Nogueira stability index, adjusted for chance, defined in [-1, 1]
        (or 1.0 if selection is perfectly stable with no variance).
        A value of 1.0 represents perfect stability.
    jaccard_stability_index : float
        The average pairwise Jaccard index across all folds, defined in [0, 1].
    selection_frequencies : np.ndarray of shape (n_features,)
        The selection frequency of each feature across the CV folds.
    mean_selected_features : float
        The mean number of selected features per fold.
    support_matrix : np.ndarray of shape (n_folds, n_features)
        A binary matrix where row i represents the feature selection mask for fold i.
    feature_names : list of str or None
        The feature names, if available.
    """
    stability_index: float
    jaccard_stability_index: float
    selection_frequencies: np.ndarray
    mean_selected_features: float
    support_matrix: np.ndarray
    feature_names: Optional[List[str]] = None


def calculate_nogueira_stability(support_matrix: np.ndarray) -> float:
    """Calculate Nogueira's stability index from a binary support matrix.

    This index is adjusted for chance and ranges from -1 to 1 (or is 1.0 if
    selection is perfectly stable with zero variance).

    Parameters
    ----------
    support_matrix : np.ndarray of shape (n_runs, n_features)
        Binary matrix of feature selections.

    Returns
    -------
    stability : float
        Nogueira stability index.
    """
    Z = np.asarray(support_matrix, dtype=int)
    if Z.ndim != 2:
        raise ValueError("support_matrix must be a 2D array.")
    M, d = Z.shape
    if M <= 1:
        raise ValueError("At least 2 runs/folds are required to estimate stability.")
    if d == 0:
        return 1.0

    # Probability (frequency) of selection for each feature
    p_hat = np.mean(Z, axis=0)
    
    # Average number of selected features per run
    k_bar = np.sum(p_hat)
    
    # If all selections are identical (all 0s or all 1s), variance is 0, so stability is 1.0
    if k_bar == 0.0 or k_bar == d:
        return 1.0

    # Unbiased sample variance of each feature selection across the M runs
    # s_j^2 = (M / (M - 1)) * p_hat_j * (1 - p_hat_j)
    var_sum = np.sum(p_hat * (1.0 - p_hat)) * (M / (M - 1.0))
    
    # Nogueira stability index formula
    stability = 1.0 - (var_sum / (k_bar * (1.0 - k_bar / d)))
    
    return float(stability)


def calculate_jaccard_stability(support_matrix: np.ndarray) -> float:
    """Calculate the average pairwise Jaccard stability index.

    Parameters
    ----------
    support_matrix : np.ndarray of shape (n_runs, n_features)
        Binary matrix of feature selections.

    Returns
    -------
    jaccard_stability : float
        Average pairwise Jaccard index.
    """
    Z = np.asarray(support_matrix, dtype=bool)
    if Z.ndim != 2:
        raise ValueError("support_matrix must be a 2D array.")
    M, d = Z.shape
    if M <= 1:
        raise ValueError("At least 2 runs/folds are required to estimate stability.")

    jaccards = []
    for i in range(M):
        for j in range(i + 1, M):
            intersection = np.sum(Z[i] & Z[j])
            union = np.sum(Z[i] | Z[j])
            if union == 0:
                # If both selected 0 features, Jaccard similarity is 1.0
                jaccards.append(1.0)
            else:
                jaccards.append(intersection / union)
    return float(np.mean(jaccards))


def _safe_indexing(X: Any, indices: Any) -> Any:
    """Helper to safely slice DataFrame/Series, numpy arrays, or lists."""
    if hasattr(X, "iloc"):
        return X.iloc[indices]
    elif hasattr(X, "ndim"):
        return X[indices]
    else:
        return [X[i] for i in indices]


def _fit_and_select(
    estimator: Any,
    X: Any,
    y: Optional[Any],
    train_idx: Any,
    fit_params: Optional[Dict[str, Any]],
) -> np.ndarray:
    """Clone, fit, and extract support mask for one fold."""
    cloned = clone(estimator)
    X_train = _safe_indexing(X, train_idx)
    y_train = _safe_indexing(y, train_idx) if y is not None else None
    
    cloned.fit(X_train, y_train, **(fit_params or {}))
    return np.asarray(cloned.get_support(), dtype=bool)


def cross_val_feature_stability(
    estimator: Any,
    X: Any,
    y: Optional[Any] = None,
    *,
    cv: Union[int, Any] = 5,
    groups: Optional[Any] = None,
    n_jobs: Optional[int] = None,
    verbose: int = 0,
    pre_dispatch: Union[str, int] = "2*n_jobs",
    fit_params: Optional[Dict[str, Any]] = None,
) -> CVStabilityResult:
    """Evaluate feature selection stability using cross-validation.

    This helper performs cross-validation by fitting the estimator on the training
    set of each fold and extracting the feature selection mask (`get_support()`).
    It then computes stability metrics across all folds.

    Parameters
    ----------
    estimator : estimator object
        An unfitted scikit-learn selector implementing the `get_support` method.
    X : array-like of shape (n_samples, n_features)
        The training input samples.
    y : array-like of shape (n_samples,), default None
        The target values (class labels or regression targets).
    cv : int, cross-validation generator, or iterable, default 5
        Determines the cross-validation splitting strategy.
    groups : array-like of shape (n_samples,), default None
        Group labels for the samples used when splitting the dataset.
    n_jobs : int or None, default None
        The number of jobs to run in parallel. None means 1.
    verbose : int, default 0
        The verbosity level.
    pre_dispatch : int or str, default "2*n_jobs"
        Controls the number of jobs that get dispatched during parallel execution.
    fit_params : dict, default None
        Parameters to pass to the fit method of the estimator.

    Returns
    -------
    result : CVStabilityResult
        A result object containing stability metrics and details.
    """
    if not hasattr(estimator, "get_support"):
        raise TypeError("estimator must be a feature selector implementing get_support().")

    cv_resolved = check_cv(cv, y, classifier=False)
    splits = list(cv_resolved.split(X, y, groups))

    results = Parallel(n_jobs=n_jobs, verbose=verbose, pre_dispatch=pre_dispatch)(
        delayed(_fit_and_select)(estimator, X, y, train_idx, fit_params)
        for train_idx, _ in splits
    )

    support_matrix = np.vstack(results)
    
    # Calculate stability metrics
    stability_index = calculate_nogueira_stability(support_matrix)
    jaccard_stability_index = calculate_jaccard_stability(support_matrix)
    selection_frequencies = np.mean(support_matrix, axis=0)
    mean_selected_features = float(np.mean(np.sum(support_matrix, axis=1)))

    feature_names = list(X.columns) if hasattr(X, "columns") else None

    return CVStabilityResult(
        stability_index=stability_index,
        jaccard_stability_index=jaccard_stability_index,
        selection_frequencies=selection_frequencies,
        mean_selected_features=mean_selected_features,
        support_matrix=support_matrix,
        feature_names=feature_names,
    )
