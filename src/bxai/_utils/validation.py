from typing import Any, Tuple
import numpy as np
import pandas as pd


def check_array_2d(X: Any) -> np.ndarray:
    """Ensure X is a 2D array or pandas DataFrame/Series, convert to numpy array."""
    if isinstance(X, (pd.DataFrame, pd.Series)):
        X_arr = X.to_numpy()
    else:
        X_arr = np.asarray(X)
    
    if X_arr.ndim == 1:
        X_arr = X_arr.reshape(-1, 1)
    elif X_arr.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {X_arr.shape}")
    
    return X_arr


def check_consistent_length(X: Any, y: Any) -> Tuple[np.ndarray, np.ndarray]:
    """Check that X and y have consistent lengths and convert both to numpy."""
    X_arr = check_array_2d(X)
    
    if isinstance(y, (pd.DataFrame, pd.Series)):
        y_arr = y.to_numpy()
    else:
        y_arr = np.asarray(y)
        
    if len(X_arr) != len(y_arr):
        raise ValueError(
            f"Found input variables with inconsistent numbers of samples: "
            f"[{len(X_arr)}, {len(y_arr)}]"
        )
    return X_arr, y_arr


def check_is_fitted(estimator: Any, attributes: str = "classes_") -> None:
    """Checks if estimator is fitted by checking for presence of attributes.
    
    If attributes is not specified, it checks standard sklearn fitted attributes.
    """
    # If the estimator has fit method but is not fitted, typical attributes aren't present.
    # We check if at least one common fitted attribute ends with an underscore.
    fitted = False
    if hasattr(estimator, attributes):
        fitted = True
    else:
        # Check standard sklearn convention: attributes ending with _ but not __
        for attr in dir(estimator):
            if attr.endswith("_") and not attr.startswith("__"):
                fitted = True
                break
                
    if not fitted:
        raise ValueError(f"This {type(estimator).__name__} instance is not fitted yet.")
