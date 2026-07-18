# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.1] - 2026-07-17

### Fixed
- Fixed broken logo image link in PyPI README display by using absolute GitHub Raw Content URL.

## [0.1.0] - 2026-07-17

### Fixed
- **`BayesianBorutaSHAP`: per-iteration random seed now derived from master RNG** —
  previously, `base_model.random_state = self.random_state` was set on the template
  before the loop, so every `clone(base_model)` inside the loop received the identical
  seed. This caused the tree's internal randomness (feature sub-sampling, bootstrap)
  to be constant across all iterations, partially undermining the diversity of shadow
  feature comparisons. The fix removes the pre-loop assignment and instead derives a
  unique `int(rng.integers(2**31 - 1))` seed for each `model_it` after cloning.
  End-to-end reproducibility is preserved: the master `rng` is still seeded by
  `self.random_state`, so the full sequence of per-iteration seeds is deterministic
  when a seed is provided.
- **Corrected Highest Density Interval (HDI) computation and labeling** —
  previously, `BARTImportance` and `ShrinkagePIP.summary()` calculated
  equal-tailed percentile intervals using `np.percentile` but labelled them as
  Highest Density Intervals (`hdi_lower` / `hdi_upper`). Since VIF and coefficient
  distributions are often right-skewed or asymmetric, equal-tailed intervals and
  true HDIs can differ meaningfully. We now utilize ArviZ's `hdi` to compute
  the actual HDI when available, falling back to equal-tailed intervals when not,
  and adding an `interval_type` column (valued `'hdi'` or `'ci'`) to the
  summaries to ensure interval type clarity.
- **`_extract_shap_importances`: eliminated silent exception swallowing** —
  the previous implementation caught all exceptions with a bare `except
  Exception: pass`, hiding SHAP API changes, CUDA errors, incompatible model
  types, and `MemoryError`. The final fallback also re-created a second
  `shap.Explainer(explainer.model, X)`, which only works for
  `TreeExplainer`; any other explainer type would raise an opaque
  `AttributeError` with no diagnostic information. The fix narrows the
  caught exceptions between the two call conventions to only
  `(NotImplementedError, TypeError)` — the only cases where a legitimate
  API-convention mismatch occurs. When both the modern `explainer(X)` and
  the legacy `explainer.shap_values(X)` calls fail, a `RuntimeError` is
  raised with the explainer type, both original error messages, and
  actionable guidance, chained via `raise … from` so the full traceback
  is preserved.
- **Input validation on hyperparameters** — previously, nonsensical hyperparameter
  combinations silently produced statistically meaningless or mathematically broken
  results. The following guards now raise `ValueError` with descriptive messages
  before any expensive computation begins:
  - `confirm_threshold <= reject_threshold` in `BetaBinomialTracker.decide` and
    `BayesianBorutaSHAP` (impossible decision: every feature would be TENTATIVE).
  - `credible_mass` outside `(0, 1)` in `BetaBinomialTracker.credible_interval /
    decide`, `NormalIGTracker.credible_interval / decide`, `BayesianBorutaSHAP`,
    `BayesianPermutation`, and `BARTImportance`.
  - `prior_alpha <= 0` or `prior_beta <= 0` in `BetaBinomialTracker` (would
    produce an invalid Beta distribution) and in `NormalIGTracker` (would produce
    an invalid Normal-Inverse-Gamma prior); same guards are propagated to the
    corresponding constructor parameters of `BayesianBorutaSHAP` and
    `BayesianPermutation`.
  - `prior_nu <= 0` in `NormalIGTracker` (the pseudo-sample-count must be
    strictly positive for a proper NIG prior).
  - `n_repeats < 2` in `BayesianPermutation` (the NIG variance estimate requires
    at least two observations; a single permutation repeat gives a degenerate
    sum-of-squares of zero).
  - `kappa_threshold` or `pip_threshold` outside `(0, 1)` in `ShrinkagePIP`
    (both are probabilities bounded in the open unit interval).
- **`check_is_fitted`: replaced unreliable custom implementation with `sklearn.utils.validation.check_is_fitted`** —
  the previous implementation walked `dir(estimator)` and returned `True` if any attribute ending
  in `_` existed. This was both unreliable (Python objects carry many single-underscore attributes
  unrelated to fitting, and the `__` guard did not prevent traversal of inherited class attrs) and
  incorrect for external sklearn estimators passed to `BayesianPermutation`: a fitted
  `RandomForestClassifier` happens to expose `n_estimators_` (an init-param echo), so an unfitted
  model of the same type would silently pass. The fix removes the bespoke function entirely and
  re-exports `sklearn.utils.validation.check_is_fitted`, which raises the canonical
  `NotFittedError`, honours `__sklearn_is_fitted__`, and is consistent with the rest of the
  sklearn ecosystem. `BayesianPermutation.fit` now imports the function directly from sklearn.
- **`ShrinkagePIP`: incorrect PIP computation for Horseshoe prior** — the previous
  implementation used a hardcoded `epsilon=1e-3` threshold and evaluated
  PIP = P(|β_j| > ε | data). Because the Horseshoe posterior is a continuous
  distribution with full support on ℝ, this trivially evaluates to ≈ 1.0 for
  every feature after good shrinkage, making the threshold useless. The fix
  introduces a prior-aware `pip_method` parameter:
  - `pip_method='kappa'` (new default for `prior='horseshoe'`): computes the
    posterior shrinkage factor κ_j = 1 / (1 + λ_j² τ²) from MCMC draws and
    uses PIP = P(κ_j < `kappa_threshold` | data). κ_j near 0 indicates that
    the local scale λ_j dominates (signal); κ_j near 1 indicates that global
    shrinkage dominates (noise).
  - `pip_method='threshold'` (default for `prior='lasso'`): retains the
    coefficient-magnitude criterion but auto-scales ε from the data
    (`std(y) / 10` for linear models, `0.1` on the log-odds scale for
    logistic) when `epsilon=None` (new default).
  - `pip_method='auto'` (default) selects the appropriate strategy based on
    the active prior.

### Added
- **Cross-validation stability helper** — Added `cross_val_feature_stability` to evaluate selector robustness, computing chance-adjusted Nogueira and Jaccard stability indices in parallel using `joblib`.
- **`feature_importances_` attribute** — Implemented standard `feature_importances_` attribute across all selectors (`BayesianBorutaSHAP`, `BayesianPermutation`, `ShrinkagePIP`, `BARTImportance`) for direct integration with SHAP plotting utilities.
- **Pipeline integration support** — Added `SelectorMixin` inheritance and support masks to `ShrinkagePIP` and `BARTImportance` so all selectors can be fitted and transformed inside standard scikit-learn `Pipeline` objects.
- **Iteration history tracking** — Added an `iteration_history_` attribute to `BayesianBorutaSHAP` recording iteration-by-iteration status changes and posterior updates for convergence diagnostics.
- **Classification support (Probit BART) in `BARTImportance`** — Added binary classification support to `BARTImportance` via a Probit link function and Bernoulli likelihood, including automatic target label binarization.
- **Interpretable perturbation space in `BayLIME`** — Added a `perturbation_space` parameter (supporting `"feature_space"` and `"interpretable"`) enabling binary on/off perturbations and binary distance computation as in original LIME.
- **`n_jobs` parallel feature evaluation in `BayesianPermutation`** — Added `n_jobs` parameter to `BayesianPermutation` to run column permutation score evaluations in parallel via `joblib`.
- **Dependency optimizations** — Made `shap` and `lightgbm` optional dependencies under a new `boruta` extra. Removed phantom dependencies `numba` and `llvmlite` to reduce disk footprint by ~250MB. Updated pandas floor to `pandas>=2.0.0`. Switched build backend to standard `hatchling`.
- `ShrinkagePIP` exposes `kappa_mean_` (mean posterior shrinkage factor per
  feature) after fitting when `pip_method='kappa'`, and `epsilon_` (the
  effective threshold used) when `pip_method='threshold'`.
- `ShrinkagePIP.summary()` now includes a `pip_method` column and either
  `kappa_mean` or `epsilon` depending on the active strategy.
- New `ShrinkagePIP` constructor parameters: `pip_method` (`'auto'`),
  `kappa_threshold` (`0.5`). `epsilon` now accepts `None` (auto-scale).
- `BayLIME` now supports a `backend` parameter: ``'analytical'`` (default,
  unchanged) and ``'mcmc'``.
- The MCMC backend builds a PyMC probabilistic model with heteroscedastic
  observation noise scaled inversely by LIME proximity weights
  (``obs_sigma = sigma_global / sqrt(proximity_weight)``), capturing the
  geometric intuition that nearby perturbations constrain the local surrogate
  more tightly.
- MCMC backend supports an optional Horseshoe prior on local coefficients via
  ``mcmc_prior='horseshoe'``, providing automatic sparsity without manual
  regularization tuning.
- ``BayLIMEExplanation`` stores raw posterior draws in ``posterior_draws_``
  when the MCMC backend is used, enabling richer diagnostics and custom HDI
  computation.
- ``BayLIMEExplanation.credible_intervals()`` now dispatches to ArviZ-based
  HDI for MCMC explanations and falls back to equal-tailed percentile
  intervals if ArviZ is unavailable.
- Expanded test suite for ``BayLIME`` from 1 to 12 tests covering both
  backends, prior injection, convergence checks, and validation error
  handling.
- Initial project structure scaffolding with `uv` package management.
- Configuration for dependencies including lightgbm, shap, scipy,
  scikit-learn, and optional groups for xgboost, catboost, and mcmc.
- Shared Bayesian inference engines: `BetaBinomialTracker` and
  `NormalIGTracker`.
- Phase 1 global non-linear feature selection: `BayesianBorutaSHAP` and
  `BayesianPermutation`.
- Phase 2 local interpretability: `BayLIME` (Bayesian local surrogate).
- Phase 3 parametric/native Bayesian importance: `ShrinkagePIP` (regularized
  GLMs via PyMC) and `BARTImportance` (split-frequency tracking via
  PyMC-BART).
- Comprehensive test suite for verification.
