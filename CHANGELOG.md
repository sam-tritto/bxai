# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed
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

## [0.1.0] - 2026-07-16

### Added
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
