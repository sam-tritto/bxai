# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `BayLIME` now supports a `backend` parameter: ``'analytical'`` (default, unchanged) and ``'mcmc'``.
- The MCMC backend builds a PyMC probabilistic model with heteroscedastic observation noise scaled inversely by LIME proximity weights (``obs_sigma = sigma_global / sqrt(proximity_weight)``), capturing the geometric intuition that nearby perturbations should constrain the local surrogate more tightly.
- MCMC backend supports an optional Horseshoe prior on local coefficients via ``mcmc_prior='horseshoe'``, providing automatic sparsity without manual regularization tuning.
- ``BayLIMEExplanation`` stores raw posterior draws in ``posterior_draws_`` when the MCMC backend is used, enabling richer diagnostics and custom HDI computation.
- ``BayLIMEExplanation.credible_intervals()`` now dispatches to ArviZ-based HDI for MCMC explanations and falls back to equal-tailed percentile intervals if ArviZ is unavailable.
- Expanded test suite for ``BayLIME`` from 1 to 12 tests covering both backends, prior injection, convergence checks, and validation error handling.



### Added
- Initial project structure scaffolding with `uv` package management.
- Configuration for dependencies including lightgbm, shap, scipy, scikit-learn, and optional groups for xgboost, catboost, and mcmc.
- Shared Bayesian inference engines: `BetaBinomialTracker` and `NormalIGTracker`.
- Phase 1 global non-linear feature selection: `BayesianBorutaSHAP` and `BayesianPermutation`.
- Phase 2 local interpretability: `BayLIME` (Bayesian local surrogate).
- Phase 3 parametric/native Bayesian importance: `ShrinkagePIP` (regularized GLMs via PyMC) and `BARTImportance` (split-frequency frequencies via PyMC-BART).
- Comprehensive test suite for verification.
