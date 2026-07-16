# bxai

Bayesian Feature Attribution and XAI Suite.

A Python package implementing rigorous, computationally efficient Bayesian methods for feature selection and model explainability. The guiding principle is "Rigorous but Fast": leveraging closed-form analytical updates wherever possible to avoid the heavy computational drag of full MCMC simulations, while maintaining statistical integrity.

## Features

- **Phase 1: Global Non-Linear Selection**
  - `BayesianBorutaSHAP`: Tree-based feature selection using SHAP values, swapping frequentist p-values for Bayesian credible intervals, with dynamic pruning for fast performance. Supports discrete (Beta-Binomial) and continuous (Normal-Inverse-Gamma) modes.
  - `BayesianPermutation`: Model-agnostic importance tracking using paired validation loss drops updated via the Student-t continuous engine.

- **Phase 2: Local Interpretability**
  - `BayLIME`: Stable, prior-informed local explanations wrapping standard/custom perturbations in a closed-form Bayesian linear regression. Can be seeded with global SHAP weights from Phase 1.

- **Phase 3: Parametric & Native Bayesian Importance**
  - `ShrinkagePIP`: High-dimensional GLMs with Horseshoe and Lasso regularizing priors, tracking Posterior Inclusion Probabilities (PIP) (requires `mcmc` extra).
  - `BARTImportance`: Variable inclusion frequency (VIF) tracking from native Bayesian Additive Regression Trees (requires `mcmc` extra).

## Installation

Install using `uv`:

```bash
uv add bxai
```

Or install with optional dependencies (e.g., PyMC for MCMC methods, catboost, xgboost):

```bash
uv add bxai --optional mcmc --optional xgboost --optional catboost
```

## Quick Start

### BayesianBorutaSHAP

```python
import lightgbm as lgb
from sklearn.datasets import make_classification
from bxai.selection import BayesianBorutaSHAP

# Generate synthetic dataset
X, y = make_classification(n_samples=500, n_features=20, n_informative=5, random_state=42)

# Fit Bayesian BorutaSHAP
clf = lgb.LGBMClassifier(random_state=42)
selector = BayesianBorutaSHAP(model=clf, mode="discrete", max_iter=20, random_state=42)
selector.fit(X, y)

print("Confirmed Features:", selector.confirmed_)
```

### BayLIME

```python
from bxai.explanation import BayLIME
import numpy as np

# Instantiate BayLIME
explainer = BayLIME(
    training_data=X,
    feature_names=[f"feat_{i}" for i in range(20)]
)

# Explain a single instance
explanation = explainer.explain_instance(
    instance=X[0],
    predict_fn=clf.predict_proba
)

print(explanation.as_dataframe())
```

## License

MIT License
