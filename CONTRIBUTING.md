# Contributing to bxai

Thank you for your interest in contributing to **bxai**! We want to make contributing to this repository structured, transparent, and rewarding.

This document outline the project's modular architecture, how to set up your local development environment, and the expectations for formatting, validation, and testing.

---

## 🗺️ 1. Architecture & Extension Hooks

`bxai` is built using a clean, modular structure where statistical inference engine logic is decoupled from high-level selectors and explainers. 

### Core Components Structure:
*   [src/bxai/_engines/](file:///Users/sam/Locals%20Only/bxai/src/bxai/_engines/) — Analytical Bayesian trackers (e.g., `BetaBinomialTracker`, `NormalIGTracker`). These handle sequential updates of prior parameters as new data evidence is collected.
*   [src/bxai/selection/](file:///Users/sam/Locals%20Only/bxai/src/bxai/selection/) — Feature selection routines (e.g., `BayesianBorutaSHAP`, `BayesianPermutation`).
*   [src/bxai/parametric/](file:///Users/sam/Locals%20Only/bxai/src/bxai/parametric/) — Parametric Bayesian model inference (e.g., Horseshoe-prior-based `ShrinkagePIP` and `BARTImportance`).
*   [src/bxai/explanation/](file:///Users/sam/Locals%20Only/bxai/src/bxai/explanation/) — Local model interpretability/attribution tools (e.g., `BayLIME`).

---

## 🔌 2. Extension Hooks

If you want to contribute new functionality, here are the standard entry points:

### 1. Adding a New Bayesian Tracking Engine
If you want to implement a new conjugate tracker (e.g. Student-t or Dirichlet-Multinomial):
1. Create a class under [src/bxai/_engines/](file:///Users/sam/Locals%20Only/bxai/src/bxai/_engines/) implementing your analytical updates.
2. Ensure your tracker exposes consistent properties like `mean`, `variance`, and methods to calculate credible intervals or Highest Density Intervals (HDI) bounds.

### 2. Adding a New Selector
If you want to add a new model-agnostic feature selection method:
1. Subclass the relevant scikit-learn base classes (e.g. `BaseEstimator`, `SelectorMixin`).
2. Add your logic to `bxai/selection/<new_method>.py`.
3. Expose it in [src/bxai/selection/__init__.py](file:///Users/sam/Locals%20Only/bxai/src/bxai/selection/__init__.py).

---

## ⚡ 3. Quickstart Developer Setup

You can set up your local development environment in just a few steps using **uv**:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/sam-tritto/bxai.git
    cd bxai
    ```

2.  **Synchronize virtual environment and install dev dependencies:**
    ```bash
    uv sync --all-extras --dev
    ```

3.  **Run the test suite (fast tests):**
    ```bash
    uv run pytest --skip-slow
    ```

4.  **Run all tests (including MCMC and slow statistical validations):**
    ```bash
    uv run pytest
    ```

---

## 🏷️ 4. Code Quality & Verification

Every contribution must pass formatting, linting, type validation, and unit tests before it can be merged.

### Format & Lint Checking:
We use `ruff` for code styling, formatting, and linting. Run these checks before submitting a PR:
```bash
# Check for lint errors
uv run ruff check .

# Automatically apply safe fixes
uv run ruff check . --fix

# Check formatting
uv run ruff format --check .

# Format code
uv run ruff format .
```

### Static Type Checks:
We use `mypy` for static type checking. All files in the codebase should satisfy:
```bash
uv run mypy src/
```

---

## 📜 5. Submission Guidelines

1.  **Open an Issue**: Before making major changes, open an issue to discuss your proposed approach and ensure alignment.
2.  **Write Tests**: Every new feature or bugfix should include corresponding unit tests in the `tests/` directory.
3.  **Update Changelog**: Summarize your changes in `CHANGELOG.md`.
4.  **Submit Pull Request**: Create a Pull Request against the `main` branch. Ensure that GitHub Actions CI passes successfully.
