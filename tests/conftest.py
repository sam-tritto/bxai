"""Shared pytest fixtures for the bxai test suite.

Fixtures are organised into three *tiers* that match the three common dataset
sizes used across the test modules:

  tiny  — n_samples=30,  n_features=4,  n_informative=2  (hyper-param validation)
  small — n_samples=100, n_features=6,  n_informative=2  (smoke / structure)
  large — n_samples=300, n_features=8,  n_informative=2  (statistical assertions)

Each tier exposes ``(X, y)`` and, where needed, a pre-fitted
``RandomForestClassifier``.  All fixtures are *session-scoped* so the
(deterministic) dataset is built once per test session, cutting repeated
``make_classification`` and ``fit`` calls.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

# ---------------------------------------------------------------------------
# CLI option: --skip-slow
# ---------------------------------------------------------------------------
# Usage:  pytest --skip-slow
# Effect: any test marked @pytest.mark.slow (which includes all
#         @pytest.mark.mcmc tests) is automatically skipped.
# This lets CI pipelines express intent more legibly than raw -m expressions.


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--skip-slow",
        action="store_true",
        default=False,
        help="Skip tests marked @pytest.mark.slow (includes all MCMC tests).",
    )


def pytest_collection_modifyitems(
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    if not config.getoption("--skip-slow"):
        return
    skip_slow = pytest.mark.skip(reason="Skipped via --skip-slow flag.")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


# ---------------------------------------------------------------------------
# Tiny dataset  (n=30, p=4, informative=2)   — hyperparameter validation
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def tiny_Xy() -> tuple[np.ndarray, np.ndarray]:
    """30-sample, 4-feature classification dataset (random_state=0)."""
    return make_classification(
        n_samples=30,
        n_features=4,
        n_informative=2,
        n_redundant=0,
        random_state=0,
    )


@pytest.fixture(scope="session")
def tiny_rf(tiny_Xy) -> RandomForestClassifier:
    """3-estimator RandomForestClassifier fitted on *tiny_Xy*."""
    X, y = tiny_Xy
    rf = RandomForestClassifier(n_estimators=3, random_state=0)
    rf.fit(X, y)
    return rf


# ---------------------------------------------------------------------------
# Small dataset  (n=100, p=6, informative=2)  — smoke / structure tests
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def small_Xy() -> tuple[np.ndarray, np.ndarray]:
    """100-sample, 6-feature classification dataset (random_state=42)."""
    return make_classification(
        n_samples=100,
        n_features=6,
        n_informative=2,
        n_redundant=0,
        random_state=42,
    )


@pytest.fixture(scope="session")
def small_rf(small_Xy) -> RandomForestClassifier:
    """10-estimator RandomForestClassifier fitted on *small_Xy*."""
    X, y = small_Xy
    rf = RandomForestClassifier(n_estimators=10, random_state=42)
    rf.fit(X, y)
    return rf


# ---------------------------------------------------------------------------
# Large dataset  (n=300, p=8, informative=2)  — statistical-assertion tests
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def large_Xy() -> tuple[np.ndarray, np.ndarray]:
    """300-sample, 8-feature classification dataset (random_state=0).

    Uses ``n_clusters_per_class=1`` and ``n_repeated=0`` for a clean
    separation between informative and noise features.
    """
    return make_classification(
        n_samples=300,
        n_features=8,
        n_informative=2,
        n_redundant=0,
        n_repeated=0,
        n_clusters_per_class=1,
        random_state=0,
    )


@pytest.fixture(scope="session")
def large_rf(large_Xy) -> RandomForestClassifier:
    """50-estimator RandomForestClassifier fitted on *large_Xy*."""
    X, y = large_Xy
    rf = RandomForestClassifier(n_estimators=50, random_state=0)
    rf.fit(X, y)
    return rf
