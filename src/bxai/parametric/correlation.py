from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.special import ndtri
from scipy.stats import gaussian_kde, rankdata
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted

from bxai._utils.hdi import compute_hdi


class BayesianCorrelation(BaseEstimator):
    """Bayesian Correlation Estimator.

    Computes the posterior distribution of correlation coefficients (Pearson's r,
    Spearman's rho, or Kendall's tau) using Markov Chain Monte Carlo (MCMC) via PyMC.

    Parameters
    ----------
    method : {'pearson', 'spearman', 'kendall'}, default 'pearson'
        The correlation method to estimate:
        - 'pearson': Pearson product-moment correlation on raw data.
        - 'spearman': Spearman rank correlation.
        - 'kendall': Kendall rank correlation.
    backend : {'quick', 'latent_copula'}, default 'quick'
        The backend/modeling framework to use:
        - 'quick': Fits a standard bivariate normal model. For 'spearman',
          the model is fitted on the ranks of the data. Not valid for 'kendall'.
        - 'latent_copula': Maps ranks to standard normal scores (probit-transform
          of empirical CDF) and fits a bivariate normal model on these latent scores.
          Uses Greiner's relations to map the latent correlation parameter back to
          Spearman's rho or Kendall's tau. Only valid for 'spearman' and 'kendall'.
    credible_mass : float, default 0.95
        Probability mass to cover in the credible interval (Highest Density Interval).
    n_samples : int, default 1000
        Number of posterior draws per chain.
    tune : int, default 1000
        Number of tuning steps per chain.
    chains : int, default 2
        Number of MCMC chains.
    cores : int or None, default None
        Number of CPU cores to use for sampling.
    progressbar : bool, default False
        Whether to display a PyMC progress bar during sampling.
    random_state : int or None, default None
        Seed for reproducibility.
    """

    mean_: float | np.ndarray
    mode_: float | np.ndarray
    hdi_lower_: float | np.ndarray
    hdi_upper_: float | np.ndarray
    hdi_: tuple[float, float] | tuple[np.ndarray, np.ndarray]
    probability_of_direction_: float | np.ndarray
    correlation_samples_: np.ndarray
    strength_: str | np.ndarray

    def __init__(
        self,
        method: str = "pearson",
        backend: str = "quick",
        credible_mass: float = 0.95,
        n_samples: int = 1000,
        tune: int = 1000,
        chains: int = 2,
        cores: int | None = None,
        progressbar: bool = False,
        random_state: int | None = None,
    ):
        self.method = method
        self.backend = backend
        self.credible_mass = credible_mass
        self.n_samples = n_samples
        self.tune = tune
        self.chains = chains
        self.cores = cores
        self.progressbar = progressbar
        self.random_state = random_state

    def _validate_hyperparams(self) -> None:
        """Validate model parameters and their combinations."""
        if self.method not in ("pearson", "spearman", "kendall"):
            raise ValueError(
                f"method must be 'pearson', 'spearman', or 'kendall'; got {self.method!r}"
            )
        if self.backend not in ("quick", "latent_copula"):
            raise ValueError(
                f"backend must be 'quick' or 'latent_copula'; got {self.backend!r}"
            )
        if self.method == "pearson" and self.backend == "latent_copula":
            raise ValueError(
                "method='pearson' is not compatible with backend='latent_copula'. "
                "Pearson correlation requires backend='quick'."
            )
        if self.method == "kendall" and self.backend == "quick":
            raise ValueError(
                "method='kendall' requires backend='latent_copula'. "
                "Kendall's tau is not supported with the 'quick' backend."
            )
        if not (0.0 < self.credible_mass < 1.0):
            raise ValueError(
                f"credible_mass must be in (0, 1); got {self.credible_mass!r}"
            )
        if self.cores is not None:
            if not isinstance(self.cores, int) or self.cores <= 0:
                raise ValueError(
                    f"cores must be a positive integer or None; got {self.cores!r}"
                )

    def fit(self, X: Any, y: Any = None) -> BayesianCorrelation:
        """Fit the Bayesian correlation model using MCMC.

        Parameters
        ----------
        X : array-like of shape (n_samples,) or (n_samples, n_features)
            If `y` is provided, `X` represents one or more features.
            If `y` is None, `X` must be a 2D array or DataFrame of shape (n_samples, 2).
        y : array-like of shape (n_samples,), default None
            The second variable (target). Only used if `X` represents features.

        Returns
        -------
        self : BayesianCorrelation
            Fitted estimator.
        """
        self._validate_hyperparams()

        try:
            import pymc as pm
            import pytensor.tensor as pt
        except ImportError:
            raise ImportError(
                "pymc and pytensor are required to use BayesianCorrelation. "
                "Install them using `pip install 'bxai[mcmc]'` or `uv sync --extra mcmc`."
            )

        # 1. Resolve inputs and variable names
        if y is not None:
            # X and y are separate arrays
            if hasattr(X, "to_numpy"):
                X_arr = X.to_numpy()
            else:
                X_arr = np.asarray(X)

            if hasattr(y, "to_numpy"):
                y_arr = y.to_numpy()
            else:
                y_arr = np.asarray(y)

            if y_arr.ndim != 1:
                raise ValueError("y must be a 1D array.")

            self.target_name_ = str(getattr(y, "name", "target"))

            if X_arr.ndim == 1:
                n_features = 1
                feature_cols = [X_arr]
                self.feature_names_ = [str(getattr(X, "name", "x"))]
            elif X_arr.ndim == 2:
                n_features = X_arr.shape[1]
                feature_cols = [X_arr[:, i] for i in range(n_features)]
                if hasattr(X, "columns"):
                    self.feature_names_ = [str(c) for c in X.columns]
                else:
                    self.feature_names_ = [f"feature_{i}" for i in range(n_features)]
            else:
                raise ValueError("X must be a 1D or 2D array.")

            if len(X_arr) != len(y_arr):
                raise ValueError(
                    f"Found input variables with inconsistent numbers of samples: [{len(X_arr)}, {len(y_arr)}]"
                )
        else:
            # X must be a 2D array of shape (n_samples, 2)
            if hasattr(X, "to_numpy"):
                X_arr = X.to_numpy()
            else:
                X_arr = np.asarray(X)

            if X_arr.ndim != 2 or X_arr.shape[1] != 2:
                raise ValueError(
                    "When y is None, X must be a 2D array or DataFrame with exactly 2 columns."
                )
            n_features = 1
            feature_cols = [X_arr[:, 0]]
            y_arr = X_arr[:, 1]
            if hasattr(X, "columns") and len(X.columns) == 2:
                self.feature_names_ = [str(X.columns[0])]
                self.target_name_ = str(X.columns[1])
            else:
                self.feature_names_ = ["x"]
                self.target_name_ = "y"

        n_obs = X_arr.shape[0]
        if n_obs < 3:
            raise ValueError(
                f"Correlation requires at least 3 observations; got {n_obs}."
            )

        # 2. Build the PyMC model
        with pm.Model():
            for j in range(n_features):
                if self.backend == "latent_copula":
                    # Convert data to normal scores
                    r_x = rankdata(feature_cols[j])
                    r_y = rankdata(y_arr)

                    # Transform to Uniform (0, 1) using Van der Waerden normal scores transformation
                    u_x = (r_x - 0.5) / n_obs
                    u_y = (r_y - 0.5) / n_obs
                    # Transform to standard normal scores
                    z_x = ndtri(u_x)
                    z_y = ndtri(u_y)

                    z_data = np.column_stack([z_x, z_y])

                    # Uniform prior on latent correlation
                    rho_latent = pm.Uniform(f"rho_latent_{j}", lower=-1.0, upper=1.0)

                    # Bivariate Normal covariance matrix with var=1.0
                    cov = pt.stack(
                        [pt.stack([1.0, rho_latent]), pt.stack([rho_latent, 1.0])]
                    )

                    # Likelihood
                    pm.MvNormal(
                        f"observed_{j}", mu=pt.zeros(2), cov=cov, observed=z_data
                    )

                    # Track deterministic mappings
                    pm.Deterministic(
                        f"kendall_tau_{j}", (2.0 / np.pi) * pt.arcsin(rho_latent)
                    )
                    pm.Deterministic(
                        f"spearman_rho_{j}", (6.0 / np.pi) * pt.arcsin(rho_latent / 2.0)
                    )

                elif self.method == "spearman":  # and backend == 'quick'
                    # Convert data to ranks
                    r_x = rankdata(feature_cols[j])
                    r_y = rankdata(y_arr)
                    ranks_data = np.column_stack([r_x, r_y])

                    mu = pm.Normal(f"mu_{j}", mu=n_obs / 2.0, sigma=10.0, shape=2)
                    sigma = pm.HalfNormal(f"sigma_{j}", sigma=10.0, shape=2)
                    rho_spearman = pm.Uniform(
                        f"rho_spearman_{j}", lower=-1.0, upper=1.0
                    )

                    cov = pt.stack(
                        [
                            pt.stack(
                                [sigma[0] ** 2, rho_spearman * sigma[0] * sigma[1]]
                            ),
                            pt.stack(
                                [rho_spearman * sigma[0] * sigma[1], sigma[1] ** 2]
                            ),
                        ]
                    )

                    pm.MvNormal(f"observed_{j}", mu=mu, cov=cov, observed=ranks_data)

                else:  # self.method == 'pearson' and backend == 'quick'
                    pair_data = np.column_stack([feature_cols[j], y_arr])

                    mu = pm.Normal(f"mu_{j}", mu=0.0, sigma=10.0, shape=2)
                    sigma = pm.HalfNormal(f"sigma_{j}", sigma=5.0, shape=2)
                    rho = pm.Uniform(f"rho_{j}", lower=-1.0, upper=1.0)

                    cov = pt.stack(
                        [
                            pt.stack([sigma[0] ** 2, rho * sigma[0] * sigma[1]]),
                            pt.stack([rho * sigma[0] * sigma[1], sigma[1] ** 2]),
                        ]
                    )

                    pm.MvNormal(f"observed_{j}", mu=mu, cov=cov, observed=pair_data)

            # Sampling step
            self.trace_ = pm.sample(
                draws=self.n_samples,
                tune=self.tune,
                chains=self.chains,
                cores=self.cores,
                random_seed=self.random_state,
                compute_convergence_checks=False,
                progressbar=False,
            )

        # 3. Extract correct target samples for all features
        trace: Any = self.trace_
        all_samples = []
        for j in range(n_features):
            if self.backend == "latent_copula":
                var_name = (
                    f"spearman_rho_{j}"
                    if self.method == "spearman"
                    else f"kendall_tau_{j}"
                )
            elif self.method == "spearman":
                var_name = f"rho_spearman_{j}"
            else:
                var_name = f"rho_{j}"

            raw_samples = trace.posterior[var_name].values
            all_samples.append(raw_samples.ravel())

        self.correlation_samples_ = np.column_stack(all_samples)

        # 4. Compute metrics
        mean_arr = np.mean(self.correlation_samples_, axis=0)

        # Estimate the mode using gaussian KDE for each feature
        modes = []
        for j in range(n_features):
            col_samples = self.correlation_samples_[:, j]
            try:
                kde = gaussian_kde(col_samples)
                grid = np.linspace(-1.0, 1.0, 1000)
                modes.append(float(grid[np.argmax(kde.evaluate(grid))]))
            except Exception:
                modes.append(float(np.mean(col_samples)))
        mode_arr = np.array(modes)

        # HDI computation
        hdi_lower_arr, hdi_upper_arr = compute_hdi(
            self.correlation_samples_, self.credible_mass
        )

        # Probability of Direction
        prop_pos = np.mean(self.correlation_samples_ > 0.0, axis=0)
        prop_neg = np.mean(self.correlation_samples_ < 0.0, axis=0)
        prob_dir_arr = np.maximum(prop_pos, prop_neg)

        # 5. Build summary DataFrame
        summary_rows = []
        for j in range(n_features):
            mean_val = float(mean_arr[j])
            hdi_l = float(hdi_lower_arr[j])
            hdi_u = float(hdi_upper_arr[j])

            # Determine strength category
            if hdi_l <= 0.0 <= hdi_u:
                strength = "Uncertain"
            elif abs(mean_val) > 0.6:
                strength = "Strong"
            else:
                strength = "Modest"

            summary_rows.append(
                {
                    "Feature": self.feature_names_[j],
                    "Target": self.target_name_,
                    "Posterior Mean": mean_val,
                    "Posterior Mode": float(mode_arr[j]),
                    "95% HDI Lower": hdi_l,
                    "95% HDI Upper": hdi_u,
                    "Prob of Direction": float(prob_dir_arr[j]),
                    "Strength": strength,
                }
            )
        self.summary_df_ = pd.DataFrame(summary_rows)
        strength_arr = np.array([row["Strength"] for row in summary_rows], dtype=object)

        # 6. Simplify attributes if n_features == 1 for backward compatibility
        if n_features == 1:
            self.mean_ = float(mean_arr[0])
            self.mode_ = float(mode_arr[0])
            self.hdi_lower_ = float(hdi_lower_arr[0])
            self.hdi_upper_ = float(hdi_upper_arr[0])
            self.hdi_ = (self.hdi_lower_, self.hdi_upper_)
            self.probability_of_direction_ = float(prob_dir_arr[0])
            self.correlation_samples_ = self.correlation_samples_.ravel()
            self.variable_names_ = [self.feature_names_[0], self.target_name_]
            self.strength_ = str(strength_arr[0])
            self.summary_df_ = self.summary_df_.rename(
                columns={"Feature": "Feature 1", "Target": "Feature 2"}
            )
        else:
            self.mean_ = mean_arr
            self.mode_ = mode_arr
            self.hdi_lower_ = hdi_lower_arr
            self.hdi_upper_ = hdi_upper_arr
            self.hdi_ = (self.hdi_lower_, self.hdi_upper_)
            self.probability_of_direction_ = prob_dir_arr
            self.variable_names_ = self.feature_names_ + [self.target_name_]
            self.strength_ = strength_arr

        return self

    def plot_posterior(self, **kwargs: Any) -> Any:
        """Plot the posterior distribution of the correlation coefficient.

        Parameters
        ----------
        **kwargs : dict
            Keyword arguments passed directly to `arviz.plot_posterior`.

        Returns
        -------
        axes : matplotlib.axes.Axes or numpy.ndarray of them
            The matplotlib axes containing the plot.
        """
        check_is_fitted(self, attributes=["correlation_samples_"])

        try:
            import arviz as az
        except ImportError:
            raise ImportError(
                "arviz is required to use plot_posterior. Please install it."
            )

        n_features = (
            1
            if self.correlation_samples_.ndim == 1
            else self.correlation_samples_.shape[1]
        )

        # Resolve the var_names of interest in the trace
        var_names = []
        for j in range(n_features):
            if self.backend == "latent_copula":
                var_name = (
                    f"spearman_rho_{j}"
                    if self.method == "spearman"
                    else f"kendall_tau_{j}"
                )
            elif self.method == "spearman":
                var_name = f"rho_spearman_{j}"
            else:
                var_name = f"rho_{j}"
            var_names.append(var_name)

        return az.plot_posterior(self.trace_, var_names=var_names, **kwargs)

    def summary(self) -> pd.DataFrame:
        """Return a summary of the fitted correlation results.

        Returns
        -------
        df : pd.DataFrame
            DataFrame containing feature name, target name, posterior mean,
            posterior mode, 95% HDI bounds, and probability of direction.
        """
        check_is_fitted(self, attributes=["summary_df_"])
        return self.summary_df_

    def plot(self, color: str | None = None) -> Any:
        """Plot the correlation results.

        For a single feature (bivariate correlation), plots the posterior density distribution
        (KDE) with a filled area, mean line, and 95% HDI.
        For multiple features, plots a horizontal interval/forest plot.

        Parameters
        ----------
        color : str or None, default None
            Custom color to use for the plot. If None,
            the color is determined by the correlation direction (green for positive,
            red for negative, and gray for intervals overlapping zero).

        Returns
        -------
        fig : matplotlib.figure.Figure
            The matplotlib figure object.
        """
        check_is_fitted(self, attributes=["correlation_samples_"])

        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib is required to plot. Install it with pip.")

        n_features = (
            1
            if self.correlation_samples_.ndim == 1
            else self.correlation_samples_.shape[1]
        )

        if n_features == 1:
            # Bivariate single-feature density/dist plot
            samples = self.correlation_samples_
            mean_val = float(self.mean_)
            hdi_l = float(self.hdi_lower_)
            hdi_u = float(self.hdi_upper_)

            if color is not None:
                plot_color = color
            else:
                if hdi_l <= 0.0 <= hdi_u:
                    plot_color = "#CC5555"  # red if the CI crosses 0
                elif abs(mean_val) > 0.6:
                    plot_color = "#82B94C"  # green if magnitude > 0.6
                else:
                    plot_color = (
                        "#D4AC0D"  # yellow if magnitude > 0 and doesn't cross 0
                    )

            fig, ax = plt.subplots(figsize=(7.5, 4.5))

            # Calculate KDE on a localized grid based on standard deviation to prevent squishing and ensure smoothness
            std_val = max(float(np.std(samples)), 1e-4)
            xmin = max(-1.0, mean_val - 4.5 * std_val)
            xmax = min(1.0, mean_val + 4.5 * std_val)

            kde = gaussian_kde(samples)
            x_grid = np.linspace(xmin, xmax, 1000)
            kde_vals = kde.evaluate(x_grid)

            # Plot filled area and boundary line
            ax.fill_between(
                x_grid,
                kde_vals,
                color=plot_color,
                alpha=0.25,
                label="Posterior Density",
            )
            ax.plot(x_grid, kde_vals, color=plot_color, linewidth=2.5)

            # Draw vertical line for mean
            ax.axvline(
                mean_val,
                color="#2C3E50",
                linestyle="--",
                linewidth=1.5,
                label=f"Mean: {mean_val:.3f}",
            )

            # Draw horizontal bar for 95% HDI
            y_max = ax.get_ylim()[1]
            hdi_y = y_max * 0.05
            ax.hlines(
                hdi_y,
                hdi_l,
                hdi_u,
                color="#34495E",
                linewidth=4,
                label=f"95% HDI: [{hdi_l:.3f}, {hdi_u:.3f}]",
            )
            ax.plot([hdi_l, hdi_u], [hdi_y, hdi_y], "o", color="#2C3E50", markersize=8)

            ax.set_title(
                f"Posterior Correlation: {self.feature_names_[0]} vs {self.target_name_}",
                fontsize=12,
                fontweight="bold",
                pad=12,
            )
            ax.set_xlabel("Correlation Parameter value (ρ)", fontsize=10)
            ax.set_ylabel("Density", fontsize=10)
            ax.set_xlim(xmin, xmax)
            ax.grid(True, linestyle=":", alpha=0.5)
            ax.legend(
                loc="upper left", frameon=True, facecolor="white", edgecolor="none"
            )
            plt.tight_layout()
            return fig

        else:
            # Forest plot for multiple features
            df = self.summary_df_
            df = df.sort_values("Posterior Mean", ascending=True)

            fig, ax = plt.subplots(figsize=(8, max(4, len(df) * 0.45)))
            ax.axvline(0.0, color="dimgray", linestyle="--", alpha=0.7)

            for idx in range(len(df)):
                mean_val = float(df.iloc[idx]["Posterior Mean"])
                hdi_l = float(df.iloc[idx]["95% HDI Lower"])
                hdi_u = float(df.iloc[idx]["95% HDI Upper"])

                if color is not None:
                    line_color = color
                else:
                    if hdi_l <= 0.0 <= hdi_u:
                        line_color = "#CC5555"  # red if the CI crosses 0
                    elif abs(mean_val) > 0.6:
                        line_color = "#82B94C"  # green if magnitude > 0.6
                    else:
                        line_color = (
                            "#D4AC0D"  # yellow if magnitude > 0 and doesn't cross 0
                        )

                ax.hlines(
                    y=idx, xmin=hdi_l, xmax=hdi_u, colors=line_color, linewidth=2.0
                )
                ax.plot(mean_val, idx, "o", color=line_color, markersize=6)

            ax.set_yticks(range(len(df)))
            ax.set_yticklabels(df["Feature"])
            ax.set_xlim(-1.05, 1.05)
            ax.set_xlabel("Correlation Coefficient (ρ)")
            ax.set_ylabel("Feature")
            ax.set_title(
                f"Bayesian Correlation Coefficients\n(credible_mass={self.credible_mass})"
            )
            plt.tight_layout()
            return fig
