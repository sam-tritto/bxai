import numpy as np

from bxai._engines.beta_binomial import BetaBinomialTracker
from bxai._engines.normal_ig import NormalIGTracker
from bxai._utils.types import FeatureStatus

# ===========================================================================
# BetaBinomialTracker
# ===========================================================================


class TestBetaBinomialTracker:
    """Tests for the Beta-Binomial conjugate update engine."""

    # -----------------------------------------------------------------------
    # Smoke / structure
    # -----------------------------------------------------------------------

    def test_initial_state(self):
        tracker = BetaBinomialTracker(n_features=3, prior_alpha=1.0, prior_beta=1.0)
        assert np.all(tracker.alpha == 1.0)
        assert np.all(tracker.beta == 1.0)

    # -----------------------------------------------------------------------
    # Posterior update direction
    # -----------------------------------------------------------------------

    def test_update_increments_alpha_on_hit(self):
        """A hit (1) must increment alpha and leave beta unchanged."""
        tracker = BetaBinomialTracker(n_features=2, prior_alpha=1.0, prior_beta=1.0)
        tracker.update(np.array([1, 0]))
        # Feature 0: hit → alpha grows, beta unchanged
        assert tracker.alpha[0] == 2.0
        assert tracker.beta[0] == 1.0
        # Feature 1: miss → beta grows, alpha unchanged
        assert tracker.alpha[1] == 1.0
        assert tracker.beta[1] == 2.0

    def test_update_increments_beta_on_miss(self):
        """A miss (0) must increment beta and leave alpha unchanged for that feature."""
        tracker = BetaBinomialTracker(n_features=1, prior_alpha=2.0, prior_beta=3.0)
        tracker.update(np.array([0]))
        assert tracker.alpha[0] == 2.0
        assert tracker.beta[0] == 4.0

    def test_repeated_hits_raise_posterior_mean(self):
        """Repeated wins must push the posterior mean strictly above 0.5."""
        tracker = BetaBinomialTracker(n_features=1)
        for _ in range(10):
            tracker.update(np.array([1]))
        posterior_mean = tracker.alpha[0] / (tracker.alpha[0] + tracker.beta[0])
        assert posterior_mean > 0.5

    def test_winner_has_higher_posterior_mean_than_loser(self):
        """Feature that always wins must have a higher posterior mean than one that always loses."""
        tracker = BetaBinomialTracker(n_features=2)
        for _ in range(15):
            tracker.update(
                np.array([1, 0])
            )  # feature 0 always wins, feature 1 always loses
        mean_0 = tracker.alpha[0] / (tracker.alpha[0] + tracker.beta[0])
        mean_1 = tracker.alpha[1] / (tracker.alpha[1] + tracker.beta[1])
        assert mean_0 > mean_1

    def test_partial_update_leaves_other_features_unchanged(self):
        """Updating only a subset of indices must not touch the other features."""
        tracker = BetaBinomialTracker(n_features=3)
        alpha_before = tracker.alpha.copy()
        beta_before = tracker.beta.copy()
        tracker.update(np.array([1, 0]), indices=np.array([0, 2]))
        # Feature 1 must be exactly as initialised
        assert tracker.alpha[1] == alpha_before[1]
        assert tracker.beta[1] == beta_before[1]

    # -----------------------------------------------------------------------
    # Exceedance probability: monotonicity
    # -----------------------------------------------------------------------

    def test_more_wins_gives_higher_exceedance(self):
        """Feature with more hits must have strictly higher P(θ > 0.5)."""
        tracker = BetaBinomialTracker(n_features=2)
        # Feature 0: 10 wins; Feature 1: 3 wins
        tracker.update(np.array([1, 1]))
        for _ in range(9):
            tracker.update(np.array([1, 0]))
        probs = tracker.exceedance_probability(0.5)
        assert probs[0] > probs[1]

    def test_exceedance_probabilities_in_unit_interval(self):
        """All exceedance probabilities must lie in [0, 1]."""
        tracker = BetaBinomialTracker(n_features=4)
        tracker.update(np.array([1, 0, 1, 0]))
        probs = tracker.exceedance_probability(0.5)
        assert np.all(probs >= 0.0)
        assert np.all(probs <= 1.0)

    def test_exceedance_decreases_with_higher_threshold(self):
        """P(θ > high_t) must be ≤ P(θ > low_t) for any feature."""
        tracker = BetaBinomialTracker(n_features=1)
        for _ in range(5):
            tracker.update(np.array([1]))
        prob_low = tracker.exceedance_probability(0.3)
        prob_high = tracker.exceedance_probability(0.7)
        assert prob_low[0] >= prob_high[0]

    def test_exceedance_probability_analytical_value(self):
        """Beta(3,1): P(θ > 0.5) = 1 − 0.5³ = 0.875."""
        tracker = BetaBinomialTracker(n_features=1, prior_alpha=3.0, prior_beta=1.0)
        prob = tracker.exceedance_probability(0.5)
        np.testing.assert_allclose(prob[0], 1 - 0.5**3, rtol=1e-6)

    # -----------------------------------------------------------------------
    # Credible interval ordering and bounds
    # -----------------------------------------------------------------------

    def test_credible_interval_lower_less_than_upper(self):
        """lower must be strictly less than upper for every feature."""
        tracker = BetaBinomialTracker(n_features=3)
        tracker.update(np.array([1, 0, 1]))
        tracker.update(np.array([0, 1, 0]))
        lower, upper = tracker.credible_interval(0.95)
        assert np.all(lower < upper)

    def test_credible_interval_within_unit_interval(self):
        """Beta credible intervals must be contained in [0, 1]."""
        tracker = BetaBinomialTracker(n_features=3)
        for _ in range(8):
            tracker.update(np.array([1, 0, 1]))
        lower, upper = tracker.credible_interval(0.95)
        assert np.all(lower >= 0.0)
        assert np.all(upper <= 1.0)

    def test_wider_credible_mass_gives_wider_interval(self):
        """A 99% CI must be wider than a 90% CI."""
        tracker = BetaBinomialTracker(n_features=1)
        for _ in range(5):
            tracker.update(np.array([1]))
        lo90, hi90 = tracker.credible_interval(0.90)
        lo99, hi99 = tracker.credible_interval(0.99)
        assert (hi99[0] - lo99[0]) > (hi90[0] - lo90[0])

    # -----------------------------------------------------------------------
    # Conjugate-formula mechanics
    # -----------------------------------------------------------------------

    def test_alpha_grows_by_exactly_n_hits(self):
        """After n hits, alpha must equal prior_alpha + n."""
        prior_alpha = 2.0
        tracker = BetaBinomialTracker(
            n_features=1, prior_alpha=prior_alpha, prior_beta=1.0
        )
        n_hits = 7
        for _ in range(n_hits):
            tracker.update(np.array([1]))
        np.testing.assert_allclose(tracker.alpha[0], prior_alpha + n_hits)

    def test_beta_grows_by_exactly_n_misses(self):
        """After n misses, beta must equal prior_beta + n."""
        prior_beta = 3.0
        tracker = BetaBinomialTracker(
            n_features=1, prior_alpha=1.0, prior_beta=prior_beta
        )
        n_misses = 5
        for _ in range(n_misses):
            tracker.update(np.array([0]))
        np.testing.assert_allclose(tracker.beta[0], prior_beta + n_misses)

    # -----------------------------------------------------------------------
    # Decision correctness
    # -----------------------------------------------------------------------

    def test_high_win_rate_leads_to_confirmed(self):
        """Feature with 15 consecutive wins must be CONFIRMED."""
        tracker = BetaBinomialTracker(n_features=1)
        for _ in range(15):
            tracker.update(np.array([1]))
        decisions = tracker.decide(confirm_threshold=0.95, reject_threshold=0.05)
        assert decisions[0] == FeatureStatus.CONFIRMED

    def test_high_loss_rate_leads_to_rejected(self):
        """Feature with 15 consecutive losses must be REJECTED."""
        tracker = BetaBinomialTracker(n_features=1)
        for _ in range(15):
            tracker.update(np.array([0]))
        decisions = tracker.decide(confirm_threshold=0.95, reject_threshold=0.05)
        assert decisions[0] == FeatureStatus.REJECTED

    def test_uncertain_feature_remains_tentative(self):
        """Feature with mixed wins/losses and sparse data should stay TENTATIVE."""
        tracker = BetaBinomialTracker(n_features=1)
        # 3 wins and 3 losses — genuinely uncertain
        tracker.update(np.array([1, 1, 1, 0, 0, 0])[:1])  # single update
        for _ in range(2):
            tracker.update(np.array([1]))
        for _ in range(2):
            tracker.update(np.array([0]))
        decisions = tracker.decide(confirm_threshold=0.99, reject_threshold=0.01)
        assert decisions[0] == FeatureStatus.TENTATIVE

    def test_winner_confirmed_loser_rejected_simultaneously(self):
        """With extreme polarisation, winner → CONFIRMED and loser → REJECTED."""
        tracker = BetaBinomialTracker(n_features=2)
        for _ in range(20):
            tracker.update(np.array([1, 0]))
        decisions = tracker.decide(confirm_threshold=0.95, reject_threshold=0.05)
        assert decisions[0] == FeatureStatus.CONFIRMED
        assert decisions[1] == FeatureStatus.REJECTED


# ===========================================================================
# NormalIGTracker
# ===========================================================================


class TestNormalIGTracker:
    """Tests for the Normal-Inverse-Gamma conjugate update engine."""

    # -----------------------------------------------------------------------
    # Smoke / structure
    # -----------------------------------------------------------------------

    def test_initial_state(self):
        tracker = NormalIGTracker(
            n_features=2, prior_mu=0.0, prior_nu=1e-2, prior_alpha=1e-2, prior_beta=1e-2
        )
        assert np.all(tracker.mu == 0.0)

    # -----------------------------------------------------------------------
    # Posterior update direction
    # -----------------------------------------------------------------------

    def test_positive_data_pushes_mu_positive(self):
        """Positive observations must pull the posterior mean above zero."""
        tracker = NormalIGTracker(n_features=1)
        data = np.array([[1.0], [2.0], [3.0]])
        tracker.update(data)
        assert tracker.mu[0] > 0.0

    def test_negative_data_pushes_mu_negative(self):
        """Negative observations must pull the posterior mean below zero."""
        tracker = NormalIGTracker(n_features=1)
        data = np.array([[-1.0], [-2.0], [-3.0]])
        tracker.update(data)
        assert tracker.mu[0] < 0.0

    def test_directional_separation_between_features(self):
        """Feature 0 (positive data) must have higher posterior mu than feature 1 (negative)."""
        tracker = NormalIGTracker(n_features=2)
        data = np.array(
            [
                [1.0, -1.0],
                [2.0, -2.0],
                [3.0, -3.0],
            ]
        )
        tracker.update(data)
        assert tracker.mu[0] > 0.0
        assert tracker.mu[1] < 0.0
        assert tracker.mu[0] > tracker.mu[1]

    def test_larger_positive_signal_gives_higher_mu(self):
        """Feature receiving data with larger mean must converge to a higher posterior mu."""
        tracker = NormalIGTracker(n_features=2)
        # Feature 0: mean ≈ 5; Feature 1: mean ≈ 1
        data = np.array(
            [
                [4.5, 0.8],
                [5.0, 1.0],
                [5.5, 1.2],
            ]
        )
        tracker.update(data)
        assert tracker.mu[0] > tracker.mu[1]

    def test_partial_update_leaves_untouched_features_unchanged(self):
        """Updating only feature 0 must not change feature 1's parameters."""
        tracker = NormalIGTracker(n_features=2)
        mu1_before = tracker.mu[1]
        nu1_before = tracker.nu[1]
        tracker.update(np.array([[3.0]]), indices=np.array([0]))
        np.testing.assert_allclose(tracker.mu[1], mu1_before)
        np.testing.assert_allclose(tracker.nu[1], nu1_before)

    # -----------------------------------------------------------------------
    # Conjugate-formula mechanics
    # -----------------------------------------------------------------------

    def test_nu_grows_by_n_samples_per_update(self):
        """nu must increase by exactly n_samples after each update."""
        prior_nu = 1e-2
        tracker = NormalIGTracker(n_features=1, prior_nu=prior_nu)
        n_samples = 5
        data = np.random.default_rng(0).standard_normal((n_samples, 1))
        tracker.update(data)
        np.testing.assert_allclose(tracker.nu[0], prior_nu + n_samples)

    def test_alpha_grows_by_half_n_samples_per_update(self):
        """alpha must increase by n_samples / 2 after each update."""
        prior_alpha = 1e-2
        tracker = NormalIGTracker(n_features=1, prior_alpha=prior_alpha)
        n_samples = 6
        data = np.random.default_rng(1).standard_normal((n_samples, 1))
        tracker.update(data)
        np.testing.assert_allclose(tracker.alpha[0], prior_alpha + 0.5 * n_samples)

    def test_cumulative_nu_after_two_updates(self):
        """nu after two sequential updates must equal prior_nu + n1 + n2."""
        prior_nu = 0.01
        tracker = NormalIGTracker(n_features=1, prior_nu=prior_nu)
        rng = np.random.default_rng(42)
        tracker.update(rng.standard_normal((3, 1)))
        tracker.update(rng.standard_normal((7, 1)))
        np.testing.assert_allclose(tracker.nu[0], prior_nu + 3 + 7)

    # -----------------------------------------------------------------------
    # Credible interval ordering
    # -----------------------------------------------------------------------

    def test_credible_interval_lower_less_than_upper(self):
        """lower bound must be strictly less than upper bound for every feature."""
        tracker = NormalIGTracker(n_features=2)
        data = np.array([[1.0, -1.0], [2.0, -2.0], [3.0, -3.0]])
        tracker.update(data)
        lower, upper = tracker.credible_interval(0.95)
        assert np.all(lower < upper)

    def test_credible_interval_for_strongly_positive_feature_is_above_zero(self):
        """After many large positive observations, the 95% CI lower bound must exceed 0."""
        tracker = NormalIGTracker(n_features=1)
        data = (
            np.full((50, 1), 5.0)
            + np.random.default_rng(7).standard_normal((50, 1)) * 0.1
        )
        tracker.update(data)
        lower, upper = tracker.credible_interval(0.95)
        assert lower[0] > 0.0

    def test_wider_credible_mass_gives_wider_interval(self):
        """99% CI must be wider than 90% CI."""
        tracker = NormalIGTracker(n_features=1)
        data = np.array([[1.0], [2.0], [3.0]])
        tracker.update(data)
        lo90, hi90 = tracker.credible_interval(0.90)
        lo99, hi99 = tracker.credible_interval(0.99)
        assert (hi99[0] - lo99[0]) > (hi90[0] - lo90[0])

    # -----------------------------------------------------------------------
    # Decision correctness
    # -----------------------------------------------------------------------

    def test_strongly_positive_feature_confirmed(self):
        """Feature with many large positive observations must be CONFIRMED."""
        tracker = NormalIGTracker(n_features=1)
        data = np.full((50, 1), 5.0)
        tracker.update(data)
        decisions = tracker.decide(credible_mass=0.95, threshold=0.0)
        assert decisions[0] == FeatureStatus.CONFIRMED

    def test_strongly_negative_feature_rejected(self):
        """Feature with many large negative observations must be REJECTED."""
        tracker = NormalIGTracker(n_features=1)
        data = np.full((50, 1), -5.0)
        tracker.update(data)
        decisions = tracker.decide(credible_mass=0.95, threshold=0.0)
        assert decisions[0] == FeatureStatus.REJECTED

    def test_zero_mean_feature_tentative(self):
        """Feature with zero-mean data and weak prior stays TENTATIVE under strict thresholds."""
        rng = np.random.default_rng(99)
        tracker = NormalIGTracker(
            n_features=1, prior_nu=1e-4, prior_alpha=1e-4, prior_beta=1e-4
        )
        # 5 observations centred at zero — interval straddles zero
        data = rng.standard_normal((5, 1)) * 0.05
        tracker.update(data)
        decisions = tracker.decide(credible_mass=0.95, threshold=0.0)
        assert decisions[0] == FeatureStatus.TENTATIVE

    def test_directional_decisions_positive_and_negative(self):
        """Feature 0 (positive) → CONFIRMED; feature 1 (negative) → REJECTED."""
        tracker = NormalIGTracker(n_features=2)
        data = np.column_stack(
            [
                np.full(50, 5.0),
                np.full(50, -5.0),
            ]
        )
        tracker.update(data)
        decisions = tracker.decide(credible_mass=0.95, threshold=0.0)
        assert decisions[0] == FeatureStatus.CONFIRMED
        assert decisions[1] == FeatureStatus.REJECTED

    def test_rope_decisions(self):
        """Test decide with a ROPE parameter (both float and tuple)."""
        tracker = NormalIGTracker(n_features=4)
        tracker.alpha = np.full(4, 50.0)
        tracker.nu = np.full(4, 100.0)
        tracker.beta = np.full(4, 1e-8)

        # Means:
        tracker.mu = np.array([0.003, 0.0, 0.001, -0.003])

        decisions_float = tracker.decide(credible_mass=0.95, rope=0.001)
        assert decisions_float[0] == FeatureStatus.CONFIRMED
        assert decisions_float[1] == FeatureStatus.REJECTED
        assert decisions_float[2] == FeatureStatus.TENTATIVE
        assert decisions_float[3] == FeatureStatus.REJECTED

        # Test tuple rope: (-0.001, 0.002)
        decisions_tuple = tracker.decide(credible_mass=0.95, rope=(-0.001, 0.002))
        assert decisions_tuple[0] == FeatureStatus.CONFIRMED
        assert decisions_tuple[1] == FeatureStatus.REJECTED
        assert decisions_tuple[2] == FeatureStatus.REJECTED
        assert decisions_tuple[3] == FeatureStatus.REJECTED

        # Test validation of invalid ROPE boundaries
        with (
            self.assertRaises(ValueError)
            if hasattr(self, "assertRaises")
            else np.testing.assert_raises(ValueError)
        ):
            tracker.decide(credible_mass=0.95, rope=-0.001)
        with (
            self.assertRaises(ValueError)
            if hasattr(self, "assertRaises")
            else np.testing.assert_raises(ValueError)
        ):
            tracker.decide(credible_mass=0.95, rope=(0.002, 0.001))
        with (
            self.assertRaises(TypeError)
            if hasattr(self, "assertRaises")
            else np.testing.assert_raises(TypeError)
        ):
            tracker.decide(credible_mass=0.95, rope="invalid")
