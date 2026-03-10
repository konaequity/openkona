"""Behavioral tests for OAPL math — verifies the core loss computation
against hand-calculated expected values from the KARL paper (Section 4.2).

These are NOT spec tests (attribute existence). They verify the actual
numerical output of compute_group_value_estimate and
compute_squared_advantage_loss.
"""

import numpy as np
import pytest

from konash.training.oapl import OAPLTrainer


class TestGroupValueEstimate:
    """V̂*(x) = β · ln(1/G · Σ exp(r/β))  [paper eq. 1]"""

    def test_uniform_rewards(self):
        """When all rewards are equal, V̂* should equal that reward."""
        trainer = OAPLTrainer(beta_value=0.05)
        v = trainer.compute_group_value_estimate([0.5, 0.5, 0.5, 0.5])
        assert abs(v - 0.5) < 1e-10

    def test_single_rollout(self):
        """With G=1, V̂* = β · ln(exp(r/β)) = r."""
        trainer = OAPLTrainer(beta_value=0.05)
        v = trainer.compute_group_value_estimate([0.7])
        assert abs(v - 0.7) < 1e-10

    def test_binary_rewards_hand_calculated(self):
        """Hand-calculated: rewards [0, 0, 1, 0], beta=0.05.

        V̂* = 0.05 * (logsumexp([0, 0, 20, 0]) - ln(4))
            = 0.05 * (20 + ln(3*exp(-20) + exp(0)) - ln(4))
            ≈ 0.05 * (20 + ln(1) - ln(4))
            = 0.05 * (20 - 1.3863)
            ≈ 0.9307
        """
        trainer = OAPLTrainer(beta_value=0.05)
        v = trainer.compute_group_value_estimate([0.0, 0.0, 1.0, 0.0])
        # The passing rollout (r=1) should pull V̂* close to 1.0 because
        # beta is small → logsumexp is dominated by the max.
        assert 0.92 < v < 0.95, f"Expected ~0.931, got {v}"

    def test_all_zero_rewards(self):
        """V̂*(all zeros) = 0."""
        trainer = OAPLTrainer(beta_value=0.05)
        v = trainer.compute_group_value_estimate([0.0, 0.0, 0.0])
        assert abs(v) < 1e-10

    def test_v_star_between_min_and_max(self):
        """V̂* should always lie between min and max reward."""
        trainer = OAPLTrainer(beta_value=0.05)
        rewards = [0.0, 0.3, 0.7, 1.0]
        v = trainer.compute_group_value_estimate(rewards)
        assert min(rewards) <= v <= max(rewards)

    def test_higher_beta_softens_estimate(self):
        """Larger beta → V̂* closer to mean. Smaller beta → closer to max."""
        rewards = [0.0, 0.0, 1.0, 0.0]
        v_small_beta = OAPLTrainer(beta_value=0.01).compute_group_value_estimate(rewards)
        v_large_beta = OAPLTrainer(beta_value=1.0).compute_group_value_estimate(rewards)
        # Small beta → V̂* near max (1.0)
        assert v_small_beta > 0.95
        # Large beta → V̂* near mean (0.25)
        assert v_large_beta < 0.6


class TestSquaredAdvantageLoss:
    """L = mean((β_kl · log(π/πref) - (r - V̂*))²)"""

    def test_zero_kl_positive_advantage(self):
        """When log_ratio=0 and advantage>0, loss = advantage²."""
        trainer = OAPLTrainer(beta_kl=0.1)
        loss = trainer.compute_squared_advantage_loss(
            log_ratios=np.array([0.0]),
            rewards=np.array([1.0]),
            value_estimates=np.array([0.5]),
        )
        # advantage = 1.0 - 0.5 = 0.5; kl_term = 0
        # loss = (0 - 0.5)² = 0.25
        assert abs(loss - 0.25) < 1e-10

    def test_kl_matches_advantage_gives_zero_loss(self):
        """When kl_term exactly equals advantage, loss = 0."""
        trainer = OAPLTrainer(beta_kl=0.1)
        # advantage = 1.0 - 0.5 = 0.5
        # kl_term = 0.1 * log_ratio = 0.5 → log_ratio = 5.0
        loss = trainer.compute_squared_advantage_loss(
            log_ratios=np.array([5.0]),
            rewards=np.array([1.0]),
            value_estimates=np.array([0.5]),
        )
        assert abs(loss) < 1e-10

    def test_mask_excludes_tokens(self):
        """Masked tokens should not contribute to loss."""
        trainer = OAPLTrainer(beta_kl=0.1)
        loss = trainer.compute_squared_advantage_loss(
            log_ratios=np.array([0.0, 999.0, 0.0]),
            rewards=np.array([1.0, 1.0, 1.0]),
            value_estimates=np.array([0.5, 0.5, 0.5]),
            mask=np.array([True, False, True]),
        )
        # Only indices 0 and 2 count; both have log_ratio=0, advantage=0.5
        # loss = mean(0.25, 0.25) = 0.25
        assert abs(loss - 0.25) < 1e-10

    def test_negative_advantage_increases_loss(self):
        """A failing rollout (r < V̂*) should produce positive loss."""
        trainer = OAPLTrainer(beta_kl=0.1)
        loss = trainer.compute_squared_advantage_loss(
            log_ratios=np.array([0.0]),
            rewards=np.array([0.0]),
            value_estimates=np.array([0.8]),
        )
        # advantage = 0.0 - 0.8 = -0.8
        # loss = (0 - (-0.8))² = 0.64
        assert abs(loss - 0.64) < 1e-10

    def test_full_oapl_pipeline_hand_calculated(self):
        """End-to-end: 4 rollouts with binary rewards, verify loss."""
        trainer = OAPLTrainer(beta_kl=0.1, beta_value=0.05)

        rewards = [0.0, 0.0, 1.0, 0.0]
        v_star = trainer.compute_group_value_estimate(rewards)

        # With log_ratios all 0 (policy = reference):
        # loss_i = (0 - (r_i - v_star))² = (r_i - v_star)²
        log_ratios = np.zeros(4)
        rewards_arr = np.array(rewards)
        v_arr = np.full(4, v_star)

        loss = trainer.compute_squared_advantage_loss(
            log_ratios, rewards_arr, v_arr
        )

        # Manually compute expected
        advantages = rewards_arr - v_star
        expected = float(np.mean(advantages ** 2))
        assert abs(loss - expected) < 1e-10

    def test_all_masked_returns_zero(self):
        """If all tokens are masked, loss should be 0."""
        trainer = OAPLTrainer(beta_kl=0.1)
        loss = trainer.compute_squared_advantage_loss(
            log_ratios=np.array([5.0, 10.0]),
            rewards=np.array([1.0, 0.0]),
            value_estimates=np.array([0.5, 0.5]),
            mask=np.array([False, False]),
        )
        assert loss == 0.0
