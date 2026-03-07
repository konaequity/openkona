from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import numpy as np


class OAPLTrainer:
    """Trainer implementing the Offline Advantage-weighted Policy Learning (OAPL)
    objective with squared-advantage loss.

    The core OAPL loss for each rollout is:

        L = (beta_kl * ln(pi(y|x) / pi_ref(y|x))  -  (r - V_hat_star))^2

    where V_hat_star is the soft optimal value estimated via logsumexp over
    group rollout rewards:

        V_hat_star = beta_value * logsumexp(rewards / beta_value)

    Attributes
    ----------
    reference_policy : callable or None
        The frozen reference policy pi_ref used for KL regularisation.
    beta_value : float
        Temperature for the soft value estimation (default 0.05).
    beta_kl : float
        Coefficient for the KL-divergence term (default 0.1).
    """

    reference_policy = None
    beta_value = 0.05
    beta_kl = 0.1

    def __init__(
        self,
        reference_policy: Optional[Callable] = None,
        beta_value: float = 0.05,
        beta_kl: float = 0.1,
    ):
        self.reference_policy = reference_policy
        self.beta_value = beta_value
        self.beta_kl = beta_kl

    # ------------------------------------------------------------------
    # Core math
    # ------------------------------------------------------------------

    def compute_group_value_estimate(self, group_rewards: List[float]) -> float:
        """Compute the soft optimal value V* for a group of rollouts.

        Uses the logsumexp trick for numerical stability:

            V* = beta_value * logsumexp(rewards / beta_value)
               = beta_value * ( max_r + log(sum(exp((r - max_r) / beta_value))) )

        This is equivalent to ``estimate_optimal_value`` but named to match
        the paper's "group value estimate" terminology.

        Parameters
        ----------
        group_rewards:
            Scalar rewards for every rollout in a prompt group.

        Returns
        -------
        float
            The estimated optimal value V_hat_star.
        """
        rewards = np.asarray(group_rewards, dtype=np.float64)
        scaled = rewards / self.beta_value
        # logsumexp for numerical stability
        max_scaled = np.max(scaled)
        lse = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
        return float(self.beta_value * lse)

    def estimate_optimal_value(self, group_rewards: List[float]) -> float:
        """Estimate V* using logsumexp over group rollout rewards.

        Alias for ``compute_group_value_estimate``.
        """
        return self.compute_group_value_estimate(group_rewards)

    def compute_squared_advantage_loss(
        self,
        log_ratios: np.ndarray,
        rewards: np.ndarray,
        value_estimates: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> float:
        """Compute the OAPL squared-advantage loss.

        For each rollout *i*:

            advantage_i = reward_i - V_hat_star_i
            kl_term_i   = beta_kl * log_ratio_i          (log(pi / pi_ref))
            loss_i       = (kl_term_i - advantage_i) ** 2

        The final loss is the mean over all (unmasked) tokens / rollouts.

        Parameters
        ----------
        log_ratios:
            Array of shape (N,) containing log(pi(y|x) / pi_ref(y|x)) for
            each rollout (or per-token, if token-level).
        rewards:
            Array of shape (N,) with scalar rewards per rollout.
        value_estimates:
            Array of shape (N,) with V_hat_star per rollout (one per group,
            broadcast to each rollout in the group).
        mask:
            Optional boolean array of shape (N,).  True keeps the element,
            False masks it out (e.g. tool-output tokens).

        Returns
        -------
        float
            Mean squared-advantage loss.
        """
        log_ratios = np.asarray(log_ratios, dtype=np.float64)
        rewards = np.asarray(rewards, dtype=np.float64)
        value_estimates = np.asarray(value_estimates, dtype=np.float64)

        advantages = rewards - value_estimates
        kl_terms = self.beta_kl * log_ratios
        residuals = kl_terms - advantages
        elementwise_loss = residuals ** 2

        if mask is not None:
            mask = np.asarray(mask, dtype=bool)
            elementwise_loss = elementwise_loss[mask]

        if elementwise_loss.size == 0:
            return 0.0

        return float(np.mean(elementwise_loss))

    def mask_non_model_tokens(
        self,
        token_ids: List[int],
        tool_output_ranges: Optional[List[tuple]] = None,
    ) -> np.ndarray:
        """Create a boolean mask that is True for model-generated tokens and
        False for tool-output (non-model) tokens.

        Parameters
        ----------
        token_ids:
            Full sequence of token ids for a rollout.
        tool_output_ranges:
            List of (start, end) index pairs marking spans that are tool
            outputs and should be masked out.

        Returns
        -------
        np.ndarray
            Boolean array of length ``len(token_ids)``.  True = include in
            loss, False = mask out.
        """
        mask = np.ones(len(token_ids), dtype=bool)
        if tool_output_ranges:
            for start, end in tool_output_ranges:
                mask[start:end] = False
        return mask

    # ------------------------------------------------------------------
    # PyTorch training (real gradient updates)
    # ------------------------------------------------------------------

    def train_epoch_torch(
        self,
        dataset: Any,
        model_engine: Any,
        learning_rate: float = 1e-5,
        max_grad_norm: float = 1.0,
    ) -> Dict[str, float]:
        """Run one OAPL training epoch with real PyTorch gradients.

        Computes per-token log-ratios between the current (LoRA) policy
        and the frozen reference (base) policy, then applies the
        squared-advantage OAPL loss with gradient accumulation per group.

        Parameters
        ----------
        dataset :
            An ``OfflineRolloutDataset`` providing ``prompts``,
            ``group_rollouts``, and ``rewards``.
        model_engine :
            A ``LocalModelEngine`` (or compatible) that provides
            ``tokenize_rollout``, ``compute_log_probs``, and
            ``trainable_params``.
        learning_rate : float
            Learning rate for AdamW.
        max_grad_norm : float
            Maximum gradient norm for clipping.

        Returns
        -------
        dict
            Training statistics: ``mean_loss``, ``num_groups``,
            ``num_rollouts``, ``learning_rate``.
        """
        import torch

        optimizer = torch.optim.AdamW(
            model_engine.trainable_params, lr=learning_rate, weight_decay=0.01,
        )

        total_loss = 0.0
        num_groups = 0
        num_rollouts = 0

        model_engine.model.train()

        for group_idx, prompt in enumerate(dataset.prompts):
            rollouts = dataset.group_rollouts[group_idx]
            group_rewards = dataset.rewards[group_idx]
            v_star = self.estimate_optimal_value(group_rewards)
            n_rollouts = len(rollouts)

            optimizer.zero_grad()
            group_loss_val = 0.0

            for local_idx, rollout in enumerate(rollouts):
                reward = group_rewards[local_idx]

                # Tokenize
                tokens = model_engine.tokenize_rollout(prompt, rollout)
                input_ids = tokens["input_ids"]
                labels = tokens["labels"]

                # Current policy log-probs (with gradient)
                log_probs, mask = model_engine.compute_log_probs(
                    input_ids, labels, use_reference=False,
                )

                # Reference policy log-probs (no gradient, LoRA disabled)
                with torch.no_grad():
                    ref_log_probs, _ = model_engine.compute_log_probs(
                        input_ids, labels, use_reference=True,
                    )

                # Mean log-ratio over valid tokens
                valid_count = mask.sum()
                if valid_count == 0:
                    continue

                log_ratio = (log_probs - ref_log_probs)
                mean_log_ratio = log_ratio.sum() / valid_count

                # OAPL squared-advantage loss
                advantage = reward - v_star
                kl_term = self.beta_kl * mean_log_ratio
                loss_i = (kl_term - advantage) ** 2

                # Scale by 1/n_rollouts for gradient accumulation
                (loss_i / n_rollouts).backward()

                group_loss_val += loss_i.item()
                num_rollouts += 1

            # Clip gradients and step
            torch.nn.utils.clip_grad_norm_(
                model_engine.trainable_params, max_norm=max_grad_norm,
            )
            optimizer.step()

            if n_rollouts > 0:
                group_loss_val /= n_rollouts
            total_loss += group_loss_val
            num_groups += 1

        model_engine.model.eval()
        mean_loss = total_loss / max(num_groups, 1)

        return {
            "mean_loss": float(mean_loss),
            "num_groups": num_groups,
            "num_rollouts": num_rollouts,
            "learning_rate": learning_rate,
        }

    # ------------------------------------------------------------------
    # High-level training API (numpy reference)
    # ------------------------------------------------------------------

    def compute_loss(
        self,
        log_probs: np.ndarray,
        ref_log_probs: np.ndarray,
        rewards: np.ndarray,
        group_indices: List[List[int]],
        mask: Optional[np.ndarray] = None,
    ) -> float:
        """Compute the full OAPL loss over a batch.

        Parameters
        ----------
        log_probs:
            Shape (N,) log-probabilities under the current policy.
        ref_log_probs:
            Shape (N,) log-probabilities under the reference policy.
        rewards:
            Shape (N,) scalar rewards per rollout.
        group_indices:
            List of lists; each inner list contains the indices of rollouts
            belonging to the same prompt group.
        mask:
            Optional boolean mask of shape (N,).

        Returns
        -------
        float
            Mean OAPL squared-advantage loss.
        """
        log_probs = np.asarray(log_probs, dtype=np.float64)
        ref_log_probs = np.asarray(ref_log_probs, dtype=np.float64)
        rewards = np.asarray(rewards, dtype=np.float64)

        log_ratios = log_probs - ref_log_probs

        # Build per-rollout value estimates from group rewards
        value_estimates = np.zeros_like(rewards)
        for group in group_indices:
            group_rewards = rewards[group].tolist()
            v_star = self.estimate_optimal_value(group_rewards)
            for idx in group:
                value_estimates[idx] = v_star

        return self.compute_squared_advantage_loss(
            log_ratios, rewards, value_estimates, mask=mask
        )

    def train_epoch(
        self,
        dataset: Any,
        policy_fn: Optional[Callable] = None,
        learning_rate: float = 1e-5,
    ) -> Dict[str, float]:
        """Run one training epoch over the dataset.

        This is a reference implementation that computes OAPL losses over
        each prompt group in the dataset.  In production this would be
        integrated with a deep-learning framework's optimiser loop.

        Parameters
        ----------
        dataset:
            An ``OfflineRolloutDataset`` (or compatible) providing
            ``prompts``, ``group_rollouts``, and ``rewards`` attributes.
        policy_fn:
            Optional callable ``policy_fn(prompt, rollout) -> log_prob``
            returning the current policy's log-probability.  If *None*,
            dummy log-probs of 0.0 are used (useful for testing).
        learning_rate:
            Learning rate (informational; the reference loop performs a
            single pseudo-gradient step per group).

        Returns
        -------
        dict
            Training statistics for the epoch: ``mean_loss``,
            ``num_groups``, and ``num_rollouts``.
        """
        total_loss = 0.0
        num_groups = 0
        num_rollouts = 0

        for group_idx, prompt in enumerate(dataset.prompts):
            rollouts = dataset.group_rollouts[group_idx]
            group_rewards = dataset.rewards[group_idx]
            v_star = self.estimate_optimal_value(group_rewards)

            group_loss = 0.0
            for local_idx, rollout in enumerate(rollouts):
                reward = group_rewards[local_idx]

                if policy_fn is not None:
                    log_prob = policy_fn(prompt, rollout)
                else:
                    log_prob = 0.0

                if self.reference_policy is not None:
                    ref_log_prob = self.reference_policy(prompt, rollout)
                else:
                    ref_log_prob = 0.0

                log_ratio = log_prob - ref_log_prob
                advantage = reward - v_star
                kl_term = self.beta_kl * log_ratio
                loss_i = (kl_term - advantage) ** 2
                group_loss += loss_i
                num_rollouts += 1

            if len(rollouts) > 0:
                group_loss /= len(rollouts)

            total_loss += group_loss
            num_groups += 1

        mean_loss = total_loss / max(num_groups, 1)

        return {
            "mean_loss": float(mean_loss),
            "num_groups": num_groups,
            "num_rollouts": num_rollouts,
            "learning_rate": learning_rate,
        }
