from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import numpy as np


class MultiTaskTrainer:
    """Combine OAPL losses from multiple in-distribution tasks.

    The trainer maintains a registry of tasks, each with its own dataset and
    (optional) loss function.  ``compute_combined_loss`` sums the per-task
    losses, optionally weighted by ``balance_by_training_tokens`` so that
    tasks with fewer tokens are up-weighted.

    Attributes
    ----------
    in_distribution_tasks : set
        The canonical set of in-distribution tasks used during training.
    """

    in_distribution_tasks = {"BrowseCompPlus", "TRECBiogen"}

    def __init__(self):
        self._tasks: Dict[str, Dict[str, Any]] = {}
        self._token_weights: Optional[Dict[str, float]] = None

    def register_task(
        self,
        task_name: str,
        dataset: Any = None,
        loss_fn: Optional[Callable] = None,
        num_training_tokens: Optional[int] = None,
    ) -> None:
        """Register a task with the trainer.

        Parameters
        ----------
        task_name:
            Unique name for the task (e.g. ``"BrowseCompPlus"``).
        dataset:
            The training dataset for this task (an ``OfflineRolloutDataset``
            or compatible object).
        loss_fn:
            Optional callable ``loss_fn(dataset) -> float`` that computes
            the scalar loss for this task.  If *None*, a default zero loss
            is assumed.
        num_training_tokens:
            Total number of training tokens for this task, used by
            ``balance_by_training_tokens`` to compute task weights.
        """
        self._tasks[task_name] = {
            "dataset": dataset,
            "loss_fn": loss_fn,
            "num_training_tokens": num_training_tokens,
        }
        # Invalidate cached weights when a new task is registered
        self._token_weights = None

    def balance_by_training_tokens(self) -> Dict[str, float]:
        """Compute per-task weights inversely proportional to token count.

        Tasks with fewer training tokens receive higher weight so that all
        tasks contribute equally in expectation.  The weights are normalised
        to sum to the number of tasks (so the mean weight is 1.0).

        Returns
        -------
        dict
            Mapping from task name to its normalised weight.
        """
        tasks_with_tokens = {
            name: info["num_training_tokens"]
            for name, info in self._tasks.items()
            if info.get("num_training_tokens") is not None
            and info["num_training_tokens"] > 0
        }

        if not tasks_with_tokens:
            # Fall back to uniform weights
            n = max(len(self._tasks), 1)
            return {name: 1.0 for name in self._tasks}

        # Inverse-token weighting: w_i = 1 / n_tokens_i, then normalise
        inverse_counts = {
            name: 1.0 / count for name, count in tasks_with_tokens.items()
        }
        total_inverse = sum(inverse_counts.values())
        num_tasks = len(tasks_with_tokens)

        weights = {
            name: (inv / total_inverse) * num_tasks
            for name, inv in inverse_counts.items()
        }

        # Tasks without token counts get weight 1.0
        for name in self._tasks:
            if name not in weights:
                weights[name] = 1.0

        self._token_weights = weights
        return weights

    def compute_combined_loss(
        self,
        weights: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """Compute the weighted sum of per-task losses.

        Parameters
        ----------
        weights:
            Optional dict of task-name to weight.  If *None*, uses the weights
            from ``balance_by_training_tokens`` (computing them if necessary).

        Returns
        -------
        dict
            Contains ``"total_loss"`` (float), ``"per_task_losses"`` (dict of
            task name to unweighted loss), and ``"weights"`` (the weights
            used).
        """
        if weights is None:
            if self._token_weights is None:
                weights = self.balance_by_training_tokens()
            else:
                weights = self._token_weights

        per_task_losses: Dict[str, float] = {}
        total_loss = 0.0

        for task_name, info in self._tasks.items():
            loss_fn = info.get("loss_fn")
            dataset = info.get("dataset")

            if loss_fn is not None:
                task_loss = float(loss_fn(dataset))
            else:
                task_loss = 0.0

            per_task_losses[task_name] = task_loss
            task_weight = weights.get(task_name, 1.0)
            total_loss += task_weight * task_loss

        return {
            "total_loss": total_loss,
            "per_task_losses": per_task_losses,
            "weights": weights,
        }
