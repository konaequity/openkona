from __future__ import annotations

import copy
from typing import Any, Callable, Dict, List, Optional


class IterativeTrainingPipeline:
    """Manages the iterative training loop: train, regenerate rollouts, repeat.

    Each iteration consists of:
      1. Train on the current dataset (``run_iteration``).
      2. Regenerate rollouts using the updated checkpoint (``regenerate_rollouts``).
      3. Promote and evaluate the new checkpoint (``promote_checkpoint``,
         ``evaluate_checkpoint``).

    The loop runs for at most ``max_iterations`` rounds.

    Attributes
    ----------
    max_iterations : int
        Maximum number of training iterations (default 3, per the paper).
    """

    max_iterations = 3

    def __init__(
        self,
        max_iterations: int = 3,
        trainer: Any = None,
        rollout_generator: Optional[Callable] = None,
        evaluator: Optional[Callable] = None,
        checkpoint_dir: Optional[str] = None,
    ):
        self.max_iterations = max_iterations
        self.trainer = trainer
        self.rollout_generator = rollout_generator
        self.evaluator = evaluator
        self.checkpoint_dir = checkpoint_dir or "/tmp/konash_checkpoints"
        self._checkpoints: List[Dict[str, Any]] = []
        self._current_iteration = 0
        self._history: List[Dict[str, Any]] = []

    def run_iteration(
        self,
        dataset: Any = None,
        iteration_idx: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Run one full training iteration.

        Trains on the provided dataset, regenerates rollouts, promotes the
        resulting checkpoint, and evaluates it.

        Parameters
        ----------
        dataset:
            The training dataset for this iteration (an
            ``OfflineRolloutDataset`` or compatible).
        iteration_idx:
            Optional explicit iteration number.  If *None*, uses the
            internal counter.

        Returns
        -------
        dict
            Iteration results including ``"iteration"``, ``"train_stats"``,
            ``"eval_results"``, and ``"checkpoint_path"``.
        """
        if iteration_idx is None:
            iteration_idx = self._current_iteration

        if iteration_idx >= self.max_iterations:
            return {
                "iteration": iteration_idx,
                "status": "skipped",
                "reason": f"Exceeded max_iterations ({self.max_iterations})",
            }

        # 1. Train
        train_stats: Dict[str, Any] = {}
        if self.trainer is not None and dataset is not None:
            if hasattr(self.trainer, "train_epoch"):
                train_stats = self.trainer.train_epoch(dataset)
            else:
                train_stats = {"status": "trained"}

        # 2. Promote checkpoint
        checkpoint_info = self.promote_checkpoint(iteration_idx=iteration_idx)

        # 3. Regenerate rollouts for next iteration
        new_rollouts = self.regenerate_rollouts(checkpoint=checkpoint_info)

        # 4. Evaluate
        eval_results = self.evaluate_checkpoint(checkpoint=checkpoint_info)

        result = {
            "iteration": iteration_idx,
            "train_stats": train_stats,
            "eval_results": eval_results,
            "checkpoint_path": checkpoint_info.get("path", ""),
            "new_rollouts": new_rollouts,
        }

        self._history.append(result)
        self._current_iteration = iteration_idx + 1

        return result

    def regenerate_rollouts(
        self,
        checkpoint: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Re-generate rollouts using the current (or specified) checkpoint.

        Parameters
        ----------
        checkpoint:
            Checkpoint metadata dict.  Passed to the rollout generator.

        Returns
        -------
        object
            The newly generated rollouts, or None if no generator is configured.
        """
        if self.rollout_generator is not None:
            return self.rollout_generator(checkpoint)
        return None

    def promote_checkpoint(
        self,
        iteration_idx: Optional[int] = None,
        model_state: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Save and promote a checkpoint.

        Parameters
        ----------
        iteration_idx:
            The iteration number for the checkpoint.
        model_state:
            Optional model state to store.  If *None*, a placeholder is used.

        Returns
        -------
        dict
            Checkpoint metadata including ``"iteration"``, ``"path"``, and
            ``"model_state"``.
        """
        if iteration_idx is None:
            iteration_idx = self._current_iteration

        checkpoint_path = f"{self.checkpoint_dir}/checkpoint_iter_{iteration_idx}"
        checkpoint_info = {
            "iteration": iteration_idx,
            "path": checkpoint_path,
            "model_state": model_state,
        }

        self._checkpoints.append(checkpoint_info)
        return checkpoint_info

    def evaluate_checkpoint(
        self,
        checkpoint: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Evaluate a checkpoint.

        Parameters
        ----------
        checkpoint:
            Checkpoint metadata dict.  Passed to the evaluator.

        Returns
        -------
        dict
            Evaluation results, or an empty dict with status if no evaluator
            is configured.
        """
        if self.evaluator is not None:
            return self.evaluator(checkpoint)
        return {"status": "no_evaluator", "checkpoint": checkpoint}


class IterationDefaults:
    """Paper-derived defaults for the iterative training loop.

    Attributes
    ----------
    max_iterations : int
        Maximum number of KARL iterations (3 in the paper).
    representative_case_task : str
        The task used as the representative case study (TRECBiogen).
    supports_large_batch_offline_updates : bool
        Whether the pipeline supports large-batch offline policy updates.
    """

    max_iterations = 3
    representative_case_task = "TRECBiogen"
    supports_large_batch_offline_updates = True
