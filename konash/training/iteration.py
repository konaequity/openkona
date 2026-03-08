"""Iterative training pipeline with real checkpoint persistence and caching.

Implements the KARL paper's core loop (Section 4.2, Figure 9):
  1. Train on the current dataset using OAPL.
  2. Save the trained LoRA adapter as a checkpoint.
  3. Regenerate all rollouts from scratch using the improved model.
  4. Evaluate the new checkpoint.
  5. The new checkpoint becomes the reference policy for the next iteration.

Supports 2-3 iterations, with each iteration yielding compounding gains.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class IterativeTrainingPipeline:
    """Manages the iterative training loop: train, checkpoint, regenerate, repeat.

    Each iteration consists of:
      1. Train on the current dataset (``run_iteration``).
      2. Save LoRA adapter weights to disk (``promote_checkpoint``).
      3. Regenerate rollouts using the updated model (``regenerate_rollouts``).
      4. Evaluate the new checkpoint (``evaluate_checkpoint``).

    The loop runs for at most ``max_iterations`` rounds.

    Attributes
    ----------
    max_iterations : int
        Maximum number of training iterations (default 3, per the paper).
    checkpoint_dir : str
        Directory for saving checkpoints and intermediate results.
    """

    max_iterations = 3

    def __init__(
        self,
        max_iterations: int = 3,
        trainer: Any = None,
        model_engine: Any = None,
        rollout_generator: Optional[Callable] = None,
        evaluator: Optional[Callable] = None,
        checkpoint_dir: Optional[str] = None,
    ):
        self.max_iterations = max_iterations
        self.trainer = trainer
        self.model_engine = model_engine
        self.rollout_generator = rollout_generator
        self.evaluator = evaluator
        self.checkpoint_dir = checkpoint_dir or "./konash_checkpoints"
        self._checkpoints: List[Dict[str, Any]] = []
        self._current_iteration = 0
        self._history: List[Dict[str, Any]] = []

        # Ensure checkpoint directory exists
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Full loop
    # ------------------------------------------------------------------

    def run_all(
        self,
        initial_dataset: Any = None,
    ) -> List[Dict[str, Any]]:
        """Run the full iterative training loop for ``max_iterations``.

        Parameters
        ----------
        initial_dataset :
            The training dataset for the first iteration.

        Returns
        -------
        list[dict]
            Results from each iteration.
        """
        dataset = initial_dataset
        results = []

        for i in range(self.max_iterations):
            logger.info("=" * 60)
            logger.info("Starting iteration %d / %d", i + 1, self.max_iterations)
            logger.info("=" * 60)

            result = self.run_iteration(dataset=dataset, iteration_idx=i)
            results.append(result)

            if result.get("status") == "skipped":
                logger.info("Iteration %d skipped: %s", i, result.get("reason"))
                break

            # Use regenerated rollouts as the dataset for the next iteration
            new_rollouts = result.get("new_rollouts")
            if new_rollouts is not None:
                dataset = new_rollouts

            logger.info(
                "Iteration %d complete. Train loss: %.4f, Eval: %s",
                i + 1,
                result.get("train_stats", {}).get("mean_loss", float("nan")),
                result.get("eval_results", {}).get("status", "n/a"),
            )

        self._save_run_history()
        return results

    def run_iteration(
        self,
        dataset: Any = None,
        iteration_idx: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Run one full training iteration.

        Trains on the provided dataset, saves checkpoint, regenerates rollouts,
        promotes and evaluates.

        Parameters
        ----------
        dataset:
            The training dataset for this iteration.
        iteration_idx:
            Optional explicit iteration number.

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

        t_start = time.time()

        # 1. Train
        train_stats: Dict[str, Any] = {}
        if self.trainer is not None and dataset is not None:
            logger.info("Training iteration %d...", iteration_idx)
            if hasattr(self.trainer, "train_epoch"):
                train_stats = self.trainer.train_epoch(dataset)
            elif hasattr(self.trainer, "train_epoch_torch"):
                train_stats = self.trainer.train_epoch_torch(dataset)
            else:
                train_stats = {"status": "no_train_method"}

        # 2. Save checkpoint
        checkpoint_info = self.promote_checkpoint(iteration_idx=iteration_idx)

        # 3. Regenerate rollouts for next iteration
        new_rollouts = self.regenerate_rollouts(checkpoint=checkpoint_info)

        # 4. Evaluate
        eval_results = self.evaluate_checkpoint(checkpoint=checkpoint_info)

        elapsed = time.time() - t_start
        result = {
            "iteration": iteration_idx,
            "train_stats": train_stats,
            "eval_results": eval_results,
            "checkpoint_path": checkpoint_info.get("path", ""),
            "new_rollouts": new_rollouts,
            "elapsed_seconds": elapsed,
            "status": "completed",
        }

        self._history.append(result)
        self._current_iteration = iteration_idx + 1

        return result

    def regenerate_rollouts(
        self,
        checkpoint: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Re-generate all rollouts from scratch using the latest checkpoint.

        This is the key step that makes iterative training work: the improved
        model generates better trajectories, providing richer training signal.

        Parameters
        ----------
        checkpoint:
            Checkpoint metadata dict.

        Returns
        -------
        object
            The newly generated rollouts, or None if no generator is configured.
        """
        if self.rollout_generator is None:
            return None

        logger.info(
            "Regenerating rollouts with checkpoint: %s",
            checkpoint.get("path", "unknown") if checkpoint else "none",
        )

        # Check for cached rollouts
        cache_path = self._rollout_cache_path(checkpoint)
        if cache_path and os.path.exists(cache_path):
            logger.info("Loading cached rollouts from %s", cache_path)
            return self._load_cached_rollouts(cache_path)

        rollouts = self.rollout_generator(checkpoint)

        # Cache the rollouts for recovery
        if cache_path and rollouts is not None:
            self._save_cached_rollouts(rollouts, cache_path)

        return rollouts

    def promote_checkpoint(
        self,
        iteration_idx: Optional[int] = None,
        model_state: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Save LoRA adapter weights and metadata to disk.

        Parameters
        ----------
        iteration_idx:
            The iteration number for the checkpoint.
        model_state:
            Optional explicit model state.

        Returns
        -------
        dict
            Checkpoint metadata including path, iteration, and timestamp.
        """
        if iteration_idx is None:
            iteration_idx = self._current_iteration

        checkpoint_path = os.path.join(
            self.checkpoint_dir, f"checkpoint_iter_{iteration_idx}"
        )
        os.makedirs(checkpoint_path, exist_ok=True)

        # Save LoRA adapter if model engine is available
        if self.model_engine is not None and hasattr(self.model_engine, "save_adapter"):
            adapter_path = os.path.join(checkpoint_path, "adapter")
            self.model_engine.save_adapter(adapter_path)
            logger.info("Saved LoRA adapter to %s", adapter_path)

        # Save metadata
        metadata = {
            "iteration": iteration_idx,
            "path": checkpoint_path,
            "timestamp": time.time(),
            "has_adapter": self.model_engine is not None,
        }

        if model_state is not None:
            metadata["model_state"] = str(model_state)

        # Write metadata to disk
        meta_path = os.path.join(checkpoint_path, "metadata.json")
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)

        self._checkpoints.append(metadata)
        logger.info("Checkpoint %d saved to %s", iteration_idx, checkpoint_path)
        return metadata

    def evaluate_checkpoint(
        self,
        checkpoint: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Evaluate a checkpoint.

        Parameters
        ----------
        checkpoint:
            Checkpoint metadata dict.

        Returns
        -------
        dict
            Evaluation results.
        """
        if self.evaluator is not None:
            return self.evaluator(checkpoint)
        return {"status": "no_evaluator", "checkpoint": checkpoint}

    def load_checkpoint(self, iteration_idx: int) -> Optional[Dict[str, Any]]:
        """Load a previously saved checkpoint.

        Parameters
        ----------
        iteration_idx:
            Which iteration's checkpoint to load.

        Returns
        -------
        dict or None
            Checkpoint metadata, or None if not found.
        """
        checkpoint_path = os.path.join(
            self.checkpoint_dir, f"checkpoint_iter_{iteration_idx}"
        )
        meta_path = os.path.join(checkpoint_path, "metadata.json")

        if not os.path.exists(meta_path):
            return None

        with open(meta_path) as f:
            metadata = json.load(f)

        # Load LoRA adapter if model engine is available
        adapter_path = os.path.join(checkpoint_path, "adapter")
        if (
            self.model_engine is not None
            and hasattr(self.model_engine, "model")
            and os.path.exists(adapter_path)
        ):
            try:
                from peft import PeftModel
                if isinstance(self.model_engine.model, PeftModel):
                    self.model_engine.model.load_adapter(adapter_path, "default")
                    logger.info("Loaded LoRA adapter from %s", adapter_path)
            except Exception as exc:
                logger.warning("Failed to load adapter from %s: %s", adapter_path, exc)

        return metadata

    def resume_from(self, iteration_idx: int) -> bool:
        """Resume training from a specific iteration checkpoint.

        Loads the checkpoint and sets the internal iteration counter.

        Returns True if successful, False if the checkpoint doesn't exist.
        """
        metadata = self.load_checkpoint(iteration_idx)
        if metadata is None:
            return False

        self._current_iteration = iteration_idx + 1
        self._checkpoints.append(metadata)
        logger.info("Resumed from iteration %d", iteration_idx)
        return True

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def _rollout_cache_path(
        self, checkpoint: Optional[Dict[str, Any]]
    ) -> Optional[str]:
        """Get the cache path for rollouts associated with a checkpoint."""
        if checkpoint is None:
            return None
        cp_path = checkpoint.get("path")
        if cp_path:
            return os.path.join(cp_path, "rollouts_cache.json")
        return None

    @staticmethod
    def _save_cached_rollouts(rollouts: Any, path: str) -> None:
        """Save rollouts to a JSON cache file."""
        try:
            with open(path, "w") as f:
                json.dump(rollouts, f, default=str)
            logger.info("Cached rollouts to %s", path)
        except Exception as exc:
            logger.warning("Failed to cache rollouts: %s", exc)

    @staticmethod
    def _load_cached_rollouts(path: str) -> Any:
        """Load rollouts from a JSON cache file."""
        with open(path) as f:
            return json.load(f)

    def _save_run_history(self) -> None:
        """Persist the full training history to disk."""
        history_path = os.path.join(self.checkpoint_dir, "training_history.json")
        try:
            # Filter out non-serializable fields
            serializable = []
            for entry in self._history:
                clean = {
                    k: v for k, v in entry.items()
                    if k != "new_rollouts"  # rollouts are cached separately
                }
                serializable.append(clean)
            with open(history_path, "w") as f:
                json.dump(serializable, f, indent=2, default=str)
            logger.info("Training history saved to %s", history_path)
        except Exception as exc:
            logger.warning("Failed to save history: %s", exc)


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
