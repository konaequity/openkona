from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple


class OfflineRolloutDataset:
    """Dataset of offline rollouts grouped by prompt, with associated rewards.

    Each item is a dict with keys 'prompt', 'rollout', and 'reward'.
    Rollouts can be grouped by prompt for group-level value estimation
    as required by the OAPL training objective.
    """

    prompts = None
    group_rollouts = None
    rewards = None

    def __init__(
        self,
        prompts: Optional[List[str]] = None,
        group_rollouts: Optional[List[List[Any]]] = None,
        rewards: Optional[List[List[float]]] = None,
    ):
        self.prompts = prompts or []
        self.group_rollouts = group_rollouts or []
        self.rewards = rewards or []

    @classmethod
    def from_rollouts(
        cls,
        rollout_data: List[Dict[str, Any]],
        prompt_key: str = "prompt",
        rollout_key: str = "rollout",
        reward_key: str = "reward",
    ) -> "OfflineRolloutDataset":
        """Build an OfflineRolloutDataset from a flat list of rollout dicts.

        Each dict in *rollout_data* must contain at minimum the keys specified
        by *prompt_key*, *rollout_key*, and *reward_key*.  Rollouts sharing the
        same prompt value are collected into the same group.

        Parameters
        ----------
        rollout_data:
            List of dicts, each representing a single rollout with its prompt
            and scalar reward.
        prompt_key:
            Key used to look up the prompt string in each dict.
        rollout_key:
            Key used to look up the rollout payload in each dict.
        reward_key:
            Key used to look up the scalar reward in each dict.

        Returns
        -------
        OfflineRolloutDataset
            A new dataset instance with rollouts grouped by prompt.
        """
        grouped: Dict[str, Tuple[List[Any], List[float]]] = {}
        prompt_order: List[str] = []

        for entry in rollout_data:
            prompt = entry[prompt_key]
            rollout = entry[rollout_key]
            reward = float(entry[reward_key])

            if prompt not in grouped:
                grouped[prompt] = ([], [])
                prompt_order.append(prompt)

            grouped[prompt][0].append(rollout)
            grouped[prompt][1].append(reward)

        prompts = prompt_order
        group_rollouts = [grouped[p][0] for p in prompts]
        rewards = [grouped[p][1] for p in prompts]

        return cls(prompts=prompts, group_rollouts=group_rollouts, rewards=rewards)

    def __len__(self) -> int:
        """Return the total number of individual rollouts across all groups."""
        return sum(len(group) for group in self.group_rollouts)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Return the idx-th rollout as a dict with prompt, rollout, and reward.

        Indexing is flattened across groups: group 0's rollouts come first,
        then group 1's, etc.
        """
        current = 0
        for group_idx, group in enumerate(self.group_rollouts):
            if idx < current + len(group):
                local_idx = idx - current
                return {
                    "prompt": self.prompts[group_idx],
                    "rollout": group[local_idx],
                    "reward": self.rewards[group_idx][local_idx],
                }
            current += len(group)
        raise IndexError(f"Index {idx} out of range for dataset of size {len(self)}")

    def group_by_prompt(
        self, rollouts: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Group rollouts by their prompt string.

        Parameters
        ----------
        rollouts:
            Optional flat list of rollout dicts. If *None*, uses the dataset's
            own stored data (flattened via __getitem__).

        Returns
        -------
        dict
            Mapping from prompt string to list of rollout dicts belonging to
            that prompt.
        """
        if rollouts is not None:
            result: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
            for entry in rollouts:
                prompt = entry.get("prompt", "")
                result[prompt].append(entry)
            return dict(result)

        # Use internal data
        result = {}
        for group_idx, prompt in enumerate(self.prompts):
            items = []
            for local_idx in range(len(self.group_rollouts[group_idx])):
                items.append(
                    {
                        "prompt": prompt,
                        "rollout": self.group_rollouts[group_idx][local_idx],
                        "reward": self.rewards[group_idx][local_idx],
                    }
                )
            result[prompt] = items
        return result
