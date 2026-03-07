from __future__ import annotations


class TrainingPromptStats:
    iteration_name = None
    task_name = None
    num_training_prompts = None

    def __init__(self, iteration_name=None, task_name=None, num_training_prompts=None):
        self.iteration_name = iteration_name
        self.task_name = task_name
        self.num_training_prompts = num_training_prompts


class TrainingPromptStatsRegistry:
    stats = {
        ("KARL Iter. 1", "BrowseCompPlus"): TrainingPromptStats(
            iteration_name="KARL Iter. 1",
            task_name="BrowseCompPlus",
            num_training_prompts=1218,
        ),
        ("KARL Iter. 1", "TRECBiogen"): TrainingPromptStats(
            iteration_name="KARL Iter. 1",
            task_name="TRECBiogen",
            num_training_prompts=6270,
        ),
        ("KARL Iter. 2", "BrowseCompPlus"): TrainingPromptStats(
            iteration_name="KARL Iter. 2",
            task_name="BrowseCompPlus",
            num_training_prompts=1336,
        ),
        ("KARL Iter. 2", "TRECBiogen"): TrainingPromptStats(
            iteration_name="KARL Iter. 2",
            task_name="TRECBiogen",
            num_training_prompts=11371,
        ),
    }
