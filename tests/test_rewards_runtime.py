from __future__ import annotations

from konash.rewards import RewardRegistry


def test_reward_registry_seeds_all_karlbench_tasks():
    registry = RewardRegistry()
    assert registry.list_rewards() == [
        "BrowseCompPlus",
        "FinanceBench",
        "FreshStack",
        "PMBench",
        "QAMPARI",
        "TRECBiogen",
    ]


def test_qampari_reward_scores_entities_per_nugget():
    registry = RewardRegistry()
    reward = registry.get("QAMPARI")

    assert reward("alpha, beta", "alpha, beta") == 1.0
    assert reward("alpha", "alpha, beta") == 1.0
    assert reward("gamma", "alpha, beta") == 0.0


def test_trecbiogen_reward_consolidates_multiple_references():
    registry = RewardRegistry()
    reward = registry.get("TRECBiogen")

    score = reward(
        "Alpha improves outcomes. Beta increases reliability.",
        reference="",
        references=[
            "Alpha improves outcomes.",
            "Beta increases reliability.",
        ],
    )

    assert score == 1.0


def test_reward_registry_exposes_metadata_for_default_rewards():
    registry = RewardRegistry()
    meta = registry.metadata("BrowseCompPlus")

    assert meta["policy"] == "BrowseCompPlus"
    assert meta["binary"] is True
    assert meta["threshold"] == 0.5
