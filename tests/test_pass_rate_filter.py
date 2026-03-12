"""Tests for PassRateFilter — verifies filtering logic, binarization,
and adaptive thresholds without requiring any API calls."""

import pytest

from konash.synthesis.filters import PassRateFilter


class FakeGroup:
    """Minimal rollout group stub with a pass_rate attribute."""

    def __init__(self, pass_rate):
        self.pass_rate = pass_rate


class TestPassRateFilterApply:

    def test_keeps_groups_within_range(self):
        f = PassRateFilter(min_pass_rate=0.1, max_pass_rate=0.9)
        groups = [FakeGroup(r) for r in [0.0, 0.1, 0.5, 0.9, 1.0]]
        kept = f.apply(groups)
        rates = [g.pass_rate for g in kept]
        assert rates == [0.1, 0.5, 0.9]

    def test_rejects_all_zero_pass_rate(self):
        f = PassRateFilter(min_pass_rate=0.1, max_pass_rate=0.9)
        kept = f.apply([FakeGroup(0.0)])
        assert kept == []

    def test_rejects_all_one_pass_rate(self):
        f = PassRateFilter(min_pass_rate=0.1, max_pass_rate=0.9)
        kept = f.apply([FakeGroup(1.0)])
        assert kept == []

    def test_no_thresholds_keeps_everything(self):
        f = PassRateFilter()
        groups = [FakeGroup(r) for r in [0.0, 0.5, 1.0]]
        kept = f.apply(groups)
        assert len(kept) == 3

    def test_min_only(self):
        f = PassRateFilter(min_pass_rate=0.3)
        groups = [FakeGroup(r) for r in [0.1, 0.3, 0.7]]
        kept = f.apply(groups)
        assert [g.pass_rate for g in kept] == [0.3, 0.7]

    def test_max_only(self):
        f = PassRateFilter(max_pass_rate=0.5)
        groups = [FakeGroup(r) for r in [0.1, 0.5, 0.9]]
        kept = f.apply(groups)
        assert [g.pass_rate for g in kept] == [0.1, 0.5]

    def test_dict_groups(self):
        """Filter should also work with dict-style groups."""
        f = PassRateFilter(min_pass_rate=0.1, max_pass_rate=0.9)
        groups = [{"pass_rate": 0.0}, {"pass_rate": 0.5}, {"pass_rate": 1.0}]
        kept = f.apply(groups)
        assert len(kept) == 1
        assert kept[0]["pass_rate"] == 0.5

    def test_empty_input(self):
        f = PassRateFilter(min_pass_rate=0.1, max_pass_rate=0.9)
        assert f.apply([]) == []

    def test_group_without_pass_rate_is_skipped(self):
        f = PassRateFilter(min_pass_rate=0.1, max_pass_rate=0.9)
        kept = f.apply([object(), FakeGroup(0.5)])
        assert len(kept) == 1


class TestBinarization:

    def test_default_threshold_is_0_5(self):
        f = PassRateFilter()
        assert f.binarization_threshold == 0.5

    def test_binarize_scores(self):
        f = PassRateFilter()
        result = f.binarize_scores([0.0, 0.3, 0.5, 0.7, 1.0])
        assert result == [0.0, 0.0, 1.0, 1.0, 1.0]

    def test_task_specific_threshold(self):
        f = PassRateFilter(task_name="TRECBiogen", iteration=1)
        assert f.binarization_threshold == 0.7
        result = f.binarize_scores([0.6, 0.7, 0.8])
        assert result == [0.0, 1.0, 1.0]


class TestAdaptiveThresholds:

    def test_browsecomp_uses_adaptive(self):
        f = PassRateFilter(task_name="BrowseCompPlus", iteration=0)
        assert f.min_pass_rate == 0.1
        assert f.max_pass_rate == 0.9

    def test_explicit_overrides_adaptive(self):
        f = PassRateFilter(
            task_name="BrowseCompPlus", iteration=0,
            min_pass_rate=0.2, max_pass_rate=0.8,
        )
        assert f.min_pass_rate == 0.2
        assert f.max_pass_rate == 0.8

    def test_unknown_task_no_thresholds(self):
        f = PassRateFilter(task_name="UnknownTask", iteration=0)
        assert f.min_pass_rate is None
        assert f.max_pass_rate is None
