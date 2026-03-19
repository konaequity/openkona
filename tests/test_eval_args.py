from __future__ import annotations

import argparse

from konash.eval.harness import add_common_args


def _parse(*argv: str):
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    return parser.parse_args(list(argv))


def test_eval_defaults_are_verbose_and_single_rollout():
    args = _parse()

    assert args.parallel == 0
    assert args.verbose is True


def test_eval_flags_can_enable_parallel_or_disable_verbose():
    args = _parse("--parallel", "5", "--no-verbose")

    assert args.parallel == 5
    assert args.verbose is False
