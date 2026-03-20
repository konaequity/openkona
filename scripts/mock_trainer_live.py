#!/usr/bin/env python3
"""Emit mock OAPL/value-model telemetry so /training/live can be tested."""

from __future__ import annotations

import argparse
import math
import time

from konash.training.logger import TrainingLogger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Emit mock live trainer telemetry")
    parser.add_argument("--project", default="trainer-live-demo", help="Project name to write under ~/.konash/projects/")
    parser.add_argument("--epochs", type=int, default=8, help="Number of mock OAPL epochs to emit")
    parser.add_argument("--delay", type=float, default=2.5, help="Seconds between epochs")
    parser.add_argument("--model", default="unsloth/GLM-4.5-Air", help="Model name to display")
    parser.add_argument("--corpus", default="financebench", help="Corpus name to display")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    log = TrainingLogger(args.project)

    log.start(iterations=1, corpus=args.corpus, model=args.model)
    print(f"Writing mock trainer telemetry to project: {args.project}")

    base_loss = 1.18
    base_entropy = 0.54

    for epoch in range(1, args.epochs + 1):
        loss = max(0.18, base_loss - 0.11 * epoch + math.sin(epoch * 0.7) * 0.025)
        entropy = max(0.08, base_entropy - 0.035 * epoch + math.cos(epoch * 0.45) * 0.018)
        log.oapl(
            iteration=1,
            epoch=epoch,
            loss=loss,
            entropy=entropy,
            kl=0.0018 * epoch,
            num_groups=96,
            num_rollouts=768,
            learning_rate=1e-6,
            duration_seconds=args.delay,
        )
        print(f"epoch {epoch}/{args.epochs}: loss={loss:.4f} entropy={entropy:.4f}")
        time.sleep(args.delay)

    log.value_model(loss=0.1432, epochs=3, duration_seconds=args.delay)
    print("value model complete")
    time.sleep(args.delay)

    log.complete(iterations=1, total_seconds=(args.epochs + 2) * args.delay)
    print("training complete")


if __name__ == "__main__":
    main()
