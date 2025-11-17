"""Entry point that wires together the Flower clients and strategy."""
from __future__ import annotations

import os

import flwr as fl

from solution.pipeline import SolutionPipeline


CFG_PATH = os.environ.get(
    "FEDASYNC_SOLUTION_CONFIG",
    os.path.join(os.path.dirname(__file__), "config.yaml"),
)


def run_solution(cfg_path: str = CFG_PATH) -> fl.simulation.history.History:
    """High-level convenience wrapper around :class:`SolutionPipeline`."""

    cfg = SolutionPipeline.load_cfg(cfg_path)
    pipeline = SolutionPipeline(cfg)
    return pipeline.run()


def main() -> None:
    run_solution()


if __name__ == "__main__":
    main()
