"""Flower-based reference solution for the FedAsync staleness project.

The :mod:`solution` namespace is intentionally split into four focused modules:

``client``
    Contains :class:`~solution.client.SolutionClient` plus training/evaluation
    helpers so every Flower participant behaves consistently.
``server``
    Hosts :class:`~solution.server.SolutionStrategy`, the FedAvg subclass that
    performs the staleness-aware aggregation described in the research brief.
``pipeline``
    Encapsulates end-to-end orchestration in the object-oriented
    :class:`~solution.pipeline.SolutionPipeline` class so experiments can be
    launched programmatically.
``run``
    Exposes a tiny entry point that instantiates the pipeline and is used by the
    ``python -m solution.run`` CLI.

Listing the modules here helps IDEs surface discoverable documentation while
keeping each module responsible for its own implementation details.
"""

__all__ = [
    "client",
    "server",
    "pipeline",
    "run",
]
