"""Server-side orchestration utilities for the Flower solution package."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import flwr as fl
from flwr.common import Parameters, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.strategy import FedAvg


@dataclass
class StrategyConfig:
    """Options that describe how :class:`SolutionStrategy` behaves."""

    fraction_fit: float = 0.6
    fraction_evaluate: float = 1.0
    min_fit_clients: int = 2
    min_evaluate_clients: int = 2
    min_available_clients: int = 2
    staleness_lambda: float = 0.5


class SolutionStrategy(FedAvg):
    """FedAvg with simple staleness-aware weighting.

    Flower already ships with a robust FedAvg implementation.  By subclassing it
    we can keep the parameter handling logic while customising the aggregation of
    client updates.  The strategy expects each client to report a ``staleness``
    metric inside :meth:`flwr.client.NumPyClient.fit`/``evaluate``.
    """

    def __init__(self, cfg: StrategyConfig, evaluate_fn: Optional[Callable] = None) -> None:
        super().__init__(
            fraction_fit=cfg.fraction_fit,
            fraction_evaluate=cfg.fraction_evaluate,
            min_fit_clients=cfg.min_fit_clients,
            min_evaluate_clients=cfg.min_evaluate_clients,
            min_available_clients=cfg.min_available_clients,
            evaluate_fn=evaluate_fn,
        )
        self.cfg = cfg

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[BaseException] | List[Tuple[fl.server.client_proxy.ClientProxy, BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, float]]:
        if not results:
            return None, {}

        weighted_updates: List[np.ndarray] | None = None
        total_weight = 0.0
        staleness_values: List[float] = []

        for _, fit_res in results:
            ndarrays = parameters_to_ndarrays(fit_res.parameters)
            staleness = float(fit_res.metrics.get("staleness", 0.0))
            weight = 1.0 / (1.0 + self.cfg.staleness_lambda * staleness)
            coeff = fit_res.num_examples * weight
            staleness_values.append(staleness)

            if weighted_updates is None:
                weighted_updates = [coeff * layer for layer in ndarrays]
            else:
                weighted_updates = [
                    current + coeff * layer for current, layer in zip(weighted_updates, ndarrays)
                ]
            total_weight += coeff

        if weighted_updates is None or total_weight == 0.0:
            return None, {"mean_staleness": float(sum(staleness_values) / max(len(staleness_values), 1))}

        averaged = [layer / total_weight for layer in weighted_updates]
        aggregated_parameters = ndarrays_to_parameters(averaged)
        metrics = {
            "mean_staleness": float(sum(staleness_values) / len(staleness_values)),
            "total_weight": float(total_weight),
        }
        return aggregated_parameters, metrics


def build_strategy(cfg: Dict, evaluate_fn: Optional[Callable] = None) -> SolutionStrategy:
    """Create :class:`SolutionStrategy` from the raw YAML dictionary."""

    total_clients = int(cfg["clients"]["total"])
    strat_cfg = StrategyConfig(
        fraction_fit=float(cfg["clients"]["fraction_fit"]),
        fraction_evaluate=float(cfg["server"].get("fraction_evaluate", 1.0)),
        min_fit_clients=int(cfg["server"].get("min_fit_clients", total_clients)),
        min_evaluate_clients=int(cfg["server"].get("min_evaluate_clients", total_clients)),
        min_available_clients=int(cfg["server"].get("min_available_clients", total_clients)),
        staleness_lambda=float(cfg["server"].get("staleness_lambda", 0.5)),
    )
    return SolutionStrategy(strat_cfg, evaluate_fn=evaluate_fn)


__all__ = ["SolutionStrategy", "StrategyConfig", "build_strategy"]
