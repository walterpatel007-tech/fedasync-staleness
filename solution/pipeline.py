"""Object-oriented orchestration layer for the Flower reference solution."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

import random

import flwr as fl
import torch
from torch.utils.data import DataLoader
import yaml

from utils.helper import get_device, set_seed
from utils.model import build_squeezenet
from utils.partitioning import DataDistributor

from .client import (
    ClientConfig,
    SolutionClient,
    evaluate_model,
    ndarrays_to_parameters,
)
from .server import build_strategy


@dataclass
class PipelineArtifacts:
    """Container storing the runtime objects the pipeline wires up."""

    model_builder: Callable[[], torch.nn.Module]
    client_fn: Callable[[str], SolutionClient]
    evaluate_fn: Callable[[int, fl.common.Parameters, Dict[str, str]], tuple[float, Dict[str, float]]]
    strategy: fl.server.strategy.Strategy
    server_config: fl.server.ServerConfig


class SolutionPipeline:
    """High-level Flower experiment runner.

    The class keeps the repository's *solution* entry point object oriented by
    exposing methods that load configuration, prepare data/model builders, and
    finally launch Flower's simulator.  External scripts can instantiate the
    pipeline directly to integrate with notebooks or other automation.
    """

    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.cfg = cfg
        self.seed = int(cfg.get("seed", 42))
        self.device = get_device()
        self.dd: Optional[DataDistributor] = None
        self.artifacts: Optional[PipelineArtifacts] = None

    # ------------------------------------------------------------------
    # Configuration helpers
    # ------------------------------------------------------------------
    @staticmethod
    def load_cfg(path: str) -> Dict[str, Any]:
        """Read a YAML configuration file from ``path``."""

        with open(path, "r", encoding="utf-8") as handle:
            return yaml.safe_load(handle)

    # ------------------------------------------------------------------
    # Build individual components
    # ------------------------------------------------------------------
    def _build_model_fn(self, num_classes: int) -> Callable[[], torch.nn.Module]:
        """Create a factory returning a fresh model per client/server call."""

        def _fn() -> torch.nn.Module:
            return build_squeezenet(num_classes=num_classes, pretrained=False)

        return _fn

    def _build_server_eval_fn(
        self,
        model_builder: Callable[[], torch.nn.Module],
        testset,
        batch_size: int,
    ) -> Callable[[int, fl.common.Parameters, Dict[str, str]], tuple[float, Dict[str, float]]]:
        """Closure Flower invokes after each round to score the global model."""

        test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)

        def evaluate(server_round: int, parameters: fl.common.Parameters, config: Dict[str, str]):
            # Each evaluation runs on a *fresh* model to avoid state leaks.
            model = model_builder().to(self.device)
            ndarrays = fl.common.parameters_to_ndarrays(parameters)
            ndarrays_to_parameters(model, ndarrays)
            loss, accuracy = evaluate_model(model, test_loader, self.device)
            return float(loss), {"accuracy": float(accuracy)}

        return evaluate

    def _build_client_fn(
        self,
        dd: DataDistributor,
        model_builder: Callable[[], torch.nn.Module],
    ) -> Callable[[str], SolutionClient]:
        """Return the callable Flower uses to spin up client instances."""

        client_cfg = ClientConfig(
            batch_size=int(self.cfg["clients"]["batch_size"]),
            local_epochs=int(self.cfg["clients"]["local_epochs"]),
            lr=float(self.cfg["clients"]["lr"]),
            momentum=float(self.cfg["clients"]["momentum"]),
            weight_decay=float(self.cfg["clients"]["weight_decay"]),
        )

        total_clients = int(self.cfg["clients"]["total"])
        train_subsets = [dd.get_client_data(i) for i in range(total_clients)]
        valset = dd.test_dataset
        delay_low, delay_high = self.cfg["clients"].get("staleness_range", [0.0, 1.0])
        delay_low, delay_high = float(delay_low), float(delay_high)
        simulated_delays = [random.uniform(delay_low, delay_high) for _ in range(total_clients)]

        def client_fn(cid: str) -> SolutionClient:
            idx = int(cid)
            return SolutionClient(
                cid=cid,
                model_builder=model_builder,
                trainset=train_subsets[idx],
                valset=valset,
                device=self.device,
                cfg=client_cfg,
                simulated_delay=simulated_delays[idx],
            )

        return client_fn

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def prepare(self) -> PipelineArtifacts:
        """Create and cache all artefacts required by :meth:`run`."""

        set_seed(self.seed)
        random.seed(self.seed)

        self.dd = DataDistributor(
            dataset_name=self.cfg["data"]["dataset"],
            data_dir=self.cfg["data"]["data_dir"],
        )
        self.dd.distribute_data(
            num_clients=int(self.cfg["clients"]["total"]),
            alpha=float(self.cfg.get("partition_alpha", 0.5)),
            seed=self.seed,
        )

        model_builder = self._build_model_fn(int(self.cfg["data"]["num_classes"]))
        client_fn = self._build_client_fn(self.dd, model_builder)
        evaluate_fn = self._build_server_eval_fn(
            model_builder=model_builder,
            testset=self.dd.test_dataset,
            batch_size=int(self.cfg["clients"]["batch_size"]),
        )
        strategy = build_strategy(self.cfg, evaluate_fn=evaluate_fn)
        server_config = fl.server.ServerConfig(num_rounds=int(self.cfg["server"]["rounds"]))

        self.artifacts = PipelineArtifacts(
            model_builder=model_builder,
            client_fn=client_fn,
            evaluate_fn=evaluate_fn,
            strategy=strategy,
            server_config=server_config,
        )
        return self.artifacts

    def run(self) -> fl.simulation.history.History:
        """Launch the Flower simulator and return the resulting history."""

        artifacts = self.artifacts or self.prepare()

        simulation_kwargs: Dict[str, Any] = {
            "client_fn": artifacts.client_fn,
            "num_clients": int(self.cfg["clients"]["total"]),
            "config": artifacts.server_config,
            "strategy": artifacts.strategy,
        }

        client_resources = self.cfg.get("simulation", {}).get("client_resources")
        if client_resources:
            simulation_kwargs["client_resources"] = client_resources

        history = fl.simulation.start_simulation(**simulation_kwargs)

        if self.cfg.get("simulation", {}).get("log_metrics", True):
            print("[Solution] Final metrics:", history.metrics_centralized)

        return history


__all__ = ["PipelineArtifacts", "SolutionPipeline"]
