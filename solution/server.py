"""Server-side orchestration utilities for the Flower solution package."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import math
import numpy as np
import flwr as fl
from flwr.common import Parameters, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

TensorList = List[np.ndarray]


def _copy_tensors(tensors: Sequence[np.ndarray]) -> TensorList:
    return [np.copy(layer) for layer in tensors]


def _zeros_like(tensors: Sequence[np.ndarray]) -> TensorList:
    return [np.zeros_like(layer) for layer in tensors]


def _dot(a: Sequence[np.ndarray], b: Sequence[np.ndarray]) -> float:
    return float(sum(np.dot(x.ravel(), y.ravel()) for x, y in zip(a, b)))


def _norm_sq(tensors: Sequence[np.ndarray]) -> float:
    return float(sum(np.dot(layer.ravel(), layer.ravel()) for layer in tensors))


def _add(a: Sequence[np.ndarray], b: Sequence[np.ndarray]) -> TensorList:
    return [x + y for x, y in zip(a, b)]


def _scale(tensors: Sequence[np.ndarray], factor: float) -> TensorList:
    return [factor * layer for layer in tensors]


def _sub(a: Sequence[np.ndarray], b: Sequence[np.ndarray]) -> TensorList:
    return [x - y for x, y in zip(a, b)]


@dataclass
class BufferedUpdate:
    """Represents one client update ``u_i`` from the PDF formulation."""

    delta: TensorList
    num_examples: int
    staleness: float
    train_loss: float
    val_loss: Optional[float]
    val_accuracy: Optional[float]

    def quality(self, beta: float, eps: float) -> float:
        """Return the ΔL_i-like alignment quality term."""

        inv_loss = 1.0 / (1.0 + max(self.train_loss, 0.0) + eps)
        acc_term = 0.0 if self.val_accuracy is None else float(self.val_accuracy)
        beta = min(max(beta, 0.0), 1.0)
        return (1.0 - beta) * inv_loss + beta * acc_term

    def stability(self, eps: float) -> float:
        """Return a guard helper (high loss → low stability)."""

        ref_loss = self.val_loss if self.val_loss is not None else self.train_loss
        return 1.0 / (1.0 + max(ref_loss, 0.0) + eps)


@dataclass
class StrategyConfig:
    """Options that describe how :class:`SolutionStrategy` behaves."""

    fraction_fit: float = 0.6
    fraction_evaluate: float = 1.0
    min_fit_clients: int = 2
    min_evaluate_clients: int = 2
    min_available_clients: int = 2
    staleness_lambda: float = 0.5
    buffer_size: int = 8
    selection_size: int = 4
    freshness_tau: float = 1.0
    align_guard: float = 1.0
    sideways_guard: float = 0.5
    beta: float = 0.3
    guard_epsilon: float = 1e-9


class SolutionStrategy(FedAvg):
    """FedAvg with the exact flow described in the attached PDF."""

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
        self.cfg.selection_size = max(1, min(cfg.selection_size, cfg.buffer_size))
        self.buffer: List[BufferedUpdate] = []
        self.current_global: Optional[TensorList] = None
        self.prev_direction: Optional[TensorList] = None

    # ------------------------------------------------------------------
    # Helper math terms from the document
    # ------------------------------------------------------------------
    def _freshness(self, staleness: float) -> float:
        scale = self.cfg.staleness_lambda / (1.0 + self.cfg.freshness_tau)
        return math.exp(-scale * max(staleness, 0.0))

    def _guard_aligned(self, update: BufferedUpdate) -> float:
        quality = update.quality(self.cfg.beta, self.cfg.guard_epsilon)
        return min(1.0, self.cfg.align_guard * quality)

    def _guard_sideways(self, update: BufferedUpdate, aligned_norm: float, sideways_norm: float) -> float:
        quality = update.quality(self.cfg.beta, self.cfg.guard_epsilon)
        stability = update.stability(self.cfg.guard_epsilon)
        curvature = sideways_norm / (aligned_norm + self.cfg.guard_epsilon)
        damping = 1.0 / (1.0 + curvature)
        return min(1.0, self.cfg.sideways_guard * quality * stability * damping)

    def _split_components(
        self,
        update: BufferedUpdate,
        direction: TensorList,
    ) -> Tuple[TensorList, TensorList, float, float]:
        direction_norm_sq = _norm_sq(direction)
        if direction_norm_sq <= self.cfg.guard_epsilon:
            aligned = _zeros_like(update.delta)
        else:
            coeff = _dot(update.delta, direction) / direction_norm_sq
            aligned = _scale(direction, coeff)
        sideways = _sub(update.delta, aligned)
        return aligned, sideways, math.sqrt(_norm_sq(aligned)), math.sqrt(_norm_sq(sideways))

    # ------------------------------------------------------------------
    # Flower Strategy overrides
    # ------------------------------------------------------------------
    def initialize_parameters(self, client_manager: fl.server.client_manager.ClientManager):
        params = super().initialize_parameters(client_manager)
        if params is not None:
            self.current_global = parameters_to_ndarrays(params)
            self.prev_direction = _zeros_like(self.current_global)
        return params

    def _collect_updates(
        self,
        results: List[Tuple[ClientProxy, fl.common.FitRes]],
    ) -> List[BufferedUpdate]:
        if not results:
            return []

        if self.current_global is None:
            reference = parameters_to_ndarrays(results[0][1].parameters)
            self.current_global = _copy_tensors(reference)
            self.prev_direction = _zeros_like(self.current_global)

        buffered: List[BufferedUpdate] = []
        for _, fit_res in results:
            ndarrays = parameters_to_ndarrays(fit_res.parameters)
            delta = _sub(ndarrays, self.current_global)
            metrics = fit_res.metrics or {}
            buffered.append(
                BufferedUpdate(
                    delta=delta,
                    num_examples=fit_res.num_examples,
                    staleness=float(metrics.get("staleness", 0.0)),
                    train_loss=float(metrics.get("train_loss", 0.0)),
                    val_loss=(
                        None
                        if metrics.get("val_loss") is None
                        else float(metrics["val_loss"])
                    ),
                    val_accuracy=(
                        None
                        if metrics.get("val_accuracy") is None
                        else float(metrics["val_accuracy"])
                    ),
                )
            )
        return buffered

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, fl.common.FitRes]],
        failures: List[BaseException] | List[Tuple[ClientProxy, BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, float]]:
        if not results:
            return None, {}

        buffered = self._collect_updates(results)
        self.buffer.extend(buffered)
        self.buffer.sort(key=lambda upd: upd.staleness)
        self.buffer = self.buffer[: self.cfg.buffer_size]

        selected = self.buffer[: self.cfg.selection_size]
        if not selected:
            return super().aggregate_fit(server_round, results, failures)

        direction = self.prev_direction or _zeros_like(selected[0].delta)
        weighted: Optional[TensorList] = None
        total_weight = 0.0
        staleness_vals: List[float] = []

        for update in selected:
            aligned, sideways, aligned_norm, sideways_norm = self._split_components(update, direction)
            guard_aligned = self._guard_aligned(update)
            guard_sideways = self._guard_sideways(update, aligned_norm, sideways_norm)
            merged = [
                guard_aligned * a_layer + guard_sideways * s_layer
                for a_layer, s_layer in zip(aligned, sideways)
            ]
            freshness = self._freshness(update.staleness)
            weight = update.num_examples * freshness

            if weighted is None:
                weighted = _scale(merged, weight)
            else:
                weighted = _add(weighted, _scale(merged, weight))

            total_weight += weight
            staleness_vals.append(update.staleness)

        # Remove the consumed updates (Guard_i(w_t, w_{t+1}) in the PDF).
        selected_ids = {id(update) for update in selected}
        self.buffer = [upd for upd in self.buffer if id(upd) not in selected_ids]

        if weighted is None or total_weight == 0.0:
            return super().aggregate_fit(server_round, results, failures)

        avg_delta = _scale(weighted, 1.0 / total_weight)
        if self.current_global is None:
            new_global = avg_delta
        else:
            new_global = _add(self.current_global, avg_delta)

        aggregated_parameters = ndarrays_to_parameters(new_global)
        if self.current_global is not None:
            self.prev_direction = _sub(new_global, self.current_global)
        else:
            self.prev_direction = _copy_tensors(new_global)
        self.current_global = _copy_tensors(new_global)

        metrics = {
            "mean_staleness": float(sum(staleness_vals) / len(staleness_vals)),
            "effective_updates": float(len(selected)),
            "buffer_retained": float(len(self.buffer)),
        }
        return aggregated_parameters, metrics


def build_strategy(cfg: Dict, evaluate_fn: Optional[Callable] = None) -> SolutionStrategy:
    """Create :class:`SolutionStrategy` from the raw YAML dictionary."""

    total_clients = int(cfg["clients"]["total"])
    server_cfg = cfg["server"]
    strat_cfg = StrategyConfig(
        fraction_fit=float(cfg["clients"]["fraction_fit"]),
        fraction_evaluate=float(server_cfg.get("fraction_evaluate", 1.0)),
        min_fit_clients=int(server_cfg.get("min_fit_clients", total_clients)),
        min_evaluate_clients=int(server_cfg.get("min_evaluate_clients", total_clients)),
        min_available_clients=int(server_cfg.get("min_available_clients", total_clients)),
        staleness_lambda=float(server_cfg.get("staleness_lambda", 0.5)),
        buffer_size=int(server_cfg.get("buffer_size", 8)),
        selection_size=int(server_cfg.get("selection_size", 4)),
        freshness_tau=float(server_cfg.get("freshness_tau", 1.0)),
        align_guard=float(server_cfg.get("align_guard", 1.0)),
        sideways_guard=float(server_cfg.get("sideways_guard", 0.5)),
        beta=float(server_cfg.get("beta", 0.3)),
        guard_epsilon=float(server_cfg.get("guard_epsilon", 1e-9)),
    )
    return SolutionStrategy(strat_cfg, evaluate_fn=evaluate_fn)


__all__ = ["SolutionStrategy", "StrategyConfig", "build_strategy"]
