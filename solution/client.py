"""Client components for the Flower based solution package."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import flwr as fl


TensorList = List[np.ndarray]
ModelBuilder = Callable[[], nn.Module]


def _to_device(batch: Tuple[torch.Tensor, torch.Tensor], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Move a batch of tensors to ``device`` for training/evaluation."""

    x, y = batch
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)


@dataclass
class ClientConfig:
    """Lightweight container with the knobs each Flower client needs."""

    batch_size: int = 32
    local_epochs: int = 1
    lr: float = 0.01
    momentum: float = 0.9
    weight_decay: float = 5e-4


def parameters_to_ndarrays(model: nn.Module) -> TensorList:
    """Return the current model weights as a list of NumPy arrays."""

    return [tensor.detach().cpu().numpy() for tensor in model.state_dict().values()]


def ndarrays_to_parameters(model: nn.Module, parameters: Sequence[np.ndarray]) -> None:
    """Load ``parameters`` into ``model``.

    The helper performs the same mapping each time because ``state_dict`` keeps a
    deterministic ordering of its keys.
    """

    state_dict_keys = list(model.state_dict().keys())
    if len(state_dict_keys) != len(parameters):
        raise ValueError("Mismatch between model parameters and received weights")

    state_dict = {}
    for key, array in zip(state_dict_keys, parameters):
        template = model.state_dict()[key]
        tensor = torch.tensor(array, dtype=template.dtype, device=template.device)
        state_dict[key] = tensor
    model.load_state_dict(state_dict, strict=True)


def train_model(model: nn.Module, loader: DataLoader, device: torch.device, cfg: ClientConfig) -> float:
    """One full local training procedure returning the final loss."""

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=cfg.lr,
        momentum=cfg.momentum,
        weight_decay=cfg.weight_decay,
    )

    model.train()
    final_loss = 0.0
    for _ in range(cfg.local_epochs):
        for batch in loader:
            x, y = _to_device(batch, device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            final_loss = loss.item()
    return float(final_loss)


def evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    """Evaluate ``model`` and return ``(loss, accuracy)``."""

    criterion = nn.CrossEntropyLoss().to(device)
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    with torch.no_grad():
        for batch in loader:
            x, y = _to_device(batch, device)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item() * x.size(0)
            preds = logits.argmax(dim=1)
            total_correct += (preds == y).sum().item()
            total_examples += x.size(0)

    if total_examples == 0:
        return 0.0, 0.0
    avg_loss = total_loss / total_examples
    accuracy = total_correct / total_examples
    return float(avg_loss), float(accuracy)


class SolutionClient(fl.client.NumPyClient):
    """Flower ``NumPyClient`` wrapping the PyTorch Lightning training loop."""

    def __init__(
        self,
        cid: str,
        model_builder: ModelBuilder,
        trainset: Dataset,
        valset: Dataset,
        device: torch.device,
        cfg: ClientConfig,
        simulated_delay: float,
    ) -> None:
        super().__init__()
        self.cid = cid
        self.device = device
        self.model_builder = model_builder
        self.model = model_builder().to(device)
        self.cfg = cfg
        self.train_loader = DataLoader(
            trainset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=0,
        )
        self.val_loader = DataLoader(
            valset,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=0,
        )
        # ``simulated_delay`` is reported back to the server so the custom
        # strategy can down-weight stale updates.
        self.simulated_delay = simulated_delay

    def get_parameters(self, config: Dict[str, str] | None = None) -> TensorList:
        return parameters_to_ndarrays(self.model)

    def fit(self, parameters: TensorList, config: Dict[str, str] | None = None):
        ndarrays_to_parameters(self.model, parameters)
        loss = train_model(self.model, self.train_loader, self.device, self.cfg)
        metrics = {
            "train_loss": loss,
            "staleness": float(self.simulated_delay),
        }
        return parameters_to_ndarrays(self.model), len(self.train_loader.dataset), metrics

    def evaluate(self, parameters: TensorList, config: Dict[str, str] | None = None):
        ndarrays_to_parameters(self.model, parameters)
        loss, accuracy = evaluate_model(self.model, self.val_loader, self.device)
        metrics = {
            "accuracy": accuracy,
            "staleness": float(self.simulated_delay),
        }
        return loss, len(self.val_loader.dataset), metrics


__all__ = [
    "ClientConfig",
    "SolutionClient",
    "evaluate_model",
    "parameters_to_ndarrays",
    "ndarrays_to_parameters",
]
