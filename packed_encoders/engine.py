"""Preparation-time engine contracts. No registry dispatch occurs during execution.

These extension contracts are provisional. An engine's prepare() must undo its
own mutations if it raises before returning a packed engine. Once returned,
close(rollback=True) owns that rollback, including derived resources and storage
relationships. Successful close preserves weight updates made while installed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol

from torch import Tensor, nn

from packed_encoders.batch import PackedBatch
from packed_encoders.pieces.base import Piece


class _Omitted:
    def __repr__(self) -> str:
        return "OMITTED"


OMITTED = _Omitted()


@dataclass(frozen=True)
class Capabilities:
    inference: bool = True
    training: bool = False
    inference_capture: bool = False
    training_capture: bool = False
    original_forward_fallback: bool = False


@dataclass(frozen=True)
class ValidationResult:
    engine: str
    details: Any


@dataclass(frozen=True)
class ModelBinding:
    adapter: ModelAdapter
    patch_target: nn.Module
    weight_source: nn.Module


class ModelAdapter(Protocol):
    name: str

    def bind(self, module: object) -> ModelBinding | None: ...

    def install(self, binding: ModelBinding, packed: PackedEngine) -> None:
        """Install the prepared public forward; do not mutate weights here."""
        ...


class PackedEngine(Protocol):
    capabilities: Capabilities
    pieces: tuple[Piece, ...]
    state: Any

    @property
    def graph_enabled(self) -> bool: ...

    @graph_enabled.setter
    def graph_enabled(self, enabled: bool) -> None: ...

    def forward_packed(self, batch: PackedBatch) -> Tensor: ...

    def validate(self, **kwargs: Any) -> ValidationResult: ...

    def configure(self, options: Mapping[str, Any]) -> None:
        """Handle repeat pack options; reject incompatible changes explicitly."""
        ...

    def set_cuda_graph(self, enabled: bool, config: Any = None) -> None: ...

    def set_train_cuda_graph(self, enabled: bool, config: Any = None) -> None: ...

    def close(self, *, rollback: bool = False) -> None:
        """Release resources; idempotent, preserving updates on successful teardown."""
        ...


class Engine(Protocol):
    name: str
    adapters: tuple[ModelAdapter, ...]

    def prepare(self, binding: ModelBinding, options: Mapping[str, Any]) -> PackedEngine: ...

    def validate(self, binding: ModelBinding, **kwargs: Any) -> ValidationResult: ...


@dataclass
class Installation:
    engine: Engine
    binding: ModelBinding
    packed: PackedEngine
    original_forward: Any
    had_forward_attribute: bool

    def restore_forward(self) -> None:
        target = self.binding.patch_target
        if self.had_forward_attribute:
            target.forward = self.original_forward
        else:
            target.__dict__.pop("forward", None)
