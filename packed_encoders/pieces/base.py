"""Executable operation contracts, independent of any model family or schedule."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Generic, ParamSpec, TypeVar

import torch

from packed_encoders.errors import ValidationError

P = ParamSpec("P")
R = TypeVar("R")


@dataclass(frozen=True)
class Contract:
    operation: str
    layout: str
    semantics: str
    autograd: bool = True
    capture: bool = True


@dataclass(frozen=True)
class PieceValidation:
    name: str
    max_abs_error: float | None
    rtol: float
    atol: float
    skipped_reason: str | None = None


@dataclass(frozen=True)
class Piece(Generic[P, R]):
    """A callable operation plus its semantic contract and independent oracle.

    Engines bind `execute` once; there is no per-call registry or generic evaluator.
    Weights and metadata are explicit arguments, so a piece has no model ownership.
    `check` validates a supplied fixture outside the hot path. Numerical validation
    is a probe within the declared contract, never a way of inferring support.
    Inputs are not mutated and outputs own storage unless the contract says otherwise.
    """

    name: str
    contract: Contract
    execute: Callable[P, R]
    reference: Callable[P, R]
    check: Callable[P, None]
    rtol: float = 0.02
    atol: float = 0.02

    def validate(self, *args: P.args, **kwargs: P.kwargs) -> PieceValidation:
        self.check(*args, **kwargs)
        with torch.no_grad():
            expected = self.reference(*args, **kwargs)
            actual = self.execute(*args, **kwargs)
        expected = expected if isinstance(expected, tuple) else (expected,)
        actual = actual if isinstance(actual, tuple) else (actual,)
        if len(actual) != len(expected):
            raise ValidationError(f"piece {self.name}: output arity differs from reference")
        worst = 0.0
        for got, ref in zip(actual, expected):
            try:
                torch.testing.assert_close(got, ref, rtol=self.rtol, atol=self.atol)
            except AssertionError as exc:
                raise ValidationError(f"piece {self.name} failed reference validation: {exc}") from exc
            worst = max(worst, float((got.float() - ref.float()).abs().max()))
        return PieceValidation(self.name, worst, self.rtol, self.atol)
