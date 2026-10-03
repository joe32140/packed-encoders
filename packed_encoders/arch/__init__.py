"""Curated engine defaults. External engines can be passed directly to pack()."""

from packed_encoders.arch.base import Architecture, match, register, registered
from packed_encoders.arch.modernbert import ModernBert

register(ModernBert(), default=True)

__all__ = ["Architecture", "match", "register", "registered"]
