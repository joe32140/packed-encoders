"""Curated engine defaults. External engines can be passed directly to pack()."""

from packed_encoders.arch.base import Architecture, match, register, registered
from packed_encoders.arch.embedding_gemma2 import EmbeddingGemma2
from packed_encoders.arch.modernbert import ModernBert

from packed_encoders.arch.qwen3_5 import Qwen35Hybrid

register(ModernBert(), default=True)
register(Qwen35Hybrid(), default=True)
register(EmbeddingGemma2(), default=True)

__all__ = ["Architecture", "match", "register", "registered"]
