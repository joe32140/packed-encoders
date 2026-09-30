"""Registered architectures. Order matters only if two plugins could match one module;
each `match()` is written to be exclusive.

| name        | backbone                                  | entry points patched                |
|-------------|-------------------------------------------|-------------------------------------|
| modernbert  | ModernBERT / Ettin / mmBERT (+ finetunes) | `ModernBertModel.forward`           |

Adding one: implement the `Architecture` protocol in `arch/<name>/`, keep `match()` exact,
and register it below.
"""

from packed_encoders.arch.base import Architecture, match, register, registered
from packed_encoders.arch.modernbert import ModernBert

register(ModernBert())

__all__ = ["Architecture", "match", "register", "registered"]
