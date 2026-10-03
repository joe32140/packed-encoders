"""Preparation-time engines with explicit curated defaults and ambiguity errors."""

from packed_encoders.engine import Engine
from packed_encoders.errors import PackedEncodersError

# Compatibility import; the old match/pack plugin contract is superseded.
Architecture = Engine
_REGISTRY: list[Engine] = []
_DEFAULTS: set[str] = set()


def register(engine: Engine, *, default: bool = False) -> Engine:
    if any(e.name == engine.name for e in _REGISTRY):
        raise ValueError(f"engine {engine.name!r} is already registered")
    _REGISTRY.append(engine)
    if default:
        _DEFAULTS.add(engine.name)
    return engine


def registered() -> tuple[Engine, ...]:
    return tuple(_REGISTRY)


def select(module: object, *, engine=None):
    candidates = []
    engines = (engine,) if engine is not None else registered()
    for candidate in engines:
        # Adapter precedence is authored within an engine (e.g. topk before HF).
        for adapter in candidate.adapters:
            binding = adapter.bind(module)
            if binding is not None:
                candidates.append((candidate, binding))
                break
    if engine is None:
        defaults = [c for c in candidates if c[0].name in _DEFAULTS]
        if defaults:
            candidates = defaults
    if len(candidates) > 1:
        names = ", ".join(p.name for p, _ in candidates)
        raise PackedEncodersError(
            f"ambiguous engines for {type(module).__name__}: {names}; "
            "pass engine=... to pack() explicitly"
        )
    return candidates[0] if candidates else None


def match(module: object):
    selected = select(module)
    return selected[0] if selected else None
