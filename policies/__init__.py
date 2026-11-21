"""Policy registry.

``make_policy({"name": "evblb", "dt": 1.0})`` builds a policy from an experiment config entry.
Other workstreams register their policies by adding an entry to ``REGISTRY``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from policies.base import BasePolicy
from policies.evblb import EVBLB, PerfectInfo
from policies.nearest import Nearest
from policies.random_policy import RandomPolicy

REGISTRY: dict[str, Callable[..., BasePolicy]] = {
    "nearest": Nearest,
    "evblb": EVBLB,
    "perfect": PerfectInfo,
    "random": RandomPolicy,
}


def make_policy(spec: dict[str, Any]) -> BasePolicy:
    """Build a policy from ``{"name": ..., **params}`` (``label`` is ignored here)."""
    params = {k: v for k, v in spec.items() if k not in ("name", "label")}
    name = spec["name"]
    if name not in REGISTRY:
        raise ValueError(f"unknown policy {name!r}; have {sorted(REGISTRY)}")
    return REGISTRY[name](**params)
