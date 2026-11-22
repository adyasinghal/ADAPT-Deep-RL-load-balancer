"""Policy registry.

``make_policy({"name": "evblb", "dt": 1.0})`` builds a policy from an experiment config entry;
``{"name": "adapt", "variant": "noWS"}`` loads the matching trained weights.
A new policy is added by giving it an entry in ``REGISTRY``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from policies.adapt import AdaptPolicy
from policies.base import BasePolicy
from policies.evblb import EVBLB, PerfectInfo
from policies.ml_evblb import MLEvblb
from policies.nearest import Nearest
from policies.random_policy import RandomPolicy
from policies.topsis import Topsis

REGISTRY: dict[str, Callable[..., BasePolicy]] = {
    "nearest": Nearest,
    "evblb": EVBLB,
    "perfect": PerfectInfo,
    "random": RandomPolicy,
    "adapt": AdaptPolicy,
    "ml_evblb": MLEvblb,
    "topsis": Topsis,
}


def make_policy(spec: dict[str, Any]) -> BasePolicy:
    """Build a policy from ``{"name": ..., **params}`` (``label`` is ignored here)."""
    params = {k: v for k, v in spec.items() if k not in ("name", "label")}
    name = spec["name"]
    if name not in REGISTRY:
        raise ValueError(f"unknown policy {name!r}; have {sorted(REGISTRY)}")
    return REGISTRY[name](**params)
