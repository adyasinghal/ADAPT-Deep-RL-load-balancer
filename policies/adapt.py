"""ADAPT at inference time, and its ablation variants.

Every server runs the same frozen Q-network on its own 30 local features and takes the best
allowed action (greedy, no exploration). Inference is plain numpy, from the ``.npz`` weights.

Variants (same code, different flags and weights):

* ``full``    -- the method: learned when and where, piggybacked load reports, warm start.
* ``noWS``    -- trained without the EVBLB warm start; identical at inference.
* ``fixedDT`` -- dispatching is only allowed once 2 s have passed since the server's last
  dispatch, so the network only chooses *where*.
* ``poll``    -- neighbour features are fresh true values, but every dispatch pays a poll
  round, like EVBLB.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from core.actions import HOLD, LOCAL
from core.config import ROOT
from core.node import NodeView
from policies.base import BasePolicy
from rl.numpy_net import NumpyQNet, greedy

VARIANTS = ("full", "noWS", "fixedDT", "poll")
WEIGHTS_DIR = ROOT / "weights"


def weights_stem(variant: str) -> Path:
    """Default weight file stem of a variant: weights/adapt or weights/adapt_<variant>."""
    return WEIGHTS_DIR / ("adapt" if variant == "full" else f"adapt_{variant}")


class AdaptPolicy(BasePolicy):
    """Greedy shared Q-network on local observations."""

    max_hold = 2.0
    capacity_mask = True

    def __init__(self, variant: str = "full", weights: str | None = None, fixed_dt: float = 2.0):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"unknown ADAPT variant {variant!r}; have {VARIANTS}")
        self.variant = variant
        self.name = "adapt" if variant == "full" else f"adapt_{variant}"
        self.fixed_dt = fixed_dt if variant == "fixedDT" else None
        poll = variant == "poll"
        self.needs_poll = poll
        self.fresh_obs = poll
        self.uses_cache = not poll  # with polling, heartbeats would never be read
        path = Path(weights) if weights else weights_stem(variant).with_suffix(".npz")
        self.weights = path if path.is_absolute() else ROOT / path
        self.net: NumpyQNet | None = None

    def reset(self, topology, rng: np.random.Generator) -> None:
        """Load the weights (once) before the first episode."""
        super().reset(topology, rng)
        if self.net is None:
            if not self.weights.exists():
                raise FileNotFoundError(f"no ADAPT weights at {self.weights}; train first")
            self.net = NumpyQNet(self.weights)

    def effective_masks(self, masks: np.ndarray, ctx: list[NodeView]) -> np.ndarray:
        """The env's masks plus the fixed-interval rule of the ``fixedDT`` variant.

        Training applies the same function, so the agent learns under the masks it will face.
        Rule 5 still wins: if HOLD is already masked, the server may dispatch.
        """
        if self.fixed_dt is None or not ctx:
            return masks
        out = masks.copy()
        now = ctx[0].now
        for i, view in enumerate(ctx):
            if out[i, HOLD] and now - view.last_dispatch < self.fixed_dt - 1e-9:
                out[i, LOCAL:] = False  # only HOLD left
        return out

    def act(self, obs: np.ndarray, masks: np.ndarray, ctx: list[NodeView]) -> np.ndarray:
        """Best allowed action for every server in one batched forward pass."""
        if self.net is None:
            raise RuntimeError("call reset() before act()")
        return greedy(self.net(obs), self.effective_masks(masks, ctx))
