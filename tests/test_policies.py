"""Tests for the baseline policies."""

import numpy as np

from policies.random_policy import RandomPolicy


def test_random_policy_respects_masks():
    rng = np.random.default_rng(0)
    masks = rng.random((200, 7)) < 0.3
    masks[:, 0] = True
    policy = RandomPolicy()
    policy.reset(None, np.random.default_rng(1))
    actions = policy.act(np.zeros((200, 30), np.float32), masks, [])
    assert masks[np.arange(200), actions].all()
