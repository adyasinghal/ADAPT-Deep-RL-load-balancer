"""Action and observation layout shared by the env, the policies and the RL code.

Action indices are fixed:

    0      HOLD   keep buffering
    1      LOCAL  move the whole buffer into this server's own queue
    2..6   SEND_j send the whole buffer to neighbour slot j = 1..5

Inside the code neighbour slots are 0-based, so SEND to slot ``s`` is action ``2 + s``.
"""

from __future__ import annotations

HOLD = 0
LOCAL = 1
FIRST_SEND = 2
N_SLOTS = 5
N_ACTIONS = FIRST_SEND + N_SLOTS
N_FEATURES = 30

ACTION_NAMES = ["HOLD", "LOCAL"] + [f"SEND_{s + 1}" for s in range(N_SLOTS)]


def send_action(slot: int) -> int:
    """Action index that sends the batch to 0-based neighbour ``slot``."""
    return FIRST_SEND + slot


def action_slot(action: int) -> int:
    """0-based neighbour slot targeted by a SEND action."""
    if action < FIRST_SEND:
        raise ValueError(f"action {action} is not a SEND action")
    return action - FIRST_SEND


def is_dispatch(action: int) -> bool:
    """True for LOCAL and every SEND action."""
    return action != HOLD
