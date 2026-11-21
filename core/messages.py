"""Protocol messages and a per-type message counter."""

from __future__ import annotations

from enum import IntEnum

from core.model import Task


class MsgKind(IntEnum):
    """Message types of the dispatch protocol (brief section 5)."""

    POLL = 0
    POLL_REPLY = 1
    OFFER = 2
    ACK = 3
    REJECT = 4
    DONE = 5
    HEARTBEAT = 6


# replies that carry the sender's (P_rem, u) load report
LOAD_REPORTS = frozenset(
    {MsgKind.POLL_REPLY, MsgKind.ACK, MsgKind.REJECT, MsgKind.DONE, MsgKind.HEARTBEAT}
)


class Message:
    """One message between two edge servers.

    Only the fields a given kind needs are filled in: OFFER has ``tasks``; ACK, REJECT and DONE
    have ``ids``; every load report has ``p_rem`` and ``util``; ACK/REJECT also carry
    ``est_finish`` (seconds until the receiver's queue would drain), which B4 uses.
    """

    __slots__ = ("kind", "src", "dst", "offer_id", "tasks", "ids", "p_rem", "util", "est_finish")

    def __init__(
        self,
        kind: MsgKind,
        src: int,
        dst: int,
        *,
        offer_id: int = -1,
        tasks: list[Task] | None = None,
        ids: list[int] | None = None,
        p_rem: float = 0.0,
        util: float = 0.0,
        est_finish: float = 0.0,
    ) -> None:
        self.kind = kind
        self.src = src
        self.dst = dst
        self.offer_id = offer_id
        self.tasks = tasks
        self.ids = ids
        self.p_rem = p_rem
        self.util = util
        self.est_finish = est_finish

    def __repr__(self) -> str:
        return f"Message({self.kind.name}, {self.src}->{self.dst}, offer={self.offer_id})"


class MessageCounter:
    """Counts every message that is *sent*, lost or not, by type."""

    def __init__(self) -> None:
        self.counts = [0] * len(MsgKind)

    def add(self, kind: MsgKind) -> None:
        """Count one message of ``kind``."""
        self.counts[kind] += 1

    @property
    def total(self) -> int:
        """All messages sent so far."""
        return sum(self.counts)

    def as_dict(self) -> dict[str, int]:
        """Counts keyed by ``msg_<KIND>``."""
        return {f"msg_{k.name}": self.counts[k] for k in MsgKind}
