"""Display names and a fixed order for the methods, shared by the tables and the plots."""

from __future__ import annotations

# label used in summary.csv -> name printed in tables and legends
NAMES = {
    "evblb_dt2": "EVBLB (B2)",
    "adapt": "ADAPT",
    "perfect": "Perfect info (B6)",
    "evblb_dt0.25": "EVBLB 0.25 s (B3)",
    "ml_evblb": "ML-EVBLB (B4)",
    "topsis": "TOPSIS (B7)",
    "nearest": "Nearest (B1)",
    "adapt_noWS": "ADAPT-noWS",
    "adapt_ep0": "ADAPT, episode 0",
    "adapt_fixedDT": "ADAPT-fixedDT",
    "adapt_poll": "ADAPT-poll",
    "evblb_dt0.5": "EVBLB 0.5 s",
    "evblb_dt1": "EVBLB 1 s",
    "evblb_dt4": "EVBLB 4 s",
}

ORDER = list(NAMES)


def name(label: str) -> str:
    """Printable name of a method label (the label itself if we don't know it)."""
    return NAMES.get(label, label)


def ordered(labels) -> list[str]:
    """Labels sorted into the fixed display order; unknown labels go last."""
    labels = list(dict.fromkeys(labels))
    return sorted(labels, key=lambda lab: ORDER.index(lab) if lab in ORDER else len(ORDER))


def scenario(workload: str, rho: float) -> str:
    """Short scenario name such as 'W1, rho 0.7'."""
    return f"{workload}, ρ = {rho:g}"
