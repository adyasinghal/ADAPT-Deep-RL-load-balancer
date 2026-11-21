"""Loading scenario configs.

A scenario is `config/default.yaml` with some keys overridden. Overrides can come from a
nested dict (e.g. the `scenario:` block of an experiment file) or from dotted keys such as
``{"workload.rho": 0.9}``.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "config" / "default.yaml"


def read_yaml(path: str | Path) -> dict[str, Any]:
    """Read a YAML file into a dict (an empty file gives an empty dict)."""
    with open(path) as fh:
        return yaml.safe_load(fh) or {}


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of ``base`` with ``override`` merged in recursively."""
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def set_dotted(cfg: dict[str, Any], dotted_key: str, value: Any) -> None:
    """Set ``cfg["a"]["b"] = value`` for ``dotted_key == "a.b"``, in place.

    Unknown keys raise a KeyError so a typo in an experiment file can't be silently ignored.
    """
    *parents, leaf = dotted_key.split(".")
    node = cfg
    for part in parents:
        if part not in node:
            raise KeyError(f"unknown config section '{part}' in '{dotted_key}'")
        node = node[part]
    if leaf not in node:
        raise KeyError(f"unknown config key '{dotted_key}'")
    node[leaf] = value


def load_config(
    overrides: dict[str, Any] | None = None,
    dotted: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Load the default scenario and apply nested and then dotted overrides."""
    cfg = read_yaml(DEFAULT_CONFIG)
    if overrides:
        cfg = deep_merge(cfg, overrides)
    for key, value in (dotted or {}).items():
        set_dotted(cfg, key, value)
    return cfg
