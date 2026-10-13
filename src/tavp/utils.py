"""Seeding, hashing and configuration helpers shared by all modules."""

from __future__ import annotations

import hashlib
import json
import zlib
from pathlib import Path
from typing import Any

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
TRADING_DAYS = 252


def stable_hash(obj: Any, length: int = 12) -> str:
    """Deterministic short hash of a JSON-serialisable object (keys sorted)."""
    payload = json.dumps(obj, sort_keys=True, default=_json_default).encode()
    return hashlib.sha256(payload).hexdigest()[:length]


def _json_default(x: Any) -> Any:
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    raise TypeError(f"not serialisable: {type(x)}")


def _key_to_int(key: Any) -> int:
    if isinstance(key, (int, np.integer)):
        return int(key)
    return zlib.crc32(str(key).encode())


def seed_sequence(master: int, *keys: Any) -> np.random.SeedSequence:
    """Independent, reproducible stream identified by a master seed and a tuple of keys."""
    return np.random.SeedSequence(master, spawn_key=tuple(_key_to_int(k) for k in keys))


def rng(master: int, *keys: Any) -> np.random.Generator:
    """NumPy generator for the stream (master, *keys)."""
    return np.random.default_rng(seed_sequence(master, *keys))


def int_seed(master: int, *keys: Any) -> int:
    """32-bit integer seed for libraries that need one (e.g. torch)."""
    return int(seed_sequence(master, *keys).generate_state(1)[0])


def load_yaml(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def deep_update(base: dict, override: dict) -> dict:
    """Recursively merge `override` into a copy of `base`."""
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | Path) -> dict:
    """Load an experiment configuration; a file may `extends:` another one."""
    path = Path(path)
    cfg = load_yaml(path)
    parent = cfg.pop("extends", None)
    if parent is not None:
        cfg = deep_update(load_config(path.parent / parent), cfg)
    return cfg
