"""
Disposable fixture repository.

The soak needs *safe but meaningful* engineering work for MR1 to reason
about — a repo with enough realistic code and tests to support inspection,
workflow generation, agent specialization, and collaboration ("what looks
fragile", "compare these two modules", "own the testing side"). It must
never be the real MR1 repo, and permitted side effects are confined to it.

`build_fixture_repo(dest)` writes a small, deterministic Python project
whose modules deliberately contain a few *plausibly fragile* persistence
paths (a non-atomic write, an unbounded cache, a swallowed exception) so
that "tell me what seems fragile here" has real answers, plus a couple of
thin tests so "propose tests" and "own the testing side" have somewhere to
land.

Nothing here imports MR1. It is inert sample code.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

# --- sample source files -------------------------------------------------
#
# These are intentionally imperfect. The fragility is the point: it gives a
# reviewer agent something real to find. Keep them small and self-contained.

_STORE_PY = '''\
"""A tiny key/value store. Deliberately fragile in a few places."""

import json
import os


class KeyValueStore:
    def __init__(self, path):
        self.path = path
        self._cache = {}  # unbounded: grows without eviction (fragile)

    def load(self):
        try:
            with open(self.path) as fh:
                self._cache = json.load(fh)
        except Exception:
            # Swallows *everything* — a corrupt file reads as empty (fragile).
            self._cache = {}
        return self._cache

    def put(self, key, value):
        self._cache[key] = value
        # Non-atomic write: a crash mid-write truncates the store (fragile).
        with open(self.path, "w") as fh:
            json.dump(self._cache, fh)

    def get(self, key, default=None):
        return self._cache.get(key, default)
'''

_CACHE_PY = '''\
"""An in-memory cache with a bounded eviction policy (the careful module)."""

from collections import OrderedDict


class BoundedCache:
    def __init__(self, capacity=128):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self._items = OrderedDict()

    def put(self, key, value):
        if key in self._items:
            self._items.move_to_end(key)
        self._items[key] = value
        while len(self._items) > self.capacity:
            self._items.popitem(last=False)

    def get(self, key, default=None):
        if key not in self._items:
            return default
        self._items.move_to_end(key)
        return self._items[key]

    def __len__(self):
        return len(self._items)
'''

_PIPELINE_PY = '''\
"""A small data pipeline that ties the store and cache together."""

from .store import KeyValueStore
from .cache import BoundedCache


def run_pipeline(store_path, records):
    store = KeyValueStore(store_path)
    store.load()
    cache = BoundedCache(capacity=64)
    for key, value in records:
        store.put(key, value)
        cache.put(key, value)
    return {"stored": len(records), "cached": len(cache)}
'''

_TEST_CACHE_PY = '''\
from fragilekit.cache import BoundedCache


def test_bounded_cache_evicts_oldest():
    cache = BoundedCache(capacity=2)
    cache.put("a", 1)
    cache.put("b", 2)
    cache.put("c", 3)
    assert cache.get("a") is None
    assert cache.get("b") == 2
    assert cache.get("c") == 3


def test_bounded_cache_rejects_bad_capacity():
    import pytest

    with pytest.raises(ValueError):
        BoundedCache(capacity=0)
'''

_README = '''\
# fragilekit

A tiny disposable sample project used by the MR1 hierarchical soak.

Modules:
- `store.py`    — a key/value store with a few known-fragile persistence paths
- `cache.py`    — a bounded LRU cache (the careful counterpart)
- `pipeline.py` — glue that runs records through both

Tests live in `tests/`. The store module is intentionally under-tested.
'''

_PYPROJECT = '''\
[project]
name = "fragilekit"
version = "0.0.1"
description = "Disposable fixture for the MR1 hierarchical soak."
'''


_FILES: dict[str, str] = {
    "README.md": _README,
    "pyproject.toml": _PYPROJECT,
    "fragilekit/__init__.py": '"""fragilekit — disposable soak fixture."""\n',
    "fragilekit/store.py": _STORE_PY,
    "fragilekit/cache.py": _CACHE_PY,
    "fragilekit/pipeline.py": _PIPELINE_PY,
    "tests/__init__.py": "",
    "tests/test_cache.py": _TEST_CACHE_PY,
}

# A short, human-readable catalogue of the seeded fragilities, so the report
# can explain what MR1 *could* have found. Not fed to MR1 — it must discover
# these on its own.
KNOWN_FRAGILITIES: tuple[dict[str, str], ...] = (
    {
        "file": "fragilekit/store.py",
        "issue": "non-atomic write in KeyValueStore.put — a crash mid-write truncates the store",
    },
    {
        "file": "fragilekit/store.py",
        "issue": "bare except in KeyValueStore.load — a corrupt file silently reads as empty",
    },
    {
        "file": "fragilekit/store.py",
        "issue": "unbounded _cache dict — grows without eviction, unlike BoundedCache",
    },
    {
        "file": "tests/",
        "issue": "store.py has no tests; only cache.py is covered",
    },
)


def build_fixture_repo(dest: Path, *, git_init: bool = True) -> Path:
    """
    Materialize the disposable fixture repo under ``dest``.

    Idempotent: existing files are overwritten, so a resumed soak rebuilds a
    clean fixture. Returns ``dest``.
    """
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    for relative, content in _FILES.items():
        path = dest / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    if git_init and not (dest / ".git").exists():
        try:
            subprocess.run(
                ["git", "init", "-q"],
                cwd=str(dest),
                check=True,
                capture_output=True,
                timeout=30,
            )
            subprocess.run(
                ["git", "add", "-A"],
                cwd=str(dest),
                check=True,
                capture_output=True,
                timeout=30,
            )
            subprocess.run(
                ["git", "-c", "user.email=soak@example.com",
                 "-c", "user.name=soak", "commit", "-q", "-m", "seed fixture"],
                cwd=str(dest),
                check=True,
                capture_output=True,
                timeout=30,
            )
        except (subprocess.SubprocessError, OSError):
            # Git is a convenience for "run read-only status commands", not a
            # requirement. A fixture without it is still fully inspectable.
            pass
    return dest
