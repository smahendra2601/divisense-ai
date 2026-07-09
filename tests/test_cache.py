"""Tests for src/cache.py (Tier 1 — disk cache decorator).

Verifies the disk_cached() TTL decorator actually memoizes calls (same
args -> cached result, function body not re-executed), that different
args get independent cache slots, that entries expire after their TTL,
and that clear() wipes everything. Uses a temporary on-disk cache
(monkeypatched in place of the shared cache) so these tests never touch
the real .diskcache/ directory or interfere with each other.
"""

from __future__ import annotations

import inspect
import time

import pytest
from diskcache import Cache

from src import cache as cache_module
from src import config


@pytest.fixture
def temp_cache(tmp_path, monkeypatch):
    """Swap the module's shared cache for an isolated temp-dir instance."""
    isolated = Cache(str(tmp_path))
    monkeypatch.setattr(cache_module, "_cache", isolated)
    yield isolated
    isolated.clear()


def test_memoizes_repeated_calls(temp_cache):
    calls = []

    @cache_module.disk_cached()
    def compute(x):
        calls.append(x)
        return x * 2

    assert compute(3) == 6
    assert compute(3) == 6
    assert compute(3) == 6
    assert calls == [3]  # function body only executed once


def test_distinct_args_get_distinct_cache_slots(temp_cache):
    calls = []

    @cache_module.disk_cached()
    def compute(x):
        calls.append(x)
        return x * 2

    assert compute(1) == 2
    assert compute(2) == 4
    assert compute(1) == 2  # served from cache, not recomputed
    assert sorted(calls) == [1, 2]


def test_ttl_expiry(temp_cache):
    calls = []

    @cache_module.disk_cached(ttl=1)
    def compute(x):
        calls.append(x)
        return x

    compute("a")
    compute("a")
    assert calls == ["a"]

    time.sleep(1.2)
    compute("a")
    assert calls == ["a", "a"]  # recomputed after the 1s TTL elapsed


def test_default_ttl_matches_config():
    sig = inspect.signature(cache_module.disk_cached)
    assert sig.parameters["ttl"].default == config.CACHE_TTL_SECONDS


def test_clear_wipes_all_entries(temp_cache):
    @cache_module.disk_cached()
    def compute(x):
        return x

    compute("keep-me")
    assert len(temp_cache) > 0

    cache_module.clear()
    assert len(temp_cache) == 0
