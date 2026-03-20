"""Tests for config environment variable loading."""

from __future__ import annotations

import pytest
from ask_the_book.config import _require


def test_require_returns_value_when_env_var_is_set(monkeypatch) -> None:
    monkeypatch.setenv("SOME_KEY", "my-value")
    assert _require("SOME_KEY") == "my-value"


def test_require_raises_when_env_var_is_missing(monkeypatch) -> None:
    monkeypatch.delenv("SOME_KEY", raising=False)
    with pytest.raises(EnvironmentError, match="SOME_KEY"):
        _require("SOME_KEY")


def test_require_raises_when_env_var_is_empty_string(monkeypatch) -> None:
    monkeypatch.setenv("SOME_KEY", "")
    with pytest.raises(EnvironmentError, match="SOME_KEY"):
        _require("SOME_KEY")
