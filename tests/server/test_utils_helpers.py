"""Unit tests for ``utils.helpers.safe_count``.

Pure-logic (no DB): safe_count wraps ``memory.count()`` so list endpoints degrade
to ``None`` on transient store failures instead of 500ing, while re-raising
programming errors. Mirrors the advisory-count contract documented in the plan.
"""

import pytest

pytest.importorskip("utils.helpers", reason="server modules not installed")

from unittest.mock import MagicMock  # noqa: E402

from utils.helpers import safe_count  # noqa: E402


def _mem(count_return=None, count_exc=None):
    mem = MagicMock()
    if count_exc is not None:
        mem.count.side_effect = count_exc
    else:
        mem.count.return_value = count_return
    return mem


class TestSafeCount:
    def test_returns_int_from_memory_count(self):
        assert safe_count(_mem(42), filters={"user_id": "u1"}) == 42

    def test_passes_filters_through(self):
        mem = _mem(7)
        safe_count(mem, filters={"user_id": "u1"})
        mem.count.assert_called_once_with(filters={"user_id": "u1"})

    def test_default_filters_none(self):
        mem = _mem(3)
        safe_count(mem)
        mem.count.assert_called_once_with(filters=None)

    def test_returns_none_when_count_unsupported(self):
        # VectorStoreBase default returns None -> passed through, not coerced to 0.
        assert safe_count(_mem(None)) is None

    @pytest.mark.parametrize("value", ["not-an-int", 3.5, [1, 2]])
    def test_normalizes_non_int_to_none(self, value):
        # Defensive: a store that returns a non-int (e.g. a malformed override).
        assert safe_count(_mem(value)) is None

    @pytest.mark.parametrize("exc", [NameError, AttributeError, SyntaxError, TypeError])
    def test_programming_errors_reraise(self, exc):
        with pytest.raises(exc):
            safe_count(_mem(None, count_exc=exc("boom")))

    def test_transient_failure_returns_none(self):
        # A store RuntimeError (connection/timeout) is swallowed -> None.
        assert safe_count(_mem(None, count_exc=RuntimeError("store down"))) is None


# --------------------------------------------------------------------------- #
# extract_memory_id parity: server copy must match the SDK copy
# --------------------------------------------------------------------------- #

from mem0.memory.utils import extract_memory_id as sdk_extract_id  # noqa: E402
from utils.helpers import extract_memory_id  # noqa: E402


class _ObjRow:
    def __init__(self, mid):
        self.id = mid


class _ObjRowNoId:
    pass


class TestExtractMemoryIdParity:
    """Server and SDK copies of extract_memory_id must produce identical output."""

    @pytest.mark.parametrize("row,label", [
        (None, "none"),
        ({"id": "abc"}, "dict-id"),
        ({"_id": "xyz"}, "dict-underscore-id"),
        ({"id": "abc", "_id": "xyz"}, "dict-both-id-priority"),
        ({}, "dict-empty"),
        (_ObjRow("obj-id"), "obj-id"),
        (_ObjRowNoId(), "obj-no-id"),
    ])
    def test_server_and_sdk_produce_same_result(self, row, label):
        assert extract_memory_id(row) == sdk_extract_id(row), label
