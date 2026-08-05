"""Unit tests for ``utils.pagination.paginate_response``.

Pure-logic envelope tests (no DB): next/previous link presence at page-size
boundaries, the ``total=None`` full-list-slicing path vs the ``total`` provided
pre-sliced path, and the empty-list / beyond-range cases. The URL string itself
is not asserted — a mock request stands in for ``Request`` so only the
``next``/``previous``/``results``/``count`` contract is exercised.
"""

import pytest

pytest.importorskip("utils.pagination", reason="server modules not installed")

from unittest.mock import MagicMock  # noqa: E402

from utils.pagination import paginate_response  # noqa: E402


def _request(page: int, page_size: int) -> MagicMock:
    """A mock Request whose ``url.include_query_params`` dynamically returns the requested page and page_size."""
    req = MagicMock()
    req.url.include_query_params.side_effect = lambda page, page_size: f"http://t/?page={page}&page_size={page_size}"
    return req


class TestPaginateResponseTotalFromList:
    """``total=None``: derive total from list length and slice here."""

    def test_first_page_has_next_when_more_pages(self):
        out = paginate_response(_request(1, 2), ["a", "b", "c", "d", "e"], 1, 2)
        assert out["count"] == 5
        assert out["results"] == ["a", "b"]
        assert out["next"] is not None
        assert out["previous"] is None

    def test_last_page_has_no_next(self):
        out = paginate_response(_request(3, 2), ["a", "b", "c", "d", "e"], 3, 2)
        assert out["results"] == ["e"]
        assert out["next"] is None
        assert out["previous"] is not None

    def test_exact_multiple_boundary_last_page_no_next(self):
        # total=4, page_size=2 -> page 2 is the last full page; next must be None
        # (off-by-one would wrongly emit a next link to an empty page 3).
        out = paginate_response(_request(2, 2), ["a", "b", "c", "d"], 2, 2)
        assert out["results"] == ["c", "d"]
        assert out["next"] is None
        assert out["previous"] is not None

    def test_exact_multiple_boundary_first_page_has_next(self):
        out = paginate_response(_request(1, 2), ["a", "b", "c", "d"], 1, 2)
        assert out["results"] == ["a", "b"]
        assert out["next"] is not None

    def test_beyond_range_returns_empty_with_previous(self):
        out = paginate_response(_request(3, 2), ["a", "b", "c", "d"], 3, 2)
        assert out["results"] == []
        assert out["count"] == 4
        assert out["next"] is None
        assert out["previous"] is not None

    def test_single_page_no_links(self):
        out = paginate_response(_request(1, 10), ["a", "b"], 1, 10)
        assert out["results"] == ["a", "b"]
        assert out["next"] is None
        assert out["previous"] is None

    def test_empty_list(self):
        out = paginate_response(_request(1, 10), [], 1, 10)
        assert out == {"count": 0, "next": None, "previous": None, "results": []}


class TestPaginateResponseTotalProvided:
    """``total`` provided: items are an already-paginated slice — not re-sliced."""

    def test_uses_provided_total_and_does_not_slice(self):
        # Caller passes the page slice [20:30] of a 100-item namespace.
        page = list(range(20, 30))
        out = paginate_response(_request(3, 10), page, 3, 10, total=100)
        assert out["count"] == 100
        assert out["results"] == page  # not re-sliced
        assert out["next"] is not None  # 20+10=30 < 100
        assert out["previous"] is not None

    def test_provided_total_last_page_no_next(self):
        page = list(range(90, 100))
        out = paginate_response(_request(10, 10), page, 10, 10, total=100)
        assert out["results"] == page
        assert out["next"] is None  # 90+10=100, not < 100
        assert out["previous"] is not None

    def test_provided_total_exact_multiple_last_page(self):
        # total=100, page_size=10 -> page 10 is last; boundary must not emit next.
        page = list(range(90, 100))
        out = paginate_response(_request(10, 10), page, 10, 10, total=100)
        assert out["next"] is None

    def test_provided_total_zero(self):
        out = paginate_response(_request(1, 10), [], 1, 10, total=0)
        assert out["count"] == 0
        assert out["next"] is None
        assert out["previous"] is None
        assert out["results"] == []


def test_previous_none_on_first_page_even_with_total():
    out = paginate_response(_request(1, 10), ["a"], 1, 10, total=1)
    assert out["previous"] is None
    assert out["next"] is None


class TestPaginateResponseHasMore:
    """``has_more`` overrides the total-based ``next`` derivation.

    Decouples pagination correctness from *total* (which may be advisory or
    unknown for stores whose ``count()`` ignores filters or counts expired rows).
    """

    def test_has_more_true_forces_next_even_when_total_says_no(self):
        # total=5, page 1/size 10: total formula says no next (10 >= 5),
        # but has_more=True forces a next link.
        out = paginate_response(_request(1, 10), ["a", "b"], 1, 10, total=5, has_more=True)
        assert out["next"] is not None
        assert out["count"] == 5

    def test_has_more_false_suppresses_next_even_when_total_says_yes(self):
        # total=100, page 1/size 10: total formula says next, but has_more=False.
        out = paginate_response(_request(1, 10), list(range(10)), 1, 10, total=100, has_more=False)
        assert out["next"] is None
        assert out["count"] == 100

    def test_has_more_none_falls_back_to_total_formula(self):
        out = paginate_response(_request(1, 10), list(range(10)), 1, 10, total=100, has_more=None)
        assert out["next"] is not None  # 0 + 10 < 100
