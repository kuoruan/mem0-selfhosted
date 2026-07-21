"""Unit tests for ``compat.scope`` filter-tree helpers.

Direct coverage of the NOT / OR-all-branches / nested-tree semantics of
``filter_tree_has_positive_key``, and the merge/append helpers, which are only
indirectly exercised via ``build_list_filters`` in ``test_server_compat.py``.

Pure logic — no DB. ``HTTPException`` is imported from fastapi for the 400 paths.
"""

import pytest

pytest.importorskip("fastapi", reason="fastapi not installed")

from compat.scope import (  # noqa: E402
    append_search_convenience_filters,
    build_categories_filter,
    filter_tree_has_positive_key,
    merge_extra_clauses_into_filters,
)
from fastapi import HTTPException  # noqa: E402


# --------------------------------------------------------------------------- #
# filter_tree_has_positive_key
# --------------------------------------------------------------------------- #
class TestFilterTreeHasPositiveKey:
    def test_flat_present(self):
        assert filter_tree_has_positive_key({"user_id": "u"}, "user_id") is True

    def test_flat_absent(self):
        assert filter_tree_has_positive_key({"user_id": "u"}, "agent_id") is False

    def test_not_clause_does_not_count_as_positive(self):
        assert filter_tree_has_positive_key({"NOT": {"user_id": "u"}}, "user_id") is False

    def test_and_any_branch_counts(self):
        tree = {"AND": [{"user_id": "u"}, {"created_at": {"gte": "2024"}}]}
        assert filter_tree_has_positive_key(tree, "user_id") is True
        assert filter_tree_has_positive_key(tree, "created_at") is True

    def test_or_all_branches_required(self):
        # both branches have user_id -> positive
        tree = {"OR": [{"user_id": "u"}, {"user_id": "u", "agent_id": "a"}]}
        assert filter_tree_has_positive_key(tree, "user_id") is True
        # not all branches have agent_id -> not positive
        assert filter_tree_has_positive_key(tree, "agent_id") is False

    def test_or_not_all_branches(self):
        tree = {"OR": [{"user_id": "u"}, {"agent_id": "a"}]}
        assert filter_tree_has_positive_key(tree, "user_id") is False
        assert filter_tree_has_positive_key(tree, "agent_id") is False

    def test_or_dict_branch(self):
        assert filter_tree_has_positive_key({"OR": {"user_id": "u"}}, "user_id") is True

    def test_nested_and_or(self):
        tree = {"AND": [{"OR": [{"user_id": "u"}, {"user_id": "u"}]}]}
        assert filter_tree_has_positive_key(tree, "user_id") is True

    def test_top_level_list_any(self):
        assert filter_tree_has_positive_key([{"user_id": "u"}, {"agent_id": "a"}], "user_id") is True
        assert filter_tree_has_positive_key([{"user_id": "u"}, {"agent_id": "a"}], "agent_id") is True
        assert filter_tree_has_positive_key([{"user_id": "u"}, {"agent_id": "a"}], "run_id") is False

    def test_non_dict_non_list(self):
        assert filter_tree_has_positive_key(None, "user_id") is False
        assert filter_tree_has_positive_key("string", "user_id") is False


# --------------------------------------------------------------------------- #
# merge_extra_clauses_into_filters
# --------------------------------------------------------------------------- #
class TestMergeExtraClauses:
    def test_no_extra_returns_unchanged(self):
        f = {"user_id": "u"}
        assert merge_extra_clauses_into_filters(f, []) == {"user_id": "u"}

    def test_flat_setdefault_adds_missing(self):
        out = merge_extra_clauses_into_filters({"user_id": "u"}, [{"agent_id": "a"}])
        assert out == {"user_id": "u", "agent_id": "a"}

    def test_flat_explicit_filter_wins_over_extra(self):
        # setdefault: an existing explicit key is NOT overridden by the extra clause.
        out = merge_extra_clauses_into_filters({"user_id": "explicit"}, [{"user_id": "extra"}])
        assert out == {"user_id": "explicit"}

    def test_logical_and_appends_extra_to_and_list(self):
        tree = {"AND": [{"user_id": "u"}], "user_id": "u"}
        out = merge_extra_clauses_into_filters(tree, [{"agent_id": "a"}])
        assert "agent_id" not in out  # appended into AND, not flat
        assert {"agent_id": "a"} in out["AND"]

    def test_logical_or_wraps_in_outer_and(self):
        tree = {"OR": [{"user_id": "u"}, {"agent_id": "a"}]}
        out = merge_extra_clauses_into_filters(tree, [{"app_id": "x"}])
        # No top-level AND -> wrapped in {"AND": [original, *extra]}
        assert list(out.keys()) == ["AND"]
        assert out["AND"][0] == tree
        assert {"app_id": "x"} in out["AND"]

    def test_logical_not_wraps_in_outer_and(self):
        tree = {"NOT": {"user_id": "u"}}
        out = merge_extra_clauses_into_filters(tree, [{"agent_id": "a"}])
        assert list(out.keys()) == ["AND"]
        assert out["AND"][0] == tree


# --------------------------------------------------------------------------- #
# append_search_convenience_filters
# --------------------------------------------------------------------------- #
class TestAppendSearchConvenience:
    def test_adds_categories_when_absent(self):
        out = append_search_convenience_filters({"user_id": "u"}, categories=["food"])
        assert out["categories"] == build_categories_filter(["food"])

    def test_does_not_override_positive_categories_flat(self):
        existing = {"categories": {"in": ["food"]}}
        out = append_search_convenience_filters(existing, categories=["x"])
        assert out["categories"] == {"in": ["food"]}

    def test_does_not_override_categories_inside_logical_tree(self):
        tree = {"AND": [{"categories": {"in": ["food"]}}]}
        out = append_search_convenience_filters(tree, categories=["x"])
        # categories already positively constrained -> no new clause appended
        assert not any({"categories": build_categories_filter(["x"])} == c for c in out.get("AND", []))

    def test_adds_metadata_keys_when_absent(self):
        out = append_search_convenience_filters({"user_id": "u"}, metadata={"topic": "ai"})
        assert out["topic"] == "ai"

    def test_does_not_override_existing_metadata_key(self):
        out = append_search_convenience_filters({"topic": "existing"}, metadata={"topic": "new"})
        assert out["topic"] == "existing"

    def test_categories_and_metadata_combined_into_logical_tree(self):
        tree = {"OR": [{"user_id": "u"}]}
        out = append_search_convenience_filters(tree, categories=["food"], metadata={"topic": "ai"})
        assert list(out.keys()) == ["AND"]
        assert out["AND"][0] == tree
        clauses = out["AND"][1:]
        assert {"categories": build_categories_filter(["food"])} in clauses
        assert {"topic": "ai"} in clauses


# --------------------------------------------------------------------------- #
# build_categories_filter
# --------------------------------------------------------------------------- #
def test_build_categories_filter_single_uses_contains():
    assert build_categories_filter(["food"]) == {"contains": "food"}


def test_build_categories_filter_multiple_uses_in():
    assert build_categories_filter(["food", "travel"]) == {"in": ["food", "travel"]}


def test_build_categories_filter_empty_list_falls_back_to_in():
    # len != 1 -> "in" path with empty list (edge case; caller normally guards)
    assert build_categories_filter([]) == {"in": []}


def test_require_entity_scope_rejects_non_string():
    # require_entity_scope calls _validate_entity_value internally,
    # which raises 400 on non-string entity ids — guard against silent type coercion.
    from compat.scope import require_entity_scope

    with pytest.raises(HTTPException) as exc:
        require_entity_scope(user_id=12345)  # type: ignore[arg-type]
    assert exc.value.status_code == 400
