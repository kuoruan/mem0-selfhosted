"""Unit tests for the server compat utilities.

Covers:
  - compat.scope: collect_direct_entity_params, require_entity_scope,
                    build_search_filters, get_entity_field
  - compat.utils: drop_none
  - compat.helpers: normalize_results, normalize_results_dict
  - compat.decorators: upstream_guard exception mapping
  - routers.compat helpers: build_list_filters, paginate_response,
                            apply_fields, build_search_kwargs,
                            resolve_existing
"""

import logging
from datetime import date, datetime, timezone
from typing import Any
from unittest.mock import MagicMock

import pytest

pytest.importorskip("fastapi", reason="fastapi not installed")

from fastapi import BackgroundTasks, HTTPException
from pydantic import ValidationError
from starlette.datastructures import URL
from mem0.exceptions import ValidationError as Mem0ValidationError
import server.compat.tasks as compat_tasks
from server.compat.events import (
    CompatEvent,
    event_cache_all,
    event_cache_clear,
    event_cache_get,
    event_cache_put,
    event_cache_update,
    resolve_event_owner_id,
)
import server.compat.entities as compat_entities
from server.compat.entities import CompatEntity, list_entities_payload
from server.compat.requests import RequestMeta
from server.compat.decorators import upstream_guard
from server.compat.helpers import (
    normalize_results,
    normalize_results_dict,
)
from server.compat.metadata import (
    build_extraction_prompt,
    merge_v1_add_metadata,
    merge_v3_add_metadata,
)
from server.compat.utils import drop_none, normalize_timestamp, parse_iso_timestamp
from server.compat.responses import (
    resolve_optional_pagination,
    warn_ignored_compat_params,
)
from server.errors import UpstreamError
from server.compat.scope import (
    build_categories_filter,
    build_search_filters,
    collect_direct_entity_params,
    get_entity_field,
    require_entity_scope,
)
from server.routers.compat import (
    MemoryBatchDeleteInput,
    MemoryBatchDeleteLegacyInput,
    MemoryBatchUpdateInput,
    MemoryBatchUpdateItem,
    MemoryAddInputV3,
    MemoryGetInputV2,
    MemoryGetInputV3,
    MemorySearchInput,
    MemorySearchInputV2,
    MemorySearchInputV3,
    MemoryUpdateInput,
    build_list_filters,
    build_search_kwargs,
    paginate_response,
    resolve_existing,
    apply_fields,
    v1_batch_delete,
    v1_batch_update,
    v1_get_entity_memories,
    v1_get_event,
    v1_list_entities,
    v1_list_events,
    v1_list_memories,
    v1_search_memories,
    v1_update_memory,
    v2_list_memories,
    v2_search_memories,
    v3_add_memory,
    v3_get_all_memories,
    v3_search_memories,
)


import uuid


# Throwaway Request stand-in for direct handler calls. The autouse
# ``_direct_call_default_operator`` fixture stubs resolve_operator (which only
# inspects ``auth``), so a shared MagicMock is fine — no test asserts on it.
_REQ = MagicMock()


@pytest.fixture(autouse=True)
def _direct_call_default_operator(monkeypatch):
    """Many tests below invoke router handlers *directly* (not via TestClient),
    so FastAPI never resolves verify_auth/get_db. With ``auth=None`` the real
    resolve_operator raises 401 because a bare MagicMock request has no
    ``state.auth_type``. Short-circuit the None path to a default admin operator;
    calls passing a real ``auth=<User>`` still resolve normally.

    NOTE: this fixture also stubs the db-backed permission/scope guards (see
    below) to no-ops for the whole module. Tests in this file exercise handler
    logic and param-forwarding ONLY — they do NOT cover authorization, which
    lives in test_entity_permissions*. Any new guard added to a compat handler
    must be explicitly tested there, not silently stubbed here.

    Stubbed guards: resolve_operator, check_query_permission,
    check_memory_scope_permission, resolve_memory_entities, authorize_write,
    get_visible_entities.
    """
    default = MagicMock(id=uuid.UUID(int=1), role="admin")
    import server.entity_permissions as _ep
    import server.routers.compat as _rc

    _real = _ep.resolve_operator

    def _stub(request, auth, db):
        if auth is None:
            return default, True
        return _real(request, auth, db)

    monkeypatch.setattr(_ep, "resolve_operator", _stub)
    monkeypatch.setattr(_rc, "resolve_operator", _stub)

    # Handlers also call db-backed permission guards (check_query_permission /
    # check_memory_scope_permission) which receive the unresolved Depends object
    # as ``db`` under direct invocation. These tests exercise handler logic /
    # param-forwarding, not authorization — stub them to no-ops. Anything that
    # needs real authorization lives in test_entity_permissions*. Tests can still
    # monkeypatch these to assert specific behaviour.
    monkeypatch.setattr(_rc, "check_query_permission", lambda filters, *a, **k: filters)
    monkeypatch.setattr(_rc, "check_memory_scope_permission", lambda *a, **k: None)
    # resolve_memory_entities reads memory.get(...) to derive an entity scope; the
    # Mem0 runtime is uninitialized under direct handler invocation. Stub it to an
    # empty scope (admin-only — and check_memory_scope_permission is already a
    # no-op). Tests asserting a 404 monkeypatch this themselves.
    monkeypatch.setattr(_rc, "resolve_memory_entities", lambda memory_id: {})
    # authorize_write (used by v3_add_memory) queries entities via db; stub it.
    monkeypatch.setattr(_rc, "authorize_write", lambda *a, **k: None)
    # get_visible_entities (used by v1_list_entities) queries db; default to empty.
    # Tests needing specific entities monkeypatch this themselves.
    monkeypatch.setattr(_rc, "get_visible_entities", lambda *a, **k: [])


# ---------------------------------------------------------------------------
# compat.entities.CompatEntity
# ---------------------------------------------------------------------------


class TestCompatEntity:
    def test_from_bucket_serializes_timestamps(self):
        created = datetime(2026, 1, 1, tzinfo=timezone.utc)
        updated = datetime(2026, 1, 2, tzinfo=timezone.utc)
        entity = CompatEntity.from_bucket(
            "user",
            "alice",
            created_at=created,
            updated_at=updated,
        )
        assert entity.type == "user"
        assert entity.name == "alice"
        assert entity.created_at == created.isoformat()
        assert entity.owner == "self-hosted"

    def test_list_entities_payload_aggregates_by_user(self, monkeypatch):
        row = MagicMock(
            payload={
                "user_id": "alice",
                "created_at": "2026-01-01T00:00:00+00:00",
                "updated_at": "2026-01-02T00:00:00+00:00",
            }
        )
        mem = MagicMock()
        mem.vector_store.list.return_value = [row]

        def _get_mem():
            return mem

        monkeypatch.setattr(compat_entities, "get_memory_instance", _get_mem)

        entities = list_entities_payload()
        assert len(entities) == 1
        assert entities[0].id == "alice"

    def test_aggregate_entity_buckets_handles_mixed_timezone_formats(self):
        payloads = [
            {"user_id": "alice", "created_at": "2026-01-02T00:00:00+00:00", "updated_at": "2026-01-03T00:00:00+00:00"},
            {"user_id": "alice", "created_at": "2026-01-01T00:00:00", "updated_at": "2026-01-04T00:00:00"},
        ]

        buckets = compat_entities.aggregate_entity_buckets(payloads, {"user": "user_id"})
        bucket = buckets[("user", "alice")]
        assert bucket["created_at"] is not None
        assert bucket["updated_at"] is not None
        assert bucket["created_at"].tzinfo is not None
        assert bucket["updated_at"].tzinfo is not None

    def test_iter_payloads_scans_single_batch_when_partial(self, monkeypatch):
        """iter_payloads returns rows from a single partial batch."""
        row = MagicMock(payload={"user_id": "alice"})
        mem = MagicMock()
        mem.vector_store.list.return_value = [row]  # flat list, 1 row < batch_size

        monkeypatch.setattr(compat_entities, "get_memory_instance", lambda: mem)

        payloads = compat_entities.iter_payloads(limit=10_000)
        assert payloads == [{"user_id": "alice"}]
        mem.vector_store.list.assert_called_once_with(filters=None, top_k=10_000, skip=0)

    def test_iter_payloads_paginates_across_full_batches(self, monkeypatch):
        """iter_payloads scans multiple full batches until a partial batch ends the scan."""
        rows_a = [MagicMock(payload={"user_id": f"u{i}"}) for i in range(5)]
        rows_b = [MagicMock(payload={"user_id": f"u{i}"}) for i in range(5, 10)]
        rows_c = [MagicMock(payload={"user_id": "last"})]  # partial
        mem = MagicMock()
        mem.vector_store.list.side_effect = [rows_a, rows_b, rows_c]

        monkeypatch.setattr(compat_entities, "get_memory_instance", lambda: mem)

        payloads = compat_entities.iter_payloads(limit=5)
        assert len(payloads) == 11  # 5 + 5 + 1
        assert [c.kwargs["skip"] for c in mem.vector_store.list.call_args_list] == [0, 5, 10]

    def test_iter_payloads_handles_qdrant_cursor_none(self, monkeypatch):
        """A qdrant (items, None) tuple signals end-of-results after one batch."""
        row = MagicMock(payload={"user_id": "alice"})
        mem = MagicMock()
        mem.vector_store.list.return_value = ([row], None)  # cursor=None → stop

        monkeypatch.setattr(compat_entities, "get_memory_instance", lambda: mem)

        payloads = compat_entities.iter_payloads(limit=5)
        assert payloads == [{"user_id": "alice"}]
        assert mem.vector_store.list.call_count == 1

    def test_iter_payloads_detects_skip_ignoring_store(self, monkeypatch):
        """A store that ignores skip (same first id, full batch) stops after one batch."""
        rows = [MagicMock(id=f"m{i}", payload={"user_id": f"u{i}"}) for i in range(5)]
        mem = MagicMock()
        # Always returns the same full batch regardless of skip.
        mem.vector_store.list.return_value = rows

        monkeypatch.setattr(compat_entities, "get_memory_instance", lambda: mem)

        payloads = compat_entities.iter_payloads(limit=5)
        assert len(payloads) == 5  # only the first batch
        assert mem.vector_store.list.call_count == 2  # 2nd call detects dup id

    def test_v1_list_entities_returns_paginated_entities_with_mixed_timestamps(self, monkeypatch):
        from server.models import Entity

        # get_visible_entities is db-backed; stub it. v1_list_entities serializes
        # the returned Entity rows into the compat envelope and paginates.
        entities = [Entity(type="user", id="alice")]
        monkeypatch.setattr("server.routers.compat.get_visible_entities", lambda *a, **k: entities)

        req = MagicMock()
        req.url = URL("http://test/v1/entities?page=1&page_size=10")

        result = v1_list_entities(request=req, page=1, page_size=10, auth=None)

        assert result["count"] == 1
        assert len(result["results"]) == 1
        assert result["results"][0].id == "alice"

    def test_v1_list_entities_respects_pagination(self, monkeypatch):
        from server.models import Entity

        entities = [Entity(type="user", id="alice"), Entity(type="agent", id="agent-1")]
        monkeypatch.setattr("server.routers.compat.get_visible_entities", lambda *a, **k: entities)

        req = MagicMock()
        req.url = URL("http://test/v1/entities?page=1&page_size=1")

        page = v1_list_entities(request=req, page=1, page_size=1, auth=None)

        assert page["count"] == 2
        assert len(page["results"]) == 1
        assert page["next"] is not None


# ---------------------------------------------------------------------------
# compat.scope
# ---------------------------------------------------------------------------


class TestGetEntityField:
    def test_user(self):
        assert get_entity_field("user") == "user_id"

    def test_agent(self):
        assert get_entity_field("agent") == "agent_id"

    def test_run(self):
        assert get_entity_field("run") == "run_id"

    def test_app(self):
        assert get_entity_field("app") == "app_id"

    def test_unknown_raises_400(self):
        with pytest.raises(HTTPException) as exc:
            get_entity_field("robot")
        assert exc.value.status_code == 400


class TestCollectDirectEntityParams:
    def test_explicit_user_id(self):
        assert collect_direct_entity_params(user_id="u1") == {"user_id": "u1"}

    def test_multiple_kwargs(self):
        result = collect_direct_entity_params(user_id="u1", agent_id="a1")
        assert result == {"user_id": "u1", "agent_id": "a1"}

    def test_app_id_kwarg(self):
        assert collect_direct_entity_params(app_id="app1") == {"app_id": "app1"}

    def test_flat_filters(self):
        result = collect_direct_entity_params(filters={"user_id": "u1", "agent_id": "a1"})
        assert result == {"user_id": "u1", "agent_id": "a1"}

    def test_non_entity_keys_in_filters_ignored(self):
        result = collect_direct_entity_params(filters={"user_id": "u1", "created_at": {"gte": "2024"}})
        assert result == {"user_id": "u1"}

    def test_kwargs_matching_filters(self):
        result = collect_direct_entity_params(user_id="u1", filters={"user_id": "u1"})
        assert result == {"user_id": "u1"}

    def test_kwargs_conflicting_with_filters_raises(self):
        with pytest.raises(HTTPException) as exc:
            collect_direct_entity_params(
                user_id="explicit",
                filters={"user_id": "from_filter"},
            )
        assert exc.value.status_code == 400

    def test_entity_filter_value_must_be_string(self):
        with pytest.raises(HTTPException) as exc:
            collect_direct_entity_params(filters={"user_id": {"in": ["u1"]}})
        assert exc.value.status_code == 400
        assert "user_id" in exc.value.detail

    def test_and_nested(self):
        result = collect_direct_entity_params(filters={"AND": [{"user_id": "u1"}, {"created_at": {"gte": "2024"}}]})
        assert result == {}

    def test_or_nested(self):
        result = collect_direct_entity_params(filters={"OR": [{"user_id": "u1"}, {"user_id": "u1", "agent_id": "a1"}]})
        assert result == {}

    def test_app_id_nested_and(self):
        result = collect_direct_entity_params(filters={"AND": [{"app_id": "app1"}, {"user_id": "u1"}]})
        assert result == {}

    def test_app_id_nested_or(self):
        result = collect_direct_entity_params(
            filters={"OR": [{"app_id": "app1"}, {"app_id": "app1", "agent_id": "a1"}]}
        )
        assert result == {}

    def test_or_without_shared_entity_scope_is_ignored(self):
        result = collect_direct_entity_params(filters={"OR": [{"user_id": "u1"}, {"agent_id": "a1"}]})
        assert result == {}

    def test_not_scope_is_ignored(self):
        result = collect_direct_entity_params(filters={"NOT": {"user_id": "u1"}})
        assert result == {}

    def test_none_values_skipped(self):
        assert collect_direct_entity_params(user_id=None, agent_id=None) == {}

    def test_empty_returns_empty(self):
        assert collect_direct_entity_params() == {}


class TestRequireEntityScope:
    def test_raises_when_empty(self):
        with pytest.raises(HTTPException) as exc:
            require_entity_scope()
        assert exc.value.status_code == 400
        assert "app_id" in exc.value.detail

    def test_custom_detail(self):
        with pytest.raises(HTTPException) as exc:
            require_entity_scope(detail="need id")
        assert exc.value.detail == "need id"

    def test_returns_scope(self):
        result = require_entity_scope(user_id="u1")
        assert result == {"user_id": "u1"}

    def test_fallback_user_id_when_empty(self):
        result = require_entity_scope(fallback_user_id="fallback")
        assert result == {"user_id": "fallback"}

    def test_explicit_takes_priority_over_fallback(self):
        result = require_entity_scope(user_id="explicit", fallback_user_id="fallback")
        assert result == {"user_id": "explicit"}

    def test_scope_from_filters(self):
        result = require_entity_scope(filters={"user_id": "u1"})
        assert result == {"user_id": "u1"}

    def test_not_scope_does_not_satisfy_requirement(self):
        with pytest.raises(HTTPException) as exc:
            require_entity_scope(filters={"NOT": {"user_id": "u1"}})
        assert exc.value.status_code == 400

    def test_scope_from_and_filter_tree(self):
        result = require_entity_scope(filters={"AND": [{"user_id": "u1"}, {"agent_id": "a1"}]})
        assert result == {"user_id": "u1", "agent_id": "a1"}

    def test_nested_entity_filter_value_must_be_string(self):
        with pytest.raises(HTTPException) as exc:
            require_entity_scope(filters={"AND": [{"user_id": {"in": ["u1"]}}]})
        assert exc.value.status_code == 400
        assert "user_id" in exc.value.detail

    def test_scope_from_shared_or_filter_tree(self):
        result = require_entity_scope(filters={"OR": [{"user_id": "u1"}, {"user_id": "u1", "agent_id": "a1"}]})
        assert result == {"user_id": "u1"}

    def test_conflicting_explicit_and_filter_tree_scope_raises(self):
        with pytest.raises(HTTPException) as exc:
            require_entity_scope(user_id="explicit", filters={"AND": [{"user_id": "from_filter"}]})
        assert exc.value.status_code == 400


class TestBuildSearchFilters:
    def test_no_filters_entity_kwarg(self):
        result = build_search_filters(user_id="u1")
        assert result == {"user_id": "u1"}

    def test_flat_filters_merged(self):
        result = build_search_filters(
            user_id="u1",
            filters={"created_at": {"gte": "2024-01-01"}},
        )
        assert result == {"user_id": "u1", "created_at": {"gte": "2024-01-01"}}

    def test_flat_filters_matching_entity_kwarg(self):
        result = build_search_filters(
            user_id="u1",
            filters={"user_id": "u1", "created_at": {"gte": "2024"}},
        )
        assert result["user_id"] == "u1"
        assert "created_at" in result

    def test_flat_filters_conflicting_entity_kwarg_raises(self):
        with pytest.raises(HTTPException) as exc:
            build_search_filters(
                user_id="explicit",
                filters={"user_id": "from_filter", "created_at": {"gte": "2024"}},
            )
        assert exc.value.status_code == 400

    def test_and_entity_in_conditions_is_copied_to_top_level_for_sdk_validation(self):
        filters = {"AND": [{"user_id": "u1"}, {"created_at": {"gte": "2024"}}]}
        result = build_search_filters(filters=filters)
        assert result == {**filters, "user_id": "u1"}

    def test_and_extra_entity_kwarg_copied_to_top_level(self):
        filters = {"AND": [{"created_at": {"gte": "2024"}}]}
        result = build_search_filters(user_id="u1", filters=filters)
        assert result == {**filters, "user_id": "u1"}

    def test_and_does_not_mutate_input(self):
        original = {"AND": [{"created_at": {"gte": "2024"}}]}
        build_search_filters(user_id="u1", filters=original)
        assert original == {"AND": [{"created_at": {"gte": "2024"}}]}

    def test_or_extra_entity_kwarg_copied_to_top_level(self):
        filters = {"OR": [{"user_id": "u1"}, {"agent_id": "a1"}]}
        result = build_search_filters(user_id="explicit", filters=filters)
        assert result == {**filters, "user_id": "explicit"}

    def test_shared_or_scope_copied_to_top_level(self):
        filters = {"OR": [{"user_id": "u1"}, {"user_id": "u1", "agent_id": "a1"}]}
        result = build_search_filters(filters=filters)
        assert result == {**filters, "user_id": "u1"}

    def test_or_without_shared_scope_raises(self):
        with pytest.raises(HTTPException) as exc:
            build_search_filters(filters={"OR": [{"user_id": "u1"}, {"created_at": {"gte": "2024"}}]})
        assert exc.value.status_code == 400

    def test_fallback_user_id_copied_to_unscoped_or(self):
        filters = {"OR": [{"user_id": "u1"}, {"created_at": {"gte": "2024"}}]}
        result = build_search_filters(filters=filters, fallback_user_id="fallback")
        assert result == {**filters, "user_id": "fallback"}

    def test_conflicting_explicit_and_and_scope_raises(self):
        with pytest.raises(HTTPException) as exc:
            build_search_filters(user_id="explicit", filters={"AND": [{"user_id": "from_filter"}]})
        assert exc.value.status_code == 400

    def test_raises_without_any_scope(self):
        with pytest.raises(HTTPException) as exc:
            build_search_filters()
        assert exc.value.status_code == 400

    def test_fallback_user_id(self):
        result = build_search_filters(fallback_user_id="fallback")
        assert result == {"user_id": "fallback"}

    def test_app_id_flat_filters_merged(self):
        result = build_search_filters(
            app_id="app1",
            filters={"created_at": {"gte": "2024-01-01"}},
        )
        assert result == {"app_id": "app1", "created_at": {"gte": "2024-01-01"}}

    def test_app_id_copied_to_top_level_for_and_filters(self):
        filters = {"AND": [{"user_id": "u1"}]}
        result = build_search_filters(app_id="app1", filters=filters)
        assert result["app_id"] == "app1"
        assert result["user_id"] == "u1"
        assert result["AND"] == [{"user_id": "u1"}]


class TestBuildCategoriesFilter:
    def test_single_category_uses_contains(self):
        assert build_categories_filter(["finance"]) == {"contains": "finance"}

    def test_multiple_categories_use_in(self):
        assert build_categories_filter(["finance", "travel"]) == {"in": ["finance", "travel"]}


# ---------------------------------------------------------------------------
# compat.responses
# ---------------------------------------------------------------------------


class TestDropNone:
    def test_removes_none_values(self):
        assert drop_none({"a": 1, "b": None, "c": "x"}) == {"a": 1, "c": "x"}

    def test_all_none(self):
        assert drop_none({"a": None, "b": None}) == {}

    def test_no_none(self):
        assert drop_none({"a": 1, "b": 2}) == {"a": 1, "b": 2}

    def test_empty_input(self):
        assert drop_none({}) == {}

    def test_does_not_remove_falsy_non_none(self):
        assert drop_none({"a": 0, "b": False, "c": ""}) == {"a": 0, "b": False, "c": ""}


class TestParseIsoTimestamp:
    def test_naive_timestamp_is_normalized_to_utc(self):
        parsed = parse_iso_timestamp("2026-01-01T12:00:00")
        assert parsed is not None
        assert parsed.tzinfo is not None
        assert parsed.utcoffset() == timezone.utc.utcoffset(None)

    def test_invalid_timestamp_returns_none(self):
        assert parse_iso_timestamp("not-a-time") is None


class TestNormalizeResults:
    def test_bare_list(self):
        raw = [{"id": "1"}, {"id": "2"}]
        assert normalize_results(raw) == raw

    def test_dict_with_results_key(self):
        raw = {"results": [{"id": "1"}], "count": 1}
        assert normalize_results(raw) == [{"id": "1"}]

    def test_empty_list(self):
        assert normalize_results([]) == []

    def test_empty_results_dict(self):
        assert normalize_results({"results": []}) == []

    def test_unknown_type_returns_empty(self):
        assert normalize_results("not a list") == []
        assert normalize_results(None) == []
        assert normalize_results(42) == []

    def test_dict_without_results_key_returns_empty(self):
        assert normalize_results({"count": 5}) == []


class TestNormalizeResultsDict:
    def test_bare_list(self):
        raw = [{"id": "1"}]
        assert normalize_results_dict(raw) == {"results": [{"id": "1"}]}

    def test_dict_with_results_key_passthrough(self):
        raw = {"results": [{"id": "1"}], "count": 1}
        assert normalize_results_dict(raw) == raw

    def test_empty_list(self):
        assert normalize_results_dict([]) == {"results": []}

    def test_unknown_type_returns_empty_results(self):
        assert normalize_results_dict(None) == {"results": []}
        assert normalize_results_dict("x") == {"results": []}

    def test_extra_fields_are_merged(self):
        raw = {"results": [{"id": "1"}], "count": 1}
        assert normalize_results_dict(raw, extra={"status": "ok"}) == {
            "results": [{"id": "1"}],
            "count": 1,
            "status": "ok",
        }


class TestMetadataMergeHelpers:
    def test_merge_v1_add_metadata_allows_explicit_empty_categories(self):
        merged = merge_v1_add_metadata(
            {"categories": ["legacy"]},
            source=None,
            platform=None,
            categories=[],
        )
        assert merged is not None
        assert merged["categories"] == []

    def test_merge_v3_add_metadata_accepts_empty_extra_metadata_dict(self):
        merged = merge_v3_add_metadata(
            {"source": "body-source", "keep": True},
            source="HEADER",
            platform="cursor",
            extra_metadata={},
        )
        assert merged == {"source": "body-source", "keep": True, "platform": "cursor"}


class TestBuildExtractionPrompt:
    def test_returns_none_when_nothing_provided(self):
        assert (
            build_extraction_prompt(
                custom_instructions=None,
                agent_custom_instructions=None,
                includes=None,
                excludes=None,
                has_agent_scope=False,
            )
            is None
        )

    def test_uses_custom_instructions_for_user_scope(self):
        prompt = build_extraction_prompt(
            custom_instructions="extract preferences",
            agent_custom_instructions=None,
            includes=None,
            excludes=None,
            has_agent_scope=False,
        )
        assert prompt == "extract preferences"

    def test_agent_custom_instructions_win_when_agent_scoped(self):
        prompt = build_extraction_prompt(
            custom_instructions="extract preferences",
            agent_custom_instructions="extract tool failures",
            includes=None,
            excludes=None,
            has_agent_scope=True,
        )
        assert prompt == "extract tool failures"

    def test_custom_instructions_used_when_not_agent_scoped_even_if_agent_instr_set(self):
        prompt = build_extraction_prompt(
            custom_instructions="extract preferences",
            agent_custom_instructions="extract tool failures",
            includes=None,
            excludes=None,
            has_agent_scope=False,
        )
        assert prompt == "extract preferences"

    def test_explicit_empty_agent_custom_instructions_falls_back_to_config(self):
        """Empty string is treated as None (no override) — SDK uses its config-level custom_instructions."""
        prompt = build_extraction_prompt(
            custom_instructions="extract preferences",
            agent_custom_instructions="",
            includes=None,
            excludes=None,
            has_agent_scope=True,
        )
        # agent_custom_instructions="" is falsy → falls back to custom_instructions
        assert prompt == "extract preferences"

    def test_includes_and_excludes_appended_as_constraints(self):
        prompt = build_extraction_prompt(
            custom_instructions="base instructions",
            agent_custom_instructions=None,
            includes="vehicles",
            excludes="politics",
            has_agent_scope=False,
        )
        assert "base instructions" in prompt
        assert "Include only: vehicles" in prompt
        assert "Exclude: politics" in prompt

    def test_constraints_only_without_base_instruction(self):
        prompt = build_extraction_prompt(
            custom_instructions=None,
            agent_custom_instructions=None,
            includes="vehicles",
            excludes=None,
            has_agent_scope=False,
        )
        assert "Include only: vehicles" in prompt
        assert "Extraction constraints:" in prompt

    def test_agent_custom_instructions_plus_constraints(self):
        prompt = build_extraction_prompt(
            custom_instructions="user prefs",
            agent_custom_instructions="tool failures",
            includes="errors",
            excludes="user details",
            has_agent_scope=True,
        )
        assert prompt.startswith("tool failures")
        assert "Include only: errors" in prompt
        assert "Exclude: user details" in prompt


class TestNormalizeTimestamp:
    def test_none_returns_none(self):
        assert normalize_timestamp(None) is None

    def test_unix_epoch_to_iso(self):
        result = normalize_timestamp(1700000000)
        # 1700000000 = 2023-11-14 in UTC
        assert result.startswith("2023-11-14")

    def test_out_of_range_raises(self):
        with pytest.raises(ValueError):
            normalize_timestamp(99999999999999999999)


class TestResolveEventOwnerId:
    def test_extracts_from_user_object(self):
        auth = MagicMock()
        auth.id = "user-1"
        assert resolve_event_owner_id(auth) == "user-1"

    def test_extracts_from_dict_id(self):
        assert resolve_event_owner_id({"id": "user-2"}) == "user-2"

    def test_list_auth_ignores_embedded_id(self):
        assert resolve_event_owner_id([{"id": "user-3"}]) is None

    def test_falls_back_to_entity_params_user_id(self):
        assert resolve_event_owner_id(None, {"user_id": "scoped-user"}) == "scoped-user"
        assert resolve_event_owner_id([], {"user_id": "scoped-user"}) == "scoped-user"
        assert resolve_event_owner_id([{"id": "user-3"}], {"user_id": "scoped-user"}) == "scoped-user"

    def test_blank_auth_owner_id_falls_back_to_scoped_user(self):
        assert resolve_event_owner_id({"id": "   "}, {"user_id": "scoped-user"}) == "scoped-user"

    def test_blank_auth_owner_id_without_scope_returns_none(self):
        assert resolve_event_owner_id({"id": "   "}) is None


class TestCompatEvent:
    def test_pending_sets_timestamps_and_empty_results(self):
        event = CompatEvent.pending("evt-1", now_iso="2024-01-01T00:00:00+00:00")
        assert event.status == "PENDING"
        assert event.results == []
        assert event.completed_at is None
        assert event.latency is None
        assert event.created_at == "2024-01-01T00:00:00+00:00"

    def test_create_add_succeeded_sets_completed_at(self):
        event = CompatEvent.create_add(
            "evt-2",
            [{"id": "m1"}],
            now_iso="2024-01-01T00:00:00+00:00",
            latency=12.5,
        )
        assert event.status == "SUCCEEDED"
        assert event.completed_at == "2024-01-01T00:00:00+00:00"
        assert event.latency == 12.5
        assert len(event.results) == 1

    def test_create_add_failed_keeps_completed_at_optional(self):
        event = CompatEvent.create_add(
            "evt-3",
            [],
            status="FAILED",
            now_iso="2024-01-01T00:00:00+00:00",
            completed_at="2024-01-01T00:00:01+00:00",
            metadata={"error": "boom"},
        )
        assert event.status == "FAILED"
        assert event.completed_at == "2024-01-01T00:00:01+00:00"
        assert event.metadata == {"error": "boom"}


class TestEventCacheCopies:
    @pytest.fixture(autouse=True)
    def _clear_events(self):
        event_cache_clear()
        yield
        event_cache_clear()

    def test_put_validates_and_detaches_from_input(self):
        event_obj = {
            "id": "evt-1",
            "event_type": "ADD",
            "status": "PENDING",
            "payload": {},
            "metadata": {"source": "caller"},
            "results": [],
            "created_at": "2024-01-01T00:00:00+00:00",
            "updated_at": "2024-01-01T00:00:00+00:00",
            "started_at": "2024-01-01T00:00:00+00:00",
            "completed_at": None,
            "latency": None,
        }

        event_cache_put("evt-1", event_obj)
        event_obj["status"] = "FAILED"

        cached = event_cache_get("evt-1")
        assert cached is not None
        assert cached["status"] == "PENDING"

    def test_get_returns_copy(self):
        event_cache_put(
            "evt-1",
            {
                "id": "evt-1",
                "event_type": "ADD",
                "status": "PENDING",
                "payload": {},
                "metadata": None,
                "results": [],
                "created_at": "2024-01-01T00:00:00+00:00",
                "updated_at": "2024-01-01T00:00:00+00:00",
                "started_at": "2024-01-01T00:00:00+00:00",
                "completed_at": None,
                "latency": None,
            },
        )

        cached = event_cache_get("evt-1")
        assert cached is not None
        cached["status"] = "FAILED"

        fresh = event_cache_get("evt-1")
        assert fresh is not None
        assert fresh["status"] == "PENDING"

    def test_all_returns_copies(self):
        event_cache_put(
            "evt-1",
            {
                "id": "evt-1",
                "event_type": "ADD",
                "status": "PENDING",
                "payload": {},
                "metadata": None,
                "results": [],
                "created_at": "2024-01-01T00:00:00+00:00",
                "updated_at": "2024-01-01T00:00:00+00:00",
                "started_at": "2024-01-01T00:00:00+00:00",
                "completed_at": None,
                "latency": None,
            },
        )

        listed = event_cache_all()
        listed[0]["status"] = "FAILED"

        fresh = event_cache_get("evt-1")
        assert fresh is not None
        assert fresh["status"] == "PENDING"

    def test_update_returns_copy(self):
        event_cache_put(
            "evt-1",
            {
                "id": "evt-1",
                "event_type": "ADD",
                "status": "PENDING",
                "payload": {},
                "metadata": None,
                "results": [],
                "created_at": "2024-01-01T00:00:00+00:00",
                "updated_at": "2024-01-01T00:00:00+00:00",
                "started_at": "2024-01-01T00:00:00+00:00",
                "completed_at": None,
                "latency": None,
            },
        )

        updated = event_cache_update("evt-1", status="SUCCEEDED")
        assert updated is not None
        updated["status"] = "FAILED"

        fresh = event_cache_get("evt-1")
        assert fresh is not None
        assert fresh["status"] == "SUCCEEDED"

    def test_update_preserves_owner_id(self):
        event_cache_put(
            "evt-1",
            {
                "id": "evt-1",
                "event_type": "ADD",
                "status": "PENDING",
                "payload": {},
                "metadata": None,
                "results": [],
                "created_at": "2024-01-01T00:00:00+00:00",
                "updated_at": "2024-01-01T00:00:00+00:00",
                "started_at": "2024-01-01T00:00:00+00:00",
                "completed_at": None,
                "latency": None,
                "owner_id": "user-1",
            },
        )

        updated = event_cache_update("evt-1", status="SUCCEEDED", owner_id="user-2")
        assert updated is not None
        assert updated["owner_id"] == "user-1"

    def test_update_rejects_invalid_status(self):
        event_cache_put("evt-1", CompatEvent.pending("evt-1", owner_id="user-1"))
        with pytest.raises(ValidationError):
            event_cache_update("evt-1", status="NOT_A_STATUS")


class TestRunV3AddMemoryTask:
    def test_warns_when_success_update_misses_event_cache(self, monkeypatch, caplog):
        monkeypatch.setattr(compat_tasks, "entity_scope_from_params", lambda params: {"user_id": "u1"})
        monkeypatch.setattr(compat_tasks, "run_memory_write", lambda callback, scope: {"results": [{"id": "m1"}]})
        monkeypatch.setattr(compat_tasks, "event_cache_update", lambda event_id, **fields: None)

        with caplog.at_level(logging.WARNING, logger="mem0.server.compat.tasks"):
            compat_tasks.run_v3_add_memory_task(
                "evt-1",
                [{"role": "user", "content": "remember"}],
                {"user_id": "u1"},
            )

        assert "completed but event cache update missed" in caplog.text

    def test_warns_when_failure_update_misses_event_cache(self, monkeypatch, caplog):
        monkeypatch.setattr(compat_tasks, "entity_scope_from_params", lambda params: {"user_id": "u1"})

        def _raise_write(_callback, _scope):
            raise RuntimeError("boom")

        monkeypatch.setattr(compat_tasks, "run_memory_write", _raise_write)
        monkeypatch.setattr(compat_tasks, "event_cache_update", lambda event_id, **fields: None)

        with caplog.at_level(logging.WARNING, logger="mem0.server.compat.tasks"):
            compat_tasks.run_v3_add_memory_task(
                "evt-2",
                [{"role": "user", "content": "remember"}],
                {"user_id": "u1"},
            )

        assert "failed but event cache update missed" in caplog.text


# ---------------------------------------------------------------------------
# build_list_filters
# ---------------------------------------------------------------------------


class TestBuildListFilters:
    def _body(self, **kwargs: Any) -> MemoryGetInputV2:
        return MemoryGetInputV2(**kwargs)

    def test_no_filters_falls_back_to_entity_params(self):
        body = self._body()
        result = build_list_filters(body, {"user_id": "u1"})
        assert result == {"user_id": "u1"}

    def test_flat_filters_preserved(self):
        body = self._body(filters={"user_id": "u1", "created_at": {"gte": "2024-01-01"}})
        result = build_list_filters(body, {"user_id": "u1"})
        assert result == {"user_id": "u1", "created_at": {"gte": "2024-01-01"}}

    def test_start_date_added(self):
        body = self._body(filters={"user_id": "u1"}, start_date="2024-01-01")
        result = build_list_filters(body, {"user_id": "u1"})
        assert result["created_at"] == {"gte": "2024-01-01"}

    def test_end_date_added(self):
        body = self._body(filters={"user_id": "u1"}, end_date="2024-12-31")
        result = build_list_filters(body, {"user_id": "u1"})
        assert result["created_at"] == {"lte": "2024-12-31"}

    def test_date_range_combined(self):
        body = self._body(
            filters={"user_id": "u1"},
            start_date="2024-01-01",
            end_date="2024-12-31",
        )
        result = build_list_filters(body, {"user_id": "u1"})
        assert result["created_at"] == {"gte": "2024-01-01", "lte": "2024-12-31"}

    def test_categories_added(self):
        body = self._body(filters={"user_id": "u1"}, categories=["finance"])
        result = build_list_filters(body, {"user_id": "u1"})
        assert result["categories"] == {"contains": "finance"}

    def test_multiple_categories_use_in_operator(self):
        body = self._body(filters={"user_id": "u1"}, categories=["finance", "travel"])
        result = build_list_filters(body, {"user_id": "u1"})
        assert result["categories"] == {"in": ["finance", "travel"]}

    def test_existing_created_at_not_overridden(self):
        """setdefault: body.filters already has created_at, date params should not override."""
        body = self._body(
            filters={"user_id": "u1", "created_at": {"gte": "2023-01-01"}},
            start_date="2024-01-01",
        )
        result = build_list_filters(body, {"user_id": "u1"})
        assert result["created_at"] == {"gte": "2023-01-01"}

    def test_existing_categories_not_overridden(self):
        body = self._body(
            filters={"user_id": "u1", "categories": {"in": ["personal"]}},
            categories=["finance"],
        )
        result = build_list_filters(body, {"user_id": "u1"})
        assert result["categories"] == {"in": ["personal"]}

    def test_not_categories_does_not_block_categories_convenience_filter(self):
        body = self._body(
            filters={"AND": [{"user_id": "u1"}, {"NOT": {"categories": {"contains": "archived"}}}]},
            categories=["finance"],
        )
        result = build_list_filters(body, {"user_id": "u1"})
        assert {"categories": {"contains": "finance"}} in result["AND"]

    def test_partial_or_created_at_does_not_block_date_convenience_filter(self):
        body = self._body(
            filters={"OR": [{"created_at": {"gte": "2023-01-01"}}, {"kind": "note"}]},
            start_date="2024-01-01",
        )
        result = build_list_filters(body, {"user_id": "u1"})
        assert {"created_at": {"gte": "2024-01-01"}} in result["AND"]

    def test_or_with_categories_in_every_branch_blocks_categories_convenience_filter(self):
        body = self._body(
            filters={"OR": [{"categories": {"contains": "personal"}}, {"categories": {"contains": "work"}}]},
            categories=["finance"],
        )
        result = build_list_filters(body, {"user_id": "u1"})
        assert result == {
            "OR": [{"categories": {"contains": "personal"}}, {"categories": {"contains": "work"}}],
            "user_id": "u1",
        }

    def test_positive_key_scan_checks_or_after_non_matching_and(self):
        body = self._body(
            filters={
                "AND": [{"kind": "note"}],
                "OR": [{"created_at": {"gte": "2023-01-01"}}, {"created_at": {"gte": "2022-01-01"}}],
            },
            start_date="2024-01-01",
        )
        result = build_list_filters(body, {"user_id": "u1"})
        assert result == {
            "AND": [{"kind": "note"}],
            "OR": [{"created_at": {"gte": "2023-01-01"}}, {"created_at": {"gte": "2022-01-01"}}],
            "user_id": "u1",
        }

    def test_and_format_skips_date_categories_merge(self):
        """Logical format: convenience fields are AND-ed at top level."""
        body = self._body(
            filters={"AND": [{"user_id": "u1"}]},
            start_date="2024-01-01",
            categories=["finance"],
        )
        result = build_list_filters(body, {"user_id": "u1"})
        assert result["user_id"] == "u1"
        assert "AND" in result
        assert {"created_at": {"gte": "2024-01-01"}} in result["AND"]
        assert {"categories": {"contains": "finance"}} in result["AND"]

    def test_and_filters_copy_entity_scope_to_top_level(self):
        body = self._body(filters={"AND": [{"user_id": "u1"}, {"created_at": {"gte": "2024"}}]})
        result = build_list_filters(body, {"user_id": "u1"})
        assert result == {"AND": [{"user_id": "u1"}, {"created_at": {"gte": "2024"}}], "user_id": "u1"}

    def test_or_filters_copy_entity_scope_to_top_level(self):
        body = self._body(filters={"OR": [{"user_id": "u1"}, {"agent_id": "a1"}]})
        result = build_list_filters(body, {"user_id": "u1", "agent_id": "a1"})
        assert result == {"OR": [{"user_id": "u1"}, {"agent_id": "a1"}], "user_id": "u1", "agent_id": "a1"}

    def test_does_not_mutate_body_filters(self):
        original = {"user_id": "u1"}
        body = self._body(filters=original, start_date="2024-01-01")
        build_list_filters(body, {"user_id": "u1"})
        assert original == {"user_id": "u1"}


# ---------------------------------------------------------------------------
# paginate_response
# ---------------------------------------------------------------------------


class TestPaginateResponse:
    def _request(self, path: str = "/v2/memories/", params: dict | None = None) -> MagicMock:
        query = "&".join(f"{k}={v}" for k, v in (params or {}).items())
        base = f"http://testserver{path}"
        if query:
            base = f"{base}?{query}"
        req = MagicMock()
        req.url = URL(base)
        req.query_params = params or {}
        return req

    def test_first_page_no_previous(self):
        req = self._request()
        items = list(range(25))
        result = paginate_response(req, items, page=1, page_size=10)
        assert result["count"] == 25
        assert result["previous"] is None
        assert result["next"] is not None
        assert result["results"] == list(range(10))

    def test_last_page_no_next(self):
        req = self._request()
        items = list(range(25))
        result = paginate_response(req, items, page=3, page_size=10)
        assert result["count"] == 25
        assert result["next"] is None
        assert result["previous"] is not None
        assert result["results"] == [20, 21, 22, 23, 24]

    def test_single_page(self):
        req = self._request()
        items = [1, 2, 3]
        result = paginate_response(req, items, page=1, page_size=10)
        assert result["count"] == 3
        assert result["next"] is None
        assert result["previous"] is None
        assert result["results"] == [1, 2, 3]

    def test_empty_items(self):
        req = self._request()
        result = paginate_response(req, [], page=1, page_size=10)
        assert result["count"] == 0
        assert result["results"] == []
        assert result["next"] is None
        assert result["previous"] is None

    def test_middle_page_has_both(self):
        req = self._request()
        items = list(range(30))
        result = paginate_response(req, items, page=2, page_size=10)
        assert result["count"] == 30
        assert result["next"] is not None
        assert result["previous"] is not None
        assert result["results"] == list(range(10, 20))

    def test_next_url_contains_page_param(self):
        req = self._request()
        items = list(range(25))
        result = paginate_response(req, items, page=1, page_size=10)
        assert "page=2" in result["next"]
        assert "page_size=10" in result["next"]

    def test_previous_url_contains_page_param(self):
        req = self._request()
        items = list(range(25))
        result = paginate_response(req, items, page=3, page_size=10)
        assert "page=2" in result["previous"]

    def test_total_uses_supplied_count_without_reslicing(self):
        """When total is provided, items is the already-paginated DB slice — it
        must not be re-sliced, and count/next/previous use the supplied total
        (this is the list_users DB-level pagination path)."""
        req = self._request()
        # DB returned page 2 (items 10-19 of 95 total); pass the slice + total.
        page_items = list(range(10, 20))
        result = paginate_response(req, page_items, page=2, page_size=10, total=95)
        assert result["count"] == 95
        assert result["results"] == list(range(10, 20))  # not re-sliced
        assert result["next"] is not None  # 20 < 95
        assert result["previous"] is not None  # page 2

    def test_total_last_page_no_next(self):
        req = self._request()
        result = paginate_response(req, [90, 91, 92, 93, 94], page=10, page_size=10, total=95)
        assert result["count"] == 95
        assert result["next"] is None  # start(90) + page_size(10) = 100 >= 95
        assert result["results"] == [90, 91, 92, 93, 94]


class TestResolveOptionalPagination:
    def test_returns_none_when_both_omitted(self):
        assert resolve_optional_pagination(None, None) is None

    def test_page_only_defaults_page_size(self):
        assert resolve_optional_pagination(2, None) == (2, 50)

    def test_page_size_only_defaults_page(self):
        assert resolve_optional_pagination(None, 25) == (1, 25)

    def test_clamps_page_size_to_max(self):
        assert resolve_optional_pagination(1, 500) == (1, 100)

    def test_normalizes_invalid_page(self):
        assert resolve_optional_pagination(0, 10) == (1, 10)


# ---------------------------------------------------------------------------
# apply_fields
# ---------------------------------------------------------------------------


class TestApplyFields:
    def test_none_fields_returns_unchanged(self):
        items = [{"id": "1", "memory": "a", "user_id": "u1"}]
        assert apply_fields(items, None) == items

    def test_empty_fields_returns_unchanged(self):
        items = [{"id": "1", "memory": "a", "user_id": "u1"}]
        assert apply_fields(items, []) == items

    def test_filters_to_requested_fields(self):
        items = [{"id": "1", "memory": "a", "user_id": "u1", "score": 0.9}]
        result = apply_fields(items, ["id", "memory"])
        assert result == [{"id": "1", "memory": "a"}]

    def test_unknown_fields_dropped(self):
        items = [{"id": "1", "memory": "a"}]
        result = apply_fields(items, ["id", "nonexistent"])
        assert result == [{"id": "1"}]


class TestWarnIgnoredCompatParams:
    def test_no_ignored_params_no_warning(self, caplog):
        with caplog.at_level(logging.WARNING):
            warn_ignored_compat_params("v3_search_memories", latest_only=None, reference_date=None)
        assert "compatibility parameters" not in caplog.text

    def test_emits_warning_for_non_none_params(self, caplog):
        with caplog.at_level(logging.WARNING):
            warn_ignored_compat_params("v3_search_memories", latest_only=False, reference_date="2024-06-01")

        assert "v3_search_memories" in caplog.text
        assert "latest_only" in caplog.text
        assert "reference_date" in caplog.text

    def test_v2_list_applies_fields_and_warns_latest_only(self, monkeypatch, caplog):
        mem = MagicMock()
        mem.count.return_value = 0
        mem.get_all.return_value = [{"id": "m1", "memory": "a", "user_id": "u1", "score": 0.9}]
        monkeypatch.setattr("server.compat.helpers.get_memory_instance", lambda: mem)

        req = MagicMock()
        req.url = URL("http://testserver/v2/memories?page=1&page_size=10")
        body = MemoryGetInputV2(filters={"user_id": "u1"}, fields=["id"], latest_only=True)

        with caplog.at_level(logging.WARNING):
            response = v2_list_memories(req, body, page=1, page_size=10, auth=None)

        assert "latest_only" in caplog.text
        assert response["results"] == [{"id": "m1"}]
        mem.get_all.assert_called_once_with(filters={"user_id": "u1"}, top_k=11, skip=0)

    def test_v3_search_warns_for_reference_date_and_latest_only(self, monkeypatch, caplog):
        mem = MagicMock()
        mem.search.return_value = []
        monkeypatch.setattr("server.routers.compat.get_memory_instance", lambda: mem)

        body = MemorySearchInputV3(
            query="hello",
            user_id="u1",
            reference_date="2024-06-01",
            latest_only=True,
        )

        with caplog.at_level(logging.WARNING):
            v3_search_memories(body, request=_REQ, auth=None)

        assert "v3_search_memories" in caplog.text
        assert "reference_date" in caplog.text
        assert "latest_only" in caplog.text
        mem.search.assert_called_once_with(query="hello", filters={"user_id": "u1"})

    def test_v3_search_accepts_keyword_search_without_422(self, monkeypatch):
        """keyword_search is accepted (no 422) but not forwarded — OSS already runs hybrid."""
        mem = MagicMock()
        mem.search.return_value = []
        monkeypatch.setattr("server.routers.compat.get_memory_instance", lambda: mem)

        body = MemorySearchInputV3(
            query="hello",
            user_id="u1",
            keyword_search=True,
        )

        v3_search_memories(body, request=_REQ, auth=None)

        # keyword_search is accepted by schema but not forwarded to SDK
        call_kwargs = mem.search.call_args.kwargs
        assert "keyword_search" not in call_kwargs


# ---------------------------------------------------------------------------
# build_search_kwargs
# ---------------------------------------------------------------------------


class TestBuildSearchKwargs:
    def test_filters_always_present(self):
        result = build_search_kwargs({"user_id": "u1"}, None, None)
        assert result == {"filters": {"user_id": "u1"}}

    def test_top_k_included(self):
        result = build_search_kwargs({"user_id": "u1"}, 5, None)
        assert result["top_k"] == 5

    def test_threshold_included(self):
        result = build_search_kwargs({"user_id": "u1"}, None, 0.7)
        assert result["threshold"] == 0.7

    def test_rerank_included(self):
        result = build_search_kwargs({"user_id": "u1"}, None, None, rerank=True)
        assert result["rerank"] is True

    def test_none_kwargs_omitted(self):
        result = build_search_kwargs({"user_id": "u1"}, None, None, None)
        assert "top_k" not in result
        assert "threshold" not in result
        assert "rerank" not in result

    def test_all_params(self):
        result = build_search_kwargs({"user_id": "u1"}, 10, 0.5, rerank=False)
        assert result == {"filters": {"user_id": "u1"}, "top_k": 10, "threshold": 0.5, "rerank": False}

    def test_zero_threshold_included(self):
        """threshold=0.0 is falsy but must be included (disables score filtering)."""
        result = build_search_kwargs({"user_id": "u1"}, None, 0.0)
        assert "threshold" in result
        assert result["threshold"] == 0.0

    def test_zero_top_k_included(self):
        result = build_search_kwargs({"user_id": "u1"}, 0, None)
        assert "top_k" in result
        assert result["top_k"] == 0


# ---------------------------------------------------------------------------
# v3_search_memories convenience fields
# ---------------------------------------------------------------------------


class TestV3SearchMemoriesConvenienceFields:
    def test_categories_and_metadata_applied_with_logical_filters(self, monkeypatch):
        mem = MagicMock()
        mem.search.return_value = {"results": []}

        def _get_mem():
            return mem

        monkeypatch.setattr("server.routers.compat.get_memory_instance", _get_mem)

        body = MemorySearchInputV3(
            query="hello",
            user_id="u1",
            filters={"OR": [{"kind": "a"}, {"kind": "b"}]},
            categories=["finance"],
            metadata={"foo": "bar"},
        )

        v3_search_memories(body, request=_REQ, auth=None)

        called = mem.search.call_args.kwargs
        assert called["query"] == "hello"
        filters = called["filters"]
        # Convenience filters wrap the scoped OR tree in an outer AND.
        assert "AND" in filters
        assert {"categories": {"contains": "finance"}} in filters["AND"]
        assert {"foo": "bar"} in filters["AND"]


# ---------------------------------------------------------------------------
# resolve_existing
# ---------------------------------------------------------------------------


class TestResolveExisting:
    def test_returns_dict_when_sdk_returns_dict(self):
        mem = MagicMock()
        mem.get.return_value = {"id": "mem-1", "memory": "hello"}
        result = resolve_existing(mem, "mem-1")
        assert result == {"id": "mem-1", "memory": "hello"}
        mem.get.assert_called_once_with("mem-1")

    def test_unwraps_single_item_list(self):
        mem = MagicMock()
        mem.get.return_value = [{"id": "mem-1", "memory": "hello"}]
        result = resolve_existing(mem, "mem-1")
        assert result == {"id": "mem-1", "memory": "hello"}

    def test_unwraps_list_takes_first_element(self):
        """When SDK returns a multi-element list, resolve_existing picks index 0."""
        mem = MagicMock()
        mem.get.return_value = [{"id": "a"}, {"id": "b"}]
        result = resolve_existing(mem, "a")
        assert result == {"id": "a"}

    def test_raises_404_on_none(self):
        mem = MagicMock()
        mem.get.return_value = None
        with pytest.raises(HTTPException) as exc:
            resolve_existing(mem, "mem-x")
        assert exc.value.status_code == 404
        assert "mem-x" in exc.value.detail

    def test_raises_404_on_empty_list(self):
        mem = MagicMock()
        mem.get.return_value = []
        with pytest.raises(HTTPException) as exc:
            resolve_existing(mem, "mem-x")
        assert exc.value.status_code == 404

    def test_raises_404_on_non_dict(self):
        mem = MagicMock()
        mem.get.return_value = "just a string"
        with pytest.raises(HTTPException) as exc:
            resolve_existing(mem, "mem-x")
        assert exc.value.status_code == 404


# ---------------------------------------------------------------------------
# v1_update_memory — field forwarding
# ---------------------------------------------------------------------------


class TestV1UpdateMemoryForwarding:
    """The route forwards only explicitly-set fields to ``Memory.update``. The SDK
    preserves existing text when ``data`` is omitted and merges metadata into the
    existing payload (non-replacing), so the route no longer pre-fetches. Not-found
    ``ValueError`` maps to 404; other ``ValueError``s fall through to
    ``@upstream_guard`` (400)."""

    @staticmethod
    def _patch_run(monkeypatch, mem):
        def _run(callback, memory_id):
            return callback(mem)

        monkeypatch.setattr("server.routers.compat.run_memory_write_for_memory_id", _run)

    def test_forwards_text_and_metadata(self, monkeypatch):
        mem = MagicMock()
        mem.update.return_value = {"message": "updated"}
        self._patch_run(monkeypatch, mem)
        v1_update_memory("mem-1", MemoryUpdateInput(text="new", metadata={"k": "v"}), request=_REQ, auth=None)
        mem.update.assert_called_once_with(memory_id="mem-1", data="new", metadata={"k": "v"})

    def test_text_only_omits_metadata(self, monkeypatch):
        mem = MagicMock()
        mem.update.return_value = {"message": "updated"}
        self._patch_run(monkeypatch, mem)
        v1_update_memory("mem-1", MemoryUpdateInput(text="new"), request=_REQ, auth=None)
        mem.update.assert_called_once_with(memory_id="mem-1", data="new")

    def test_metadata_only_omits_data(self, monkeypatch):
        """text omitted -> ``data`` is not forwarded; the SDK preserves existing text."""
        mem = MagicMock()
        mem.update.return_value = {"message": "updated"}
        self._patch_run(monkeypatch, mem)
        v1_update_memory("mem-1", MemoryUpdateInput(metadata={"k": "v"}), request=_REQ, auth=None)
        mem.update.assert_called_once_with(memory_id="mem-1", metadata={"k": "v"})

    def test_timestamp_backdates_created_at(self, monkeypatch):
        """timestamp is converted to a UTC ISO created_at in metadata, not stored as a dead timestamp key."""
        mem = MagicMock()
        mem.update.return_value = {"message": "updated"}
        self._patch_run(monkeypatch, mem)
        v1_update_memory(
            "mem-1",
            MemoryUpdateInput(timestamp=1700000000, metadata={"k": "v"}),
            request=_REQ,
            auth=None,
        )
        call_metadata = mem.update.call_args.kwargs["metadata"]
        assert call_metadata["k"] == "v"
        assert call_metadata["created_at"].startswith("2023-11-14")
        # timestamp itself is NOT leaked into metadata
        assert "timestamp" not in call_metadata

    def test_invalid_timestamp_returns_422(self, monkeypatch):
        """Out-of-range timestamp produces a 422, not a 500."""
        with pytest.raises(HTTPException) as exc:
            v1_update_memory(
                "mem-1",
                MemoryUpdateInput(timestamp=99999999999999999999, metadata={}),
                request=_REQ,
                auth=None,
            )
        assert exc.value.status_code == 422

    def test_not_found_returns_404(self, monkeypatch):
        mem = MagicMock()
        mem.update.side_effect = ValueError("Memory with id mem-x not found. Please provide a valid 'memory_id'")
        self._patch_run(monkeypatch, mem)
        with pytest.raises(HTTPException) as exc:
            v1_update_memory("mem-x", MemoryUpdateInput(text="new"), request=_REQ, auth=None)
        assert exc.value.status_code == 404

    def test_other_value_error_maps_to_400(self, monkeypatch):
        """Non-not-found ValueError is re-raised; @upstream_guard converts it to 400."""
        mem = MagicMock()
        mem.update.side_effect = ValueError("data must be a non-empty string")
        self._patch_run(monkeypatch, mem)
        with pytest.raises(HTTPException) as exc:
            v1_update_memory("mem-1", MemoryUpdateInput(text="new"), request=_REQ, auth=None)
        assert exc.value.status_code == 400

    def test_no_fields_rejected_400(self):
        """No text/metadata/timestamp/expiration_date set -> 400 (nothing to update)."""
        with pytest.raises(HTTPException) as exc:
            v1_update_memory("mem-1", MemoryUpdateInput(), request=_REQ, auth=None)
        assert exc.value.status_code == 400

    def test_invalid_expiration_date_returns_422(self):
        """Invalid date string is rejected at schema validation (422), not 400."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            MemoryUpdateInput(expiration_date="not-a-date")

    def test_v3_add_invalid_expiration_date_returns_422(self):
        """Invalid date string is rejected at schema validation (422), not 400."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            MemoryAddInputV3(
                messages=[{"role": "user", "content": "x"}],
                user_id="u1",
                expiration_date="not-a-date",
            )


# ---------------------------------------------------------------------------
# v1_batch_update — route-level boundaries
# ---------------------------------------------------------------------------


class TestV1BatchUpdateRoute:
    @staticmethod
    def _patch_run(monkeypatch, mem):
        def _run(callback, memory_id):
            return callback(mem)

        monkeypatch.setattr("server.routers.compat.run_memory_write_for_memory_id", _run)

    def test_too_many_items_returns_400(self):
        items = [MemoryBatchUpdateItem(memory_id=f"m{i}", text="x") for i in range(101)]
        with pytest.raises(HTTPException) as exc:
            v1_batch_update(MemoryBatchUpdateInput(memories=items), request=_REQ, auth=None)
        assert exc.value.status_code == 400
        assert "100" in exc.value.detail

    def test_item_missing_text_and_metadata_returns_400(self):
        items = [MemoryBatchUpdateItem(memory_id="m1")]
        with pytest.raises(HTTPException) as exc:
            v1_batch_update(MemoryBatchUpdateInput(memories=items), request=_REQ, auth=None)
        assert exc.value.status_code == 400
        assert "m1" in exc.value.detail

    def test_mixed_valid_and_not_found_counts_successes(self, monkeypatch):
        mem = MagicMock()

        def _update(memory_id, **kw):
            if memory_id == "m2":
                raise ValueError("Memory with id m2 not found. Please provide a valid 'memory_id'")
            return {"message": "updated"}

        mem.update.side_effect = _update
        self._patch_run(monkeypatch, mem)
        items = [
            MemoryBatchUpdateItem(memory_id="m1", text="a"),
            MemoryBatchUpdateItem(memory_id="m2", text="b"),
            MemoryBatchUpdateItem(memory_id="m3", metadata={"k": "v"}),
        ]
        result = v1_batch_update(MemoryBatchUpdateInput(memories=items), request=_REQ, auth=None)
        assert result == {"message": "Memories updated successfully, count: 2."}

    def test_all_valid_counts_successes(self, monkeypatch):
        mem = MagicMock()
        mem.update.return_value = {"message": "updated"}
        self._patch_run(monkeypatch, mem)
        items = [
            MemoryBatchUpdateItem(memory_id="m1", text="a"),
            MemoryBatchUpdateItem(memory_id="m2", metadata={"k": "v"}),
        ]
        result = v1_batch_update(MemoryBatchUpdateInput(memories=items), request=_REQ, auth=None)
        assert result == {"message": "Memories updated successfully, count: 2."}


# ---------------------------------------------------------------------------
# v1_batch_delete — route-level boundaries
# ---------------------------------------------------------------------------


class TestV1BatchDeleteRoute:
    @staticmethod
    def _patch_run(monkeypatch, mem):
        def _run(callback, memory_id):
            return callback(mem)

        monkeypatch.setattr("server.routers.compat.run_memory_write_for_memory_id", _run)

    def test_too_many_returns_400(self):
        ids = [f"m{i}" for i in range(1001)]
        with pytest.raises(HTTPException) as exc:
            v1_batch_delete(MemoryBatchDeleteInput(memory_ids=ids), request=_REQ, auth=None)
        assert exc.value.status_code == 400

    def test_mixed_valid_and_not_found_counts_successes(self, monkeypatch):
        mem = MagicMock()

        def _delete(memory_id):
            if memory_id == "m2":
                raise ValueError("Memory with id m2 not found")
            return None

        mem.delete.side_effect = _delete
        self._patch_run(monkeypatch, mem)
        result = v1_batch_delete(MemoryBatchDeleteInput(memory_ids=["m1", "m2", "m3"]), request=_REQ, auth=None)
        assert result == {"message": "Memories deleted successfully, count: 2."}

    def test_legacy_format_accepted(self, monkeypatch):
        mem = MagicMock()
        mem.delete.return_value = None
        self._patch_run(monkeypatch, mem)
        body = MemoryBatchDeleteLegacyInput(memories=[{"memory_id": "m1"}])
        result = v1_batch_delete(body, request=_REQ, auth=None)
        assert result == {"message": "Memories deleted successfully, count: 1."}


# ---------------------------------------------------------------------------
# upstream_guard exception mapping
# ---------------------------------------------------------------------------


class TestUpstreamGuardExceptionMapping:
    def _make_wrapped(self, side_effect=None):
        """Create a function wrapped with @upstream_guard that raises the given side_effect."""

        @upstream_guard
        def handler():
            if side_effect:
                raise side_effect
            return "ok"

        return handler

    def test_mem0_validation_error_maps_to_400(self):
        wrapped = self._make_wrapped(Mem0ValidationError("bad input", error_code="VAL_001"))
        with pytest.raises(HTTPException) as exc:
            wrapped()
        assert exc.value.status_code == 400
        assert "bad input" in exc.value.detail

    def test_value_error_maps_to_400(self):
        wrapped = self._make_wrapped(ValueError("invalid parameter"))
        with pytest.raises(HTTPException) as exc:
            wrapped()
        assert exc.value.status_code == 400
        assert "invalid parameter" in exc.value.detail

    def test_http_exception_passes_through(self):
        original = HTTPException(status_code=403, detail="forbidden")
        wrapped = self._make_wrapped(original)
        with pytest.raises(HTTPException) as exc:
            wrapped()
        assert exc.value is original
        assert exc.value.status_code == 403

    def test_other_exception_maps_to_502(self):
        wrapped = self._make_wrapped(RuntimeError("something broke"))
        with pytest.raises(UpstreamError) as exc:
            wrapped()
        assert exc.value.status_code == 502

    def test_no_exception_returns_normally(self):
        wrapped = self._make_wrapped()
        assert wrapped() == "ok"


class TestBatchDeleteInputValidation:
    def test_legacy_payload_rejects_empty_memories(self):
        with pytest.raises(ValidationError) as exc:
            MemoryBatchDeleteLegacyInput(memories=[])
        assert "must not be empty" in str(exc.value)

    def test_payload_rejects_empty_memory_ids(self):
        with pytest.raises(ValidationError) as exc:
            MemoryBatchDeleteInput(memory_ids=[])
        assert "must not be empty" in str(exc.value)


class TestV1ListMemories:
    def test_filtered_path_returns_bare_results_list(self, monkeypatch):
        mem = MagicMock()
        mem.get_all.return_value = [{"id": "m1"}]

        def _get_mem():
            return mem

        monkeypatch.setattr("server.routers.compat.get_memory_instance", _get_mem)

        result = v1_list_memories(request=_REQ, user_id="u1", auth=None)

        assert result == [{"id": "m1"}]
        mem.get_all.assert_called_once_with(filters={"user_id": "u1"})


class TestSyntheticEvents:
    @pytest.fixture(autouse=True)
    def _clear_events(self):
        event_cache_clear()
        yield
        event_cache_clear()

    @staticmethod
    def _run_background_tasks(tasks: BackgroundTasks) -> None:
        for task in tasks.tasks:
            task.func(*task.args, **task.kwargs)

    @staticmethod
    def _patch_memory(monkeypatch, get_mem) -> None:
        for target in (
            "server.routers.compat.get_memory_instance",
            "server.server_state.get_memory_instance",
            "server.memory_lock.get_memory_instance",
            "server.compat.tasks.get_memory_instance",
            "server.compat.tasks.run_memory_write",
            "server.routers.compat.run_memory_write",
            "memory_lock.get_memory_instance",
            "compat.tasks.get_memory_instance",
            "compat.tasks.run_memory_write",
            "routers.compat.run_memory_write",
        ):
            if target.endswith("run_memory_write"):
                monkeypatch.setattr(target, lambda fn, entity_scope=None, **_: fn(get_mem()), raising=False)
            else:
                monkeypatch.setattr(target, get_mem, raising=False)

    def test_v3_add_returns_event_id_and_event_is_fetchable(self, monkeypatch):
        mem = MagicMock()
        mem.add.return_value = {"results": [{"id": "m1", "memory": "saved"}]}

        def get_mem():
            return mem

        self._patch_memory(monkeypatch, get_mem)

        tasks = BackgroundTasks()

        result = v3_add_memory(
            MemoryAddInputV3(messages=[{"role": "user", "content": "remember"}], app_id="app1", infer=True),
            background_tasks=tasks,
            meta=RequestMeta(),
            request=_REQ,
            auth=None,
        )

        assert result["status"] == "PENDING"
        assert result["event_id"]

        pending_event = v1_get_event(result["event_id"], auth=None)
        assert pending_event["status"] == "PENDING"

        self._run_background_tasks(tasks)

        event = v1_get_event(result["event_id"], auth=None)
        assert event["id"] == result["event_id"]
        assert event["status"] == "SUCCEEDED"
        assert event["results"] == [{"id": "m1", "memory": "saved"}]

    def test_v1_events_paginates_cached_events(self, monkeypatch):
        mem = MagicMock()
        mem.add.return_value = {"results": [{"id": "m1", "memory": "saved"}]}

        def get_mem():
            return mem

        self._patch_memory(monkeypatch, get_mem)

        tasks = BackgroundTasks()

        v3_add_memory(
            MemoryAddInputV3(messages=[{"role": "user", "content": "first"}], app_id="app1", infer=True),
            background_tasks=tasks,
            meta=RequestMeta(),
            request=_REQ,
            auth=None,
        )
        v3_add_memory(
            MemoryAddInputV3(messages=[{"role": "user", "content": "second"}], app_id="app1", infer=True),
            background_tasks=tasks,
            meta=RequestMeta(),
            request=_REQ,
            auth=None,
        )
        self._run_background_tasks(tasks)

        req1 = MagicMock()
        req1.url.path = "/v1/events"
        req1.query_params = {"page": "1", "page_size": "1"}
        req2 = MagicMock()
        req2.url.path = "/v1/events"
        req2.query_params = {"page": "2", "page_size": "1"}

        first_page = v1_list_events(request=req1, page=1, page_size=1, auth=None)
        second_page = v1_list_events(request=req2, page=2, page_size=1, auth=None)

        assert first_page["count"] == 2
        assert len(first_page["results"]) == 1
        assert len(second_page["results"]) == 1
        assert first_page["next"] is not None
        assert second_page["previous"] is not None

    def test_v1_get_event_denied_for_other_user(self, monkeypatch):
        mem = MagicMock()
        mem.add.return_value = {"results": [{"id": "m1", "memory": "saved"}]}

        def get_mem():
            return mem

        self._patch_memory(monkeypatch, get_mem)

        tasks = BackgroundTasks()

        owner = MagicMock()
        owner.id = "user-1"
        other = MagicMock()
        other.id = "user-2"

        result = v3_add_memory(
            MemoryAddInputV3(messages=[{"role": "user", "content": "remember"}], app_id="app1", infer=True),
            background_tasks=tasks,
            meta=RequestMeta(),
            request=_REQ,
            auth=owner,
        )

        with pytest.raises(HTTPException) as exc:
            v1_get_event(result["event_id"], auth=other)
        assert exc.value.status_code == 404

    def test_v1_list_events_filters_by_owner(self, monkeypatch):
        mem = MagicMock()
        mem.add.return_value = {"results": [{"id": "m1", "memory": "saved"}]}

        def get_mem():
            return mem

        self._patch_memory(monkeypatch, get_mem)

        tasks = BackgroundTasks()

        user1 = MagicMock()
        user1.id = "user-1"
        user2 = MagicMock()
        user2.id = "user-2"

        v3_add_memory(
            MemoryAddInputV3(messages=[{"role": "user", "content": "u1"}], app_id="app1", infer=True),
            background_tasks=tasks,
            meta=RequestMeta(),
            request=_REQ,
            auth=user1,
        )
        v3_add_memory(
            MemoryAddInputV3(messages=[{"role": "user", "content": "u2"}], app_id="app1", infer=True),
            background_tasks=tasks,
            meta=RequestMeta(),
            request=_REQ,
            auth=user2,
        )
        self._run_background_tasks(tasks)

        req = MagicMock()
        req.url.path = "/v1/events"
        req.query_params = {"page": "1", "page_size": "10"}

        listed = v1_list_events(request=req, page=1, page_size=10, auth=user1)
        assert listed["count"] == 1
        assert len(listed["results"]) == 1
        assert listed["results"][0]["owner_id"] == "user-1"

    def test_v3_add_infer_false_returns_results_immediately(self, monkeypatch):
        mem = MagicMock()
        mem.add.return_value = {"results": [{"id": "m1", "memory": "verbatim"}]}

        def get_mem():
            return mem

        self._patch_memory(monkeypatch, get_mem)

        tasks = BackgroundTasks()
        result = v3_add_memory(
            MemoryAddInputV3(
                messages=[{"role": "user", "content": "remember"}],
                app_id="app1",
                infer=False,
            ),
            background_tasks=tasks,
            meta=RequestMeta(),
            request=_REQ,
            auth=None,
        )

        assert result == {
            "results": [{"id": "m1", "memory": "verbatim"}],
            "message": "Memory added successfully.",
            "event_id": None,
            "status": "SUCCEEDED",
        }
        assert tasks.tasks == []
        mem.add.assert_called_once()
        call_kwargs = mem.add.call_args.kwargs
        assert call_kwargs["infer"] is False

    def test_v3_add_infer_false_failure_surfaces_from_add(self, monkeypatch):
        mem = MagicMock()
        mem.add.side_effect = RuntimeError("boom")

        def get_mem():
            return mem

        self._patch_memory(monkeypatch, get_mem)

        with pytest.raises(UpstreamError):
            v3_add_memory(
                MemoryAddInputV3(
                    messages=[{"role": "user", "content": "remember"}],
                    app_id="app1",
                    infer=False,
                ),
                background_tasks=BackgroundTasks(),
                meta=RequestMeta(),
                request=_REQ,
                auth=None,
            )

    def test_v3_add_event_latency_is_recorded_in_milliseconds(self, monkeypatch):
        mem = MagicMock()
        mem.add.return_value = {"results": [{"id": "m1", "memory": "saved"}]}

        def get_mem():
            return mem

        self._patch_memory(monkeypatch, get_mem)
        monkeypatch.setattr(compat_tasks.time, "perf_counter", MagicMock(side_effect=[10.0, 10.25]))

        tasks = BackgroundTasks()
        result = v3_add_memory(
            MemoryAddInputV3(messages=[{"role": "user", "content": "remember"}], app_id="app1", infer=True),
            background_tasks=tasks,
            meta=RequestMeta(),
            request=_REQ,
            auth=None,
        )

        self._run_background_tasks(tasks)

        event = v1_get_event(result["event_id"], auth=None)
        assert event["latency"] == 250.0

    def test_v3_add_marks_event_failed_when_add_raises(self, monkeypatch):
        mem = MagicMock()
        mem.add.side_effect = RuntimeError("boom")

        def get_mem():
            return mem

        self._patch_memory(monkeypatch, get_mem)

        tasks = BackgroundTasks()
        result = v3_add_memory(
            MemoryAddInputV3(messages=[{"role": "user", "content": "remember"}], app_id="app1", infer=True),
            background_tasks=tasks,
            meta=RequestMeta(),
            request=_REQ,
            auth=None,
        )

        assert result["status"] == "PENDING"
        self._run_background_tasks(tasks)

        event = v1_get_event(result["event_id"], auth=None)
        assert event["status"] == "FAILED"
        assert event["metadata"] is not None
        assert "boom" in event["metadata"]["error"]

    def test_v3_add_forwards_expiration_date(self, monkeypatch):
        """expiration_date is forwarded to memory.add() when provided."""
        mem = MagicMock()
        mem.add.return_value = {"results": [{"id": "m1", "memory": "saved"}]}

        def get_mem():
            return mem

        self._patch_memory(monkeypatch, get_mem)

        tasks = BackgroundTasks()
        result = v3_add_memory(
            MemoryAddInputV3(
                messages=[{"role": "user", "content": "remember"}],
                app_id="app1",
                infer=False,
                expiration_date="2099-12-31",
            ),
            background_tasks=tasks,
            meta=RequestMeta(),
            request=_REQ,
            auth=None,
        )
        assert result["status"] == "SUCCEEDED"
        call_kwargs = mem.add.call_args.kwargs
        assert call_kwargs["expiration_date"] == date(2099, 12, 31)

    def test_v3_add_deduced_memories_rewrite_messages_when_infer_false(self, monkeypatch):
        """deduced_memories with infer=False: each fact becomes its own message."""
        mem = MagicMock()
        mem.add.return_value = {"results": [{"id": "m1"}, {"id": "m2"}]}

        def get_mem():
            return mem

        self._patch_memory(monkeypatch, get_mem)

        tasks = BackgroundTasks()
        v3_add_memory(
            MemoryAddInputV3(
                messages=[{"role": "user", "content": "joined blob"}],
                user_id="u1",
                infer=False,
                deduced_memories=["fact A", "fact B"],
            ),
            background_tasks=tasks,
            meta=RequestMeta(),
            request=_REQ,
            auth=None,
        )

        call_messages = mem.add.call_args.kwargs["messages"]
        assert call_messages == [
            {"role": "user", "content": "fact A"},
            {"role": "user", "content": "fact B"},
        ]
        # deduced_memories is NOT leaked into metadata
        assert "deduced_memories" not in mem.add.call_args.kwargs.get("metadata", {})

    def test_v3_add_deduced_memories_ignored_when_infer_true(self, monkeypatch):
        """deduced_memories with infer=True: original messages preserved (LLM extraction path)."""
        mem = MagicMock()
        mem.add.return_value = {"results": [{"id": "m1"}]}

        def get_mem():
            return mem

        self._patch_memory(monkeypatch, get_mem)

        tasks = BackgroundTasks()
        v3_add_memory(
            MemoryAddInputV3(
                messages=[{"role": "user", "content": "original"}],
                user_id="u1",
                infer=True,
                deduced_memories=["fact A"],
            ),
            background_tasks=tasks,
            meta=RequestMeta(),
            request=_REQ,
            auth=None,
        )
        self._run_background_tasks(tasks)

        call_messages = mem.add.call_args.kwargs["messages"]
        assert call_messages == [{"role": "user", "content": "original"}]
        assert "deduced_memories" not in (mem.add.call_args.kwargs.get("metadata") or {})

    def test_v3_add_timestamp_backdates_created_at(self, monkeypatch):
        """timestamp sets created_at in metadata (OSS _create_memory preserves existing created_at)."""
        mem = MagicMock()
        mem.add.return_value = {"results": [{"id": "m1", "memory": "saved"}]}

        def get_mem():
            return mem

        self._patch_memory(monkeypatch, get_mem)

        tasks = BackgroundTasks()
        v3_add_memory(
            MemoryAddInputV3(
                messages=[{"role": "user", "content": "remember"}],
                user_id="u1",
                infer=False,
                timestamp=1700000000,
            ),
            background_tasks=tasks,
            meta=RequestMeta(),
            request=_REQ,
            auth=None,
        )
        metadata = mem.add.call_args.kwargs["metadata"]
        assert metadata["created_at"].startswith("2023-11-14")
        # timestamp itself is NOT leaked into metadata
        assert "timestamp" not in metadata

    def test_v3_add_structured_data_schema_accepted_but_unsupported(self, monkeypatch, caplog):
        """structured_data_schema is accepted (no 422) but logged as unsupported."""
        mem = MagicMock()
        mem.add.return_value = {"results": [{"id": "m1", "memory": "saved"}]}

        def get_mem():
            return mem

        self._patch_memory(monkeypatch, get_mem)

        tasks = BackgroundTasks()
        with caplog.at_level(logging.WARNING):
            v3_add_memory(
                MemoryAddInputV3(
                    messages=[{"role": "user", "content": "remember"}],
                    user_id="u1",
                    infer=False,
                    structured_data_schema={"type": "object"},
                ),
                background_tasks=tasks,
                meta=RequestMeta(),
                request=_REQ,
                auth=None,
            )
        # schema accepted (no 422) and warning logged
        assert "structured_data_schema" in caplog.text
        assert "unsupported" in caplog.text.lower()
        # NOT persisted in metadata
        assert "structured_data_schema" not in (mem.add.call_args.kwargs.get("metadata") or {})

    def test_v3_add_empty_deduced_memories_preserves_original_messages(self, monkeypatch):
        """deduced_memories=[] does not rewrite messages."""
        mem = MagicMock()
        mem.add.return_value = {"results": [{"id": "m1", "memory": "saved"}]}

        def get_mem():
            return mem

        self._patch_memory(monkeypatch, get_mem)

        tasks = BackgroundTasks()
        v3_add_memory(
            MemoryAddInputV3(
                messages=[{"role": "user", "content": "original"}],
                user_id="u1",
                infer=False,
                deduced_memories=[],
            ),
            background_tasks=tasks,
            meta=RequestMeta(),
            request=_REQ,
            auth=None,
        )
        call_messages = mem.add.call_args.kwargs["messages"]
        assert call_messages == [{"role": "user", "content": "original"}]

    def test_v3_add_deduced_memories_rejects_non_string_entries(self):
        """List[str] schema rejects non-string entries with 422."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            MemoryAddInputV3(
                messages=[{"role": "user", "content": "x"}],
                user_id="u1",
                infer=False,
                deduced_memories=["valid", 123],
            )

    def test_v3_add_invalid_timestamp_returns_422(self, monkeypatch):
        """Out-of-range timestamp produces a 422, not a 500."""
        with pytest.raises(HTTPException) as exc_info:
            v3_add_memory(
                MemoryAddInputV3(
                    messages=[{"role": "user", "content": "remember"}],
                    user_id="u1",
                    infer=False,
                    timestamp=99999999999999999999,
                ),
                background_tasks=BackgroundTasks(),
                meta=RequestMeta(),
                request=_REQ,
                auth=None,
            )
        assert exc_info.value.status_code == 422

    def test_v3_add_timestamp_applied_for_async_path(self, monkeypatch):
        """timestamp backdates created_at even on the infer=True async path."""
        mem = MagicMock()
        mem.add.return_value = {"results": [{"id": "m1", "memory": "saved"}]}

        def get_mem():
            return mem

        self._patch_memory(monkeypatch, get_mem)

        tasks = BackgroundTasks()
        v3_add_memory(
            MemoryAddInputV3(
                messages=[{"role": "user", "content": "remember"}],
                user_id="u1",
                infer=True,
                timestamp=1700000000,
            ),
            background_tasks=tasks,
            meta=RequestMeta(),
            request=_REQ,
            auth=None,
        )
        self._run_background_tasks(tasks)
        metadata = mem.add.call_args.kwargs["metadata"]
        assert metadata["created_at"].startswith("2023-11-14")

    def test_v3_add_passes_agent_custom_instructions_as_prompt(self, monkeypatch):
        """agent_custom_instructions reaches SDK as prompt= when agent-scoped and infer=True."""
        mem = MagicMock()
        mem.add.return_value = {"results": [{"id": "m1", "memory": "saved"}]}

        def get_mem():
            return mem

        self._patch_memory(monkeypatch, get_mem)

        tasks = BackgroundTasks()
        result = v3_add_memory(
            MemoryAddInputV3(
                messages=[{"role": "user", "content": "remember"}],
                agent_id="agent1",
                infer=True,
                agent_custom_instructions="extract tool failures",
            ),
            background_tasks=tasks,
            meta=RequestMeta(),
            request=_REQ,
            auth=None,
        )
        assert result["status"] == "PENDING"
        self._run_background_tasks(tasks)

        call_kwargs = mem.add.call_args.kwargs
        assert call_kwargs["prompt"] == "extract tool failures"
        # instruction fields are NOT leaked into metadata
        assert "custom_instructions" not in call_kwargs.get("metadata", {})
        assert "agent_custom_instructions" not in call_kwargs.get("metadata", {})

    def test_v3_add_passes_custom_instructions_with_constraints_as_prompt(self, monkeypatch):
        """custom_instructions + includes/excludes are merged into a single prompt."""
        mem = MagicMock()
        mem.add.return_value = {"results": [{"id": "m1", "memory": "saved"}]}

        def get_mem():
            return mem

        self._patch_memory(monkeypatch, get_mem)

        tasks = BackgroundTasks()
        v3_add_memory(
            MemoryAddInputV3(
                messages=[{"role": "user", "content": "remember"}],
                user_id="u1",
                infer=True,
                custom_instructions="base instructions",
                includes="vehicles",
                excludes="politics",
            ),
            background_tasks=tasks,
            meta=RequestMeta(),
            request=_REQ,
            auth=None,
        )
        self._run_background_tasks(tasks)

        prompt = mem.add.call_args.kwargs["prompt"]
        assert "base instructions" in prompt
        assert "Include only: vehicles" in prompt
        assert "Exclude: politics" in prompt

    def test_v3_add_infer_false_omits_prompt(self, monkeypatch):
        """infer=False skips extraction, so no prompt is forwarded."""
        mem = MagicMock()
        mem.add.return_value = {"results": [{"id": "m1", "memory": "saved"}]}

        def get_mem():
            return mem

        self._patch_memory(monkeypatch, get_mem)

        tasks = BackgroundTasks()
        result = v3_add_memory(
            MemoryAddInputV3(
                messages=[{"role": "user", "content": "remember"}],
                user_id="u1",
                infer=False,
                custom_instructions="base instructions",
                includes="vehicles",
            ),
            background_tasks=tasks,
            meta=RequestMeta(),
            request=_REQ,
            auth=None,
        )
        assert result["status"] == "SUCCEEDED"
        assert "prompt" not in mem.add.call_args.kwargs

    def test_v3_add_schema_accepts_all_new_fields_without_422(self, monkeypatch):
        """agent_custom_instructions, includes, excludes are all accepted by MemoryAddInputV3 (extra='forbid')."""
        mem = MagicMock()
        mem.add.return_value = {"results": [{"id": "m1", "memory": "saved"}]}

        def get_mem():
            return mem

        self._patch_memory(monkeypatch, get_mem)

        tasks = BackgroundTasks()
        result = v3_add_memory(
            MemoryAddInputV3(
                messages=[{"role": "user", "content": "remember"}],
                agent_id="agent1",
                infer=False,
                custom_instructions="base",
                agent_custom_instructions="tool failures",
                includes="vehicles",
                excludes="politics",
            ),
            background_tasks=tasks,
            meta=RequestMeta(),
            request=_REQ,
            auth=None,
        )
        assert result["status"] == "SUCCEEDED"


# ---------------------------------------------------------------------------
# Model field validation — new / updated fields
# ---------------------------------------------------------------------------


class TestMemoryUpdateInputExpirationDate:
    def test_accepts_expiration_date(self):
        body = MemoryUpdateInput(text="hello", expiration_date="2099-12-31")
        assert body.expiration_date == date(2099, 12, 31)

    def test_expiration_date_none_by_default(self):
        body = MemoryUpdateInput(text="hello")
        assert body.expiration_date is None

    def test_expiration_date_omitted_not_in_fields_set(self):
        """When omitted, model_fields_set does NOT contain expiration_date."""
        body = MemoryUpdateInput(text="hello")
        assert "expiration_date" not in body.model_fields_set

    def test_expiration_date_explicit_null_in_fields_set(self):
        """When explicitly set to None, model_fields_set DOES contain it."""
        body = MemoryUpdateInput(text="hello", expiration_date=None)
        assert body.expiration_date is None
        assert "expiration_date" in body.model_fields_set

    def test_explicit_null_only_clears_expiration_date(self, monkeypatch):
        mem = MagicMock()
        mem.update.return_value = {"message": "updated"}

        def _run_memory_write_for_memory_id(callback, memory_id):
            assert memory_id == "mem-1"
            return callback(mem)

        monkeypatch.setattr(
            "server.routers.compat.run_memory_write_for_memory_id",
            _run_memory_write_for_memory_id,
        )

        result = v1_update_memory("mem-1", MemoryUpdateInput(expiration_date=None), request=_REQ, auth=None)

        assert result == {"message": "updated"}
        # Only expiration_date is forwarded; text/metadata are omitted (None) so
        # the SDK preserves the existing values. expiration_date=None means "clear".
        mem.update.assert_called_once_with(memory_id="mem-1", expiration_date=None)

    def test_expiration_date_set_is_forwarded(self, monkeypatch):
        mem = MagicMock()
        mem.update.return_value = {"message": "updated"}

        def _run_memory_write_for_memory_id(callback, memory_id):
            return callback(mem)

        monkeypatch.setattr(
            "server.routers.compat.run_memory_write_for_memory_id",
            _run_memory_write_for_memory_id,
        )
        v1_update_memory("mem-1", MemoryUpdateInput(text="new", expiration_date="2099-12-31"), request=_REQ, auth=None)

        mem.update.assert_called_once_with(memory_id="mem-1", data="new", expiration_date=date(2099, 12, 31))

    def test_rejects_unknown_field(self):
        with pytest.raises(ValidationError):
            MemoryUpdateInput(unknown_field="bad")


class TestMemoryGetInputV2Fields:
    def test_accepts_fields(self):
        body = MemoryGetInputV2(filters={"user_id": "u1"}, fields=["id", "memory"])
        assert body.fields == ["id", "memory"]

    def test_accepts_latest_only(self):
        body = MemoryGetInputV2(filters={"user_id": "u1"}, latest_only=True)
        assert body.latest_only is True

    def test_accepts_show_expired(self):
        body = MemoryGetInputV2(filters={"user_id": "u1"}, show_expired=True)
        assert body.show_expired is True

    def test_rejects_unknown_field(self):
        with pytest.raises(ValidationError):
            MemoryGetInputV2(unknown_field="bad")


class TestMemoryGetInputV3Fields:
    def test_accepts_all_defined_fields(self):
        body = MemoryGetInputV3(
            filters={"user_id": "u1"},
            start_date="2024-01-01",
            end_date="2024-12-31",
            categories=["finance"],
            show_expired=True,
            latest_only=True,
        )
        assert body.filters == {"user_id": "u1"}
        assert body.start_date == "2024-01-01"
        assert body.end_date == "2024-12-31"
        assert body.categories == ["finance"]
        assert body.show_expired is True
        assert body.latest_only is True

    def test_rejects_unknown_field(self):
        with pytest.raises(ValidationError):
            MemoryGetInputV3(unknown_field="bad")

    def test_fields_not_in_v3(self):
        """MemoryGetInputV3 must NOT have 'fields' — that's v2 only."""
        with pytest.raises(ValidationError):
            MemoryGetInputV3(fields=["id"])


class TestMemorySearchInputLatestOnly:
    def test_accepts_latest_only(self):
        body = MemorySearchInput(query="hello", latest_only=True)
        assert body.latest_only is True

    def test_rejects_unknown_field(self):
        with pytest.raises(ValidationError):
            MemorySearchInput(unknown_field="bad")


class TestMemorySearchInputV2LatestOnly:
    def test_accepts_latest_only(self):
        body = MemorySearchInputV2(query="hello", filters={"user_id": "u1"}, latest_only=True)
        assert body.latest_only is True

    def test_rejects_unknown_field(self):
        with pytest.raises(ValidationError):
            MemorySearchInputV2(unknown_field="bad")


class TestMemorySearchInputV3LatestOnly:
    def test_accepts_latest_only_and_reference_date(self):
        body = MemorySearchInputV3(
            query="hello",
            filters={"user_id": "u1"},
            latest_only=True,
            reference_date="2024-06-01",
        )
        assert body.latest_only is True
        assert body.reference_date == "2024-06-01"

    def test_rejects_unknown_field(self):
        with pytest.raises(ValidationError):
            MemorySearchInputV3(unknown_field="bad")


# ---------------------------------------------------------------------------
# build_search_kwargs — show_expired passthrough
# ---------------------------------------------------------------------------


class TestBuildSearchKwargsShowExpired:
    def test_show_expired_false_passed_through(self):
        kwargs = build_search_kwargs({"user_id": "u1"}, top_k=5, threshold=0.5, show_expired=False)
        assert kwargs["show_expired"] is False

    def test_show_expired_true_passed_through(self):
        kwargs = build_search_kwargs({"user_id": "u1"}, top_k=None, threshold=None, show_expired=True)
        assert kwargs["show_expired"] is True

    def test_show_expired_none_not_included(self):
        kwargs = build_search_kwargs({"user_id": "u1"}, top_k=5, threshold=None, show_expired=None)
        assert "show_expired" not in kwargs


# ---------------------------------------------------------------------------
# v1_list_memories — show_expired query param passthrough
# ---------------------------------------------------------------------------


class TestV1ListMemoriesShowExpired:
    def test_show_expired_passed_to_get_all(self, monkeypatch):
        mem = MagicMock()
        mem.get_all.return_value = [{"id": "m1"}]

        def _get_mem():
            return mem

        monkeypatch.setattr("server.routers.compat.get_memory_instance", _get_mem)

        v1_list_memories(request=_REQ, user_id="u1", show_expired=True, auth=None)

        mem.get_all.assert_called_once_with(filters={"user_id": "u1"}, show_expired=True)

    def test_show_expired_false_passed_through(self, monkeypatch):
        mem = MagicMock()
        mem.get_all.return_value = [{"id": "m1"}]

        def _get_mem():
            return mem

        monkeypatch.setattr("server.routers.compat.get_memory_instance", _get_mem)

        v1_list_memories(request=_REQ, user_id="u1", show_expired=False, auth=None)

        mem.get_all.assert_called_once_with(filters={"user_id": "u1"}, show_expired=False)

    def test_show_expired_true_forwarded_to_get_all(self, monkeypatch):
        mem = MagicMock()
        mem.get_all.return_value = [{"id": "m1"}]
        monkeypatch.setattr("server.routers.compat.get_memory_instance", lambda: mem)

        v1_list_memories(request=_REQ, show_expired=True, auth=None)

        _, kwargs = mem.get_all.call_args
        assert kwargs.get("show_expired") is True

    def test_show_expired_omitted_not_forwarded_to_get_all(self, monkeypatch):
        mem = MagicMock()
        mem.get_all.return_value = [{"id": "m1"}]
        monkeypatch.setattr("server.routers.compat.get_memory_instance", lambda: mem)

        v1_list_memories(request=_REQ, auth=None)

        _, kwargs = mem.get_all.call_args
        assert "show_expired" not in kwargs


# ---------------------------------------------------------------------------
# v3_get_all_memories — show_expired passthrough
# ---------------------------------------------------------------------------


class TestV3GetAllMemoriesShowExpired:
    def test_show_expired_passed_to_get_all(self, monkeypatch):
        mem = MagicMock()
        mem.count.return_value = 0
        mem.get_all.return_value = [{"id": "m1"}]

        def _get_mem():
            return mem

        monkeypatch.setattr("server.compat.helpers.get_memory_instance", _get_mem)

        req = MagicMock()
        req.url.path = "/v3/memories"
        req.query_params = {"page": "1", "page_size": "10"}
        body = MemoryGetInputV3(filters={"user_id": "u1"}, show_expired=True)

        v3_get_all_memories(request=req, body=body, page=1, page_size=10, auth=None)

        mem.get_all.assert_called_once_with(filters={"user_id": "u1"}, show_expired=True, top_k=11, skip=0)


# ---------------------------------------------------------------------------
# v3_search_memories — show_expired passthrough
# ---------------------------------------------------------------------------


class TestV3SearchMemoriesShowExpired:
    def test_show_expired_passed_to_search(self, monkeypatch):
        mem = MagicMock()
        mem.search.return_value = {"results": []}

        def _get_mem():
            return mem

        monkeypatch.setattr("server.routers.compat.get_memory_instance", _get_mem)

        body = MemorySearchInputV3(query="hello", user_id="u1", show_expired=True)
        v3_search_memories(body, request=_REQ, auth=None)

        assert mem.search.call_args.kwargs["show_expired"] is True

    def test_show_expired_defaults_to_none(self, monkeypatch):
        mem = MagicMock()
        mem.search.return_value = {"results": []}

        def _get_mem():
            return mem

        monkeypatch.setattr("server.routers.compat.get_memory_instance", _get_mem)

        body = MemorySearchInputV3(query="hello", user_id="u1")
        v3_search_memories(body, request=_REQ, auth=None)

        assert "show_expired" not in mem.search.call_args.kwargs

    def test_v3_search_applies_fields(self, monkeypatch):
        mem = MagicMock()
        mem.search.return_value = [{"id": "m1", "memory": "a", "score": 0.9}]

        def _get_mem():
            return mem

        monkeypatch.setattr("server.routers.compat.get_memory_instance", _get_mem)

        body = MemorySearchInputV3(query="hello", user_id="u1", fields=["id", "memory"])
        response = v3_search_memories(body, request=_REQ, auth=None)

        assert response == {"results": [{"id": "m1", "memory": "a"}]}

    def test_v3_search_applies_fields_v1_0_output(self, monkeypatch):
        mem = MagicMock()
        mem.search.return_value = [{"id": "m1", "memory": "a", "score": 0.9}]

        def _get_mem():
            return mem

        monkeypatch.setattr("server.routers.compat.get_memory_instance", _get_mem)

        body = MemorySearchInputV3(query="hello", user_id="u1", fields=["id"], output_format="v1.0")
        result = v3_search_memories(body, request=_REQ, auth=None)

        assert result == [{"id": "m1"}]


# ---------------------------------------------------------------------------
# v2_list_memories — show_expired passthrough
# ---------------------------------------------------------------------------


class TestV2ListMemoriesShowExpired:
    def test_show_expired_passed_to_get_all(self, monkeypatch):
        mem = MagicMock()
        mem.count.return_value = 0
        mem.get_all.return_value = [{"id": "m1"}]

        def _get_mem():
            return mem

        monkeypatch.setattr("server.compat.helpers.get_memory_instance", _get_mem)

        req = MagicMock()
        req.url.path = "/v2/memories"
        req.query_params = {"page": "1", "page_size": "10"}
        body = MemoryGetInputV2(filters={"user_id": "u1"}, show_expired=True)

        v2_list_memories(request=req, body=body, page=1, page_size=10, auth=None)

        mem.get_all.assert_called_once_with(filters={"user_id": "u1"}, show_expired=True, top_k=11, skip=0)


# ---------------------------------------------------------------------------
# v2_search_memories — show_expired passthrough
# ---------------------------------------------------------------------------


class TestV2SearchMemoriesShowExpired:
    def test_show_expired_passed_to_search(self, monkeypatch):
        mem = MagicMock()
        mem.search.return_value = {"results": []}

        def _get_mem():
            return mem

        monkeypatch.setattr("server.routers.compat.get_memory_instance", _get_mem)

        body = MemorySearchInputV2(query="hello", user_id="u1", show_expired=True)

        v2_search_memories(body, request=_REQ, auth=None)

        assert mem.search.call_args.kwargs["show_expired"] is True

    def test_v2_search_applies_fields(self, monkeypatch):
        mem = MagicMock()
        mem.search.return_value = {"results": [{"id": "m1", "memory": "a", "score": 0.9}]}

        def _get_mem():
            return mem

        monkeypatch.setattr("server.routers.compat.get_memory_instance", _get_mem)

        body = MemorySearchInputV2(query="hello", user_id="u1", fields=["id", "memory"])
        response = v2_search_memories(body, request=_REQ, auth=None)

        assert "results" in response
        assert response["results"] == [{"id": "m1", "memory": "a"}]


# ---------------------------------------------------------------------------
# v1_search_memories — show_expired passthrough
# ---------------------------------------------------------------------------


class TestV1SearchMemoriesShowExpired:
    def test_show_expired_passed_to_search(self, monkeypatch):
        mem = MagicMock()
        mem.search.return_value = {"results": []}

        def _get_mem():
            return mem

        monkeypatch.setattr("server.routers.compat.get_memory_instance", _get_mem)

        body = MemorySearchInput(query="hello", user_id="u1", show_expired=True)

        v1_search_memories(body, request=_REQ, auth=None)

        assert mem.search.call_args.kwargs["show_expired"] is True

    def test_v1_search_applies_fields(self, monkeypatch):
        mem = MagicMock()
        mem.search.return_value = [{"id": "m1", "memory": "a", "score": 0.9}]

        def _get_mem():
            return mem

        monkeypatch.setattr("server.routers.compat.get_memory_instance", _get_mem)

        body = MemorySearchInput(query="hello", user_id="u1", fields=["id", "memory"])

        result = v1_search_memories(body, request=_REQ, auth=None)

        assert result == [{"id": "m1", "memory": "a"}]


# ---------------------------------------------------------------------------
# v1_get_entity_memories — show_expired passthrough
# ---------------------------------------------------------------------------


class TestV1GetEntityMemoriesShowExpired:
    def test_show_expired_passed_to_get_all(self, monkeypatch):
        mem = MagicMock()
        mem.get_all.return_value = [{"id": "m1"}]

        def _get_mem():
            return mem

        monkeypatch.setattr("server.routers.compat.get_memory_instance", _get_mem)

        v1_get_entity_memories(entity_type="user", entity_id="alice", request=_REQ, show_expired=True, auth=None)

        mem.get_all.assert_called_once_with(filters={"user_id": "alice"}, show_expired=True)
