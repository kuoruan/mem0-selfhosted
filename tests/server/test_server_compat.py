"""Unit tests for the server compat utilities.

Covers:
  - compat.scope: collect_direct_entity_params, require_entity_scope,
                    build_search_filters, get_entity_field
  - compat.utils: drop_none
  - compat.helpers: normalize_results, normalize_results_dict
  - compat.decorators: upstream_guard exception mapping
  - routers.compat helpers: build_list_filters, paginate_response,
                            warn_unsupported_fields, build_search_kwargs,
                            resolve_existing, merge_and_update
"""

import logging
from datetime import datetime, timezone
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
from server.compat.metadata import merge_v1_add_metadata, merge_v3_add_metadata
from server.compat.utils import drop_none, parse_iso_timestamp
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
    MemoryAddInputV3,
    MemoryGetInputV2,
    MemoryGetInputV3,
    MemorySearchInput,
    MemorySearchInputV2,
    MemorySearchInputV3,
    MemoryUpdateInput,
    build_list_filters,
    build_search_kwargs,
    merge_and_update,
    paginate_response,
    resolve_existing,
    warn_unsupported_fields,
    v1_get_event,
    v1_list_entities,
    v1_list_events,
    v1_list_memories,
    v1_update_memory,
    v2_list_memories,
    v3_add_memory,
    v3_search_memories,
)


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
            total_memories=3,
            created_at=created,
            updated_at=updated,
        )
        assert entity.type == "user"
        assert entity.name == "alice"
        assert entity.total_memories == 3
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
        assert entities[0].total_memories == 1

    def test_aggregate_entity_buckets_handles_mixed_timezone_formats(self):
        payloads = [
            {"user_id": "alice", "created_at": "2026-01-02T00:00:00+00:00", "updated_at": "2026-01-03T00:00:00+00:00"},
            {"user_id": "alice", "created_at": "2026-01-01T00:00:00", "updated_at": "2026-01-04T00:00:00"},
        ]

        buckets = compat_entities.aggregate_entity_buckets(payloads, {"user": "user_id"})
        bucket = buckets[("user", "alice")]
        assert bucket["total_memories"] == 2
        assert bucket["created_at"] is not None
        assert bucket["updated_at"] is not None
        assert bucket["created_at"].tzinfo is not None
        assert bucket["updated_at"].tzinfo is not None

    def test_iter_payloads_uses_store_count_when_available(self, monkeypatch):
        row = MagicMock(payload={"user_id": "alice"})
        mem = MagicMock()
        mem.vector_store.col_info.return_value = {"count": 15_000}
        mem.vector_store.list.return_value = [row]

        monkeypatch.setattr(compat_entities, "get_memory_instance", lambda: mem)

        payloads = compat_entities.iter_payloads()
        assert payloads == [{"user_id": "alice"}]
        mem.vector_store.list.assert_called_once_with(top_k=15_000)

    def test_iter_payloads_uses_object_store_count_when_available(self, monkeypatch):
        row = MagicMock(payload={"user_id": "alice"})
        mem = MagicMock()
        mem.vector_store.col_info.return_value = MagicMock(points_count=12_000)
        mem.vector_store.list.return_value = [row]

        monkeypatch.setattr(compat_entities, "get_memory_instance", lambda: mem)

        payloads = compat_entities.iter_payloads()
        assert payloads == [{"user_id": "alice"}]
        mem.vector_store.list.assert_called_once_with(top_k=12_000)

    def test_iter_payloads_warns_when_backend_returns_paged_tuple(self, monkeypatch, caplog):
        row = MagicMock(payload={"user_id": "alice"})
        mem = MagicMock()
        mem.vector_store.col_info.return_value = {"count": 5}
        mem.vector_store.list.return_value = ([row], "next-offset")

        monkeypatch.setattr(compat_entities, "get_memory_instance", lambda: mem)

        with caplog.at_level(logging.WARNING, logger="mem0.server.compat.entities"):
            payloads = compat_entities.iter_payloads(limit=5)

        assert payloads == [{"user_id": "alice"}]
        assert "paged result while building entities" in caplog.text

    def test_iter_payloads_retries_with_larger_top_k_when_count_missing(self, monkeypatch):
        row1 = MagicMock(payload={"user_id": "alice"})
        row2 = MagicMock(payload={"user_id": "bob"})
        mem = MagicMock()
        mem.vector_store.col_info.return_value = {}
        mem.vector_store.list.side_effect = [([row1], "next-offset"), [row1, row2]]

        monkeypatch.setattr(compat_entities, "get_memory_instance", lambda: mem)

        payloads = compat_entities.iter_payloads(limit=2)
        assert payloads == [{"user_id": "alice"}, {"user_id": "bob"}]
        assert [call.kwargs["top_k"] for call in mem.vector_store.list.call_args_list] == [2, 4]

    def test_v1_list_entities_returns_paginated_entities_with_mixed_timestamps(self, monkeypatch):
        row1 = MagicMock(
            payload={
                "user_id": "alice",
                "created_at": "2026-01-01T00:00:00",
                "updated_at": "2026-01-01T01:00:00+00:00",
            }
        )
        row2 = MagicMock(
            payload={
                "user_id": "alice",
                "created_at": "2026-01-02T00:00:00+00:00",
                "updated_at": "2026-01-03T00:00:00",
            }
        )
        mem = MagicMock()
        mem.vector_store.col_info.return_value = {"count": 2}
        mem.vector_store.list.return_value = [row1, row2]

        monkeypatch.setattr("server.routers.compat.get_memory_instance", lambda: mem)
        monkeypatch.setattr("server.compat.entities.get_memory_instance", lambda: mem)

        req = MagicMock()
        req.url = URL("http://test/v1/entities?page=1&page_size=10")

        result = v1_list_entities(request=req, page=1, page_size=10, _auth=None)

        assert result["count"] == 1
        assert len(result["results"]) == 1
        entity = result["results"][0]
        assert entity.id == "alice"
        assert entity.total_memories == 2

    def test_v1_list_entities_respects_pagination(self, monkeypatch):
        rows = [
            MagicMock(payload={"user_id": "alice", "created_at": "2026-01-01T00:00:00+00:00"}),
            MagicMock(payload={"agent_id": "agent-1", "created_at": "2026-01-01T00:00:00+00:00"}),
        ]
        mem = MagicMock()
        mem.vector_store.col_info.return_value = {"count": 2}
        mem.vector_store.list.return_value = rows

        monkeypatch.setattr("server.routers.compat.get_memory_instance", lambda: mem)
        monkeypatch.setattr("server.compat.entities.get_memory_instance", lambda: mem)

        req = MagicMock()
        req.url = URL("http://test/v1/entities?page=1&page_size=1")

        page = v1_list_entities(request=req, page=1, page_size=1, _auth=None)

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
        result = collect_direct_entity_params(filters={"OR": [{"app_id": "app1"}, {"app_id": "app1", "agent_id": "a1"}]})
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

    def test_update_preserves_owner_user_id(self):
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
                "owner_user_id": "user-1",
            },
        )

        updated = event_cache_update("evt-1", status="SUCCEEDED", owner_user_id="user-2")
        assert updated is not None
        assert updated["owner_user_id"] == "user-1"

    def test_update_rejects_invalid_status(self):
        event_cache_put("evt-1", CompatEvent.pending("evt-1", owner_user_id="user-1"))
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
# warn_unsupported_fields
# ---------------------------------------------------------------------------


class TestWarnUnsupportedFields:
    def test_no_fields_no_warning(self, caplog):
        with caplog.at_level(logging.WARNING):
            warn_unsupported_fields(None, "v3_search_memories")
        assert "fields" not in caplog.text

    def test_empty_list_no_warning(self, caplog):
        with caplog.at_level(logging.WARNING):
            warn_unsupported_fields([], "v3_search_memories")
        assert "fields" not in caplog.text

    def test_fields_emits_warning(self, caplog):
        with caplog.at_level(logging.WARNING):
            warn_unsupported_fields(["id", "memory"], "v2_search_memories")
        assert "v2_search_memories" in caplog.text
        assert "fields" in caplog.text.lower()

    def test_warning_includes_field_names(self, caplog):
        with caplog.at_level(logging.WARNING):
            warn_unsupported_fields(["score"], "v3_search_memories")
        assert "score" in caplog.text


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

    def test_v2_list_warns_for_fields_and_latest_only(self, monkeypatch, caplog):
        mem = MagicMock()
        mem.get_all.return_value = []
        monkeypatch.setattr("server.routers.compat.get_memory_instance", lambda: mem)

        req = MagicMock()
        req.url = URL("http://testserver/v2/memories?page=1&page_size=10")
        body = MemoryGetInputV2(filters={"user_id": "u1"}, fields=["id"], latest_only=True)

        with caplog.at_level(logging.WARNING):
            v2_list_memories(req, body, page=1, page_size=10, _auth=None)

        assert "v2_list_memories" in caplog.text
        assert "fields" in caplog.text.lower()
        assert "latest_only" in caplog.text
        mem.get_all.assert_called_once_with(filters={"user_id": "u1"})

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
            v3_search_memories(body, _auth=None)

        assert "v3_search_memories" in caplog.text
        assert "reference_date" in caplog.text
        assert "latest_only" in caplog.text
        mem.search.assert_called_once_with(query="hello", filters={"user_id": "u1"})


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

        v3_search_memories(body, _auth=None)

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
# merge_and_update
# ---------------------------------------------------------------------------


class TestMergeAndUpdate:
    def test_new_text_overwrites_existing(self):
        mem = MagicMock()
        mem.get.return_value = {"id": "mem-1", "memory": "old text", "metadata": {}}
        mem.update.return_value = {"message": "updated"}
        merge_and_update(mem, "mem-1", text="new text")
        mem.update.assert_called_once_with(memory_id="mem-1", data="new text", metadata={})

    def test_preserves_existing_text_when_none(self):
        mem = MagicMock()
        mem.get.return_value = {"id": "mem-1", "memory": "old text", "metadata": {"key": "val"}}
        mem.update.return_value = {"message": "updated"}
        merge_and_update(mem, "mem-1", text=None)
        mem.update.assert_called_once_with(
            memory_id="mem-1", data="old text", metadata={"key": "val"}
        )

    def test_preserves_existing_text_via_text_key(self):
        """Some SDK responses use 'text' instead of 'memory' for the content field."""
        mem = MagicMock()
        mem.get.return_value = {"id": "mem-1", "text": "via text key", "metadata": {}}
        mem.update.return_value = {"message": "updated"}
        merge_and_update(mem, "mem-1")
        mem.update.assert_called_once_with(memory_id="mem-1", data="via text key", metadata={})

    def test_metadata_new_keys_override_existing(self):
        mem = MagicMock()
        mem.get.return_value = {"id": "mem-1", "memory": "txt", "metadata": {"a": 1, "b": 2}}
        mem.update.return_value = {"message": "updated"}
        merge_and_update(mem, "mem-1", metadata={"b": 99, "c": 3})
        mem.update.assert_called_once_with(
            memory_id="mem-1", data="txt", metadata={"a": 1, "b": 99, "c": 3}
        )

    def test_metadata_none_keeps_existing(self):
        mem = MagicMock()
        mem.get.return_value = {"id": "mem-1", "memory": "txt", "metadata": {"x": 1}}
        mem.update.return_value = {"message": "updated"}
        merge_and_update(mem, "mem-1", metadata=None)
        mem.update.assert_called_once_with(memory_id="mem-1", data="txt", metadata={"x": 1})

    def test_raises_404_when_memory_missing(self):
        """Delegates to resolve_existing which raises 404 for missing memory."""
        mem = MagicMock()
        mem.get.return_value = None
        with pytest.raises(HTTPException) as exc:
            merge_and_update(mem, "nonexistent", text="new")
        assert exc.value.status_code == 404

    def test_handles_missing_metadata_on_existing(self):
        mem = MagicMock()
        mem.get.return_value = {"id": "mem-1", "memory": "txt"}
        mem.update.return_value = {"message": "updated"}
        merge_and_update(mem, "mem-1", metadata={"new_key": "val"})
        mem.update.assert_called_once_with(
            memory_id="mem-1", data="txt", metadata={"new_key": "val"}
        )


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

        result = v1_list_memories(request=MagicMock(), user_id="u1", auth=None)

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
            auth=None,
        )
        v3_add_memory(
            MemoryAddInputV3(messages=[{"role": "user", "content": "second"}], app_id="app1", infer=True),
            background_tasks=tasks,
            meta=RequestMeta(),
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
            auth=user1,
        )
        v3_add_memory(
            MemoryAddInputV3(messages=[{"role": "user", "content": "u2"}], app_id="app1", infer=True),
            background_tasks=tasks,
            meta=RequestMeta(),
            auth=user2,
        )
        self._run_background_tasks(tasks)

        req = MagicMock()
        req.url.path = "/v1/events"
        req.query_params = {"page": "1", "page_size": "10"}

        listed = v1_list_events(request=req, page=1, page_size=10, auth=user1)
        assert listed["count"] == 1
        assert len(listed["results"]) == 1
        assert listed["results"][0]["owner_user_id"] == "user-1"

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
            auth=None,
        )

        assert result["status"] == "SUCCEEDED"
        call_kwargs = mem.add.call_args.kwargs
        assert call_kwargs["expiration_date"] == "2099-12-31"


# ---------------------------------------------------------------------------
# Model field validation — new / updated fields
# ---------------------------------------------------------------------------


class TestMemoryUpdateInputExpirationDate:
    def test_accepts_expiration_date(self):
        body = MemoryUpdateInput(text="hello", expiration_date="2099-12-31")
        assert body.expiration_date == "2099-12-31"

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
        mem.get.return_value = {"id": "mem-1", "memory": "txt", "metadata": {"kind": "note"}}
        mem.update.return_value = {"message": "updated"}

        def _run_memory_write_for_memory_id(callback, memory_id):
            assert memory_id == "mem-1"
            return callback(mem)

        monkeypatch.setattr(
            "server.routers.compat.run_memory_write_for_memory_id",
            _run_memory_write_for_memory_id,
        )

        result = v1_update_memory("mem-1", MemoryUpdateInput(expiration_date=None), _auth=None)

        assert result == {"message": "updated"}
        mem.update.assert_called_once_with(
            memory_id="mem-1",
            data="txt",
            metadata={"kind": "note"},
            expiration_date=None,
        )

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
# merge_and_update — expiration_date passthrough
# ---------------------------------------------------------------------------


class TestMergeAndUpdateExpirationDate:
    def test_expiration_date_passed_to_mem_update(self):
        mem = MagicMock()
        mem.get.return_value = {"id": "mem-1", "memory": "txt", "metadata": {}}
        mem.update.return_value = {"message": "updated"}
        merge_and_update(mem, "mem-1", expiration_date="2099-12-31")
        mem.update.assert_called_once_with(
            memory_id="mem-1", data="txt", metadata={}, expiration_date="2099-12-31"
        )

    def test_expiration_date_none_passed_to_clear(self):
        """When expiration_date is explicitly None, it MUST be passed to mem.update().
        The SDK interprets None as "clear the expiration date"."""
        mem = MagicMock()
        mem.get.return_value = {"id": "mem-1", "memory": "txt", "metadata": {}}
        mem.update.return_value = {"message": "updated"}
        merge_and_update(mem, "mem-1", expiration_date=None)
        mem.update.assert_called_once_with(
            memory_id="mem-1", data="txt", metadata={}, expiration_date=None
        )

    def test_expiration_date_omitted_preserves_existing(self):
        """When expiration_date is not passed (defaults to _UNSET), it is NOT forwarded."""
        mem = MagicMock()
        mem.get.return_value = {"id": "mem-1", "memory": "txt", "metadata": {}}
        mem.update.return_value = {"message": "updated"}
        merge_and_update(mem, "mem-1")
        mem.update.assert_called_once_with(
            memory_id="mem-1", data="txt", metadata={}
        )


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

        v1_list_memories(request=MagicMock(), user_id="u1", show_expired=True, auth=None)

        mem.get_all.assert_called_once_with(filters={"user_id": "u1"}, show_expired=True)

    def test_show_expired_false_passed_through(self, monkeypatch):
        mem = MagicMock()
        mem.get_all.return_value = [{"id": "m1"}]

        def _get_mem():
            return mem

        monkeypatch.setattr("server.routers.compat.get_memory_instance", _get_mem)

        v1_list_memories(request=MagicMock(), user_id="u1", show_expired=False, auth=None)

        mem.get_all.assert_called_once_with(filters={"user_id": "u1"}, show_expired=False)

    def test_admin_path_passes_show_expired_to_list_all_memories(self, monkeypatch):
        list_all = MagicMock(return_value={"results": [{"id": "m1"}]})

        monkeypatch.setattr("server.routers.compat.ensure_admin", lambda request, auth: None)
        monkeypatch.setattr("server.routers.compat.list_all_memories", list_all)

        result = v1_list_memories(request=MagicMock(), show_expired=True, auth=MagicMock())

        assert result == [{"id": "m1"}]
        list_all.assert_called_once_with(limit=None, show_expired=True)

    def test_admin_path_default_hides_expired(self, monkeypatch):
        list_all = MagicMock(return_value={"results": [{"id": "m1"}]})

        monkeypatch.setattr("server.routers.compat.ensure_admin", lambda request, auth: None)
        monkeypatch.setattr("server.routers.compat.list_all_memories", list_all)

        v1_list_memories(request=MagicMock(), auth=MagicMock())

        list_all.assert_called_once_with(limit=None, show_expired=None)


# ---------------------------------------------------------------------------
# v3_get_all_memories — show_expired passthrough
# ---------------------------------------------------------------------------


class TestV3GetAllMemoriesShowExpired:
    def test_show_expired_passed_to_get_all(self, monkeypatch):
        mem = MagicMock()
        mem.get_all.return_value = [{"id": "m1"}]

        def _get_mem():
            return mem

        monkeypatch.setattr("server.routers.compat.get_memory_instance", _get_mem)

        req = MagicMock()
        req.url.path = "/v3/memories"
        req.query_params = {"page": "1", "page_size": "10"}
        body = MemoryGetInputV3(filters={"user_id": "u1"}, show_expired=True)

        from server.routers.compat import v3_get_all_memories

        v3_get_all_memories(request=req, body=body, page=1, page_size=10, _auth=None)

        mem.get_all.assert_called_once_with(filters={"user_id": "u1"}, show_expired=True)


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
        v3_search_memories(body, _auth=None)

        assert mem.search.call_args.kwargs["show_expired"] is True

    def test_show_expired_defaults_to_none(self, monkeypatch):
        mem = MagicMock()
        mem.search.return_value = {"results": []}

        def _get_mem():
            return mem

        monkeypatch.setattr("server.routers.compat.get_memory_instance", _get_mem)

        body = MemorySearchInputV3(query="hello", user_id="u1")
        v3_search_memories(body, _auth=None)

        assert "show_expired" not in mem.search.call_args.kwargs


# ---------------------------------------------------------------------------
# v2_list_memories — show_expired passthrough
# ---------------------------------------------------------------------------


class TestV2ListMemoriesShowExpired:
    def test_show_expired_passed_to_get_all(self, monkeypatch):
        mem = MagicMock()
        mem.get_all.return_value = [{"id": "m1"}]

        def _get_mem():
            return mem

        monkeypatch.setattr("server.routers.compat.get_memory_instance", _get_mem)

        req = MagicMock()
        req.url.path = "/v2/memories"
        req.query_params = {"page": "1", "page_size": "10"}
        body = MemoryGetInputV2(filters={"user_id": "u1"}, show_expired=True)

        from server.routers.compat import v2_list_memories

        v2_list_memories(request=req, body=body, page=1, page_size=10, _auth=None)

        mem.get_all.assert_called_once_with(filters={"user_id": "u1"}, show_expired=True)


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

        from server.routers.compat import v2_search_memories

        v2_search_memories(body, _auth=None)

        assert mem.search.call_args.kwargs["show_expired"] is True


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

        from server.routers.compat import v1_search_memories

        v1_search_memories(body, _auth=None)

        assert mem.search.call_args.kwargs["show_expired"] is True


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

        from server.routers.compat import v1_get_entity_memories

        v1_get_entity_memories(entity_type="user", entity_id="alice", show_expired=True, _auth=None)

        mem.get_all.assert_called_once_with(filters={"user_id": "alice"}, show_expired=True)
