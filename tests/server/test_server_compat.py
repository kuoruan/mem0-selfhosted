"""Unit tests for the server compat utilities.

Covers:
  - compat.scope: collect_entity_params, require_entity_scope,
                    build_search_filters, get_entity_field
  - compat.responses: drop_none, normalize_results, normalize_results_dict
  - compat.decorators: upstream_guard exception mapping
  - routers.compat helpers: build_list_filters, paginate_response,
                            warn_unsupported_fields, build_search_kwargs,
                            resolve_existing, merge_and_update
"""

import logging
import sys
from typing import Any
from unittest.mock import MagicMock

import pytest

pytest.importorskip("fastapi", reason="fastapi not installed")

from fastapi import BackgroundTasks, HTTPException
from pydantic import ValidationError
from starlette.datastructures import URL
from mem0.exceptions import ValidationError as Mem0ValidationError
from server.compat.events import event_cache_all, event_cache_clear, event_cache_get, event_cache_put, event_cache_update
from server.compat.requests import RequestMeta
from server.compat.decorators import upstream_guard
from server.compat.responses import (
    drop_none,
    normalize_results,
    normalize_results_dict,
    resolve_optional_pagination,
)
from server.errors import UpstreamError
from server.compat.scope import (
    build_categories_filter,
    build_search_filters,
    collect_entity_params,
    get_entity_field,
    require_entity_scope,
)
from server.routers.compat import (
    MemoryBatchDeleteInput,
    MemoryBatchDeleteLegacyInput,
    MemoryAddInputV3,
    MemoryGetInputV2,
    build_list_filters,
    build_search_kwargs,
    merge_and_update,
    paginate_response,
    resolve_existing,
    warn_unsupported_fields,
    v1_get_event,
    v1_list_events,
    v1_list_memories,
    v3_add_memory,
)


# ---------------------------------------------------------------------------
# compat.entities.CompatEntity
# ---------------------------------------------------------------------------


class TestCompatEntity:
    def test_from_bucket_serializes_timestamps(self):
        from compat.entities import CompatEntity
        from datetime import datetime, timezone

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
        import compat.entities as entities_mod
        from compat.entities import list_entities_payload

        row = MagicMock(
            payload={
                "user_id": "alice",
                "created_at": "2026-01-01T00:00:00+00:00",
                "updated_at": "2026-01-02T00:00:00+00:00",
            }
        )
        mem = MagicMock()
        mem.vector_store.list.return_value = [row]
        monkeypatch.setattr(entities_mod, "get_memory_instance", lambda: mem)

        entities = list_entities_payload()
        assert len(entities) == 1
        assert entities[0].id == "alice"
        assert entities[0].total_memories == 1


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


class TestCollectEntityParams:
    def test_explicit_user_id(self):
        assert collect_entity_params(user_id="u1") == {"user_id": "u1"}

    def test_multiple_kwargs(self):
        result = collect_entity_params(user_id="u1", agent_id="a1")
        assert result == {"user_id": "u1", "agent_id": "a1"}

    def test_app_id_kwarg(self):
        assert collect_entity_params(app_id="app1") == {"app_id": "app1"}

    def test_flat_filters(self):
        result = collect_entity_params(filters={"user_id": "u1", "agent_id": "a1"})
        assert result == {"user_id": "u1", "agent_id": "a1"}

    def test_non_entity_keys_in_filters_ignored(self):
        result = collect_entity_params(filters={"user_id": "u1", "created_at": {"gte": "2024"}})
        assert result == {"user_id": "u1"}

    def test_kwargs_override_filters(self):
        result = collect_entity_params(
            user_id="explicit",
            filters={"user_id": "from_filter"},
        )
        assert result == {"user_id": "explicit"}

    def test_and_nested(self):
        result = collect_entity_params(filters={"AND": [{"user_id": "u1"}, {"created_at": {"gte": "2024"}}]})
        assert result == {"user_id": "u1"}

    def test_or_nested(self):
        result = collect_entity_params(filters={"OR": [{"user_id": "u1"}, {"agent_id": "a1"}]})
        assert result == {"user_id": "u1", "agent_id": "a1"}

    def test_app_id_nested_and(self):
        result = collect_entity_params(filters={"AND": [{"app_id": "app1"}, {"user_id": "u1"}]})
        assert result == {"app_id": "app1", "user_id": "u1"}

    def test_app_id_nested_or(self):
        result = collect_entity_params(filters={"OR": [{"app_id": "app1"}, {"agent_id": "a1"}]})
        assert result == {"app_id": "app1", "agent_id": "a1"}

    def test_none_values_skipped(self):
        assert collect_entity_params(user_id=None, agent_id=None) == {}

    def test_empty_returns_empty(self):
        assert collect_entity_params() == {}




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

    def test_flat_filters_entity_already_present(self):
        """Entity kwarg should overwrite same key already in flat filters."""
        result = build_search_filters(
            user_id="explicit",
            filters={"user_id": "from_filter", "created_at": {"gte": "2024"}},
        )
        assert result["user_id"] == "explicit"
        assert "created_at" in result

    def test_and_entity_in_conditions_not_re_injected(self):
        """user_id already inside AND conditions: nothing extra to inject."""
        filters = {"AND": [{"user_id": "u1"}, {"created_at": {"gte": "2024"}}]}
        result = build_search_filters(filters=filters)
        # result should equal the original AND structure, not add user_id at top level
        assert result == filters
        assert "user_id" not in result

    def test_and_extra_entity_kwarg_injected_into_and(self):
        """user_id from explicit kwarg not present in AND conditions: inject into AND list."""
        filters = {"AND": [{"created_at": {"gte": "2024"}}]}
        result = build_search_filters(user_id="u1", filters=filters)
        assert "AND" in result
        assert "user_id" not in result  # not at top level
        and_items = result["AND"]
        assert any(item.get("user_id") == "u1" for item in and_items)

    def test_and_does_not_mutate_input(self):
        """Injecting into AND must not modify the original filters object."""
        original = {"AND": [{"created_at": {"gte": "2024"}}]}
        build_search_filters(user_id="u1", filters=original)
        assert original == {"AND": [{"created_at": {"gte": "2024"}}]}

    def test_or_extra_entity_kwarg_wraps_in_outer_and(self):
        """user_id from explicit kwarg with OR filters: wrap in AND."""
        filters = {"OR": [{"user_id": "u1"}, {"agent_id": "a1"}]}
        result = build_search_filters(user_id="explicit", filters=filters)
        assert "AND" in result
        assert "OR" not in result  # OR is nested inside AND now
        outer_and = result["AND"]
        assert any("OR" in item for item in outer_and)
        assert any(item.get("user_id") == "explicit" for item in outer_and)

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

    def test_app_id_injected_into_and_filters(self):
        filters = {"AND": [{"user_id": "u1"}]}
        result = build_search_filters(app_id="app1", filters=filters)
        assert any(item.get("app_id") == "app1" for item in result["AND"])


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


class TestResolveEventOwnerId:
    def test_extracts_from_user_object(self):
        from compat.events import resolve_event_owner_id

        auth = MagicMock()
        auth.id = "user-1"
        assert resolve_event_owner_id(auth) == "user-1"

    def test_extracts_from_dict_id(self):
        from compat.events import resolve_event_owner_id

        assert resolve_event_owner_id({"id": "user-2"}) == "user-2"

    def test_extracts_id_from_auth_list(self):
        from compat.events import resolve_event_owner_id

        assert resolve_event_owner_id([{"id": "user-3"}]) == "user-3"

    def test_falls_back_to_entity_params_user_id(self):
        from compat.events import resolve_event_owner_id

        assert resolve_event_owner_id(None, {"user_id": "scoped-user"}) == "scoped-user"
        assert resolve_event_owner_id([], {"user_id": "scoped-user"}) == "scoped-user"


class TestCompatEvent:
    def test_pending_sets_timestamps_and_empty_results(self):
        from compat.events import CompatEvent

        event = CompatEvent.pending("evt-1", now_iso="2024-01-01T00:00:00+00:00")
        assert event.status == "PENDING"
        assert event.results == []
        assert event.completed_at is None
        assert event.latency is None
        assert event.created_at == "2024-01-01T00:00:00+00:00"

    def test_create_add_succeeded_sets_completed_at(self):
        from compat.events import CompatEvent

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
        from compat.events import CompatEvent

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

    def test_and_format_skips_date_categories_merge(self):
        """Logical format: convenience fields are AND-ed at top level."""
        body = self._body(
            filters={"AND": [{"user_id": "u1"}]},
            start_date="2024-01-01",
            categories=["finance"],
        )
        result = build_list_filters(body, {"user_id": "u1"})
        assert "AND" in result
        assert {"created_at": {"gte": "2024-01-01"}} in result["AND"]
        assert {"categories": {"contains": "finance"}} in result["AND"]

    def test_and_filters_do_not_mix_top_level_entity_keys(self):
        body = self._body(filters={"AND": [{"user_id": "u1"}, {"created_at": {"gte": "2024"}}]})
        result = build_list_filters(body, {"user_id": "u1"})
        assert result == {"AND": [{"user_id": "u1"}, {"created_at": {"gte": "2024"}}]}

    def test_or_filters_do_not_mix_top_level_entity_keys(self):
        body = self._body(filters={"OR": [{"user_id": "u1"}, {"agent_id": "a1"}]})
        result = build_list_filters(body, {"user_id": "u1", "agent_id": "a1"})
        assert result == {"OR": [{"user_id": "u1"}, {"agent_id": "a1"}]}

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
        from server.routers.compat import MemorySearchInputV3, v3_search_memories

        mem = MagicMock()
        mem.search.return_value = {"results": []}
        monkeypatch.setattr("server.routers.compat.get_memory_instance", lambda: mem)

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
        # build_search_filters wraps OR with outer AND for explicit user_id
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
        mem.update.assert_called_once_with(memory_id="mem-1", data="old text", metadata={"key": "val"})

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
        mem.update.assert_called_once_with(memory_id="mem-1", data="txt", metadata={"a": 1, "b": 99, "c": 3})

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
        mem.update.assert_called_once_with(memory_id="mem-1", data="txt", metadata={"new_key": "val"})


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

        monkeypatch.setattr("server.routers.compat.get_memory_instance", lambda: mem)

        result = v1_list_memories(request=MagicMock(), user_id="u1", auth=None)

        assert result == [{"id": "m1"}]
        mem.get_all.assert_called_once_with(filters={"user_id": "u1"})


class TestSyntheticEvents:
    @pytest.fixture(autouse=True)
    def _clear_events(self):
        event_cache_clear()
        compat_events = sys.modules.get("compat.events")
        if compat_events is not None and hasattr(compat_events, "event_cache_clear"):
            compat_events.event_cache_clear()
        yield
        event_cache_clear()
        compat_events = sys.modules.get("compat.events")
        if compat_events is not None and hasattr(compat_events, "event_cache_clear"):
            compat_events.event_cache_clear()

    @staticmethod
    def _run_background_tasks(tasks: BackgroundTasks) -> None:
        for task in tasks.tasks:
            task.func(*task.args, **task.kwargs)

    def test_v3_add_returns_event_id_and_event_is_fetchable(self, monkeypatch):
        mem = MagicMock()
        mem.add.return_value = {"results": [{"id": "m1", "memory": "saved"}]}
        get_mem = lambda: mem
        monkeypatch.setattr("server.routers.compat.get_memory_instance", get_mem)
        monkeypatch.setattr("server.server_state.get_memory_instance", get_mem)

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
        get_mem = lambda: mem
        monkeypatch.setattr("server.routers.compat.get_memory_instance", get_mem)
        monkeypatch.setattr("server.server_state.get_memory_instance", get_mem)

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
        get_mem = lambda: mem
        monkeypatch.setattr("server.routers.compat.get_memory_instance", get_mem)
        monkeypatch.setattr("server.server_state.get_memory_instance", get_mem)

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
        get_mem = lambda: mem
        monkeypatch.setattr("server.routers.compat.get_memory_instance", get_mem)
        monkeypatch.setattr("server.server_state.get_memory_instance", get_mem)

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
        get_mem = lambda: mem
        monkeypatch.setattr("server.routers.compat.get_memory_instance", get_mem)
        monkeypatch.setattr("server.server_state.get_memory_instance", get_mem)

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
        get_mem = lambda: mem
        monkeypatch.setattr("server.routers.compat.get_memory_instance", get_mem)
        monkeypatch.setattr("server.server_state.get_memory_instance", get_mem)

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
        get_mem = lambda: mem
        monkeypatch.setattr("server.routers.compat.get_memory_instance", get_mem)
        monkeypatch.setattr("server.server_state.get_memory_instance", get_mem)
        monkeypatch.setattr("server.compat.tasks.time.perf_counter", MagicMock(side_effect=[10.0, 10.25]))

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
        get_mem = lambda: mem
        monkeypatch.setattr("server.routers.compat.get_memory_instance", get_mem)
        monkeypatch.setattr("server.server_state.get_memory_instance", get_mem)

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
