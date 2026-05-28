"""Entity-scope resolution for REST and MCP handlers.

Provides helpers to collect, validate, and merge entity-identifying parameters
(``user_id``, ``agent_id``, ``app_id``, ``run_id``) from request bodies and query strings.
"""

from typing import Any, Optional

from fastapi import HTTPException

ENTITY_PARAMS = frozenset({"user_id", "agent_id", "app_id", "run_id"})

COMPAT_TYPE_TO_FIELD: dict[str, str] = {
    "user": "user_id",
    "agent": "agent_id",
    "app": "app_id",
    "run": "run_id",
}
VALID_ENTITY_TYPES = frozenset(COMPAT_TYPE_TO_FIELD)


def build_categories_filter(categories: list[str]) -> dict[str, Any]:
    """Build a categories filter matching platform operator semantics."""
    if len(categories) == 1:
        return {"contains": categories[0]}
    return {"in": categories}


def _scan_filters(
    filters: dict[str, Any],
) -> dict[str, str]:
    """Recursively walk a filter tree.

    Collects and returns entity params (``ENTITY_PARAMS``) found at any depth.
    Handles flat format, AND/OR/NOT list conditions, and arbitrary nesting.
    """
    result: dict[str, str] = {}
    for key in ENTITY_PARAMS:
        if filters.get(key) is not None:
            result[key] = filters[key]
    for op in ("AND", "OR", "NOT"):
        sub = filters.get(op)
        if isinstance(sub, list):
            for cond in sub:
                if isinstance(cond, dict):
                    sub_result = _scan_filters(cond)
                    for k, v in sub_result.items():
                        if k in result and result[k] != v:
                            raise HTTPException(
                                status_code=400,
                                detail=f"Conflicting values for '{k}' in filter conditions.",
                            )
                    result.update(sub_result)
        elif isinstance(sub, dict):
            sub_result = _scan_filters(sub)
            for k, v in sub_result.items():
                if k in result and result[k] != v:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Conflicting values for '{k}' in filter conditions.",
                    )
            result.update(sub_result)
    return result


def collect_entity_params(
    *,
    user_id: Optional[str] = None,
    agent_id: Optional[str] = None,
    app_id: Optional[str] = None,
    run_id: Optional[str] = None,
    filters: Optional[dict[str, Any]] = None,
) -> dict[str, str]:
    """Collect non-None entity params, preferring explicit kwargs over *filters*."""
    merged: dict[str, Any] = {}
    if filters:
        merged.update(_scan_filters(filters))
    for key, val in (("user_id", user_id), ("agent_id", agent_id), ("app_id", app_id), ("run_id", run_id)):
        if val is not None:
            merged[key] = val
    return merged


def require_entity_scope(
    *,
    user_id: Optional[str] = None,
    agent_id: Optional[str] = None,
    app_id: Optional[str] = None,
    run_id: Optional[str] = None,
    filters: Optional[dict[str, Any]] = None,
    detail: str = "One of the filters: user_id, agent_id, app_id or run_id is required!",
    fallback_user_id: Optional[str] = None,
) -> dict[str, str]:
    """Like ``collect_entity_params`` but raises 400 when no scope is found.

    If *fallback_user_id* is given and no entity params are present, returns
    ``{"user_id": fallback_user_id}`` instead of raising.
    """
    params = collect_entity_params(
        user_id=user_id,
        agent_id=agent_id,
        app_id=app_id,
        run_id=run_id,
        filters=filters,
    )
    if not params:
        if fallback_user_id:
            return {"user_id": fallback_user_id}
        raise HTTPException(status_code=400, detail=detail)
    return params


def build_search_filters(
    *,
    user_id: Optional[str] = None,
    agent_id: Optional[str] = None,
    app_id: Optional[str] = None,
    run_id: Optional[str] = None,
    filters: Optional[dict[str, Any]] = None,
    detail: str = "At least one of the filters: agent_id, user_id, app_id or run_id is required!",
    fallback_user_id: Optional[str] = None,
) -> dict[str, Any]:
    """Resolve scope then merge into *filters* dict for ``Memory.search`` / ``get_all``."""
    scope = require_entity_scope(
        user_id=user_id,
        agent_id=agent_id,
        app_id=app_id,
        run_id=run_id,
        filters=filters,
        detail=detail,
        fallback_user_id=fallback_user_id,
    )
    merged: dict[str, Any] = dict(filters) if filters else {}
    if "AND" not in merged and "OR" not in merged and "NOT" not in merged:
        # Flat format: safe to merge entity scope at top level.
        merged.update(scope)
        return merged
    # AND/OR/NOT format: merging scope at top level would produce a malformed mixed-format dict
    # (e.g. {"AND": [...], "user_id": "bob"}).  Only inject entity params that came from
    # explicit kwargs (plus fallback) — those already inside AND/OR conditions are already
    # present in 'merged' and do not need re-injection.
    kwargs_scope = collect_entity_params(user_id=user_id, agent_id=agent_id, app_id=app_id, run_id=run_id)
    if not kwargs_scope and fallback_user_id:
        kwargs_scope = {"user_id": fallback_user_id}
    if not kwargs_scope:
        # All scope came from within the AND/OR/NOT conditions; nothing extra to inject.
        return merged
    if "AND" in merged and isinstance(merged["AND"], list):
        # Append extra entity conditions into the AND list (new list to avoid mutating caller's filters).
        merged["AND"] = [*merged["AND"], *({k: v} for k, v in kwargs_scope.items())]
    else:
        # OR/NOT format: wrap in an outer AND to restrict to the entity without modifying inner logic.
        # Each entity param becomes its own dict for uniform filter structure.
        merged = {"AND": [merged, *({k: v} for k, v in kwargs_scope.items())]}
    return merged


def filter_tree_has_key(filters: Any, needle: str) -> bool:
    """Return True if *needle* appears as a key anywhere in a filter tree."""
    if isinstance(filters, dict):
        if needle in filters:
            return True
        for op in ("AND", "OR", "NOT"):
            if filter_tree_has_key(filters.get(op), needle):
                return True
    elif isinstance(filters, list):
        return any(filter_tree_has_key(item, needle) for item in filters)
    return False


def merge_extra_clauses_into_filters(
    filters: dict[str, Any],
    extra_clauses: list[dict[str, Any]],
) -> dict[str, Any]:
    """Merge convenience filter clauses into flat or logical filter trees.

    For flat dicts, uses ``setdefault`` so explicit filter keys win. For AND/OR/NOT
    trees, appends to a top-level ``AND`` list or wraps the tree in an outer ``AND``.
    """
    if not extra_clauses:
        return filters

    merged = dict(filters)
    has_logical = any(key in merged for key in ("AND", "OR", "NOT"))
    if not has_logical:
        for clause in extra_clauses:
            for key, value in clause.items():
                merged.setdefault(key, value)
        return merged

    if "AND" in merged and isinstance(merged["AND"], list):
        merged["AND"] = [*merged["AND"], *extra_clauses]
        return merged
    return {"AND": [merged, *extra_clauses]}


def append_search_convenience_filters(
    filters: dict[str, Any],
    *,
    categories: Optional[list[str]] = None,
    metadata: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Merge top-level search body ``categories`` / ``metadata`` into SDK filters."""
    extra_clauses: list[dict[str, Any]] = []
    if categories and not filter_tree_has_key(filters, "categories"):
        extra_clauses.append({"categories": build_categories_filter(categories)})
    if metadata:
        for key, value in metadata.items():
            if not filter_tree_has_key(filters, key):
                extra_clauses.append({key: value})
    return merge_extra_clauses_into_filters(filters, extra_clauses)


def get_entity_field(entity_type: str) -> str:
    """Map entity type name (``"user"``) to payload field name (``"user_id"``).

    Raises 400 for unknown types.
    """
    field = COMPAT_TYPE_TO_FIELD.get(entity_type)
    if field is None:
        raise HTTPException(status_code=400, detail="Invalid entity type")
    return field


def build_list_filters(body: Any, entity_params: dict[str, str]) -> dict[str, Any]:
    """Build SDK filter dict for get_all from request body and entity params."""
    sdk_filters: dict[str, Any] = dict(body.filters) if body.filters else dict(entity_params)

    extra_clauses: list[dict[str, Any]] = []
    date_filter: dict[str, str] = {}
    if body.start_date:
        date_filter["gte"] = body.start_date
    if body.end_date:
        date_filter["lte"] = body.end_date
    if date_filter and not filter_tree_has_key(sdk_filters, "created_at"):
        extra_clauses.append({"created_at": date_filter})
    if body.categories and not filter_tree_has_key(sdk_filters, "categories"):
        extra_clauses.append({"categories": build_categories_filter(body.categories)})

    return merge_extra_clauses_into_filters(sdk_filters, extra_clauses)
