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


def _validate_entity_value(key: str, value: Any) -> str:
    """Validate entity scope values before forwarding them to the SDK."""
    if not isinstance(value, str):
        raise HTTPException(status_code=400, detail=f"'{key}' must be a string entity id.")
    return value


def _scan_filters(
    filters: dict[str, Any],
) -> dict[str, str]:
    """Recursively walk a filter tree.

    Collects positive entity params (``ENTITY_PARAMS``) that constrain the whole
    tree. ``NOT`` clauses do not count as positive scope, and ``OR`` only counts
    entity params shared by every branch with the same value.
    """
    result: dict[str, str] = {}
    for key in ENTITY_PARAMS:
        if filters.get(key) is not None:
            result[key] = _validate_entity_value(key, filters[key])

    and_sub = filters.get("AND")
    if isinstance(and_sub, list):
        for cond in and_sub:
            if isinstance(cond, dict):
                _merge_entity_params(result, _scan_filters(cond))
    elif isinstance(and_sub, dict):
        _merge_entity_params(result, _scan_filters(and_sub))

    or_sub = filters.get("OR")
    if isinstance(or_sub, list):
        _merge_entity_params(result, _shared_or_entity_params(or_sub))
    elif isinstance(or_sub, dict):
        _merge_entity_params(result, _scan_filters(or_sub))

    return result


def _merge_entity_params(target: dict[str, str], source: dict[str, str]) -> None:
    for key, value in source.items():
        if key in target and target[key] != value:
            raise HTTPException(
                status_code=400,
                detail=f"Conflicting values for '{key}' in filter conditions.",
            )
        target[key] = value


def _shared_or_entity_params(conditions: list[Any]) -> dict[str, str]:
    """Return entity params that positively constrain every branch of an OR."""
    branch_scopes = [_scan_filters(cond) for cond in conditions if isinstance(cond, dict)]
    if len(branch_scopes) != len(conditions) or not branch_scopes:
        return {}

    shared: dict[str, str] = {}
    for key in ENTITY_PARAMS:
        first = branch_scopes[0].get(key)
        if first is None:
            continue
        if all(scope.get(key) == first for scope in branch_scopes[1:]):
            shared[key] = first
    return shared


def collect_direct_entity_params(
    *,
    user_id: Optional[str] = None,
    agent_id: Optional[str] = None,
    app_id: Optional[str] = None,
    run_id: Optional[str] = None,
    filters: Optional[dict[str, Any]] = None,
) -> dict[str, str]:
    """Collect entity params that can be safely forwarded as direct SDK kwargs.

    Write paths such as ``Memory.add`` need concrete ownership kwargs
    (``user_id=...``), not query predicates. This helper therefore only reads
    explicit arguments and flat/top-level entity keys in ``filters``. It does
    not scan ``AND`` / ``OR`` / ``NOT`` trees; those are handled by
    ``require_entity_scope`` for read-path scope validation.
    """
    merged: dict[str, Any] = {}
    if filters:
        for key in ENTITY_PARAMS:
            if filters.get(key) is not None:
                merged[key] = _validate_entity_value(key, filters[key])
    for key, val in (("user_id", user_id), ("agent_id", agent_id), ("app_id", app_id), ("run_id", run_id)):
        if val is not None:
            val = _validate_entity_value(key, val)
            if key in merged and merged[key] != val:
                raise HTTPException(
                    status_code=400,
                    detail=f"Conflicting values for '{key}' in filters and explicit parameters.",
                )
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
    """Resolve required entity scope for read paths.

    Starts with direct params from ``collect_direct_entity_params``, then scans
    positive scope constraints from logical filter trees. If *fallback_user_id*
    is given and no entity params are present, returns ``{"user_id":
    fallback_user_id}`` instead of raising.
    """
    params = collect_direct_entity_params(
        user_id=user_id,
        agent_id=agent_id,
        app_id=app_id,
        run_id=run_id,
        filters=filters,
    )
    if filters:
        _merge_entity_params(params, _scan_filters(filters))
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
    """Resolve scope then copy it to top-level ``filters`` for SDK validation."""
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
    # The OSS SDK validates top-level entity scope before it processes logical
    # operators, so duplicate the effective scope while preserving the original tree.
    merged.update(scope)
    return merged


def filter_tree_has_positive_key(filters: Any, needle: str) -> bool:
    """Return True when *needle* is already constrained by a positive filter.

    ``NOT`` does not count as a positive constraint. For ``OR``, a key only
    constrains the whole tree if every branch contains a positive constraint for it.
    """
    if isinstance(filters, dict):
        if needle in filters:
            return True
        and_sub = filters.get("AND")
        if isinstance(and_sub, list):
            if any(filter_tree_has_positive_key(item, needle) for item in and_sub):
                return True
        if isinstance(and_sub, dict) and filter_tree_has_positive_key(and_sub, needle):
            return True

        or_sub = filters.get("OR")
        if isinstance(or_sub, list) and or_sub:
            if all(filter_tree_has_positive_key(item, needle) for item in or_sub):
                return True
        if isinstance(or_sub, dict) and filter_tree_has_positive_key(or_sub, needle):
            return True
    elif isinstance(filters, list):
        return any(filter_tree_has_positive_key(item, needle) for item in filters)
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
    if _has_logical_operators(merged):
        return _merge_into_logical_tree(merged, extra_clauses)
    return _merge_into_flat_filters(merged, extra_clauses)


def _has_logical_operators(filters: dict[str, Any]) -> bool:
    """Return whether the top-level filter dict contains logical operators."""
    return any(key in filters for key in ("AND", "OR", "NOT"))


def _merge_into_flat_filters(filters: dict[str, Any], extra_clauses: list[dict[str, Any]]) -> dict[str, Any]:
    """Merge convenience clauses into a flat filter dict using setdefault semantics."""
    merged = dict(filters)
    for clause in extra_clauses:
        for key, value in clause.items():
            merged.setdefault(key, value)
    return merged


def _merge_into_logical_tree(filters: dict[str, Any], extra_clauses: list[dict[str, Any]]) -> dict[str, Any]:
    """Append convenience clauses to a logical filter tree as top-level AND terms."""
    merged = dict(filters)
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
    if categories and not filter_tree_has_positive_key(filters, "categories"):
        extra_clauses.append({"categories": build_categories_filter(categories)})
    if metadata:
        for key, value in metadata.items():
            if not filter_tree_has_positive_key(filters, key):
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
    sdk_filters: dict[str, Any] = dict(body.filters) if body.filters else {}
    sdk_filters.update(entity_params)

    extra_clauses: list[dict[str, Any]] = []
    date_filter: dict[str, str] = {}
    if body.start_date:
        date_filter["gte"] = body.start_date
    if body.end_date:
        date_filter["lte"] = body.end_date
    if date_filter and not filter_tree_has_positive_key(sdk_filters, "created_at"):
        extra_clauses.append({"created_at": date_filter})
    if body.categories and not filter_tree_has_positive_key(sdk_filters, "categories"):
        extra_clauses.append({"categories": build_categories_filter(body.categories)})

    return merge_extra_clauses_into_filters(sdk_filters, extra_clauses)
