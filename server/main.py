import asyncio
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

import telemetry
from auth import ensure_admin, is_bootstrap_admin, require_admin, verify_auth
from auth_config import get_auth_config
from bg_tasks import prune_loop
from db import SessionLocal, get_db
from dotenv import load_dotenv
from errors import (
    UpstreamError,
    install_request_id_logging,
    new_request_id,
    request_id_var,
    upstream_error,
    upstream_error_handler,
)
from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from mcp_server import setup_mcp_server
from memory_lock import entity_scope_from_params, run_memory_write, run_memory_write_for_memory_id
from models import RequestLog, User
from pydantic import BaseModel, Field
from rate_limit import limiter
from routers import api_keys as api_keys_router
from routers import auth as auth_router
from routers import compat as compat_router
from routers import entities as entities_router
from routers import oidc as oidc_router
from routers import requests as requests_router
from routers import users as users_router
from schemas import MessageResponse
from server_state import (
    ALL_MEMORIES_LIMIT,
    get_current_config,
    get_memory_instance,
    initialize_state,
    list_all_memories,
    set_session_factory,
    update_config,
)
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from entity_permissions import (
    authorize_write,
    bulk_delete_memories,
    check_memory_scope_permission,
    check_query_permission,
    inject_default_user_id,
    reject_bootstrap_memory_mutation,
    resolve_memory_entities,
    resolve_operator,
)
from utils.helpers import is_wildcard

from mem0.exceptions import ValidationError as Mem0ValidationError

load_dotenv()

install_request_id_logging()

LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()

_log_level = getattr(logging, LOG_LEVEL, logging.INFO)
logging.basicConfig(level=_log_level, format="%(asctime)s - %(levelname)s - [%(request_id)s] %(message)s")

MIN_KEY_LENGTH = 16
SENSITIVE_CONFIG_KEYS = {
    "admin_api_key",
    "api_key",
    "authorization",
    "jwt_secret",
    "password",
    "password_hash",
    "secret",
    "token",
}
SKIPPED_REQUEST_LOG_PATHS = {"/api/health", "/docs", "/redoc", "/openapi.json", "/v1/ping"}
SKIPPED_REQUEST_LOG_PREFIXES = ("/requests",)

BUNDLED_LLM_PROVIDERS = ("openai", "anthropic", "gemini", "deepseek", "ollama")
BUNDLED_EMBEDDER_PROVIDERS = ("openai", "gemini", "ollama")


def _warn_if_unconfigured() -> None:
    """Pre-auth deployments upgrading into this build will 401 everywhere until
    an admin key or admin user exists. Surface the fix before the support tickets."""
    try:
        with SessionLocal() as session:
            if session.scalar(select(func.count(User.id))) > 0:
                return
    except Exception:
        return

    logging.warning(
        "\n%s\n"
        "  Auth is enabled by default and this server has no admin configured.\n"
        "  Protected endpoints will return 401 until you either:\n"
        "    1. Set ADMIN_API_KEY=<long-random-value>  (fastest, no client changes)\n"
        "    2. Register an admin at http://<host>:3000/setup\n"
        "    3. Set AUTH_DISABLED=true                 (local development only)\n"
        "  Docs: https://docs.mem0.ai/open-source/features/rest-api#authentication\n"
        "%s",
        "=" * 72,
        "=" * 72,
    )


_auth_config = get_auth_config()

if not _auth_config.auth_disabled and not _auth_config.jwt_secret:
    raise RuntimeError(
        "JWT_SECRET is required. Set it in .env (generate with `openssl rand -base64 48`) "
        "or set AUTH_DISABLED=true for local development only."
    )

if _auth_config.auth_disabled:
    logging.warning("AUTH_DISABLED is enabled. Protected endpoints are open for local development only.")
elif _auth_config.admin_api_key and len(_auth_config.admin_api_key) < MIN_KEY_LENGTH:
    logging.warning(
        "ADMIN_API_KEY is shorter than %d characters - consider using a longer key for production.",
        MIN_KEY_LENGTH,
    )
elif not _auth_config.admin_api_key:
    _warn_if_unconfigured()

if _auth_config.oidc and _auth_config.oidc.providers:
    if os.environ.get("OIDC_STATE_STORE", "memory").lower() == "memory":
        logging.warning(
            "OIDC is enabled with the in-memory state/exchange store. "
            "Login state is held per-process, so multi-worker (e.g. uvicorn --workers N) "
            "or multi-replica deployments will see intermittent 'invalid_state' / 'expired' "
            "login failures. For such deployments, implement a shared backend (Redis/DB), "
            "select it via OIDC_STATE_STORE, and rebuild."
        )

telemetry.log_status()

POSTGRES_HOST = os.environ.get("POSTGRES_HOST", "postgres")
POSTGRES_PORT = os.environ.get("POSTGRES_PORT", "5432")
POSTGRES_DB = os.environ.get("POSTGRES_DB", "postgres")
POSTGRES_USER = os.environ.get("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.environ.get("POSTGRES_PASSWORD", "postgres")
POSTGRES_COLLECTION_NAME = os.environ.get("POSTGRES_COLLECTION_NAME", "memories")

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
HISTORY_DB_PATH = os.environ.get("HISTORY_DB_PATH", "/app/history/history.db")
DEFAULT_LLM_MODEL = os.environ.get("MEM0_DEFAULT_LLM_MODEL", "gpt-4.1-nano-2025-04-14")
DEFAULT_EMBEDDER_MODEL = os.environ.get("MEM0_DEFAULT_EMBEDDER_MODEL", "text-embedding-3-small")
CONFIG_PATH = os.environ.get("MEM0_CONFIG_PATH")

DEFAULT_CONFIG = {
    "version": "v1.1",
    "vector_store": {
        "provider": "pgvector",
        "config": {
            "host": POSTGRES_HOST,
            "port": int(POSTGRES_PORT),
            "dbname": POSTGRES_DB,
            "user": POSTGRES_USER,
            "password": POSTGRES_PASSWORD,
            "collection_name": POSTGRES_COLLECTION_NAME,
        },
    },
    "llm": {
        "provider": "openai",
        "config": {"api_key": OPENAI_API_KEY, "temperature": 0.2, "model": DEFAULT_LLM_MODEL},
    },
    "embedder": {"provider": "openai", "config": {"api_key": OPENAI_API_KEY, "model": DEFAULT_EMBEDDER_MODEL}},
    "history_db_path": HISTORY_DB_PATH,
}


set_session_factory(SessionLocal)
initialize_state(DEFAULT_CONFIG, config_path=CONFIG_PATH)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start background tasks on startup, clean up on shutdown."""
    prune_task = asyncio.create_task(prune_loop())
    try:
        yield
    finally:
        prune_task.cancel()
        try:
            await prune_task
        except asyncio.CancelledError:
            pass
        logging.info("Graceful shutdown complete.")


app = FastAPI(
    lifespan=lifespan,
    title="Mem0 REST APIs",
    description=(
        "A REST API for managing and searching memories for your AI Agents and Apps.\n\n"
        "## Authentication\n"
        "Supports Bearer JWT tokens, per-user API keys via `X-API-Key` header, "
        "or the legacy `ADMIN_API_KEY` environment variable. Set `AUTH_DISABLED=true` for local development only."
    ),
    version="1.0.0",
    redirect_slashes=False,
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_exception_handler(UpstreamError, upstream_error_handler)


DASHBOARD_URL = os.environ.get("DASHBOARD_URL", "http://localhost:3000")
_cors_origins = [origin.strip() for origin in DASHBOARD_URL.split(",") if origin.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router.router)
app.include_router(api_keys_router.router)
app.include_router(oidc_router.router)
app.include_router(compat_router.router)
app.include_router(entities_router.router)
app.include_router(requests_router.router)
app.include_router(users_router.router)
setup_mcp_server(app)


class Message(BaseModel):
    role: str = Field(..., description="Role of the message (user or assistant).")
    content: str = Field(..., description="Message content.")


class MemoryCreate(BaseModel):
    messages: List[Message] = Field(..., description="List of messages to store.")
    user_id: Optional[str] = None
    agent_id: Optional[str] = None
    app_id: Optional[str] = None
    run_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    expiration_date: Optional[str] = Field(None, description="Expiration date in YYYY-MM-DD format.")
    infer: Optional[bool] = Field(None, description="Whether to extract facts from messages. Defaults to True.")
    memory_type: Optional[str] = Field(None, description="Type of memory to store (e.g. 'core').")
    prompt: Optional[str] = Field(None, description="Custom prompt to use for fact extraction.")


class MemoryUpdate(BaseModel):
    text: Optional[str] = Field(None, description="New content to update the memory with.")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Metadata to update.")
    expiration_date: Optional[str] = Field(None, description="Expiration date in YYYY-MM-DD format, or null to clear.")


class SearchRequest(BaseModel):
    query: str = Field(..., description="Search query.")
    user_id: Optional[str] = Field(None, description="Deprecated: pass inside `filters` instead.", deprecated=True)
    run_id: Optional[str] = Field(None, description="Deprecated: pass inside `filters` instead.", deprecated=True)
    agent_id: Optional[str] = Field(None, description="Deprecated: pass inside `filters` instead.", deprecated=True)
    app_id: Optional[str] = Field(None, description="Deprecated: pass inside `filters` instead.", deprecated=True)
    filters: Optional[Dict[str, Any]] = None
    top_k: Optional[int] = Field(None, description="Maximum number of results to return.")
    threshold: Optional[float] = Field(None, description="Minimum similarity score for results.")
    explain: Optional[bool] = Field(None, description="Include score details for each search result.")
    show_expired: Optional[bool] = Field(None, description="Include expired memories.")


class GenerateInstructionsRequest(BaseModel):
    use_case: str = Field(..., description="Description of what the user will use Mem0 for.")


def _client_error(exc: Exception) -> HTTPException:
    """Map core validation / not-found errors to 4xx so clients can tell a bad
    request from an upstream outage. 'not found' is a 404, everything else a 400."""
    detail = str(exc)
    status_code = 404 if isinstance(exc, ValueError) and "not found" in detail.lower() else 400
    return HTTPException(status_code=status_code, detail=detail)


def _redact_config(value: Any, key: str | None = None) -> Any:
    if isinstance(value, dict):
        return {item_key: _redact_config(item_value, item_key) for item_key, item_value in value.items()}
    if isinstance(value, list):
        return [_redact_config(item_value, key) for item_value in value]
    if key is not None and key.lower() in SENSITIVE_CONFIG_KEYS:
        return "[redacted]" if value else value
    return value


def _validate_bundled_providers(config: Dict[str, Any]) -> None:
    llm = config.get("llm")
    if isinstance(llm, dict) and (provider := llm.get("provider")) and provider not in BUNDLED_LLM_PROVIDERS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"LLM provider '{provider}' is not bundled in this image. "
                f"Bundled providers: {', '.join(BUNDLED_LLM_PROVIDERS)}. "
                "To use another provider, install its Python package, rebuild the container, "
                "and extend BUNDLED_LLM_PROVIDERS in server/main.py."
            ),
        )

    embedder = config.get("embedder")
    if (
        isinstance(embedder, dict)
        and (provider := embedder.get("provider"))
        and provider not in BUNDLED_EMBEDDER_PROVIDERS
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                f"Embedder provider '{provider}' is not bundled in this image. "
                f"Bundled providers: {', '.join(BUNDLED_EMBEDDER_PROVIDERS)}. "
                "To use another provider, install its Python package, rebuild the container, "
                "and extend BUNDLED_EMBEDDER_PROVIDERS in server/main.py."
            ),
        )


def _should_log_request(request: Request) -> bool:
    if request.method == "OPTIONS":
        return False
    path = request.url.path
    for skipped in SKIPPED_REQUEST_LOG_PATHS:
        if path == skipped or path.startswith(skipped.rstrip("/") + "/"):
            return False
    return not path.startswith(SKIPPED_REQUEST_LOG_PREFIXES)


def _persist_request_log(
    method: str, path: str, status_code: int, latency_ms: float, auth_type: str, user_id: str | None = None
) -> None:
    with SessionLocal() as session:
        try:
            session.add(
                RequestLog(
                    method=method,
                    path=path,
                    status_code=status_code,
                    latency_ms=latency_ms,
                    auth_type=auth_type,
                    user_id=uuid.UUID(user_id) if user_id else None,
                )
            )
            session.commit()
        except Exception:
            session.rollback()
            logging.exception("Failed to persist request log")


@app.middleware("http")
async def log_requests(request: Request, call_next):
    request.state.auth_type = getattr(request.state, "auth_type", "none")
    rid = new_request_id()
    token = request_id_var.set(rid)
    start = time.perf_counter()
    status_code = 500

    try:
        response = await call_next(request)
        status_code = response.status_code
        response.headers["X-Request-ID"] = rid
        return response
    except Exception:
        status_code = 500
        raise
    finally:
        request_id_var.reset(token)
        if _should_log_request(request):
            asyncio.get_running_loop().run_in_executor(
                None,
                _persist_request_log,
                request.method,
                request.url.path,
                status_code,
                round((time.perf_counter() - start) * 1000, 2),
                getattr(request.state, "auth_type", "none"),
                getattr(request.state, "user_id", None),
            )


@app.get("/api/health", summary="Health check")
def health_check():
    """Return server health status including DB and vector store connectivity."""
    checks = {"server": "ok"}

    # Check DB connectivity
    try:
        with SessionLocal() as session:
            session.execute(select(1))
            checks["db"] = "ok"
    except Exception as exc:
        checks["db"] = f"error: {exc}"

    # Check vector store via Memory instance
    try:
        get_memory_instance()
        checks["vector_store"] = "ok"
    except Exception as exc:
        checks["vector_store"] = f"error: {exc}"

    status_code = 200 if all(v == "ok" for v in checks.values()) else 503
    return JSONResponse(content=checks, status_code=status_code)


@app.get("/configure", summary="Get current Mem0 configuration")
def get_config(_auth=Depends(require_admin)):
    return _redact_config(get_current_config())


@app.get("/configure/providers", summary="List bundled LLM and embedder providers")
def list_bundled_providers(_auth=Depends(require_admin)):
    return {"llm": list(BUNDLED_LLM_PROVIDERS), "embedder": list(BUNDLED_EMBEDDER_PROVIDERS)}


@app.post("/configure", summary="Configure Mem0")
def set_config(config: Dict[str, Any], _auth=Depends(require_admin)):
    """Set memory configuration. Requires admin role."""
    _validate_bundled_providers(config)
    update_config(config)
    return {"message": "Configuration set successfully"}


@app.post("/generate-instructions", summary="Generate custom instructions from a use case")
def generate_instructions(req: GenerateInstructionsRequest, _auth=Depends(verify_auth)):
    """Generate custom instructions and a contextual test message tailored to a use case."""
    try:
        llm = get_memory_instance().llm
        prompt = (
            "You are configuring a memory system. Given the use case below, produce two things:\n"
            "1. INSTRUCTIONS: A short paragraph of custom instructions telling the memory extraction system "
            "what kinds of facts, preferences, and context to prioritize. Be specific to the use case.\n"
            "2. TEST_MESSAGE: A single realistic sentence a user in this use case would say, suitable for "
            "testing that the memory system works.\n\n"
            "Respond in exactly this format (no markdown, no extra text):\n"
            "INSTRUCTIONS: <your instructions>\n"
            f"TEST_MESSAGE: <your test message>\n\nUse case: {req.use_case}"
        )
        response = llm.generate_response([{"role": "user", "content": prompt}])
        instructions = response
        test_message = "I like to hike on weekends."
        if "INSTRUCTIONS:" in response and "TEST_MESSAGE:" in response:
            parts = response.split("TEST_MESSAGE:")
            instructions = parts[0].replace("INSTRUCTIONS:", "").strip()
            test_message = parts[1].strip()
        return {"custom_instructions": instructions, "test_message": test_message}
    except Exception:
        raise upstream_error()


@app.post("/memories", summary="Create memories")
def add_memory(
    memory_create: MemoryCreate,
    request: Request,
    auth=Depends(verify_auth),
    db: Session = Depends(get_db),
):
    """Store new memories."""
    operator, is_admin = resolve_operator(request, auth, db)
    reject_bootstrap_memory_mutation(operator)
    entity_params = {
        k: v
        for k, v in {
            "user_id": memory_create.user_id,
            "agent_id": memory_create.agent_id,
            "app_id": memory_create.app_id,
            "run_id": memory_create.run_id,
        }.items()
        if v
    }
    entity_params = inject_default_user_id(entity_params, operator)
    if not entity_params:
        # Only the admin_api_key bypass with no entity params reaches here (real
        # users get their own user_id injected above). The scope lock and SDK both
        # reject a scopeless write; fail fast with a clear message.
        raise HTTPException(
            status_code=400,
            detail="At least one identifier (user_id, agent_id, app_id, run_id) is required.",
        )

    authorize_write(entity_params, operator, db, bypass=is_admin)

    params = {k: v for k, v in memory_create.model_dump().items() if v is not None and k != "messages"}
    params.update(entity_params)
    try:
        response = run_memory_write(
            lambda memory: memory.add(messages=[m.model_dump() for m in memory_create.messages], **params),
            entity_params,
        )
        if response.get("results"):
            telemetry.log_dashboard_nudge_once(DASHBOARD_URL)
        return JSONResponse(content=response)
    except (ValueError, Mem0ValidationError) as e:
        raise _client_error(e)
    except HTTPException:
        raise
    except Exception:
        raise upstream_error()


@app.get("/memories", summary="Get memories")
def get_all_memories(
    request: Request,
    user_id: Optional[str] = None,
    run_id: Optional[str] = None,
    agent_id: Optional[str] = None,
    app_id: Optional[str] = None,
    top_k: Optional[int] = Query(None, ge=0, le=ALL_MEMORIES_LIMIT),
    show_expired: bool = Query(False),
    auth=Depends(verify_auth),
    db: Session = Depends(get_db),
):
    """Retrieve stored memories. With no identifier, returns the caller's own memories
    (the admin_api_key bootstrap bypass lists everything).

    When both ``user_id`` and ``app_id`` are provided, this endpoint enforces a
    **dual permission check**: the caller must hold read permission on both the
    user entity and the app entity. The compat (``/v1/``, ``/v2/``, ``/v3/``) and
    MCP paths use the legacy app-primary-gate behaviour instead (``user_id`` is
    stripped when ``app_id`` is present — see ``strip_user_id_for_app_gate``)."""
    operator, is_admin = resolve_operator(request, auth, db)
    try:
        filters = {
            k: v
            for k, v in {"user_id": user_id, "run_id": run_id, "agent_id": agent_id, "app_id": app_id}.items()
            if v
        }
        if not filters:
            if is_bootstrap_admin(operator):
                # admin_api_key / system bypass has no per-user scope to default to.
                ensure_admin(request, auth)
                return list_all_memories(limit=top_k if top_k is not None else ALL_MEMORIES_LIMIT, show_expired=show_expired)
            filters = {"user_id": str(operator.id)}
        # Admin user_id="*" shortcut: bypass the filter rewrite and list everything.
        if is_wildcard(filters.get("user_id")) and is_admin and len(filters) == 1:
            return list_all_memories(limit=top_k if top_k is not None else ALL_MEMORIES_LIMIT, show_expired=show_expired)
        filters = check_query_permission(filters, operator.id, db, bypass=is_admin)
        params = {"filters": filters}
        if top_k is not None:
            params["top_k"] = top_k
        params["show_expired"] = show_expired
        return get_memory_instance().get_all(**params)
    except HTTPException:
        raise
    except Exception:
        raise upstream_error()


@app.get("/memories/{memory_id}", summary="Get a memory")
def get_memory(
    memory_id: str,
    request: Request,
    auth=Depends(verify_auth),
    db: Session = Depends(get_db),
):
    """Retrieve a specific memory by ID."""
    operator, is_admin = resolve_operator(request, auth, db)
    scope = resolve_memory_entities(memory_id)
    check_memory_scope_permission(scope, operator.id, "read", db, bypass=is_admin)
    try:
        return get_memory_instance().get(memory_id)
    except HTTPException:
        raise
    except Exception:
        raise upstream_error()


@app.post("/search", summary="Search memories")
def search_memories(
    search_req: SearchRequest,
    request: Request,
    auth=Depends(verify_auth),
    db: Session = Depends(get_db),
):
    """Search for memories based on a query."""
    operator, is_admin = resolve_operator(request, auth, db)
    try:
        filters = search_req.filters or {}
        deprecated_keys = []
        for entity_key in ("user_id", "agent_id", "app_id", "run_id"):
            entity_val = getattr(search_req, entity_key, None)
            if entity_val:
                filters[entity_key] = entity_val
                deprecated_keys.append(entity_key)
        if deprecated_keys:
            logging.warning(
                "Top-level %s in /search is deprecated. Use filters={%s} instead.",
                ", ".join(deprecated_keys),
                ", ".join(f'"{k}": "..."' for k in deprecated_keys),
            )
        if not filters and not is_bootstrap_admin(operator):
            filters = {"user_id": str(operator.id)}
        filters = check_query_permission(filters, operator.id, db, bypass=is_admin)
        params = {}
        if search_req.top_k is not None:
            params["top_k"] = search_req.top_k
        if search_req.threshold is not None:
            params["threshold"] = search_req.threshold
        if search_req.explain is not None:
            params["explain"] = search_req.explain
        if search_req.show_expired is not None:
            params["show_expired"] = search_req.show_expired
        return get_memory_instance().search(query=search_req.query, filters=filters, **params)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception:
        raise upstream_error()


@app.put("/memories/{memory_id}", summary="Update a memory")
def update_memory(
    memory_id: str,
    updated_memory: MemoryUpdate,
    request: Request,
    auth=Depends(verify_auth),
    db: Session = Depends(get_db),
):
    """Update an existing memory."""
    operator, is_admin = resolve_operator(request, auth, db)
    reject_bootstrap_memory_mutation(operator)
    scope = resolve_memory_entities(memory_id)
    check_memory_scope_permission(scope, operator.id, "write", db, bypass=is_admin)
    try:
        fields_set = getattr(updated_memory, "model_fields_set", getattr(updated_memory, "__fields_set__", set()))
        params = {"memory_id": memory_id}
        if "text" in fields_set:
            params["data"] = updated_memory.text
        if "metadata" in fields_set:
            params["metadata"] = updated_memory.metadata
        if "expiration_date" in fields_set:
            params["expiration_date"] = updated_memory.expiration_date
        return run_memory_write_for_memory_id(
            lambda memory: memory.update(**params),
            memory_id,
        )
    except (ValueError, Mem0ValidationError) as e:
        raise _client_error(e)
    except HTTPException:
        raise
    except Exception:
        raise upstream_error()


@app.get("/memories/{memory_id}/history", summary="Get memory history")
def memory_history(
    memory_id: str,
    request: Request,
    auth=Depends(verify_auth),
    db: Session = Depends(get_db),
):
    """Retrieve memory history."""
    operator, is_admin = resolve_operator(request, auth, db)
    scope = resolve_memory_entities(memory_id)
    check_memory_scope_permission(scope, operator.id, "read", db, bypass=is_admin)
    try:
        return get_memory_instance().history(memory_id=memory_id)
    except HTTPException:
        raise
    except Exception:
        raise upstream_error()


@app.delete("/memories/{memory_id}", summary="Delete a memory", response_model=MessageResponse)
def delete_memory(
    memory_id: str,
    request: Request,
    auth=Depends(verify_auth),
    db: Session = Depends(get_db),
):
    """Delete a specific memory by ID."""
    operator, is_admin = resolve_operator(request, auth, db)
    reject_bootstrap_memory_mutation(operator)
    scope = resolve_memory_entities(memory_id)
    check_memory_scope_permission(scope, operator.id, "write", db, bypass=is_admin)
    try:
        run_memory_write_for_memory_id(lambda memory: memory.delete(memory_id=memory_id), memory_id)
        return MessageResponse(message="Memory deleted successfully")
    except (ValueError, Mem0ValidationError) as e:
        raise _client_error(e)
    except HTTPException:
        raise
    except Exception:
        raise upstream_error()


@app.delete("/memories", summary="Delete all memories", response_model=MessageResponse)
def delete_all_memories(
    request: Request,
    user_id: Optional[str] = None,
    run_id: Optional[str] = None,
    agent_id: Optional[str] = None,
    app_id: Optional[str] = None,
    auth=Depends(verify_auth),
    db: Session = Depends(get_db),
):
    """Delete all memories for a given identifier.

    Requires admin permission on every matched memory's full scope, so an entity
    admin can delete within their own scope but a cross-scope match fails the batch.
    """
    operator, is_admin = resolve_operator(request, auth, db)
    reject_bootstrap_memory_mutation(operator)
    params = {
        k: v for k, v in {"user_id": user_id, "run_id": run_id, "agent_id": agent_id, "app_id": app_id}.items() if v
    }
    if not params:
        raise HTTPException(status_code=400, detail="At least one identifier is required.")
    # Scope agent/run-only deletes to the caller's user namespace (skip for app-scoped
    # deletes — injecting user_id would narrow the delete scope). Without this, a
    # delete_all(agent_id=riley) prescans every user's same-named agent and the bulk
    # admin check fails the whole batch with 403. Mirrors compat v1 / MCP.
    params = inject_default_user_id(params, operator, skip_if_has_app=True)
    try:
        run_memory_write(
            lambda m: bulk_delete_memories(
                m, params, operator.id, db, bypass=is_admin,
            ),
            entity_scope_from_params(params),
        )
        return MessageResponse(message="All relevant memories deleted")
    except HTTPException:
        raise
    except Exception:
        raise upstream_error()


@app.post("/reset", summary="Reset all memories")
def reset_memory(_auth=Depends(require_admin)):
    """Completely reset stored memories. Requires admin role."""
    try:
        run_memory_write(lambda memory: memory.reset(), global_lock=True)
        return {"message": "All memories reset"}
    except Exception:
        raise upstream_error()


@app.get("/", summary="Redirect to the OpenAPI documentation", include_in_schema=False)
def home():
    """Redirect to the OpenAPI documentation."""
    return RedirectResponse(url="/docs")
