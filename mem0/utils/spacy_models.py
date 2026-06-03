"""
Shared spaCy model loader.

Consolidates spaCy model loading into a single module so that
entity_extraction and lemmatization share cached instances per model
instead of each loading their own copy from disk.

When ``auto_download=True`` and a model is missing, it is downloaded
in a **background thread** so that the calling request returns
immediately (with NLP features temporarily disabled).  Subsequent
requests will pick up the model once the download completes.
"""

import importlib
import logging
import os
import subprocess
import sys
import threading
import time
from typing import Any, Optional

from mem0.configs.nlp.config import NlpConfig

logger = logging.getLogger(__name__)

_DEFAULT_NLP_CONFIG = NlpConfig()

_lock = threading.RLock()

# Successfully loaded models: cache_key → nlp instance.
_nlp_cache: dict[str, Any] = {}

# Failed loads: cache_key → timestamp. Automatically retried after TTL.
_LOAD_FAILED_TTL = 300.0
_load_failed: dict[str, float] = {}

# Keys currently being downloaded by a background thread.
_downloading: set[str] = set()

# Lazy-imported spaCy module: None = not yet resolved, False = not installed.
_spacy: Any = None


def _get_spacy():
    """Return the ``spacy`` module, or ``None`` if it is not installed.

    The result is cached after the first call so ``ImportError`` is only
    raised once.
    """
    global _spacy
    # Fast path: already resolved (module or sentinel).
    if _spacy is not None:
        return _spacy or None
    with _lock:
        if _spacy is not None:
            return _spacy or None
        try:
            import spacy as _spacy_mod

            _spacy = _spacy_mod
            return _spacy_mod
        except ImportError:
            _spacy = False
            return None


def _ensure_model_dir(model_dir: Optional[str]) -> str:
    """Create *model_dir* (if set) and register it on ``sys.path``."""
    if not model_dir:
        return ""
    abs_dir = os.path.abspath(model_dir)
    # Fast path: already registered from a previous call.
    if abs_dir in sys.path:
        return abs_dir
    os.makedirs(abs_dir, exist_ok=True)
    with _lock:
        if abs_dir not in sys.path:
            sys.path.insert(0, abs_dir)
            importlib.invalidate_caches()
    return abs_dir


def _cache_key(model_name: str, model_dir: str, disable: Optional[tuple[str, ...]]) -> str:
    base = f"{model_dir}:{model_name}" if model_dir else model_name
    if not disable:
        return base
    return f"{base}:{'|'.join(sorted(disable))}"


def _dl_key(model_name: str, model_dir: str) -> str:
    """Return a model-level key for download tracking (excludes disable config)."""
    return f"{model_dir}:{model_name}"


def _is_model_available(model_name: str, model_dir: str) -> bool:
    """Return ``True`` if the spaCy model is already installed / on disk."""
    spacy = _get_spacy()
    if spacy is None:
        return False
    if spacy.util.is_package(model_name) or os.path.exists(model_name):
        return True
    if model_dir and os.path.isfile(os.path.join(model_dir, model_name, "meta.json")):
        return True
    return False


def _download_model(model_name: str, model_dir: str, download_url: Optional[str]) -> None:
    """Download a spaCy model via pip, optionally from *download_url* mirror.

    Resolves the model version and wheel filename from spaCy's compatibility
    table, then installs via ``pip install``.  Bypasses ``spacy.cli.download``
    entirely — this avoids URL validation that breaks GitHub proxy mirrors
    (``urljoin`` normalises ``//`` → ``/``, failing the ``startswith`` check)
    and avoids version-dependent function signatures (``custom_url`` parameter
    was added in spaCy 3.8.8).
    """
    logger.info("Downloading spaCy model %s...", model_name)

    try:
        import spacy.about
        from spacy.cli.download import get_compatibility, get_model_filename, get_version

        compatibility = get_compatibility()
        version = get_version(model_name, compatibility)
        filename = get_model_filename(model_name, version)
    except (Exception, SystemExit) as e:
        raise RuntimeError(
            f"Failed to resolve spaCy model download for {model_name}: {e}"
        ) from e

    base = download_url or spacy.about.__download_url__
    full_url = f"{base.rstrip('/')}/{filename}"

    cmd = [sys.executable, "-m", "pip", "install", "--upgrade"]
    if model_dir:
        cmd.extend(["--target", model_dir])
    cmd.append(full_url)

    logger.info("pip install %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        error_msg = result.stderr.strip() or result.stdout.strip() or "No output"
        raise RuntimeError(
            f"pip install failed (exit {result.returncode}): {error_msg}"
        )

    logger.info("spaCy model %s downloaded successfully", model_name)


def _background_download(
    cache_key: str,
    dl_key: str,
    model_name: str,
    model_dir: str,
    download_url: Optional[str],
    disable: Optional[tuple[str, ...]],
) -> None:
    """Download and load a spaCy model in a background thread."""
    try:
        _download_model(model_name, model_dir, download_url)
        importlib.invalidate_caches()

        nlp = _load_model(model_name, disable)
        with _lock:
            _nlp_cache[cache_key] = nlp
        logger.info("spaCy model loaded in background: %s (disable=%s)", model_name, disable)
    except Exception as e:
        logger.warning("Background download/load of spaCy model %s failed: %s", model_name, e)
        _mark_download_failed(cache_key, dl_key)
    finally:
        with _lock:
            _downloading.discard(dl_key)


def _is_load_failed(key: str) -> bool:
    """Return ``True`` if *key* is cached as failed and not yet expired."""
    ts = _load_failed.get(key)
    if ts is None:
        return False
    return time.monotonic() - ts < _LOAD_FAILED_TTL


def _mark_download_failed(cache_key: str, dl_key: str) -> None:
    """Record a download/load failure and clear the in-progress flag."""
    with _lock:
        _downloading.discard(dl_key)
        _load_failed[cache_key] = time.monotonic()


def _load_model(model_name: str, disable: Optional[tuple[str, ...]] = None):
    """Load *model_name*, optionally disabling pipeline components.

    When *disable* is provided, the list is filtered to only include
    pipeline components that actually exist in the model.  If
    ``spacy.load(..., disable=...)`` raises ``ValueError``, the model
    is loaded without disable and pipes are disabled post-load.
    """
    spacy = _get_spacy()

    if not disable:
        return spacy.load(model_name)

    filtered: tuple[str, ...] = disable
    try:
        meta = spacy.util.get_model_meta(model_name)
        pipeline = meta.get("pipeline", [])
        filtered = tuple(c for c in disable if c in pipeline)
    except (ValueError, KeyError, ImportError, OSError):
        pass
    try:
        return spacy.load(model_name, disable=list(filtered))
    except ValueError:
        nlp = spacy.load(model_name)
        nlp.disable_pipes(*[c for c in filtered if c in nlp.pipe_names])
        return nlp


def _load_spacy_model(
    model_name: str,
    *,
    model_dir: Optional[str],
    download_url: Optional[str],
    disable: Optional[tuple[str, ...]],
    auto_download: bool,
):
    """Load (and optionally download) a spaCy model, caching the result.

    When the model is missing and ``auto_download`` is True, a background
    thread is started to download it.  The function returns ``None``
    immediately so the caller can proceed without NLP features.  Once the
    download completes, subsequent calls will return the cached model.
    """
    key = _cache_key(model_name, model_dir or "", disable)
    dl_key = _dl_key(model_name, model_dir or "")

    # Fast path (no lock).
    if _is_load_failed(key):
        return None
    if key in _nlp_cache:
        return _nlp_cache[key]

    with _lock:
        # Purge expired failure entries so they become retryable.
        now = time.monotonic()
        for k, ts in list(_load_failed.items()):
            if now - ts >= _LOAD_FAILED_TTL:
                del _load_failed[k]
        # Double-check under lock.
        if _is_load_failed(key):
            return None
        if key in _nlp_cache:
            return _nlp_cache[key]

        # Already downloading in background?
        if dl_key in _downloading:
            return None

        # Ensure spaCy is installed before spawning a download thread.
        if _get_spacy() is None:
            logger.warning(
                "spaCy is not installed. NLP features will be disabled. "
                "Install it with: pip install mem0ai[nlp]"
            )
            _load_failed[key] = time.monotonic()
            return None

        # Prepare model directory.
        try:
            actual_dir = _ensure_model_dir(model_dir)
        except OSError as e:
            logger.warning("Cannot create model_dir %s: %s", model_dir, e)
            _load_failed[key] = time.monotonic()
            return None

        # Model already on disk → try to load synchronously.
        if _is_model_available(model_name, actual_dir):
            try:
                nlp = _load_model(model_name, disable)
                _nlp_cache[key] = nlp
                logger.info("spaCy model loaded: %s (disable=%s)", model_name, disable)
                return nlp
            except Exception as e:
                logger.warning("Failed to load spaCy model %s: %s", model_name, e)
                if not auto_download:
                    _load_failed[key] = time.monotonic()
                    return None
                # Model exists but incompatible (version mismatch?) →
                # fall through to re-download.

        # Model not available and auto_download disabled.
        if not auto_download:
            hint = f"python -m spacy download {model_name}"
            if model_dir:
                hint += f" or place it under {model_dir}"
            logger.warning(
                "spaCy model '%s' is not installed and auto_download is disabled. "
                "Install manually: %s",
                model_name, hint,
            )
            _load_failed[key] = time.monotonic()
            return None

        # Start background download.
        _downloading.add(dl_key)
        logger.info(
            "Starting background download of spaCy model %s (model_dir=%s)...",
            model_name, actual_dir or "<default>",
        )
        thread = threading.Thread(
            target=_background_download,
            args=(key, dl_key, model_name, actual_dir, download_url, disable),
            daemon=True,
            name=f"spacy-download-{model_name}",
        )
        thread.start()
        return None


def _load_nlp_model(
    nlp_config: Optional[NlpConfig], variant: str, disable: Optional[tuple[str, ...]]
):
    config = nlp_config if nlp_config is not None else _DEFAULT_NLP_CONFIG
    if not config.enabled:
        return None
    return _load_spacy_model(
        config.resolve_model(variant=variant),
        model_dir=config.model_dir,
        download_url=config.download_url,
        disable=disable,
        auto_download=config.auto_download,
    )


def get_nlp_full(nlp_config: Optional[NlpConfig] = None):
    """Return spaCy model with all pipelines (NER, tagger, etc.) for entity extraction."""
    return _load_nlp_model(nlp_config, "full", None)


def get_nlp_lemma(nlp_config: Optional[NlpConfig] = None):
    """Return spaCy model with NER/parser disabled for BM25 text processing."""
    return _load_nlp_model(nlp_config, "lemma", ("ner", "parser"))


def reset_spacy_cache() -> None:
    """Clear cached models, failure flags, and spacy module state (for tests)."""
    global _spacy
    with _lock:
        _nlp_cache.clear()
        _load_failed.clear()
        _downloading.clear()
        _spacy = None
