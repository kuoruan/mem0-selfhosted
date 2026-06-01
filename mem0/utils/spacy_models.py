"""
Shared spaCy model loader.

Consolidates spaCy model loading into a single module so that
entity_extraction and lemmatization share cached instances per model
instead of each loading their own copy from disk.
"""

import logging
import os
import sys
import threading
import time
from typing import Any, Optional

from mem0.configs.nlp.config import NlpConfig

logger = logging.getLogger(__name__)

_DEFAULT_NLP_CONFIG = NlpConfig()
_lock = threading.Lock()
_nlp_cache: dict[str, Any] = {}
# Failed loads: cache_key -> timestamp. Entries expire after _LOAD_FAILED_TTL seconds.
_load_failed: dict[str, float] = {}
_LOAD_FAILED_TTL = 300.0


def _ensure_model_dir(model_dir: Optional[str]) -> str:
    """Create *model_dir* (if set) and register it on ``sys.path``.

    When a ``model_dir`` is configured, spaCy models are downloaded there via
    ``pip install --target`` and loaded via ``spacy.load(model_name)`` which
    finds the package through ``is_package()`` → ``importlib.import_module()``
    because *model_dir* is on ``sys.path``.
    """
    if not model_dir:
        return ""
    os.makedirs(model_dir, exist_ok=True)
    if model_dir not in sys.path:
        sys.path.insert(0, model_dir)
    return model_dir


def _cache_key(model_name: str, model_dir: str, disable: Optional[tuple[str, ...]]) -> str:
    base = f"{model_dir}:{model_name}" if model_dir else model_name
    if not disable:
        return base
    return f"{base}:{'|'.join(sorted(disable))}"


def _ensure_model_available(
    model_name: str, *, model_dir: str, download_url: Optional[str], auto_download: bool
) -> None:
    """Download the spaCy model if it is not already available anywhere."""
    # 1. Already installed as a Python package (site-packages or model_dir via sys.path).
    try:
        import spacy

        if spacy.util.is_package(model_name):
            return
    except ImportError as e:
        raise ImportError(
            "spaCy is not installed. Install it with: pip install mem0ai[nlp]"
        ) from e

    # 2. Already on disk — explicit local path (e.g. model="/path/to/model").
    if os.path.exists(model_name):
        return
    # 3. Check the model_dir in case sys.path hasn't picked it up yet.
    if model_dir and os.path.isdir(os.path.join(model_dir, model_name)):
        return

    if not auto_download:
        hint = (
            f"spacy download {model_name}"
            if not model_dir
            else f"spacy download {model_name} or place it under {model_dir}"
        )
        raise RuntimeError(
            f"spaCy model '{model_name}' is not installed. Install manually: {hint}"
        )

    logger.info("Downloading spaCy model %s...", model_name)
    try:
        from spacy.cli import download

        if model_dir:
            # All positional so "--target" and model_dir land in *pip_args.
            download(model_name, False, False, download_url, "--target", model_dir)
        else:
            download(model_name, False, False, download_url)
        logger.info("spaCy model %s downloaded successfully", model_name)
    except SystemExit as e:
        if e.code not in (0, None):
            raise RuntimeError(
                f"Failed to download spaCy model {model_name}: {e}. "
                f"Please install manually: python -m spacy download {model_name}"
            ) from e
    except Exception as e:
        raise RuntimeError(
            f"Failed to download spaCy model {model_name}: {e}. "
            f"Please install manually: python -m spacy download {model_name}"
        ) from e


def _resolve_nlp_config(nlp_config: Optional[NlpConfig]) -> NlpConfig:
    return nlp_config if nlp_config is not None else _DEFAULT_NLP_CONFIG


def _is_load_failed(key: str) -> bool:
    ts = _load_failed.get(key)
    return ts is not None and time.monotonic() - ts < _LOAD_FAILED_TTL


def _load_spacy_model(
    model_name: str,
    *,
    model_dir: str,
    download_url: Optional[str],
    disable: Optional[tuple[str, ...]],
    auto_download: bool,
):
    """Load (and optionally download) a spaCy model, caching the result."""
    key = _cache_key(model_name, model_dir, disable)
    if _is_load_failed(key):
        return None
    if key in _nlp_cache:
        return _nlp_cache[key]

    with _lock:
        if _is_load_failed(key):
            return None
        if key in _nlp_cache:
            return _nlp_cache[key]

        try:
            _ensure_model_available(
                model_name, model_dir=model_dir,
                download_url=download_url, auto_download=auto_download,
            )
            import spacy
        except (ImportError, RuntimeError) as e:
            logger.warning(
                "spaCy model '%s' is not available, NLP features will be disabled. (%s)",
                model_name, e,
            )
            _load_failed[key] = time.monotonic()
            return None

        try:
            if disable:
                try:
                    meta = spacy.util.get_model_meta(model_name)
                    pipeline = meta.get("pipeline", [])
                    actual_disable = [c for c in disable if c in pipeline]
                except Exception:
                    actual_disable = disable
                try:
                    nlp = spacy.load(model_name, disable=actual_disable)
                except ValueError:
                    nlp = spacy.load(model_name)
                    nlp.disable_pipes(*[c for c in disable if c in nlp.pipe_names])
            else:
                nlp = spacy.load(model_name)
            _nlp_cache[key] = nlp
            logger.info("spaCy model loaded: %s (disable=%s)", model_name, disable)
            return nlp
        except Exception as e:
            logger.warning("Failed to load spaCy model %s: %s", model_name, e)
            _load_failed[key] = time.monotonic()
            return None


def get_nlp_full(nlp_config: Optional[NlpConfig] = None):
    """Return spaCy model with all pipelines (NER, tagger, etc.) for entity extraction."""
    config = _resolve_nlp_config(nlp_config)
    if not config.enabled:
        return None
    model_dir = _ensure_model_dir(config.model_dir)
    model_name = config.resolve_model(variant="full")
    return _load_spacy_model(
        model_name, model_dir=model_dir, download_url=config.download_url,
        disable=None, auto_download=config.auto_download,
    )


def get_nlp_lemma(nlp_config: Optional[NlpConfig] = None):
    """Return spaCy model with NER/parser disabled for BM25 text processing."""
    config = _resolve_nlp_config(nlp_config)
    if not config.enabled:
        return None
    model_dir = _ensure_model_dir(config.model_dir)
    model_name = config.resolve_model(variant="lemma")
    return _load_spacy_model(
        model_name, model_dir=model_dir, download_url=config.download_url,
        disable=("ner", "parser"), auto_download=config.auto_download,
    )


def reset_spacy_cache() -> None:
    """Clear cached models and failure flags (for tests)."""
    global _nlp_cache, _load_failed
    with _lock:
        _nlp_cache = {}
        _load_failed = {}
