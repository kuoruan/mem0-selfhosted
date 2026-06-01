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
    """Create *model_dir* (if set) and register it on ``sys.path``."""
    if not model_dir:
        return ""
    abs_dir = os.path.abspath(model_dir)
    os.makedirs(abs_dir, exist_ok=True)
    with _lock:
        if abs_dir not in sys.path:
            sys.path.insert(0, abs_dir)
    return abs_dir


def _cache_key(model_name: str, model_dir: str, disable: Optional[tuple[str, ...]]) -> str:
    base = f"{model_dir}:{model_name}" if model_dir else model_name
    if not disable:
        return base
    return f"{base}:{'|'.join(sorted(disable))}"


def _download_model(model_name: str, model_dir: str, download_url: Optional[str]) -> None:
    """Call ``spacy.cli.download``, optionally via *download_url* mirror.

    spaCy 3.8.x has two bugs in ``download_model()`` that make ``custom_url``
    unusable for mirrors: it resets *base_url* to the GitHub default when the
    URL lacks a trailing ``/``, and it validates the final URL against
    ``about.__download_url__``.  We work around both by temporarily patching
    ``about.__download_url__`` to the mirror URL.
    """
    import spacy.about
    from spacy.cli import download

    logger.info("Downloading spaCy model %s...", model_name)
    pip_args = ["--target", model_dir] if model_dir else []
    if download_url:
        saved = spacy.about.__download_url__
        spacy.about.__download_url__ = download_url
        try:
            download(model_name, *pip_args)
        finally:
            spacy.about.__download_url__ = saved
    else:
        download(model_name, *pip_args)
    logger.info("spaCy model %s downloaded successfully", model_name)


def _ensure_model_available(
    model_name: str, *, model_dir: str, download_url: Optional[str], auto_download: bool
) -> None:
    """Raise if the spaCy model is unavailable and *auto_download* is False, otherwise download it."""
    try:
        import spacy
    except ImportError as e:
        raise ImportError(
            "spaCy is not installed. Install it with: pip install mem0ai[nlp]"
        ) from e

    if spacy.util.is_package(model_name) or os.path.exists(model_name):
        return
    if model_dir and os.path.isdir(os.path.join(model_dir, model_name)):
        return

    if not auto_download:
        hint = f"spacy download {model_name}"
        if model_dir:
            hint += f" or place it under {model_dir}"
        raise RuntimeError(
            f"spaCy model '{model_name}' is not installed. Install manually: {hint}"
        )

    try:
        _download_model(model_name, model_dir, download_url)
    except SystemExit as e:
        if e.code not in (0, None):
            raise RuntimeError(
                f"Failed to download spaCy model {model_name} (exit {e.code}). "
                f"Install manually: python -m spacy download {model_name}"
            ) from e
    except Exception as e:
        raise RuntimeError(
            f"Failed to download spaCy model {model_name}: {e}. "
            f"Install manually: python -m spacy download {model_name}"
        ) from e


def _is_load_failed(key: str) -> bool:
    ts = _load_failed.get(key)
    return ts is not None and time.monotonic() - ts < _LOAD_FAILED_TTL


def _load_with_disable(model_name: str, disable: tuple[str, ...]):
    """Load *model_name* with *disable*, filtering to only existing pipes."""
    import spacy

    try:
        meta = spacy.util.get_model_meta(model_name)
        pipeline = meta.get("pipeline", [])
        disable = tuple(c for c in disable if c in pipeline)
    except Exception:
        pass
    try:
        return spacy.load(model_name, disable=list(disable))
    except ValueError:
        nlp = spacy.load(model_name)
        nlp.disable_pipes(*[c for c in disable if c in nlp.pipe_names])
        return nlp


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
        except (ImportError, RuntimeError) as e:
            logger.warning(
                "spaCy model '%s' is not available, NLP features will be disabled. (%s)",
                model_name, e,
            )
            _load_failed[key] = time.monotonic()
            return None

        try:
            import spacy

            nlp = _load_with_disable(model_name, disable) if disable else spacy.load(model_name)
            _nlp_cache[key] = nlp
            logger.info("spaCy model loaded: %s (disable=%s)", model_name, disable)
            return nlp
        except Exception as e:
            logger.warning("Failed to load spaCy model %s: %s", model_name, e)
            _load_failed[key] = time.monotonic()
            return None


def _load_nlp_model(
    nlp_config: Optional[NlpConfig], variant: str, disable: Optional[tuple[str, ...]]
):
    config = nlp_config if nlp_config is not None else _DEFAULT_NLP_CONFIG
    if not config.enabled:
        return None
    return _load_spacy_model(
        config.resolve_model(variant=variant),
        model_dir=_ensure_model_dir(config.model_dir),
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
    """Clear cached models and failure flags (for tests)."""
    global _nlp_cache, _load_failed
    with _lock:
        _nlp_cache = {}
        _load_failed = {}
