"""
Shared spaCy model loader.

Consolidates spaCy model loading into a single module so that
entity_extraction and lemmatization share cached instances per model
instead of each loading their own copy from disk.
"""

import logging
import os
import threading
from typing import Any, Optional

from mem0.configs.nlp.config import NlpConfig

logger = logging.getLogger(__name__)

_DEFAULT_NLP_CONFIG = NlpConfig()
_lock = threading.Lock()
# Successful loads only; failures are tracked in _load_failed.
_nlp_cache: dict[str, Any] = {}
_load_failed: set[str] = set()


def _cache_key(model_name: str, disable: Optional[tuple[str, ...]]) -> str:
    if not disable:
        return model_name
    return f"{model_name}:{'|'.join(sorted(disable))}"


def _ensure_model_available(model_name: str, *, auto_download: bool) -> None:
    """Download the spaCy model if installed but package is missing."""
    try:
        import spacy
    except ImportError as e:
        raise ImportError(
            "spaCy is not installed. Install it with: pip install mem0ai[nlp]"
        ) from e

    if os.path.exists(model_name) or spacy.util.is_package(model_name):
        return

    if not auto_download:
        raise RuntimeError(
            f"spaCy model '{model_name}' is not installed. "
            f"Install manually: python -m spacy download {model_name}"
        )

    logger.info("Downloading spaCy model %s...", model_name)
    try:
        from spacy.cli import download

        download(model_name)
        logger.info("spaCy model %s downloaded successfully", model_name)
    except (Exception, SystemExit) as e:
        raise RuntimeError(
            f"Failed to download spaCy model {model_name}: {e}. "
            f"Please install manually: python -m spacy download {model_name}"
        ) from e


def _resolve_nlp_config(nlp_config: Optional[NlpConfig]) -> NlpConfig:
    return nlp_config if nlp_config is not None else _DEFAULT_NLP_CONFIG


def _load_spacy_model(model_name: str, *, disable: Optional[tuple[str, ...]], auto_download: bool):
    key = _cache_key(model_name, disable)
    if key in _load_failed:
        return None
    if key in _nlp_cache:
        return _nlp_cache[key]

    with _lock:
        if key in _load_failed:
            return None
        if key in _nlp_cache:
            return _nlp_cache[key]

        try:
            _ensure_model_available(model_name, auto_download=auto_download)

            import spacy

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
                    for c in disable:
                        if c in nlp.pipe_names:
                            nlp.disable_pipe(c)
            else:
                nlp = spacy.load(model_name)
            _nlp_cache[key] = nlp
            logger.info("spaCy model loaded: %s (disable=%s)", model_name, disable)
            return nlp
        except Exception as e:
            logger.warning("Failed to load spaCy model %s: %s", model_name, e)
            _load_failed.add(key)
            return None


def get_nlp_full(nlp_config: Optional[NlpConfig] = None):
    """Return spaCy model with all pipelines (NER, tagger, etc.) for entity extraction."""
    config = _resolve_nlp_config(nlp_config)
    if not config.enabled:
        return None
    model_name = config.resolve_model(variant="full")
    return _load_spacy_model(model_name, disable=None, auto_download=config.auto_download)


def get_nlp_lemma(nlp_config: Optional[NlpConfig] = None):
    """Return spaCy model with NER/parser disabled for BM25 text processing."""
    config = _resolve_nlp_config(nlp_config)
    if not config.enabled:
        return None
    model_name = config.resolve_model(variant="lemma")
    return _load_spacy_model(model_name, disable=("ner", "parser"), auto_download=config.auto_download)


def reset_spacy_cache() -> None:
    """Clear cached models and failure flags (for tests)."""
    global _nlp_cache, _load_failed
    with _lock:
        _nlp_cache = {}
        _load_failed = set()
