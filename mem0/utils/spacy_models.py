"""
Shared spaCy model manager.

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
from functools import cache
from types import ModuleType
from typing import Any, Optional

from mem0.configs.nlp.config import NlpConfig

logger = logging.getLogger(__name__)

_DEFAULT_NLP_CONFIG = NlpConfig()

# How long before a failed load is retried (seconds).
_LOAD_FAILED_TTL = 300.0


# ---------------------------------------------------------------------------
# spaCy module resolution (memoised, no manager state involved)
# ---------------------------------------------------------------------------


@cache
def _resolve_spacy() -> ModuleType | None:
    """Return the ``spacy`` module, or ``None`` if unavailable.

    Pure memoisation: resolved once per process, cached by Python's
    ``functools.cache``.  Thread-safe in practice because Python's
    import system is process-global and idempotent — concurrent calls
    may all execute the import but always obtain the same module object.
    """
    try:
        import spacy

        return spacy
    except ImportError:
        return None


class SpacyModelManager:
    """Thread-safe manager for spaCy model loading, caching, and downloads.

    Encapsulates all mutable state for spaCy model management:
    loaded model cache, failure tracking, download coordination, and
    compatibility/version lookups.  Designed as a process-level singleton
    (see module-level ``_manager``) so that all callers share cached instances.

    Lifecycle:
        - State is initialised lazily on first use.
        - ``reset()`` clears all caches in-place (for test isolation).
        - A fresh instance can be created if a truly clean slate is needed.
    """

    def __init__(self) -> None:
        # RLock required because ensure_model_dir()
        # may re-enter _lock from load_spacy_model() Phase 1.
        self._lock = threading.RLock()
        self._compat_lock = threading.Lock()

        # Successfully loaded models: cache_key → nlp instance.
        self._nlp_cache: dict[str, Any] = {}

        # Failed loads: cache_key → timestamp.  Automatically retried after TTL.
        self._load_failed: dict[str, float] = {}

        # Keys currently being downloaded by a background thread.
        self._downloading: set[str] = set()

        # Cached compatibility table from ``get_compatibility()``.
        self._compatibility_cache: dict[str, Any] | None = None

        # Cached expected versions (only successful results): model_name → version string.
        self._expected_version_cache: dict[str, str] = {}
        # Cached model metadata (only successful results): model_name → meta dict.
        self._model_meta_cache: dict[str, dict] = {}

    # ---- spaCy module resolution ----

    @staticmethod
    def get_spacy() -> ModuleType | None:
        """Return the ``spacy`` module, or ``None`` if it is not installed."""
        return _resolve_spacy()

    # ---- Model directory ----

    def ensure_model_dir(self, model_dir: Optional[str]) -> str:
        """Create *model_dir* (if set) and register it on ``sys.path``."""
        if not model_dir:
            return ""
        abs_dir = os.path.abspath(model_dir)
        # Ensure directory exists even if already on sys.path (it may
        # have been deleted between calls).
        os.makedirs(abs_dir, exist_ok=True)
        if abs_dir in sys.path:
            return abs_dir
        with self._lock:
            if abs_dir not in sys.path:
                sys.path.insert(0, abs_dir)
                importlib.invalidate_caches()
        return abs_dir

    # ---- Cache key helpers ----

    @staticmethod
    def cache_key(model_name: str, model_dir: str, disable: Optional[tuple[str, ...]]) -> str:
        base = f"{model_dir}:{model_name}" if model_dir else model_name
        if not disable:
            return base
        return f"{base}:{'|'.join(sorted(disable))}"

    @staticmethod
    def dl_key(model_name: str, model_dir: str) -> str:
        """Model-level key for download tracking (excludes ``disable`` config)."""
        return f"{model_dir}:{model_name}"

    # ---- Version / compatibility ----

    def get_expected_version(self, model_name: str) -> Optional[str]:
        """Return the version expected by the current spaCy installation.

        Uses ``get_compatibility()`` (HTTP) outside the global ``_lock`` so
        it is never held during a network call.  A dedicated ``_compat_lock``
        uses double-checked locking to ensure only one thread fetches the
        compatibility table.  The table is cached globally so different
        models share one HTTP request.  Only successful results are cached
        — transient failures are retried.
        """
        # Lock-free fast path for the common case (version already resolved).
        version = self._expected_version_cache.get(model_name)
        if version is not None:
            return version

        if self.get_spacy() is None:
            return None

        # Import outside any lock — Python caches module imports after the
        # first call, but the initial import may touch disk.
        try:
            from spacy.cli.download import get_compatibility
        except ImportError:
            return None

        # Fetch compatibility table outside the global lock to avoid blocking
        # other threads during the HTTP call.
        compat = self._compatibility_cache
        if compat is None:
            with self._compat_lock:
                compat = self._compatibility_cache
                if compat is None:
                    try:
                        temp = get_compatibility()
                        if temp:
                            self._compatibility_cache = temp
                            compat = temp
                    except Exception as e:
                        logger.warning("Failed to fetch compatibility table: %s", e)

        try:
            from spacy.cli.download import get_version
        except ImportError as e:
            logger.warning("Failed to import spacy.cli.download.get_version: %s", e)
            return None

        with self._lock:
            version = self._expected_version_cache.get(model_name)
            if version is not None:
                return version

            # Re-check the global cache inside the lock in case another thread
            # fetched the table after our lock-free read above.
            if not compat:
                compat = self._compatibility_cache
            if compat:
                try:
                    version = get_version(model_name, compat)
                except (Exception, SystemExit) as e:
                    logger.warning("Failed to resolve expected version for %s: %s", model_name, e)

            if version is not None:
                self._expected_version_cache[model_name] = version
            return version

    def get_model_meta(self, model_name: str) -> Optional[dict]:
        """Return the spaCy model meta dict for *model_name*, with caching.

        Only successful results are cached — failures (e.g. model not yet
        installed) are retried on subsequent calls.

        Thread-safe: fast-path read is lock-free; cache miss acquires ``_lock``
        to prevent duplicate disk reads from concurrent threads.
        """
        # Lock-free fast path for the common case.
        meta = self._model_meta_cache.get(model_name)
        if meta is not None:
            return meta

        spacy = self.get_spacy()
        if spacy is None:
            return None

        with self._lock:
            # Double-check under lock.
            cached_meta = self._model_meta_cache.get(model_name)
            if cached_meta is not None:
                return cached_meta

            # Read meta inside the lock to prevent concurrent duplicate
            # disk reads.
            try:
                meta = spacy.util.get_model_meta(model_name)
            except Exception:
                meta = None

            if meta is not None:
                self._model_meta_cache[model_name] = meta
            return meta

    def check_model_version(self, model_name: str) -> bool:
        """Return ``True`` if the installed *model_name* version matches the expected one.

        Returns ``True`` when the expected version cannot be resolved (assume OK).
        Returns ``False`` when the expected version is known but the installed
        metadata cannot be read — this indicates a corrupt or unreadable model
        that should be re-downloaded.
        """
        expected = self.get_expected_version(model_name)
        if expected is None:
            return True
        meta = self.get_model_meta(model_name)
        if meta is None:
            return False
        installed = meta.get("version", "")
        if installed and installed != expected:
            logger.info(
                "Model %s version mismatch: installed=%s, expected=%s",
                model_name,
                installed,
                expected,
            )
            return False
        return True

    # ---- Model availability ----

    def is_model_available(self, model_name: str, model_dir: str) -> bool:
        """Return ``True`` if the spaCy model is already installed / on disk."""
        spacy = self.get_spacy()
        if spacy is None:
            return False
        if spacy.util.is_package(model_name) or os.path.exists(model_name):
            return True
        if model_dir and os.path.isfile(os.path.join(model_dir, model_name, "meta.json")):
            return True
        return False

    # ---- Download ----

    def _download_model(self, model_name: str, model_dir: str, download_url: Optional[str]) -> None:
        """Download a spaCy model via pip, optionally from *download_url* mirror.

        Resolves the model version and wheel filename from spaCy's compatibility
        table, then installs via ``pip install``.  Bypasses ``spacy.cli.download``
        entirely — this avoids URL validation that breaks GitHub proxy mirrors
        (``urljoin`` normalises ``//`` → ``/``, failing the ``startswith`` check)
        and avoids version-dependent function signatures (``custom_url`` parameter
        was added in spaCy 3.8.8).
        """

        spacy = self.get_spacy()
        if spacy is None:
            raise RuntimeError("spaCy is not installed")

        logger.info("Downloading spaCy model %s...", model_name)

        try:
            from spacy.cli.download import get_model_filename

            version = self.get_expected_version(model_name)
            if version is None:
                raise RuntimeError(f"Cannot resolve expected version for {model_name}")
            filename = get_model_filename(model_name, version)
        except (Exception, SystemExit) as e:
            raise RuntimeError(
                f"Failed to resolve spaCy model download for {model_name}: {e}"
            ) from e

        base = download_url or spacy.about.__download_url__
        full_url = f"{base.rstrip('/')}/{filename}"

        cmd = [sys.executable, "-m", "pip", "install", "--upgrade", "--upgrade-strategy", "only-if-needed"]
        if model_dir:
            cmd.extend(["--target", model_dir])
        cmd.append(full_url)

        logger.info("Installing spaCy model %s into %s", model_name, model_dir or "<default>")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            error_msg = result.stderr.strip() or result.stdout.strip() or "No output"
            raise RuntimeError(f"pip install failed (exit {result.returncode}): {error_msg}")

        logger.info("spaCy model %s downloaded successfully", model_name)

    def _background_download(
        self,
        cache_key: str,
        dl_key: str,
        model_name: str,
        model_dir: str,
        download_url: Optional[str],
        disable: Optional[tuple[str, ...]],
    ) -> None:
        """Download and load a spaCy model in a background thread."""
        try:
            self._download_model(model_name, model_dir, download_url)
            importlib.invalidate_caches()
            # Purge stale meta cache from version-check failure so the
            # re-downloaded model is queried fresh (pipeline may differ).
            # Also clear failure state so the model is usable immediately.
            with self._lock:
                self._model_meta_cache.pop(model_name, None)
                self._load_failed.pop(cache_key, None)

            nlp = self._load_model(model_name, disable)
            with self._lock:
                self._nlp_cache[cache_key] = nlp
            logger.info("spaCy model loaded in background: %s (disable=%s)", model_name, disable)
        except Exception as e:
            logger.warning("Background download/load of spaCy model %s failed: %s", model_name, e)
            self._mark_download_failed(cache_key)
        finally:
            with self._lock:
                self._downloading.discard(dl_key)

    # ---- Failure tracking ----

    def is_load_failed(self, key: str) -> bool:
        """Return ``True`` if *key* is cached as failed and not yet expired.

        Lock-free best-effort check.  Benign races are acceptable: a
        concurrent clear/expire may cause a stale ``True`` or ``False``,
        but correctness is eventually consistent.
        """
        ts = self._load_failed.get(key)
        if ts is None:
            return False
        return time.monotonic() - ts < _LOAD_FAILED_TTL

    def _mark_download_failed(self, cache_key: str) -> None:
        """Record a download/load failure."""
        with self._lock:
            self._load_failed[cache_key] = time.monotonic()

    # ---- Model loading ----

    def _load_model(self, model_name: str, disable: Optional[tuple[str, ...]] = None):
        """Load *model_name*, optionally disabling pipeline components.

        When *disable* is provided, the list is first tried as-is.  Only
        on ``ValueError`` (component not in model) does the method fall
        back to reading model metadata to filter the list — this avoids
        an extra ``meta.json`` disk read on the hot path.
        """
        spacy = self.get_spacy()

        if not disable:
            return spacy.load(model_name)

        try:
            return spacy.load(model_name, disable=list(disable))
        except ValueError:
            # One or more disabled components don't exist in this model.
            # Read metadata to filter, then retry.
            pass

        filtered: tuple[str, ...] = disable
        meta = self.get_model_meta(model_name)
        if meta is not None:
            pipeline = meta.get("pipeline", [])
            filtered = tuple(c for c in disable if c in pipeline)

        try:
            return spacy.load(model_name, disable=list(filtered))
        except ValueError:
            nlp = spacy.load(model_name)
            nlp.disable_pipes(*[c for c in filtered if c in nlp.pipe_names])
            return nlp

    # ---- Main orchestrator ----

    def load_spacy_model(
        self,
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

        Locking strategy: the global ``_lock`` is only held briefly for reading
        or writing shared caches.  Slow I/O (model loading, version checks)
        is performed outside the global lock to minimise contention.
        """
        key = self.cache_key(model_name, model_dir or "", disable)
        dl_key = self.dl_key(model_name, model_dir or "")
        actual_dir = ""  # Set in Phase 1; used in Phase 4.

        # Fast path (no lock).
        if self.is_load_failed(key):
            return None
        if key in self._nlp_cache:
            return self._nlp_cache[key]

        # Phase 1: Read shared state under lock — no slow I/O.
        with self._lock:
            # Lazy purge: only check the current key's TTL instead of
            # scanning the entire dict on every call.
            ts = self._load_failed.get(key)
            if ts is not None and time.monotonic() - ts >= _LOAD_FAILED_TTL:
                del self._load_failed[key]
            # Double-check under lock.
            if self.is_load_failed(key):
                return None
            if key in self._nlp_cache:
                return self._nlp_cache[key]

            # Already downloading in background?
            if dl_key in self._downloading:
                return None

            # Ensure spaCy is installed before spawning a download thread.
            if self.get_spacy() is None:
                logger.warning(
                    "spaCy is not installed. NLP features will be disabled. "
                    "Install it with: pip install mem0ai[nlp]"
                )
                self._load_failed[key] = time.monotonic()
                return None

            # Prepare model directory.
            try:
                actual_dir = self.ensure_model_dir(model_dir)
            except OSError as e:
                logger.warning("Cannot create model_dir %s: %s", model_dir, e)
                self._load_failed[key] = time.monotonic()
                return None

            model_on_disk = self.is_model_available(model_name, actual_dir)

        # Phase 2: Try loading existing model (outside lock).
        is_compatible = True
        if model_on_disk:
            result = self._try_load_from_disk(model_name, key, disable)
            if result is not None:
                return result
            # Loading failed — check if version mismatch explains it.
            is_compatible = self.check_model_version(model_name)

        # Phase 3+4: Decide next step and optionally start background download.
        return self._decide_and_download(
            key, dl_key, model_name, actual_dir, download_url,
            disable, auto_download, model_on_disk, is_compatible,
        )

    def _try_load_from_disk(
        self, model_name: str, key: str, disable: Optional[tuple[str, ...]]
    ):
        """Phase 2: Attempt to load model from disk.

        Returns the loaded nlp instance on success, or ``None`` on failure
        (caller should proceed to version check / download).
        """
        try:
            nlp = self._load_model(model_name, disable)
            with self._lock:
                # Another thread may have cached it first — reuse that
                # instance so our duplicate can be garbage collected.
                if key in self._nlp_cache:
                    nlp = self._nlp_cache[key]
                else:
                    self._nlp_cache[key] = nlp
            logger.info("spaCy model loaded: %s (disable=%s)", model_name, disable)
            return nlp
        except Exception as e:
            logger.warning("Failed to load spaCy model %s: %s", model_name, e)
            return None

    def _decide_and_download(
        self,
        key: str,
        dl_key: str,
        model_name: str,
        actual_dir: str,
        download_url: Optional[str],
        disable: Optional[tuple[str, ...]],
        auto_download: bool,
        model_on_disk: bool,
        is_compatible: bool,
    ):
        """Phase 3+4: Update shared state and decide next step.

        If a background download is warranted, mark the key as downloading
        and spawn the thread.
        """
        with self._lock:
            # Another thread may have progressed while we were outside the lock.
            if key in self._nlp_cache:
                return self._nlp_cache[key]
            if dl_key in self._downloading:
                return None

            # Model on disk but version matches — re-download is futile
            # because pip's only-if-needed strategy won't replace the same version.
            if model_on_disk and is_compatible:
                self._load_failed[key] = time.monotonic()
                return None

            # auto_download disabled — log a helpful hint and bail out.
            if not auto_download:
                if model_on_disk:
                    reason = "version is incompatible with current spaCy"
                    extra = f" or re-install under {actual_dir}" if actual_dir else ""
                else:
                    reason = "is not installed and auto_download is disabled"
                    extra = f" or place it under {actual_dir}" if actual_dir else ""
                logger.warning(
                    "spaCy model '%s' %s. Install manually: python -m spacy download %s%s",
                    model_name,
                    reason,
                    model_name,
                    extra,
                )
                self._load_failed[key] = time.monotonic()
                return None

            # auto_download enabled and (version mismatch or model missing)
            # → fall through to start background download.
            self._downloading.add(dl_key)

        # Phase 4: Start background download (outside lock).
        return self._start_background_download(
            key, dl_key, model_name, actual_dir, download_url, disable,
        )

    def _start_background_download(
        self,
        key: str,
        dl_key: str,
        model_name: str,
        actual_dir: str,
        download_url: Optional[str],
        disable: Optional[tuple[str, ...]],
    ):
        """Phase 4: Spawn a background thread to download and load the model."""
        logger.info(
            "Starting background download of spaCy model %s (model_dir=%s)...",
            model_name,
            actual_dir or "<default>",
        )
        thread = threading.Thread(
            target=self._background_download,
            args=(key, dl_key, model_name, actual_dir, download_url, disable),
            daemon=True,
            name=f"spacy-download-{model_name}",
        )
        try:
            thread.start()
        except Exception:
            with self._lock:
                self._downloading.discard(dl_key)
            raise
        return None

    # ---- Reset ----

    # Lock ordering invariant: ``_compat_lock`` must always be acquired
    # before ``_lock`` (never the reverse) to prevent deadlock.  Both
    # ``reset()`` and ``get_expected_version()`` follow this order.

    def reset(self) -> None:
        """Clear all cached state (for tests)."""
        with self._compat_lock:
            self._compatibility_cache = None
        with self._lock:
            self._nlp_cache.clear()
            self._load_failed.clear()
            self._downloading.clear()
            self._expected_version_cache.clear()
            self._model_meta_cache.clear()
        _resolve_spacy.cache_clear()


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_manager = SpacyModelManager()

# ---------------------------------------------------------------------------
# Module-level public API
# ---------------------------------------------------------------------------


def _load_nlp_model(
    nlp_config: Optional[NlpConfig], variant: str, disable: Optional[tuple[str, ...]]
):
    config = nlp_config if nlp_config is not None else _DEFAULT_NLP_CONFIG
    if not config.enabled:
        return None
    return _manager.load_spacy_model(
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
    _manager.reset()
