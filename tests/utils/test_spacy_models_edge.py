"""Tests for spacy_models edge cases (spaCy required, model download not required)."""
import os
import sys
import tempfile
import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from mem0.configs.nlp.config import NlpConfig
from mem0.utils import spacy_models

pytest.importorskip("spacy")


@pytest.fixture(autouse=True)
def _reset_cache():
    spacy_models.reset_spacy_cache()
    yield
    # Join background download threads before resetting to prevent them
    # from writing stale entries into the freshly-cleared caches of the
    # next test.
    for thread in threading.enumerate():
        if thread.name.startswith("spacy-download-"):
            thread.join(timeout=5.0)
    spacy_models.reset_spacy_cache()


class TestDisableFiltering:
    """Test that disable list is filtered to only existing pipeline components."""

    @patch("spacy.util.get_model_meta")
    @patch("mem0.utils.spacy_models._manager.is_model_available", return_value=True)
    @patch("spacy.load")
    def test_disable_filtered_to_existing_components(self, mock_load, mock_ensure, mock_get_meta):
        """Only existing pipeline components should be in disable list."""
        # Model has "tagger" and "ner" but NO "parser".
        # First load attempt with unfiltered disable raises ValueError,
        # triggering the meta-based filter fallback.
        mock_get_meta.return_value = {"pipeline": ["tagger", "ner"]}
        mock_nlp = MagicMock()
        mock_load.side_effect = [ValueError("unknown component"), mock_nlp]

        config = NlpConfig(language="en", auto_download=False)
        result = spacy_models.get_nlp_lemma(config)  # disable=("ner", "parser")

        assert result is mock_nlp
        # Second call should have filtered disable (only "ner", not "parser").
        second_call = mock_load.call_args_list[1]
        _, kwargs = second_call
        assert "disable" in kwargs
        assert "ner" in kwargs["disable"]
        assert "parser" not in kwargs["disable"]

    @patch("spacy.util.get_model_meta")
    @patch("mem0.utils.spacy_models._manager.is_model_available", return_value=True)
    @patch("spacy.load")
    def test_disable_empty_when_no_components_exist(self, mock_load, mock_ensure, mock_get_meta):
        """Empty disable list when no pipeline components match."""
        # Model has NO pipeline components.  First load raises ValueError
        # because none of the requested disables exist, triggering meta
        # filter which produces an empty list.
        mock_get_meta.return_value = {"pipeline": []}
        mock_nlp = MagicMock()
        mock_load.side_effect = [ValueError("unknown component"), mock_nlp]

        config = NlpConfig(language="en", auto_download=False)
        result = spacy_models.get_nlp_lemma(config)

        assert result is mock_nlp
        second_call = mock_load.call_args_list[1]
        _, kwargs = second_call
        assert "disable" in kwargs
        assert len(kwargs["disable"]) == 0

    @patch("spacy.util.get_model_meta", side_effect=ValueError("meta failed"))
    @patch("mem0.utils.spacy_models._manager.is_model_available", return_value=True)
    @patch("spacy.load")
    def test_disable_falls_back_when_meta_fails(self, mock_load, mock_ensure, mock_get_meta):
        """If get_model_meta fails, use original disable list."""
        mock_load.return_value = MagicMock()

        config = NlpConfig(language="en", auto_download=False)
        spacy_models.get_nlp_lemma(config)

        call_args = mock_load.call_args
        assert call_args is not None
        _, kwargs = call_args
        # Should fall back to original disable list
        assert "ner" in kwargs["disable"]
        assert "parser" in kwargs["disable"]


class TestLoadSpacyModel:
    """Test _load_spacy_model edge cases."""

    @patch("mem0.utils.spacy_models._manager.is_model_available", return_value=False)
    def test_model_not_available_auto_download_false_cached(self, mock_available):
        """When model is missing and auto_download=False, failure should be cached."""
        config = NlpConfig(language="en", auto_download=False)
        first = spacy_models.get_nlp_full(config)
        second = spacy_models.get_nlp_full(config)

        assert first is None
        assert second is None
        # is_model_available should only be called once due to _load_failed cache
        mock_available.assert_called_once()

    @patch("mem0.utils.spacy_models._manager.is_model_available", return_value=True)
    @patch("spacy.load", side_effect=RuntimeError("load failed"))
    def test_load_failure_cached(self, mock_load, mock_available):
        """Model load failure should be cached."""
        config = NlpConfig(language="en", auto_download=False)
        first = spacy_models.get_nlp_full(config)
        second = spacy_models.get_nlp_full(config)

        assert first is None
        assert second is None
        # spacy.load should be called at most once (may be 0 if spacy not importable)
        assert mock_load.call_count <= 1

    @patch("mem0.utils.spacy_models._manager.is_model_available", return_value=True)
    @patch("spacy.load")
    def test_caching_different_models(self, mock_load, mock_available):
        """Different model configs should have separate cache entries."""
        mock_nlp_en = MagicMock()
        mock_nlp_de = MagicMock()

        def side_effect(*args, **kwargs):
            if "en_core_web_sm" in args:
                return mock_nlp_en
            return mock_nlp_de

        mock_load.side_effect = side_effect

        config_en = NlpConfig(language="en", auto_download=False)
        config_de = NlpConfig(language="de", auto_download=False)

        result_en = spacy_models.get_nlp_full(config_en)
        result_de = spacy_models.get_nlp_full(config_de)

        assert result_en is mock_nlp_en
        assert result_de is mock_nlp_de
        assert mock_load.call_count == 2


class TestEnsureCacheDir:
    """Test ensure_model_dir behavior."""

    def test_none_returns_empty_string(self):
        result = spacy_models._manager.ensure_model_dir(None)
        assert result == ""

    def test_creates_dir_and_adds_to_sys_path(self):

        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = os.path.join(tmpdir, "spacy_models")
            result = spacy_models._manager.ensure_model_dir(model_dir)
            assert result == model_dir
            assert os.path.isdir(model_dir)
            assert sys.path[0] == model_dir
            # Restore sys.path
            sys.path.remove(model_dir)

    def test_existing_dir_returns_path(self):

        with tempfile.TemporaryDirectory() as tmpdir:
            result = spacy_models._manager.ensure_model_dir(tmpdir)
            assert result == tmpdir

    def test_does_not_duplicate_sys_path_entry(self):

        with tempfile.TemporaryDirectory() as tmpdir:
            spacy_models._manager.ensure_model_dir(tmpdir)  # first call
            path_count = sys.path.count(tmpdir)
            spacy_models._manager.ensure_model_dir(tmpdir)  # second call
            assert sys.path.count(tmpdir) == path_count  # no duplicate
            sys.path.remove(tmpdir)


class TestGetNlpWithCacheDir:
    """Test get_nlp_full / get_nlp_lemma integration with model_dir."""

    @patch("mem0.utils.spacy_models._manager.is_model_available", return_value=True)
    @patch("spacy.load")
    def test_full_loads_model_by_name(self, mock_load, mock_ensure):
        """spacy.load should be called with the model name (not a path)."""
        mock_load.return_value = MagicMock()

        config = NlpConfig(language="en", auto_download=False)
        spacy_models.get_nlp_full(config)

        mock_load.assert_called_once_with("en_core_web_sm")

    @patch("mem0.utils.spacy_models._manager.is_model_available", return_value=True)
    @patch("spacy.load")
    def test_lemma_loads_model_by_name(self, mock_load, mock_ensure):
        """Lemma loader should also pass model name to spacy.load."""
        mock_load.return_value = MagicMock()

        config = NlpConfig(language="en", auto_download=False)
        spacy_models.get_nlp_lemma(config)

        load_name = mock_load.call_args[0][0]
        assert load_name == "en_core_web_sm"

    @patch("mem0.utils.spacy_models._manager.ensure_model_dir", return_value="")
    @patch("mem0.utils.spacy_models._manager.is_model_available", return_value=True)
    @patch("spacy.load")
    def test_disabled_skips_everything(self, mock_load, mock_ensure, mock_cache):
        """When NLP is disabled, no model loading should happen."""
        config = NlpConfig(enabled=False, model_dir="/tmp/spacy_models")
        result = spacy_models.get_nlp_full(config)

        assert result is None
        mock_load.assert_not_called()
        mock_ensure.assert_not_called()


class TestBackgroundDownload:
    """Test background download thread behaviour.

    When ``auto_download=True`` and the model is not on disk, a daemon
    thread is spawned to download and load it.  The calling thread
    returns ``None`` immediately.
    """

    @patch("mem0.utils.spacy_models._manager._download_model")
    @patch("mem0.utils.spacy_models._manager.is_model_available", return_value=False)
    def test_returns_none_immediately(self, mock_available, mock_download):
        """auto_download=True returns None without blocking."""
        config = NlpConfig(language="en", auto_download=True)
        assert spacy_models.get_nlp_full(config) is None

    @patch("mem0.utils.spacy_models._manager.is_model_available", return_value=False)
    def test_downloading_flag_set_during_download(self, mock_available):
        """_downloading tracks in-progress downloads."""
        started = threading.Event()

        def block(*_a, **_kw):
            started.set()
            # Keep the thread inside _download_model long enough for
            # the assertion below to run.
            time.sleep(0.5)

        with patch("mem0.utils.spacy_models._manager._download_model", block):
            spacy_models.get_nlp_full(NlpConfig(language="en", auto_download=True))
            assert started.wait(timeout=2.0), "download thread did not start"
            # _downloading uses download_key (model_dir:model_name), not cache_key.
            assert ":en_core_web_sm" in spacy_models._manager._downloading

    @patch("mem0.utils.spacy_models._manager.is_model_available", return_value=False)
    def test_no_duplicate_download_threads(self, mock_available):
        """Second call while downloading returns None without extra thread."""
        started = threading.Event()

        def block(*_a, **_kw):
            started.set()
            time.sleep(0.5)

        with patch("mem0.utils.spacy_models._manager._download_model", block):
            first = spacy_models.get_nlp_full(NlpConfig(language="en", auto_download=True))
            second = spacy_models.get_nlp_full(NlpConfig(language="en", auto_download=True))

        assert first is None
        assert second is None
        assert started.wait(timeout=2.0)

    @patch("mem0.utils.spacy_models._manager._download_model")
    @patch("mem0.utils.spacy_models._manager.is_model_available", return_value=False)
    def test_model_cached_after_successful_download(self, mock_available, mock_download):
        """Once the background download and load succeed, the model is cached."""
        mock_nlp = MagicMock()
        mock_spacy = MagicMock()
        mock_spacy.load.return_value = mock_nlp

        with patch("mem0.utils.spacy_models._manager.get_spacy", return_value=mock_spacy):
            config = NlpConfig(language="en", auto_download=True)
            first = spacy_models.get_nlp_full(config)
            assert first is None

            # Wait for background thread to cache the model.
            key = spacy_models._manager.cache_key("en_core_web_sm", "", None)
            for _ in range(50):
                if key in spacy_models._manager._nlp_cache:
                    break
                time.sleep(0.05)
            else:
                pytest.fail("Background download did not complete within 2.5 s")

        second = spacy_models.get_nlp_full(config)
        assert second is mock_nlp
        mock_download.assert_called_once()

    @patch("mem0.utils.spacy_models._manager.is_model_available", return_value=False)
    def test_download_failure_cleans_up_downloading(self, mock_available):
        """When download fails, _downloading is cleared and failure cached."""

        def fail(*_a, **_kw):
            raise RuntimeError("simulated download failure")

        with patch("mem0.utils.spacy_models._manager._download_model", fail):
            spacy_models.get_nlp_full(NlpConfig(language="en", auto_download=True))

        # Wait for the background thread to finish to avoid flakiness.
        for thread in threading.enumerate():
            if thread.name == "spacy-download-en_core_web_sm":
                thread.join(timeout=5.0)
        assert ":en_core_web_sm" not in spacy_models._manager._downloading

        # Failure should be cached.
        second = spacy_models.get_nlp_full(NlpConfig(language="en", auto_download=True))
        assert second is None
        assert mock_available.call_count <= 1


class TestGetExpectedVersion:
    """Test get_expected_version caching and error handling."""

    def test_caches_per_model(self):
        """Same model returns cached value; HTTP call only once."""
        with patch(
            "spacy.cli.download.get_compatibility", return_value={"en_core_web_sm": ["3.7.0"]}
        ) as mock_comp:
            v1 = spacy_models._manager.get_expected_version("en_core_web_sm")
            v2 = spacy_models._manager.get_expected_version("en_core_web_sm")

        assert v1 == "3.7.0"
        assert v2 == "3.7.0"
        mock_comp.assert_called_once()

    def test_handles_different_models(self):
        """Different models share the compatibility table — one HTTP call total."""
        comp = {"en_core_web_sm": ["3.7.0"], "de_core_news_sm": ["3.7.0"]}
        with patch(
            "spacy.cli.download.get_compatibility", return_value=comp
        ) as mock_comp:
            v_en = spacy_models._manager.get_expected_version("en_core_web_sm")
            v_de = spacy_models._manager.get_expected_version("de_core_news_sm")

        assert v_en == "3.7.0"
        assert v_de == "3.7.0"
        # Compatibility table is cached globally; only one HTTP call total.
        mock_comp.assert_called_once()

    @patch("spacy.cli.download.get_compatibility", side_effect=RuntimeError("network down"))
    def test_returns_none_on_http_failure(self, mock_comp):
        """Transient network failures are NOT cached — retried on next call."""
        result1 = spacy_models._manager.get_expected_version("en_core_web_sm")
        result2 = spacy_models._manager.get_expected_version("en_core_web_sm")

        assert result1 is None
        assert result2 is None
        # Failure not cached → retried on each call.
        assert mock_comp.call_count == 2

    def test_returns_none_when_spacy_not_installed(self):
        """When spacy is not installed, returns None immediately (no import attempt)."""
        with patch("mem0.utils.spacy_models._manager.get_spacy", return_value=None):
            spacy_models._manager._expected_version_cache.clear()
            result = spacy_models._manager.get_expected_version("en_core_web_sm")
            assert result is None


class TestCompatLockConcurrency:
    """Test that _compat_lock prevents redundant HTTP requests under concurrency."""

    def test_concurrent_threads_single_http_call(self):
        """Multiple threads requesting at the same time trigger only one get_compatibility() call."""
        call_count = {"n": 0}

        def slow_get_compatibility():
            """Simulate a slow HTTP response to widen the race window."""
            call_count["n"] += 1
            time.sleep(0.3)
            return {"en_core_web_sm": ["3.7.0"]}

        results = [None] * 4
        errors = [None] * 4

        def worker(idx):
            try:
                results[idx] = spacy_models._manager.get_expected_version("en_core_web_sm")
            except Exception as exc:
                errors[idx] = exc

        with patch("spacy.cli.download.get_compatibility", side_effect=slow_get_compatibility):
            threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=5.0)

        # All threads got the correct result.
        assert all(e is None for e in errors), f"Threads errored: {errors}"
        assert all(r == "3.7.0" for r in results), f"Unexpected results: {results}"
        # Only one HTTP call was made — _compat_lock serialised the fetch.
        assert call_count["n"] == 1, f"Expected 1 HTTP call, got {call_count['n']}"

    def test_second_call_reuses_cached_table(self):
        """After first fetch, subsequent calls skip _compat_lock entirely."""
        with patch(
            "spacy.cli.download.get_compatibility", return_value={"en_core_web_sm": ["3.7.0"]}
        ) as mock_comp:
            spacy_models._manager.get_expected_version("en_core_web_sm")
            spacy_models._manager.get_expected_version("en_core_web_sm")

        mock_comp.assert_called_once()

    def test_compat_lock_not_held_during_version_resolution(self):
        """_compat_lock blocks a second thread while the first fetches, then the second reuses the cache."""
        fetch_entered = threading.Event()
        fetch_proceed = threading.Event()
        call_count = {"n": 0}

        def slow_compat():
            call_count["n"] += 1
            fetch_entered.set()
            fetch_proceed.wait(timeout=5.0)
            return {"en_core_web_sm": ["3.7.0"], "de_core_news_sm": ["3.7.0"]}

        mock_comp = MagicMock(side_effect=slow_compat)
        with patch("spacy.cli.download.get_compatibility", mock_comp):
            t1 = threading.Thread(
                target=lambda: spacy_models._manager.get_expected_version("en_core_web_sm")
            )
            t1.start()

            # Wait for the first thread to enter the compat fetch.
            assert fetch_entered.wait(timeout=5.0)

            # Start a second thread while the first still holds _compat_lock.
            # It should block on _compat_lock, NOT start a second HTTP call.
            t2_result = [None]
            t2_error = [None]

            def t2_worker():
                try:
                    t2_result[0] = spacy_models._manager.get_expected_version("de_core_news_sm")
                except Exception as exc:
                    t2_error[0] = exc

            t2 = threading.Thread(target=t2_worker)
            t2.start()

            # Give t2 a moment — it should be blocked on _compat_lock.
            time.sleep(0.2)
            assert call_count["n"] == 1, "Second thread should not have started a separate HTTP call"

            # Let the first thread finish — this releases _compat_lock.
            fetch_proceed.set()
            t1.join(timeout=5.0)
            t2.join(timeout=5.0)

        # Both threads got their results.
        assert t2_error[0] is None, f"Second thread errored: {t2_error[0]}"
        # Only one get_compatibility() call — t2 reused the cached table.
        assert call_count["n"] == 1, f"Expected 1 HTTP call, got {call_count['n']}"


class TestGetModelMeta:
    """Test get_model_meta caching and error handling."""

    @patch("spacy.util.get_model_meta")
    def test_caches_meta(self, mock_get_meta):
        """Same model returns cached meta; get_model_meta called once."""
        mock_get_meta.return_value = {"version": "3.7.0", "pipeline": ["ner"]}

        m1 = spacy_models._manager.get_model_meta("en_core_web_sm")
        m2 = spacy_models._manager.get_model_meta("en_core_web_sm")

        assert m1 == {"version": "3.7.0", "pipeline": ["ner"]}
        assert m2 is m1  # same object from cache
        mock_get_meta.assert_called_once_with("en_core_web_sm")

    def test_returns_none_when_spacy_missing(self):
        """When get_spacy returns None, get_model_meta returns None."""
        key = "some_missing_model"
        # This model won't be in cache and get_spacy is mocked to return None.
        with patch("mem0.utils.spacy_models._manager.get_spacy", return_value=None):
            # Also need to make sure it's not cached from a previous call.
            spacy_models._manager._model_meta_cache.pop(key, None)
            result = spacy_models._manager.get_model_meta(key)
        assert result is None

    @patch("spacy.util.get_model_meta", side_effect=OSError("disk error"))
    def test_returns_none_on_error(self, mock_get_meta):
        """Transient disk failures are NOT cached — retried on next call."""
        result1 = spacy_models._manager.get_model_meta("en_core_web_sm")
        result2 = spacy_models._manager.get_model_meta("en_core_web_sm")

        assert result1 is None
        assert result2 is None
        # Failure not cached → retried on each call.
        assert mock_get_meta.call_count == 2


class TestCheckModelVersion:
    """Test check_model_version behaviour."""

    def test_version_match_returns_true(self):
        """When installed version matches expected, returns True."""
        with patch(
            "mem0.utils.spacy_models._manager.get_expected_version", return_value="3.7.0"
        ), patch(
            "mem0.utils.spacy_models._manager.get_model_meta", return_value={"version": "3.7.0"}
        ):
            assert spacy_models._manager.check_model_version("en_core_web_sm") is True

    def test_version_mismatch_returns_false(self):
        """When installed version differs from expected, returns False."""
        with patch(
            "mem0.utils.spacy_models._manager.get_expected_version", return_value="3.7.0"
        ), patch(
            "mem0.utils.spacy_models._manager.get_model_meta", return_value={"version": "3.6.0"}
        ):
            assert spacy_models._manager.check_model_version("en_core_web_sm") is False

    def test_expected_none_returns_true(self):
        """When expected version is unresolvable, assume OK."""
        with patch("mem0.utils.spacy_models._manager.get_expected_version", return_value=None):
            assert spacy_models._manager.check_model_version("en_core_web_sm") is True

    def test_meta_none_with_expected_returns_false(self):
        """When expected version is known but meta is unreadable, treat as incompatible."""
        with patch(
            "mem0.utils.spacy_models._manager.get_expected_version", return_value="3.7.0"
        ), patch("mem0.utils.spacy_models._manager.get_model_meta", return_value=None):
            assert spacy_models._manager.check_model_version("en_core_web_sm") is False

    def test_empty_installed_version_returns_true(self):
        """When installed version is empty string, assume OK."""
        with patch(
            "mem0.utils.spacy_models._manager.get_expected_version", return_value="3.7.0"
        ), patch(
            "mem0.utils.spacy_models._manager.get_model_meta", return_value={"version": ""}
        ):
            assert spacy_models._manager.check_model_version("en_core_web_sm") is True

    def test_no_version_key_returns_true(self):
        """When meta has no 'version' key, assume OK."""
        with patch(
            "mem0.utils.spacy_models._manager.get_expected_version", return_value="3.7.0"
        ), patch("mem0.utils.spacy_models._manager.get_model_meta", return_value={}):
            assert spacy_models._manager.check_model_version("en_core_web_sm") is True


class TestLoadSpacyModelVersionIntegration:
    """Integration tests for version-aware load_spacy_model.

    Version check is deferred to the *failure* path: the model is loaded
    first, and ``check_model_version`` is only called if ``spacy.load``
    raises.  This avoids a synchronous HTTP call on every model load.
    """

    @patch("mem0.utils.spacy_models._manager.is_model_available", return_value=True)
    @patch("spacy.load")
    def test_version_ok_loads_synchronously(self, mock_load, mock_available):
        """Model loads successfully — no version check needed."""
        mock_nlp = MagicMock()
        mock_load.return_value = mock_nlp

        config = NlpConfig(language="en", auto_download=False)
        result = spacy_models.get_nlp_full(config)

        assert result is mock_nlp
        mock_load.assert_called_once()

    @patch("mem0.utils.spacy_models._manager.check_model_version", return_value=False)
    @patch("mem0.utils.spacy_models._manager.is_model_available", return_value=True)
    @patch("spacy.load", side_effect=RuntimeError("incompatible model"))
    def test_version_mismatch_auto_download_false(self, mock_load, mock_available, mock_check):
        """Load fails + version mismatch + auto_download=False → None with failure cached."""
        config = NlpConfig(language="en", auto_download=False)
        result = spacy_models.get_nlp_full(config)

        assert result is None
        # Subsequent call should hit the failure cache, not re-check version.
        with patch("mem0.utils.spacy_models._manager.check_model_version") as mock_check2:
            result2 = spacy_models.get_nlp_full(config)
            assert result2 is None
            mock_check2.assert_not_called()

    @patch("mem0.utils.spacy_models._manager.check_model_version", return_value=False)
    @patch("mem0.utils.spacy_models._manager.is_model_available", return_value=True)
    @patch("spacy.load", side_effect=RuntimeError("incompatible model"))
    def test_version_mismatch_auto_download_true(self, mock_load, mock_available, mock_check):
        """Load fails + version mismatch + auto_download=True → background download."""
        with patch("mem0.utils.spacy_models._manager._download_model") as mock_download:
            config = NlpConfig(language="en", auto_download=True)
            result = spacy_models.get_nlp_full(config)

        # Returns None immediately (background download in progress).
        assert result is None
        # Wait for the daemon thread to call _download_model.
        for thread in threading.enumerate():
            if thread.name == "spacy-download-en_core_web_sm":
                thread.join(timeout=5.0)
        mock_download.assert_called_once()

    @patch("mem0.utils.spacy_models._manager.check_model_version", return_value=False)
    @patch("mem0.utils.spacy_models._manager.is_model_available", return_value=True)
    @patch("spacy.load")
    def test_version_mismatch_redownload_clears_meta_cache(self, mock_load, mock_available, mock_check):
        """After re-download, stale model meta cache is purged."""
        # First load from disk raises (simulating version mismatch).
        mock_load.side_effect = [RuntimeError("incompatible model"), MagicMock()]

        # Pre-populate _model_meta_cache with old data.
        spacy_models._manager._model_meta_cache["en_core_web_sm"] = {
            "version": "3.6.0",
            "pipeline": ["old_pipe"],
        }
        assert "en_core_web_sm" in spacy_models._manager._model_meta_cache

        with patch("mem0.utils.spacy_models._manager._download_model"):
            config = NlpConfig(language="en", auto_download=True)
            spacy_models.get_nlp_full(config)

            # Wait for background thread to complete.
            key = spacy_models._manager.cache_key("en_core_web_sm", "", None)
            for _ in range(50):
                if key in spacy_models._manager._nlp_cache:
                    break
                time.sleep(0.05)

        # After successful background download, old meta should be purged.
        assert "en_core_web_sm" not in spacy_models._manager._model_meta_cache

    @patch("mem0.utils.spacy_models._manager.is_model_available", return_value=True)
    @patch("spacy.load", side_effect=RuntimeError("bad model"))
    def test_another_thread_finished_during_version_check(self, mock_load, mock_available):
        """If another thread loads the model while we check version, return it."""
        mock_nlp = MagicMock()

        def check_version_side_effect(model_name):
            # Simulate another thread finishing the work during our version check.
            spacy_models._manager._nlp_cache[
                spacy_models._manager.cache_key("en_core_web_sm", "", None)
            ] = mock_nlp
            return False  # version mismatch on our side

        with patch(
            "mem0.utils.spacy_models._manager.check_model_version",
            side_effect=check_version_side_effect,
        ):
            config = NlpConfig(language="en", auto_download=True)
            result = spacy_models.get_nlp_full(config)

        # Should have returned the model from cache, not None and not started a thread.
        assert result is mock_nlp

    @patch("mem0.utils.spacy_models._manager.is_model_available", return_value=True)
    @patch("spacy.load", side_effect=RuntimeError("bad model"))
    def test_another_thread_started_download_during_version_check(self, mock_load, mock_available):
        """If another thread started a download while we check version, bail out."""
        dl_key = spacy_models._manager.dl_key("en_core_web_sm", "")

        def check_version_with_download(*_a, **_kw):
            spacy_models._manager._downloading.add(dl_key)
            return False

        with patch.object(
            spacy_models._manager, "check_model_version", side_effect=check_version_with_download
        ):
            config = NlpConfig(language="en", auto_download=True)
            result = spacy_models.get_nlp_full(config)

        assert result is None


class TestDownloadModelPipArgs:
    """Test _download_model pip invocation."""

    @patch("mem0.utils.spacy_models._manager.get_expected_version", return_value="3.7.0")
    @patch(
        "spacy.cli.download.get_model_filename",
        return_value="en_core_web_sm-3.7.0-py3-none-any.whl",
    )
    @patch(
        "spacy.about.__download_url__",
        "https://github.com/explosion/spacy-models/releases/download",
    )
    @patch("subprocess.run")
    def test_pip_includes_upgrade_strategy(self, mock_run, mock_filename, mock_version):
        """pip install command includes --upgrade and --upgrade-strategy only-if-needed."""
        mock_run.return_value.returncode = 0

        spacy_models._manager._download_model("en_core_web_sm", "", None)

        cmd = mock_run.call_args[0][0]
        assert "--upgrade" in cmd
        assert "--upgrade-strategy" in cmd
        assert "only-if-needed" in cmd
        # Ensure they appear in correct order (--upgrade comes before --upgrade-strategy).
        upgrade_idx = cmd.index("--upgrade")
        strategy_idx = cmd.index("--upgrade-strategy")
        assert strategy_idx > upgrade_idx

    @patch("mem0.utils.spacy_models._manager.get_expected_version", return_value="3.7.0")
    @patch(
        "spacy.cli.download.get_model_filename",
        return_value="en_core_web_sm-3.7.0-py3-none-any.whl",
    )
    @patch(
        "spacy.about.__download_url__",
        "https://github.com/explosion/spacy-models/releases/download",
    )
    @patch("subprocess.run")
    def test_pip_includes_target_when_model_dir(self, mock_run, mock_filename, mock_version):
        """When model_dir is set, pip install uses --target."""
        mock_run.return_value.returncode = 0

        spacy_models._manager._download_model("en_core_web_sm", "/tmp/models", None)

        cmd = mock_run.call_args[0][0]
        assert "--target" in cmd
        target_idx = cmd.index("--target")
        assert cmd[target_idx + 1] == "/tmp/models"
