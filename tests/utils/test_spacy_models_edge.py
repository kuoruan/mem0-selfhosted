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
    spacy_models.reset_spacy_cache()


class TestDisableFiltering:
    """Test that disable list is filtered to only existing pipeline components."""

    @patch("spacy.util.get_model_meta")
    @patch("mem0.utils.spacy_models._is_model_available", return_value=True)
    @patch("spacy.load")
    def test_disable_filtered_to_existing_components(self, mock_load, mock_ensure, mock_get_meta):
        """Only existing pipeline components should be in disable list."""
        # Model has "tagger" and "ner" but NO "parser"
        mock_get_meta.return_value = {"pipeline": ["tagger", "ner"]}
        mock_load.return_value = MagicMock()

        config = NlpConfig(language="en", auto_download=False)
        spacy_models.get_nlp_lemma(config)  # disable=("ner", "parser")

        # Should only pass "ner" (not "parser") since model doesn't have parser
        call_args = mock_load.call_args
        assert call_args is not None, "spacy.load should have been called"
        _, kwargs = call_args
        assert "disable" in kwargs
        assert "ner" in kwargs["disable"]
        assert "parser" not in kwargs["disable"]

    @patch("spacy.util.get_model_meta")
    @patch("mem0.utils.spacy_models._is_model_available", return_value=True)
    @patch("spacy.load")
    def test_disable_empty_when_no_components_exist(self, mock_load, mock_ensure, mock_get_meta):
        """Empty disable list when no pipeline components match."""
        # Model has NO pipeline components
        mock_get_meta.return_value = {"pipeline": []}
        mock_load.return_value = MagicMock()

        config = NlpConfig(language="en", auto_download=False)
        spacy_models.get_nlp_lemma(config)

        call_args = mock_load.call_args
        assert call_args is not None
        _, kwargs = call_args
        assert "disable" in kwargs
        assert len(kwargs["disable"]) == 0

    @patch("spacy.util.get_model_meta", side_effect=ValueError("meta failed"))
    @patch("mem0.utils.spacy_models._is_model_available", return_value=True)
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

    @patch("mem0.utils.spacy_models._is_model_available", return_value=False)
    def test_model_not_available_auto_download_false_cached(self, mock_available):
        """When model is missing and auto_download=False, failure should be cached."""
        config = NlpConfig(language="en", auto_download=False)
        first = spacy_models.get_nlp_full(config)
        second = spacy_models.get_nlp_full(config)

        assert first is None
        assert second is None
        # _is_model_available should only be called once due to _load_failed cache
        mock_available.assert_called_once()

    @patch("mem0.utils.spacy_models._is_model_available", return_value=True)
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

    @patch("mem0.utils.spacy_models._is_model_available", return_value=True)
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
    """Test _ensure_model_dir behavior."""

    def test_none_returns_empty_string(self):
        result = spacy_models._ensure_model_dir(None)
        assert result == ""

    def test_creates_dir_and_adds_to_sys_path(self):

        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = os.path.join(tmpdir, "spacy_models")
            result = spacy_models._ensure_model_dir(model_dir)
            assert result == model_dir
            assert os.path.isdir(model_dir)
            assert sys.path[0] == model_dir
            # Restore sys.path
            sys.path.remove(model_dir)

    def test_existing_dir_returns_path(self):

        with tempfile.TemporaryDirectory() as tmpdir:
            result = spacy_models._ensure_model_dir(tmpdir)
            assert result == tmpdir

    def test_does_not_duplicate_sys_path_entry(self):

        with tempfile.TemporaryDirectory() as tmpdir:
            spacy_models._ensure_model_dir(tmpdir)  # first call
            path_count = sys.path.count(tmpdir)
            spacy_models._ensure_model_dir(tmpdir)  # second call
            assert sys.path.count(tmpdir) == path_count  # no duplicate
            sys.path.remove(tmpdir)


class TestGetNlpWithCacheDir:
    """Test get_nlp_full / get_nlp_lemma integration with model_dir."""

    @patch("mem0.utils.spacy_models._is_model_available", return_value=True)
    @patch("spacy.load")
    def test_full_loads_model_by_name(self, mock_load, mock_ensure):
        """spacy.load should be called with the model name (not a path)."""
        mock_load.return_value = MagicMock()

        config = NlpConfig(language="en", auto_download=False)
        spacy_models.get_nlp_full(config)

        mock_load.assert_called_once_with("en_core_web_sm")

    @patch("mem0.utils.spacy_models._is_model_available", return_value=True)
    @patch("spacy.load")
    def test_lemma_loads_model_by_name(self, mock_load, mock_ensure):
        """Lemma loader should also pass model name to spacy.load."""
        mock_load.return_value = MagicMock()

        config = NlpConfig(language="en", auto_download=False)
        spacy_models.get_nlp_lemma(config)

        load_name = mock_load.call_args[0][0]
        assert load_name == "en_core_web_sm"

    @patch("mem0.utils.spacy_models._ensure_model_dir", return_value="")
    @patch("mem0.utils.spacy_models._is_model_available", return_value=True)
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

    @patch("mem0.utils.spacy_models._download_model")
    @patch("mem0.utils.spacy_models._is_model_available", return_value=False)
    def test_returns_none_immediately(self, mock_available, mock_download):
        """auto_download=True returns None without blocking."""
        config = NlpConfig(language="en", auto_download=True)
        assert spacy_models.get_nlp_full(config) is None

    @patch("mem0.utils.spacy_models._is_model_available", return_value=False)
    def test_downloading_flag_set_during_download(self, mock_available):
        """_downloading tracks in-progress downloads."""
        started = threading.Event()

        def block(*_a, **_kw):
            started.set()
            # Keep the thread inside _download_model long enough for
            # the assertion below to run.
            time.sleep(0.5)

        with patch("mem0.utils.spacy_models._download_model", block):
            spacy_models.get_nlp_full(NlpConfig(language="en", auto_download=True))
            assert started.wait(timeout=2.0), "download thread did not start"
            # _downloading uses download_key (model_dir:model_name), not cache_key.
            assert ":en_core_web_sm" in spacy_models._downloading

    @patch("mem0.utils.spacy_models._is_model_available", return_value=False)
    def test_no_duplicate_download_threads(self, mock_available):
        """Second call while downloading returns None without extra thread."""
        started = threading.Event()

        def block(*_a, **_kw):
            started.set()
            time.sleep(0.5)

        with patch("mem0.utils.spacy_models._download_model", block):
            first = spacy_models.get_nlp_full(NlpConfig(language="en", auto_download=True))
            second = spacy_models.get_nlp_full(NlpConfig(language="en", auto_download=True))

        assert first is None
        assert second is None
        assert started.wait(timeout=2.0)

    @patch("mem0.utils.spacy_models._download_model")
    @patch("mem0.utils.spacy_models._is_model_available", return_value=False)
    def test_model_cached_after_successful_download(self, mock_available, mock_download):
        """Once the background download and load succeed, the model is cached."""
        mock_nlp = MagicMock()
        mock_spacy = MagicMock()
        mock_spacy.load.return_value = mock_nlp

        with patch("mem0.utils.spacy_models._get_spacy", return_value=mock_spacy):
            config = NlpConfig(language="en", auto_download=True)
            first = spacy_models.get_nlp_full(config)
            assert first is None

            # Wait for background thread to cache the model.
            key = spacy_models._cache_key("en_core_web_sm", "", None)
            for _ in range(50):
                if key in spacy_models._nlp_cache:
                    break
                time.sleep(0.05)
            else:
                pytest.fail("Background download did not complete within 2.5 s")

        second = spacy_models.get_nlp_full(config)
        assert second is mock_nlp
        mock_download.assert_called_once()

    @patch("mem0.utils.spacy_models._is_model_available", return_value=False)
    def test_download_failure_cleans_up_downloading(self, mock_available):
        """When download fails, _downloading is cleared and failure cached."""
        def fail(*_a, **_kw):
            raise RuntimeError("simulated download failure")

        with patch("mem0.utils.spacy_models._download_model", fail):
            spacy_models.get_nlp_full(NlpConfig(language="en", auto_download=True))

        # Wait for the background thread to finish to avoid flakiness.
        for thread in threading.enumerate():
            if thread.name == "spacy-download-en_core_web_sm":
                thread.join(timeout=5.0)
        assert ":en_core_web_sm" not in spacy_models._downloading

        # Failure should be cached.
        second = spacy_models.get_nlp_full(NlpConfig(language="en", auto_download=True))
        assert second is None
        assert mock_available.call_count <= 1
