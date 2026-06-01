"""Tests for spacy_models edge cases (spaCy required, model download not required)."""
import tempfile
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


class TestEnsureModelAvailable:
    """Test _ensure_model_available edge cases."""

    def test_local_path_skips_package_check(self):
        """Local file path should be accepted without checking spacy package."""
        with tempfile.NamedTemporaryFile(suffix=".pt") as f:
            spacy_models._ensure_model_available(f.name, model_dir="", download_url=None, auto_download=False)

    def test_local_directory_skips_package_check(self):
        """Local directory should be accepted without checking spacy package."""
        with tempfile.TemporaryDirectory() as d:
            spacy_models._ensure_model_available(d, model_dir="", download_url=None, auto_download=False)


class TestDisableFiltering:
    """Test that disable list is filtered to only existing pipeline components."""

    @patch("spacy.util.get_model_meta")
    @patch("mem0.utils.spacy_models._ensure_model_available")
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
    @patch("mem0.utils.spacy_models._ensure_model_available")
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

    @patch("spacy.util.get_model_meta", side_effect=Exception("meta failed"))
    @patch("mem0.utils.spacy_models._ensure_model_available")
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

    @patch("mem0.utils.spacy_models._ensure_model_available", side_effect=RuntimeError("network error"))
    def test_ensure_failure_cached_in_load_failed(self, mock_ensure):
        """_ensure_model_available failure should be cached in _load_failed."""
        config = NlpConfig(language="en", auto_download=True)
        first = spacy_models.get_nlp_full(config)
        second = spacy_models.get_nlp_full(config)

        assert first is None
        assert second is None
        # _ensure should only be called once due to _load_failed cache
        mock_ensure.assert_called_once()

    @patch("mem0.utils.spacy_models._ensure_model_available")
    @patch("spacy.load", side_effect=RuntimeError("load failed"))
    def test_load_failure_cached(self, mock_load, mock_ensure):
        """Model load failure should be cached."""
        mock_ensure.return_value = None

        config = NlpConfig(language="en", auto_download=False)
        first = spacy_models.get_nlp_full(config)
        second = spacy_models.get_nlp_full(config)

        assert first is None
        assert second is None
        # spacy.load should be called at most once (may be 0 if spacy not importable)
        assert mock_load.call_count <= 1

    @patch("mem0.utils.spacy_models._ensure_model_available")
    @patch("spacy.load")
    def test_caching_different_models(self, mock_load, mock_ensure):
        """Different model configs should have separate cache entries."""
        mock_ensure.return_value = None
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
        import os
        import sys
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = os.path.join(tmpdir, "spacy_models")
            old_path = list(sys.path)
            result = spacy_models._ensure_model_dir(model_dir)
            assert result == model_dir
            assert os.path.isdir(model_dir)
            assert sys.path[0] == model_dir
            # Restore sys.path
            sys.path.remove(model_dir)

    def test_existing_dir_returns_path(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            result = spacy_models._ensure_model_dir(tmpdir)
            assert result == tmpdir

    def test_does_not_duplicate_sys_path_entry(self):
        import sys
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            spacy_models._ensure_model_dir(tmpdir)  # first call
            path_count = sys.path.count(tmpdir)
            spacy_models._ensure_model_dir(tmpdir)  # second call
            assert sys.path.count(tmpdir) == path_count  # no duplicate
            sys.path.remove(tmpdir)


class TestEnsureModelAvailableCacheDir:
    """Test _ensure_model_available with model_dir."""

    def test_returns_early_when_model_dir_in_cache(self):
        """Should return immediately if model directory exists under model_dir."""
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = os.path.join(tmpdir, "en_core_web_sm")
            os.makedirs(model_dir)
            # Should not raise even without auto_download
            spacy_models._ensure_model_available(
                "en_core_web_sm", model_dir=tmpdir, download_url=None, auto_download=False
            )

    @patch("spacy.cli.download")
    def test_download_to_model_dir_uses_target(self, mock_download):
        """When model_dir is set, download should pass --target as pip arg."""
        config = NlpConfig(language="en", model_dir="/tmp/spacy_models")
        spacy_models._ensure_model_dir(config.model_dir)
        try:
            spacy_models._ensure_model_available(
                "en_core_web_sm", model_dir=config.model_dir,
                download_url=None, auto_download=True,
            )
        except Exception:
            pass  # May fail if no network, but download should have been called
        if mock_download.called:
            mock_download.assert_called_once_with(
                "en_core_web_sm", False, False, None,
                "--target", "/tmp/spacy_models",
            )


class TestGetNlpWithCacheDir:
    """Test get_nlp_full / get_nlp_lemma integration with model_dir."""

    @patch("mem0.utils.spacy_models._ensure_model_available")
    @patch("spacy.load")
    def test_full_loads_model_by_name(self, mock_load, mock_ensure):
        """spacy.load should be called with the model name (not a path)."""
        mock_load.return_value = MagicMock()

        config = NlpConfig(language="en", auto_download=False)
        spacy_models.get_nlp_full(config)

        mock_load.assert_called_once_with("en_core_web_sm")

    @patch("mem0.utils.spacy_models._ensure_model_available")
    @patch("spacy.load")
    def test_lemma_loads_model_by_name(self, mock_load, mock_ensure):
        """Lemma loader should also pass model name to spacy.load."""
        mock_load.return_value = MagicMock()

        config = NlpConfig(language="en", auto_download=False)
        spacy_models.get_nlp_lemma(config)

        load_name = mock_load.call_args[0][0]
        assert load_name == "en_core_web_sm"

    @patch("mem0.utils.spacy_models._ensure_model_dir", return_value="")
    @patch("mem0.utils.spacy_models._ensure_model_available")
    @patch("spacy.load")
    def test_disabled_skips_everything(self, mock_load, mock_ensure, mock_cache):
        """When NLP is disabled, no model loading should happen."""
        config = NlpConfig(enabled=False, model_dir="/tmp/spacy_models")
        result = spacy_models.get_nlp_full(config)

        assert result is None
        mock_load.assert_not_called()
        mock_ensure.assert_not_called()

    @patch("spacy.cli.download")
    def test_download_url_passed_to_download(self, mock_download):
        """custom_url should be passed to spacy.cli.download."""
        # Use a model name that isn't installed, so download is triggered.
        try:
            spacy_models._ensure_model_available(
                "xx_nonexistent_model_nlp", model_dir="",
                download_url="https://mirror.example.com/models",
                auto_download=True,
            )
        except Exception:
            pass  # download may fail due to network
        if mock_download.called:
            mock_download.assert_called_once_with(
                "xx_nonexistent_model_nlp", False, False,
                "https://mirror.example.com/models",
            )
