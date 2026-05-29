"""Tests for spacy_models edge cases (no spaCy installation required)."""
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from mem0.configs.nlp.config import NlpConfig
from mem0.utils import spacy_models


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
            # This should not raise - local file exists
            try:
                spacy_models._ensure_model_available(f.name, auto_download=False)
            except Exception as e:
                # Failing is ok only if it's an import error (spacy not installed)
                # but NOT a RuntimeError saying package not found
                if "spaCy is not installed" in str(e):
                    pytest.skip("spaCy not installed")
                raise

    def test_local_directory_skips_package_check(self):
        """Local directory should be accepted without checking spacy package."""
        with tempfile.TemporaryDirectory() as d:
            try:
                spacy_models._ensure_model_available(d, auto_download=False)
            except Exception as e:
                if "spaCy is not installed" in str(e):
                    pytest.skip("spaCy not installed")


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
        # spacy.load should only be called once due to caching
        mock_load.assert_called_once()

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
