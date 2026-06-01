from unittest.mock import MagicMock, patch

import pytest

from mem0.configs.nlp.config import NlpConfig
from mem0.utils import spacy_models

pytest.importorskip("spacy")


@pytest.fixture(autouse=True)
def _reset_spacy_cache():
    spacy_models.reset_spacy_cache()
    yield
    spacy_models.reset_spacy_cache()


class TestSpacyModels:
    def test_resolve_german_model_name(self):
        config = NlpConfig(language="de")
        assert config.resolve_model() == "de_core_news_sm"

    @patch("mem0.utils.spacy_models._ensure_model_available")
    @patch("spacy.load")
    def test_loads_and_caches_per_model(self, mock_load, mock_ensure):
        mock_nlp = MagicMock()
        mock_load.return_value = mock_nlp

        config = NlpConfig(language="en", auto_download=False)
        first = spacy_models.get_nlp_full(config)
        second = spacy_models.get_nlp_full(config)

        assert first is mock_nlp
        assert second is mock_nlp
        mock_load.assert_called_once_with("en_core_web_sm")

    @patch("mem0.utils.spacy_models._ensure_model_available")
    @patch("spacy.load")
    def test_lemma_pipeline_disables_ner_and_parser(self, mock_load, mock_ensure):
        mock_load.return_value = MagicMock()

        config = NlpConfig(language="en", auto_download=False)
        spacy_models.get_nlp_lemma(config)

        call_args = mock_load.call_args
        assert call_args is not None
        _, kwargs = call_args
        assert "disable" in kwargs
        assert set(kwargs["disable"]) == {"ner", "parser"}

    @patch("mem0.utils.spacy_models._ensure_model_available", side_effect=RuntimeError("missing"))
    def test_failed_load_returns_none(self, mock_ensure):
        config = NlpConfig(language="en", auto_download=False)
        assert spacy_models.get_nlp_full(config) is None
        assert spacy_models.get_nlp_full(config) is None

    def test_disabled_returns_none_without_loading(self):
        config = NlpConfig(enabled=False)
        assert spacy_models.get_nlp_full(config) is None
        assert spacy_models.get_nlp_lemma(config) is None
