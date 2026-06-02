import pytest

from mem0.configs.nlp.config import NlpConfig


class TestNlpConfig:
    def test_default_english_model(self):
        config = NlpConfig()
        assert config.resolve_model("full") == "en_core_web_sm"
        assert config.resolve_model("lemma") == "en_core_web_sm"
        assert config.enabled is True
        assert config.entity_extraction == "auto"

    def test_language_code_mapping(self):
        config = NlpConfig(language="zh")
        assert config.resolve_model() == "zh_core_web_sm"
        assert config.uses_ner_extraction is True

    def test_language_subtag(self):
        config = NlpConfig(language="zh-cn")
        assert config.language_code == "zh"
        assert config.resolve_model() == "zh_core_web_sm"

    def test_explicit_model_override(self):
        config = NlpConfig(language="zh", model="en_core_web_sm")
        assert config.resolve_model() == "en_core_web_sm"

    def test_separate_lemma_model(self):
        config = NlpConfig(language="en", model="en_core_web_md", lemma_model="en_core_web_sm")
        assert config.resolve_model("full") == "en_core_web_md"
        assert config.resolve_model("lemma") == "en_core_web_sm"

    def test_entity_extraction_ner_mode(self):
        config = NlpConfig(language="en", entity_extraction="ner")
        assert config.uses_ner_extraction is True

    def test_entity_extraction_heuristic_mode(self):
        config = NlpConfig(language="zh", entity_extraction="heuristic")
        assert config.uses_ner_extraction is False

    def test_unsupported_language_raises_on_construct(self):
        with pytest.raises(ValueError, match="Unsupported NLP language"):
            NlpConfig(language="invalid")

    def test_unsupported_language_when_disabled_ok(self):
        config = NlpConfig(enabled=False, language="invalid")
        assert config.enabled is False
        assert config.language_code == "invalid"
        assert config.resolve_model() is None

    def test_unsupported_language_with_explicit_model_ok(self):
        config = NlpConfig(language="invalid", model="en_core_web_sm")
        assert config.resolve_model() == "en_core_web_sm"

    def test_model_dir_default_none(self):
        config = NlpConfig()
        assert config.model_dir is None

    def test_model_dir_custom_path(self):
        config = NlpConfig(model_dir="/custom/spacy/data")
        assert config.model_dir == "/custom/spacy/data"

    def test_model_dir_with_disabled_does_not_affect_resolve(self):
        """model_dir has no effect on model resolution."""
        config = NlpConfig(enabled=False, model_dir="/tmp/spacy")
        assert config.model_dir == "/tmp/spacy"
        assert config.resolve_model() is None

    def test_download_url_default_none(self):
        config = NlpConfig()
        assert config.download_url is None

    def test_download_url_custom(self):
        config = NlpConfig(download_url="https://mirrors.example.com/spacy-models/releases/download")
        assert config.download_url == "https://mirrors.example.com/spacy-models/releases/download"
