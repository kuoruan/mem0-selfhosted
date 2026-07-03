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

    def test_entity_extraction_auto_mode_uses_ner(self):
        """auto mode always enables NER (with heuristics for non-CJK)."""
        config_en = NlpConfig(language="en", entity_extraction="auto")
        assert config_en.uses_ner_extraction is True
        config_zh = NlpConfig(language="zh", entity_extraction="auto")
        assert config_zh.uses_ner_extraction is True

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

    # ── CJK language coverage ──────────────────────────────────────────

    def test_cjk_languages_resolve_model(self):
        """zh, ja, ko should resolve to their respective spaCy models."""
        assert NlpConfig(language="zh").resolve_model() == "zh_core_web_sm"
        assert NlpConfig(language="ja").resolve_model() == "ja_core_news_sm"
        assert NlpConfig(language="ko").resolve_model() == "ko_core_news_sm"

    def test_cjk_languages_use_ner_in_auto_mode(self):
        """All CJK languages (zh, ja, ko) should enable NER in auto mode."""
        for lang in ("zh", "ja", "ko"):
            assert NlpConfig(language=lang, entity_extraction="auto").uses_ner_extraction is True, \
                f"auto mode should use NER for {lang}"

    def test_cjk_languages_heuristic_disables_ner(self):
        """CJK languages in heuristic mode should disable NER."""
        for lang in ("zh", "ja", "ko"):
            assert NlpConfig(language=lang, entity_extraction="heuristic").uses_ner_extraction is False, \
                f"heuristic mode should disable NER for {lang}"

    # ── language normalization ─────────────────────────────────────────

    def test_language_uppercase_normalized(self):
        """Uppercase language codes are normalized to lowercase."""
        assert NlpConfig(language="ZH").language == "zh"
        assert NlpConfig(language="EN").language == "en"

    def test_language_underscore_normalized(self):
        """Underscore-separated locales are normalized (zh_cn → zh, zh-cn)."""
        config = NlpConfig(language="zh_cn")
        assert config.language == "zh_cn"
        assert config.language_code == "zh"

    # ── resolve_model edge cases ───────────────────────────────────────

    def test_resolve_model_defaults_to_full(self):
        """resolve_model() without variant defaults to 'full'."""
        assert NlpConfig().resolve_model() == "en_core_web_sm"

    def test_lemma_model_without_explicit_model(self):
        """lemma_model alone affects only lemma variant; full variant uses language default."""
        config = NlpConfig(language="en", lemma_model="en_core_web_md")
        assert config.resolve_model("full") == "en_core_web_sm"  # language default
        assert config.resolve_model("lemma") == "en_core_web_md"  # explicit lemma

    def test_resolve_model_returns_none_when_disabled(self):
        """resolve_model returns None when enabled=False regardless of language."""
        config = NlpConfig(enabled=False, language="zh")
        assert config.resolve_model() is None
        assert config.resolve_model("lemma") is None

    # ── extra fields forbidden ─────────────────────────────────────────

    def test_extra_fields_forbidden(self):
        """NlpConfig forbids extra/unknown fields (model_config extra='forbid')."""
        with pytest.raises(ValueError):
            NlpConfig(language="en", unknown_field="value")
