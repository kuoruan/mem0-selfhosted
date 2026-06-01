"""Tests for NlpConfig edge cases."""
import pytest

from mem0.configs.nlp.config import NlpConfig


class TestNlpConfigEdgeCases:
    """NlpConfig edge case tests for feat/spacy changes."""

    def test_language_code_with_underscore(self):
        """language_code should handle underscore separation."""
        config = NlpConfig(language="zh_CN", auto_download=False)
        assert config.language_code == "zh"

    @pytest.mark.parametrize("lang,expected", [
        ("en_US", "en"),
        ("zh_TW", "zh"),
        ("ja_JP", "ja"),
    ])
    def test_various_language_underscore_codes(self, lang, expected):
        """Various underscore language codes should resolve correctly."""
        config = NlpConfig(language=lang, auto_download=False)
        assert config.language_code == expected
        # Should not raise ValueError since language_code maps correctly
        model = config.resolve_model()
        assert model is not None

    def test_unsupported_language_underscore_raises(self):
        """Underscore separated unsupported language should raise."""
        with pytest.raises(ValueError, match="Unsupported NLP language"):
            NlpConfig(language="invalid_lang")

    def test_resolve_model_returns_none_when_disabled(self):
        """resolve_model returns None when NLP is disabled."""
        config = NlpConfig(enabled=False)
        assert config.resolve_model() is None
        assert config.resolve_model("full") is None
        assert config.resolve_model("lemma") is None

    def test_disabled_config_still_has_defaults(self):
        """Disabled config should still set defaults but not validate language."""
        config = NlpConfig(enabled=False)
        assert config.language == "en"
        assert config.entity_extraction == "auto"

    def test_disabled_with_unsupported_language(self):
        """Disabled config with unsupported language should not raise."""
        config = NlpConfig(enabled=False, language="invalid")
        assert config.resolve_model() is None
        assert config.enabled is False
