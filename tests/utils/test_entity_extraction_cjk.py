"""Tests for CJK entity extraction behavior (no spaCy model required)."""
from unittest.mock import MagicMock, patch

import pytest

from mem0.utils.entity_extraction import _extract_entities_from_doc, _finalize_entities


class TestCjkEntityExtraction:
    """CJK-specific entity extraction tests."""

    def test_cjk_min_length_accepts_single_char(self):
        """CJK languages should accept 1-character entities in NER mode."""
        doc = MagicMock()
        doc.text = "我去了北京"
        doc.__iter__.return_value = []
        doc.noun_chunks = []

        ent = MagicMock()
        ent.text = "北京"
        ent.label_ = "GPE"
        doc.ents = [ent]

        result = _extract_entities_from_doc(doc, entity_extraction="ner", language_code="zh")
        assert any("北京" in e[1] for e in result), f"Expected '北京' in results, got {result}"

    def test_cjk_accepts_two_char_entity(self):
        """CJK 2-character entities should be kept."""
        doc = MagicMock()
        doc.text = "我喜欢中国"
        doc.__iter__.return_value = []
        doc.noun_chunks = []

        ent = MagicMock()
        ent.text = "中国"
        ent.label_ = "GPE"
        doc.ents = [ent]

        result = _extract_entities_from_doc(doc, entity_extraction="ner", language_code="zh")
        assert any("中国" in e[1] for e in result), f"Expected '中国' in results, got {result}"

    def test_cjk_keeps_substring_entities(self):
        """CJK entities that are substrings of longer ones should be kept."""
        entities = [
            ("PROPER", "北京"),
            ("PROPER", "北京大学"),
        ]
        result = _finalize_entities(entities, language_code="zh")
        texts = [e[1] for e in result]
        assert "北京" in texts, "北京 should be kept in CJK mode"
        assert "北京大学" in texts, "北京大学 should be kept in CJK mode"

    def test_non_cjk_removes_substrings(self):
        """Non-CJK entities that are substrings should be removed."""
        entities = [
            ("PROPER", "New"),
            ("PROPER", "New York"),
        ]
        result = _finalize_entities(entities, language_code="en")
        texts = [e[1] for e in result]
        assert "New" not in texts, "New should be removed as substring of New York"
        assert "New York" in texts

    def test_cjk_finalize_min_length_one(self):
        """_finalize_entities should accept 1-char entities for CJK."""
        result = _finalize_entities([("PROPER", "中")], language_code="zh")
        assert len(result) == 1
        assert result[0][1] == "中"

    def test_non_cjk_finalize_requires_min_length_three(self):
        """_finalize_entities should require min 3 chars for non-CJK."""
        result = _finalize_entities([("PROPER", "ab")], language_code="en")
        assert len(result) == 0

    def test_cjk_deduplication_case_insensitive(self):
        """CJK entity dedup should still work."""
        entities = [
            ("PROPER", "北京"),
            ("PROPER", "北京"),
        ]
        result = _finalize_entities(entities, language_code="zh")
        assert len(result) == 1


class TestNounChunksErrorHandling:
    """Test handling of noun_chunks NotImplementedError and ValueError."""

    def test_not_implemented_error_caught(self):
        """NotImplementedError on noun_chunks should be caught."""
        doc = MagicMock()
        doc.text = "Test text"
        doc.__iter__.return_value = []
        doc.ents = []
        doc.noun_chunks.__iter__.side_effect = NotImplementedError("no parser")

        # Should not raise
        result = _extract_entities_from_doc(doc, entity_extraction="auto", language_code="en")
        assert isinstance(result, list)

    def test_value_error_caught(self):
        """ValueError on noun_chunks should be caught."""
        doc = MagicMock()
        doc.text = "Test text"
        doc.__iter__.return_value = []
        doc.ents = []
        doc.noun_chunks.__iter__.side_effect = ValueError("no parser pipeline")

        # Should not raise
        result = _extract_entities_from_doc(doc, entity_extraction="auto", language_code="en")
        assert isinstance(result, list)

    def test_noun_chunks_works_when_available(self):
        """noun_chunks should work normally when available."""
        doc = MagicMock()
        doc.text = "machine learning is great"
        doc.__iter__.return_value = []

        token = MagicMock()
        token.text = "machine"
        token.lemma_ = "machine"
        token.pos_ = "NOUN"
        token.dep_ = "compound"
        token.is_stop = False
        token.i = 0

        token2 = MagicMock()
        token2.text = "learning"
        token2.lemma_ = "learning"
        token2.pos_ = "NOUN"
        token2.dep_ = "nmod"
        token2.is_stop = False
        token2.i = 1

        doc.__iter__.return_value = [token, token2]
        doc.ents = []

        # Create a mock noun_chunk
        chunk = MagicMock()
        chunk.__iter__.return_value = [token, token2]
        doc.noun_chunks = [chunk]

        result = _extract_entities_from_doc(doc, entity_extraction="auto", language_code="en")
        # Should not raise and return a list
        assert isinstance(result, list)
