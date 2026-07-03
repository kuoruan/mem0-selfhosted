"""Tests for CJK entity extraction behavior (no spaCy model required)."""
from unittest.mock import MagicMock

from mem0.utils.entity_extraction import (
    _EntityCandidate,
    _add_proper_name_candidates,
    _add_quoted_candidates,
    _extract_entities_from_doc,
    _resolve_candidates,
)


def _make_token(text, *, pos_="PROPN", dep_="nsubj", is_stop=False, head_pos="VERB", i=0):
    """A spaCy-Token-like mock sufficient for the NER candidate path."""
    tok = MagicMock()
    tok.text = text
    tok.text_with_ws = text
    tok.lemma_ = text.lower()
    tok.pos_ = pos_
    tok.dep_ = dep_
    tok.is_stop = is_stop
    tok.i = i
    tok.head.pos_ = head_pos
    return tok


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
        ent.__iter__.return_value = [_make_token("北京")]
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
        ent.__iter__.return_value = [_make_token("中国")]
        doc.ents = [ent]

        result = _extract_entities_from_doc(doc, entity_extraction="ner", language_code="zh")
        assert any("中国" in e[1] for e in result), f"Expected '中国' in results, got {result}"

    def test_cjk_keeps_substring_entities(self):
        """CJK: distinct-text entities are all kept — the candidate resolver does not
        perform text-substring removal (span-overlap only), so CJK short entities
        that happen to be substrings of longer ones survive."""
        candidates = [
            _EntityCandidate("PROPER", "北京", "spacy_ner", -1, -1, 0.95, 0),
            _EntityCandidate("PROPER", "北京大学", "spacy_ner", -1, -1, 0.95, 0),
        ]
        result = _resolve_candidates(candidates)
        texts = [e[1] for e in result]
        assert "北京" in texts, "北京 should be kept"
        assert "北京大学" in texts, "北京大学 should be kept"

    def test_overlapping_spans_deduped(self):
        """A shorter entity whose span overlaps a longer one is dropped in favor of
        the longer (the new architecture's equivalent of substring dedup)."""
        candidates = [
            _EntityCandidate("PROPER", "New", "spacy_ner", 0, 3, 0.95, 0),
            _EntityCandidate("PROPER", "New York", "spacy_ner", 0, 8, 0.95, 0),
        ]
        result = _resolve_candidates(candidates)
        texts = [e[1] for e in result]
        assert "New York" in texts
        assert "New" not in texts

    def test_cjk_ner_accepts_one_char(self):
        """CJK NER accepts 1-character entities (min_len=1)."""
        doc = MagicMock()
        doc.text = "中"
        doc.__iter__.return_value = []
        doc.noun_chunks = []

        ent = MagicMock()
        ent.text = "中"
        ent.label_ = "GPE"
        ent.__iter__.return_value = [_make_token("中")]
        doc.ents = [ent]

        result = _extract_entities_from_doc(doc, entity_extraction="ner", language_code="zh")
        assert any("中" in e[1] for e in result), f"Expected '中' in results, got {result}"

    def test_non_cjk_ner_requires_min_length_three(self):
        """Non-CJK NER rejects entities shorter than 3 characters (min_len=3)."""
        doc = MagicMock()
        doc.text = "Ab"
        doc.__iter__.return_value = []
        doc.noun_chunks = []

        ent = MagicMock()
        ent.text = "Ab"
        ent.label_ = "GPE"
        ent.__iter__.return_value = [_make_token("Ab")]
        doc.ents = [ent]

        result = _extract_entities_from_doc(doc, entity_extraction="ner", language_code="en")
        assert result == [], f"Expected empty result for 2-char non-CJK entity, got {result}"

    def test_deduplication_case_insensitive(self):
        """Duplicate entities (case-insensitive normalized text) are deduped to one."""
        candidates = [
            _EntityCandidate("PROPER", "北京", "spacy_ner", -1, -1, 0.95, 0),
            _EntityCandidate("PROPER", "北京", "spacy_ner", -1, -1, 0.95, 0),
        ]
        result = _resolve_candidates(candidates)
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

    def test_type_error_caught(self):
        """TypeError on noun_chunks (e.g. parser disabled) should be caught."""
        doc = MagicMock()
        doc.text = "Test text"
        doc.__iter__.return_value = []
        doc.ents = []
        doc.noun_chunks.__iter__.side_effect = TypeError("no parser")

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


class TestCjkProperNameGuard:
    """_add_proper_name_candidates should be a no-op for CJK (is_cjk=True)."""

    def test_cjk_guard_returns_immediately(self):
        """_add_proper_name_candidates with is_cjk=True returns without adding candidates."""
        candidates: list = []
        _add_proper_name_candidates([], candidates, is_cjk=True)
        assert candidates == []

    def test_non_cjk_still_works(self):
        """_add_proper_name_candidates with is_cjk=False still processes tokens."""
        candidates: list = []
        # With empty tokens, should not crash and not add anything
        _add_proper_name_candidates([], candidates, is_cjk=False)
        assert candidates == []


class TestCjkQuotedExtraction:
    """CJK quote forms (「」, 『』) are extracted."""

    def test_corner_bracket_quoted(self):
        """「...」 (corner brackets) should be extracted as QUOTED."""
        candidates: list = []
        _add_quoted_candidates("我读了「三体」这本书", candidates, min_len=1)
        texts = {c.text for c in candidates}
        assert "三体" in texts, f"Expected '三体' from 「三体」, got {texts}"

    def test_white_corner_bracket_quoted(self):
        """『...』 (white corner brackets) should be extracted as QUOTED."""
        candidates: list = []
        _add_quoted_candidates("『吾輩は猫である』を読んだ", candidates, min_len=1)
        texts = {c.text for c in candidates}
        assert "吾輩は猫である" in texts, f"Expected '吾輩は猫である' from 『...』, got {texts}"

    def test_cjk_quoted_min_len_one(self):
        """CJK quoted text with min_len=1 accepts short CJK phrases."""
        candidates: list = []
        _add_quoted_candidates("「中」", candidates, min_len=1)
        texts = {c.text for c in candidates}
        assert "中" in texts, f"Expected single CJK char from quotes, got {texts}"

    def test_cjk_quoted_rejects_below_min_len(self):
        """CJK quoted text shorter than min_len is rejected."""
        candidates: list = []
        _add_quoted_candidates("「中」", candidates, min_len=2)
        texts = {c.text for c in candidates}
        assert "中" not in texts, f"Expected single char rejected with min_len=2, got {texts}"


class TestCjkModeBehavior:
    """CJK auto/heuristic mode integration tests."""

    def _make_doc(self, text, ents=None, noun_chunks=None):
        doc = MagicMock()
        doc.text = text
        doc.ents = ents or []
        doc.noun_chunks = noun_chunks or []
        doc.__iter__.return_value = []
        return doc

    def _make_ent(self, text, label="GPE"):
        ent = MagicMock()
        ent.text = text
        ent.label_ = label
        ent.__iter__.return_value = [_make_token(text)]
        return ent

    def test_cjk_auto_does_not_run_topic_phrases(self):
        """CJK auto mode: run_heuristics=False → no topic phrases from noun_chunks."""
        doc = self._make_doc("test", ents=[self._make_ent("北京")])
        # If run_heuristics were True, this would call noun_chunks
        doc.noun_chunks = MagicMock(side_effect=Exception("should not be called"))
        result = _extract_entities_from_doc(doc, entity_extraction="auto", language_code="zh")
        assert any("北京" in e[1] for e in result)

    def test_cjk_ner_mode_does_not_run_topic_phrases(self):
        """CJK ner mode: run_heuristics=False → no topic phrases."""
        doc = self._make_doc("test", ents=[self._make_ent("北京")])
        doc.noun_chunks = MagicMock(side_effect=Exception("should not be called"))
        result = _extract_entities_from_doc(doc, entity_extraction="ner", language_code="zh")
        assert any("北京" in e[1] for e in result)

    def test_cjk_heuristic_skips_ner(self):
        """CJK heuristic mode: use_ner=False → NER entity is not extracted."""
        doc = self._make_doc("北京", ents=[self._make_ent("北京")])
        result = _extract_entities_from_doc(doc, entity_extraction="heuristic", language_code="zh")
        assert not any("北京" in e[1] for e in result)

    def test_cjk_heuristic_runs_topic_phrases(self):
        """CJK heuristic mode: run_heuristics=True, is_cjk=True for topic phrases."""
        token = MagicMock()
        token.text = "机器"
        token.lemma_ = "机器"
        token.pos_ = "NOUN"
        token.dep_ = "compound"
        token.is_stop = False
        token.i = 0

        token2 = MagicMock()
        token2.text = "学习"
        token2.lemma_ = "学习"
        token2.pos_ = "NOUN"
        token2.dep_ = "nmod"
        token2.is_stop = False
        token2.i = 1

        chunk = MagicMock()
        chunk.__iter__.return_value = [token, token2]

        doc = self._make_doc("机器学习", noun_chunks=[chunk])
        doc.__iter__.return_value = [token, token2]
        result = _extract_entities_from_doc(doc, entity_extraction="heuristic", language_code="zh")
        texts = {e[1] for e in result}
        # is_cjk=True allows space-less CJK topic phrases; tokens joined with spaces
        assert "机器 学习" in texts, f"Expected CJK topic phrase '机器 学习', got {texts}"


class TestCjkTopicPhrase:
    """CJK topic phrase extraction with is_cjk=True."""

    def _make_token(self, text, pos_="NOUN", dep_="compound", i=0):
        tok = MagicMock()
        tok.text = text
        tok.text_with_ws = text
        tok.lemma_ = text
        tok.pos_ = pos_
        tok.dep_ = dep_
        tok.is_stop = False
        tok.i = i
        return tok

    def test_cjk_topic_phrase_no_space_accepted(self):
        """is_cjk=True accepts phrases without spaces (CJK languages don't use spaces)."""
        from mem0.utils.entity_extraction import _add_topic_phrase_candidates

        token1 = self._make_token("自然", "NOUN", "compound", i=0)
        token2 = self._make_token("语言", "NOUN", "nmod", i=1)
        token3 = self._make_token("处理", "NOUN", "nmod", i=2)

        chunk = MagicMock()
        chunk.__iter__.return_value = [token1, token2, token3]

        doc = MagicMock()
        doc.noun_chunks = [chunk]

        candidates: list = []
        _add_topic_phrase_candidates(doc, candidates, min_len=1, is_cjk=True)
        texts = {c.text for c in candidates}
        assert "自然 语言 处理" in texts, f"Expected CJK topic phrase, got {texts}"

    def test_non_cjk_topic_phrase_requires_space(self):
        """is_cjk=False rejects phrases without spaces (Latin languages need spaces)."""
        from mem0.utils.entity_extraction import _add_topic_phrase_candidates

        token1 = self._make_token("machine", "NOUN", "compound", i=0)
        token2 = self._make_token("learning", "NOUN", "nmod", i=1)

        chunk = MagicMock()
        chunk.__iter__.return_value = [token1, token2]

        doc = MagicMock()
        doc.noun_chunks = [chunk]

        candidates: list = []
        # min_len=3, is_cjk=False → "machine learning" has space, should be accepted
        _add_topic_phrase_candidates(doc, candidates, min_len=3, is_cjk=False)
        texts = {c.text for c in candidates}
        assert "machine learning" in texts, f"Expected English topic phrase, got {texts}"
