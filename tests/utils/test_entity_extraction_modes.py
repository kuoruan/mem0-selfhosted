"""Tests for entity_extraction mode behavior (no spaCy model required)."""

from unittest.mock import MagicMock

import pytest

from mem0.utils.entity_extraction import _extract_entities_from_doc


def _make_doc(text: str, *, ents=None, noun_chunks=None):
    doc = MagicMock()
    doc.text = text
    doc.ents = ents or []

    if noun_chunks is None:
        noun_chunks = []

    doc.noun_chunks = noun_chunks

    # Empty token loop for PROPER heuristic path
    doc.__iter__ = lambda self: iter([])
    return doc


class TestEntityExtractionModes:
    def test_ner_mode_skips_noun_chunk_heuristics(self):
        ent = MagicMock()
        ent.text = "OpenAI"
        ent.label_ = "ORG"

        chunk = MagicMock()
        chunk.__iter__ = lambda self: iter([MagicMock(text="machine", lemma_="machine", pos_="NOUN")])

        doc = _make_doc("OpenAI and machine learning", ents=[ent], noun_chunks=[chunk])
        result = _extract_entities_from_doc(doc, entity_extraction="ner", language_code="en")

        texts = [e[1] for e in result]
        assert "OpenAI" in texts
        assert not any("machine" in t for t in texts)

    def test_ner_mode_keeps_quoted_text(self):
        doc = _make_doc('She read "The Great Gatsby"')
        result = _extract_entities_from_doc(doc, entity_extraction="ner", language_code="en")
        assert any("Great Gatsby" in e[1] for e in result)

    def test_heuristic_mode_skips_spacy_ner(self):
        ent = MagicMock()
        ent.text = "OpenAI"
        ent.label_ = "ORG"

        doc = _make_doc("OpenAI", ents=[ent])
        result = _extract_entities_from_doc(doc, entity_extraction="heuristic", language_code="en")
        assert not any("OpenAI" in e[1] for e in result)
