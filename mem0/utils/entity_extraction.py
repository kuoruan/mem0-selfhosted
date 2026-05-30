"""
Entity extraction from text using spaCy NLP.

Extracts four types of entities from text:
- **Proper nouns**: Capitalized multi-word sequences (person names, places, brands)
- **Quoted text**: Text in single or double quotes (titles, specific terms)
- **Noun compounds**: Multi-word noun phrases with specific modifiers (e.g., "machine learning")
- **Noun fallback**: Single nouns from circumstantial compound patterns

Public API:
    extract_entities(text: str) -> List[Tuple[str, str]]

Internal:
    _extract_entities_from_doc(doc) -> List[Tuple[str, str]]
"""

import logging
import re
from typing import List, Optional, Tuple

from mem0.configs.nlp.config import CJK_LANGUAGES, EntityExtractionMode, NlpConfig

logger = logging.getLogger(__name__)

_NER_LABELS_AS_PROPER = frozenset(
    {"PERSON", "ORG", "GPE", "LOC", "PRODUCT", "EVENT", "WORK_OF_ART", "FAC", "LANGUAGE", "NORP", "LAW"}
)

# Words that are too generic to be useful as entity heads
_GENERIC_HEADS = {
    "thing", "stuff", "way", "time", "experience", "situation", "case",
    "fact", "matter", "issue", "idea", "thought", "feeling", "place",
    "area", "part", "kind", "type", "sort", "lot", "bit", "day", "year",
    "week", "month", "moment", "instance", "example", "technique",
    "method", "approach", "process", "step", "tool", "result", "outcome",
    "goal", "task", "item", "topic", "scale", "size", "level", "degree",
    "amount", "number", "style", "look", "color", "colour", "shape",
    "form", "piece", "section", "side", "end", "edge", "surface", "point",
}

# Modifiers that describe circumstance, not content
_CIRCUMSTANTIAL_MODS = {
    "solo", "individual", "team", "group", "joint", "collaborative",
    "first", "last", "next", "previous", "final", "initial", "main", "side",
}

# Adjectives too vague to make a compound entity specific
_NON_SPECIFIC_ADJ = {
    "many", "few", "several", "some", "any", "all", "most", "more",
    "less", "much", "little", "enough", "various", "numerous", "multiple",
    "countless", "great", "good", "bad", "nice", "terrible", "awful",
    "awesome", "amazing", "wonderful", "horrible", "excellent", "poor",
    "best", "worst", "fine", "okay", "new", "old", "recent", "past",
    "future", "current", "previous", "next", "last", "first", "latest",
    "early", "late", "former", "modern", "ancient", "big", "small",
    "large", "tiny", "huge", "enormous", "long", "short", "tall", "high",
    "low", "wide", "narrow", "thick", "thin", "deep", "shallow",
    "similar", "different", "same", "other", "another", "such", "certain",
    "important", "main", "major", "minor", "key", "primary", "real",
    "actual", "true", "whole", "entire", "full", "complete", "total",
    "basic", "simple", "interesting", "boring", "exciting", "special",
    "particular", "general", "common", "unique", "rare", "typical",
    "usual", "normal", "regular", "possible", "likely", "potential",
    "available", "necessary", "only", "solo", "individual", "team",
    "group", "joint", "collaborative", "final", "initial", "side",
}

# Generic tail words to strip from compound entities
_GENERIC_ENDINGS = {
    "work", "works", "job", "jobs", "task", "tasks", "stuff", "things",
    "thing", "info", "information", "details", "data", "content",
    "material", "materials", "activities", "activity", "efforts", "effort",
    "options", "option", "choices", "choice", "results", "result",
    "output", "outputs", "products", "product", "items", "item",
}

# Capitalized single words that are too generic to be proper nouns
_GENERIC_CAPS = {
    "works", "items", "things", "stuff", "resources", "options", "tips",
    "ideas", "steps", "ways", "methods", "tools", "features", "benefits",
    "examples", "details", "notes", "instructions", "guidelines",
    "recommendations", "suggestions", "overview", "summary", "conclusion",
    "introduction", "pros", "cons", "advantages", "disadvantages",
}

# Markdown/formatting markers to skip during extraction
_FORMATTING_MARKERS = {"*", "-", "+", "\u2022", "\u2013", "\u2014", "#", "##", "###", "**", "__"}


def _is_sentence_start(tokens: list, idx: int) -> bool:
    """Check if a token is at the start of a sentence or after formatting."""
    if idx == 0:
        return True
    tok = tokens[idx]
    if tok.is_sent_start:
        return True
    prev = tokens[idx - 1].text
    return prev in ".!?:" or prev in _FORMATTING_MARKERS or "\n" in prev


def _strip_generic_ending(toks: list) -> list:
    """Remove generic trailing words from compound token sequences."""
    if len(toks) <= 1:
        return toks
    last = toks[-1].lemma_.lower() if hasattr(toks[-1], "lemma_") else toks[-1].lower()
    return toks[:-1] if last in _GENERIC_ENDINGS and len(toks) > 2 else toks


def _lemmatize_compound(toks: list) -> str:
    """Join compound tokens, lemmatizing nouns."""
    return " ".join(t.lemma_ if t.pos_ == "NOUN" else t.text for t in toks)


_ARTIFACT_STAR_RE = re.compile(r"\s\*\s|\s\*$|^\*\s")
_ARTIFACT_PREFIXES = ("\u2022", "-", "+", "\u2013", "\u2014")


def _has_artifacts(txt: str) -> bool:
    """Check for formatting artifacts that indicate non-entity text."""
    return any(
        [
            "**" in txt or "__" in txt or ":*" in txt,
            _ARTIFACT_STAR_RE.search(txt),
            "  " in txt or "\n" in txt or "\t" in txt,
            len(txt) > 100,
            txt.startswith(_ARTIFACT_PREFIXES),
        ]
    )


def extract_entities(text: str, *, nlp_config: Optional[NlpConfig] = None) -> List[Tuple[str, str]]:
    """Extract named entities, quoted text, and noun compounds from text.

    This is the public API that accepts a string. It loads the spaCy model
    internally and delegates to _extract_entities_from_doc().

    Args:
        text: Input text to extract entities from.

    Returns:
        Deduplicated list of (entity_type, entity_text) tuples.
        Entity types: PROPER, QUOTED, COMPOUND, NOUN.
        Returns empty list if spaCy is unavailable.
    """
    from mem0.utils.spacy_models import get_nlp_full

    config = nlp_config or NlpConfig()
    nlp = get_nlp_full(config)
    if nlp is None:
        return []

    doc = nlp(text)
    return _extract_entities_from_doc(
        doc,
        entity_extraction=config.entity_extraction,
        language_code=config.language_code,
    )


def extract_entities_batch(
    texts: List[str],
    batch_size: int = 32,
    *,
    nlp_config: Optional[NlpConfig] = None,
) -> List[List[Tuple[str, str]]]:
    """Extract entities from multiple texts using spaCy's nlp.pipe() for batched NER.

    Uses spaCy's efficient batch processing pipeline instead of calling
    nlp() individually per text. Significantly faster for multiple texts.

    Args:
        texts: List of input texts to extract entities from.
        batch_size: Number of texts to process in each spaCy batch.

    Returns:
        List of entity lists, one per input text. Each entity list contains
        (entity_type, entity_text) tuples. Returns list of empty lists if
        spaCy is unavailable.
    """
    if not texts:
        return []

    from mem0.utils.spacy_models import get_nlp_full

    config = nlp_config or NlpConfig()
    nlp = get_nlp_full(config)
    if nlp is None:
        return [[] for _ in texts]

    results = []
    for doc in nlp.pipe(texts, batch_size=batch_size):
        results.append(
            _extract_entities_from_doc(
                doc,
                entity_extraction=config.entity_extraction,
                language_code=config.language_code,
            )
        )
    return results


def _extract_entities_from_doc(
    doc,
    *,
    entity_extraction: EntityExtractionMode = "auto",
    language_code: str = "en",
) -> List[Tuple[str, str]]:
    """Extract entities from a spaCy Doc object.

    Ported from platform's shared.core.utils.entity_extraction.extract_entities().
    """
    # Mode matrix:
    # auto + non-CJK  → heuristic only (original behavior)
    # auto + CJK      → NER + noun_chunks (skip PROPER heuristic)
    # ner             → NER + quoted only
    # heuristic       → heuristic only (no NER)
    if entity_extraction == "ner":
        run_ner, run_heuristics = True, False
    elif entity_extraction == "heuristic":
        run_ner, run_heuristics = False, True
    else:
        run_ner = language_code in CJK_LANGUAGES
        run_heuristics = True

    entities: List[Tuple[str, str]] = []
    text = doc.text
    tokens = list(doc)

    # === spaCy NER ===
    if run_ner:
        for ent in doc.ents:
            ent_text = ent.text.strip()
            min_len = 1 if language_code in CJK_LANGUAGES else 3
            if len(ent_text) < min_len:
                continue
            entity_type = "PROPER" if ent.label_ in _NER_LABELS_AS_PROPER else "COMPOUND"
            entities.append((entity_type, ent_text))

    # === PROPER NOUN SEQUENCES (Latin-script heuristics) ===
    i = 0
    if not run_heuristics or run_ner:
        i = len(tokens)
    while i < len(tokens):
        tok = tokens[i]
        if tok.text in _FORMATTING_MARKERS:
            i += 1
            continue
        is_cap = tok.text and tok.text[0].isupper()
        is_label = i + 1 < len(tokens) and tokens[i + 1].text == ":"

        if is_cap and not is_label and tok.pos_ in {"PROPN", "NOUN", "ADJ"}:
            seq = [(tok, i)]
            j = i + 1
            while j < len(tokens):
                t = tokens[j]
                if (t.text and t.text[0].isupper()) or t.text.lower() in {
                    "'s", "of", "the", "in", "and", "for", "at", "is",
                }:
                    seq.append((t, j))
                    j += 1
                else:
                    break
            # Strip trailing function words
            while seq and seq[-1][0].text.lower() in {"of", "the", "in", "and", "for", "at", "is", "'s"}:
                seq.pop()
            if seq:
                has_mid_cap = any(
                    not _is_sentence_start(tokens, idx)
                    for (t, idx) in seq
                    if t.text[0].isupper() and t.text.lower() not in {"'s", "of", "the", "in", "and", "for", "at", "is"}
                )
                if has_mid_cap:
                    phrase = "".join(t.text_with_ws for (t, idx) in seq).strip()
                    if len(phrase) > 2:
                        entities.append(("PROPER", phrase))
            i = j
        else:
            i += 1

    # === QUOTED TEXT ===
    for m in re.finditer(r'"([^"]+)"', text):
        if len(m.group(1).strip()) > 2:
            entities.append(("QUOTED", m.group(1).strip()))
    for m in re.finditer(r"(?:^|[\s\(\[{,;])'([^']+)'(?=[\s\.,;:!?\)\]]|$)", text):
        if len(m.group(1).strip()) > 2:
            entities.append(("QUOTED", m.group(1).strip()))

    # === NOUN-NOUN COMPOUNDS ===
    if not run_heuristics:
        return _finalize_entities(entities, language_code=language_code)

    try:
        noun_chunks = list(doc.noun_chunks)
    except (NotImplementedError, ValueError):
        noun_chunks = []

    for chunk in noun_chunks:
        chunk_tokens = list(chunk)
        split_indices: list = []
        poss_splits: list = []
        for idx, tok in enumerate(chunk_tokens):
            if tok.dep_ == "case" and tok.text in {"'s", "\u2019s", "'"}:
                split_indices.append(idx)
                poss_splits.append(idx)
            elif tok.pos_ == "PUNCT" and tok.text in {"'", '"', "\u2018", "\u2019", "\u201c", "\u201d"}:
                split_indices.append(idx)

        if split_indices:
            groups: list = []
            prev = 0
            for split_idx in split_indices:
                if split_idx > prev:
                    groups.append(chunk_tokens[prev:split_idx])
                if split_idx in poss_splits:
                    next_split = next((s for s in split_indices if s > split_idx), None)
                    owned = chunk_tokens[split_idx + 1: next_split if next_split else len(chunk_tokens)]
                    if owned:
                        first_content = next((t for t in owned if t.pos_ not in {"PUNCT", "PART"}), None)
                        if not (first_content and first_content.text and first_content.text[0].isupper()):
                            prev = next_split if next_split else len(chunk_tokens)
                            continue
                prev = split_idx + 1
            if prev < len(chunk_tokens):
                groups.append(chunk_tokens[prev:])
        else:
            groups = [chunk_tokens]

        for group in groups:
            if not group:
                continue
            head = next((t for t in reversed(group) if t.pos_ in {"NOUN", "PROPN"}), None)
            if not head:
                continue
            head_generic = head.lemma_.lower() in _GENERIC_HEADS
            content = [
                t
                for t in group
                if t.pos_ not in {"DET", "PRON", "PUNCT", "PART", "ADP", "SCONJ", "NUM"} and (t.pos_ == "ADJ" or not t.is_stop)
            ]
            if not content:
                continue

            compound_toks = [t for t in content if t.dep_ == "compound"]
            adj_toks = [t for t in content if t.pos_ == "ADJ" or t.dep_ == "amod"]
            has_spec_adj = any(t.lemma_.lower() not in _NON_SPECIFIC_ADJ for t in adj_toks)
            if head_generic and not has_spec_adj and not compound_toks:
                continue

            if compound_toks:
                is_circ = any(t.lemma_.lower() in _CIRCUMSTANTIAL_MODS for t in compound_toks)
                if is_circ:
                    val = head.lemma_ if head.pos_ == "NOUN" else head.text
                    if len(val) > 2:
                        entities.append(("NOUN", val))
                else:
                    filtered = _strip_generic_ending(
                        [t for t in content if not (t.pos_ == "ADJ" and t.lemma_.lower() in _NON_SPECIFIC_ADJ)]
                    )
                    if filtered:
                        phrase = _lemmatize_compound(filtered)
                        if len(phrase) > 3 and " " in phrase:
                            entities.append(("COMPOUND", phrase))
            elif len(content) > 1 and has_spec_adj:
                filtered = _strip_generic_ending(
                    [t for t in content if not ((t.pos_ == "ADJ" or t.dep_ == "amod") and t.lemma_.lower() in _NON_SPECIFIC_ADJ)]
                )
                if filtered:
                    phrase = _lemmatize_compound(filtered)
                    if len(phrase) > 3 and " " in phrase:
                        entities.append(("COMPOUND", phrase))

    # === FALLBACK: Mis-tagged VERB heads ===
    processed = {e[1].lower() for e in entities if e[0] == "COMPOUND"}
    generic_verb_heads = _GENERIC_HEADS | {"find", "buy", "purchase", "sale", "deal", "trip", "visit"}

    for tok in doc:
        if tok.pos_ == "VERB" and tok.dep_ in {"pobj", "dobj", "nsubj"}:
            comps = sorted([t for t in tok.children if t.dep_ == "compound"], key=lambda t: t.i)
            if comps:
                phrase_toks = comps if tok.lemma_.lower() in generic_verb_heads else comps + [tok]
                phrase = " ".join(t.text for t in phrase_toks)
                if phrase.lower() not in processed and len(phrase) > 3 and " " in phrase:
                    entities.append(("COMPOUND", phrase))
                    processed.add(phrase.lower())

    return _finalize_entities(entities, language_code=language_code)


def _finalize_entities(entities: List[Tuple[str, str]], language_code: str = "en") -> List[Tuple[str, str]]:
    # === DEDUPLICATION & CLEANUP ===
    min_len = 1 if language_code in CJK_LANGUAGES else 3
    seen: set = set()
    deduped = []
    for t, e in entities:
        k = e.lower().strip()
        if k not in seen and len(k) >= min_len:
            seen.add(k)
            deduped.append((t, e))

    cleaned: List[Tuple[str, str]] = []
    for etype, etext in deduped:
        txt = re.sub(r"^\*+\s*|\s*\*+$", "", etext.strip())
        txt = re.sub(r"\s*:+$", "", txt)
        txt = re.sub(r"^\d+\s*\.\s*", "", txt)
        if not txt or len(txt) < min_len or _has_artifacts(txt):
            continue
        if etype == "PROPER" and " " not in txt and txt.lower() in _GENERIC_CAPS:
            continue
        cleaned.append((etype, txt))

    # Keep best type per entity (PROPER > COMPOUND > QUOTED > NOUN)
    type_pri = {"PROPER": 0, "COMPOUND": 1, "QUOTED": 2, "NOUN": 3, "VERB": 4}
    best: dict = {}
    for t, e in cleaned:
        k = e.lower()
        if k not in best or type_pri.get(t, 99) < type_pri.get(best[k][0], 99):
            best[k] = (t, e)
    deduped = list(best.values())

    # Remove entities that are substrings of longer entities
    # Skip for CJK languages where short entities are frequently valid substrings of longer ones
    if language_code not in CJK_LANGUAGES:
        all_lower = [e[1].lower() for e in deduped]
        deduped = [(t, e) for t, e in deduped if not any(e.lower() != o and e.lower() in o for o in all_lower)]
    return deduped
