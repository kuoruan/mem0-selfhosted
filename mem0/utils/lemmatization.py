"""
BM25 lemmatization for consistent keyword matching.

Uses spaCy's lemmatizer for better handling of:
- Verb forms: attending/attends/attended -> attend
- Comparatives/superlatives: older/oldest -> old
- Plurals: memories -> memory
- Avoids over-stemming: organization != organize

Also includes original -ing forms alongside lemmas to handle cases
where spaCy's context-dependent lemmatization produces inconsistent
results (e.g., "meeting" as noun vs verb -> different lemmas).
"""

import logging
from typing import Optional

from mem0.configs.nlp.config import NlpConfig

logger = logging.getLogger(__name__)


def lemmatize_for_bm25(text: str, *, nlp_config: Optional[NlpConfig] = None) -> str:
    """Lemmatize text for BM25 matching.

    Returns space-joined lemmas for full-text search. Falls back to
    the original text if spaCy is unavailable.
    """
    from mem0.utils.spacy_models import get_nlp_lemma

    nlp = get_nlp_lemma(nlp_config)
    if nlp is None:
        return text

    doc = nlp(text)
    tokens = []

    for token in doc:
        if token.is_punct or token.is_stop:
            continue

        lemma = token.lemma_.lower()
        if lemma.isalnum():
            tokens.append(lemma)

        # Also add original if it ends in -ing and differs from lemma.
        # This handles noun/verb ambiguity (meeting/meet, attending/attend).
        token_text = token.text.lower()
        if token_text.endswith("ing") and token_text != lemma and token_text.isalnum():
            tokens.append(token_text)

    return " ".join(tokens)
