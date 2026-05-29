from mem0.configs.nlp.config import NlpConfig
from mem0.utils.lemmatization import lemmatize_for_bm25
from mem0.utils import spacy_models  # module import: tests call get_nlp_* and may use reset_spacy_cache


class TestNlpDisabled:
    def test_lemmatize_returns_original_when_disabled(self):
        nlp = NlpConfig(enabled=False)
        text = "The cats are running"
        assert lemmatize_for_bm25(text, nlp_config=nlp) == text

    def test_spacy_loaders_return_none_when_disabled(self):
        nlp = NlpConfig(enabled=False)
        assert spacy_models.get_nlp_full(nlp) is None
        assert spacy_models.get_nlp_lemma(nlp) is None
