from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

# ISO 639-1 language code -> default spaCy model package name.
LANGUAGE_MODEL_MAP: dict[str, str] = {
    "en": "en_core_web_sm",
    "zh": "zh_core_web_sm",
    "de": "de_core_news_sm",
    "fr": "fr_core_news_sm",
    "es": "es_core_news_sm",
    "it": "it_core_news_sm",
    "pt": "pt_core_news_sm",
    "nl": "nl_core_news_sm",
    "ja": "ja_core_news_sm",
    "ko": "ko_core_news_sm",
    "xx": "xx_ent_wiki_sm",  # multilingual NER fallback (set language="xx" explicitly)
}

# Languages where capitalization-based PROPER noun heuristics do not apply.
CJK_LANGUAGES = frozenset({"zh", "ja", "ko"})

EntityExtractionMode = Literal["auto", "heuristic", "ner"]
SUPPORTED_LANGUAGES = frozenset(LANGUAGE_MODEL_MAP.keys())


class NlpConfig(BaseModel):
    """spaCy settings for BM25 lemmatization and entity extraction.

    Typical usage:
        NlpConfig(language="zh")
        NlpConfig(model="en_core_web_md")
        NlpConfig(language="en", lemma_model="en_core_web_sm")  # lighter lemma pipeline
    """

    enabled: bool = Field(
        default=True,
        description="When False, skip spaCy (semantic search only; no BM25 lemmatization or entity linking).",
    )
    language: str = Field(
        default="en",
        description=(
            f"ISO 639-1 language code. Selects a default spaCy model when `model` is unset. "
            f"Supported: {', '.join(sorted(SUPPORTED_LANGUAGES))}."
        ),
    )
    model: Optional[str] = Field(
        default=None,
        description="spaCy package name for entity extraction (overrides `language`).",
    )
    lemma_model: Optional[str] = Field(
        default=None,
        description="spaCy package for BM25 lemmatization; defaults to `model` or the language default.",
    )
    entity_extraction: EntityExtractionMode = Field(
        default="auto",
        description=(
            "Entity extraction strategy: "
            "`auto` — NER for zh/ja/ko, capitalization heuristics otherwise; "
            "`ner` — spaCy NER only; "
            "`heuristic` — English-style rules (poor for CJK)."
        ),
    )
    auto_download: bool = Field(
        default=True,
        description="Download missing spaCy models on first load.",
    )

    model_config = {"extra": "forbid"}

    @field_validator("language")
    @classmethod
    def _normalize_language(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("language must be a non-empty ISO 639-1 code")
        return value.strip().lower()

    @model_validator(mode="after")
    def _validate_language_when_no_model(self) -> "NlpConfig":
        if not self.enabled:
            return self
        if not self.model and self.language_code not in SUPPORTED_LANGUAGES:
            supported = ", ".join(sorted(SUPPORTED_LANGUAGES))
            raise ValueError(
                f"Unsupported NLP language '{self.language}'. "
                f"Set `model` explicitly, or use one of: {supported}"
            )
        return self

    @property
    def language_code(self) -> str:
        """Primary language subtag (e.g. zh from zh-cn or zh_cn)."""
        return self.language.replace("_", "-").split("-")[0]

    @property
    def uses_ner_extraction(self) -> bool:
        """Whether entity extraction should prefer spaCy NER over capitalization heuristics."""
        if self.entity_extraction == "ner":
            return True
        if self.entity_extraction == "heuristic":
            return False
        return self.language_code in CJK_LANGUAGES

    def resolve_model(self, variant: Literal["full", "lemma"] = "full") -> Optional[str]:
        """Resolve the spaCy package name for entity extraction (full) or BM25 (lemma)."""
        if not self.enabled:
            return None
        if variant == "lemma" and self.lemma_model:
            return self.lemma_model
        if self.model:
            return self.model
        mapped = LANGUAGE_MODEL_MAP.get(self.language_code)
        if mapped is None:
            supported = ", ".join(sorted(SUPPORTED_LANGUAGES))
            raise ValueError(
                f"Unsupported NLP language '{self.language}'. "
                f"Set `model` explicitly, or use one of: {supported}"
            )
        return mapped
