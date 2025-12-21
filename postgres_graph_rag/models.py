from typing import TypedDict, Optional


class ProviderConfig(TypedDict):
    extraction_model: str
    embedding_model: str
    dimension: int


# Default configurations for the current era
OPENAI_DEFAULT_CONFIG: ProviderConfig = {
    "extraction_model": "gpt-5-nano-2025-08-07",
    "embedding_model": "text-embedding-3-small",
    "dimension": 1536,
}

GOOGLE_DEFAULT_CONFIG: ProviderConfig = {
    "extraction_model": "gemini-2.5-flash-lite",
    "embedding_model": "text-embedding-004",
    "dimension": 768,
}
