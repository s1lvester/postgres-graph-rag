import json
from typing import List, Optional
from pydantic import BaseModel, Field
import openai
from google import genai
from google.genai import types
from .models import ProviderConfig, OPENAI_DEFAULT_CONFIG, GOOGLE_DEFAULT_CONFIG


class Triplet(BaseModel):
    subject: str = Field(
        ..., description="The entity that is the subject of the relationship"
    )
    predicate: str = Field(
        ..., description="The relationship between the subject and the object"
    )
    object: str = Field(
        ..., description="The entity that is the object of the relationship"
    )


class ExtractionResult(BaseModel):
    triplets: List[Triplet]


class LLMExtractor:
    def __init__(
        self,
        config: ProviderConfig,
        openai_api_key: Optional[str] = None,
        google_api_key: Optional[str] = None,
    ):
        self.config = config
        self.openai_client = None
        self.google_client = None

        if openai_api_key:
            self.openai_client = openai.AsyncOpenAI(api_key=openai_api_key)
        if google_api_key:
            # Use the .aio attribute for async operations
            self.google_client = genai.Client(api_key=google_api_key).aio

    async def extract_triplets(self, text: str) -> List[Triplet]:
        """Extracts entities and relationships from text using the configured LLM."""
        prompt = (
            "You are an expert knowledge graph extractor. Your task is to decompose the given text "
            "into atomic subject-predicate-object triplets.\n\n"
            "Guidelines:\n"
            "1. Entities (Subject/Object): Use proper nouns or specific concepts. Avoid pronouns (he, she, it, they).\n"
            "2. Predicates: Use short, active verbs or clear relationship terms (e.g., 'works_at', 'developed', 'is_located_in').\n"
            "3. Atomicity: Each triplet must represent a single, distinct fact.\n"
            "4. Normalization: Clean up entity names (e.g., 'Apple Inc.' and 'Apple' should be 'Apple').\n"
            "5. Context: Only extract facts explicitly stated in the text.\n\n"
            "Return a JSON list of triplets with the keys: 'subject', 'predicate', 'object'."
        )

        model = self.config["extraction_model"]
        if "gpt" in model and self.openai_client:
            return await self._extract_openai(text, prompt)
        elif "gemini" in model and self.google_client:
            return await self._extract_google(text, prompt)
        else:
            raise ValueError(f"Model {model} not supported or API key missing.")

    async def _extract_openai(self, text: str, prompt: str) -> List[Triplet]:
        completion = await self.openai_client.beta.chat.completions.parse(
            model=self.config["extraction_model"],
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": text},
            ],
            response_format=ExtractionResult,
        )
        return completion.choices[0].message.parsed.triplets

    async def _extract_google(self, text: str, prompt: str) -> List[Triplet]:
        response = await self.google_client.models.generate_content(
            model=self.config["extraction_model"],
            contents=[prompt, text],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=ExtractionResult,
            ),
        )
        if response.parsed:
            return response.parsed.triplets
        return []

    async def get_embedding(self, text: str) -> List[float]:
        model = self.config["embedding_model"]
        if self.openai_client and "text-embedding" in model:
            response = await self.openai_client.embeddings.create(
                input=text, model=model
            )
            return response.data[0].embedding
        elif self.google_client and (
            "text-embedding" in model or "embedding" in model
        ):
            response = await self.google_client.models.embed_content(
                model=model, contents=text
            )
            return response.embeddings[0].values
        raise ValueError("API key missing or provider mismatch for embeddings.")
