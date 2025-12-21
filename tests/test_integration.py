import os
import pytest
import pytest_asyncio
from postgres_graph_rag import PostgresGraphRAG
from postgres_graph_rag.models import (
    OPENAI_DEFAULT_CONFIG,
    GOOGLE_DEFAULT_CONFIG,
)
from dotenv import load_dotenv

load_dotenv()

POSTGRES_URL = os.getenv("POSTGRES_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


@pytest.mark.skipif(not POSTGRES_URL, reason="POSTGRES_URL not set")
class TestIntegration:

    @pytest.mark.asyncio
    @pytest.mark.skipif(not OPENAI_API_KEY, reason="OPENAI_API_KEY not set")
    async def test_openai_e2e(self):
        # Frictionless instantiation
        rag = PostgresGraphRAG(
            postgres_url=POSTGRES_URL,
            openai_api_key=OPENAI_API_KEY,
            config=OPENAI_DEFAULT_CONFIG,
        )
        try:
            await self._run_e2e_flow(rag, "OpenAI")
        finally:
            await rag.close()

    @pytest.mark.asyncio
    @pytest.mark.skipif(not GOOGLE_API_KEY, reason="GOOGLE_API_KEY not set")
    async def test_google_e2e(self):
        # Frictionless instantiation
        rag = PostgresGraphRAG(
            postgres_url=POSTGRES_URL,
            google_api_key=GOOGLE_API_KEY,
            config=GOOGLE_DEFAULT_CONFIG,
        )
        try:
            await self._run_e2e_flow(rag, "Google")
        finally:
            await rag.close()

    async def _run_e2e_flow(self, rag, provider_name):
        await rag.db._init_pool()
        async with rag.db.pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute("DROP TABLE IF EXISTS graph_edges CASCADE")
                await cur.execute("DROP TABLE IF EXISTS graph_nodes CASCADE")
                await conn.commit()

        await rag.setup()

        namespace = f"test-{provider_name.lower()}"
        test_text = f"The {provider_name} team developed a new RAG system. This system uses Postgres."

        # Ingest with namespace
        await rag.add_texts(test_text, namespace=namespace)

        # Query with same namespace
        context = await rag.query(
            f"What did the {provider_name} team develop?",
            namespace=namespace,
            hops=2,
        )

        assert provider_name in context
        assert "Postgres" in context or "PostgreSQL" in context

        # Verify other namespace is empty
        empty_context = await rag.query("Any team?", namespace="other-empty-ns")
        assert "No relevant context found" in empty_context
