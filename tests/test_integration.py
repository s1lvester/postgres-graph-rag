import os
import pytest
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

    @pytest.mark.skipif(not OPENAI_API_KEY, reason="OPENAI_API_KEY not set")
    def test_openai_e2e(self):
        rag = PostgresGraphRAG(
            postgres_url=POSTGRES_URL,
            openai_api_key=OPENAI_API_KEY,
            config=OPENAI_DEFAULT_CONFIG,
        )
        self._run_e2e_flow(rag, "OpenAI")

    @pytest.mark.skipif(not GOOGLE_API_KEY, reason="GOOGLE_API_KEY not set")
    def test_google_e2e(self):
        rag = PostgresGraphRAG(
            postgres_url=POSTGRES_URL,
            google_api_key=GOOGLE_API_KEY,
            config=GOOGLE_DEFAULT_CONFIG,
        )
        self._run_e2e_flow(rag, "Google")

    def _run_e2e_flow(self, rag, provider_name):
        with rag.db._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("DROP TABLE IF EXISTS edges CASCADE")
                cur.execute("DROP TABLE IF EXISTS nodes CASCADE")
                conn.commit()

        rag.setup()
        test_text = f"The {provider_name} team developed a new RAG system. This system uses Postgres."
        rag.ingest(test_text)
        context = rag.query(
            f"What did the {provider_name} team develop?", hops=2
        )
        assert provider_name in context
        assert "Postgres" in context
