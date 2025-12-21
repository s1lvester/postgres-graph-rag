import os
import pytest
from postgres_graph_rag import PostgresGraphRAG
from postgres_graph_rag.models import OPENAI_DEFAULT_CONFIG
from dotenv import load_dotenv

load_dotenv()

POSTGRES_URL = os.getenv("POSTGRES_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


@pytest.fixture
def rag_class():
    if not POSTGRES_URL or not OPENAI_API_KEY:
        pytest.skip("POSTGRES_URL or OPENAI_API_KEY not set")
    return PostgresGraphRAG


@pytest.mark.asyncio
async def test_entity_resolution_across_chunks(rag_class):
    """
    Test Case: Entity Resolution & Merging
    Goal: Verify that if an entity is mentioned in two different ingestion calls,
    the system resolves them to the same database node and merges their metadata.
    """
    rag = rag_class(
        postgres_url=POSTGRES_URL,
        openai_api_key=OPENAI_API_KEY,
        config=OPENAI_DEFAULT_CONFIG,
    )
    try:
        # Clean state
        await rag.db._init_pool()
        async with rag.db.pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute("DROP TABLE IF EXISTS graph_edges CASCADE")
                await cur.execute("DROP TABLE IF EXISTS graph_nodes CASCADE")
                await conn.commit()

        await rag.setup()

        # 1. Ingest first fact
        await rag.add_texts(
            "Apple Inc. is a technology company.", namespace="res-test"
        )

        # 2. Ingest second fact mentioning the same entity
        await rag.add_texts(
            "Apple Inc. is headquartered in Cupertino.", namespace="res-test"
        )

        # 3. Verify database state
        async with rag.db.pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT count(*) FROM graph_nodes WHERE namespace = 'res-test'"
                )
                row = await cur.fetchone()
                node_count = row["count"]
                assert node_count <= 3
    finally:
        await rag.close()


@pytest.mark.asyncio
async def test_deep_path_traversal(rag_class):
    """
    Test Case: Deep Path Traversal (3+ Hops)
    Goal: Ensure the recursive CTE can navigate through a chain of relationships
    without losing context or failing on cycle detection.
    """
    rag = rag_class(
        postgres_url=POSTGRES_URL,
        openai_api_key=OPENAI_API_KEY,
        config=OPENAI_DEFAULT_CONFIG,
    )
    try:
        await rag.setup()
        # Building a chain: A -> B -> C -> D
        await rag.add_texts(
            "Johny Srouji leads the Hardware Team.", namespace="path-test"
        )
        await rag.add_texts(
            "The Hardware Team designed the M4 Chip.", namespace="path-test"
        )
        await rag.add_texts(
            "The M4 Chip uses ARM Architecture.", namespace="path-test"
        )

        # Querying the start of the chain with 3 hops should reach the end
        context = await rag.query(
            "Tell me about Johny Srouji's work", namespace="path-test", hops=3
        )

        assert "Johny Srouji" in context
        assert "M4" in context
        assert "ARM" in context
    finally:
        await rag.close()


@pytest.mark.asyncio
async def test_strict_namespace_isolation(rag_class):
    """
    Test Case: Strict Namespace Isolation
    Goal: Confirm that data in one namespace is completely invisible to queries
    performed in another namespace, preventing data leakage.
    """
    rag = rag_class(
        postgres_url=POSTGRES_URL,
        openai_api_key=OPENAI_API_KEY,
        config=OPENAI_DEFAULT_CONFIG,
    )
    try:
        await rag.setup()
        # Fact for Namespace A
        await rag.add_texts(
            "The secret code is 12345.", namespace="internal-vault"
        )

        # Fact for Namespace B
        await rag.add_texts(
            "The secret code is 99999.", namespace="guest-vault"
        )

        # Querying Namespace A should NOT see facts from B
        context_a = await rag.query(
            "What is the secret code?", namespace="internal-vault"
        )
        assert "12345" in context_a
        assert "99999" not in context_a

        # Querying Namespace B should NOT see facts from A
        context_b = await rag.query(
            "What is the secret code?", namespace="guest-vault"
        )
        assert "99999" in context_b
        assert "12345" not in context_b
    finally:
        await rag.close()


@pytest.mark.asyncio
async def test_metadata_integrity_via_jsonb_merge(rag_class):
    """
    Test Case: Metadata Integrity (JSONB Merge)
    Goal: Verify that the Postgres || operator correctly merges metadata
    instead of overwriting it when the same entity/edge is updated.
    """
    rag = rag_class(
        postgres_url=POSTGRES_URL,
        openai_api_key=OPENAI_API_KEY,
        config=OPENAI_DEFAULT_CONFIG,
    )
    try:
        await rag.setup()
        # 1. First ingestion with custom metadata
        await rag.add_texts(
            "London is a city.",
            namespace="meta-test",
            metadata={"source": "book_1"},
        )

        # 2. Second ingestion for the same fact with different source
        await rag.add_texts(
            "London is a city.",
            namespace="meta-test",
            metadata={"quality": "high"},
        )

        # 3. Check if both fields exist in the database
        async with rag.db.pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT metadata FROM graph_nodes WHERE content ILIKE '%London%' AND namespace = 'meta-test'"
                )
                row = await cur.fetchone()
                metadata = row["metadata"]

                assert metadata.get("source") == "book_1"
                assert metadata.get("quality") == "high"
    finally:
        await rag.close()
