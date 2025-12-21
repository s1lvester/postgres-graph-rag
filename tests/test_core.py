import pytest
from unittest.mock import AsyncMock
from postgres_graph_rag import PostgresGraphRAG
from postgres_graph_rag.extractor import Triplet
from postgres_graph_rag.models import OPENAI_DEFAULT_CONFIG


@pytest.fixture
def mock_db_class(mocker):
    return mocker.patch("postgres_graph_rag.core.DatabaseManager")


@pytest.fixture
def mock_extractor_class(mocker):
    return mocker.patch("postgres_graph_rag.core.LLMExtractor")


@pytest.fixture
def rag(mock_db_class, mock_extractor_class):
    # Synchronous initialization
    return PostgresGraphRAG(
        postgres_url="postgresql://user:pass@localhost:5432/db",
        openai_api_key="test_key",
        config=OPENAI_DEFAULT_CONFIG,
    )


@pytest.mark.asyncio
async def test_ingest(rag, mock_db_class, mock_extractor_class):
    mock_db = mock_db_class.return_value
    mock_extractor = mock_extractor_class.return_value
    mock_extractor.config = OPENAI_DEFAULT_CONFIG

    # Make these async mocks
    mock_extractor.extract_triplets = AsyncMock()
    mock_extractor.extract_triplets.return_value = [
        Triplet(subject="Apple", predicate="released", object="M4")
    ]

    mock_extractor.get_embedding = AsyncMock()
    mock_extractor.get_embedding.return_value = [0.1] * 1536

    mock_db.upsert_node = AsyncMock()
    mock_db.upsert_node.side_effect = ["uuid1", "uuid2"]

    mock_db.upsert_edge = AsyncMock()

    await rag.add_texts("Apple released the M4.", namespace="test-ns")

    mock_extractor.extract_triplets.assert_called_once()
    assert mock_db.upsert_node.call_count == 2
    mock_db.upsert_node.assert_any_call(
        "Apple", [0.1] * 1536, namespace="test-ns", metadata=None
    )
    mock_db.upsert_edge.assert_called_once_with(
        "uuid1", "uuid2", "released", namespace="test-ns", metadata=None
    )


@pytest.mark.asyncio
async def test_query(rag, mock_db_class, mock_extractor_class):
    mock_db = mock_db_class.return_value
    mock_extractor = mock_extractor_class.return_value

    # Make these async mocks
    mock_extractor.get_embedding = AsyncMock()
    mock_extractor.get_embedding.return_value = [0.1] * 1536

    mock_db.vector_search = AsyncMock()
    mock_db.vector_search.return_value = [{"id": "uuid1", "content": "Apple"}]

    mock_db.traverse_graph = AsyncMock()
    mock_db.traverse_graph.return_value = {
        "nodes": [
            {"id": "uuid1", "content": "Apple"},
            {"id": "uuid2", "content": "M4"},
        ],
        "edges": [
            {
                "source_node_id": "uuid1",
                "target_node_id": "uuid2",
                "relation": "released",
                "source_content": "Apple",
                "target_content": "M4",
                "weight": 1.0,
            }
        ],
    }

    context = await rag.query("What did Apple release?", namespace="test-ns")

    mock_db.vector_search.assert_called_once_with(
        [0.1] * 1536, namespace="test-ns", top_k=5
    )
    assert "Apple" in context
    assert "M4" in context
