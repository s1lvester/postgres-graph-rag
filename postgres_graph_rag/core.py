from typing import List, Dict, Any, Optional, Union, Callable
import pandas as pd
from .database import DatabaseManager
from .extractor import LLMExtractor
from .models import ProviderConfig, OPENAI_DEFAULT_CONFIG, GOOGLE_DEFAULT_CONFIG


def simple_chunker(
    text: str, size: int = 1000, overlap: int = 100
) -> List[str]:
    """Default simple character-based chunking."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start += size - overlap
    return chunks


class PostgresGraphRAG:
    def __init__(
        self,
        postgres_url: str,
        openai_api_key: Optional[str] = None,
        google_api_key: Optional[str] = None,
        config: Optional[ProviderConfig] = None,
        chunker: Optional[Callable[[str], List[str]]] = None,
    ):
        """
        Initializes the PostgresGraphRAG instance.
        This is a standard synchronous initialization.
        """
        self.db = DatabaseManager(postgres_url)

        if config is None:
            config = (
                GOOGLE_DEFAULT_CONFIG
                if google_api_key
                else OPENAI_DEFAULT_CONFIG
            )

        self.extractor = LLMExtractor(
            config=config,
            openai_api_key=openai_api_key,
            google_api_key=google_api_key,
        )
        self.chunker = chunker or simple_chunker

    async def __aenter__(self):
        """Supports async context manager usage."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Closes resources when exiting the context."""
        await self.close()

    async def close(self):
        """Manually closes the database connection pool."""
        await self.db.close()

    async def setup(self):
        """Initializes the migration-safe database schema."""
        dimension = self.extractor.config["dimension"]
        await self.db.setup_database(embedding_dimension=dimension)

    async def add_texts(
        self,
        texts: Union[str, List[str]],
        namespace: str = "default",
        metadata: Dict[str, Any] = None,
    ):
        """Ingests one or more texts into a specific namespace."""
        if isinstance(texts, str):
            texts = [texts]

        for text in texts:
            chunks = self.chunker(text)
            for chunk in chunks:
                triplets = await self.extractor.extract_triplets(chunk)
                for triplet in triplets:
                    sub_emb = await self.extractor.get_embedding(
                        triplet.subject
                    )
                    sub_id = await self.db.upsert_node(
                        triplet.subject,
                        sub_emb,
                        namespace=namespace,
                        metadata=metadata,
                    )

                    obj_emb = await self.extractor.get_embedding(triplet.object)
                    obj_id = await self.db.upsert_node(
                        triplet.object,
                        obj_emb,
                        namespace=namespace,
                        metadata=metadata,
                    )

                    await self.db.upsert_edge(
                        sub_id,
                        obj_id,
                        triplet.predicate,
                        namespace=namespace,
                        metadata=metadata,
                    )

    async def query(
        self,
        question: str,
        namespace: str = "default",
        hops: int = 2,
        top_k: int = 5,
    ) -> str:
        """Searches the graph and returns enriched context."""
        query_emb = await self.extractor.get_embedding(question)
        seed_nodes = await self.db.vector_search(
            query_emb, namespace=namespace, top_k=top_k
        )
        seed_ids = [n["id"] for n in seed_nodes]
        graph_data = await self.db.traverse_graph(
            seed_ids, namespace=namespace, max_hops=hops
        )
        return self._format_context(graph_data)

    def _format_context(
        self, graph_data: Dict[str, List[Dict[str, Any]]]
    ) -> str:
        if not graph_data["nodes"] and not graph_data["edges"]:
            return "No relevant context found."

        nodes_df = pd.DataFrame(graph_data["nodes"])
        edges_df = pd.DataFrame(graph_data["edges"])

        context_parts = ["Relevant Entities and Relationships:"]

        if not nodes_df.empty:
            context_parts.append("\nEntities:")
            for _, row in nodes_df.iterrows():
                context_parts.append(f"- {row['content']}")

        if not edges_df.empty:
            context_parts.append("\nRelationships:")
            for _, row in edges_df.iterrows():
                context_parts.append(
                    f"- {row['source_content']} --[{row['relation']}]--> {row['target_content']}"
                )

        return "\n".join(context_parts)
