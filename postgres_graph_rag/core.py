import asyncio
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
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Ingests one or more texts into a specific namespace.
        Optimized to use parallel extraction and batch embeddings.
        """
        if isinstance(texts, str):
            texts = [texts]

        # 1. Extraction (Parallel across all texts and chunks)
        all_chunks = []
        for text in texts:
            all_chunks.extend(self.chunker(text))

        extraction_tasks = [
            self.extractor.extract_triplets(c) for c in all_chunks
        ]
        all_triplets_lists = await asyncio.gather(*extraction_tasks)

        # Flatten and deduplicate entities
        all_triplets = []
        unique_entities = set()
        for triplets in all_triplets_lists:
            if not triplets:
                continue
            for t in triplets:
                all_triplets.append(t)
                unique_entities.add(t.subject)
                unique_entities.add(t.object)

        if not unique_entities:
            return

        # 2. Batch Embedding Retrieval
        entity_list = list(unique_entities)
        embeddings = await self.extractor.get_embedding(entity_list)
        entity_to_emb = dict(zip(entity_list, embeddings))

        # 3. Batch DB Write
        await self.db._init_pool()
        async with self.db.pool.connection() as conn:
            # Prepare nodes for batch upsert
            nodes_data = [
                {
                    "content": entity,
                    "embedding": entity_to_emb[entity],
                    "metadata": metadata,
                }
                for entity in entity_list
            ]

            # Upsert nodes and get their IDs
            node_ids_list = await self.db.upsert_nodes_batch(
                nodes_data, namespace=namespace, connection=conn
            )
            content_to_id = dict(zip(entity_list, node_ids_list))

            # Prepare edges for batch upsert
            edges_data = [
                {
                    "source_id": content_to_id[t.subject],
                    "target_id": content_to_id[t.object],
                    "relation": t.predicate,
                    "metadata": metadata,
                }
                for t in all_triplets
            ]

            # Upsert edges
            await self.db.upsert_edges_batch(
                edges_data, namespace=namespace, connection=conn
            )

            # Commit the whole batch in one transaction
            await conn.commit()

    async def _process_single_text(
        self, text: str, namespace: str, metadata: Optional[Dict[str, Any]]
    ):
        """Deprecated: Internal helper to process a single text blob (extraction -> embedding -> db)."""
        await self.add_texts([text], namespace=namespace, metadata=metadata)

    async def query(
        self,
        question: str,
        namespace: str = "default",
        hops: int = 2,
        top_k: int = 5,
    ) -> str:
        """Searches the graph and returns enriched context."""
        # get_embedding returns List[float] when given a string
        query_emb = await self.extractor.get_embedding(question)
        if (
            isinstance(query_emb, list)
            and query_emb
            and isinstance(query_emb[0], list)
        ):
            # This should not happen since question is a string, but for type safety:
            query_emb = query_emb[0]

        await self.db._init_pool()
        async with self.db.pool.connection() as conn:
            seed_nodes = await self.db.vector_search(
                query_emb, namespace=namespace, top_k=top_k, connection=conn
            )
            seed_ids = [n["id"] for n in seed_nodes]
            graph_data = await self.db.traverse_graph(
                seed_ids, namespace=namespace, max_hops=hops, connection=conn
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
