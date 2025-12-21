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
        """Ingests one or more texts into a specific namespace."""
        if isinstance(texts, str):
            texts = [texts]

        for text in texts:
            chunks = self.chunker(text)
            for chunk in chunks:
                # 1. Extraction (per chunk)
                triplets = await self.extractor.extract_triplets(chunk)
                if not triplets:
                    continue

                # 2. Collect unique entities in this chunk
                entities = set()
                for t in triplets:
                    entities.add(t.subject)
                    entities.add(t.object)

                # 3. Parallel Embedding Retrieval
                entity_list = list(entities)
                embedding_tasks = [
                    self.extractor.get_embedding(e) for e in entity_list
                ]
                embeddings = await asyncio.gather(*embedding_tasks)
                entity_to_emb = dict(zip(entity_list, embeddings))

                # 4. Batch DB Write
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
                        for t in triplets
                    ]

                    # Upsert edges
                    await self.db.upsert_edges_batch(
                        edges_data, namespace=namespace, connection=conn
                    )

                    # Commit the whole chunk in one transaction
                    await conn.commit()

    async def query(
        self,
        question: str,
        namespace: str = "default",
        hops: int = 2,
        top_k: int = 5,
    ) -> str:
        """Searches the graph and returns enriched context."""
        query_emb = await self.extractor.get_embedding(question)

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
