from typing import List, Dict, Any, Optional
import pandas as pd
from .database import DatabaseManager
from .extractor import LLMExtractor
from .models import ProviderConfig, OPENAI_DEFAULT_CONFIG, GOOGLE_DEFAULT_CONFIG


class PostgresGraphRAG:
    def __init__(
        self,
        postgres_url: str,
        openai_api_key: Optional[str] = None,
        google_api_key: Optional[str] = None,
        config: Optional[ProviderConfig] = None,
    ):
        self.db = DatabaseManager(postgres_url)

        # Determine default config if none provided
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

    def setup(self):
        """Initializes the migration-safe database schema."""
        dimension = self.extractor.config["dimension"]
        self.db.setup_database(embedding_dimension=dimension)

    def ingest(
        self,
        text: str,
        namespace: str = "default",
        chunk_size: int = 1000,
        chunk_overlap: int = 100,
        metadata: Dict[str, Any] = None,
    ):
        """Ingests text into a specific namespace."""
        chunks = self._chunk_text(text, chunk_size, chunk_overlap)
        for chunk in chunks:
            triplets = self.extractor.extract_triplets(chunk)
            for triplet in triplets:
                # 1. Upsert subject node
                sub_emb = self.extractor.get_embedding(triplet.subject)
                sub_id = self.db.upsert_node(
                    triplet.subject,
                    sub_emb,
                    namespace=namespace,
                    metadata=metadata,
                )

                # 2. Upsert object node
                obj_emb = self.extractor.get_embedding(triplet.object)
                obj_id = self.db.upsert_node(
                    triplet.object,
                    obj_emb,
                    namespace=namespace,
                    metadata=metadata,
                )

                # 3. Upsert edge
                self.db.upsert_edge(
                    sub_id,
                    obj_id,
                    triplet.predicate,
                    namespace=namespace,
                    metadata=metadata,
                )

    def query(
        self,
        question: str,
        namespace: str = "default",
        hops: int = 2,
        top_k: int = 5,
    ) -> str:
        """Searches the graph within a namespace and returns enriched context."""
        query_emb = self.extractor.get_embedding(question)
        seed_nodes = self.db.vector_search(
            query_emb, namespace=namespace, top_k=top_k
        )
        seed_ids = [n["id"] for n in seed_nodes]
        graph_data = self.db.traverse_graph(
            seed_ids, namespace=namespace, max_hops=hops
        )
        return self._format_context(graph_data)

    def _chunk_text(self, text: str, size: int, overlap: int) -> List[str]:
        """Simple recursive character-based chunking."""
        chunks = []
        start = 0
        while start < len(text):
            end = start + size
            chunks.append(text[start:end])
            start += size - overlap
        return chunks

    def _format_context(
        self, graph_data: Dict[str, List[Dict[str, Any]]]
    ) -> str:
        """Converts graph data into a readable text block for LLM context."""
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
