import json
import uuid
from typing import List, Dict, Any, Optional
import psycopg
from psycopg.rows import dict_row


class DatabaseManager:
    def __init__(self, connection_url: str):
        self.connection_url = connection_url

    def _get_connection(self):
        return psycopg.connect(self.connection_url, row_factory=dict_row)

    def setup_database(self, embedding_dimension: int = 1536):
        """Initializes the database schema."""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                # Enable pgvector extension
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector")

                # Create nodes table with dynamic vector size
                cur.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS nodes (
                        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        content TEXT UNIQUE NOT NULL,
                        metadata JSONB DEFAULT '{{}}',
                        embedding VECTOR({embedding_dimension})
                    )
                """
                )

                # Create edges table
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS edges (
                        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        source_id UUID REFERENCES nodes(id) ON DELETE CASCADE,
                        target_id UUID REFERENCES nodes(id) ON DELETE CASCADE,
                        relation TEXT NOT NULL,
                        metadata JSONB DEFAULT '{}',
                        UNIQUE(source_id, target_id, relation)
                    )
                """
                )

                # Create indices
                cur.execute(
                    "CREATE INDEX IF NOT EXISTS idx_nodes_embedding ON nodes USING hnsw (embedding vector_cosine_ops)"
                )
                cur.execute(
                    "CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source_id)"
                )
                cur.execute(
                    "CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target_id)"
                )

                conn.commit()

    def upsert_node(
        self,
        content: str,
        embedding: List[float],
        metadata: Dict[str, Any] = None,
    ) -> str:
        """Inserts or updates a node and returns its ID."""
        metadata = metadata or {}
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO nodes (content, embedding, metadata)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (content) DO UPDATE SET
                        embedding = EXCLUDED.embedding,
                        metadata = nodes.metadata || EXCLUDED.metadata
                    RETURNING id
                """,
                    (content, embedding, json.dumps(metadata)),
                )
                row = cur.fetchone()
                conn.commit()
                return str(row["id"])

    def upsert_edge(
        self,
        source_id: str,
        target_id: str,
        relation: str,
        metadata: Dict[str, Any] = None,
    ):
        """Inserts or updates an edge."""
        metadata = metadata or {}
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO edges (source_id, target_id, relation, metadata)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (source_id, target_id, relation) DO UPDATE SET
                        metadata = edges.metadata || EXCLUDED.metadata
                """,
                    (source_id, target_id, relation, json.dumps(metadata)),
                )
                conn.commit()

    def vector_search(
        self, query_embedding: List[float], top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Finds nodes most similar to the query embedding."""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, content, metadata, (embedding <=> %s::vector) as distance
                    FROM nodes
                    ORDER BY distance ASC
                    LIMIT %s
                """,
                    (query_embedding, top_k),
                )
                return cur.fetchall()

    def traverse_graph(
        self, seed_node_ids: List[str], max_hops: int = 2
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Performs a recursive traversal to find neighbors within N hops."""
        if not seed_node_ids:
            return {"nodes": [], "edges": []}

        with self._get_connection() as conn:
            with conn.cursor() as cur:
                # Recursive CTE for graph expansion
                cur.execute(
                    """
                    WITH RECURSIVE graph_expansion AS (
                        -- Base case: seed nodes
                        SELECT id, content, metadata, 0 as depth, ARRAY[id] as visited
                        FROM nodes
                        WHERE id = ANY(%s)

                        UNION ALL

                        -- Recursive step: find outgoing edges
                        SELECT n.id, n.content, n.metadata, ge.depth + 1, ge.visited || n.id
                        FROM nodes n
                        JOIN edges e ON n.id = e.target_id
                        JOIN graph_expansion ge ON e.source_id = ge.id
                        WHERE ge.depth < %s AND NOT (n.id = ANY(ge.visited))
                    )
                    SELECT DISTINCT id, content, metadata FROM graph_expansion
                """,
                    (seed_node_ids, max_hops),
                )
                nodes = cur.fetchall()
                node_ids = [n["id"] for n in nodes]

                # Get all edges between the discovered nodes
                if node_ids:
                    cur.execute(
                        """
                        SELECT e.source_id, e.target_id, e.relation, e.metadata,
                               s.content as source_content, t.content as target_content
                        FROM edges e
                        JOIN nodes s ON e.source_id = s.id
                        JOIN nodes t ON e.target_id = t.id
                        WHERE e.source_id = ANY(%s) AND e.target_id = ANY(%s)
                    """,
                        (node_ids, node_ids),
                    )
                    edges = cur.fetchall()
                else:
                    edges = []

                return {"nodes": nodes, "edges": edges}
