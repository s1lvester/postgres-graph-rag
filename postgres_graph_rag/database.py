import json
from typing import List, Dict, Any, Optional, Protocol, Union
import psycopg
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool


class DatabaseManager:
    def __init__(self, connection_url: str):
        self.connection_url = connection_url
        self.pool: Optional[AsyncConnectionPool] = None

    async def _init_pool(self):
        """Lazily initializes the connection pool if it doesn't exist."""
        if self.pool is None:
            self.pool = AsyncConnectionPool(
                self.connection_url,
                open=False,  # Wait for explicit open
                kwargs={"row_factory": dict_row},
            )
            await self.pool.open()

    async def close(self):
        """Closes the connection pool."""
        if self.pool:
            await self.pool.close()
            self.pool = None

    async def setup_database(self, embedding_dimension: int = 1536):
        """Initializes the migration-safe 'Forever Schema'."""
        await self._init_pool()
        async with self.pool.connection() as conn:
            async with conn.cursor() as cur:
                # Enable pgvector extension
                await cur.execute("CREATE EXTENSION IF NOT EXISTS vector")

                # Create graph_nodes table (The Entities)
                await cur.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS graph_nodes (
                        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        namespace VARCHAR(255) NOT NULL,
                        content TEXT NOT NULL,
                        embedding VECTOR({embedding_dimension}),
                        metadata JSONB DEFAULT '{{}}',
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                    )
                """
                )

                # Create graph_edges table (The Relationships)
                await cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS graph_edges (
                        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        namespace VARCHAR(255) NOT NULL,
                        source_node_id UUID REFERENCES graph_nodes(id) ON DELETE CASCADE,
                        target_node_id UUID REFERENCES graph_nodes(id) ON DELETE CASCADE,
                        relation TEXT NOT NULL,
                        weight FLOAT DEFAULT 1.0,
                        metadata JSONB DEFAULT '{}',
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(namespace, source_node_id, target_node_id, relation)
                    )
                """
                )

                # Indices for fast lookups and namespacing
                await cur.execute(
                    "CREATE UNIQUE INDEX IF NOT EXISTS idx_graph_nodes_namespace_content ON graph_nodes (namespace, content)"
                )
                await cur.execute(
                    "CREATE INDEX IF NOT EXISTS idx_graph_nodes_embedding ON graph_nodes USING hnsw (embedding vector_cosine_ops)"
                )
                await cur.execute(
                    "CREATE INDEX IF NOT EXISTS idx_graph_edges_source ON graph_edges (source_node_id)"
                )
                await cur.execute(
                    "CREATE INDEX IF NOT EXISTS idx_graph_edges_target ON graph_edges (target_node_id)"
                )
                await cur.execute(
                    "CREATE INDEX IF NOT EXISTS idx_graph_edges_namespace ON graph_edges (namespace)"
                )

                await conn.commit()

    async def upsert_node(
        self,
        content: str,
        embedding: List[float],
        namespace: str = "default",
        metadata: Dict[str, Any] = None,
    ) -> str:
        """Inserts or updates a node within a namespace and returns its ID."""
        metadata = metadata or {}
        await self._init_pool()
        async with self.pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    INSERT INTO graph_nodes (namespace, content, embedding, metadata)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (namespace, content) DO UPDATE SET
                        embedding = EXCLUDED.embedding,
                        metadata = graph_nodes.metadata || EXCLUDED.metadata
                    RETURNING id
                """,
                    (namespace, content, embedding, json.dumps(metadata)),
                )
                row = await cur.fetchone()
                await conn.commit()
                return str(row["id"])

    async def upsert_edge(
        self,
        source_id: str,
        target_id: str,
        relation: str,
        namespace: str = "default",
        weight: float = 1.0,
        metadata: Dict[str, Any] = None,
    ):
        """Inserts or updates an edge within a namespace."""
        metadata = metadata or {}
        await self._init_pool()
        async with self.pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    INSERT INTO graph_edges (namespace, source_node_id, target_node_id, relation, weight, metadata)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (namespace, source_node_id, target_node_id, relation) DO UPDATE SET
                        weight = EXCLUDED.weight,
                        metadata = graph_edges.metadata || EXCLUDED.metadata
                """,
                    (
                        namespace,
                        source_id,
                        target_id,
                        relation,
                        weight,
                        json.dumps(metadata),
                    ),
                )
                await conn.commit()

    async def vector_search(
        self,
        query_embedding: List[float],
        namespace: str = "default",
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Finds nodes within a namespace most similar to the query embedding."""
        await self._init_pool()
        async with self.pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    SELECT id, content, metadata, (embedding <=> %s::vector) as distance
                    FROM graph_nodes
                    WHERE namespace = %s
                    ORDER BY distance ASC
                    LIMIT %s
                """,
                    (query_embedding, namespace, top_k),
                )
                return await cur.fetchall()

    async def traverse_graph(
        self,
        seed_node_ids: List[str],
        namespace: str = "default",
        max_hops: int = 2,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Performs a namespaced recursive traversal to find neighbors within N hops."""
        if not seed_node_ids:
            return {"nodes": [], "edges": []}

        await self._init_pool()
        async with self.pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    WITH RECURSIVE graph_expansion AS (
                        -- Base case: seed nodes
                        SELECT id, content, metadata, 0 as depth, ARRAY[id] as visited
                        FROM graph_nodes
                        WHERE id = ANY(%s) AND namespace = %s

                        UNION ALL

                        -- Recursive step: find outgoing edges
                        SELECT n.id, n.content, n.metadata, ge.depth + 1, ge.visited || n.id
                        FROM graph_nodes n
                        JOIN graph_edges e ON n.id = e.target_node_id
                        JOIN graph_expansion ge ON e.source_node_id = ge.id
                        WHERE ge.depth < %s 
                          AND e.namespace = %s
                          AND NOT (n.id = ANY(ge.visited))
                    )
                    SELECT DISTINCT id, content, metadata FROM graph_expansion
                """,
                    (seed_node_ids, namespace, max_hops, namespace),
                )
                nodes = await cur.fetchall()
                node_ids = [n["id"] for n in nodes]

                if node_ids:
                    await cur.execute(
                        """
                        SELECT e.source_node_id, e.target_node_id, e.relation, e.metadata, e.weight,
                               s.content as source_content, t.content as target_content
                        FROM graph_edges e
                        JOIN graph_nodes s ON e.source_node_id = s.id
                        JOIN graph_nodes t ON e.target_node_id = t.id
                        WHERE e.source_node_id = ANY(%s) 
                          AND e.target_node_id = ANY(%s)
                          AND e.namespace = %s
                    """,
                        (node_ids, node_ids, namespace),
                    )
                    edges = await cur.fetchall()
                else:
                    edges = []

                return {"nodes": nodes, "edges": edges}
