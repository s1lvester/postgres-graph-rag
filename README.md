# Postgres Graph RAG (The Lean MVP)

A high-performance, Postgres-native GraphRAG library using a migration-safe "Forever Schema". No complex orchestration frameworks—just pure Python and SQL.

## Core Philosophy
- **Infrastructure:** Postgres is the only database (via `pgvector`).
- **Intelligence:** Hosted SLMs (GPT-5.2 or Gemini 2.5) for extraction.
- **Simplicity:** Pure Python + SQL.
- **Scalability:** Namespace-aware design (Multi-tenancy) and JSONB metadata.

## Installation

Using `uv` (recommended):
```bash
uv sync --extra test
```

Or using `pip`:
```bash
pip install .
```

## Running Tests

```bash
uv run pytest
```

## Developer API

```python
from postgres_graph_rag import PostgresGraphRAG

# Initialize
rag = PostgresGraphRAG(
    postgres_url="postgresql://user:password@localhost:5432/dbname",
    openai_api_key="your_openai_key"
)

# 1. Setup DB (Creates the "Forever Schema" tables)
rag.setup()

# 2. Ingest (with optional namespace/multi-tenancy)
rag.ingest(
    "Apple's hardware team, led by Johny Srouji, designed the M4 chip.",
    namespace="project_alpha"
)

# 3. Query (Returns graph-enriched context from specific namespace)
context = rag.query(
    "Who leads the team that made the M4?", 
    namespace="project_alpha",
    hops=2
)
print(context)
```

## Features
- **Forever Schema:** Uses `graph_nodes` and `graph_edges` with JSONB metadata. No `ALTER TABLE` needed for future features.
- **Multi-Tenancy:** First-class support for `namespace` to separate data (e.g., per user or per project).
- **Atomic Upserts:** Uses `ON CONFLICT` with JSONB merging (`||`) for reliable data ingestion.
- **Recursive Traversal:** Optimized Postgres Recursive CTE for efficient N-hop neighbor discovery with cycle detection.

## Database Schema
The library manages two tables:
- `graph_nodes`: Stores entities, namespaces, embeddings, and flexible JSONB metadata.
- `graph_edges`: Stores relationships, namespaces, weights, and JSONB metadata.

## Dependencies
- `psycopg3`
- `pgvector`
- `openai`
- `google-genai`
- `pandas`
- `pydantic`
