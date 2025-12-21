# postgres-graph-rag (The Lean MVP)

A high-performance, Postgres-native GraphRAG library. No complex orchestration frameworks—just pure Python and SQL.

## Core Philosophy
- **Infrastructure:** Postgres is the only database (via `pgvector`).
- **Intelligence:** Hosted SLMs (GPT-5.2 or Gemini 2.5) for extraction.
- **Simplicity:** Pure Python + SQL.

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
    # model defaults to gpt-5-nano-2025-08-07
)

# Or for Gemini
rag = PostgresGraphRAG(
    postgres_url="your_dsn",
    google_api_key="your_key"
    # model defaults to gemini-2.5-flash-lite
)

# 1. Setup DB (Creates tables and extensions)
rag.setup()

# 2. Ingest
rag.ingest("Apple's hardware team, led by Johny Srouji, designed the M4 chip.")

# 3. Query (Returns graph-enriched context)
context = rag.query("Who leads the team that made the M4?", hops=2)
print(context)
```

## Features
- **Atomic Upserts:** Uses `ON CONFLICT` logic to ensure entity names are unique.
- **Recursive Traversal:** Uses a Postgres Recursive CTE for efficient N-hop neighbor discovery with cycle detection.
- **Structured Extraction:** Uses OpenAI Structured Outputs and Google GenAI Controlled Generation for reliable triplet extraction.

## Database Schema
The library manages two tables:
- `nodes`: Stores entities, their metadata, and embeddings.
- `edges`: Stores relationships between entities.

## Dependencies
- `psycopg3`
- `pgvector`
- `openai`
- `google-genai`
- `pandas`
- `pydantic`
