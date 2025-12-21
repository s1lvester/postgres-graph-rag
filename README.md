# Postgres Graph RAG

A high-performance, Postgres-native GraphRAG library using a migration-safe "Forever Schema". No complex orchestration frameworks—just pure Python and SQL.

## Core Philosophy
- **Infrastructure:** Postgres is the only database (via `pgvector`).
- **Intelligence:** Hosted SLMs (GPT-5.2 or Gemini 2.5) for extraction.
- **Simplicity:** Native Async Python + SQL.
- **Scalability:** High-performance connection pooling and namespace-aware design (Multi-tenancy).

---

## Installation

Using `uv` (recommended):
```bash
uv sync --extra test
```

Or using `pip`:
```bash
pip install "postgres-graph-rag[pool]"
```

---

## Getting Started (Interactive & Frictionless)

The library is designed to be interactive-friendly. You can instantiate it normally and use `await` at the top level in Notebooks or REPLs. The database connection pool is initialized lazily upon the first request.

```python
from postgres_graph_rag import PostgresGraphRAG

# 1. Simple Instantiation
rag = PostgresGraphRAG(
    postgres_url="postgresql://user:password@localhost:5432/dbname",
    openai_api_key="sk-..." # Or use google_api_key
)

async def quick_start():
    # 2. Setup (Creates tables and pgvector extension if missing)
    await rag.setup()

    # 3. Add Knowledge (Atomic upserts with automatic entity resolution)
    await rag.add_texts(
        "Johny Srouji leads the hardware team at Apple.", 
        namespace="apple_research"
    )

    # 4. Hybrid Query (Vector Search + Recursive Graph Traversal)
    context = await rag.query(
        "Who is leading the hardware efforts?", 
        namespace="apple_research",
        hops=2
    )
    print(context)
    
    # 5. Cleanup (Closes the connection pool)
    await rag.close()
```

---

## Advanced Usage & Modes

### 1. The Production Way: Async Context Manager
For applications (like FastAPI or background workers), use the `async with` pattern to ensure the connection pool is always closed correctly, even if errors occur.

```python
async with PostgresGraphRAG(postgres_url=DSN, openai_api_key=KEY) as rag:
    await rag.add_texts("The M4 chip uses ARM architecture.")
    # No need to call rag.close(), it happens automatically!
```

### 2. Custom Chunking (Inversion of Control)
Don't like the default character splitter? Inject your own. You can pass any callable that takes a string and returns a list of strings.

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Create your favorite chunker
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

# Inject it into the library
rag = PostgresGraphRAG(
    postgres_url=DSN,
    openai_api_key=KEY,
    chunker=splitter.split_text # Just pass the method
)
```

### 3. Custom Provider Configuration
You can control exactly which models are used for extraction and embeddings.

```python
from postgres_graph_rag.models import ProviderConfig

custom_config: ProviderConfig = {
    "extraction_model": "gpt-5-nano-2025-08-07",
    "embedding_model": "text-embedding-3-large",
    "dimension": 3072 # Must match the model's output
}

rag = PostgresGraphRAG(..., config=custom_config)
```

### 4. Multi-Tenancy (Namespacing)
Isolate data for different users or projects within the same database tables.

```python
# User A's private graph
await rag.add_texts("My secret key is 123.", namespace="user_a")

# User B's private graph
await rag.add_texts("My secret key is 999.", namespace="user_b")

# Queries are strictly isolated
res = await rag.query("What is my key?", namespace="user_a") # Returns 123
```

---

## Features under the Hood

- **Forever Schema:** Uses `graph_nodes` and `graph_edges` with JSONB metadata. No `ALTER TABLE` Akrobatik needed for future metadata fields.
- **Connection Pooling:** Uses `psycopg_pool.AsyncConnectionPool` for high-concurrency performance.
- **Cycle Detection:** The recursive CTE uses a `visited` array to prevent infinite loops in complex graphs.
- **Atomic JSONB Merges:** Metadata is merged using the Postgres `||` operator during ingestion, preserving historical data.

---

## Dependencies
- `psycopg[pool]`
- `pgvector`
- `openai`
- `google-genai`
- `pandas`
- `pydantic`
