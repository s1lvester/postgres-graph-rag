# postgres-graph-rag 🐘🕸️

### High-Precision GraphRAG. Native to PostgreSQL.

Most RAG systems are Flatlanders. They use vector similarity to find related text, but they are fundamentally blind to **relationships**. If you ask your RAG "How is Person A connected to Project B through their shared dependencies?", standard vector search fails because the answer isn't in a single chunk—it’s in the **links** between them.

**Postgres Graph RAG** bridges this reasoning gap by turning your existing PostgreSQL database into a structured knowledge engine. 

### Why this exists:
1.  **Infrastructure Nightmare:** Building "Smart RAG" usually means adding a Graph DB (Neo4j) to your stack. Now you have a distributed systems nightmare: keeping your Relational DB, Vector DB, and Graph DB in sync.
2.  **Flatland Problem:** Vector similarity is just probabilistic matching. It doesn't understand hierarchy, causality, or directed relationships (e.g., "A leads B" vs "B leads A").
3.  **Batch Bottleneck:** Existing GraphRAG research (like Microsoft's) is batch-heavy and token-expensive. It can't handle real-time, incremental updates.

### Postgres-Native Solution:
This library is built for **Postgres Maximalists**. It leverages the engine you already trust to do the heavy lifting:
*   **Recursive Retrieval:** Instead of expensive LLM-agent loops, we use **SQL Recursive CTEs** to perform multi-hop reasoning. It’s deterministic, 10x faster, and handles "neighbor-of-neighbor" walks natively.
*   **Atomic Consistency:** Vectors, nodes, and relationships live in one ACID-compliant engine. One transaction. Zero sync lag.
*   **Forever Schema:** Using `JSONB` metadata and a namespaced design, the schema is migration-proof. You can evolve your graph's logic without ever running `ALTER TABLE`.

---

## Core Philosophy
- **Infrastructure:** Postgres is the only database (via `pgvector`).
- **Intelligence:** Hosted SLMs (**GPT-5.2** or **Gemini 2.5**) for extraction. Freely configurable.
- **Simplicity:** Native Async Python + SQL.
- **Scalability:** High-performance connection pooling and namespace-aware design (Multi-tenancy).

---

## Installation

```bash
pip install postgres-graph-rag
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

## 🗺️ Roadmap & Future Vision

This project follows the **"Postgres Maximalism"** philosophy: Stop building new infrastructure and start using the full power of the database you already own.

### ✅ Phase 1: Foundation (Current Release)
- [x] **Postgres-Native Schema:** Migration-proof design using JSONB and namespacing.
- [x] **Recursive Reasoning:** Multi-hop graph traversal implemented via SQL Recursive CTEs.
- [x] **Incremental Ingestion:** Atomic upserts for nodes and edges (no expensive batch rebuilds).
- [x] **Hosted SLM Extraction:** Native support for OpenAI and Google Gemini for high-speed, low-cost tagging.
- [x] **Async Architecture:** Production-ready with high-performance connection pooling.

### 🏗️ Phase 2: High-Precision Retrieval (Next Up)

- [ ] **Hybrid Search (BM25 + Vector):** Integrate Postgres Full-Text Search with `pgvector`. Keyword precision meets semantic depth.
- [ ] **Advanced Entity Resolution (ER):** Automatic merging of similar entities (e.g., "Elon" and "Elon Musk") using `pg_trgm` fuzzy matching and vector distance during ingestion.
- [ ] **Relationship Scoring:** Dynamic edge weighting based on mention frequency and extraction confidence scores.
- [ ] **Metadata Pruning:** Filter graph paths based on metadata (e.g., "Only traverse relationships from documents updated in the last 90 days").

### 📊 Phase 3: Global Intelligence & Scaling

- [ ] **SQL-Native Community Detection:** Implement clustering algorithms directly in SQL to identify thematic communities (a lightweight alternative to Leiden).
- [ ] **Global Summarization:** Automated summary generation for clusters to answer "What are the key trends across these 10,000 documents?".
- [ ] **Graph Observability:** Built-in tracing to visualize the "Reasoning Path"—showing exactly why the agent connected Node A to Node C.

### 🛡️ Phase 4: Enterprise & Agentic Features

- [ ] **Agent Identity & Auth (Signal #1):** Integration with Postgres Row-Level Security (RLS) to ensure agents only traverse paths the specific user is authorized to see.
- [ ] **MCP Server Support:** Native Model Context Protocol implementation so `pg-graph-rag` can be used as a direct tool in Claude, Cursor, and other IDEs.
- [ ] **Telemetry & Usage Analytics:** Per-namespace token tracking and latency monitoring for cost-conscious scaling.

---

### 💡 User-Driven Priorities
We prioritize features that reduce **Operational Overhead**. If you need a feature that further consolidates the "Standard Stack" (Vector + Graph + Relational) into Postgres, open an issue!

**Launch Status:** 🚀 MVP is live. Focus is now on **Hybrid Search** and **Automatic Entity Resolution**.

---

## Development

If you want to contribute or run the tests locally:

```bash
# Clone the repo and sync dependencies
uv sync --extra test

# Run tests
uv run pytest
```
