# Knowledge Graph Pipeline

A two-stage pipeline that transforms plain text files into a queryable Neo4j knowledge graph, powered by OpenAI for both extraction and natural language querying.

---

## Overview

```
.txt files  ──►  txt_triplet_ingest.py  ──►  Neo4j Graph  ──►  neo4j_query_interface.py  ──►  Natural Language Answer
```

The system has two components that work in sequence:

1. **Ingestion** (`txt_triplet_ingest.py`) — reads raw text, extracts structured knowledge triplets using an LLM, deduplicates them via embedding similarity, and writes the resulting graph into Neo4j.
2. **Query Interface** (`neo4j_query_interface.py`) — accepts a natural language question, translates it to a Cypher query using an LLM, executes it against Neo4j, and returns a human-readable answer.

---

## Prerequisites

- Python 3.9+
- A running [Neo4j](https://neo4j.com/) instance (local or remote)
- An [OpenAI API key](https://platform.openai.com/api-keys)

### Install dependencies

```bash
pip install openai neo4j python-dotenv
```

---

## Configuration

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=sk-...

NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password
NEO4J_DATABASE=neo4j          # optional, defaults to "neo4j"
```

---

## Usage

### Step 1 — Ingest text files into Neo4j

```bash
python txt_triplet_ingest.py path/to/file1.txt path/to/file2.txt
```

If no files are provided, the script defaults to `data/lecture1.txt` and `data/lecture2.txt`.

**Optional flags:**

| Flag | Default | Description |
|---|---|---|
| `--chunk-size` | `3500` | Max characters per extraction chunk |
| `--max-workers` | `4` | Parallel workers for extraction |
| `--node-threshold` | `0.88` | Cosine similarity threshold for merging duplicate nodes |
| `--relation-threshold` | `0.90` | Cosine similarity threshold for merging duplicate relation labels |
| `--min-confidence` | `0.65` | Minimum LLM confidence score to keep a triplet |
| `--model` | `gpt-4o-mini` | OpenAI chat model used for extraction |
| `--embedding-model` | `text-embedding-3-small` | OpenAI embedding model used for similarity merging |
| `--save-json` | *(none)* | Save the merged graph payload to a JSON file |
| `--search-query` | *(none)* | Run a semantic search after ingestion and print results |
| `--search-top-k` | `5` | Number of semantic search results to return |
| `--print-limit` | `10` | Number of nodes/edges to print in the summary |

**Example with options:**

```bash
python txt_triplet_ingest.py notes.txt \
  --chunk-size 2000 \
  --min-confidence 0.7 \
  --save-json output/graph.json \
  --search-query "machine learning"
```

---

### Step 2 — Query the graph in natural language

Edit the `__main__` block in `neo4j_query_interface.py` and set your question:

```python
query = "Tell me something about Google"
result = interface.query(query)
print(result["response"])
```

Then run:

```bash
python neo4j_query_interface.py
```

The interface will print the generated Cypher query, the raw results, and a natural language summary.

---

## How It Works

### Ingestion pipeline (`txt_triplet_ingest.py`)

1. **Read** — loads `.txt` files from disk.
2. **Chunk** — splits each file into overlapping text chunks (~3500 chars by default) to fit within LLM context limits.
3. **Extract** — sends each chunk to GPT-4o-mini and asks it to return a list of `(subject, relation, object)` triplets with a confidence score.
4. **Filter** — drops any triplet whose confidence is below `--min-confidence`.
5. **Embed** — generates embeddings for all entity names and relation labels using `text-embedding-3-small`.
6. **Merge** — clusters near-duplicate nodes and relations using cosine similarity; keeps a weighted average embedding and a canonical name for each cluster.
7. **Write** — upserts nodes (`Entity`, `SourceFile`) and edges into Neo4j, including embeddings for later semantic search.

### Query interface (`neo4j_query_interface.py`)

1. **Schema fetch** — introspects the live Neo4j database to retrieve node labels, relationship types, and property keys.
2. **Cypher translation** — sends the schema and the user's question to GPT-4o-mini, which returns a valid Cypher query.
3. **Execute** — runs the Cypher query against Neo4j and collects the raw records.
4. **Summarize** — passes the raw results back to GPT-4o-mini to produce a concise natural language answer.

---

## Project Structure

```
.
├── txt_triplet_ingest.py      # Ingestion pipeline
├── neo4j_query_interface.py   # Natural language query interface
├── data/
│   ├── lecture1.txt           # Example input files
│   └── lecture2.txt
└── .env                       # API keys and Neo4j credentials (not committed)
```

---

## Notes

- Both scripts read credentials from environment variables or `.env`. Never hard-code secrets in source files.
- The ingestion script is idempotent — re-running it on the same files will merge new information into existing nodes rather than creating duplicates, thanks to the similarity-based merging step.
- Neo4j vector index (`entity_embedding_index`, `source_file_embedding_index`) is used for semantic search if available; otherwise the pipeline falls back to in-memory cosine similarity.
