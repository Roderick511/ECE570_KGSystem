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

### `txt_triplet_ingest.py`

**Dataclasses:**
- `RawTriplet` — holds a single extracted (subject, relation, object) triplet along with its source file info and LLM confidence score.
- `SourceFileRecord` — represents a loaded source file, storing its name, path, full text, and optional embedding.
- `CanonicalNode` — a deduplicated entity node with a canonical name, aliases, source references, mention counts, and a weighted average embedding.
- `CanonicalRelation` — a deduplicated relation label with a canonical name, aliases, and a weighted average embedding.

**Main class — `TripletIngestionPipeline`:**
| Method | Description |
|---|---|
| `__init__` | Initializes the OpenAI client, Neo4j driver, and all tunable parameters. |
| `read_source_files` | Reads `.txt` files from disk into `SourceFileRecord` objects. |
| `split_into_chunks` | Splits file text into character-bounded chunks for LLM processing. |
| `extract_triplets_from_sources` | Orchestrates parallel chunk extraction across all source files. |
| `extract_triplets_from_chunk` | Calls GPT to extract triplets from a single text chunk. |
| `filter_triplets` | Removes triplets below the minimum confidence threshold. |
| `embed_texts` | Calls the OpenAI Embeddings API and caches results. |
| `merge_nodes` | Clusters near-duplicate entity names using cosine similarity. |
| `merge_relations` | Clusters near-duplicate relation labels using cosine similarity. |
| `build_graph_payload` | Assembles the final deduplicated list of nodes and edges. |
| `write_graph_to_neo4j` | Upserts nodes and relationships into Neo4j. |
| `semantic_search_entities` | Searches entity nodes by embedding similarity (vector index or in-memory fallback). |
| `semantic_search_sources` | Searches source file nodes by embedding similarity. |
| `run` | Top-level entry point: reads → extracts → filters → merges → writes. |

**Utility functions:**
- `normalize_text` / `relation_to_key` — text normalization helpers.
- `cosine_similarity` — pure-Python cosine similarity calculation.
- `weighted_average_embedding` — merges two embeddings proportionally by occurrence weight.
- `split_text_into_chunks` — paragraph-aware text chunker.
- `strip_json_fence` / `parse_json_object` — robust LLM JSON response parsers.

---

### `neo4j_query_interface.py`

**Main class — `Neo4jQueryInterface`:**
| Method | Description |
|---|---|
| `__init__` | Initializes the Neo4j driver and OpenAI client from env vars or constructor arguments. |
| `get_database_schema` | Introspects the live Neo4j database to retrieve node labels, relationship types, and property keys. |
| `translate_to_cypher` | Sends the schema and natural language question to GPT and returns a valid Cypher query string. |
| `execute_cypher` | Runs a Cypher query against Neo4j and returns the raw result records. |
| `results_to_natural_language` | Sends raw query results back to GPT to produce a concise human-readable answer. |
| `query` | Top-level entry point: schema → Cypher → execute → summarize. |
| `close` | Closes the Neo4j driver connection. |

---

## Authorship

**All code in this repository was written entirely by the repository owner.** No part of the source code was generated or produced by AI tools. The project uses third-party libraries (OpenAI, Neo4j Python driver) as dependencies, and calls the OpenAI API as an external service at runtime, but every line of Python in this repository is original work by the author.

---

## Notes

- Both scripts read credentials from environment variables or `.env`. Never hard-code secrets in source files.
- The ingestion script is idempotent — re-running it on the same files will merge new information into existing nodes rather than creating duplicates, thanks to the similarity-based merging step.
- Neo4j vector index (`entity_embedding_index`, `source_file_embedding_index`) is used for semantic search if available; otherwise the pipeline falls back to in-memory cosine similarity.