import argparse
import json
import math
import os
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional convenience dependency
    def load_dotenv() -> bool:
        return False
load_dotenv()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_text(value: str) -> str:
    return " ".join(value.strip().lower().split())


def relation_to_key(value: str) -> str:
    key = normalize_text(value)
    return key.replace(" ", "_")


def cosine_similarity(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
    numerator = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return numerator / (norm_a * norm_b)


def weighted_average_embedding(
    current_embedding: Optional[List[float]],
    current_weight: int,
    new_embedding: Optional[List[float]],
    new_weight: int = 1,
) -> Optional[List[float]]:
    if current_embedding is None:
        return list(new_embedding) if new_embedding else None
    if new_embedding is None:
        return current_embedding

    total_weight = current_weight + new_weight
    return [
        ((current_embedding[idx] * current_weight) + (new_embedding[idx] * new_weight)) / total_weight
        for idx in range(len(current_embedding))
    ]


def split_text_into_chunks(text: str, max_chars: int = 3500) -> List[str]:
    paragraphs = [segment.strip() for segment in text.splitlines() if segment.strip()]
    if not paragraphs:
        return []

    chunks: List[str] = []
    current: List[str] = []
    current_length = 0

    for paragraph in paragraphs:
        if len(paragraph) > max_chars:
            if current:
                chunks.append("\n".join(current))
                current = []
                current_length = 0
            for start in range(0, len(paragraph), max_chars):
                chunks.append(paragraph[start : start + max_chars])
            continue

        projected = current_length + len(paragraph) + (1 if current else 0)
        if projected > max_chars and current:
            chunks.append("\n".join(current))
            current = [paragraph]
            current_length = len(paragraph)
        else:
            current.append(paragraph)
            current_length = projected

    if current:
        chunks.append("\n".join(current))

    return chunks


def strip_json_fence(value: str) -> str:
    cleaned = value.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.replace("```json", "", 1).replace("```", "")
    return cleaned.strip()


def parse_json_object(value: str) -> Dict[str, Any]:
    cleaned = strip_json_fence(value)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1 or start >= end:
            raise
        return json.loads(cleaned[start : end + 1])


@dataclass
class RawTriplet:
    subject: str
    relation: str
    object: str
    source_file: str
    source_path: str
    chunk_index: int
    confidence: float = 1.0


@dataclass
class SourceFileRecord:
    name: str
    path: str
    text: str
    embedding: Optional[List[float]] = None


@dataclass
class CanonicalNode:
    canonical_name: str
    normalized_name: str
    aliases: set = field(default_factory=set)
    sources: set = field(default_factory=set)
    weight: int = 0
    embedding: Optional[List[float]] = None
    mention_counts: Dict[str, int] = field(default_factory=lambda: defaultdict(int))


@dataclass
class CanonicalRelation:
    canonical_name: str
    normalized_name: str
    aliases: set = field(default_factory=set)
    weight: int = 0
    embedding: Optional[List[float]] = None


class TripletIngestionPipeline:
    def __init__(
        self,
        openai_model: str = "gpt-4o-mini",
        embedding_model: str = "text-embedding-3-small",
        node_similarity_threshold: float = 0.88,
        relation_similarity_threshold: float = 0.9,
        min_triplet_confidence: float = 0.65,
        chunk_size: int = 3500,
        max_workers: int = 4,
        neo4j_uri: Optional[str] = None,
        neo4j_user: Optional[str] = None,
        neo4j_password: Optional[str] = None,
        neo4j_database: Optional[str] = None,
    ):
        self.openai_model = openai_model
        self.embedding_model = embedding_model
        self.node_similarity_threshold = node_similarity_threshold
        self.relation_similarity_threshold = relation_similarity_threshold
        self.min_triplet_confidence = min_triplet_confidence
        self.chunk_size = chunk_size
        self.max_workers = max_workers

        try:
            from openai import OpenAI  # type: ignore
        except ImportError as exc:
            raise ImportError("Missing dependency 'openai'. Install it with: pip install openai") from exc

        try:
            from neo4j import GraphDatabase  # type: ignore
        except ImportError as exc:
            raise ImportError("Missing dependency 'neo4j'. Install it with: pip install neo4j") from exc

        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        self.neo4j_uri = neo4j_uri or os.getenv("NEO4J_URI", "neo4j://127.0.0.1:7687").strip().strip('"')
        self.neo4j_user = (
            neo4j_user
            or os.getenv("NEO4J_USER")
            or os.getenv("NEO4J_USERNAME")
            or "neo4j"
        )
        self.neo4j_password = (
            neo4j_password
            or os.getenv("NEO4J_PASSWORD")
            or "password"
        )
        self.neo4j_database = (
            neo4j_database
            or os.getenv("NEO4J_DATABASE")
            or "neo4j"
        )

        self.driver = GraphDatabase.driver(
            self.neo4j_uri,
            auth=(self.neo4j_user.strip().strip('"'), self.neo4j_password.strip().strip('"')),
        )

        self._embedding_cache: Dict[str, List[float]] = {}

    def close(self) -> None:
        self.driver.close()

    def read_source_files(self, paths: Sequence[str]) -> List[SourceFileRecord]:
        def _read(path_value: str) -> SourceFileRecord:
            path = Path(path_value).expanduser().resolve()
            text = path.read_text(encoding="utf-8")
            return SourceFileRecord(name=path.name, path=str(path), text=text)

        records: List[SourceFileRecord] = []
        with ThreadPoolExecutor(max_workers=min(self.max_workers, max(1, len(paths)))) as executor:
            futures = {executor.submit(_read, path): path for path in paths}
            for future in as_completed(futures):
                records.append(future.result())

        records.sort(key=lambda item: item.name)
        if records:
            embeddings = self.embed_texts([record.text for record in records])
            for record in records:
                record.embedding = embeddings.get(record.text)
        return records

    def extract_triplets_from_chunk(
        self,
        source: SourceFileRecord,
        chunk_text: str,
        chunk_index: int,
        chunk_total: int,
    ) -> List[RawTriplet]:
        system_prompt = """You extract knowledge graph triplets from raw text.
Return JSON only with the shape:
{
  "triplets": [
    {"subject": "...", "relation": "...", "object": "...", "confidence": 0.0}
  ]
}

Rules:
- Extract only facts clearly supported by the text.
- Keep subject and object as concise entity names or noun phrases.
- Keep relation as a short verb phrase.
- Do not invent facts.
- Avoid duplicates inside one chunk.
- Preserve the source language when possible.
- confidence must be a number between 0 and 1.
- Lower confidence for vague entities, malformed snippets, isolated numbers, code fragments, or partial phrases.
- If the chunk has no useful facts, return {"triplets": []}.
"""

        user_prompt = f"""Source file: {source.name}
Chunk: {chunk_index + 1}/{chunk_total}

Text:
{chunk_text}
"""

        response = self.openai_client.chat.completions.create(
            model=self.openai_model,
            temperature=0.1,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )

        payload = parse_json_object(response.choices[0].message.content or "")

        triplets: List[RawTriplet] = []
        seen = set()
        for item in payload.get("triplets", []):
            subject = str(item.get("subject", "")).strip()
            relation = str(item.get("relation", "")).strip()
            obj = str(item.get("object", "")).strip()
            confidence = self._coerce_confidence(item.get("confidence", 0.5))
            if not subject or not relation or not obj:
                continue
            dedupe_key = (
                normalize_text(subject),
                normalize_text(relation),
                normalize_text(obj),
            )
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            triplets.append(
                RawTriplet(
                    subject=subject,
                    relation=relation,
                    object=obj,
                    source_file=source.name,
                    source_path=source.path,
                    chunk_index=chunk_index,
                    confidence=confidence,
                )
            )
        return triplets

    @staticmethod
    def _coerce_confidence(value: Any) -> float:
        try:
            confidence = float(value)
        except (TypeError, ValueError):
            confidence = 0.5
        return max(0.0, min(1.0, confidence))

    @staticmethod
    def _looks_like_numeric_noise(value: str) -> bool:
        cleaned = value.strip()
        compact = cleaned.replace(",", "").replace(".", "").replace("%", "").replace("-", "")
        return bool(cleaned) and compact.isdigit()

    @staticmethod
    def _looks_like_list_or_vector(value: str) -> bool:
        cleaned = value.strip()
        return (
            (cleaned.startswith("[") and cleaned.endswith("]"))
            or (cleaned.startswith("(") and cleaned.endswith(")"))
            or (cleaned.startswith("{") and cleaned.endswith("}"))
        )

    @staticmethod
    def _looks_like_fragment_noise(value: str) -> bool:
        cleaned = normalize_text(value)
        low_signal_terms = {
            "a class",
            "an object",
            "a function",
            "a method",
            "a variable",
            "the data",
            "this",
            "that",
            "it",
            "they",
            "he",
            "she",
            "we",
            "you",
            "i",
        }
        if cleaned in low_signal_terms:
            return True
        if len(cleaned) <= 1:
            return True
        if re.fullmatch(r"[\W_]+", cleaned):
            return True
        return False

    def is_valid_entity(self, value: str) -> bool:
        cleaned = value.strip()
        if not cleaned:
            return False
        if self._looks_like_numeric_noise(cleaned):
            return False
        if self._looks_like_list_or_vector(cleaned):
            return False
        if self._looks_like_fragment_noise(cleaned):
            return False
        return True

    def is_valid_relation(self, value: str) -> bool:
        cleaned = normalize_text(value)
        if not cleaned:
            return False
        if self._looks_like_numeric_noise(cleaned):
            return False
        if self._looks_like_list_or_vector(cleaned):
            return False
        if len(cleaned) <= 1:
            return False
        return True

    def filter_triplets(self, triplets: Sequence[RawTriplet]) -> List[RawTriplet]:
        filtered: List[RawTriplet] = []
        seen = set()

        for triplet in triplets:
            if triplet.confidence < self.min_triplet_confidence:
                continue
            if not self.is_valid_entity(triplet.subject):
                continue
            if not self.is_valid_entity(triplet.object):
                continue
            if not self.is_valid_relation(triplet.relation):
                continue

            dedupe_key = (
                normalize_text(triplet.subject),
                normalize_text(triplet.relation),
                normalize_text(triplet.object),
                triplet.source_file,
            )
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            filtered.append(triplet)

        return filtered

    def extract_triplets_from_sources(self, sources: Sequence[SourceFileRecord]) -> List[RawTriplet]:
        tasks: List[Tuple[SourceFileRecord, str, int, int]] = []
        for source in sources:
            chunks = split_text_into_chunks(source.text, self.chunk_size)
            for chunk_index, chunk in enumerate(chunks):
                tasks.append((source, chunk, chunk_index, len(chunks)))

        triplets: List[RawTriplet] = []
        with ThreadPoolExecutor(max_workers=min(self.max_workers, max(1, len(tasks)))) as executor:
            futures = {
                executor.submit(self.extract_triplets_from_chunk, source, chunk, chunk_index, chunk_total): (
                    source.name,
                    chunk_index,
                )
                for source, chunk, chunk_index, chunk_total in tasks
            }
            for future in as_completed(futures):
                triplets.extend(future.result())

        return triplets

    def embed_texts(self, texts: Iterable[str]) -> Dict[str, List[float]]:
        uncached = []
        ordered = []
        for text in texts:
            if text not in ordered:
                ordered.append(text)
            if text not in self._embedding_cache:
                uncached.append(text)

        batch_size = 100
        for start in range(0, len(uncached), batch_size):
            batch = uncached[start : start + batch_size]
            if not batch:
                continue
            response = self.openai_client.embeddings.create(
                model=self.embedding_model,
                input=batch,
            )
            for item, text in zip(response.data, batch):
                self._embedding_cache[text] = item.embedding

        return {text: self._embedding_cache[text] for text in ordered}

    def merge_nodes(self, triplets: Sequence[RawTriplet]) -> Tuple[Dict[str, CanonicalNode], Dict[str, str]]:
        mention_counter: Counter[str] = Counter()
        original_to_normalized: Dict[str, str] = {}

        for triplet in triplets:
            for entity in (triplet.subject, triplet.object):
                normalized = normalize_text(entity)
                mention_counter[normalized] += 1
                original_to_normalized.setdefault(entity, normalized)

        unique_texts = list(original_to_normalized.keys())
        embeddings = self.embed_texts(unique_texts)
        canonical_nodes: Dict[str, CanonicalNode] = {}
        original_to_canonical: Dict[str, str] = {}

        ordered_entities = sorted(
            unique_texts,
            key=lambda item: (-mention_counter[normalize_text(item)], normalize_text(item)),
        )

        for entity in ordered_entities:
            normalized = normalize_text(entity)
            embedding = embeddings.get(entity)

            if normalized in canonical_nodes:
                node = canonical_nodes[normalized]
                node.aliases.add(entity)
                original_to_canonical[entity] = normalized
                continue

            best_key = None
            best_score = -1.0

            for key, candidate in canonical_nodes.items():
                if candidate.embedding is None or embedding is None:
                    continue
                score = cosine_similarity(embedding, candidate.embedding)
                if score > best_score:
                    best_key = key
                    best_score = score

            if best_key and best_score >= self.node_similarity_threshold:
                node = canonical_nodes[best_key]
                node.aliases.add(entity)
                original_to_canonical[entity] = best_key
                continue

            canonical_nodes[normalized] = CanonicalNode(
                canonical_name=entity,
                normalized_name=normalized,
                aliases={entity},
                embedding=embedding,
            )
            original_to_canonical[entity] = normalized

        for triplet in triplets:
            for entity in (triplet.subject, triplet.object):
                canonical_key = original_to_canonical[entity]
                node = canonical_nodes[canonical_key]
                node.weight += 1
                node.sources.add(triplet.source_file)
                node.mention_counts[triplet.source_file] += 1
                node.embedding = weighted_average_embedding(node.embedding, max(node.weight - 1, 0), embeddings.get(entity))

        return canonical_nodes, original_to_canonical

    def merge_relation_labels(self, triplets: Sequence[RawTriplet]) -> Dict[str, CanonicalRelation]:
        unique_relations = sorted({triplet.relation for triplet in triplets}, key=normalize_text)
        embeddings = self.embed_texts(unique_relations)
        canonical_relations: Dict[str, CanonicalRelation] = {}

        relation_counter = Counter(normalize_text(triplet.relation) for triplet in triplets)

        for relation in sorted(unique_relations, key=lambda item: (-relation_counter[normalize_text(item)], normalize_text(item))):
            normalized = normalize_text(relation)
            embedding = embeddings.get(relation)

            if normalized in canonical_relations:
                canonical_relations[normalized].aliases.add(relation)
                continue

            best_key = None
            best_score = -1.0
            for key, candidate in canonical_relations.items():
                if candidate.embedding is None or embedding is None:
                    continue
                score = cosine_similarity(embedding, candidate.embedding)
                if score > best_score:
                    best_key = key
                    best_score = score

            if best_key and best_score >= self.relation_similarity_threshold:
                canonical_relations[best_key].aliases.add(relation)
                continue

            canonical_relations[normalized] = CanonicalRelation(
                canonical_name=relation,
                normalized_name=normalized,
                aliases={relation},
                embedding=embedding,
            )

        for triplet in triplets:
            normalized = normalize_text(triplet.relation)
            relation = canonical_relations.get(normalized)
            if relation is not None:
                relation.weight += 1
                relation.embedding = weighted_average_embedding(
                    relation.embedding,
                    max(relation.weight - 1, 0),
                    embeddings.get(triplet.relation),
                )
                continue

            for relation in canonical_relations.values():
                if triplet.relation in relation.aliases:
                    relation.weight += 1
                    relation.embedding = weighted_average_embedding(
                        relation.embedding,
                        max(relation.weight - 1, 0),
                        embeddings.get(triplet.relation),
                    )
                    break

        return canonical_relations

    def build_graph_payload(
        self,
        triplets: Sequence[RawTriplet],
        sources: Optional[Sequence[SourceFileRecord]] = None,
    ) -> Dict[str, Any]:
        canonical_nodes, original_to_canonical = self.merge_nodes(triplets)
        canonical_relations = self.merge_relation_labels(triplets)

        relation_lookup: Dict[str, str] = {}
        for key, relation in canonical_relations.items():
            relation_lookup[key] = key
            for alias in relation.aliases:
                relation_lookup[normalize_text(alias)] = key

        edge_index: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
        source_files: Dict[str, Dict[str, Any]] = {
            source.name: {"name": source.name, "path": source.path}
            for source in (sources or [])
        }

        for triplet in triplets:
            source_files[triplet.source_file] = {
                "name": triplet.source_file,
                "path": triplet.source_path,
            }

            subject_key = original_to_canonical[triplet.subject]
            object_key = original_to_canonical[triplet.object]
            relation_key = relation_lookup[normalize_text(triplet.relation)]
            relation = canonical_relations[relation_key]

            edge_key = (subject_key, relation_key, object_key)
            if edge_key not in edge_index:
                edge_index[edge_key] = {
                    "source": subject_key,
                    "target": object_key,
                    "relation_type": relation.canonical_name,
                    "relation_key": relation_to_key(relation.canonical_name),
                    "aliases": set(relation.aliases),
                    "weight": 0,
                    "sources": set(),
                }

            edge = edge_index[edge_key]
            edge["weight"] += 1
            edge["sources"].add(triplet.source_file)

        nodes_payload = [
            {
                "canonical_name": node.canonical_name,
                "normalized_name": node.normalized_name,
                "aliases": sorted(node.aliases),
                "sources": sorted(node.sources),
                "weight": node.weight,
                "mention_counts": dict(node.mention_counts),
                "embedding": node.embedding,
            }
            for node in canonical_nodes.values()
        ]

        edges_payload = [
            {
                "source": edge["source"],
                "target": edge["target"],
                "relation_type": edge["relation_type"],
                "relation_key": edge["relation_key"],
                "aliases": sorted(edge["aliases"]),
                "sources": sorted(edge["sources"]),
                "weight": edge["weight"],
            }
            for edge in edge_index.values()
        ]

        return {
            "nodes": nodes_payload,
            "edges": edges_payload,
            "sources": [
                {
                    "name": source["name"],
                    "path": source["path"],
                    "embedding": next(
                        (
                            record.embedding
                            for record in (sources or [])
                            if record.name == source["name"] and record.path == source["path"]
                        ),
                        None,
                    ),
                }
                for source in source_files.values()
            ],
            "stats": {
                "raw_triplets": len(triplets),
                "canonical_nodes": len(nodes_payload),
                "canonical_edges": len(edges_payload),
                "source_files": len(source_files),
            },
        }

    def ensure_constraints(self) -> None:
        statements = [
            "CREATE CONSTRAINT source_file_name IF NOT EXISTS FOR (s:SourceFile) REQUIRE s.name IS UNIQUE",
            "CREATE CONSTRAINT entity_normalized_name IF NOT EXISTS FOR (e:Entity) REQUIRE e.normalized_name IS UNIQUE",
        ]
        with self.driver.session(database=self.neo4j_database) as session:
            for statement in statements:
                session.run(statement)

    def ensure_vector_indexes(self, payload: Dict[str, Any]) -> None:
        entity_dimension = next((len(node["embedding"]) for node in payload["nodes"] if node.get("embedding")), None)
        source_dimension = next((len(source["embedding"]) for source in payload["sources"] if source.get("embedding")), None)

        statements = []
        if entity_dimension:
            statements.append(
                f"""
                CREATE VECTOR INDEX entity_embedding_index IF NOT EXISTS
                FOR (e:Entity) ON (e.embedding)
                OPTIONS {{indexConfig: {{
                    `vector.dimensions`: {entity_dimension},
                    `vector.similarity_function`: 'cosine'
                }}}}
                """
            )
        if source_dimension:
            statements.append(
                f"""
                CREATE VECTOR INDEX source_file_embedding_index IF NOT EXISTS
                FOR (s:SourceFile) ON (s.embedding)
                OPTIONS {{indexConfig: {{
                    `vector.dimensions`: {source_dimension},
                    `vector.similarity_function`: 'cosine'
                }}}}
                """
            )

        if not statements:
            return

        with self.driver.session(database=self.neo4j_database) as session:
            for statement in statements:
                try:
                    session.run(statement)
                except Exception:
                    pass

    def write_graph_to_neo4j(self, payload: Dict[str, Any]) -> None:
        self.ensure_constraints()
        self.ensure_vector_indexes(payload)
        ingested_at = utc_now()

        with self.driver.session(database=self.neo4j_database) as session:
            session.execute_write(self._write_sources, payload["sources"], ingested_at)
            session.execute_write(self._write_nodes, payload["nodes"], ingested_at)
            session.execute_write(self._write_mentions, payload["nodes"], ingested_at)
            session.execute_write(self._write_edges, payload["edges"], ingested_at)

    @staticmethod
    def _write_sources(tx, sources: List[Dict[str, Any]], ingested_at: str) -> None:
        tx.run(
            """
            UNWIND $sources AS source
            MERGE (s:SourceFile {name: source.name})
            SET s.path = source.path,
                s.embedding = source.embedding,
                s.last_ingested_at = $ingested_at
            """,
            sources=sources,
            ingested_at=ingested_at,
        )

    @staticmethod
    def _write_nodes(tx, nodes: List[Dict[str, Any]], ingested_at: str) -> None:
        tx.run(
            """
            UNWIND $nodes AS node
            MERGE (e:Entity {normalized_name: node.normalized_name})
            ON CREATE SET e.created_at = $ingested_at
            SET e.canonical_name = node.canonical_name,
                e.aliases = node.aliases,
                e.sources = node.sources,
                e.embedding = node.embedding,
                e.weight = node.weight,
                e.last_ingested_at = $ingested_at
            """,
            nodes=nodes,
            ingested_at=ingested_at,
        )

    @staticmethod
    def _write_mentions(tx, nodes: List[Dict[str, Any]], ingested_at: str) -> None:
        mention_rows = []
        for node in nodes:
            for source_name, count in node["mention_counts"].items():
                mention_rows.append(
                    {
                        "normalized_name": node["normalized_name"],
                        "source_name": source_name,
                        "count": count,
                    }
                )

        if not mention_rows:
            return

        tx.run(
            """
            UNWIND $rows AS row
            MATCH (e:Entity {normalized_name: row.normalized_name})
            MATCH (s:SourceFile {name: row.source_name})
            MERGE (e)-[r:MENTIONED_IN]->(s)
            SET r.weight = row.count,
                r.last_ingested_at = $ingested_at
            """,
            rows=mention_rows,
            ingested_at=ingested_at,
        )

    @staticmethod
    def _write_edges(tx, edges: List[Dict[str, Any]], ingested_at: str) -> None:
        tx.run(
            """
            UNWIND $edges AS edge
            MATCH (source:Entity {normalized_name: edge.source})
            MATCH (target:Entity {normalized_name: edge.target})
            MERGE (source)-[r:RELATES_TO {relation_key: edge.relation_key}]->(target)
            ON CREATE SET r.created_at = $ingested_at
            SET r.relation_type = edge.relation_type,
                r.aliases = edge.aliases,
                r.sources = edge.sources,
                r.weight = edge.weight,
                r.last_ingested_at = $ingested_at
            """,
            edges=edges,
            ingested_at=ingested_at,
        )

    def run(self, paths: Sequence[str]) -> Dict[str, Any]:
        if not paths:
            raise ValueError("Please provide at least one .txt file path.")

        sources = self.read_source_files(paths)
        raw_triplets = self.extract_triplets_from_sources(sources)
        triplets = self.filter_triplets(raw_triplets)
        payload = self.build_graph_payload(triplets, sources=sources)
        payload["stats"]["filtered_out_triplets"] = len(raw_triplets) - len(triplets)
        self.write_graph_to_neo4j(payload)
        return payload

    def semantic_search_entities(self, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        query_embedding = self.embed_texts([query_text])[query_text]
        with self.driver.session(database=self.neo4j_database) as session:
            try:
                result = session.run(
                    """
                    CALL db.index.vector.queryNodes('entity_embedding_index', $top_k, $embedding)
                    YIELD node, score
                    RETURN node.canonical_name AS canonical_name,
                           node.normalized_name AS normalized_name,
                           node.sources AS sources,
                           node.weight AS weight,
                           score
                    ORDER BY score DESC
                    """,
                    top_k=top_k,
                    embedding=query_embedding,
                )
                return [record.data() for record in result]
            except Exception:
                result = session.run(
                    """
                    MATCH (e:Entity)
                    WHERE e.embedding IS NOT NULL
                    RETURN e.canonical_name AS canonical_name,
                           e.normalized_name AS normalized_name,
                           e.sources AS sources,
                           e.weight AS weight,
                           e.embedding AS embedding
                    """
                )
                rows = [record.data() for record in result]
                for row in rows:
                    row["score"] = cosine_similarity(query_embedding, row["embedding"])
                    row.pop("embedding", None)
                rows.sort(key=lambda item: item["score"], reverse=True)
                return rows[:top_k]

    def semantic_search_sources(self, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        query_embedding = self.embed_texts([query_text])[query_text]
        with self.driver.session(database=self.neo4j_database) as session:
            try:
                result = session.run(
                    """
                    CALL db.index.vector.queryNodes('source_file_embedding_index', $top_k, $embedding)
                    YIELD node, score
                    RETURN node.name AS name,
                           node.path AS path,
                           score
                    ORDER BY score DESC
                    """,
                    top_k=top_k,
                    embedding=query_embedding,
                )
                return [record.data() for record in result]
            except Exception:
                result = session.run(
                    """
                    MATCH (s:SourceFile)
                    WHERE s.embedding IS NOT NULL
                    RETURN s.name AS name,
                           s.path AS path,
                           s.embedding AS embedding
                    """
                )
                rows = [record.data() for record in result]
                for row in rows:
                    row["score"] = cosine_similarity(query_embedding, row["embedding"])
                    row.pop("embedding", None)
                rows.sort(key=lambda item: item["score"], reverse=True)
                return rows[:top_k]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read TXT files, extract triplets with OpenAI, merge similar nodes/relations, and write them into Neo4j."
    )
    parser.add_argument(
        "files",
        nargs="*",
        help="One or more TXT files to ingest. Defaults to data/lecture1.txt and data/lecture2.txt.",
    )
    parser.add_argument("--chunk-size", type=int, default=3500, help="Approximate characters per extraction chunk.")
    parser.add_argument("--max-workers", type=int, default=4, help="Parallel workers for file reads and extraction calls.")
    parser.add_argument("--node-threshold", type=float, default=0.88, help="Cosine similarity threshold for merging nodes.")
    parser.add_argument("--relation-threshold", type=float, default=0.9, help="Cosine similarity threshold for merging relation labels.")
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.65,
        help="Minimum LLM confidence required for a triplet to survive filtering.",
    )
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI chat model used for triplet extraction.")
    parser.add_argument("--embedding-model", default="text-embedding-3-small", help="Embedding model used for similarity merging.")
    parser.add_argument("--database", default=None, help="Neo4j database name override.")
    parser.add_argument(
        "--save-json",
        default=None,
        help="Optional path to save the intermediate merged graph payload as JSON.",
    )
    parser.add_argument(
        "--print-limit",
        type=int,
        default=10,
        help="How many merged nodes and edges to print after ingestion.",
    )
    parser.add_argument(
        "--search-query",
        default=None,
        help="Optional semantic search query to run after ingestion.",
    )
    parser.add_argument(
        "--search-top-k",
        type=int,
        default=5,
        help="Number of semantic search results to return.",
    )
    return parser


def resolve_default_files(files: Sequence[str]) -> List[str]:
    if files:
        return list(files)

    project_root = Path(__file__).resolve().parent
    default_files = [
        project_root / "data" / "lecture1.txt",
        project_root / "data" / "lecture2.txt",
    ]
    return [str(path) for path in default_files]


def print_payload_summary(payload: Dict[str, Any], files: Sequence[str], print_limit: int) -> None:
    print("\n=== Input Files ===")
    for file_path in files:
        print(file_path)

    print("\n=== Stats ===")
    print(json.dumps(payload["stats"], ensure_ascii=False, indent=2))

    print("\n=== Nodes ===")
    sorted_nodes = sorted(
        payload["nodes"],
        key=lambda item: (-item["weight"], item["canonical_name"].lower()),
    )
    for node in sorted_nodes[:print_limit]:
        print(
            f"- {node['canonical_name']} | weight={node['weight']} | "
            f"sources={','.join(node['sources'])} | aliases={', '.join(node['aliases'])}"
        )

    print("\n=== Edges ===")
    sorted_edges = sorted(
        payload["edges"],
        key=lambda item: (-item["weight"], item["relation_type"].lower(), item["source"], item["target"]),
    )
    for edge in sorted_edges[:print_limit]:
        print(
            f"- ({edge['source']}) -[{edge['relation_type']} / weight={edge['weight']}]-> "
            f"({edge['target']}) | sources={','.join(edge['sources'])}"
        )


def print_search_results(title: str, results: Sequence[Dict[str, Any]]) -> None:
    print(f"\n=== {title} ===")
    if not results:
        print("No results")
        return

    for item in results:
        name = item.get("canonical_name") or item.get("name") or item.get("path")
        score = item.get("score")
        score_text = f"{score:.4f}" if isinstance(score, float) else str(score)
        print(f"- {name} | score={score_text}")


def main() -> None:
    args = build_arg_parser().parse_args()
    files = resolve_default_files(args.files)

    pipeline = TripletIngestionPipeline(
        openai_model=args.model,
        embedding_model=args.embedding_model,
        node_similarity_threshold=args.node_threshold,
        relation_similarity_threshold=args.relation_threshold,
        min_triplet_confidence=args.min_confidence,
        chunk_size=args.chunk_size,
        max_workers=args.max_workers,
        neo4j_database=args.database,
    )

    try:
        payload = pipeline.run(files)
        if args.save_json:
            output_path = Path(args.save_json).expanduser().resolve()
            output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"\nSaved merged payload JSON to: {output_path}")

        print_payload_summary(payload, files, args.print_limit)
        if args.search_query:
            entity_results = pipeline.semantic_search_entities(args.search_query, top_k=args.search_top_k)
            source_results = pipeline.semantic_search_sources(args.search_query, top_k=args.search_top_k)
            print_search_results(f"Entity Search: {args.search_query}", entity_results)
            print_search_results(f"Source Search: {args.search_query}", source_results)
    finally:
        pipeline.close()


if __name__ == "__main__":
    main()
