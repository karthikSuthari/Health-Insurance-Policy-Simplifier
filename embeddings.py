"""
Phase 3a: Embeddings Ingestion
================================
Loads parsed chunks from JSON, embeds them using HuggingFace all-MiniLM-L6-v2,
and stores everything in a persistent ChromaDB collection.

Skips re-embedding if the collection already exists with the expected count.

Requirements:
    pip install chromadb sentence-transformers

Usage:
    python embeddings.py            # ingest all chunks
    python embeddings.py --reset    # delete existing collection first
    python embeddings.py --test     # offline self-test
"""

import json
import sys
import time
import logging
from pathlib import Path

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# ─── Configuration ────────────────────────────────────────────────────────────

CHUNKS_FILE = Path("./data/chunks/all_chunks.json")
CHROMA_DIR = Path("./data/chromadb")
COLLECTION_NAME = "insurance_policies"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
BATCH_SIZE = 128  # ChromaDB upsert batch size

# ─── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("embeddings")

# ─── Core ─────────────────────────────────────────────────────────────────────


class EmbeddingStore:
    """Manages ChromaDB collection and HuggingFace embedding model."""

    def __init__(
        self,
        chroma_dir: Path = CHROMA_DIR,
        collection_name: str = COLLECTION_NAME,
        model_name: str = EMBEDDING_MODEL,
    ):
        self.chroma_dir = chroma_dir
        self.collection_name = collection_name
        self.model_name = model_name

        self._model: SentenceTransformer | None = None
        self._client: chromadb.ClientAPI | None = None
        self._collection: chromadb.Collection | None = None

    # ── Lazy loaders ─────────────────────────────────────────────────────

    def _get_model(self) -> SentenceTransformer:
        if self._model is None:
            log.info("Loading embedding model: %s …", self.model_name)
            t0 = time.time()
            self._model = SentenceTransformer(self.model_name)
            log.info("  Model loaded in %.1fs", time.time() - t0)
        return self._model

    def _get_client(self) -> chromadb.ClientAPI:
        if self._client is None:
            self.chroma_dir.mkdir(parents=True, exist_ok=True)
            log.info("Connecting to ChromaDB at %s", self.chroma_dir.resolve())
            self._client = chromadb.PersistentClient(
                path=str(self.chroma_dir),
                settings=Settings(anonymized_telemetry=False),
            )
        return self._client

    def _get_collection(self) -> chromadb.Collection:
        if self._collection is None:
            client = self._get_client()
            self._collection = client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},  # cosine similarity
            )
            log.info(
                "Collection '%s' – current count: %d",
                self.collection_name,
                self._collection.count(),
            )
        return self._collection

    # ── Public API ───────────────────────────────────────────────────────

    def collection_count(self) -> int:
        """Return the number of embeddings currently in the collection."""
        return self._get_collection().count()

    def reset_collection(self) -> None:
        """Delete and recreate the collection."""
        client = self._get_client()
        try:
            client.delete_collection(self.collection_name)
            log.info("Deleted existing collection '%s'.", self.collection_name)
        except Exception:
            pass
        self._collection = None
        self._get_collection()

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts and return vectors."""
        model = self._get_model()
        embeddings = model.encode(
            texts,
            show_progress_bar=False,
            batch_size=BATCH_SIZE,
            normalize_embeddings=True,
        )
        return embeddings.tolist()

    def ingest_chunks(
        self,
        chunks: list[dict],
        force: bool = False,
    ) -> dict:
        """
        Embed and upsert all chunks into ChromaDB.

        If the collection already has the expected number of documents and
        *force* is False, skips ingestion entirely.

        Returns:
            Summary dict with counts and timing.
        """
        collection = self._get_collection()
        existing = collection.count()

        if existing >= len(chunks) and not force:
            log.info(
                "SKIP: collection already has %d embeddings (expected %d). "
                "Use --reset to re-embed.",
                existing,
                len(chunks),
            )
            return {
                "status": "skipped",
                "reason": "collection already populated",
                "existing_count": existing,
                "expected_count": len(chunks),
            }

        log.info("Ingesting %d chunks (batch size %d) …", len(chunks), BATCH_SIZE)
        t_start = time.time()

        # Process in batches
        total_embedded = 0
        for batch_start in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[batch_start : batch_start + BATCH_SIZE]

            ids = [c["chunk_id"] for c in batch]
            texts = [c["text"] for c in batch]
            metadatas = [
                {
                    "filename": c["filename"],
                    "page_number": c["page_number"],
                    "page_end": c["page_end"],
                    "section_title": c["section_title"],
                    "token_count": c["token_count"],
                    "char_start": c["char_start"],
                    "char_end": c["char_end"],
                }
                for c in batch
            ]

            # Embed
            embeddings = self.embed_texts(texts)

            # Upsert (idempotent – safe to re-run)
            collection.upsert(
                ids=ids,
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas,
            )

            total_embedded += len(batch)
            elapsed = time.time() - t_start
            rate = total_embedded / elapsed if elapsed > 0 else 0
            log.info(
                "  %d / %d embedded  (%.0f chunks/s  |  %.1fs elapsed)",
                total_embedded,
                len(chunks),
                rate,
                elapsed,
            )

        total_time = time.time() - t_start
        final_count = collection.count()

        summary = {
            "status": "success",
            "chunks_ingested": total_embedded,
            "collection_count": final_count,
            "embedding_model": self.model_name,
            "time_seconds": round(total_time, 1),
            "chunks_per_second": round(total_embedded / total_time, 1) if total_time > 0 else 0,
        }

        log.info("=" * 60)
        log.info(
            "DONE  |  %d chunks embedded in %.1fs  (%.0f/s)  |  Collection: %d",
            total_embedded,
            total_time,
            summary["chunks_per_second"],
            final_count,
        )
        log.info("=" * 60)

        return summary

    def query(
        self,
        query_text: str,
        n_results: int = 8,
    ) -> list[dict]:
        """
        Query the collection with a single text string.

        Returns:
            List of dicts: {chunk_id, text, score, metadata}
            Sorted by relevance (highest score first).
        """
        collection = self._get_collection()
        if collection.count() == 0:
            log.warning("Collection is empty — run ingestion first.")
            return []

        query_embedding = self.embed_texts([query_text])[0]

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(n_results, collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        hits: list[dict] = []
        for i in range(len(results["ids"][0])):
            # ChromaDB returns cosine *distance* (0 = identical); convert to score
            distance = results["distances"][0][i]
            score = 1.0 - distance  # cosine similarity

            hits.append({
                "chunk_id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "score": round(score, 4),
                "metadata": results["metadatas"][0][i],
            })

        return hits


# ─── File Loader ──────────────────────────────────────────────────────────────


def load_chunks(path: Path = CHUNKS_FILE) -> list[dict]:
    """Load chunks JSON produced by pdf_parser.py."""
    if not path.exists():
        raise FileNotFoundError(
            f"Chunks file not found: {path}\n"
            "Run pdf_parser.py first to generate it."
        )
    log.info("Loading chunks from %s …", path)
    with open(path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    log.info("  Loaded %d chunks.", len(chunks))
    return chunks


# ─── Main Ingestion ──────────────────────────────────────────────────────────


def run_ingestion(reset: bool = False) -> dict:
    """Full pipeline: load chunks → embed → store."""
    chunks = load_chunks()
    store = EmbeddingStore()
    if reset:
        store.reset_collection()
    return store.ingest_chunks(chunks)


# ─── Self-Test ────────────────────────────────────────────────────────────────


def _self_test():
    """Quick offline test: embed a few fake chunks, query, verify results."""
    import tempfile
    import shutil

    log.info("Running self-test …")

    tmp_dir = Path(tempfile.mkdtemp(prefix="chroma_test_"))
    try:
        store = EmbeddingStore(chroma_dir=tmp_dir, collection_name="test")

        fake_chunks = [
            {
                "chunk_id": "test_001",
                "text": "Knee replacement surgery is covered under this policy up to Rs 5,00,000.",
                "token_count": 15,
                "filename": "test.pdf",
                "page_number": 1,
                "page_end": 1,
                "section_title": "Surgical Benefits",
                "char_start": 0,
                "char_end": 72,
            },
            {
                "chunk_id": "test_002",
                "text": "Pre-existing diseases are excluded for the first 48 months of the policy.",
                "token_count": 14,
                "filename": "test.pdf",
                "page_number": 3,
                "page_end": 3,
                "section_title": "Exclusions",
                "char_start": 200,
                "char_end": 272,
            },
            {
                "chunk_id": "test_003",
                "text": "Maternity expenses are covered after a waiting period of 9 months.",
                "token_count": 13,
                "filename": "test.pdf",
                "page_number": 5,
                "page_end": 5,
                "section_title": "Benefits",
                "char_start": 400,
                "char_end": 466,
            },
        ]

        summary = store.ingest_chunks(fake_chunks, force=True)
        assert summary["status"] == "success"
        assert summary["chunks_ingested"] == 3
        log.info("  Ingestion … OK")

        # Skip test should work now
        summary2 = store.ingest_chunks(fake_chunks, force=False)
        assert summary2["status"] == "skipped"
        log.info("  Skip logic … OK")

        # Query
        results = store.query("Is knee surgery covered?", n_results=3)
        assert len(results) > 0
        assert results[0]["chunk_id"] == "test_001"  # most relevant
        assert results[0]["score"] > 0.3
        log.info("  Query … OK  (top hit: '%s', score=%.4f)", results[0]["chunk_id"], results[0]["score"])

        log.info("Self-test PASSED.")

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if "--test" in sys.argv:
        _self_test()
        sys.exit(0)

    reset = "--reset" in sys.argv
    summary = run_ingestion(reset=reset)
    print(json.dumps(summary, indent=2))
