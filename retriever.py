"""
Phase 3b: Multi-Query Retriever
==================================
Takes a user question, expands it into 3 query variants using Ollama (llama3),
retrieves top-8 chunks per variant from ChromaDB, deduplicates by chunk_id,
and returns the top-10 unique chunks sorted by best score.

Requirements:
    pip install chromadb sentence-transformers requests

Usage:
    python retriever.py "Is knee replacement surgery covered?"
    python retriever.py --test
"""

import json
import os
import sys
import time
import logging
from pathlib import Path

import requests as http_requests

from embeddings import EmbeddingStore, CHROMA_DIR, COLLECTION_NAME, EMBEDDING_MODEL

# ─── Configuration ────────────────────────────────────────────────────────────

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")
TOP_K_PER_QUERY = 8       # chunks retrieved per query variant
FINAL_TOP_K = 10           # unique chunks returned after dedup
NUM_QUERY_VARIANTS = 3     # how many expanded queries Ollama generates

# ─── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("retriever")

# ─── Query Expansion via Ollama ───────────────────────────────────────────────

EXPANSION_PROMPT = """\
You are a health insurance domain expert. A user wants to find out whether \
something is covered by their health insurance policy.

Given the user's question below, generate exactly {n} alternative search \
queries that would help retrieve relevant passages from an insurance policy \
document. Each query should approach the topic from a different angle:

1. A more specific / technical version of the question.
2. A broader version that captures related concepts.
3. A version using common insurance terminology (exclusions, sub-limits, \
waiting periods, etc.).

USER QUESTION: {question}

Respond ONLY with a JSON array of {n} strings — no markdown, no explanation.
Example: ["query one", "query two", "query three"]
"""


def _call_ollama(
    prompt: str,
    system: str = "",
    temperature: float = 0.4,
    base_url: str = OLLAMA_BASE_URL,
    model: str = OLLAMA_MODEL,
    json_mode: bool = False,
) -> str:
    """
    Call the local Ollama REST API and return the assistant's response text.
    """
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": temperature},
    }
    if json_mode:
        payload["format"] = "json"

    resp = http_requests.post(
        f"{base_url}/api/chat",
        json=payload,
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"].strip()


def expand_query(
    question: str,
    n: int = NUM_QUERY_VARIANTS,
    **kwargs,
) -> list[str]:
    """
    Use Ollama (llama3) to generate *n* semantically diverse query variants.

    Falls back to simple heuristic expansion if the API call fails.
    """
    try:
        t0 = time.time()

        raw = _call_ollama(
            prompt=EXPANSION_PROMPT.format(question=question, n=n),
            temperature=0.4,
        )

        elapsed = time.time() - t0
        log.info("Ollama query expansion (%.1fs): %s", elapsed, raw)

        # Strip markdown fences if present
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[-1]
        if cleaned.endswith("```"):
            cleaned = cleaned.rsplit("```", 1)[0]
        cleaned = cleaned.strip()

        # Parse JSON array
        variants = json.loads(cleaned)
        if isinstance(variants, list) and len(variants) >= 1:
            return [str(v) for v in variants[:n]]

        log.warning("Unexpected Ollama output format — falling back to heuristic.")
        return _heuristic_expand(question, n)

    except Exception as e:
        log.error("Ollama query expansion failed: %s — using heuristic.", e)
        return _heuristic_expand(question, n)


def _heuristic_expand(question: str, n: int) -> list[str]:
    """
    Cheap fallback: rewrite the question in a few ways without an LLM.
    """
    variants = [
        question,
        f"coverage policy benefits {question}",
        f"exclusions waiting period sub-limit {question}",
    ]
    log.info("Heuristic expansion: %s", variants[:n])
    return variants[:n]


# ─── Multi-Query Retrieval ────────────────────────────────────────────────────


class MultiQueryRetriever:
    """
    Retrieves relevant chunks using multi-query expansion + deduplication.
    """

    def __init__(
        self,
        store: EmbeddingStore | None = None,
        **kwargs,
    ):
        self.store = store or EmbeddingStore()

    def retrieve(
        self,
        question: str,
        top_k_per_query: int = TOP_K_PER_QUERY,
        final_top_k: int = FINAL_TOP_K,
        num_variants: int = NUM_QUERY_VARIANTS,
    ) -> dict:
        """
        End-to-end retrieval pipeline:
          1. Expand question into variants
          2. Retrieve top_k_per_query chunks per variant
          3. Deduplicate by chunk_id, keeping best score
          4. Return top final_top_k unique chunks

        Returns:
            {
                "question": str,
                "query_variants": [str, ...],
                "total_retrieved": int,
                "unique_chunks": int,
                "results": [
                    {"chunk_id", "text", "score", "metadata"}, …
                ]
            }
        """
        t0 = time.time()

        # 1. Expand
        variants = expand_query(question, n=num_variants)
        # Always include the original question as well
        all_queries = [question] + [v for v in variants if v.lower() != question.lower()]
        log.info("Queries to run (%d): %s", len(all_queries), all_queries)

        # 2. Retrieve per variant
        all_hits: list[dict] = []
        for i, q in enumerate(all_queries):
            hits = self.store.query(q, n_results=top_k_per_query)
            log.info(
                "  Query %d/%d  →  %d hits  (best score: %.4f)",
                i + 1,
                len(all_queries),
                len(hits),
                hits[0]["score"] if hits else 0.0,
            )
            all_hits.extend(hits)

        total_retrieved = len(all_hits)

        # 3. Deduplicate — keep the best (highest) score per chunk_id
        best_by_id: dict[str, dict] = {}
        for hit in all_hits:
            cid = hit["chunk_id"]
            if cid not in best_by_id or hit["score"] > best_by_id[cid]["score"]:
                best_by_id[cid] = hit

        # 4. Sort by score descending, take top N
        unique_sorted = sorted(best_by_id.values(), key=lambda h: h["score"], reverse=True)
        top_results = unique_sorted[:final_top_k]

        elapsed = time.time() - t0
        log.info(
            "Retrieved %d total → %d unique → returning top %d  (%.2fs)",
            total_retrieved,
            len(best_by_id),
            len(top_results),
            elapsed,
        )

        return {
            "question": question,
            "query_variants": all_queries,
            "total_retrieved": total_retrieved,
            "unique_chunks": len(best_by_id),
            "results": top_results,
            "time_seconds": round(elapsed, 2),
        }


# ─── Pretty Printer ──────────────────────────────────────────────────────────


def print_results(result: dict) -> None:
    """Print retrieval results in a human-readable format."""
    print("\n" + "═" * 70)
    print(f"  QUESTION: {result['question']}")
    print(f"  QUERIES:  {result['query_variants']}")
    print(f"  TOTAL HITS: {result['total_retrieved']}  →  UNIQUE: {result['unique_chunks']}  →  TOP: {len(result['results'])}")
    print("═" * 70)

    for i, r in enumerate(result["results"], 1):
        meta = r["metadata"]
        print(f"\n── [{i}] Score: {r['score']:.4f} ──")
        print(f"  File:    {meta['filename']}")
        print(f"  Page:    {meta['page_number']}–{meta['page_end']}")
        print(f"  Section: {meta['section_title']}")
        # Show first 250 chars of text
        snippet = r["text"][:250].replace("\n", " ")
        print(f"  Text:    {snippet}{'…' if len(r['text']) > 250 else ''}")

    print("\n" + "═" * 70)


# ─── Test Function ────────────────────────────────────────────────────────────


def test_retrieval(question: str = "knee surgery coverage"):
    """
    Test retrieval with a sample insurance question.
    Works with the real ChromaDB collection.
    """
    log.info("=" * 60)
    log.info("TEST RETRIEVAL: '%s'", question)

    store = EmbeddingStore()
    count = store.collection_count()
    if count == 0:
        log.error("ChromaDB collection is empty. Run embeddings.py first.")
        sys.exit(1)
    log.info("Collection has %d embeddings.", count)

    retriever = MultiQueryRetriever(store=store)
    result = retriever.retrieve(question)
    print_results(result)

    # Validate structure
    assert "results" in result
    assert len(result["results"]) <= FINAL_TOP_K
    for r in result["results"]:
        assert "chunk_id" in r
        assert "score" in r
        assert "text" in r
        assert "metadata" in r

    log.info("Test PASSED — %d results returned.", len(result["results"]))
    return result


# ─── Offline Self-Test ────────────────────────────────────────────────────────


def _offline_test():
    """Test heuristic expansion and dedup logic without API or ChromaDB."""
    log.info("Running offline self-test …")

    # Heuristic expansion
    variants = _heuristic_expand("Is knee surgery covered?", 3)
    assert len(variants) == 3
    assert "knee surgery" in variants[0].lower()
    log.info("  _heuristic_expand … OK")

    # Dedup logic
    hits = [
        {"chunk_id": "a", "score": 0.8, "text": "t1", "metadata": {}},
        {"chunk_id": "b", "score": 0.7, "text": "t2", "metadata": {}},
        {"chunk_id": "a", "score": 0.9, "text": "t1", "metadata": {}},  # duplicate, higher score
        {"chunk_id": "c", "score": 0.6, "text": "t3", "metadata": {}},
    ]
    best_by_id: dict[str, dict] = {}
    for hit in hits:
        cid = hit["chunk_id"]
        if cid not in best_by_id or hit["score"] > best_by_id[cid]["score"]:
            best_by_id[cid] = hit

    assert len(best_by_id) == 3  # a, b, c
    assert best_by_id["a"]["score"] == 0.9  # kept the higher score
    log.info("  dedup logic … OK")

    log.info("Offline self-test PASSED.")


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if "--offline-test" in sys.argv:
        _offline_test()
        sys.exit(0)

    if "--test" in sys.argv:
        _offline_test()
        question = " ".join(a for a in sys.argv[1:] if not a.startswith("--")) or "knee surgery coverage"
        test_retrieval(question)
        sys.exit(0)

    if len(sys.argv) > 1 and not sys.argv[1].startswith("--"):
        question = " ".join(sys.argv[1:])
        store = EmbeddingStore()
        retriever = MultiQueryRetriever(store=store)
        result = retriever.retrieve(question)
        print_results(result)
        # Also output machine-readable JSON
        print(json.dumps(result, indent=2, default=str))
    else:
        print("Usage:")
        print('  python retriever.py "Is knee replacement surgery covered?"')
        print("  python retriever.py --test")
        print("  python retriever.py --offline-test")
        sys.exit(1)
