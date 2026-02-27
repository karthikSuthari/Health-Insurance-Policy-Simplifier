"""
Phase 5: FastAPI Backend
==========================
REST API that wraps the RAG pipeline.

Endpoints:
    POST /ask          — Answer an insurance question (JSON body: {"question": "..."})
    GET  /health       — Health-check
    GET  /stats        — ChromaDB collection statistics
    GET  /ask          — Quick question via query parameter (?q=...)

Requirements:
    pip install fastapi uvicorn requests

Usage:
    python api.py                       # start on port 8000
    python api.py --port 8080           # custom port
    python api.py --test                # offline self-test
"""

import json
import os
import sys
import time
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

import requests as http_requests

from embeddings import EmbeddingStore
from retriever import MultiQueryRetriever
from answer_chain import AnswerChain


def _check_ollama() -> bool:
    """Return True if the Ollama server is reachable."""
    try:
        r = http_requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
        return r.status_code == 200
    except Exception:
        return False

# ─── Configuration ────────────────────────────────────────────────────────────

HOST = os.getenv("API_HOST", "0.0.0.0")
PORT = int(os.getenv("API_PORT", "8000"))
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# ─── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("api")

# ─── Shared State ─────────────────────────────────────────────────────────────

_store: EmbeddingStore | None = None
_chain: AnswerChain | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise heavy objects once at startup."""
    global _store, _chain
    log.info("Initialising embedding store & answer chain …")
    t0 = time.time()
    _store = EmbeddingStore()
    retriever = MultiQueryRetriever(store=_store)
    _chain = AnswerChain(retriever=retriever)
    count = _store.collection_count()
    log.info("Ready — %d embeddings loaded (%.1fs)", count, time.time() - t0)
    yield
    log.info("Shutting down.")


# ─── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Health Insurance Policy Simplifier",
    description="Ask plain-English questions about health insurance policies.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Request / Response Models ────────────────────────────────────────────────


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=500, examples=["Is knee replacement surgery covered?"])
    top_k: int = Field(default=10, ge=1, le=30, description="Number of chunks to retrieve")


class Citation(BaseModel):
    filename: str = ""
    page: int | str = ""
    section: str = ""
    quote: str = ""


class AskResponse(BaseModel):
    model_config = {"populate_by_name": True}

    answer: str = Field(..., description="Yes / No / Partial / Unknown")
    explanation: str = ""
    confidence: float = 0.0
    citations: list[Citation] = []
    caveats: list[str] = []
    meta: dict = Field(default_factory=dict, alias="_meta")


class HealthResponse(BaseModel):
    status: str
    embeddings: int
    model: str


class StatsResponse(BaseModel):
    collection_name: str
    embedding_count: int
    embedding_model: str
    llm_model: str
    ollama_available: bool


# ─── Endpoints ────────────────────────────────────────────────────────────────


@app.get("/health", response_model=HealthResponse, tags=["system"])
async def health_check():
    """Lightweight health-check."""
    count = _store.collection_count() if _store else 0
    return HealthResponse(
        status="ok" if count > 0 else "empty",
        embeddings=count,
        model="all-MiniLM-L6-v2",
    )


@app.get("/stats", response_model=StatsResponse, tags=["system"])
async def stats():
    """Collection statistics."""
    count = _store.collection_count() if _store else 0
    return StatsResponse(
        collection_name="insurance_policies",
        embedding_count=count,
        embedding_model="all-MiniLM-L6-v2",
        llm_model=_chain.model if _chain else "n/a",
        ollama_available=_check_ollama(),
    )


@app.post("/ask", tags=["query"])
async def ask_post(req: AskRequest):
    """
    Answer a health insurance question.

    Returns a structured JSON response with:
    - answer: Yes / No / Partial
    - explanation: plain-English summary
    - citations: list of exact quotes with page/section references
    - caveats: conditions, waiting periods, sub-limits
    """
    if not _chain:
        raise HTTPException(status_code=503, detail="Service not ready. Try again shortly.")

    log.info("POST /ask  question=%r  top_k=%d", req.question, req.top_k)
    t0 = time.time()

    try:
        result = _chain.answer(req.question, top_k=req.top_k)
    except Exception as e:
        log.error("Answer chain error: %s", e)
        raise HTTPException(status_code=500, detail=f"Processing error: {e}")

    log.info("Answered in %.2fs  →  %s", time.time() - t0, result.get("answer", "?"))
    return result


@app.get("/ask", tags=["query"])
async def ask_get(
    q: str = Query(..., min_length=3, max_length=500, description="Your insurance question"),
    top_k: int = Query(default=10, ge=1, le=30),
):
    """Quick query via GET parameter. Same response as POST /ask."""
    req = AskRequest(question=q, top_k=top_k)
    return await ask_post(req)


@app.get("/pdfs", tags=["system"])
async def list_pdfs():
    """Return sorted list of policy PDF filenames."""
    from pathlib import Path

    pdf_dir = Path("./data/policies")
    if not pdf_dir.exists():
        return {"pdfs": []}
    names = sorted(f.name for f in pdf_dir.glob("*.pdf"))
    return {"pdfs": names}


# ─── Self-Test ────────────────────────────────────────────────────────────────


def _self_test():
    """Validate request/response models without starting the server."""
    log.info("Running API self-test …")

    # Request model validation
    req = AskRequest(question="Is surgery covered?", top_k=10)
    assert req.question == "Is surgery covered?"
    log.info("  AskRequest parsing … OK")

    # Response model
    resp = AskResponse(
        answer="Yes",
        explanation="Covered up to 5 lakh.",
        confidence=0.9,
        citations=[Citation(filename="test.pdf", page=5, section="Benefits", quote="surgery is covered")],
        caveats=["24-month waiting period"],
        _meta={"question": "test", "total_time_s": 1.2},
    )
    assert resp.answer == "Yes"
    assert len(resp.citations) == 1
    log.info("  AskResponse model … OK")

    # Health response
    hr = HealthResponse(status="ok", embeddings=1410, model="all-MiniLM-L6-v2")
    assert hr.status == "ok"
    log.info("  HealthResponse model … OK")

    log.info("API self-test PASSED.")


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if "--test" in sys.argv:
        _self_test()
        sys.exit(0)

    port = PORT
    if "--port" in sys.argv:
        idx = sys.argv.index("--port")
        if idx + 1 < len(sys.argv):
            port = int(sys.argv[idx + 1])

    log.info("Starting server on %s:%d …", HOST, port)
    uvicorn.run("api:app", host=HOST, port=port, reload=False, log_level="info")
