# InsureIQ — Health Insurance Policy Simplifier

A production-ready **RAG (Retrieval-Augmented Generation)** system that answers health insurance coverage questions (e.g., *"Is knee surgery covered?"*) by reading complex policy PDFs and returning **Yes/No answers with exact citations** (page, section, quote).

## Architecture

```
Google Drive PDFs → Parser/Chunker → Embeddings (ChromaDB)
    → Multi-Query Retriever → Answer Chain (Ollama llama3)
    → FastAPI REST API → InsureIQ Streamlit UI
```

## Project Structure

| File / Folder | Purpose |
|---|---|
| `drive_downloader.py` | Downloads PDFs from a shared Google Drive folder via service account |
| `pdf_parser.py` | Extracts text, detects sections, chunks at ~800 tokens with overlap |
| `embeddings.py` | Embeds chunks using `all-MiniLM-L6-v2` and stores in ChromaDB |
| `retriever.py` | Multi-query retrieval with Ollama llama3 query expansion |
| `answer_chain.py` | RAG answer generation — retrieves chunks, sends to Ollama, returns structured JSON |
| `api.py` | FastAPI REST backend (`POST /ask`, `GET /health`, `GET /stats`, `GET /pdfs`) |
| `app.py` | **InsureIQ** — production Streamlit UI with 3-tab layout, verdict banners, citations |
| `agent.py` | `CoverageAgent` class — standalone Ollama-powered coverage checker |
| `test_suite.py` | Demo validation script (10 predefined questions) |
| `assets/style.css` | CSS variable reference |
| `components/answer_card.py` | Reusable answer-card rendering component |

## Installation & Setup

### Prerequisites

| Requirement | Details |
|---|---|
| **Python** | 3.10 or later — [python.org/downloads](https://www.python.org/downloads/) |
| **Ollama** | Local LLM runtime — [ollama.com](https://ollama.com) |
| **Git** | (optional) To clone this repo |
| **Google Service Account** | Only needed if downloading PDFs from Google Drive |

### Step 1 — Clone the repository

```bash
git clone https://github.com/karthikSuthari/Health-Insurance-Policy-Simplifier.git
cd Health-Insurance-Policy-Simplifier
```

### Step 2 — Create a virtual environment

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3 — Install Python dependencies

```bash
pip install -r requirements.txt
```

This installs: `google-api-python-client`, `google-auth`, `pdfplumber`, `tiktoken`, `chromadb`, `sentence-transformers`, `fastapi`, `uvicorn`, `streamlit`, `requests`.

### Step 4 — Install & start Ollama

1. Download and install Ollama from [ollama.com](https://ollama.com).
2. Pull the llama3 model (~4.7 GB):

```bash
ollama pull llama3
```

3. Make sure the Ollama server is running (it starts automatically on install, or run manually):

```bash
ollama serve
```

> Ollama runs on `http://localhost:11434` by default.

### Step 5 — Prepare the data (PDF pipeline)

```bash
# 5a — Download PDFs from Google Drive (requires credentials.json in project root)
python drive_downloader.py

# 5b — Parse & chunk all PDFs into JSON
python pdf_parser.py
# → Creates data/all_chunks.json (~5 MB, ~1,400 chunks)

# 5c — Generate embeddings and store in ChromaDB
python embeddings.py --reset
# → Creates data/chromadb/ with 1,410 embeddings (384-dim vectors)
```

> **Note:** If HuggingFace model download stalls, set the environment variable first:
> - Windows: `$env:HF_HUB_DISABLE_XET="1"`
> - Linux/macOS: `export HF_HUB_DISABLE_XET=1`

### Step 6 — Start the application

Open **two terminals** (both with the virtual environment activated):

**Terminal 1 — FastAPI backend (port 8000):**

```bash
python api.py
```

Wait until you see `Ready — 1410 embeddings loaded` in the console (~15–20 seconds on first run while loading the embedding model).

**Terminal 2 — Streamlit frontend (port 8501):**

```bash
streamlit run app.py
```

### Step 7 — Open the app

Open your browser and go to:

```
http://localhost:8501
```

You should see the **InsureIQ** dashboard with live API status in the top bar.

---

## Quick Start (TL;DR)

```bash
git clone https://github.com/karthikSuthari/Health-Insurance-Policy-Simplifier.git
cd Health-Insurance-Policy-Simplifier
python -m venv .venv && .venv\Scripts\activate      # Windows
pip install -r requirements.txt
ollama pull llama3

python drive_downloader.py    # download PDFs
python pdf_parser.py          # parse & chunk
python embeddings.py --reset  # embed into ChromaDB

python api.py                 # Terminal 1 — API on :8000
streamlit run app.py          # Terminal 2 — UI  on :8501
```

### API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/ask` | Answer a question — body: `{"question": "...", "top_k": 10}` |
| `GET` | `/ask?q=...` | Quick query via URL parameter |
| `GET` | `/health` | Health check (status + embedding count) |
| `GET` | `/stats` | Collection statistics + Ollama status |
| `GET` | `/pdfs` | List indexed policy PDF filenames |
| `GET` | `/docs` | Interactive Swagger UI |

### API Response Format

```json
{
  "answer": "Yes | No | Partial | Unknown",
  "explanation": "Plain-English summary of coverage",
  "confidence": 0.92,
  "citations": [
    {
      "filename": "policy.pdf",
      "page": 12,
      "section": "Section 4 — In-Patient Treatment",
      "quote": "Exact text from the policy document…"
    }
  ],
  "caveats": ["24-month waiting period for pre-existing conditions"],
  "_meta": {
    "retrieval_time_s": 1.2,
    "generation_time_s": 8.4,
    "total_time_s": 9.6,
    "chunks_used": 5
  }
}
```

## InsureIQ UI

The Streamlit frontend (`app.py`) provides a professional, production-ready interface:

### Features

| Feature | Description |
|---|---|
| **Top Bar** | Live API health status, embedding count, policy count |
| **Sidebar** | PDF policy selector, 6 quick-question buttons, compare toggle, settings |
| **Tab 1 — Ask** | Free-text question input → animated verdict banner (COVERED / NOT COVERED / CONDITIONAL / UNCLEAR), confidence circle, stat cards, plain-English explanation, collapsible citations with colour-coded cards, caveat pills, chat history |
| **Tab 2 — Compare** | Side-by-side coverage comparison across up to 3 policies |
| **Tab 3 — Dashboard** | Embedding stats, model info, Ollama status, recent queries, full policy list |
| **Footer** | Session query count, elapsed time, model branding |

### Design System

- **Font:** Inter (Google Fonts)
- **Palette:** Navy `#0A2342` · Blue `#1B6CA8` · Cyan `#00B4D8` · Green `#2ECC71` · Amber `#F39C12` · Red `#E74C3C`
- **Cards:** 16px radius, subtle shadows, hover lift
- **Animations:** fadeSlide verdicts, popIn confidence circles, slideIn citations
- **Mobile responsive** via CSS media queries

### Screenshots

Open **http://localhost:8501** after starting both servers.

## Stats

- **32** policy PDFs (33.68 MB)
- **1,410** chunks (avg 784 tokens each)
- **1,410** embeddings in ChromaDB (384-dim, cosine similarity)
- **10/10** test suite pass rate (100%)

## Tech Stack

- Python 3.11
- pdfplumber · tiktoken · sentence-transformers (`all-MiniLM-L6-v2`, 384-dim)
- ChromaDB (local persistent, cosine similarity)
- Ollama llama3 (local LLM — query expansion + answer generation, JSON mode)
- FastAPI + Uvicorn (REST API)
- Streamlit (InsureIQ production UI)
