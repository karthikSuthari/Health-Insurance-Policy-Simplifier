"""
Coverage Agent — Ollama (llama3) powered policy analyst
=======================================================
A CoverageAgent class that:
  1. Accepts a natural-language question + optional policy filename filter
  2. Retrieves relevant chunks via retriever.py
  3. Calls Ollama llama3 with a system prompt and chunks as context
  4. Forces JSON output matching this schema:
     {covered, confidence, explanation, citations:[{file,page,section,quote}], caveats}
  5. Validates and parses the JSON response
  6. Falls back gracefully if the model returns non-JSON

Requirements:
    pip install chromadb sentence-transformers requests

Usage:
    python agent.py                          # run 5 sample questions
    python agent.py --test                   # offline self-test
    python agent.py "Is maternity covered?"  # single question
"""

import json
import os
import re
import sys
import time
import logging
from typing import Any

import requests as http_requests

from embeddings import EmbeddingStore
from retriever import MultiQueryRetriever

# ─── Configuration ────────────────────────────────────────────────────────────

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")
MAX_CONTEXT_CHARS = 12_000   # max chars of chunk context sent to LLM
FINAL_TOP_K = 5              # chunks passed to LLM

# ─── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("agent")

# ─── JSON Schema (documentation) ─────────────────────────────────────────────

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "covered":     {"type": "string", "enum": ["Yes", "No", "Partial", "Unknown"]},
        "confidence":  {"type": "number", "minimum": 0, "maximum": 1},
        "explanation": {"type": "string"},
        "citations":   {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "file":    {"type": "string"},
                    "page":    {"type": "integer"},
                    "section": {"type": "string"},
                    "quote":   {"type": "string"},
                },
                "required": ["file", "page", "section", "quote"],
            },
        },
        "caveats": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["covered", "confidence", "explanation", "citations", "caveats"],
}

# ─── System Prompt ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a health insurance policy analyst. The user will give you a question \
and policy document excerpts. You MUST answer the EXACT question asked.

STEPS:
1. Read the user's question carefully.
2. Search the provided excerpts for text that directly answers the question.
3. Determine: is the item covered ("Yes"), excluded ("No"), conditionally \
covered ("Partial"), or not found in the excerpts ("Unknown").
4. Write a concise 2-4 sentence explanation that DIRECTLY answers the question.
5. Provide exact quote(s) from the excerpts as citations — never leave quotes empty.
6. List any relevant caveats (waiting periods, sub-limits, conditions).

RULES:
- "covered": "Yes" if clearly covered. "No" if excluded. "Partial" if covered \
with conditions/waiting periods. "Unknown" if excerpts don't address it.
- "confidence": 0.9+ if the excerpts clearly answer the question. 0.5-0.8 if \
somewhat relevant. Below 0.5 if unsure.
- Each citation MUST include a "quote" with at least one full sentence copied \
verbatim from the excerpts.
- Only mention caveats relevant to the user's specific question.
- Do NOT include general policy information unrelated to the question.

Respond with ONLY this JSON (no other text):
{"covered": "Yes", "confidence": 0.92, "explanation": "Knee replacement surgery is covered as an inpatient surgical procedure under Section 4.", "citations": [{"file": "policy.pdf", "page": 5, "section": "Benefits", "quote": "All daycare and inpatient surgical procedures including joint replacement are covered up to the sum insured"}], "caveats": ["48-month waiting period for joint replacements"]}
"""

USER_TEMPLATE = """\
QUESTION: {question}

Below are the most relevant excerpts from health insurance policy documents. \
Each excerpt is tagged with its source file, page number, and section.

{context}

Remember: Your explanation must directly answer "{question}".
Respond with ONLY the JSON: {{"covered": ..., "confidence": ..., "explanation": ..., "citations": [...], "caveats": [...]}}
"""

# ─── Ollama Caller ────────────────────────────────────────────────────────────


def _call_ollama(
    prompt: str,
    system: str = "",
    temperature: float = 0.2,
    base_url: str = OLLAMA_BASE_URL,
    model: str = OLLAMA_MODEL,
    json_mode: bool = False,
) -> str:
    """Call the local Ollama REST API and return the assistant's response text."""
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
        timeout=180,
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"].strip()


# ─── Context Builder ─────────────────────────────────────────────────────────


def _build_context(results: list[dict], max_chars: int = MAX_CONTEXT_CHARS) -> str:
    """Format retrieved chunks into a numbered context string for the prompt."""
    parts: list[str] = []
    total = 0

    for i, r in enumerate(results, 1):
        meta = r["metadata"]
        header = (
            f"[Excerpt {i}]  "
            f"File: {meta['filename']}  |  "
            f"Page: {meta['page_number']}–{meta['page_end']}  |  "
            f"Section: {meta['section_title']}"
        )
        block = f"{header}\n{r['text']}\n"

        if total + len(block) > max_chars:
            remaining = max_chars - total
            if remaining > 200:
                block = block[:remaining] + "\n…[truncated]"
                parts.append(block)
            break

        parts.append(block)
        total += len(block)

    return "\n".join(parts)


# ─── JSON Helpers ─────────────────────────────────────────────────────────────


def _extract_json(text: str) -> dict | None:
    """
    Try to extract a JSON object from the model's response.
    Handles markdown fences, leading/trailing text, etc.
    """
    cleaned = text.strip()

    # Strip markdown code fences
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*\n?", "", cleaned)
        cleaned = re.sub(r"\n?```\s*$", "", cleaned)
        cleaned = cleaned.strip()

    # Try direct parse
    try:
        obj = json.loads(cleaned)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    # Try to find the first { ... } block
    match = re.search(r"\{[\s\S]*\}", cleaned)
    if match:
        try:
            obj = json.loads(match.group())
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass

    return None


def _flatten_value(val: Any) -> Any:
    """Recursively unwrap nested {type, value} wrappers llama3 sometimes produces."""
    if isinstance(val, dict):
        if "value" in val and len(val) <= 3:
            return _flatten_value(val["value"])
        return {k: _flatten_value(v) for k, v in val.items()}
    if isinstance(val, list):
        return [_flatten_value(v) for v in val]
    return val


def _validate_response(data: dict) -> dict:
    """
    Validate and normalise a parsed response dict to match the expected schema.
    Fills in defaults for any missing fields. Handles llama3 quirks.
    """
    # Flatten any nested {type, value} structures
    data = _flatten_value(data)
    result = {}

    # covered (also accept "answer" key)
    covered = data.get("covered", data.get("answer", "Unknown"))
    if isinstance(covered, str) and covered.capitalize() in ("Yes", "No", "Partial", "Unknown"):
        result["covered"] = covered.capitalize()
    else:
        result["covered"] = "Unknown"

    # confidence
    conf = data.get("confidence", 0.0)
    try:
        conf = float(conf)
        result["confidence"] = max(0.0, min(1.0, conf))
    except (TypeError, ValueError):
        result["confidence"] = 0.0

    # explanation
    explanation = data.get("explanation", "")
    if isinstance(explanation, str):
        result["explanation"] = explanation.strip()
    elif isinstance(explanation, dict) and "value" in explanation:
        result["explanation"] = str(explanation["value"]).strip()
    else:
        result["explanation"] = str(explanation)

    # citations — accept many key variants llama3 may use
    raw_citations = data.get("citations", [])
    citations = []
    if isinstance(raw_citations, list):
        for c in raw_citations:
            if isinstance(c, dict):
                c = _flatten_value(c)
                citations.append({
                    "file": str(c.get("file", c.get("filename", ""))),
                    "page": _safe_int(c.get("page", c.get("page_number", 0))),
                    "section": str(c.get("section", c.get("section_title", ""))),
                    "quote": str(c.get("quote", c.get("text", ""))),
                })
    result["citations"] = citations

    # caveats
    raw_caveats = data.get("caveats", [])
    caveats = []
    if isinstance(raw_caveats, list):
        for cv in raw_caveats:
            if isinstance(cv, str) and cv.strip():
                caveats.append(cv.strip())
            elif isinstance(cv, dict):
                # llama3 sometimes returns {"caveat": "..."} or {"description": "..."}
                for v in cv.values():
                    if isinstance(v, str) and v.strip():
                        caveats.append(v.strip())
                        break
            elif cv is not None:
                caveats.append(str(cv))
    elif isinstance(raw_caveats, str) and raw_caveats.strip():
        caveats = [raw_caveats.strip()]
    result["caveats"] = caveats

    # Collect unknown top-level keys into explanation if explanation is short
    known_keys = {"covered", "answer", "confidence", "explanation",
                  "citations", "caveats", "type", "value"}
    extras = []
    for k, v in data.items():
        if k not in known_keys and isinstance(v, str) and len(v) > 10:
            extras.append(v)
    if extras and len(result["explanation"]) < 30:
        result["explanation"] = " ".join(extras)[:1000]

    return result


def _safe_int(val: Any) -> int:
    """Convert a value to int, returning 0 on failure."""
    try:
        return int(val)
    except (TypeError, ValueError):
        return 0


def _fallback_response(question: str, error: str) -> dict:
    """Return a graceful fallback when the model returns non-JSON."""
    return {
        "covered": "Unknown",
        "confidence": 0.0,
        "explanation": f"Could not determine coverage. The model response "
                       f"could not be parsed as valid JSON. Error: {error}",
        "citations": [],
        "caveats": [],
        "_meta": {
            "question": question,
            "error": error,
            "fallback": True,
        },
    }


# ─── Citation Backfill ────────────────────────────────────────────────────────


def _best_sentence(text: str, question: str) -> str:
    """Pick the sentence from *text* most relevant to *question*."""
    q_words = set(re.findall(r"\w{3,}", question.lower()))
    sentences = re.split(r"(?<=[.;])\s+", text)
    scored = []
    for s in sentences:
        s = s.strip()
        if len(s) < 20:
            continue
        s_words = set(re.findall(r"\w{3,}", s.lower()))
        overlap = len(q_words & s_words)
        scored.append((overlap, s))
    scored.sort(key=lambda x: -x[0])
    if scored:
        return scored[0][1]
    for s in sentences:
        if len(s.strip()) >= 30:
            return s.strip()
    return text[:300]


def _backfill_citations(
    citations: list[dict],
    chunks: list[dict],
    question: str,
) -> list[dict]:
    """
    Ensure every citation has a non-empty quote. If a quote is empty,
    find the most relevant sentence from the matching chunk.
    If no citations exist at all, create them from the top chunks.
    """
    # Backfill empty quotes in existing citations
    for cit in citations:
        if cit.get("quote", "").strip():
            continue
        fname = cit.get("file", "")
        page = cit.get("page", 0)
        # Try exact file+page match
        for chunk in chunks:
            m = chunk["metadata"]
            if (fname.lower() in m.get("filename", "").lower()
                    and int(m.get("page_number", -1)) <= page <= int(m.get("page_end", -1))):
                cit["quote"] = _best_sentence(chunk["text"], question)
                break
        # Fall back to loose filename match
        if not cit.get("quote", "").strip():
            for chunk in chunks:
                if fname.lower() in chunk["metadata"].get("filename", "").lower():
                    cit["quote"] = _best_sentence(chunk["text"], question)
                    break

    # If no citations at all, create from top chunks
    if not citations:
        for chunk in chunks[:3]:
            m = chunk["metadata"]
            citations.append({
                "file": m.get("filename", ""),
                "page": int(m.get("page_number", 0)),
                "section": m.get("section_title", ""),
                "quote": _best_sentence(chunk["text"], question),
            })

    # Remove any citation that still has no quote
    citations = [c for c in citations if c.get("quote", "").strip()]

    return citations


# ─── Coverage Agent ──────────────────────────────────────────────────────────


class CoverageAgent:
    """
    End-to-end coverage analysis agent:
      question → retrieve chunks → Ollama llama3 → validated JSON answer.
    """

    def __init__(
        self,
        model: str = OLLAMA_MODEL,
        base_url: str = OLLAMA_BASE_URL,
        retriever: MultiQueryRetriever | None = None,
    ):
        self.model = model
        self.base_url = base_url
        self.retriever = retriever or MultiQueryRetriever()

        # Verify Ollama is reachable
        if not self._check_ollama():
            log.warning("Ollama is not reachable at %s — calls will fail.", base_url)
        else:
            log.info("CoverageAgent ready — model=%s  base_url=%s", self.model, self.base_url)

    def _check_ollama(self) -> bool:
        """Return True if the Ollama server is reachable."""
        try:
            r = http_requests.get(f"{self.base_url}/api/tags", timeout=3)
            return r.status_code == 200
        except Exception:
            return False

    def ask(
        self,
        question: str,
        filename_filter: str | None = None,
        top_k: int = FINAL_TOP_K,
    ) -> dict:
        """
        Analyse a health insurance coverage question.

        Args:
            question:        Natural-language question.
            filename_filter: Optional policy filename to restrict retrieval to.
            top_k:           Number of chunks to pass to the model.

        Returns:
            Dict with keys: covered, confidence, explanation, citations, caveats, _meta
        """
        t0 = time.time()

        # ── 1. Retrieve relevant chunks ──────────────────────────────────
        log.info("Retrieving chunks for: %r", question)
        retrieval = self.retriever.retrieve(question, final_top_k=top_k)
        chunks = retrieval.get("results", [])

        # ── 2. Apply optional filename filter ────────────────────────────
        if filename_filter:
            filter_lower = filename_filter.lower()
            chunks = [
                c for c in chunks
                if filter_lower in c["metadata"].get("filename", "").lower()
            ]
            log.info("Filename filter %r → %d chunks remaining", filename_filter, len(chunks))

        if not chunks:
            log.warning("No chunks found for question: %r", question)
            return {
                "covered": "Unknown",
                "confidence": 0.0,
                "explanation": "No relevant policy excerpts were found for this question.",
                "citations": [],
                "caveats": [],
                "_meta": {
                    "question": question,
                    "filename_filter": filename_filter,
                    "chunks_retrieved": 0,
                    "total_time_s": round(time.time() - t0, 2),
                },
            }

        # ── 3. Build context ─────────────────────────────────────────────
        context = _build_context(chunks)
        user_prompt = USER_TEMPLATE.format(question=question, context=context)

        # ── 4. Call Ollama (llama3) with JSON mode ───────────────────────
        log.info("Calling Ollama %s (%d chunks, %d chars context) …",
                 self.model, len(chunks), len(context))

        try:
            t_llm = time.time()
            raw_response = _call_ollama(
                prompt=user_prompt,
                system=SYSTEM_PROMPT,
                temperature=0.2,
                base_url=self.base_url,
                model=self.model,
                json_mode=True,
            )
            llm_time = time.time() - t_llm
            log.info("Ollama responded in %.1fs (%d chars)", llm_time, len(raw_response))
        except http_requests.exceptions.ConnectionError:
            log.error("Cannot reach Ollama at %s", self.base_url)
            return _fallback_response(question, f"Ollama not reachable at {self.base_url}")
        except Exception as e:
            log.error("Ollama error: %s", e)
            return _fallback_response(question, str(e))

        # ── 5. Parse & validate JSON ─────────────────────────────────────
        parsed = _extract_json(raw_response)
        if parsed is None:
            log.warning("Model returned non-JSON. Raw: %s", raw_response[:500])
            result = _fallback_response(question, "Response was not valid JSON")
            # Try to salvage the explanation from the raw text
            if len(raw_response.strip()) > 20:
                result["explanation"] = raw_response.strip()[:1000]
            result["_meta"]["raw_response"] = raw_response[:2000]
            return result

        result = _validate_response(parsed)

        # ── 6. Backfill empty citation quotes ────────────────────────────
        result["citations"] = _backfill_citations(result["citations"], chunks, question)

        # ── 7. Attach metadata ───────────────────────────────────────────
        result["_meta"] = {
            "question": question,
            "filename_filter": filename_filter,
            "model": self.model,
            "chunks_retrieved": len(chunks),
            "query_variants": retrieval.get("query_variants", []),
            "llm_time_s": round(llm_time, 2),
            "total_time_s": round(time.time() - t0, 2),
        }

        log.info(
            "Result: covered=%s  confidence=%.2f  citations=%d  caveats=%d",
            result["covered"],
            result["confidence"],
            len(result["citations"]),
            len(result["caveats"]),
        )
        return result


# ─── Pretty Printer ──────────────────────────────────────────────────────────


def _print_result(question: str, result: dict) -> None:
    """Print a formatted result to the console."""
    print("\n" + "=" * 72)
    print(f"  Q: {question}")
    print("=" * 72)
    print(f"  Covered:    {result.get('covered', '?')}")
    print(f"  Confidence: {result.get('confidence', 0):.0%}")
    print(f"  Explanation: {result.get('explanation', '—')}")

    citations = result.get("citations", [])
    if citations:
        print(f"\n  Citations ({len(citations)}):")
        for i, c in enumerate(citations, 1):
            print(f"    [{i}] {c.get('file', '?')} — p.{c.get('page', '?')}, "
                  f"  {c.get('section', '?')}")
            quote = c.get("quote", "")
            if quote:
                if len(quote) > 120:
                    quote = quote[:120] + "…"
                print(f"        \u201c{quote}\u201d")

    caveats = result.get("caveats", [])
    if caveats:
        print(f"\n  Caveats ({len(caveats)}):")
        for cv in caveats:
            print(f"    \u2022 {cv}")

    meta = result.get("_meta", {})
    if meta:
        print(f"\n  LLM: {meta.get('llm_time_s', '?')}s  |  "
              f"Total: {meta.get('total_time_s', '?')}s  |  "
              f"Chunks: {meta.get('chunks_retrieved', '?')}")
    print()


# ─── Self-Test ────────────────────────────────────────────────────────────────


def _self_test():
    """Validate helpers without needing a running Ollama server."""
    log.info("Running agent self-test …")

    # JSON extraction
    raw1 = '```json\n{"covered": "Yes", "confidence": 0.9, "explanation": "test", "citations": [], "caveats": []}\n```'
    parsed1 = _extract_json(raw1)
    assert parsed1 is not None
    assert parsed1["covered"] == "Yes"
    log.info("  JSON extraction (markdown fence) … OK")

    raw2 = 'Here is the answer: {"covered": "No", "confidence": 0.5, "explanation": "x", "citations": [], "caveats": []}'
    parsed2 = _extract_json(raw2)
    assert parsed2 is not None
    assert parsed2["covered"] == "No"
    log.info("  JSON extraction (leading text) … OK")

    assert _extract_json("This is not JSON at all") is None
    log.info("  JSON extraction (non-JSON) … OK")

    # Validation — correct schema
    valid = _validate_response({
        "covered": "Partial",
        "confidence": 0.85,
        "explanation": "Covered with conditions.",
        "citations": [{"file": "a.pdf", "page": 3, "section": "S1", "quote": "text"}],
        "caveats": ["30-day wait"],
    })
    assert valid["covered"] == "Partial"
    assert valid["confidence"] == 0.85
    assert len(valid["citations"]) == 1
    log.info("  Validation (correct schema) … OK")

    # Validation — llama3 key variants
    mapped = _validate_response({
        "answer": "Yes",
        "confidence": "0.7",
        "explanation": "ok",
        "citations": [{"filename": "b.pdf", "page_number": 5, "section_title": "X", "quote": "q"}],
        "caveats": ["wait"],
    })
    assert mapped["covered"] == "Yes"
    assert mapped["confidence"] == 0.7
    assert mapped["citations"][0]["file"] == "b.pdf"
    log.info("  Validation (key mapping) … OK")

    # Validation — nested {type, value} wrappers
    nested = _validate_response({
        "covered": {"type": "string", "value": "Yes"},
        "confidence": {"type": "number", "value": 0.8},
        "explanation": {"type": "string", "value": "Covered"},
        "citations": [],
        "caveats": [],
    })
    assert nested["covered"] == "Yes"
    assert nested["confidence"] == 0.8
    log.info("  Validation (nested wrappers) … OK")

    # Fallback
    fb = _fallback_response("test?", "parse error")
    assert fb["covered"] == "Unknown"
    assert fb["_meta"]["fallback"] is True
    log.info("  Fallback response … OK")

    # Best sentence
    text = "The policy covers hospitalization. Knee replacement is included under surgical benefits. Dental is excluded."
    best = _best_sentence(text, "Is knee replacement covered?")
    assert "knee" in best.lower() or "replacement" in best.lower()
    log.info("  Best sentence scoring … OK")

    log.info("Agent self-test PASSED.")


# ─── Sample Questions ────────────────────────────────────────────────────────

SAMPLE_QUESTIONS = [
    "Is knee replacement surgery covered under the policy?",
    "What is the waiting period for pre-existing diseases like diabetes?",
    "Does the policy cover maternity and newborn baby expenses?",
    "Are dental treatments covered?",
    "Is there coverage for mental health or psychiatric treatment?",
]


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if "--test" in sys.argv:
        _self_test()
        sys.exit(0)

    # Custom single question
    custom_q = None
    for arg in sys.argv[1:]:
        if not arg.startswith("--"):
            custom_q = arg
            break

    # Initialise agent
    agent = CoverageAgent()

    questions = [custom_q] if custom_q else SAMPLE_QUESTIONS

    print(f"\n{'─' * 72}")
    print(f"  Coverage Agent — Ollama {OLLAMA_MODEL}")
    print(f"  Running {len(questions)} question(s)")
    print(f"{'─' * 72}")

    for q in questions:
        result = agent.ask(q)
        _print_result(q, result)
