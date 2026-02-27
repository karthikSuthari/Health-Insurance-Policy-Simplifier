"""
Phase 4: Answer Generation Chain
====================================
Takes a user question, retrieves relevant policy chunks via the multi-query
retriever, then sends them to Ollama (llama3) to produce a structured JSON answer:

    {
        "answer":      "Yes" | "No" | "Partial",
        "explanation": str,
        "confidence":  float (0-1),
        "citations":   [ { "filename", "page", "section", "quote" }, … ],
        "caveats":     [ str, … ]
    }

Requirements:
    pip install chromadb sentence-transformers requests

Usage:
    python answer_chain.py "Is knee replacement surgery covered?"
    python answer_chain.py --test
"""

import json
import os
import sys
import time
import logging

import requests as http_requests

from embeddings import EmbeddingStore
from retriever import MultiQueryRetriever, _call_ollama

# ─── Configuration ────────────────────────────────────────────────────────────

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")
MAX_CONTEXT_CHARS = 12_000          # max chars of chunk context sent to LLM
FINAL_TOP_K = 5                      # chunks passed to LLM

# ─── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("answer_chain")

# ─── System Prompt ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a health insurance policy analyst. The user asks a specific question \
and provides policy excerpts. You must answer THAT EXACT QUESTION.

STEPS:
1. Read the user's question carefully.
2. Search the excerpts for text that directly relates to the question.
3. Determine: is the thing the user asked about covered, excluded, or partially covered?
4. Write a 2-4 sentence explanation that directly answers the question.
5. Quote the specific text from excerpts that supports your answer.
6. List any conditions, waiting periods, or sub-limits as caveats.

RULES:
- "answer": "Yes" if clearly covered. "No" if excluded or not mentioned. "Partial" only if covered WITH conditions.
- "confidence": 0.9+ if excerpts clearly answer the question. 0.5-0.8 if somewhat relevant. Below 0.5 if unsure.
- Do NOT list general policy exclusions. ONLY mention what is relevant to the user's question.
- The explanation MUST mention what the user asked about (e.g. if they ask about knee surgery, talk about knee surgery).
- CITATIONS ARE REQUIRED. Each citation MUST include a "quote" with the EXACT sentence or phrase copied from the excerpt. Do NOT leave the quote empty. Copy at least one full sentence from the excerpt.

Respond with ONLY this JSON:
{"answer": "Yes", "explanation": "Knee replacement surgery is covered as an inpatient surgical procedure under Section 4.", "confidence": 0.92, "citations": [{"filename": "policy.pdf", "page": 5, "section": "Benefits", "quote": "All daycare and inpatient surgical procedures including joint replacement are covered up to the sum insured"}], "caveats": ["48-month waiting period for joint replacements"]}
"""

USER_TEMPLATE = """\
QUESTION: {question}

Below are the most relevant excerpts from various health insurance policy \
documents. Each excerpt is tagged with its source file, page number, and \
section.

{context}

Remember: Your explanation must directly answer "{question}". Do not discuss unrelated exclusions.
Respond with ONLY the JSON: {{"answer": ..., "explanation": ..., "confidence": ..., "citations": [...], "caveats": [...]}}
"""


# ─── Context Builder ─────────────────────────────────────────────────────────


def build_context(results: list[dict], max_chars: int = MAX_CONTEXT_CHARS) -> str:
    """
    Format retrieved chunks into a numbered context string for the LLM prompt.
    Truncates if total exceeds *max_chars*.
    """
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


# ─── Answer Chain ─────────────────────────────────────────────────────────────


class AnswerChain:
    """
    End-to-end pipeline:  question → retrieve → generate answer.
    """

    def __init__(
        self,
        model: str = OLLAMA_MODEL,
        retriever: MultiQueryRetriever | None = None,
        **kwargs,
    ):
        self.model = model
        self.retriever = retriever or MultiQueryRetriever()

    # ── Core ────────────────────────────────────────────────────────────

    def answer(
        self,
        question: str,
        top_k: int = FINAL_TOP_K,
    ) -> dict:
        """
        Full pipeline: retrieve chunks → build prompt → call Claude → parse JSON.

        Returns dict with keys:
            answer, explanation, confidence, citations, caveats,
            _meta: { question, retrieval_time, generation_time, total_time,
                     chunks_used, model }
        """
        t_start = time.time()

        # 1. Retrieve
        retrieval = self.retriever.retrieve(question, final_top_k=top_k)
        chunks = retrieval["results"]
        t_retrieval = time.time() - t_start
        log.info("Retrieved %d chunks in %.2fs", len(chunks), t_retrieval)

        if not chunks:
            return self._empty_answer(question, "No relevant policy excerpts found.")

        # 2. Build prompt
        context = build_context(chunks)
        user_msg = USER_TEMPLATE.format(question=question, context=context)

        # 3. Call Ollama (llama3)
        t_gen_start = time.time()
        try:
            raw = _call_ollama(
                prompt=user_msg,
                system=SYSTEM_PROMPT,
                temperature=0.2,
                model=self.model,
                json_mode=True,
            )
        except Exception as e:
            log.error("Ollama generation failed: %s", e)
            return self._empty_answer(question, f"LLM call failed: {e}")

        t_generation = time.time() - t_gen_start
        t_total = time.time() - t_start
        log.info("Generated answer in %.2fs  (total: %.2fs)", t_generation, t_total)

        # 4. Parse JSON
        parsed = self._parse_response(raw)

        # 5. Attach metadata (before backfill so question is available)
        parsed["_meta"] = {
            "question": question,
            "retrieval_time_s": round(t_retrieval, 2),
            "generation_time_s": round(t_generation, 2),
            "total_time_s": round(t_total, 2),
            "chunks_used": len(chunks),
            "model": self.model,
        }

        # 6. Backfill empty citation quotes with relevant text from chunks
        parsed = self._backfill_citations(parsed, chunks)

        return parsed

    # ── Helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def _backfill_citations(parsed: dict, chunks: list[dict]) -> dict:
        """
        Ensure every citation has a non-empty, relevant quote.
        - If the LLM left a quote empty, find the best matching sentence
          from the retrieved chunk.
        - If no citations at all, create them from the top chunks,
          highlighting the most relevant sentences to the question.
        """
        import re as _re

        question = parsed.get("_meta", {}).get("question", "") or ""
        question_words = set(question.lower().split())
        citations = parsed.get("citations", [])

        # Build lookup: (filename, page_str) -> chunk
        chunk_lookup: dict[tuple, dict] = {}
        for c in chunks:
            meta = c.get("metadata", {})
            key = (meta.get("filename", ""), str(meta.get("page_number", "")))
            if key not in chunk_lookup:
                chunk_lookup[key] = c

        def _best_sentences(text: str, n: int = 3) -> str:
            """Pick the most question-relevant sentences from a chunk."""
            # Split on sentence boundaries
            sentences = _re.split(r'(?<=[.!?])\s+', text.strip())
            if not sentences:
                return text[:400]
            # Score each sentence by overlap with question words
            scored = []
            for s in sentences:
                s_clean = s.strip()
                if len(s_clean) < 10:
                    continue
                words = set(s_clean.lower().split())
                overlap = len(question_words & words)
                scored.append((overlap, s_clean))
            scored.sort(key=lambda x: x[0], reverse=True)
            # Take top n sentences, preserving original order
            top = [s for _, s in scored[:n]]
            ordered = [s for s in sentences if s.strip() in top]
            if not ordered:
                ordered = [s.strip() for s in sentences[:n]]
            result = " ".join(ordered)
            if len(result) > 500:
                result = result[:500] + "…"
            return result

        # 1. Backfill empty quotes in existing citations
        for cit in citations:
            if not cit.get("quote") or len(str(cit.get("quote", ""))) < 5:
                key = (cit.get("filename", ""), str(cit.get("page", "")))
                chunk = chunk_lookup.get(key)
                if chunk:
                    cit["quote"] = _best_sentences(chunk.get("text", ""))
                else:
                    # Try loose match by filename only
                    for (fn, _), ch in chunk_lookup.items():
                        if fn == cit.get("filename", ""):
                            cit["quote"] = _best_sentences(ch.get("text", ""))
                            break
                # Last resort: use any top chunk
                if not cit.get("quote") and chunks:
                    cit["quote"] = _best_sentences(chunks[0].get("text", ""))

        # 2. If still no citations, create from top chunks
        if not citations and chunks:
            for c in chunks[:3]:
                meta = c.get("metadata", {})
                citations.append({
                    "filename": meta.get("filename", ""),
                    "page": meta.get("page_number", ""),
                    "section": meta.get("section_title", ""),
                    "quote": _best_sentences(c.get("text", "")),
                })

        # 3. Final pass: remove any citation that still has no quote
        citations = [c for c in citations if c.get("quote")]

        parsed["citations"] = citations
        return parsed

    @staticmethod
    def _parse_response(raw: str) -> dict:
        """Parse LLM's JSON response, with lenient fallback."""
        import re

        cleaned = raw.strip()

        # Strategy 1: strip markdown fences
        if "```" in cleaned:
            # Extract content between first ``` and last ```
            m = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", cleaned, re.DOTALL)
            if m:
                cleaned = m.group(1).strip()

        # Strategy 2: extract first { ... } block if there's surrounding text
        if not cleaned.startswith("{"):
            start = cleaned.find("{")
            if start != -1:
                cleaned = cleaned[start:]
        if not cleaned.endswith("}"):
            end = cleaned.rfind("}")
            if end != -1:
                cleaned = cleaned[: end + 1]

        cleaned = cleaned.strip()

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            log.warning("Failed to parse LLM response as JSON. Returning raw text.")
            data = {
                "answer": "Unknown",
                "explanation": cleaned[:1000],
                "confidence": 0.0,
                "citations": [],
                "caveats": ["LLM response was not valid JSON"],
            }

        # --- Post-process: normalise llama3 non-conformant output ---
        data = AnswerChain._normalise_llm_output(data)

        # Ensure required keys exist
        for key in ("answer", "explanation", "confidence", "citations", "caveats"):
            data.setdefault(key, [] if key in ("citations", "caveats") else "")

        return data

    @staticmethod
    def _flatten_value(v):
        """Recursively flatten nested {type, value} structures llama3 produces."""
        if isinstance(v, str):
            return v
        if isinstance(v, (int, float)):
            return v
        if isinstance(v, dict):
            # Handle {type: "string", value: "..."} pattern
            if "value" in v:
                return AnswerChain._flatten_value(v["value"])
            # Handle {type: "object", value: {…}} pattern
            if "type" in v and len(v) <= 2:
                for k2 in v:
                    if k2 != "type":
                        return AnswerChain._flatten_value(v[k2])
            return v
        if isinstance(v, list):
            return [AnswerChain._flatten_value(item) for item in v]
        return v

    @staticmethod
    def _normalise_llm_output(data: dict) -> dict:
        """Normalise whatever llama3 returns into the expected schema."""
        # 1. Flatten nested {type, value} wrappers on all keys
        for k in list(data.keys()):
            data[k] = AnswerChain._flatten_value(data[k])

        # 2. Fix "answer" — must be a simple string
        answer_val = data.get("answer", "")
        if isinstance(answer_val, dict):
            # Try to extract something useful
            answer_val = str(answer_val.get("value", "")) or str(answer_val)
        if isinstance(answer_val, list):
            answer_val = ", ".join(str(x) for x in answer_val)
        answer_val = str(answer_val).strip()

        # 3. Fix "confidence" — must be a float
        conf = data.get("confidence", 0)
        if isinstance(conf, str):
            try:
                conf = float(conf)
            except ValueError:
                conf = 0.5
        if not isinstance(conf, (int, float)):
            conf = 0.5
        data["confidence"] = float(conf)

        # 4. Fix "citations" — must be list of dicts with filename/page/section/quote
        citations = data.get("citations", [])
        if isinstance(citations, list):
            clean_citations = []
            for c in citations:
                c = AnswerChain._flatten_value(c)
                if isinstance(c, str):
                    clean_citations.append({"filename": "", "page": "", "section": "", "quote": c})
                elif isinstance(c, dict):
                    clean_citations.append({
                        "filename": str(c.get("filename", "") or ""),
                        "page": c.get("page", ""),
                        "section": str(c.get("section", "") or ""),
                        "quote": str(c.get("quote", "") or c.get("value", "") or ""),
                    })
            data["citations"] = clean_citations

        # 5. Fix "caveats" — must be list of plain strings
        caveats = data.get("caveats", [])
        if isinstance(caveats, str):
            # Single string instead of list
            caveats = [caveats] if caveats else []
        if isinstance(caveats, list):
            clean_caveats = []
            for c in caveats:
                c = AnswerChain._flatten_value(c)
                if isinstance(c, str) and c.strip():
                    clean_caveats.append(c.strip())
                elif isinstance(c, dict):
                    # Extract description or value from nested dict
                    text = c.get("description", "") or c.get("value", "") or c.get("desc", "")
                    if text:
                        clean_caveats.append(str(text).strip())
                elif c is not None:
                    s = str(c).strip()
                    if s and s != "None":
                        clean_caveats.append(s)
            data["caveats"] = clean_caveats
        else:
            data["caveats"] = []

        # 6. Collect text from non-standard keys
        explanation_val = data.get("explanation", "") or ""
        extra_parts = []
        for k in list(data.keys()):
            if k in ("answer", "explanation", "confidence", "citations", "caveats"):
                continue
            v = data[k]
            if isinstance(v, str) and v:
                extra_parts.append(v)
            elif isinstance(v, list):
                for item in v:
                    if isinstance(item, dict):
                        desc = item.get("description", "") or item.get("desc", "") or ""
                        if desc:
                            extra_parts.append(str(desc))
                    elif isinstance(item, str):
                        extra_parts.append(item)
            del data[k]

        if extra_parts and not explanation_val:
            data["explanation"] = ". ".join(extra_parts[:5])[:1500]
            explanation_val = data["explanation"]

        # 7. Only fix answer if it's truly empty/invalid — respect the model's choice
        valid_answers = {"yes", "no", "partial"}
        if isinstance(answer_val, str) and answer_val.lower() in valid_answers:
            # Model gave a valid answer — keep it as-is (capitalised)
            data["answer"] = answer_val.capitalize()
        elif not answer_val or (isinstance(answer_val, str) and answer_val.lower() not in valid_answers):
            # Only infer if the model didn't give a valid answer
            combined = (explanation_val or str(answer_val) or "").lower()
            if any(w in combined for w in ["covered", "eligible", "payable", "included"]):
                if any(w in combined for w in ["condition", "waiting", "sub-limit", "co-pay", "limit"]):
                    data["answer"] = "Partial"
                else:
                    data["answer"] = "Yes"
            elif any(w in combined for w in ["excluded", "not covered", "not eligible", "not mentioned"]):
                data["answer"] = "No"
            else:
                data["answer"] = "Partial"
            if not data.get("confidence"):
                data["confidence"] = 0.5
        else:
            data["answer"] = str(answer_val).strip()
        return data

    @staticmethod
    def _empty_answer(question: str, reason: str) -> dict:
        return {
            "answer": "Unknown",
            "explanation": reason,
            "confidence": 0.0,
            "citations": [],
            "caveats": [reason],
            "_meta": {
                "question": question,
                "retrieval_time_s": 0,
                "generation_time_s": 0,
                "total_time_s": 0,
                "chunks_used": 0,
                "model": OLLAMA_MODEL,
            },
        }


# ─── Pretty Print ────────────────────────────────────────────────────────────


def print_answer(result: dict) -> None:
    """Human-readable output."""
    meta = result.get("_meta", {})

    print("\n" + "═" * 70)
    print(f"  QUESTION : {meta.get('question', '?')}")
    print(f"  ANSWER   : {result['answer']}")
    print(f"  CONFIDENCE: {result['confidence']}")
    print("═" * 70)
    print(f"\n{result['explanation']}\n")

    if result["caveats"]:
        print("⚠ CAVEATS:")
        for c in result["caveats"]:
            print(f"  • {c}")
        print()

    if result["citations"]:
        print(f"CITATIONS ({len(result['citations'])}):")
        for i, cit in enumerate(result["citations"], 1):
            print(f"  [{i}] {cit.get('filename', '?')}  p.{cit.get('page', '?')}  §{cit.get('section', '?')}")
            print(f"      \"{cit.get('quote', '')}\"")
        print()

    if meta:
        print(
            f"⏱ Retrieval {meta.get('retrieval_time_s', 0)}s  |  "
            f"Generation {meta.get('generation_time_s', 0)}s  |  "
            f"Total {meta.get('total_time_s', 0)}s  |  "
            f"Chunks {meta.get('chunks_used', 0)}"
        )
    print("═" * 70)


# ─── Self-Test ────────────────────────────────────────────────────────────────


def _self_test():
    """Test context building and JSON parsing without API calls."""
    log.info("Running self-test …")

    # 1. build_context
    fake = [
        {
            "text": "Surgery is covered up to Rs 5 lakh.",
            "score": 0.9,
            "chunk_id": "c1",
            "metadata": {
                "filename": "test.pdf",
                "page_number": 1,
                "page_end": 1,
                "section_title": "Benefits",
            },
        }
    ]
    ctx = build_context(fake)
    assert "Excerpt 1" in ctx
    assert "Surgery" in ctx
    log.info("  build_context … OK")

    # 2. _parse_response — valid JSON
    good_json = '{"answer":"Yes","explanation":"Covered.","confidence":0.9,"citations":[],"caveats":[]}'
    parsed = AnswerChain._parse_response(good_json)
    assert parsed["answer"] == "Yes"
    log.info("  _parse_response (valid) … OK")

    # 3. _parse_response — markdown-wrapped JSON
    wrapped = '```json\n{"answer":"No","explanation":"Not covered.","confidence":0.8,"citations":[],"caveats":[]}\n```'
    parsed2 = AnswerChain._parse_response(wrapped)
    assert parsed2["answer"] == "No"
    log.info("  _parse_response (markdown) … OK")

    # 4. _parse_response — invalid JSON fallback
    parsed3 = AnswerChain._parse_response("This is not JSON at all")
    assert parsed3["answer"] == "Unknown"
    assert parsed3["confidence"] == 0.0
    log.info("  _parse_response (fallback) … OK")

    log.info("Self-test PASSED.")


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if "--test" in sys.argv:
        _self_test()
        sys.exit(0)

    if len(sys.argv) > 1 and not sys.argv[1].startswith("--"):
        question = " ".join(sys.argv[1:])
        chain = AnswerChain()
        result = chain.answer(question)
        print_answer(result)
        print("\n" + json.dumps(result, indent=2, ensure_ascii=False, default=str))
    else:
        print("Usage:")
        print('  python answer_chain.py "Is knee replacement surgery covered?"')
        print("  python answer_chain.py --test")
        sys.exit(1)
