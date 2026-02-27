"""
Phase 2: PDF Parser & Chunker
===============================
Extracts text page-by-page from health insurance PDFs using pdfplumber,
detects section headers via heuristics, chunks at ~800 tokens with 100-token
overlap (never splitting mid-sentence), and exports metadata-rich JSON.

Requirements:
    pip install pdfplumber tiktoken

Usage:
    # Parse all PDFs in data/policies/:
        python pdf_parser.py

    # Parse a single PDF:
        python pdf_parser.py "data/policies/some-file.pdf"

    # Run self-test on first available PDF:
        python pdf_parser.py --test
"""

import json
import os
import re
import sys
import time
import logging
from pathlib import Path
from dataclasses import dataclass, asdict, field

import pdfplumber
import tiktoken

# ─── Configuration ────────────────────────────────────────────────────────────

POLICIES_DIR = Path("./data/policies")
OUTPUT_DIR = Path("./data/chunks")
TARGET_TOKENS = 800
OVERLAP_TOKENS = 100
ENCODING_NAME = "cl100k_base"  # tokenizer used by Claude / GPT-4

# ─── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("pdf_parser")

# ─── Data Models ──────────────────────────────────────────────────────────────


@dataclass
class PageText:
    """Raw extracted text for a single PDF page."""

    page_number: int  # 1-based
    text: str
    char_count: int
    line_count: int


@dataclass
class Section:
    """A detected section header with its position."""

    title: str
    page_number: int
    line_index: int  # line offset within the page


@dataclass
class Chunk:
    """A text chunk with full provenance metadata."""

    chunk_id: str
    text: str
    token_count: int
    filename: str
    page_number: int
    page_end: int  # chunk may span pages
    section_title: str
    char_start: int  # offset in full-document text
    char_end: int


# ─── Tokenizer ────────────────────────────────────────────────────────────────

_encoder: tiktoken.Encoding | None = None


def _get_encoder() -> tiktoken.Encoding:
    global _encoder
    if _encoder is None:
        _encoder = tiktoken.get_encoding(ENCODING_NAME)
    return _encoder


def count_tokens(text: str) -> int:
    return len(_get_encoder().encode(text))


# ─── Section Header Detection ────────────────────────────────────────────────

# Pattern 1: Numbered sections  e.g. "4.1 Exclusions", "IV. Benefits"
_RE_NUMBERED = re.compile(
    r"^(?:"
    r"\d{1,3}(?:\.\d{1,3}){0,3}"          # 1, 1.2, 1.2.3
    r"|[IVXLC]+\."                          # Roman numerals
    r"|[A-Z]\."                             # A. B. C.
    r")\s+\S",
)

# Pattern 2: ALL-CAPS lines (≥4 alpha chars, ≤120 chars, >60 % uppercase)
_RE_ALLCAPS = re.compile(r"^[A-Z][A-Z \-&/,\(\):'\"]{3,119}$")

# Pattern 3: Common insurance section keywords at start of line
_SECTION_KEYWORDS = re.compile(
    r"^(?:SECTION|PART|CHAPTER|SCHEDULE|ANNEXURE|APPENDIX|TABLE OF|"
    r"DEFINITIONS?|EXCLUSIONS?|INCLUSIONS?|BENEFITS?|COVERAGE|"
    r"GENERAL\s+(?:TERMS|CONDITIONS|PROVISIONS)|"
    r"CLAIM|PREMIUM|WAITING\s+PERIOD|PRE-?EXISTING|RENEWAL|"
    r"GRIEVANCE|PORTABILITY|FREE\s+LOOK|CANCELLATION"
    r")\b",
    re.IGNORECASE,
)


def is_section_header(line: str) -> bool:
    """
    Heuristic: return True if *line* looks like a section heading.
    We require the line to be reasonably short (≤ 120 chars) and match at
    least one structural pattern.
    """
    stripped = line.strip()
    if not stripped or len(stripped) > 120:
        return False
    # Very short lines that are just numbers aren't headers
    if len(stripped) < 4:
        return False

    # Numbered heading  (e.g. "4.1 Scope of Cover")
    if _RE_NUMBERED.match(stripped):
        return True

    # ALL-CAPS line with enough alpha chars
    alpha_chars = sum(1 for c in stripped if c.isalpha())
    if alpha_chars >= 4 and stripped == stripped.upper() and _RE_ALLCAPS.match(stripped):
        return True

    # Keyword-prefixed lines
    if _SECTION_KEYWORDS.match(stripped):
        return True

    return False


def clean_header(raw: str) -> str:
    """Normalise a detected header string."""
    s = raw.strip()
    # Remove leading numbering for a cleaner title (keep original numbering too)
    return s[:100]


# ─── PDF Text Extraction ─────────────────────────────────────────────────────


def extract_pages(pdf_path: str | Path) -> list[PageText]:
    """
    Extract text from every page of a PDF using pdfplumber.

    Returns list of PageText, one per page.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    pages: list[PageText] = []
    try:
        with pdfplumber.open(str(pdf_path)) as pdf:
            log.info("Opened '%s'  (%d pages)", pdf_path.name, len(pdf.pages))
            for i, page in enumerate(pdf.pages, 1):
                text = page.extract_text() or ""
                # pdfplumber sometimes returns None for scanned/image pages
                pages.append(
                    PageText(
                        page_number=i,
                        text=text,
                        char_count=len(text),
                        line_count=text.count("\n") + 1 if text else 0,
                    )
                )
    except Exception as e:
        log.error("Failed to open/read PDF '%s': %s", pdf_path.name, e)
        raise

    total_chars = sum(p.char_count for p in pages)
    total_lines = sum(p.line_count for p in pages)
    empty = sum(1 for p in pages if p.char_count == 0)
    log.info(
        "  Extracted %d pages  |  %d chars  |  %d lines  |  %d empty pages",
        len(pages), total_chars, total_lines, empty,
    )
    return pages


# ─── Section Detection (across all pages) ────────────────────────────────────


def detect_sections(pages: list[PageText]) -> list[Section]:
    """
    Walk through every line of every page and detect section headers.
    Returns a list of Section objects in document order.
    """
    sections: list[Section] = []
    for page in pages:
        for line_idx, line in enumerate(page.text.split("\n")):
            if is_section_header(line):
                sections.append(
                    Section(
                        title=clean_header(line),
                        page_number=page.page_number,
                        line_index=line_idx,
                    )
                )
    log.info("  Detected %d section headers", len(sections))
    return sections


# ─── Sentence-Aware Chunking ─────────────────────────────────────────────────

# Simple but robust sentence boundary regex
_SENT_BOUNDARY = re.compile(
    r"(?<=[.!?;])\s+(?=[A-Z0-9(])"   # period/!/? + space + capital letter
    r"|(?<=\n)\s*(?=\S)"              # newline boundary (paragraph break)
)


def _split_sentences(text: str) -> list[str]:
    """
    Split text into sentence-ish segments.
    Falls back to splitting on newlines if regex produces nothing useful.
    """
    parts = _SENT_BOUNDARY.split(text)
    # Filter out empty parts
    parts = [p for p in parts if p.strip()]
    if not parts:
        # fallback: split on newlines
        parts = [p for p in text.split("\n") if p.strip()]
    return parts if parts else [text]


def _build_page_map(pages: list[PageText]) -> list[tuple[int, int]]:
    """
    Build a list of (page_number, char_offset) tuples so we can map any
    character offset in the concatenated document back to a page number.
    """
    mapping: list[tuple[int, int]] = []
    offset = 0
    for page in pages:
        mapping.append((page.page_number, offset))
        offset += page.char_count + 1  # +1 for the \n we insert between pages
    return mapping


def _page_for_offset(page_map: list[tuple[int, int]], offset: int) -> int:
    """Return the 1-based page number that contains *offset*."""
    result = page_map[0][0]
    for page_num, start in page_map:
        if start > offset:
            break
        result = page_num
    return result


def _section_for_offset(
    sections: list[Section],
    pages: list[PageText],
    page_map: list[tuple[int, int]],
    offset: int,
) -> str:
    """
    Return the section title active at *offset*.
    Walks sections in reverse to find the last header before this offset.
    """
    if not sections:
        return "Unknown"

    # Convert each section to a char-offset in the combined doc
    section_offsets: list[tuple[int, str]] = []
    for sec in sections:
        # Find the char offset where this section starts
        page_start_offset = 0
        for pnum, poff in page_map:
            if pnum == sec.page_number:
                page_start_offset = poff
                break
        # Approximate: line_index * avg ~60 chars per line
        page_text = ""
        for p in pages:
            if p.page_number == sec.page_number:
                page_text = p.text
                break
        lines = page_text.split("\n")
        line_offset = sum(len(lines[i]) + 1 for i in range(min(sec.line_index, len(lines))))
        section_offsets.append((page_start_offset + line_offset, sec.title))

    # Find the last section whose offset ≤ our offset
    current_title = sections[0].title
    for sec_off, sec_title in section_offsets:
        if sec_off > offset:
            break
        current_title = sec_title
    return current_title


def chunk_document(
    pages: list[PageText],
    sections: list[Section],
    filename: str,
    target_tokens: int = TARGET_TOKENS,
    overlap_tokens: int = OVERLAP_TOKENS,
) -> list[Chunk]:
    """
    Chunk the full document into pieces of ~target_tokens with overlap.
    Never splits mid-sentence.
    """
    # Concatenate all pages into one document (preserve page boundaries)
    full_text = "\n".join(p.text for p in pages)
    if not full_text.strip():
        log.warning("  Document has no extractable text!")
        return []

    page_map = _build_page_map(pages)
    sentences = _split_sentences(full_text)

    chunks: list[Chunk] = []
    enc = _get_encoder()

    # Pre-compute token counts for each sentence
    sent_tokens = [len(enc.encode(s)) for s in sentences]

    i = 0
    chunk_idx = 0

    while i < len(sentences):
        # Accumulate sentences until we hit target_tokens
        chunk_sents: list[str] = []
        chunk_tok_count = 0
        start_sent_idx = i

        while i < len(sentences) and chunk_tok_count + sent_tokens[i] <= target_tokens:
            chunk_sents.append(sentences[i])
            chunk_tok_count += sent_tokens[i]
            i += 1

        # If a single sentence exceeds target_tokens, include it anyway
        # (never leave a sentence behind)
        if not chunk_sents and i < len(sentences):
            chunk_sents.append(sentences[i])
            chunk_tok_count = sent_tokens[i]
            i += 1

        if not chunk_sents:
            break

        chunk_text = " ".join(chunk_sents)
        char_start = full_text.find(chunk_sents[0])
        # More robust: track cumulative position
        char_start_approx = sum(len(sentences[j]) + 1 for j in range(start_sent_idx))
        char_end_approx = char_start_approx + len(chunk_text)

        page_start = _page_for_offset(page_map, char_start_approx)
        page_end = _page_for_offset(page_map, char_end_approx)
        section = _section_for_offset(sections, pages, page_map, char_start_approx)

        chunk_idx += 1
        chunks.append(
            Chunk(
                chunk_id=f"{Path(filename).stem}__chunk_{chunk_idx:04d}",
                text=chunk_text,
                token_count=chunk_tok_count,
                filename=filename,
                page_number=page_start,
                page_end=page_end,
                section_title=section,
                char_start=char_start_approx,
                char_end=char_end_approx,
            )
        )

        # --- overlap: rewind so the next chunk starts ~overlap_tokens before end ---
        if i < len(sentences):
            rewind_tokens = 0
            rewind_count = 0
            j = i - 1
            while j >= start_sent_idx and rewind_tokens < overlap_tokens:
                rewind_tokens += sent_tokens[j]
                rewind_count += 1
                j -= 1
            # Don't rewind past the start of current chunk
            i = max(i - rewind_count, start_sent_idx + 1)

    log.info(
        "  Chunked into %d pieces  |  avg %.0f tokens/chunk  |  range [%d – %d]",
        len(chunks),
        sum(c.token_count for c in chunks) / max(len(chunks), 1),
        min((c.token_count for c in chunks), default=0),
        max((c.token_count for c in chunks), default=0),
    )
    return chunks


# ─── Single PDF Pipeline ─────────────────────────────────────────────────────


def parse_pdf(pdf_path: str | Path) -> list[Chunk]:
    """
    Full pipeline for one PDF: extract → detect sections → chunk.
    """
    pdf_path = Path(pdf_path)
    log.info("=" * 60)
    log.info("PARSING: %s", pdf_path.name)

    t0 = time.time()
    pages = extract_pages(pdf_path)
    sections = detect_sections(pages)
    chunks = chunk_document(pages, sections, filename=pdf_path.name)
    elapsed = time.time() - t0

    log.info("  Done in %.1fs  |  %d chunks", elapsed, len(chunks))
    return chunks


# ─── Batch Pipeline ──────────────────────────────────────────────────────────


def parse_all_pdfs(
    input_dir: Path = POLICIES_DIR,
    output_dir: Path = OUTPUT_DIR,
) -> dict:
    """
    Parse every PDF in *input_dir*, write per-file and combined JSON to *output_dir*.

    Returns:
        Summary dict.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(
        [f for f in input_dir.iterdir() if f.suffix.lower() == ".pdf"],
        key=lambda p: p.name.lower(),
    )
    log.info("Found %d PDF file(s) in %s", len(pdf_files), input_dir)

    all_chunks: list[dict] = []
    file_stats: list[dict] = []
    errors: list[dict] = []

    for idx, pdf_path in enumerate(pdf_files, 1):
        log.info("[%d/%d] %s", idx, len(pdf_files), pdf_path.name)
        try:
            chunks = parse_pdf(pdf_path)
            chunk_dicts = [asdict(c) for c in chunks]
            all_chunks.extend(chunk_dicts)

            # Per-file JSON
            per_file = output_dir / f"{pdf_path.stem}_chunks.json"
            per_file.write_text(json.dumps(chunk_dicts, indent=2, ensure_ascii=False), encoding="utf-8")

            file_stats.append({
                "file": pdf_path.name,
                "pages": 0,  # filled below
                "chunks": len(chunks),
                "total_tokens": sum(c.token_count for c in chunks),
                "status": "ok",
            })
            # Patch page count (re-open is cheap, but let's derive from chunks)
            if chunks:
                file_stats[-1]["pages"] = max(c.page_end for c in chunks)

        except Exception as e:
            log.error("FAILED on %s: %s", pdf_path.name, e)
            errors.append({"file": pdf_path.name, "error": str(e)})
            file_stats.append({
                "file": pdf_path.name,
                "pages": 0,
                "chunks": 0,
                "total_tokens": 0,
                "status": f"error: {e}",
            })

    # Write combined chunks
    combined_path = output_dir / "all_chunks.json"
    combined_path.write_text(json.dumps(all_chunks, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("Combined JSON saved: %s  (%d chunks)", combined_path, len(all_chunks))

    summary = {
        "total_files": len(pdf_files),
        "successful": len(pdf_files) - len(errors),
        "failed": len(errors),
        "total_chunks": len(all_chunks),
        "total_tokens": sum(c.get("token_count", 0) for c in all_chunks),
        "avg_tokens_per_chunk": (
            round(sum(c.get("token_count", 0) for c in all_chunks) / max(len(all_chunks), 1))
        ),
        "files": file_stats,
        "errors": errors,
    }

    # Save summary
    summary_path = output_dir / "_parse_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("Summary saved: %s", summary_path)

    return summary


# ─── Quick Test ───────────────────────────────────────────────────────────────


def _quick_test():
    """Parse the first available PDF and print detailed stats."""
    log.info("=" * 60)
    log.info("QUICK TEST — parsing first PDF found")

    pdf_files = sorted(
        [f for f in POLICIES_DIR.iterdir() if f.suffix.lower() == ".pdf"],
        key=lambda p: p.stat().st_size,  # start with smallest for speed
    )
    if not pdf_files:
        log.error("No PDFs found in %s", POLICIES_DIR)
        sys.exit(1)

    test_pdf = pdf_files[0]
    log.info("Test file: %s  (%d bytes)", test_pdf.name, test_pdf.stat().st_size)

    chunks = parse_pdf(test_pdf)

    # ── Print stats ──
    print("\n" + "─" * 60)
    print(f"FILE:           {test_pdf.name}")
    print(f"CHUNKS:         {len(chunks)}")
    if chunks:
        tokens = [c.token_count for c in chunks]
        pages_seen = sorted(set(c.page_number for c in chunks))
        sections_seen = sorted(set(c.section_title for c in chunks))
        print(f"TOKEN RANGE:    {min(tokens)} – {max(tokens)}  (target: {TARGET_TOKENS})")
        print(f"AVG TOKENS:     {sum(tokens) / len(tokens):.0f}")
        print(f"TOTAL TOKENS:   {sum(tokens)}")
        print(f"PAGES COVERED:  {pages_seen[0]} – {pages_seen[-1]}")
        print(f"SECTIONS ({len(sections_seen)}):")
        for s in sections_seen[:15]:
            print(f"  • {s}")
        if len(sections_seen) > 15:
            print(f"  … and {len(sections_seen) - 15} more")

        # Sample chunk
        mid = len(chunks) // 2
        sample = chunks[mid]
        print(f"\n── SAMPLE CHUNK (#{mid + 1}) ──")
        print(f"  ID:      {sample.chunk_id}")
        print(f"  Tokens:  {sample.token_count}")
        print(f"  Page:    {sample.page_number}–{sample.page_end}")
        print(f"  Section: {sample.section_title}")
        print(f"  Text:    {sample.text[:300]}{'…' if len(sample.text) > 300 else ''}")
    print("─" * 60)

    # Save test output
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / f"_test_{test_pdf.stem}_chunks.json"
    out.write_text(json.dumps([asdict(c) for c in chunks], indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("Test chunks saved: %s", out)


# ─── Self-Test (no PDFs needed) ──────────────────────────────────────────────


def _unit_test():
    """Run offline unit tests on heuristics and chunking."""
    log.info("Running offline unit tests …")

    # ── Section header detection ─────────────────────────────────────────
    assert is_section_header("DEFINITIONS") is True
    assert is_section_header("SECTION 4: EXCLUSIONS") is True
    assert is_section_header("4.1 Scope of Cover") is True
    assert is_section_header("IV. Benefits") is True
    assert is_section_header("A. General Conditions") is True
    assert is_section_header("GENERAL TERMS AND CONDITIONS") is True
    assert is_section_header("PRE-EXISTING DISEASES") is True
    assert is_section_header("") is False
    assert is_section_header("a") is False
    assert is_section_header("This is a normal paragraph about coverage.") is False
    # Long line should not be a header
    assert is_section_header("A" * 130) is False
    log.info("  is_section_header … OK")

    # ── Sentence splitting ───────────────────────────────────────────────
    text = "First sentence. Second sentence. Third one here."
    sents = _split_sentences(text)
    assert len(sents) >= 2, f"Expected ≥2 sentences, got {len(sents)}"
    log.info("  _split_sentences … OK")

    # ── Token counting ───────────────────────────────────────────────────
    n = count_tokens("Hello world, this is a test.")
    assert 3 < n < 20, f"Unexpected token count: {n}"
    log.info("  count_tokens … OK")

    # ── Chunking with fake pages ─────────────────────────────────────────
    fake_pages = [
        PageText(page_number=1, text="DEFINITIONS\n" + ("word " * 500), char_count=2500, line_count=10),
        PageText(page_number=2, text="EXCLUSIONS\n" + ("text " * 500), char_count=2500, line_count=10),
    ]
    fake_sections = detect_sections(fake_pages)
    chunks = chunk_document(fake_pages, fake_sections, "test.pdf")
    assert len(chunks) > 0, "Should produce at least one chunk"
    for c in chunks:
        assert c.token_count <= TARGET_TOKENS + 100, f"Chunk too large: {c.token_count} tokens"
        assert c.filename == "test.pdf"
        assert c.page_number >= 1
        assert c.section_title
    log.info("  chunk_document … OK  (%d chunks from fake data)", len(chunks))

    log.info("All unit tests PASSED.")


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if "--unit" in sys.argv:
        _unit_test()
        sys.exit(0)

    if "--test" in sys.argv:
        _unit_test()
        _quick_test()
        sys.exit(0)

    if len(sys.argv) > 1 and not sys.argv[1].startswith("--"):
        # Single file mode
        path = Path(sys.argv[1])
        chunks = parse_pdf(path)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        out = OUTPUT_DIR / f"{path.stem}_chunks.json"
        out.write_text(json.dumps([asdict(c) for c in chunks], indent=2, ensure_ascii=False), encoding="utf-8")
        log.info("Saved %d chunks → %s", len(chunks), out)
    else:
        # Batch mode — all PDFs
        summary = parse_all_pdfs()
        print(json.dumps(summary, indent=2))
