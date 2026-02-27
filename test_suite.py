"""
Demo Validation Test Suite
============================
Runs 10 predefined coverage questions against the CoverageAgent and reports:
  1. Each Q&A with citation count
  2. Flags responses with 0 citations (retrieval failure)
  3. Flags responses where JSON parsing failed
  4. Summary: pass rate, average response time, average citations per answer

Usage:
    python test_suite.py
"""

import time
import sys
import os

# ─── Configuration ────────────────────────────────────────────────────────────

# Ensure HuggingFace xet transport doesn't interfere
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

# ─── Test Questions ───────────────────────────────────────────────────────────

TEST_QUESTIONS = [
    "Is knee replacement surgery covered?",
    "What is the waiting period for pre-existing diseases?",
    "Are maternity expenses covered?",
    "Is dental treatment covered under this policy?",
    "What are the exclusions for room rent?",
    "Is AYUSH treatment covered?",
    "What is the co-payment clause?",
    "Is cataract surgery covered?",
    "Are mental health treatments covered?",
    "Is organ transplant surgery covered?",
]

# ─── ANSI Colours ─────────────────────────────────────────────────────────────

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"

# ─── Helpers ──────────────────────────────────────────────────────────────────


def _status_icon(ok: bool) -> str:
    return f"{GREEN}PASS{RESET}" if ok else f"{RED}FAIL{RESET}"


def _divider(char: str = "─", width: int = 78) -> str:
    return char * width


# ─── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    print(f"\n{BOLD}{'=' * 78}{RESET}")
    print(f"{BOLD}  Health Insurance Policy Simplifier — Demo Validation Suite{RESET}")
    print(f"{BOLD}{'=' * 78}{RESET}\n")

    # ── Initialise agent ──────────────────────────────────────────────────
    print(f"{DIM}Initialising CoverageAgent …{RESET}")
    t_init = time.time()

    try:
        from agent import CoverageAgent
    except ImportError as e:
        print(f"{RED}ERROR: Cannot import CoverageAgent — {e}{RESET}")
        sys.exit(1)

    try:
        agent = CoverageAgent()
    except Exception as e:
        print(f"{RED}ERROR: Failed to create CoverageAgent — {e}{RESET}")
        sys.exit(1)

    init_time = time.time() - t_init
    print(f"{GREEN}Agent ready in {init_time:.1f}s{RESET}\n")

    # ── Run tests ─────────────────────────────────────────────────────────
    results: list[dict] = []
    total = len(TEST_QUESTIONS)

    for i, question in enumerate(TEST_QUESTIONS, 1):
        print(f"{_divider()}")
        print(f"{BOLD}[{i}/{total}]{RESET} {question}")
        print(f"{_divider()}")

        t0 = time.time()
        try:
            answer = agent.ask(question)
            elapsed = time.time() - t0
            json_ok = True
        except Exception as e:
            elapsed = time.time() - t0
            answer = {
                "covered": "ERROR",
                "confidence": 0.0,
                "explanation": str(e),
                "citations": [],
                "caveats": [],
            }
            json_ok = False

        # Extract fields
        covered = answer.get("covered", "Unknown")
        confidence = answer.get("confidence", 0.0)
        explanation = answer.get("explanation", "")
        citations = answer.get("citations", [])
        caveats = answer.get("caveats", [])
        meta = answer.get("_meta", {})

        # Detect JSON parse failure (fallback responses have _meta.error)
        if meta.get("error"):
            json_ok = False

        citation_count = len(citations)
        retrieval_ok = citation_count > 0

        # Determine pass/fail
        passed = json_ok and retrieval_ok

        # Print result
        print(f"  Answer:      {BOLD}{covered}{RESET}")
        print(f"  Confidence:  {confidence:.0%}")
        print(f"  Explanation: {explanation[:120]}{'…' if len(explanation) > 120 else ''}")
        print(f"  Citations:   {citation_count}", end="")
        if not retrieval_ok:
            print(f"  {RED}⚠ RETRIEVAL FAILURE (0 citations){RESET}")
        else:
            print()
        if not json_ok:
            print(f"  {RED}⚠ JSON PARSE FAILURE{RESET}")
        if caveats:
            print(f"  Caveats:     {len(caveats)} — {', '.join(str(c)[:50] for c in caveats[:3])}")
        print(f"  Time:        {elapsed:.1f}s")
        print(f"  Status:      {_status_icon(passed)}")
        print()

        results.append({
            "question": question,
            "covered": covered,
            "confidence": confidence,
            "citations": citation_count,
            "caveats": len(caveats),
            "json_ok": json_ok,
            "retrieval_ok": retrieval_ok,
            "passed": passed,
            "time_s": elapsed,
        })

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{BOLD}{'=' * 78}{RESET}")
    print(f"{BOLD}  SUMMARY{RESET}")
    print(f"{BOLD}{'=' * 78}{RESET}\n")

    pass_count = sum(1 for r in results if r["passed"])
    fail_count = total - pass_count
    json_fails = sum(1 for r in results if not r["json_ok"])
    retrieval_fails = sum(1 for r in results if not r["retrieval_ok"])
    avg_time = sum(r["time_s"] for r in results) / total if total else 0
    avg_citations = sum(r["citations"] for r in results) / total if total else 0
    avg_confidence = sum(r["confidence"] for r in results) / total if total else 0
    total_time = sum(r["time_s"] for r in results)

    # Per-question summary table
    print(f"  {'#':<4} {'Status':<8} {'Answer':<10} {'Conf':>6} {'Cites':>6} {'Time':>7}  Question")
    print(f"  {'─'*4} {'─'*8} {'─'*10} {'─'*6} {'─'*6} {'─'*7}  {'─'*30}")
    for i, r in enumerate(results, 1):
        status = f"{GREEN}PASS{RESET}" if r["passed"] else f"{RED}FAIL{RESET}"
        print(
            f"  {i:<4} {status:<17} {r['covered']:<10} {r['confidence']:>5.0%} {r['citations']:>6} {r['time_s']:>6.1f}s  {r['question'][:40]}"
        )

    print(f"\n  {_divider('─', 70)}\n")

    # Aggregate metrics
    pass_rate = (pass_count / total * 100) if total else 0
    print(f"  {'Pass Rate:':<25} {BOLD}{pass_rate:.0f}%{RESET}  ({pass_count}/{total})")
    print(f"  {'JSON Parse Failures:':<25} {BOLD}{json_fails}{RESET}")
    print(f"  {'Retrieval Failures:':<25} {BOLD}{retrieval_fails}{RESET}")
    print(f"  {'Avg Response Time:':<25} {BOLD}{avg_time:.1f}s{RESET}")
    print(f"  {'Avg Citations/Answer:':<25} {BOLD}{avg_citations:.1f}{RESET}")
    print(f"  {'Avg Confidence:':<25} {BOLD}{avg_confidence:.0%}{RESET}")
    print(f"  {'Total Execution Time:':<25} {BOLD}{total_time:.1f}s{RESET}")

    print(f"\n{BOLD}{'=' * 78}{RESET}")
    if fail_count == 0:
        print(f"{GREEN}{BOLD}  ✓ ALL {total} TESTS PASSED{RESET}")
    else:
        print(f"{RED}{BOLD}  ✗ {fail_count} TEST(S) FAILED{RESET}")
        # List failures
        for i, r in enumerate(results, 1):
            if not r["passed"]:
                reasons = []
                if not r["json_ok"]:
                    reasons.append("JSON parse failure")
                if not r["retrieval_ok"]:
                    reasons.append("0 citations")
                print(f"    [{i}] {r['question']}  — {', '.join(reasons)}")
    print(f"{BOLD}{'=' * 78}{RESET}\n")

    sys.exit(0 if fail_count == 0 else 1)


if __name__ == "__main__":
    main()
