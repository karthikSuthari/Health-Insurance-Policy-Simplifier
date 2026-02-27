"""
Phase 6: Streamlit Frontend
===============================
Interactive UI for querying health insurance policies.

Features:
    • Text input for plain-English questions
    • Answer badge (Yes / No / Partial) with colour coding
    • Confidence meter
    • Expandable citations with exact quotes
    • Caveat/condition list
    • Response timing

Requirements:
    pip install streamlit requests

Usage:
    streamlit run app.py
    streamlit run app.py -- --api http://localhost:8080  (custom API URL)
"""

import sys
import time
import requests
import streamlit as st

# ─── Configuration ────────────────────────────────────────────────────────────

DEFAULT_API = "http://localhost:8000"


def get_api_url() -> str:
    """Read API URL from CLI args or default."""
    for i, arg in enumerate(sys.argv):
        if arg == "--api" and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return DEFAULT_API


API_URL = get_api_url()

# ─── Page Config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Health Insurance Policy Simplifier",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────

st.markdown(
    """
    <style>
    .answer-badge {
        display: inline-block;
        padding: 0.35em 1em;
        border-radius: 0.4em;
        font-weight: 700;
        font-size: 1.5em;
        margin-bottom: 0.5em;
    }
    .badge-yes    { background: #d4edda; color: #155724; }
    .badge-no     { background: #f8d7da; color: #721c24; }
    .badge-partial{ background: #fff3cd; color: #856404; }
    .badge-unknown{ background: #e2e3e5; color: #383d41; }
    .citation-box {
        background: #f8f9fa;
        border-left: 4px solid #0d6efd;
        padding: 0.75em 1em;
        margin: 0.5em 0;
        border-radius: 4px;
        font-size: 0.92em;
        color: #000000;
    }
    .timing-bar {
        font-size: 0.82em;
        color: #6c757d;
        margin-top: 1em;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─── Sidebar ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("⚙️ Settings")
    api_url = st.text_input("API URL", value=API_URL, help="FastAPI backend URL")
    top_k = st.slider("Chunks to retrieve", min_value=1, max_value=30, value=10)

    st.divider()
    st.subheader("📊 System Status")

    # Health check
    try:
        resp = requests.get(f"{api_url}/health", timeout=5)
        if resp.ok:
            data = resp.json()
            st.success(f"API: **{data['status']}**")
            st.metric("Embeddings", f"{data['embeddings']:,}")
            st.metric("Model", data["model"])
        else:
            st.error(f"API returned {resp.status_code}")
    except requests.ConnectionError:
        st.warning("API not reachable. Start the backend:\n```\npython api.py\n```")
    except Exception as e:
        st.error(f"Health check failed: {e}")

    st.divider()
    st.caption("Health Insurance Policy Simplifier v1.0")

# ─── Main Content ────────────────────────────────────────────────────────────

st.title("🏥 Health Insurance Policy Simplifier")
st.markdown(
    "Ask a plain-English question about your health insurance policy. "
    "The system searches **32 policy documents** and returns a structured answer with citations."
)

# ─── Sample Questions ────────────────────────────────────────────────────────

SAMPLE_QUESTIONS = [
    "Is knee replacement surgery covered?",
    "What is the waiting period for pre-existing diseases?",
    "Are maternity expenses covered?",
    "Is dental treatment covered under this policy?",
    "What are the exclusions for room rent?",
    "Is AYUSH treatment covered?",
    "What is the co-payment clause?",
    "Is cataract surgery covered?",
]

st.markdown("**💡 Try a sample question:**")
cols = st.columns(4)
for i, q in enumerate(SAMPLE_QUESTIONS):
    with cols[i % 4]:
        if st.button(q, key=f"sample_{i}", use_container_width=True):
            st.session_state["question"] = q

# ─── Question Input ──────────────────────────────────────────────────────────

question = st.text_input(
    "🔍 Your question",
    value=st.session_state.get("question", ""),
    placeholder="e.g. Is knee replacement surgery covered?",
    max_chars=500,
    key="question_input",
)

# Sync sample-button selection into the text input
if "question" in st.session_state and st.session_state.get("question") != question:
    question = st.session_state["question"]

ask_clicked = st.button("🔎 Ask", type="primary", use_container_width=True)

# ─── Answer Display ──────────────────────────────────────────────────────────

result = None
elapsed = 0.0

if ask_clicked and question and len(question.strip()) >= 3:
    with st.spinner("Searching policies and generating answer…"):
        t0 = time.time()
        try:
            resp = requests.post(
                f"{api_url}/ask",
                json={"question": question.strip(), "top_k": top_k},
                timeout=120,
            )
            elapsed = time.time() - t0

            if not resp.ok:
                st.error(f"API error ({resp.status_code}): {resp.text}")
            else:
                result = resp.json()

        except requests.ConnectionError:
            st.error(
                "Cannot reach the API. Make sure the backend is running:\n"
                "```\npython api.py\n```"
            )
        except requests.Timeout:
            st.error("Request timed out. The question may be too complex. Try again.")
        except Exception as e:
            st.error(f"Unexpected error: {e}")

if result:
    answer = result.get("answer", "Unknown")
    explanation = result.get("explanation", "")
    confidence = result.get("confidence", 0.0)
    citations = result.get("citations", [])
    caveats = result.get("caveats", [])
    meta = result.get("_meta", {})

    # ── Answer badge ────────────────────────────────────────────────
    badge_class = {
        "Yes": "badge-yes",
        "No": "badge-no",
        "Partial": "badge-partial",
    }.get(answer, "badge-unknown")

    st.markdown(f'<div class="answer-badge {badge_class}">{answer}</div>', unsafe_allow_html=True)

    # ── Confidence ──────────────────────────────────────────────────
    conf_pct = int(confidence * 100)
    st.progress(confidence, text=f"Confidence: {conf_pct}%")

    # ── Explanation ─────────────────────────────────────────────────
    st.markdown(f"### Explanation\n{explanation}")

    # ── Caveats ────────────────────────────────────────────────────
    if caveats:
        st.markdown("### ⚠️ Conditions & Caveats")
        for cav in caveats:
            st.markdown(f"- {cav}")

    # ── Citations ──────────────────────────────────────────────────
    if citations:
        st.markdown(f"### 📄 Citations ({len(citations)})")
        for i, cit in enumerate(citations, 1):
            fname = cit.get("filename", "?") or "Unknown file"
            page = cit.get("page", "?") or "?"
            section = cit.get("section", "?") or "?"
            quote = cit.get("quote", "") or ""
            header = f"[{i}] {fname} — p.{page}, §{section}"
            with st.expander(header, expanded=True):
                if quote:
                    st.markdown(
                        f'<div class="citation-box">“{quote}”</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f'<div class="citation-box"><em>Referenced in {fname}, page {page}, section {section}</em></div>',
                        unsafe_allow_html=True,
                    )
                st.caption(f"File: {fname}  |  Page: {page}  |  Section: {section}")

    # ── Timing ─────────────────────────────────────────────────────
    ret_time = meta.get("retrieval_time_s", 0)
    gen_time = meta.get("generation_time_s", 0)
    total_time = meta.get("total_time_s", 0)
    chunks_used = meta.get("chunks_used", 0)

    st.markdown(
        f'<div class="timing-bar">'
        f"⏱ Retrieval: {ret_time}s &nbsp;|&nbsp; "
        f"Generation: {gen_time}s &nbsp;|&nbsp; "
        f"Total: {total_time}s &nbsp;|&nbsp; "
        f"Chunks: {chunks_used} &nbsp;|&nbsp; "
        f"Round-trip: {elapsed:.2f}s"
        f"</div>",
        unsafe_allow_html=True,
    )

elif ask_clicked:
    st.warning("Please enter a question with at least 3 characters.")
