"""
Phase 6: Streamlit Frontend
===============================
Professional UI for querying health insurance policies.

Features:
    • Sidebar with policy PDF selector dropdown
    • Text input for free-form coverage questions
    • 5 sample questions as clickable buttons
    • Results panel: YES / NO / UNCLEAR badge, confidence bar,
      plain-English explanation, expandable citations, caveats list
    • Calls FastAPI backend at localhost:8000/ask

Requirements:
    pip install streamlit requests

Usage:
    streamlit run app.py
"""

import sys
import time
import requests
import streamlit as st

# ─── Configuration ────────────────────────────────────────────────────────────

DEFAULT_API = "http://localhost:8000"


def _get_api_url() -> str:
    for i, arg in enumerate(sys.argv):
        if arg == "--api" and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return DEFAULT_API


API_URL = _get_api_url()

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
/* ── Answer badge ────────────────────────────── */
.badge-wrap {
    display: flex;
    align-items: center;
    gap: 1rem;
    margin-bottom: 1rem;
}
.answer-badge {
    display: inline-block;
    padding: 0.5em 1.4em;
    border-radius: 0.5em;
    font-weight: 800;
    font-size: 1.8em;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    flex-shrink: 0;
}
.badge-yes     { background: #d4edda; color: #155724; border: 2px solid #28a745; }
.badge-no      { background: #f8d7da; color: #721c24; border: 2px solid #dc3545; }
.badge-partial { background: #fff3cd; color: #856404; border: 2px solid #ffc107; }
.badge-unclear { background: #e2e3e5; color: #383d41; border: 2px solid #6c757d; }

/* ── Confidence bar ──────────────────────────── */
.conf-outer {
    flex: 1;
    background: #e9ecef;
    border-radius: 0.4em;
    height: 28px;
    overflow: hidden;
}
.conf-inner {
    height: 100%;
    border-radius: 0.4em;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 700;
    font-size: 0.85em;
    color: #fff;
    transition: width 0.5s ease;
}

/* ── Section cards ───────────────────────────── */
.section-card {
    background: #ffffff;
    border: 1px solid #dee2e6;
    border-radius: 0.6em;
    padding: 1.2em 1.4em;
    margin-bottom: 1em;
    color: #212529;
}
.section-card h4 { margin-top: 0; color: #212529; }
.section-card p  { color: #212529; }

/* ── Citation box ────────────────────────────── */
.citation-box {
    background: #f8f9fa;
    border-left: 4px solid #0d6efd;
    padding: 0.75em 1em;
    margin: 0.5em 0;
    border-radius: 4px;
    font-size: 0.93em;
    color: #000000;
    line-height: 1.55;
}
.citation-meta {
    font-size: 0.8em;
    color: #6c757d;
    margin-top: 0.3em;
}

/* ── Caveat pills ────────────────────────────── */
.caveat-pill {
    display: inline-block;
    background: #fff3cd;
    color: #856404;
    border: 1px solid #ffc107;
    border-radius: 2em;
    padding: 0.3em 0.9em;
    margin: 0.25em 0.3em 0.25em 0;
    font-size: 0.88em;
}

/* ── Timing footer ───────────────────────────── */
.timing-bar {
    font-size: 0.78em;
    color: #6c757d;
    text-align: right;
    margin-top: 0.5em;
    padding-top: 0.5em;
    border-top: 1px solid #dee2e6;
}

/* ── Sample-question buttons ─────────────────── */
div[data-testid="stHorizontalBlock"] div.stButton > button {
    font-size: 0.82em;
    border-radius: 1.5em;
    border: 1px solid #0d6efd;
    color: #0d6efd;
    background: transparent;
    transition: all 0.15s;
}
div[data-testid="stHorizontalBlock"] div.stButton > button:hover {
    background: #0d6efd;
    color: #fff;
}
</style>
""",
    unsafe_allow_html=True,
)

# ─── Helper Functions ─────────────────────────────────────────────────────────


@st.cache_data(ttl=300)
def _fetch_pdfs(api_url: str) -> list[str]:
    """Get available policy PDF names from the API."""
    try:
        r = requests.get(f"{api_url}/pdfs", timeout=5)
        if r.ok:
            return r.json().get("pdfs", [])
    except Exception:
        pass
    return []


def _health_check(api_url: str) -> dict | None:
    try:
        r = requests.get(f"{api_url}/health", timeout=5)
        return r.json() if r.ok else None
    except Exception:
        return None


def _conf_color(v: float) -> str:
    if v >= 0.75:
        return "#28a745"  # green
    if v >= 0.50:
        return "#ffc107"  # yellow
    return "#dc3545"      # red


def _badge_class(answer: str) -> str:
    a = answer.strip().lower()
    return {"yes": "badge-yes", "no": "badge-no", "partial": "badge-partial"}.get(a, "badge-unclear")


def _badge_label(answer: str) -> str:
    a = answer.strip().lower()
    if a in ("yes", "no", "partial"):
        return answer.strip().upper()
    return "UNCLEAR"


# ─── Sidebar ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🏥 Policy Simplifier")
    st.caption("Powered by Ollama llama3 · v1.0")
    st.divider()

    # ── Policy PDF dropdown ────────────────────────────────────────
    st.subheader("📂 Select Policy")
    pdf_list = _fetch_pdfs(API_URL)
    pdf_options = ["All Policies"] + pdf_list
    selected_pdf = st.selectbox(
        "Policy document",
        options=pdf_options,
        index=0,
        help="Filter results to a specific policy PDF, or search across all.",
    )

    st.divider()

    # ── Settings ───────────────────────────────────────────────────
    st.subheader("⚙️ Settings")
    api_url = st.text_input("API endpoint", value=API_URL, help="FastAPI backend URL")
    top_k = st.slider("Chunks to retrieve", min_value=1, max_value=30, value=10)

    st.divider()

    # ── Health check ───────────────────────────────────────────────
    st.subheader("📊 System Status")
    health = _health_check(api_url)
    if health:
        st.success(f"API: **{health['status'].upper()}**")
        c1, c2 = st.columns(2)
        c1.metric("Embeddings", f"{health['embeddings']:,}")
        c2.metric("Model", health["model"])
    else:
        st.error("API offline — run `python api.py`")

# ─── Main Content ────────────────────────────────────────────────────────────

st.markdown(
    '<h1 style="margin-bottom:0;">🏥 Health Insurance Policy Simplifier</h1>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p style="color:#6c757d;font-size:1.05em;margin-top:0.2em;">'
    "Ask a plain-English coverage question. "
    "The system searches <strong>32 policy documents</strong> and returns a structured, cited answer."
    "</p>",
    unsafe_allow_html=True,
)

# ─── 5 Sample Questions as Clickable Buttons ─────────────────────────────────

SAMPLE_QUESTIONS = [
    "Is knee replacement surgery covered?",
    "What is the waiting period for pre-existing diseases?",
    "Are maternity expenses covered?",
    "Is dental treatment covered under this policy?",
    "What are the exclusions for room rent?",
]

st.markdown("##### 💡 Quick Questions")
sq_cols = st.columns(len(SAMPLE_QUESTIONS))
for i, q in enumerate(SAMPLE_QUESTIONS):
    with sq_cols[i]:
        if st.button(q, key=f"sq_{i}", use_container_width=True):
            st.session_state["question"] = q

# ─── Question Input ──────────────────────────────────────────────────────────

st.markdown("")  # spacer
question = st.text_input(
    "🔍 Your coverage question",
    value=st.session_state.get("question", ""),
    placeholder="e.g. Is cataract surgery covered under my policy?",
    max_chars=500,
    key="question_input",
)

# Sync sample-button click into the text box
if "question" in st.session_state and st.session_state["question"] != question:
    question = st.session_state["question"]

ask_clicked = st.button("🔎  Get Answer", type="primary", use_container_width=True)

# ─── Query API ────────────────────────────────────────────────────────────────

result = None
elapsed = 0.0

if ask_clicked and question and len(question.strip()) >= 3:
    with st.spinner("Searching policies and generating answer …"):
        t0 = time.time()
        try:
            payload: dict = {"question": question.strip(), "top_k": top_k}
            resp = requests.post(f"{api_url}/ask", json=payload, timeout=180)
            elapsed = time.time() - t0
            if resp.ok:
                result = resp.json()
            else:
                st.error(f"API error ({resp.status_code}): {resp.text}")
        except requests.ConnectionError:
            st.error("Cannot reach the API. Start the backend:\n```\npython api.py\n```")
        except requests.Timeout:
            st.error("Request timed out — try a simpler question.")
        except Exception as e:
            st.error(f"Unexpected error: {e}")
elif ask_clicked:
    st.warning("Please enter a question with at least 3 characters.")

# ─── Results Panel ────────────────────────────────────────────────────────────

if result:
    answer = result.get("answer", "Unknown")
    explanation = result.get("explanation", "No explanation provided.")
    confidence = result.get("confidence", 0.0)
    citations = result.get("citations", [])
    caveats = result.get("caveats", [])
    meta = result.get("_meta", {})

    # Filter citations to selected PDF if not "All Policies"
    if selected_pdf != "All Policies" and citations:
        filtered = [c for c in citations if c.get("filename", "") == selected_pdf]
        if filtered:
            citations = filtered

    st.divider()

    # ── 1. Big YES / NO / UNCLEAR badge + confidence bar ──────────
    badge_cls = _badge_class(answer)
    badge_lbl = _badge_label(answer)
    conf_pct = max(0, min(100, int(confidence * 100)))
    conf_clr = _conf_color(confidence)

    st.markdown(
        f"""
        <div class="badge-wrap">
            <div class="answer-badge {badge_cls}">{badge_lbl}</div>
            <div class="conf-outer">
                <div class="conf-inner" style="width:{conf_pct}%;background:{conf_clr};">
                    {conf_pct}% confidence
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── 2. Plain-English explanation ───────────────────────────────
    st.markdown(
        f"""
        <div class="section-card">
            <h4>📝 Explanation</h4>
            <p>{explanation}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── 3. Caveats list ───────────────────────────────────────────
    if caveats:
        pills_html = "".join(f'<span class="caveat-pill">⚠️ {c}</span>' for c in caveats)
        st.markdown(
            f"""
            <div class="section-card">
                <h4>⚠️ Conditions &amp; Caveats</h4>
                {pills_html}
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── 4. Expandable citations ───────────────────────────────────
    if citations:
        st.markdown(f"### 📄 Citations ({len(citations)})")
        for idx, cit in enumerate(citations, 1):
            fname = cit.get("filename", "Unknown file") or "Unknown file"
            page = cit.get("page", "?") or "?"
            section = cit.get("section", "") or ""
            quote = cit.get("quote", "") or ""

            header_parts = [f"[{idx}]  {fname}"]
            if str(page) != "?":
                header_parts.append(f"p. {page}")
            if section:
                header_parts.append(f"§ {section}")
            header = "  —  ".join(header_parts)

            with st.expander(header, expanded=True):
                if quote:
                    st.markdown(
                        f'<div class="citation-box">"{quote}"</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f'<div class="citation-box"><em>Referenced in {fname}, page {page}</em></div>',
                        unsafe_allow_html=True,
                    )
                st.markdown(
                    f'<div class="citation-meta">📁 {fname} &nbsp;|&nbsp; 📄 Page {page}'
                    + (f" &nbsp;|&nbsp; 📑 {section}" if section else "")
                    + "</div>",
                    unsafe_allow_html=True,
                )
    else:
        st.info("No specific citations were found for this question.")

    # ── 5. Timing footer ──────────────────────────────────────────
    ret_t = meta.get("retrieval_time_s", 0)
    gen_t = meta.get("generation_time_s", 0)
    total_t = meta.get("total_time_s", 0)
    chunks_n = meta.get("chunks_used", 0)

    st.markdown(
        f'<div class="timing-bar">'
        f"⏱ Retrieval: {ret_t:.1f}s &nbsp;|&nbsp; "
        f"Generation: {gen_t:.1f}s &nbsp;|&nbsp; "
        f"Total: {total_t:.1f}s &nbsp;|&nbsp; "
        f"Chunks: {chunks_n} &nbsp;|&nbsp; "
        f"Round-trip: {elapsed:.2f}s"
        f"</div>",
        unsafe_allow_html=True,
    )
