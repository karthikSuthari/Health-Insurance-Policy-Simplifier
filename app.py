"""
InsureIQ — Health Insurance Policy Simplifier
================================================
Production-ready Streamlit UI with professional design.
Connects to FastAPI backend at localhost:8000.

Usage:  streamlit run app.py
"""

import sys
import time
import json
from datetime import datetime
from typing import Any
import requests
import streamlit as st

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="InsureIQ — Know What's Covered",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — API CLIENT
# ═══════════════════════════════════════════════════════════════════════════════

class APIClient:
    BASE_URL = "http://localhost:8000"

    @staticmethod
    def health_check() -> dict:
        try:
            r = requests.get(f"{APIClient.BASE_URL}/health", timeout=3)
            return r.json() if r.ok else {"status": "offline"}
        except Exception:
            return {"status": "offline"}

    @staticmethod
    @st.cache_data(ttl=300)
    def get_policies() -> list[str]:
        try:
            r = requests.get(f"{APIClient.BASE_URL}/pdfs", timeout=5)
            return r.json().get("pdfs", []) if r.ok else []
        except Exception:
            return []

    @staticmethod
    def get_stats() -> dict:
        try:
            r = requests.get(f"{APIClient.BASE_URL}/stats", timeout=5)
            return r.json() if r.ok else {}
        except Exception:
            return {}

    @staticmethod
    def ask_question(question: str, top_k: int = 10) -> dict:
        r = requests.post(
            f"{APIClient.BASE_URL}/ask",
            json={"question": question, "top_k": top_k},
            timeout=180,
        )
        if r.ok:
            return r.json()
        raise Exception(f"API error {r.status_code}: {r.text[:200]}")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — SESSION STATE
# ═══════════════════════════════════════════════════════════════════════════════

_DEFAULTS: dict[str, Any] = {
    "chat_history": [],
    "last_answer": None,
    "last_citations": [],
    "active_policy": "All Policies",
    "compare_policies": [],
    "compare_mode": False,
    "settings": {
        "confidence_threshold": 0.75,
        "show_raw_json": False,
        "auto_expand_citations": False,
        "top_k": 10,
    },
    "query_count": 0,
    "session_start": datetime.now().isoformat(),
    "trigger_question": None,
}

for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — GLOBAL CSS
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

:root {
    --navy:  #0A2342;
    --blue:  #1B6CA8;
    --cyan:  #00B4D8;
    --green: #2ECC71;
    --amber: #F39C12;
    --red:   #E74C3C;
    --gray:  #636E72;
    --light: #F8F9FA;
    --white: #FFFFFF;
    --text:  #2D3436;
}

html, body, [class*="st-"] { font-family: 'Inter', sans-serif !important; }
#MainMenu, header, footer { visibility: hidden !important; }
.block-container { padding-top: 1.5rem !important; max-width: 1200px; }

/* Sidebar spacing fixes */
section[data-testid="stSidebar"] .stMarkdown p { margin-bottom: 0.3rem !important; }
section[data-testid="stSidebar"] .stSelectbox,
section[data-testid="stSidebar"] .stMultiSelect { margin-bottom: 0.5rem !important; }
section[data-testid="stSidebar"] .stButton { margin-bottom: 2px !important; }
section[data-testid="stSidebar"] .stButton > button { padding: 6px 14px !important; font-size: 13px !important; }
section[data-testid="stSidebar"] .stExpander { margin-top: 0.5rem !important; }
section[data-testid="stSidebar"] .stExpander [data-testid="stExpanderDetails"] > div { gap: 0.75rem !important; display: flex; flex-direction: column; }
section[data-testid="stSidebar"] hr { margin: 0.5rem 0 !important; }
section[data-testid="stSidebar"] .stSlider { margin-bottom: 0.5rem !important; }
section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] { margin-bottom: 0 !important; }

/* Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--light); }
::-webkit-scrollbar-thumb { background: var(--navy); border-radius: 3px; }

/* Buttons */
.stButton > button {
    border-radius: 10px !important;
    padding: 10px 20px !important;
    font-weight: 600 !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover { transform: translateY(-1px); }
.stButton > button:active { transform: scale(0.97); }

/* Cards */
.iq-card {
    background: var(--white);
    border-radius: 16px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.08);
    padding: 24px;
    margin-bottom: 16px;
    transition: transform 0.2s, box-shadow 0.2s;
    color: var(--text);
}
.iq-card:hover { transform: translateY(-2px); box-shadow: 0 8px 30px rgba(0,0,0,0.12); }

.iq-glass {
    background: rgba(255,255,255,0.1);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255,255,255,0.2);
    border-radius: 16px;
    padding: 24px;
}

/* Top bar */
.iq-topbar {
    background: var(--navy);
    color: white;
    padding: 12px 28px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    border-radius: 12px;
    margin-bottom: 1rem;
}
.iq-topbar .logo { font-size: 22px; font-weight: 800; letter-spacing: -0.5px; }
.iq-topbar .logo span { color: var(--cyan); }
.iq-topbar .status { font-size: 13px; display: flex; gap: 18px; align-items: center; flex-wrap: wrap; }

@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.4} }
.dot { width:8px; height:8px; border-radius:50%; display:inline-block; margin-right:5px; animation: pulse 2s infinite; }
.dot-green { background: var(--green); }
.dot-red   { background: var(--red); }

/* Verdict banners */
.verdict-banner {
    border-radius: 16px;
    padding: 24px 28px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 16px;
    animation: fadeSlide 0.5s ease;
}
@keyframes fadeSlide { from { opacity:0; transform:translateY(12px); } to { opacity:1; transform:translateY(0); } }

.verdict-yes     { background: linear-gradient(135deg,#F0FFF4,#DAFBE1); border: 2px solid var(--green); border-left: 6px solid var(--green); }
.verdict-no      { background: linear-gradient(135deg,#FFF5F5,#FFE0E0); border: 2px solid var(--red); border-left: 6px solid var(--red); }
.verdict-partial { background: linear-gradient(135deg,#FFFBF0,#FFF3CD); border: 2px solid var(--amber); border-left: 6px solid var(--amber); }
.verdict-unknown { background: linear-gradient(135deg,#F8F9FA,#ECEFF1); border: 2px solid var(--gray); border-left: 6px solid var(--gray); }

.verdict-text { font-size: 28px; font-weight: 900; margin: 0; }
.verdict-icon { font-size: 36px; margin-right: 14px; }

/* Confidence circle */
.conf-circle {
    width: 90px; height: 90px;
    border-radius: 50%;
    display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    font-weight: 900; font-size: 26px;
    color: white; flex-shrink: 0;
    animation: popIn 0.6s ease;
}
@keyframes popIn { from { transform:scale(0.5); opacity:0; } to { transform:scale(1); opacity:1; } }

/* Stat cards */
.stat-card {
    background: var(--white);
    border-radius: 12px;
    padding: 18px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.06);
    text-align: center;
    color: var(--text);
}
.stat-card .stat-val { font-size: 22px; font-weight: 800; }
.stat-card .stat-lbl { font-size: 12px; color: var(--gray); text-transform: uppercase; letter-spacing: 0.5px; margin-top: 4px; }

/* Citation card */
.cite-card {
    border-radius: 10px;
    padding: 16px 18px;
    margin: 10px 0;
    font-size: 14px;
    line-height: 1.65;
    color: #000;
    animation: slideIn 0.4s ease;
    overflow: hidden;
    word-wrap: break-word;
    overflow-wrap: break-word;
}
@keyframes slideIn { from { opacity:0; transform:translateX(-16px); } to { opacity:1; transform:translateX(0); } }
.cite-green   { background: #F0FFF4; border-left: 4px solid var(--green); }
.cite-red     { background: #FFF5F5; border-left: 4px solid var(--red); }
.cite-amber   { background: #FFFBF0; border-left: 4px solid var(--amber); }
.cite-blue    { background: #EBF5FB; border-left: 4px solid var(--blue); }
.cite-card .cite-quote {
    font-family: 'Georgia', serif;
    font-style: italic;
    margin: 10px 0;
    padding: 10px 14px;
    background: rgba(255,255,255,0.7);
    border-radius: 6px;
    word-wrap: break-word;
    overflow-wrap: break-word;
}
.cite-card .cite-meta {
    font-size: 11px;
    color: var(--gray);
    margin-top: 8px;
    padding-top: 6px;
    border-top: 1px solid rgba(0,0,0,0.06);
}
.cite-card .cite-badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 12px;
    font-size: 11px;
    font-weight: 600;
    vertical-align: middle;
    margin-left: 0;
    margin-top: 4px;
    max-width: 100%;
    word-wrap: break-word;
    overflow-wrap: break-word;
    white-space: normal;
}

/* Caveat pills */
.cav-pill {
    display: inline-block;
    background: #FFF3CD;
    color: #856404;
    border: 1px solid #F39C12;
    border-radius: 20px;
    padding: 5px 14px;
    margin: 4px 4px;
    font-size: 13px;
    max-width: 100%;
    word-wrap: break-word;
    overflow-wrap: break-word;
    box-sizing: border-box;
}
.cav-container { display: flex; flex-wrap: wrap; gap: 6px; }

/* Plain English box */
.plain-box {
    background: linear-gradient(135deg,#EBF5FB,#E8F8F5);
    border-radius: 12px;
    padding: 18px 22px;
    border-left: 4px solid var(--cyan);
    margin: 12px 0;
    color: var(--text);
    line-height: 1.7;
}

/* Sidebar pills */
.sq-pill {
    display: inline-block;
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.2);
    border-radius: 20px;
    padding: 6px 14px;
    font-size: 12px;
    color: rgba(255,255,255,0.85);
    cursor: pointer;
    transition: all 0.2s;
    margin: 3px 2px;
    text-decoration: none;
}
.sq-pill:hover { background: rgba(0,180,216,0.25); border-color: var(--cyan); color: white; }

/* Chat messages */
.chat-user {
    background: var(--navy); color:white;
    border-radius:12px 12px 0 12px;
    padding:12px 16px; margin:8px 0; margin-left:20%;
    text-align:right; font-size:14px;
    word-wrap: break-word; overflow-wrap: break-word;
}
.chat-agent {
    background:var(--white); color:var(--text);
    border:1px solid #dee2e6;
    border-radius:0 12px 12px 12px;
    padding:12px 16px; margin:8px 0; margin-right:20%;
    font-size:14px;
    word-wrap: break-word; overflow-wrap: break-word;
}
.chat-time { font-size:11px; color:var(--gray); margin-top:4px; }

/* Footer */
.iq-footer {
    background: var(--navy);
    color: rgba(255,255,255,0.6);
    font-size: 12px;
    padding: 12px 28px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-radius: 12px;
    margin-top: 2rem;
    flex-wrap: wrap;
    gap: 8px;
}

/* Metric card top border colors */
.stat-blue   { border-top: 4px solid var(--blue); }
.stat-green  { border-top: 4px solid var(--green); }
.stat-red    { border-top: 4px solid var(--red); }
.stat-amber  { border-top: 4px solid var(--amber); }

/* Tab styling */
.stTabs [data-baseweb="tab-list"] button { font-size:14px; font-weight:600; padding:12px 20px; }
.stTabs [data-baseweb="tab-list"] button[aria-selected="true"] { border-bottom:3px solid var(--cyan); }

/* ── Custom collapsible sections (replaces st.expander for output) ── */
.iq-collapse {
    border: 1px solid #dee2e6;
    border-radius: 12px;
    margin: 10px 0;
    overflow: hidden;
    background: var(--white);
}
.iq-collapse summary {
    display: flex;
    align-items: center;
    padding: 14px 18px;
    cursor: pointer;
    font-weight: 600;
    font-size: 14px;
    color: var(--navy);
    background: #F8F9FA;
    list-style: none;
    user-select: none;
    transition: background 0.2s;
    gap: 10px;
}
.iq-collapse summary:hover { background: #EEF1F4; }
.iq-collapse summary::-webkit-details-marker { display: none; }
.iq-collapse summary::marker { display: none; content: ""; }
.iq-collapse summary .iq-chevron {
    flex-shrink: 0;
    width: 20px;
    height: 20px;
    transition: transform 0.25s ease;
    color: #636E72;
}
.iq-collapse[open] summary .iq-chevron { transform: rotate(90deg); }
.iq-collapse summary .iq-collapse-label { flex: 1; }
.iq-collapse-body {
    padding: 14px 18px;
    word-wrap: break-word;
    overflow-wrap: break-word;
}

/* ── Streamlit expander: hide Material Icons text (for Settings / Raw JSON) ── */
[data-testid="stExpander"] [data-testid="stExpanderToggleIcon"] {
    font-size: 0 !important;
    color: transparent !important;
    overflow: hidden !important;
    width: 0 !important;
    min-width: 0 !important;
    max-width: 0 !important;
    height: 0 !important;
    padding: 0 !important;
    margin: 0 !important;
    line-height: 0 !important;
    display: none !important;
    visibility: hidden !important;
}
[data-testid="stExpander"] details > summary {
    display: flex !important;
    align-items: center !important;
    padding: 12px 16px !important;
    cursor: pointer !important;
    list-style: none !important;
}
[data-testid="stExpander"] details > summary::marker,
[data-testid="stExpander"] details > summary::-webkit-details-marker {
    display: none !important;
    content: "" !important;
}
[data-testid="stExpander"] details > summary p {
    margin: 0 !important;
    font-weight: 600 !important;
    font-size: 14px !important;
}
[data-testid="stExpander"] [data-testid="stExpanderDetails"] {
    padding: 12px 16px !important;
    overflow-x: hidden !important;
    word-wrap: break-word !important;
}

/* Sidebar expander: also fully hide the toggle icon */
section[data-testid="stSidebar"] [data-testid="stExpanderToggleIcon"] {
    display: none !important;
    visibility: hidden !important;
    width: 0 !important;
    height: 0 !important;
    overflow: hidden !important;
    font-size: 0 !important;
}
section[data-testid="stSidebar"] [data-testid="stExpander"] details > summary {
    display: flex !important;
    align-items: center !important;
    padding: 10px 12px !important;
    list-style: none !important;
}
section[data-testid="stSidebar"] [data-testid="stExpander"] details > summary p {
    margin: 0 !important;
    font-weight: 600 !important;
    font-size: 14px !important;
}

/* Prevent text overflow in cards */
.iq-card p, .plain-box span, .cite-card strong { word-wrap: break-word; overflow-wrap: break-word; }
.stat-card { min-height: 90px; }

/* Mobile */
@media (max-width: 768px) {
    .verdict-banner { flex-direction: column; text-align: center; gap: 12px; padding: 16px; }
    .iq-topbar { flex-direction: column; gap: 8px; padding: 10px 16px; }
    .iq-footer { flex-direction: column; gap: 4px; text-align: center; }
    .iq-card { padding: 16px; }
    .conf-circle { width: 70px; height: 70px; font-size: 20px; }
    .stat-card { padding: 12px; }
    .stat-card .stat-val { font-size: 18px; }
}
</style>
""", unsafe_allow_html=True)

# ── JavaScript: strip Material Icon text nodes from remaining st.expander ──
st.markdown("""
<script>
(function hideExpanderIconText() {
    function strip() {
        document.querySelectorAll('[data-testid="stExpanderToggleIcon"]').forEach(function(el) {
            el.childNodes.forEach(function(n) {
                if (n.nodeType === 3) n.textContent = '';
            });
            el.style.fontSize = '0';
            el.style.width = '0';
            el.style.height = '0';
            el.style.overflow = 'hidden';
            el.style.display = 'none';
        });
    }
    strip();
    var obs = new MutationObserver(strip);
    obs.observe(document.body, {childList: true, subtree: true});
})();
</script>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def _verdict_config(answer: str) -> dict:
    a = answer.strip().lower()
    configs = {
        "yes":     {"cls": "verdict-yes",     "icon": "✅", "label": "COVERED",       "color": "#2ECC71"},
        "no":      {"cls": "verdict-no",      "icon": "❌", "label": "NOT COVERED",   "color": "#E74C3C"},
        "partial": {"cls": "verdict-partial",  "icon": "⚠️", "label": "CONDITIONAL",  "color": "#F39C12"},
    }
    return configs.get(a, {"cls": "verdict-unknown", "icon": "❓", "label": "UNCLEAR", "color": "#636E72"})


def _conf_label(conf: float) -> tuple[str, str]:
    if conf >= 0.80:
        return "● High", "#2ECC71"
    if conf >= 0.60:
        return "● Moderate", "#F39C12"
    return "● Low", "#E74C3C"


def _trunc(text: str, limit: int) -> str:
    """Truncate *text* at a word boundary, never mid-word."""
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit(" ", 1)[0]
    return cut.rstrip(".,;: ") + " …"


def _elapsed_str() -> str:
    try:
        start = datetime.fromisoformat(st.session_state["session_start"])
        delta = datetime.now() - start
        mins = int(delta.total_seconds() // 60)
        return f"{mins}m" if mins > 0 else f"{int(delta.total_seconds())}s"
    except Exception:
        return "0s"


def _render_topbar(health: dict, policy_count: int):
    status = health.get("status", "offline")
    emb = health.get("embeddings", 0)
    if status != "offline":
        dot_cls, status_text = "dot-green", f"API Connected · {emb:,} embeddings"
    else:
        dot_cls, status_text = "dot-red", "API Offline — Start backend first"

    st.markdown(f"""
    <div class="iq-topbar">
        <div class="logo">🛡️ Insure<span>IQ</span></div>
        <div class="status">
            <span><span class="dot {dot_cls}"></span>{status_text}</span>
            <span>📄 {policy_count} Policies</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


def _render_verdict(result: dict):
    answer = result.get("answer", "Unknown")
    confidence = result.get("confidence", 0.0)
    vc = _verdict_config(answer)
    conf_pct = max(0, min(100, int(confidence * 100)))
    cl, cc = _conf_label(confidence)

    st.markdown(f"""
    <div class="verdict-banner {vc['cls']}">
        <div style="display:flex;align-items:center;">
            <span class="verdict-icon">{vc['icon']}</span>
            <p class="verdict-text" style="color:{vc['color']}">{vc['label']}</p>
        </div>
        <div style="text-align:center;">
            <div class="conf-circle" style="background:{vc['color']};">
                {conf_pct}%
            </div>
            <div style="font-size:12px;color:#636E72;margin-top:4px;">Confidence</div>
            <div style="font-size:12px;color:{cc};font-weight:600;">{cl}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def _render_stat_cards(result: dict):
    citations = result.get("citations", [])
    caveats = result.get("caveats", [])
    meta = result.get("_meta", {})
    n_cites = len(citations)
    n_cav = len(caveats)
    n_chunks = meta.get("chunks_used", meta.get("chunks_retrieved", 0))

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""<div class="stat-card stat-blue">
            <div style="font-size:24px;">📎</div>
            <div class="stat-val" style="color:#1B6CA8;">{n_cites}</div>
            <div class="stat-lbl">Citations Found</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="stat-card stat-amber">
            <div style="font-size:24px;">⚠️</div>
            <div class="stat-val" style="color:#F39C12;">{n_cav}</div>
            <div class="stat-lbl">Caveats</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="stat-card stat-green">
            <div style="font-size:24px;">📄</div>
            <div class="stat-val" style="color:#2ECC71;">{n_chunks}</div>
            <div class="stat-lbl">Chunks Analyzed</div>
        </div>""", unsafe_allow_html=True)


def _render_explanation(result: dict):
    explanation = result.get("explanation", "No explanation available.")
    st.markdown(f"""<div class="iq-card">
        <h4 style="color:#0A2342;margin-top:0;">📋 What This Means</h4>
        <p style="font-size:15px;line-height:1.8;color:#2D3436;">{explanation}</p>
    </div>""", unsafe_allow_html=True)

    st.markdown(f"""<div class="plain-box">
        <span style="font-style:italic;color:#636E72;font-size:13px;">In simple terms:</span><br>
        <span style="font-size:15px;">{explanation}</span>
    </div>""", unsafe_allow_html=True)


# ── SVG chevron used for custom collapsible sections ──
_CHEVRON_SVG = '<svg class="iq-chevron" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="9 18 15 12 9 6"></polyline></svg>'


def _render_citations(citations: list, auto_expand: bool):
    if not citations:
        st.info("No specific citations were found.")
        return

    cards_html = []
    for i, cit in enumerate(citations):
        fname = cit.get("filename", "Unknown") or "Unknown"
        page = cit.get("page", "?") or "?"
        section = cit.get("section", "") or ""
        quote = cit.get("quote", "") or ""

        style_cls = "cite-blue"
        if any(w in section.lower() for w in ["exclusion", "exclude", "not covered"]):
            style_cls = "cite-red"
        elif any(w in section.lower() for w in ["condition", "waiting", "limit"]):
            style_cls = "cite-amber"
        elif any(w in section.lower() for w in ["cover", "benefit", "eligible"]):
            style_cls = "cite-green"

        quote_html = f'<div class="cite-quote">"{quote}"</div>' if quote else ""
        section_html = f'<div style="margin-top:4px;"><span class="cite-badge" style="background:rgba(0,0,0,0.06);color:#2D3436;">{section}</span></div>' if section else ""

        cards_html.append(f"""<div class="cite-card {style_cls}">
            <div style="font-weight:700;font-size:14px;">[{i+1}] {fname} — Page {page}</div>
            {section_html}
            {quote_html}
            <div class="cite-meta">📁 {fname} &nbsp;|&nbsp; 📄 Page {page}</div>
        </div>""")

    open_attr = ' open' if auto_expand else ''
    body = '\n'.join(cards_html)
    st.markdown(f"""<details class="iq-collapse"{open_attr}>
        <summary>{_CHEVRON_SVG}<span class="iq-collapse-label">📎 Policy Citations ({len(citations)} found)</span></summary>
        <div class="iq-collapse-body">{body}</div>
    </details>""", unsafe_allow_html=True)


def _render_caveats(caveats: list):
    if not caveats:
        return
    pills = "".join(f'<span class="cav-pill">⚠️ {c}</span>' for c in caveats if c)
    body = f'<div class="cav-container">{pills}</div>' if pills else '<span style="color:#636E72;">No special conditions found.</span>'
    st.markdown(f"""<details class="iq-collapse">
        <summary>{_CHEVRON_SVG}<span class="iq-collapse-label">⚠️ Conditions &amp; Requirements ({len(caveats)})</span></summary>
        <div class="iq-collapse-body">{body}</div>
    </details>""", unsafe_allow_html=True)


def _render_chat_history():
    history = st.session_state.get("chat_history", [])
    if not history:
        return
    msgs = []
    for item in history:
        ts = item.get("timestamp", "")
        q = item.get("question", "")
        a = item.get("answer", {})
        verdict = a.get("answer", "?")
        vc = _verdict_config(verdict)
        expl = _trunc(a.get('explanation', ''), 120)
        msgs.append(f"""
            <div class="chat-user">{q}</div>
            <div class="chat-agent">
                <div style="margin-bottom:6px;">
                    <span class="cite-badge" style="background:{vc['color']};color:white;padding:3px 10px;border-radius:10px;font-size:11px;">{vc['label']}</span>
                </div>
                <div style="font-size:13px;line-height:1.6;">{expl}</div>
                <div class="chat-time">{ts}</div>
            </div>""")
    body = '\n'.join(msgs)
    st.markdown(f"""<details class="iq-collapse">
        <summary>{_CHEVRON_SVG}<span class="iq-collapse-label">💬 Previous Questions ({len(history)})</span></summary>
        <div class="iq-collapse-body">{body}</div>
    </details>""", unsafe_allow_html=True)


def _render_footer():
    qc = st.session_state.get("query_count", 0)
    elapsed = _elapsed_str()
    st.markdown(f"""
    <div class="iq-footer">
        <span>🛡️ InsureIQ · AI-Powered Insurance Analysis</span>
        <span>⚡ Powered by Ollama llama3 + ChromaDB</span>
        <span>Session: {qc} queries · {elapsed} elapsed</span>
    </div>
    """, unsafe_allow_html=True)


def _do_query(question: str, top_k: int):
    """Execute query and store results."""
    result = APIClient.ask_question(question, top_k=top_k)
    st.session_state["last_answer"] = result
    st.session_state["last_citations"] = result.get("citations", [])
    st.session_state["query_count"] = st.session_state.get("query_count", 0) + 1
    history = st.session_state.get("chat_history", [])
    history.insert(0, {
        "question": question,
        "answer": result,
        "timestamp": datetime.now().strftime("%H:%M:%S"),
    })
    st.session_state["chat_history"] = history[:20]
    return result

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════

health = APIClient.health_check()
policies = APIClient.get_policies()

with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:12px 0 4px 0;">
        <div style="font-size:28px;font-weight:900;color:white;letter-spacing:-0.5px;">
            🛡️ Insure<span style="color:#00B4D8;">IQ</span>
        </div>
        <div style="font-size:12px;color:rgba(255,255,255,0.5);margin-top:2px;">
            Powered by Ollama llama3 · v1.0
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    # ── Policy Selector ────────────────────────────────────────────
    st.markdown('<div style="color:#00B4D8;font-size:11px;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;margin-bottom:4px;">YOUR POLICIES</div>', unsafe_allow_html=True)
    pdf_options = ["All Policies"] + [f"📄 {p}" for p in policies]
    sel = st.selectbox("Policy document", options=pdf_options, index=0, label_visibility="collapsed")
    st.session_state["active_policy"] = sel
    st.caption(f"{len(policies)} policies indexed")

    # ── Compare Mode ───────────────────────────────────────────────
    compare_on = st.toggle("⚖️ Compare Mode", value=st.session_state.get("compare_mode", False))
    st.session_state["compare_mode"] = compare_on
    if compare_on and policies:
        cmp = st.multiselect("Compare up to 3", policies[:20], max_selections=3)
        st.session_state["compare_policies"] = cmp
        if cmp:
            st.caption(f"Comparing {len(cmp)} policies")

    st.divider()

    # ── Quick Questions ────────────────────────────────────────────
    st.markdown('<div style="color:#00B4D8;font-size:11px;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;margin-bottom:4px;">QUICK QUESTIONS</div>', unsafe_allow_html=True)
    QUICK_QS = [
        ("🧠", "Mental Health", "Are mental health treatments covered?"),
        ("🚑", "Emergency Room", "Is emergency room treatment covered?"),
        ("💊", "Prescription Drugs", "Are prescription drugs covered under the policy?"),
        ("🔬", "MRI / CT Scan", "Is MRI or CT scan covered?"),
        ("🤰", "Maternity Care", "Are maternity expenses covered?"),
    ]
    for icon, label, full_q in QUICK_QS:
        if st.button(f"{icon} {label}", key=f"qq_{label}", use_container_width=True):
            st.session_state["trigger_question"] = full_q

    st.divider()

    # ── Settings ───────────────────────────────────────────────────
    with st.expander("⚙️ Settings", expanded=False):
        settings = st.session_state["settings"]
        settings["confidence_threshold"] = st.slider(
            "Min confidence", 0.5, 1.0, settings["confidence_threshold"], 0.05,
            help="Minimum confidence to show answer",
            key="sld_conf",
        )
        settings["show_raw_json"] = st.checkbox("Show raw JSON", value=settings["show_raw_json"], help="Debug: show full API response", key="chk_json")
        settings["auto_expand_citations"] = st.checkbox("Auto-expand citations", value=settings["auto_expand_citations"], key="chk_cit")
        settings["top_k"] = st.slider("Chunks to retrieve", 1, 30, settings.get("top_k", 10), key="sld_topk")
    st.session_state["settings"] = settings

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — TOP BAR
# ═══════════════════════════════════════════════════════════════════════════════

_render_topbar(health, len(policies))

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — API OFFLINE STATE
# ═══════════════════════════════════════════════════════════════════════════════

api_online = health.get("status", "offline") != "offline"

if not api_online:
    st.markdown("""
    <div style="background:linear-gradient(135deg,#FFF5F5,#FFE0E0);border:2px solid #E74C3C;border-radius:12px;padding:20px 24px;margin-bottom:16px;color:#2D3436;">
        <strong style="font-size:16px;">⚠️ Backend API is offline</strong><br>
        <span style="font-size:14px;">Start the server: <code>python api.py</code></span>
    </div>
    """, unsafe_allow_html=True)
    if st.button("🔄 Retry Connection"):
        st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 9 — TABS
# ═══════════════════════════════════════════════════════════════════════════════

tab1, tab2, tab3 = st.tabs(["💬  Ask a Question", "⚖️  Compare Policies", "📊  Policy Dashboard"])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — ASK A QUESTION
# ═══════════════════════════════════════════════════════════════════════════════

with tab1:
    if not api_online and not st.session_state.get("last_answer"):
        # Empty state
        st.markdown("""
        <div style="text-align:center;padding:60px 20px;">
            <div style="font-size:80px;">🛡️</div>
            <h2 style="color:#0A2342;font-weight:800;">Welcome to InsureIQ</h2>
            <p style="color:#636E72;font-size:16px;">Start the backend API, then ask any coverage question.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        # ── Question Input Card ────────────────────────────────────

        # Handle triggered question from sidebar quick-question buttons
        if st.session_state.get("trigger_question"):
            st.session_state["q_input"] = st.session_state["trigger_question"]
            st.session_state["trigger_question"] = None

        question = st.text_area(
            "Ask anything about your coverage",
            placeholder="Ask anything about your coverage...\ne.g. Is emergency room treatment covered?",
            height=80,
            max_chars=500,
            key="q_input",
            label_visibility="collapsed",
        )

        lc, rc = st.columns([3, 1])
        with lc:
            st.caption(f"{len(question)}/500 characters")
        with rc:
            ask_clicked = st.button("🔍 Check Coverage", type="primary", use_container_width=True)

        # ── Execute Query ──────────────────────────────────────────
        result = st.session_state.get("last_answer")

        if ask_clicked and question and len(question.strip()) >= 3 and api_online:
            with st.spinner("🔍 Searching policies & analyzing coverage…"):
                try:
                    result = _do_query(question.strip(), st.session_state["settings"].get("top_k", 10))
                except Exception as e:
                    st.markdown(f"""<div class="iq-card" style="border-left:4px solid #F39C12;">
                        <strong>⚠️ Unable to process your question</strong>
                        <p style="color:#636E72;font-size:14px;">{e}</p>
                        <ul style="color:#636E72;font-size:13px;">
                            <li>Try rephrasing your question</li>
                            <li>Check that a policy is selected</li>
                            <li>Ensure the backend is running</li>
                        </ul>
                    </div>""", unsafe_allow_html=True)
                    result = None
        elif ask_clicked and (not question or len(question.strip()) < 3):
            st.warning("Please enter a coverage question (at least 3 characters).")
        elif ask_clicked and not api_online:
            st.error("Cannot reach the API. Start the backend first.")

        # ── Render Results ─────────────────────────────────────────
        if result:
            st.divider()
            _render_verdict(result)
            _render_stat_cards(result)
            _render_explanation(result)
            _render_caveats(result.get("caveats", []))
            _render_citations(result.get("citations", []), st.session_state["settings"]["auto_expand_citations"])

            # Action buttons
            bc1, bc2 = st.columns(2)
            with bc1:
                if st.button("🔁 Ask Follow-up", use_container_width=True):
                    st.session_state["trigger_question"] = ""
                    st.rerun()
            with bc2:
                if st.button("📤 Copy Summary", use_container_width=True):
                    summary = f"Q: {result.get('_meta',{}).get('question','')}\nA: {result.get('answer','')} ({int(result.get('confidence',0)*100)}%)\n{result.get('explanation','')}"
                    st.code(summary, language=None)
                    st.success("✅ Summary ready — copy from above")

            # Raw JSON debug
            if st.session_state["settings"]["show_raw_json"]:
                with st.expander("🔧 Raw JSON Response", expanded=False):
                    st.json(result)

            # Timing
            meta = result.get("_meta", {})
            ret_t = meta.get("retrieval_time_s", 0)
            gen_t = meta.get("generation_time_s", 0)
            total_t = meta.get("total_time_s", 0)
            st.markdown(f"""<div class="iq-footer" style="margin:1rem 0;border-radius:10px;">
                <span>⏱ Retrieval: {ret_t:.1f}s</span>
                <span>🤖 Generation: {gen_t:.1f}s</span>
                <span>⏱ Total: {total_t:.1f}s</span>
            </div>""", unsafe_allow_html=True)

            # Chat history
            _render_chat_history()

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — COMPARE POLICIES
# ═══════════════════════════════════════════════════════════════════════════════

with tab2:
    if not st.session_state.get("compare_mode"):
        st.markdown("""
        <div style="text-align:center;padding:60px 20px;">
            <div style="font-size:64px;">⚖️</div>
            <h3 style="color:#0A2342;">Enable Compare Mode</h3>
            <p style="color:#636E72;">Toggle <strong>⚖️ Compare Mode</strong> in the sidebar to compare coverage across multiple policies.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        cmp_policies = st.session_state.get("compare_policies", [])
        if not cmp_policies:
            st.info("Select up to 3 policies in the sidebar to compare.")
        else:
            cmp_q = st.text_input("Coverage question to compare", placeholder="e.g. Is knee replacement covered?", key="cmp_q")
            if st.button("⚖️ Compare Coverage", type="primary", use_container_width=True) and cmp_q and api_online:
                cols = st.columns(len(cmp_policies))
                for i, policy in enumerate(cmp_policies):
                    with cols[i]:
                        with st.spinner(f"Checking {_trunc(policy, 25)}"):
                            try:
                                res = APIClient.ask_question(cmp_q, top_k=10)
                                vc = _verdict_config(res.get("answer", "Unknown"))
                                conf = int(res.get("confidence", 0) * 100)
                                cites = len(res.get("citations", []))
                                caveats = res.get("caveats", [])
                                st.markdown(f"""<div class="iq-card" style="border-top:4px solid {vc['color']};">
                                    <div style="font-size:13px;color:#636E72;margin-bottom:8px;">📄 {_trunc(policy, 35)}</div>
                                    <div style="font-size:32px;text-align:center;">{vc['icon']}</div>
                                    <div style="text-align:center;font-weight:800;font-size:18px;color:{vc['color']};">{vc['label']}</div>
                                    <div style="text-align:center;font-size:28px;font-weight:900;color:{vc['color']};margin:8px 0;">{conf}%</div>
                                    <div style="font-size:13px;color:#636E72;text-align:center;">📎 {cites} citations</div>
                                    <div style="margin-top:8px;">{''.join(f'<span class="cav-pill" style="font-size:11px;">{_trunc(c, 50)}</span>' for c in caveats[:3])}</div>
                                </div>""", unsafe_allow_html=True)
                            except Exception as e:
                                st.error(f"Error: {e}")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — POLICY DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════

with tab3:
    stats = APIClient.get_stats()

    # ── Metric Row ─────────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    emb_count = stats.get("embedding_count", health.get("embeddings", 0))
    with m1:
        st.markdown(f"""<div class="stat-card stat-blue">
            <div style="font-size:36px;font-weight:900;color:#1B6CA8;">{emb_count:,}</div>
            <div class="stat-lbl">Total Chunks</div>
        </div>""", unsafe_allow_html=True)
    with m2:
        st.markdown(f"""<div class="stat-card stat-green">
            <div style="font-size:36px;font-weight:900;color:#2ECC71;">{len(policies)}</div>
            <div class="stat-lbl">Policies Loaded</div>
        </div>""", unsafe_allow_html=True)
    with m3:
        ollama_ok = stats.get("ollama_available", False)
        st.markdown(f"""<div class="stat-card stat-amber">
            <div style="font-size:36px;font-weight:900;color:{'#2ECC71' if ollama_ok else '#E74C3C'};">{'✅' if ollama_ok else '❌'}</div>
            <div class="stat-lbl">Ollama Status</div>
        </div>""", unsafe_allow_html=True)
    with m4:
        qc = st.session_state.get("query_count", 0)
        st.markdown(f"""<div class="stat-card stat-red">
            <div style="font-size:36px;font-weight:900;color:#0A2342;">{qc}</div>
            <div class="stat-lbl">Session Queries</div>
        </div>""", unsafe_allow_html=True)

    st.divider()

    # ── Two columns: model info + recent queries ───────────────────
    dl, dr = st.columns(2)

    with dl:
        st.markdown("#### 🤖 Model Information")
        st.markdown(f"""<div class="iq-card">
            <p><strong>Embedding Model:</strong> {stats.get('embedding_model', 'all-MiniLM-L6-v2')}</p>
            <p><strong>LLM Model:</strong> {stats.get('llm_model', 'llama3')}</p>
            <p><strong>Collection:</strong> {stats.get('collection_name', 'insurance_policies')}</p>
            <p><strong>Vector Count:</strong> {emb_count:,}</p>
        </div>""", unsafe_allow_html=True)

    with dr:
        st.markdown("#### 🕐 Recent Questions")
        history = st.session_state.get("chat_history", [])
        if history:
            for item in history[:5]:
                vc = _verdict_config(item["answer"].get("answer", "?"))
                st.markdown(f"""<div style="padding:8px 12px;border-left:3px solid {vc['color']};margin:6px 0;background:#FAFAFA;border-radius:0 8px 8px 0;">
                    <span style="font-size:13px;color:#2D3436;">{_trunc(item['question'], 50)}</span>
                    <span class="cite-badge" style="background:{vc['color']};color:white;padding:1px 8px;border-radius:8px;font-size:10px;float:right;">{vc['label']}</span>
                    <div style="font-size:11px;color:#636E72;">{item['timestamp']}</div>
                </div>""", unsafe_allow_html=True)
        else:
            st.markdown('<p style="color:#636E72;font-size:14px;">No queries yet this session.</p>', unsafe_allow_html=True)

    st.divider()

    # ── Policy List ────────────────────────────────────────────────
    st.markdown("#### 🗂️ Loaded Policies")
    if policies:
        for p in policies:
            st.markdown(f"""<div style="display:flex;align-items:center;justify-content:space-between;padding:10px 16px;background:white;border:1px solid #E0E0E0;border-radius:10px;margin:4px 0;">
                <span style="font-size:14px;color:#2D3436;">📄 {p}</span>
            </div>""", unsafe_allow_html=True)
    else:
        st.info("No policies loaded. Sync from Google Drive or check data/policies folder.")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 10 — FOOTER
# ═══════════════════════════════════════════════════════════════════════════════

_render_footer()
