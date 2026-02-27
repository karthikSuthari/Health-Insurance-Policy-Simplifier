"""
components/answer_card.py
Reusable answer rendering for InsureIQ.
"""

import streamlit as st


def verdict_config(answer: str) -> dict:
    a = answer.strip().lower()
    _map = {
        "yes":     {"cls": "verdict-yes",     "icon": "✅", "label": "COVERED",       "color": "#2ECC71"},
        "no":      {"cls": "verdict-no",      "icon": "❌", "label": "NOT COVERED",   "color": "#E74C3C"},
        "partial": {"cls": "verdict-partial",  "icon": "⚠️", "label": "CONDITIONAL",  "color": "#F39C12"},
    }
    return _map.get(a, {"cls": "verdict-unknown", "icon": "❓", "label": "UNCLEAR", "color": "#636E72"})


def render_answer_card(result: dict, *, expanded_citations: bool = False):
    """Render a complete answer card with verdict, explanation, citations."""
    answer = result.get("answer", "Unknown")
    confidence = result.get("confidence", 0.0)
    explanation = result.get("explanation", "")
    citations = result.get("citations", [])
    caveats = result.get("caveats", [])
    vc = verdict_config(answer)
    pct = max(0, min(100, int(confidence * 100)))

    # Verdict
    st.markdown(f"""
    <div class="verdict-banner {vc['cls']}">
        <div style="display:flex;align-items:center;">
            <span class="verdict-icon">{vc['icon']}</span>
            <p class="verdict-text" style="color:{vc['color']}">{vc['label']}</p>
        </div>
        <div class="conf-circle" style="background:{vc['color']};">{pct}%</div>
    </div>
    """, unsafe_allow_html=True)

    # Explanation
    st.markdown(f"""<div class="iq-card">
        <h4 style="color:#0A2342;margin-top:0;">📋 Explanation</h4>
        <p style="font-size:15px;line-height:1.7;">{explanation}</p>
    </div>""", unsafe_allow_html=True)

    # Caveats
    if caveats:
        pills = "".join(f'<span class="cav-pill">⚠️ {c}</span>' for c in caveats if c)
        if pills:
            st.markdown(pills, unsafe_allow_html=True)

    # Citations
    if citations:
        with st.expander(f"📎 Citations ({len(citations)})", expanded=expanded_citations):
            for i, cit in enumerate(citations):
                fname = cit.get("filename", "?")
                page = cit.get("page", "?")
                quote = cit.get("quote", "")
                q_html = f'<div style="font-style:italic;font-size:13px;color:#444;margin-top:6px;">"{quote}"</div>' if quote else ""
                st.markdown(f"""<div class="cite-card cite-blue">
                    <strong>[{i+1}]</strong> {fname} — Page {page}{q_html}
                </div>""", unsafe_allow_html=True)
