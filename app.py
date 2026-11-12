"""VibeMatch Agent - Streamlit UI.

    streamlit run app.py      (binds to localhost via .streamlit/config.toml)
"""
import pandas as pd
import streamlit as st

from src.agent import VibeMatchAgent, configure_logging
from src.feedback import apply_feedback
from src.recommender import MODES

configure_logging()
st.set_page_config(page_title="VibeMatch Agent", page_icon="🎵", layout="wide")

ss = st.session_state
ss.setdefault("weights", MODES["balanced"])
ss.setdefault("mode", "balanced")
ss.setdefault("excluded", [])
ss.setdefault("feedback_log", [])
ss.setdefault("result", None)

with st.sidebar:
    st.header("Settings")
    mode = st.selectbox("Scoring mode", list(MODES), index=list(MODES).index(ss.mode))
    if mode != ss.mode:
        ss.mode, ss.weights = mode, MODES[mode]
    use_rag = st.toggle("Use retrieval (RAG)", value=True,
                        help="Off = baseline that only understands the catalog's exact labels.")
    st.caption("Current weights (change with 👍/👎):")
    st.json({k: round(v, 2) for k, v in ss.weights.__dict__.items()}, expanded=False)
    if ss.excluded:
        st.caption(f"Excluded song ids: {ss.excluded}")
    if st.button("Reset feedback"):
        ss.weights, ss.excluded, ss.feedback_log = MODES[ss.mode], [], []
        st.rerun()

st.title("🎵 VibeMatch Agent")
st.caption("Describe what you want to hear. The agent retrieves listening guidance, builds a playlist "
           "from the 60-song catalog, checks it, repairs what fails, and explains every pick.")

examples = ["rainy sunday study session, nothing with lyrics, about 40 minutes, start calm and build up a little",
            "gym workout, warm up then go hard then cool down, 30 minutes",
            "something sad", "bossa nova to relax"]
pick = st.selectbox("Try an example", ["(type your own)"] + examples)
request = st.text_area("Request", value="" if pick == "(type your own)" else pick, height=80)


def run():
    agent = VibeMatchAgent(mode=ss.weights, use_rag=use_rag, exclude_ids=ss.excluded, save_trace=True)
    ss.result = agent.run(request)


if st.button("Build playlist", type="primary"):
    run()

r = ss.result
if r is None:
    st.stop()

if r.status in ("refused", "error"):
    st.error(r.message)
    st.stop()
if r.status == "needs_clarification":
    st.info(f"🤔 {r.message}")
    st.stop()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Songs", len(r.items))
c2.metric("Length", f"{r.total_minutes} min")
c3.metric("Confidence", f"{r.confidence:.2f}")
c4.metric("Repair rounds", r.rounds)
if r.needs_review:
    st.warning("Needs review: confidence is low or a check failed. Look at the checks below before using this playlist.")
st.write(r.intro)
for w in r.warnings:
    st.caption(f"ℹ️ {w}")

st.subheader("Playlist")
for it in r.items:
    s = it.song
    cols = st.columns([6, 1, 1])
    with cols[0].expander(f"{it.slot}. **{s['title']}** - {s['artist']}  ·  {s['genre']}/{s['mood']}  ·  "
                          f"energy {s['energy']:.2f}  ·  {it.segment}"):
        st.write("Why: " + "; ".join(it.reasons))
        if it.citations:
            st.write("Sources: " + ", ".join(f"`{c}`" for c in it.citations))
    if cols[1].button("👍", key=f"up{s['id']}"):
        ss.weights, note = apply_feedback(ss.weights, r.profile, s, liked=True)
        ss.feedback_log.append(note)
        st.rerun()
    if cols[2].button("👎", key=f"down{s['id']}"):
        ss.weights, note = apply_feedback(ss.weights, r.profile, s, liked=False)
        ss.excluded.append(s["id"])
        ss.feedback_log.append(note)
        st.rerun()

if ss.feedback_log:
    st.subheader("Your feedback")
    for note in ss.feedback_log:
        st.write("- " + note)
    if st.button("Rebuild with my feedback"):
        run()
        st.rerun()

st.subheader("Checks")
st.dataframe(pd.DataFrame([{"check": c.name, "result": "pass" if c.passed else "FAIL", "detail": c.detail}
                           for c in r.checks]), hide_index=True, use_container_width=True)

with st.expander("Agent trace (every step, for debugging and grading)"):
    st.caption(f"LLM: {r.llm}. Saved to {r.trace_path}")
    st.json(r.trace)
