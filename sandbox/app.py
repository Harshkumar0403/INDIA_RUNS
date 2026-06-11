"""
sandbox/app.py — ArcRank
=========================
Streamlit application for the ArcRank intelligent hiring system.

Three sections:
  1. Upload Your Candidates  — user uploads JSON, gets ranked CSV + table
  2. Live Demo               — pre-loaded sample_candidates.json
  3. Full 100k Ranking       — triggers rank.py on the complete dataset

Deploy:
  streamlit run sandbox/app.py
"""

import sys
import json
import time
import subprocess
import pandas as pd
import streamlit as st
from pathlib import Path
from io import StringIO

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# ── page config ───────────────────────────────────────────────────
st.set_page_config(
    page_title="ArcRank — Intelligent Hiring",
    page_icon="🕸️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── custom CSS ────────────────────────────────────────────────────
st.markdown("""
<style>
    .arcrank-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem 2.5rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
    }
    .arcrank-title {
        font-size: 2.6rem;
        font-weight: 800;
        color: #e2e8f0;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .arcrank-subtitle {
        font-size: 1.05rem;
        color: #94a3b8;
        margin-top: 0.4rem;
    }
    .section-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }
    .metric-pill {
        display: inline-block;
        background: #dbeafe;
        color: #1e40af;
        border-radius: 20px;
        padding: 2px 12px;
        font-size: 0.8rem;
        font-weight: 600;
        margin: 2px;
    }
    .tag-green  { background:#dcfce7; color:#166534; }
    .tag-purple { background:#f3e8ff; color:#6b21a8; }
    .tag-orange { background:#ffedd5; color:#9a3412; }
    div[data-testid="stHorizontalBlock"] { gap: 1.5rem; }
</style>
""", unsafe_allow_html=True)

# ── header ────────────────────────────────────────────────────────
st.markdown("""
<div class="arcrank-header">
  <div class="arcrank-title">🕸️ ArcRank</div>
  <div class="arcrank-subtitle">Intelligent Hiring via Career Arc Analysis</div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
**ArcRank** ranks candidates the way a senior technical recruiter would —
by understanding career narratives, not matching keywords.

Built on an **Event-Centric Knowledge Graph** framework (adapted from NLP narrative research),
ArcRank models each candidate's career as a sequence of typed events with causal progression logic.
A Staff ML Engineer whose arc shows `Data Engineer → ML Engineer → Tech Lead at product companies`
scores higher than someone with identical keywords but a disconnected or consulting-heavy career.

<span class="metric-pill">25-dim feature matrix</span>
<span class="metric-pill tag-green">IDF-weighted skill scoring</span>
<span class="metric-pill tag-purple">Career KG arc alignment</span>
<span class="metric-pill tag-orange">Behavioral availability gates</span>
<span class="metric-pill">CPU-only · No API calls · &lt;5 min</span>
""", unsafe_allow_html=True)

st.markdown("---")

# ── sidebar: filters ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🎛️ Display Filters")
    st.caption("Applied to results tables after ranking.")

    min_score = st.slider(
        "Minimum score", 0.0, 1.0, 0.0, 0.05,
        help="Hide candidates below this score threshold"
    )
    top_n_display = st.selectbox(
        "Rows to display",
        [10, 25, 50, 100],
        index=0,
        help="Number of top candidates shown in the table"
    )
    show_reasoning = st.toggle(
        "Show reasoning column", value=True,
        help="Toggle reasoning text in the results table"
    )
    score_decimals = st.selectbox(
        "Score decimal places", [2, 4, 6], index=1
    )

    st.markdown("---")
    st.markdown("### 📊 Score Legend")
    st.markdown("""
| Range | Signal |
|-------|--------|
| 0.90 – 0.95 | Excellent match |
| 0.80 – 0.89 | Strong match |
| 0.70 – 0.79 | Good match |
| 0.60 – 0.69 | Moderate match |
| 0.50 – 0.59 | Weak match |
    """)

    st.markdown("---")
    st.caption("ArcRank · IIT Guwahati · 2026")


# ── shared helpers ────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading ArcRank models...")
def load_ranker():
    from mini_ranker import rank_candidates, rows_to_csv
    return rank_candidates, rows_to_csv

def load_sample_candidates():
    for p in [
        ROOT / "INDIA_RUNS_Assets" / "sample_candidates.json",
        Path("INDIA_RUNS_Assets") / "sample_candidates.json",
        Path("sample_candidates.json"),
    ]:
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    return None

def clean_text(text: str) -> str:
    return (str(text)
            .replace("\u2014", "--").replace("\u2013", "-")
            .replace("\u2192", "->").replace("\u2019", "'")
            .replace("\u201c", '"').replace("\u201d", '"'))

def rows_to_csv_clean(rows: list) -> str:
    import csv
    buf = StringIO()
    w   = csv.DictWriter(buf, fieldnames=["candidate_id","rank","score","reasoning"])
    w.writeheader()
    for r in rows:
        w.writerow({
            "candidate_id": r["candidate_id"],
            "rank":         r["rank"],
            "score":        r["score"],
            "reasoning":    clean_text(r.get("reasoning","")),
        })
    return buf.getvalue()

def build_display_df(rows: list) -> pd.DataFrame:
    """Build filtered, display-ready DataFrame from ranked rows."""
    fmt = f"{{:.{score_decimals}f}}"
    cols = ["Rank", "Candidate ID", "Score"]
    if show_reasoning:
        cols.append("Reasoning")

    data = []
    for r in rows:
        if float(r["score"]) < min_score:
            continue
        row = {
            "Rank":         int(r["rank"]),
            "Candidate ID": r["candidate_id"],
            "Score":        fmt.format(float(r["score"])),
        }
        if show_reasoning:
            raw = clean_text(r.get("reasoning", ""))
            row["Reasoning"] = raw[:220] + ("..." if len(raw) > 220 else "")
        data.append(row)
    return pd.DataFrame(data[:top_n_display])

def score_chart(rows: list):
    """Bar chart of score distribution."""
    scores = [float(r["score"]) for r in rows]
    df = pd.DataFrame({"Score": scores})
    st.bar_chart(df, y="Score", use_container_width=True, height=220)

def run_ranking_with_ui(candidates: list) -> tuple:
    """Run mini-ranker with live progress display."""
    rank_fn, _ = load_ranker()

    progress = st.progress(0, text="Initialising pipeline...")
    log_box  = st.empty()
    logs     = []

    import builtins
    _orig = builtins.print
    def _capture(*args, **kwargs):
        msg = " ".join(str(a) for a in args).strip()
        if msg:
            logs.append(msg)
            log_box.code("\n".join(logs[-6:]), language=None)
    builtins.print = _capture

    t0 = time.time()
    try:
        progress.progress(15, text="Extracting features & building KGs...")
        rows = rank_fn(candidates, verbose=True, use_reasoning=True)
        progress.progress(95, text="Generating CSV...")
        csv_str = rows_to_csv_clean(rows)
        progress.progress(100, text="Done!")
    finally:
        builtins.print = _orig

    elapsed = time.time() - t0
    log_box.empty()
    progress.empty()
    return rows, csv_str, elapsed

def show_results(rows: list, csv_str: str, elapsed: float,
                 source_label: str, dl_filename: str):
    """Common results block: metrics, download, table, chart."""
    # ── metrics row ──────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Candidates ranked",  len(rows))
    m2.metric("Top score",          f"{max(float(r['score']) for r in rows):.4f}")
    m3.metric("Processing time",    f"{elapsed:.1f}s")
    eligible = sum(1 for r in rows if float(r["score"]) >= 0.60)
    m4.metric("Strong matches (≥0.60)", eligible)

    # ── download ─────────────────────────────────────────────────
    st.download_button(
        label     = f"⬇️ Download {dl_filename}",
        data      = csv_str,
        file_name = dl_filename,
        mime      = "text/csv",
        use_container_width=False,
    )

    # ── table ─────────────────────────────────────────────────────
    df = build_display_df(rows)
    if df.empty:
        st.info("No candidates meet the current filter thresholds. "
                "Lower the minimum score in the sidebar.")
    else:
        col_cfg = {
            "Rank":         st.column_config.NumberColumn(width="small"),
            "Candidate ID": st.column_config.TextColumn(width="medium"),
            "Score":        st.column_config.TextColumn(width="small"),
        }
        if show_reasoning:
            col_cfg["Reasoning"] = st.column_config.TextColumn(width="large")
        st.dataframe(df, use_container_width=True, hide_index=True,
                     column_config=col_cfg)

    # ── chart ─────────────────────────────────────────────────────
    with st.expander("📊 Score distribution", expanded=False):
        score_chart(rows)


# ══════════════════════════════════════════════════════════════════
# SECTION 1 — Upload Your Candidates
# ══════════════════════════════════════════════════════════════════
st.markdown("## 📂 Section 1 — Upload Your Candidates")
st.markdown("""
Upload a JSON file (array format) or JSONL file following the standard candidate schema.
ArcRank will extract features, build career KGs, compute semantic similarity,
and return a ranked shortlist with grounded reasoning.
*Recommended: 50–200 candidates for best speed. Processing takes ~10–30 seconds on CPU.*
""")

with st.container():
    uploaded_file = st.file_uploader(
        "Drop your candidates file here",
        type=["json", "jsonl"],
        help="JSON array [ {...}, {...} ] or JSONL (one candidate per line). "
             "Must follow the standard candidate schema."
    )

    u_col1, u_col2 = st.columns([2, 1])
    with u_col1:
        rank_btn = st.button(
            "🚀 Rank Candidates",
            disabled=(uploaded_file is None),
            use_container_width=True,
            type="primary",
        )
    with u_col2:
        if uploaded_file is None:
            st.caption("← Upload a file first")

if rank_btn and uploaded_file:
    content = uploaded_file.read().decode("utf-8").strip()
    if content.startswith("["):
        candidates = json.loads(content)
    else:
        candidates = [json.loads(l) for l in content.splitlines() if l.strip()]

    if len(candidates) > 500:
        st.warning(f"Large file: {len(candidates)} candidates — truncating to 500 for speed.")
        candidates = candidates[:500]

    st.info(f"Loaded **{len(candidates)}** candidates from `{uploaded_file.name}`")

    with st.spinner("Running ArcRank pipeline..."):
        rows, csv_str, elapsed = run_ranking_with_ui(candidates)

    st.success(f"✅ Ranked **{len(rows)}** candidates in **{elapsed:.1f}s**")
    show_results(rows, csv_str, elapsed,
                 source_label=uploaded_file.name,
                 dl_filename="arcrank_results.csv")

st.markdown("---")

# ══════════════════════════════════════════════════════════════════
# SECTION 2 — Live Demo
# ══════════════════════════════════════════════════════════════════
st.markdown("## 🎬 Section 2 — Live Demo")
st.markdown("""
See ArcRank in action on a pre-loaded sample of 50 candidates from the hackathon dataset.
This sample intentionally includes irrelevant profiles (HR managers, accountants, mechanical engineers)
alongside genuinely qualified ML/AI engineers — demonstrating how ArcRank cuts through the noise.
*Click the button to run the full pipeline end-to-end in seconds.*
""")

demo_btn = st.button(
    "▶️ Run Demo",
    use_container_width=False,
    type="secondary",
)

if demo_btn:
    sample_cands = load_sample_candidates()
    if sample_cands is None:
        st.error("sample_candidates.json not found. "
                 "Make sure INDIA_RUNS_Assets/sample_candidates.json is present.")
    else:
        st.info(f"Loaded **{len(sample_cands)}** candidates from `sample_candidates.json`")

        with st.spinner("Running ArcRank pipeline on sample data..."):
            rows, csv_str, elapsed = run_ranking_with_ui(sample_cands)

        st.success(f"✅ Demo complete — ranked **{len(rows)}** candidates in **{elapsed:.1f}s**")

        # Show eliminated count too
        eliminated = len(sample_cands) - len(rows)
        if eliminated > 0:
            st.caption(
                f"ℹ️ {eliminated} candidates eliminated by hard gates "
                f"(disqualified title, consulting-only, CV/speech-domain, honeypot)"
            )

        show_results(rows, csv_str, elapsed,
                     source_label="sample_candidates.json",
                     dl_filename="sample_rank.csv")

st.markdown("---")



# ── footer ────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center; color:#94a3b8; font-size:0.82em; padding: 1rem 0;'>
    🕸️ <strong>ArcRank</strong> — Intelligent Hiring via Career Arc Analysis &nbsp;·&nbsp;
    Event-Centric KG &nbsp;·&nbsp; ONNX MiniLM-L6-v2 &nbsp;·&nbsp;
    CPU-only &nbsp;·&nbsp; No external API calls
</div>
""", unsafe_allow_html=True)
