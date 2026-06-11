import sys
import json
import time
import subprocess
import tempfile
from io import StringIO
from pathlib import Path
from functools import lru_cache

import pandas as pd
import gradio as gr
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))


# ============================================================
# CACHED RANKER
# ============================================================
@lru_cache(maxsize=1)
def load_ranker():
    from mini_ranker import rank_candidates, rows_to_csv
    return rank_candidates, rows_to_csv


# ============================================================
# SAMPLE LOADER
# ============================================================
def load_sample_candidates():
    paths = [
        ROOT / "INDIA_RUNS_Assets" / "sample_candidates.json",
        Path("INDIA_RUNS_Assets") / "sample_candidates.json",
        Path("sample_candidates.json"),
    ]

    for p in paths:
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))

    return None


# ============================================================
# TEXT CLEANER
# ============================================================
def clean_text(text: str):
    return (
        str(text)
        .replace("\u2014", "--")
        .replace("\u2013", "-")
        .replace("\u2192", "->")
        .replace("\u2019", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
    )


# ============================================================
# CSV CONVERTER
# ============================================================
def rows_to_csv_clean(rows):

    import csv

    buf = StringIO()

    writer = csv.DictWriter(
        buf,
        fieldnames=["candidate_id", "rank", "score", "reasoning"]
    )

    writer.writeheader()

    for r in rows:
        writer.writerow({
            "candidate_id": r["candidate_id"],
            "rank": r["rank"],
            "score": r["score"],
            "reasoning": clean_text(r.get("reasoning", ""))
        })

    return buf.getvalue()


# ============================================================
# DISPLAY DATAFRAME
# ============================================================
def build_display_df(
        rows,
        min_score,
        top_n_display,
        show_reasoning,
        score_decimals):

    fmt = f"{{:.{score_decimals}f}}"

    data = []

    for r in rows:

        if float(r["score"]) < min_score:
            continue

        row = {
            "Rank": int(r["rank"]),
            "Candidate ID": r["candidate_id"],
            "Score": fmt.format(float(r["score"]))
        }

        if show_reasoning:
            txt = clean_text(r.get("reasoning", ""))

            row["Reasoning"] = (
                txt[:220] + "..."
                if len(txt) > 220 else txt
            )

        data.append(row)

    return pd.DataFrame(data[:top_n_display])


# ============================================================
# SCORE CHART
# ============================================================
def score_chart(rows):

    scores = [float(r["score"]) for r in rows]

    fig, ax = plt.subplots(figsize=(7, 3))

    ax.hist(scores, bins=20)

    ax.set_xlabel("Score")
    ax.set_ylabel("Count")
    ax.set_title("Score Distribution")

    return fig


# ============================================================
# RANKING FUNCTION
# ============================================================
def run_ranking(candidates):

    rank_fn, _ = load_ranker()

    t0 = time.time()

    rows = rank_fn(
        candidates,
        verbose=True,
        use_reasoning=True
    )

    elapsed = time.time() - t0

    csv_str = rows_to_csv_clean(rows)

    return rows, csv_str, elapsed


# ============================================================
# UPLOAD HANDLER
# ============================================================
def upload_rank(
        file,
        min_score,
        top_n_display,
        show_reasoning,
        score_decimals):

    if file is None:
        return None, None, None, "Please upload a file."

    content = Path(file.name).read_text(
        encoding="utf-8"
    ).strip()

    if content.startswith("["):
        candidates = json.loads(content)
    else:
        candidates = [
            json.loads(x)
            for x in content.splitlines()
            if x.strip()
        ]

    if len(candidates) > 500:
        candidates = candidates[:500]

    rows, csv_str, elapsed = run_ranking(candidates)

    df = build_display_df(
        rows,
        min_score,
        top_n_display,
        show_reasoning,
        score_decimals
    )

    fig = score_chart(rows)

    tmp_csv = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".csv"
    )

    tmp_csv.write(csv_str.encode())
    tmp_csv.close()

    metrics = f"""
### ✅ Ranking Complete

- Candidates ranked: **{len(rows)}**
- Top score: **{max(float(r['score']) for r in rows):.4f}**
- Processing time: **{elapsed:.1f}s**
"""

    return df, fig, tmp_csv.name, metrics


# ============================================================
# CUSTOM CSS
# ============================================================
css = """
body {
    background: #0f172a;
}

.gradio-container {
    max-width: 1500px !important;
}

.header {
    background: linear-gradient(
        135deg,
        #1a1a2e 0%,
        #16213e 50%,
        #0f3460 100%
    );

    padding: 30px;
    border-radius: 15px;
}
"""


# ============================================================
# UI
# ============================================================
with gr.Blocks(css=css, title="ArcRank") as demo:

    gr.HTML("""
    <div class="header">
        <h1 style="color:white;">
            🕸️ ArcRank
        </h1>

        <h3 style="color:#cbd5e1;">
            Intelligent Hiring via Career Arc Analysis
        </h3>
    </div>
    """)

    with gr.Row():

        with gr.Column(scale=1):

            min_score = gr.Slider(
                0,
                1,
                value=0,
                step=0.05,
                label="Minimum Score"
            )

            top_n_display = gr.Dropdown(
                [10, 25, 50, 100],
                value=10,
                label="Rows to Display"
            )

            show_reasoning = gr.Checkbox(
                value=True,
                label="Show Reasoning"
            )

            score_decimals = gr.Dropdown(
                [2, 4, 6],
                value=4,
                label="Decimal Places"
            )

        with gr.Column(scale=4):

            with gr.Tab("📂 Upload Candidates"):

                gr.Markdown("""
Upload JSON or JSONL candidate files.

Recommended size:
50–200 candidates.
""")

                upload_file = gr.File(
                    file_types=[".json", ".jsonl"]
                )

                rank_btn = gr.Button(
                    "🚀 Rank Candidates",
                    variant="primary"
                )

                metrics_md = gr.Markdown()

                result_df = gr.DataFrame()

                score_plot = gr.Plot()

                csv_output = gr.File()

                rank_btn.click(
                    fn=upload_rank,
                    inputs=[
                        upload_file,
                        min_score,
                        top_n_display,
                        show_reasoning,
                        score_decimals
                    ],
                    outputs=[
                        result_df,
                        score_plot,
                        csv_output,
                        metrics_md
                    ]
                )            # ====================================================
            # TAB 2 — LIVE DEMO
            # ====================================================

            with gr.Tab("🎬 Live Demo"):

                gr.Markdown("""
Run ArcRank on the bundled sample candidates.
""")

                demo_btn = gr.Button(
                    "▶️ Run Demo",
                    variant="secondary"
                )

                demo_metrics = gr.Markdown()

                demo_df = gr.DataFrame()

                demo_plot = gr.Plot()

                demo_csv = gr.File()


                def run_demo(
                        min_score,
                        top_n_display,
                        show_reasoning,
                        score_decimals):

                    candidates = load_sample_candidates()

                    if candidates is None:
                        return None, None, None, \
                            "sample_candidates.json not found."

                    rows, csv_str, elapsed = run_ranking(candidates)

                    df = build_display_df(
                        rows,
                        min_score,
                        top_n_display,
                        show_reasoning,
                        score_decimals
                    )

                    fig = score_chart(rows)

                    tmp_csv = tempfile.NamedTemporaryFile(
                        delete=False,
                        suffix=".csv"
                    )

                    tmp_csv.write(csv_str.encode())
                    tmp_csv.close()

                    metrics = f"""
### ✅ Demo Complete

- Candidates ranked: **{len(rows)}**
- Top score: **{max(float(r['score']) for r in rows):.4f}**
- Processing time: **{elapsed:.1f}s**
"""

                    return df, fig, tmp_csv.name, metrics


                demo_btn.click(
                    fn=run_demo,
                    inputs=[
                        min_score,
                        top_n_display,
                        show_reasoning,
                        score_decimals
                    ],
                    outputs=[
                        demo_df,
                        demo_plot,
                        demo_csv,
                        demo_metrics
                    ]
                )


            # ====================================================
            # TAB 3 — FULL RANKING
            # ====================================================

            with gr.Tab("🏆 Full Ranking"):

                gr.Markdown("""
Run rank.py on the complete dataset.
Requires prebuilt artifacts.
""")

                full_btn = gr.Button(
                    "⚡ Run Full Ranking",
                    variant="primary"
                )

                full_logs = gr.Textbox(
                    label="Logs",
                    lines=15
                )

                full_csv = gr.File()


                def run_full():

                    t0 = time.time()

                    try:

                        result = subprocess.run(
                            [sys.executable, str(ROOT / "rank.py")],
                            capture_output=True,
                            text=True,
                            cwd=str(ROOT)
                        )

                        elapsed = time.time() - t0

                        if result.returncode != 0:

                            return (
                                result.stderr,
                                None
                            )

                        output_csv = ROOT / "Harsh_0403.csv"

                        logs = f"""
Completed in {elapsed:.1f} sec

{result.stdout[-3000:]}
"""

                        return logs, str(output_csv)

                    except Exception as e:

                        return str(e), None


                full_btn.click(
                    fn=run_full,
                    outputs=[
                        full_logs,
                        full_csv
                    ]
                )


    # ============================================================
    # FOOTER
    # ============================================================

    gr.HTML("""
    <hr>

    <center>

    <p style="color:#94a3b8;">

    🕸️ ArcRank —
    Intelligent Hiring via Career Arc Analysis

    <br><br>

    Event-Centric KG · ONNX MiniLM-L6-v2 ·
    CPU-only · No external API calls

    </p>

    </center>
    """)


# ================================================================
# MAIN
# ================================================================

if __name__ == "__main__":

    demo.launch(share=True)
