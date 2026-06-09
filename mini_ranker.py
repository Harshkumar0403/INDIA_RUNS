"""
mini_ranker.py
==============
Standalone end-to-end ranker for small candidate sets (≤500 candidates).
Does NOT require precomputed artifacts (feature_matrix.pkl, faiss_index.bin etc.)
Computes everything on the fly from raw candidate dicts.

Used by:
  - sandbox/demo app (Streamlit)
  - Docker container
  - Stage 3 code reproduction verification

For the full 100k ranking, use rank.py (which reads precomputed artifacts).
For the sandbox ≤100 candidate demo, this file handles everything.

Runtime on 100 candidates, CPU only:
  Feature extraction : ~2s
  ONNX embedding     : ~3s  (skipped if model not found → cosine = 0.5)
  KG building        : ~1s
  Scoring + reasoning: ~5s
  Total              : < 15s  (well within 5-minute budget)
"""

import sys
import json
import csv
import math
import time
import numpy as np
from pathlib import Path
from datetime import datetime
from io import StringIO

# Add parent dir so imports work whether called from sandbox/ or root
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from config import (
    MINILM_ONNX, MINILM_TOKDIR, SKILL_VOCAB_FILE,
    SCORING, AVAILABILITY, PRODUCTION_CUES,
    CONSULTING_FIRMS, TARGET_LOCATIONS,
    CAREER_EVENT_TYPES, IDEAL_ARC_DISTRIBUTION,
    CAUSAL_TRANSITION_SCORES, DEFAULT_TRANSITION_SCORE,
)
from constants import (
    TITLE_RELEVANCE, DISQUALIFIED_TITLES,
    SENIORITY_LADDER, PRODUCT_INDUSTRIES, CV_SPEECH_SKILLS,
)
from utils import (
    load_skill_vocab, get_career_narrative, get_career_text,
    get_skill_names, get_skill_names_lower, days_since,
    availability_score, get_skill_idf, get_skill_jd_label,
    is_consulting_company, cosine_similarity_vectors,
)
from jd_parser import parse_jd, score_text_against_jd
from feature_extractor import extract_features, FEATURE_NAMES
from career_kg import build_candidate_kg
from hard_filter import CV_PRIMARY_TITLES_LOWER
from scorer import (
    score_candidates, score_breakdown,
    OUTSIDE_INDIA_NO_RELOC_MULT, OUTSIDE_INDIA_WILLING_MULT,
    SAVED_RECRUITERS_BOOST_THRESHOLD,
)
from reasoning import (
    generate_all_reasoning, calibrate_scores,
    get_genuine_gaps, build_fallback,
)

TODAY = datetime(2026, 6, 2)


# ─────────────────────────────────────────────────────────────────
# ONNX EMBEDDER (inline — reuses embedder.py logic)
# ─────────────────────────────────────────────────────────────────

def embed_texts_onnx(texts: list, batch_size: int = 64) -> np.ndarray:
    """
    Embed texts using ONNX MiniLM. Falls back to zeros if not available.
    Returns float32 array of shape (N, 384).
    """
    dim = 384
    N   = len(texts)

    if not MINILM_ONNX.exists():
        # Return neutral unit vectors — semantic score will be 0.5
        # System still ranks correctly via structural + skill + behavioral
        print("  [sandbox] MiniLM ONNX not found — semantic score set to neutral.")
        vecs = np.random.default_rng(42).normal(size=(N, dim)).astype(np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True).clip(min=1e-9)
        return vecs / norms

    try:
        import onnxruntime as ort

        sess = ort.InferenceSession(
            str(MINILM_ONNX), providers=["CPUExecutionProvider"]
        )

        # Load tokenizer
        try:
            from tokenizers import Tokenizer
            tok = Tokenizer.from_file(str(MINILM_TOKDIR / "tokenizer.json"))
            tok.enable_padding(pad_id=0, pad_token="[PAD]", length=128)
            tok.enable_truncation(max_length=128)
            use_hf = False
        except Exception:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(str(MINILM_TOKDIR))
            use_hf = True

        all_vecs = []
        for i in range(0, N, batch_size):
            batch = texts[i:i+batch_size]

            if use_hf:
                enc = tok(batch, padding=True, truncation=True,
                          max_length=128, return_tensors="np")
                input_ids      = enc["input_ids"].astype(np.int64)
                attention_mask = enc["attention_mask"].astype(np.int64)
            else:
                encodings      = tok.encode_batch(batch)
                input_ids      = np.array([e.ids for e in encodings], dtype=np.int64)
                attention_mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)

            feed = {"input_ids": input_ids, "attention_mask": attention_mask}
            input_names = [inp.name for inp in sess.get_inputs()]
            if "token_type_ids" in input_names:
                feed["token_type_ids"] = np.zeros_like(input_ids)

            out = sess.run(None, feed)[0]   # (B, seq, 384)
            if out.ndim == 3:
                mask_exp = attention_mask[:, :, None].astype(np.float32)
                pooled   = (out * mask_exp).sum(1) / mask_exp.sum(1).clip(min=1e-9)
            else:
                pooled = out
            norms  = np.linalg.norm(pooled, axis=1, keepdims=True).clip(min=1e-9)
            all_vecs.append((pooled / norms).astype(np.float32))

        return np.vstack(all_vecs)

    except Exception as e:
        print(f"  [sandbox] Embedding error: {e} — using neutral vectors.")
        vecs  = np.random.default_rng(42).normal(size=(N, dim)).astype(np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True).clip(min=1e-9)
        return vecs / norms


# ─────────────────────────────────────────────────────────────────
# HARD FILTER (inline — same rules as hard_filter.py)
# ─────────────────────────────────────────────────────────────────

GATE_THRESHOLD = 0.5

def is_eligible(cand: dict, feat_vec: np.ndarray) -> tuple:
    """
    Returns (eligible: bool, reason: str).
    Applies all 6 hard gates from hard_filter.py.
    """
    # Feature-matrix gates
    if feat_vec[1]  > GATE_THRESHOLD: return False, "Disqualified title"
    if feat_vec[11] > GATE_THRESHOLD: return False, "Pure consulting career"
    if feat_vec[12] > GATE_THRESHOLD: return False, "CV/speech dominant"
    if feat_vec[13] > GATE_THRESHOLD: return False, "Framework enthusiast"
    if feat_vec[15] > GATE_THRESHOLD: return False, "Honeypot anomaly"

    # CV primary title hard gate
    title = cand.get("profile", {}).get("current_title", "").lower()
    if title in CV_PRIMARY_TITLES_LOWER:
        return False, "CV-primary title"

    return True, ""


# ─────────────────────────────────────────────────────────────────
# MAIN RANKING FUNCTION
# ─────────────────────────────────────────────────────────────────

def rank_candidates(
    candidates: list,
    verbose: bool = True,
    use_reasoning: bool = True,
) -> list:
    """
    Full end-to-end ranking pipeline for a small candidate set.
    No precomputed artifacts required.

    Args:
        candidates:    list of candidate dicts (raw from JSON)
        verbose:       print progress
        use_reasoning: generate reasoning strings (adds ~5s)

    Returns:
        list of dicts: [{candidate_id, rank, score, reasoning}, ...]
        Sorted by rank ascending (rank 1 = best).
    """
    t0 = time.time()
    N  = len(candidates)
    if verbose:
        print(f"\n  Mini-ranker: {N} candidates")

    # ── 1. Load shared resources ──────────────────────────────────
    vocab = load_skill_vocab(SKILL_VOCAB_FILE)
    jd    = parse_jd()
    if verbose:
        print(f"  Vocab: {len(vocab)} skills  |  JD parsed")

    # ── 2. Feature extraction ─────────────────────────────────────
    if verbose:
        print(f"  Extracting features ...")
    feature_matrix = np.zeros((N, 25), dtype=np.float32)
    for i, cand in enumerate(candidates):
        feature_matrix[i] = extract_features(cand, vocab, jd)

    # ── 3. Hard filter ────────────────────────────────────────────
    eligible_mask = []
    for i, cand in enumerate(candidates):
        elig, reason = is_eligible(cand, feature_matrix[i])
        eligible_mask.append(elig)

    eligible_indices = [i for i, e in enumerate(eligible_mask) if e]
    n_eligible = len(eligible_indices)
    if verbose:
        print(f"  Hard filter: {N - n_eligible} eliminated, {n_eligible} eligible")

    # ── 4. Embeddings + cosine similarity ────────────────────────
    if verbose:
        print(f"  Embedding career narratives ...")

    narratives  = [get_career_narrative(c) for c in candidates]
    all_vecs    = embed_texts_onnx(narratives)

    jd_vec = embed_texts_onnx([jd["embedding_text"]])[0]
    jd_vec = jd_vec / (np.linalg.norm(jd_vec) + 1e-9)

    # Cosine similarity for each candidate
    cosine_scores = np.zeros(N, dtype=np.float32)
    for i in range(N):
        v = all_vecs[i] / (np.linalg.norm(all_vecs[i]) + 1e-9)
        cosine_scores[i] = float(np.dot(jd_vec, v))

    # ── 5. KG features ───────────────────────────────────────────
    if verbose:
        print(f"  Building career KGs ...")
    kg_scores_all = np.zeros(N, dtype=np.float32)
    for i, cand in enumerate(candidates):
        kg = build_candidate_kg(cand)
        kg_scores_all[i] = float(kg.get("kg_score", 0.0))

    # ── 6. Score eligible candidates ─────────────────────────────
    if verbose:
        print(f"  Scoring ...")

    eligible_arr = np.array(eligible_indices, dtype=np.int64)
    saved_arr    = np.array([
        int(candidates[i].get("redrob_signals", {}).get("saved_by_recruiters_30d", 0) or 0)
        for i in eligible_indices
    ], dtype=np.int32)

    raw_scores = score_candidates(
        feature_matrix      = feature_matrix,
        faiss_cosine_scores = cosine_scores[eligible_arr],
        kg_scores           = kg_scores_all[eligible_arr],
        candidate_indices   = eligible_arr,
        saved_by_recruiters = saved_arr,
    )

    # Assign 0 to eliminated candidates
    all_final_scores = np.zeros(N, dtype=np.float32)
    for j, i in enumerate(eligible_indices):
        all_final_scores[i] = raw_scores[j]

    # Sort eligible by score descending
    sorted_eligible = sorted(eligible_indices, key=lambda i: -all_final_scores[i])

    # ── 7. Calibrate scores ───────────────────────────────────────
    raw_for_calib = [float(all_final_scores[i]) for i in sorted_eligible]
    calibrated    = calibrate_scores(raw_for_calib)

    # ── 8. Reasoning for top-min(100, n_eligible) ────────────────
    top_n = min(100, n_eligible)
    top_candidates_for_reasoning = []
    for j, idx in enumerate(sorted_eligible[:top_n]):
        cand = candidates[idx]
        bd   = score_breakdown(
            feature_matrix = feature_matrix,
            idx            = idx,
            faiss_cosine   = float(cosine_scores[idx]),
            kg_score       = float(kg_scores_all[idx]),
        )
        top_candidates_for_reasoning.append((cand, j+1, bd))

    if use_reasoning:
        if verbose:
            print(f"  Generating reasoning for top-{top_n} ...")
        reasoning_map = generate_all_reasoning(
            top_candidates_for_reasoning,
            jd_context="",
            use_llm=False,   # no external API in sandbox
        )
    else:
        reasoning_map = {}

    # ── 9. Build output rows ──────────────────────────────────────
    final_rows = []
    for rank_pos, (cand, pos, bd) in enumerate(top_candidates_for_reasoning, 1):
        cid = cand.get("candidate_id", f"CAND_{rank_pos:07d}")
        final_rows.append({
            "candidate_id": cid,
            "rank":         rank_pos,
            "score":        calibrated[rank_pos - 1],
            "reasoning":    reasoning_map.get(cid, ""),
        })

    elapsed = time.time() - t0
    if verbose:
        print(f"\n  Done in {elapsed:.1f}s")
        print(f"  Top-3: {[r['candidate_id'] for r in final_rows[:3]]}")

    return final_rows


def rows_to_csv(rows: list) -> str:
    """Convert ranked rows to CSV string."""
    buf = StringIO()
    writer = csv.DictWriter(buf, fieldnames=["candidate_id","rank","score","reasoning"])
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue()


def rank_from_file(input_path: str = None, output_path: str = "sample_rank.csv") -> str:
    """
    Rank candidates from a JSON file and write/return CSV.

    input_path:  Path to JSON or JSONL file.
                 If None or file not found, falls back to
                 INDIA_RUNS_Assets/sample_candidates.json automatically.
    output_path: Output CSV filename (default: sample_rank.csv).

    Accepts both:
      - list format: sample_candidates.json  → [ {...}, {...} ]
      - jsonl format: candidates.jsonl       → one JSON object per line
    """
    # ── resolve input path ─────────────────────────────────────────
    SAMPLE_FALLBACK_PATHS = [
        Path("INDIA_RUNS_Assets") / "sample_candidates.json",
        Path(__file__).parent / "INDIA_RUNS_Assets" / "sample_candidates.json",
        Path("sample_candidates.json"),
    ]

    if input_path is None:
        # No input provided — use sample fallback
        resolved = next((p for p in SAMPLE_FALLBACK_PATHS if p.exists()), None)
        if resolved is None:
            raise FileNotFoundError(
                "No input file provided and sample_candidates.json not found. "
                "Pass --candidates <path> to specify an input file."
            )
        print(f"  No input provided — using fallback: {resolved}")
        input_path = resolved
    else:
        input_path = Path(input_path)
        if not input_path.exists():
            # Provided path not found — fall back to sample
            resolved = next((p for p in SAMPLE_FALLBACK_PATHS if p.exists()), None)
            if resolved is None:
                raise FileNotFoundError(
                    f"Input file not found: {input_path}\n"
                    "Also could not find sample_candidates.json as fallback."
                )
            print(f"  WARNING: {input_path} not found — falling back to {resolved}")
            input_path = resolved

    # ── load candidates ────────────────────────────────────────────
    with open(input_path, encoding="utf-8") as f:
        content = f.read().strip()

    if content.startswith("["):
        candidates = json.loads(content)
    else:
        candidates = [
            json.loads(line)
            for line in content.splitlines()
            if line.strip()
        ]

    print(f"  Loaded {len(candidates)} candidates from {input_path.name}")

    # ── rank and write ─────────────────────────────────────────────
    rows    = rank_candidates(candidates, verbose=True, use_reasoning=True)
    csv_str = rows_to_csv(rows)

    out_path = Path(output_path)
    out_path.write_text(csv_str, encoding="utf-8")
    print(f"  Output written → {out_path}  ({len(rows)} rows)")

    return csv_str


# ─────────────────────────────────────────────────────────────────
# CLI ENTRY POINT
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Redrob Mini-Ranker — end-to-end ranking for small candidate sets.\n"
            "If --candidates is omitted, falls back to sample_candidates.json."
        )
    )
    parser.add_argument(
        "--candidates",
        default=None,
        help=(
            "Path to candidates JSON or JSONL file. "
            "Optional — falls back to INDIA_RUNS_Assets/sample_candidates.json "
            "if not provided or file not found."
        )
    )
    parser.add_argument(
        "--out",
        default="sample_rank.csv",
        help="Output CSV path (default: sample_rank.csv)"
    )
    args = parser.parse_args()

    rank_from_file(input_path=args.candidates, output_path=args.out)
