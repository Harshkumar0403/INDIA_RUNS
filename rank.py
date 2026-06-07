"""
rank.py
=======
Main online ranking entry point.
Must complete in ≤5 min on CPU with ≤16GB RAM.
No network calls. Reads only pre-built artifacts.

Pipeline:
  1. Load all artifacts from disk (feature matrix, FAISS, KG features)
  2. Hard filter — eliminate disqualified candidates (~60-70%)
  3. FAISS retrieval — top-5000 semantically similar to JD
  4. Score matrix — 5-dim multiplicative scoring
  5. KG deep analysis — top-500 get causal arc scoring
  6. Final ranking — top-100 with reasoning strings
  7. Write submission.csv

Run:
    python rank.py
Output:
    submission.csv  (100 rows: candidate_id, rank, score, reasoning)
"""

import sys
import csv
import time
import numpy as np
import faiss
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    CANDIDATES_FILE, SAMPLE_FILE,
    FEATURE_MATRIX_FILE, CANDIDATE_IDS_FILE,
    FAISS_INDEX_FILE, KG_FEATURES_FILE,
    JD_EMBEDDING_FILE, SUBMISSION_FILE,
    FAISS as FAISS_CFG,
)
from utils import load_candidates, load_pickle
from hard_filter import apply_hard_filters, get_eligible_indices
from scorer import score_candidates, score_breakdown
from reasoning import generate_all_reasoning, calibrate_scores
from jd_parser import parse_jd, get_jd_context_for_reasoning


def load_artifacts(verbose: bool = True) -> dict:
    """Load all pre-built offline artifacts from disk."""
    if verbose:
        print("  Loading artifacts ...")

    missing = []
    for path in [FEATURE_MATRIX_FILE, CANDIDATE_IDS_FILE,
                 FAISS_INDEX_FILE, KG_FEATURES_FILE, JD_EMBEDDING_FILE]:
        if not path.exists():
            missing.append(str(path))

    if missing:
        print(f"\n  ERROR: Missing artifacts:")
        for m in missing:
            print(f"    {m}")
        print(f"\n  Run first:  python build_index.py")
        sys.exit(1)

    feature_matrix = load_pickle(FEATURE_MATRIX_FILE)
    candidate_ids  = load_pickle(CANDIDATE_IDS_FILE)
    faiss_index    = faiss.read_index(str(FAISS_INDEX_FILE))
    kg_features    = load_pickle(KG_FEATURES_FILE)
    jd_embedding   = np.load(str(JD_EMBEDDING_FILE))

    if verbose:
        print(f"  feature_matrix : {feature_matrix.shape}")
        print(f"  candidate_ids  : {len(candidate_ids):,}")
        print(f"  faiss_index    : {faiss_index.ntotal:,} vectors")
        print(f"  kg_features    : {len(kg_features):,} candidates")
        print(f"  jd_embedding   : {jd_embedding.shape}")

    return {
        "feature_matrix": feature_matrix,
        "candidate_ids":  candidate_ids,
        "faiss_index":    faiss_index,
        "kg_features":    kg_features,
        "jd_embedding":   jd_embedding,
    }


def rank_candidates(verbose: bool = True) -> list:
    """
    Full online ranking pipeline.
    Returns list of dicts: [{candidate_id, rank, score, reasoning}, ...]
    """
    t_start = time.time()

    # ── Load ──────────────────────────────────────────────────────
    artifacts = load_artifacts(verbose=verbose)
    fm        = artifacts["feature_matrix"]
    ids       = artifacts["candidate_ids"]
    index     = artifacts["faiss_index"]
    kg_feats  = artifacts["kg_features"]
    jd_vec    = artifacts["jd_embedding"]
    N         = len(ids)

    jd         = parse_jd()
    jd_context = get_jd_context_for_reasoning(jd)

    # ── Step 1: Hard filter ───────────────────────────────────────
    if verbose:
        print(f"\n  [1/5] Hard filter ({N:,} candidates) ...")
    eligible_mask, gate_log = apply_hard_filters(fm, ids, verbose=verbose)
    eligible_indices = get_eligible_indices(eligible_mask)
    n_eligible = len(eligible_indices)
    if verbose:
        print(f"  Eligible after gates: {n_eligible:,}")

    t1 = time.time()
    if verbose:
        print(f"  Hard filter time: {t1-t_start:.1f}s")

    # ── Step 2: FAISS retrieval ───────────────────────────────────
    if verbose:
        print(f"\n  [2/5] FAISS retrieval (top-{FAISS_CFG['top_k_retrieval']:,}) ...")

    query = jd_vec.reshape(1, -1).astype(np.float32)
    faiss.normalize_L2(query)

    # Search ALL candidates in FAISS, then filter to eligible
    k_search = min(FAISS_CFG["top_k_retrieval"] * 3, N)
    scores_raw, faiss_indices = index.search(query, k_search)
    scores_raw  = scores_raw[0]
    faiss_indices = faiss_indices[0]

    # Filter to only eligible candidates
    eligible_set = set(eligible_indices.tolist())
    filtered     = [
        (idx, float(sc))
        for idx, sc in zip(faiss_indices, scores_raw)
        if idx in eligible_set and idx >= 0
    ][:FAISS_CFG["top_k_retrieval"]]

    top_k_indices  = np.array([x[0] for x in filtered], dtype=np.int64)
    top_k_cosines  = np.array([x[1] for x in filtered], dtype=np.float32)
    n_retrieved    = len(top_k_indices)

    if verbose:
        print(f"  Retrieved: {n_retrieved:,} eligible candidates")
        t2 = time.time()
        print(f"  FAISS time: {t2-t1:.1f}s")

    # ── Step 3: KG scores for retrieved candidates ────────────────
    if verbose:
        print(f"\n  [3/5] KG arc scores ...")

    kg_score_arr = np.zeros(n_retrieved, dtype=np.float32)
    for j, idx in enumerate(top_k_indices):
        cid = ids[idx]
        kg  = kg_feats.get(cid, {})
        kg_score_arr[j] = float(kg.get("kg_score", 0.0))

    # ── Step 4: Scoring matrix ────────────────────────────────────
    if verbose:
        print(f"\n  [4/5] Scoring matrix ({n_retrieved:,} candidates) ...")

    final_scores = score_candidates(
        feature_matrix      = fm,
        faiss_cosine_scores = top_k_cosines,
        kg_scores           = kg_score_arr,
        candidate_indices   = top_k_indices,
    )

    # Sort by final score descending
    sort_order    = np.argsort(-final_scores)
    sorted_indices = top_k_indices[sort_order]
    sorted_scores  = final_scores[sort_order]
    sorted_cosines = top_k_cosines[sort_order]
    sorted_kg      = kg_score_arr[sort_order]

    if verbose:
        t3 = time.time()
        print(f"  Scoring time: {t3-(t2 if 't2' in dir() else t_start):.1f}s")
        print(f"  Top score: {sorted_scores[0]:.4f}")
        print(f"  Score at rank 100: {sorted_scores[min(99,len(sorted_scores)-1)]:.4f}")

    # ── Step 5: Top-100 with full breakdown + reasoning ───────────
    top_n   = min(FAISS_CFG["top_k_output"], len(sorted_indices))
    if verbose:
        print(f"\n  [5/5] Generating reasoning for top-{top_n} ...")

    # Load full candidates for reasoning (only top-100 needed)
    all_candidates = load_candidates(CANDIDATES_FILE, SAMPLE_FILE, verbose=False)
    cand_id_to_obj = {c.get("candidate_id",""): c for c in all_candidates}

    top_candidates_for_reasoning = []
    final_rows = []

    for rank, (fm_idx, score, cosine, kg_s) in enumerate(
        zip(sorted_indices[:top_n],
            sorted_scores[:top_n],
            sorted_cosines[:top_n],
            sorted_kg[:top_n]),
        start=1
    ):
        cid      = ids[fm_idx]
        cand_obj = cand_id_to_obj.get(cid, {})

        # Score breakdown for reasoning
        breakdown = score_breakdown(
            feature_matrix = fm,
            idx            = fm_idx,
            faiss_cosine   = float(cosine),
            kg_score       = float(kg_s),
        )

        top_candidates_for_reasoning.append((cand_obj, rank, breakdown))
        final_rows.append({
            "candidate_id": cid,
            "rank":         rank,
            "score":        round(float(score), 6),
            "breakdown":    breakdown,
            "cand_obj":     cand_obj,
        })

    # Generate reasoning strings
    reasoning_map = generate_all_reasoning(
        top_candidates_for_reasoning,
        jd_context=jd_context,
        use_t5=True,
    )

    # Attach reasoning to rows
    for row in final_rows:
        row["reasoning"] = reasoning_map.get(row["candidate_id"], "")

    # Calibrate scores to recruiter-readable range [0.50, 0.95]
    raw_scores    = [row["score"] for row in final_rows]
    cal_scores    = calibrate_scores(raw_scores)
    for row, cal in zip(final_rows, cal_scores):
        row["score"] = cal

    t_total = time.time() - t_start
    if verbose:
        print(f"\n  Total pipeline time: {t_total:.1f}s  ({t_total/60:.1f} min)")
        print(f"  (Constraint: ≤300s / 5 min)")
        if t_total > 300:
            print(f"  WARNING: Exceeded 5-minute limit!")

    return final_rows


def write_submission(final_rows: list, output_path: Path = None) -> Path:
    """Write submission CSV in the required format."""
    path = output_path or SUBMISSION_FILE

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["candidate_id", "rank", "score", "reasoning"]
        )
        writer.writeheader()
        for row in final_rows:
            writer.writerow({
                "candidate_id": row["candidate_id"],
                "rank":         row["rank"],
                "score":        row["score"],
                "reasoning":    row["reasoning"],
            })

    size_kb = path.stat().st_size / 1024
    print(f"\n  Submission written → {path}  ({size_kb:.1f} KB)")
    print(f"  Rows: {len(final_rows)}")
    return path


def print_top10_preview(final_rows: list) -> None:
    """Print a preview of the top-10 ranked candidates."""
    print(f"\n  Top-10 Preview:")
    print(f"  {'Rank':>4s}  {'CandidateID':>18s}  {'Score':>7s}  "
          f"{'Sem':>5s}  {'Str':>5s}  {'Skill':>5s}  {'Avail':>5s}  Title")
    print("  " + "-" * 100)

    for row in final_rows[:10]:
        bd    = row.get("breakdown", {})
        title = row["cand_obj"].get("profile", {}).get("current_title", "?")[:30]
        print(
            f"  {row['rank']:>4d}  {row['candidate_id']:>18s}  "
            f"{row['score']:>7.4f}  "
            f"{bd.get('semantic_score',0):>5.2f}  "
            f"{bd.get('structural_score',0):>5.2f}  "
            f"{bd.get('skill_idf_score',0):>5.2f}  "
            f"{bd.get('availability_mult',0):>5.2f}  "
            f"{title}"
        )
        print(f"         {row['reasoning'][:90]}")


def main():
    print("\n" + "=" * 60)
    print("  Redrob Ranking Pipeline — rank.py")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    final_rows = rank_candidates(verbose=True)
    print_top10_preview(final_rows)
    write_submission(final_rows)

    print(f"\n  Done. Submit: {SUBMISSION_FILE}")


if __name__ == "__main__":
    main()
