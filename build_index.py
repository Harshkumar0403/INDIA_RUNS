"""
build_index.py
==============
Offline pipeline orchestrator. Run ONCE before ranking.
Calls feature_extractor, career_kg, and embedder in sequence.
All outputs cached to artifacts/ directory.

Steps:
  1. Extract 25-dim feature matrix for all candidates
  2. Build career KG features (event nodes + causal edges)
  3. Embed career narratives → FAISS index
  4. Embed JD → query vector

Total runtime on 100k candidates (CPU):
  Features: ~15 min
  KG:       ~10 min
  Embeddings (ONNX): ~25 min
  Total:    ~50 min

Run:
    python build_index.py
    python build_index.py --step features   # only step 1
    python build_index.py --step kg         # only step 2
    python build_index.py --step embed      # only step 3
"""

import sys
import time
import argparse
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    CANDIDATES_FILE, SAMPLE_FILE,
    FEATURE_MATRIX_FILE, CANDIDATE_IDS_FILE,
    KG_FEATURES_FILE, FAISS_INDEX_FILE,
    JD_EMBEDDING_FILE, ROLE_EMBEDDINGS_FILE,
    SKILL_VOCAB_FILE,
)
from utils import load_candidates

def step_features(candidates):
    from feature_extractor import build_feature_matrix, report_matrix_stats
    from jd_parser import parse_jd
    from utils import load_skill_vocab, save_pickle

    print("\n[Step 1] Building feature matrix ...")
    vocab  = load_skill_vocab(SKILL_VOCAB_FILE)
    jd     = parse_jd()
    matrix, ids = build_feature_matrix(candidates, vocab, jd, verbose=True)
    report_matrix_stats(matrix, ids)
    save_pickle(matrix, FEATURE_MATRIX_FILE)
    save_pickle(ids,    CANDIDATE_IDS_FILE)
    print(f"  Feature matrix saved.")

def step_kg(candidates):
    from career_kg import build_all_kg_features, report_kg_stats
    from utils import save_pickle

    print("\n[Step 2] Building career KG features ...")
    kg_features = build_all_kg_features(candidates, verbose=True)
    report_kg_stats(kg_features)
    save_pickle(kg_features, KG_FEATURES_FILE)
    print(f"  KG features saved.")

def step_embed(candidates):
    from embedder import ONNXEmbedder, build_faiss_index, save_faiss_index, build_role_embeddings
    from jd_parser import parse_jd
    from config import MINILM_ONNX, MINILM_TOKDIR, EMBEDDING_BATCH_SIZE
    from utils import get_career_narrative, save_pickle
    import numpy as np

    print("\n[Step 3] Building embeddings and FAISS index ...")
    embedder = ONNXEmbedder(MINILM_ONNX, MINILM_TOKDIR)
    jd       = parse_jd()

    # JD embedding
    jd_vec = embedder.embed_single(jd["embedding_text"])
    np.save(str(JD_EMBEDDING_FILE), jd_vec)
    print(f"  JD embedding saved.")

    # Career narratives → FAISS
    narratives  = [get_career_narrative(c) for c in candidates]
    career_vecs = embedder.embed_texts(narratives, EMBEDDING_BATCH_SIZE, verbose=True)
    index       = build_faiss_index(career_vecs)
    save_faiss_index(index, FAISS_INDEX_FILE)

    # Role-level embeddings
    role_embs = build_role_embeddings(candidates, embedder, EMBEDDING_BATCH_SIZE)
    save_pickle(role_embs, ROLE_EMBEDDINGS_FILE)
    print(f"  Role embeddings saved.")


def main():
    parser = argparse.ArgumentParser(description="Build offline index artifacts")
    parser.add_argument(
        "--step",
        choices=["features", "kg", "embed", "all"],
        default="all",
        help="Which pipeline step to run (default: all)"
    )
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  Offline Index Builder")
    print(f"  Step: {args.step}")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    t0 = time.time()
    candidates = load_candidates(CANDIDATES_FILE, SAMPLE_FILE, verbose=True)

    if args.step in ("features", "all"):
        t1 = time.time()
        step_features(candidates)
        print(f"  Features done in {(time.time()-t1)/60:.1f} min")

    if args.step in ("kg", "all"):
        t1 = time.time()
        step_kg(candidates)
        print(f"  KG done in {(time.time()-t1)/60:.1f} min")

    if args.step in ("embed", "all"):
        t1 = time.time()
        step_embed(candidates)
        print(f"  Embeddings done in {(time.time()-t1)/60:.1f} min")

    total = (time.time() - t0) / 60
    print(f"\n  Total time: {total:.1f} min")
    print(f"  Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\n  Artifacts ready in: artifacts/")
    print(f"  Next step: python rank.py")


if __name__ == "__main__":
    main()
