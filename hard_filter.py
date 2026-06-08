"""
hard_filter.py
==============
Layer 1 of the online ranking pipeline.
Binary gate — candidates failing ANY rule get score=0 immediately.
This runs BEFORE any scoring, eliminating ~60-70% of candidates.

Rules derived directly from JD and EDA findings:
  Gate 1: disqualified title (HR Manager, Accountant etc.)
  Gate 2: pure consulting career (entire career at TCS/Infosys etc.)
  Gate 3: CV/speech/robotics dominant (JD explicit rejection)
  Gate 4: framework enthusiast (LangChain-only, no depth)
  Gate 5: honeypot anomaly (impossible skill durations/tenures)

EDA showed ~60-70% of candidates eliminated by these gates.
Running gates first saves compute on the remaining pipeline.
"""

import sys
import numpy as np
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    FEATURE_MATRIX_FILE, CANDIDATE_IDS_FILE,
    HARD_FILTER,
)
from utils import load_pickle

# Feature matrix column indices
IDX_DQ_TITLE      = 1
IDX_PURE_CONSULT  = 11
IDX_CV_SPEECH     = 12
IDX_FW_ENTHUSIAST = 13
IDX_HONEYPOT      = 15

# Gate threshold (features are binary 0/1 — use 0.5 as boundary)
GATE_THRESHOLD = 0.5

# Issue 5: CV-primary titles — hard gate, not just structural cap.
# These candidates should be eliminated at the gate layer, not
# allowed through with a capped structural score.
# Kept separate from IDX_DQ_TITLE so the elimination log is clear.
CV_PRIMARY_TITLES_LOWER = {
    'computer vision engineer', 'cv engineer', 'vision engineer',
    'computer vision researcher', 'computer vision scientist',
}


def apply_hard_filters(
    feature_matrix: np.ndarray,
    candidate_ids: list,
    candidates: list = None,
    verbose: bool = True,
) -> tuple:
    """
    Apply binary gate filters to the feature matrix.

    Args:
        feature_matrix: shape (N, 25) float32
        candidate_ids:  list of candidate_id strings
        candidates:     optional list of full candidate dicts for CV title gate
        verbose:        print elimination stats

    Returns:
        eligible_mask    — boolean array shape (N,), True = eligible
        elimination_log  — dict with per-gate elimination counts
    """
    N    = len(candidate_ids)
    mask = np.ones(N, dtype=bool)   # start: all eligible

    gates = [
        (IDX_DQ_TITLE,     "Disqualified title"),
        (IDX_PURE_CONSULT, "Pure consulting career"),
        (IDX_CV_SPEECH,    "CV/speech/robotics dominant"),
        (IDX_FW_ENTHUSIAST,"Framework enthusiast"),
        (IDX_HONEYPOT,     "Honeypot anomaly"),
    ]

    elimination_log = {}

    for feat_idx, gate_name in gates:
        gate_fired       = feature_matrix[:, feat_idx] > GATE_THRESHOLD
        newly_eliminated = gate_fired & mask
        count            = int(newly_eliminated.sum())
        mask             = mask & ~gate_fired
        elimination_log[gate_name] = count
        if verbose:
            remaining = int(mask.sum())
            print(f"  {gate_name:<35s}  eliminated: {count:>6,}  "
                  f"remaining: {remaining:>6,}")

    # Issue 5: CV-primary title hard gate
    # Applied using candidate dicts if available, else skip
    if candidates is not None:
        cv_gate = np.array([
            c.get("profile", {}).get("current_title", "").lower()
            in CV_PRIMARY_TITLES_LOWER
            for c in candidates
        ], dtype=bool)
        newly_eliminated = cv_gate & mask
        count = int(newly_eliminated.sum())
        mask  = mask & ~cv_gate
        elimination_log["CV-primary title (hard gate)"] = count
        if verbose:
            remaining = int(mask.sum())
            print(f"  {'CV-primary title (hard gate)':<35s}  eliminated: {count:>6,}  "
                  f"remaining: {remaining:>6,}")

    total_eliminated = N - int(mask.sum())
    elimination_log["total_eliminated"] = total_eliminated
    elimination_log["total_eligible"]   = int(mask.sum())

    if verbose:
        print(f"\n  {'─'*60}")
        print(f"  Total eliminated : {total_eliminated:,}  "
              f"({total_eliminated/N*100:.1f}%)")
        print(f"  Eligible         : {int(mask.sum()):,}  "
              f"({mask.sum()/N*100:.1f}%)")

    return mask, elimination_log


def get_eligible_indices(mask: np.ndarray) -> np.ndarray:
    """Return integer indices of eligible candidates."""
    return np.where(mask)[0]


def get_eligible_ids(mask: np.ndarray, candidate_ids: list) -> list:
    """Return candidate_id strings for eligible candidates."""
    return [candidate_ids[i] for i in get_eligible_indices(mask)]


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  Hard Filter — binary gate layer")
    print("=" * 60 + "\n")

    if not FEATURE_MATRIX_FILE.exists():
        print(f"  ERROR: {FEATURE_MATRIX_FILE} not found.")
        print(f"  Run: python feature_extractor.py  first.")
    else:
        matrix = load_pickle(FEATURE_MATRIX_FILE)
        ids    = load_pickle(CANDIDATE_IDS_FILE)

        print(f"  Loaded feature matrix: {matrix.shape}")
        mask, log = apply_hard_filters(matrix, ids, verbose=True)
        print(f"\n  Gate log: {log}")
