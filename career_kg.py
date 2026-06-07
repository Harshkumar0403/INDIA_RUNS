"""
career_kg.py
============
Builds event-centric knowledge graphs for each candidate's career.
Directly adapted from the Vrittanta paper's KG framework.

Paper mapping:
  Story events (MOV, COM, COG...) → Career events (CORE_ML_ROLE, ...)
  Temporal edges                  → Chronological role transitions
  Causal edges                    → Logical career progression edges
  KL-divergence from real dist.   → Arc alignment vs JD ideal arc
  Transition matrix Pearson r     → Career coherence score

For each candidate we compute:
  arc_vector         — distribution over event types (8-dim)
  arc_alignment      — cosine similarity vs JD ideal arc
  causal_score       — avg causal edge weight along career path
  kg_total_score     — weighted combination

Run:
    python career_kg.py
Produces:
    artifacts/kg_features.pkl  (dict: candidate_id → kg_feature_dict)
"""

import sys
import json
import math
from datetime import datetime
from pathlib import Path
from collections import Counter
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs): return x

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    CANDIDATES_FILE, SAMPLE_FILE, KG_FEATURES_FILE,
    CONSULTING_FIRMS,
    CAREER_EVENT_TYPES, IDEAL_ARC_DISTRIBUTION,
    CAUSAL_TRANSITION_SCORES, DEFAULT_TRANSITION_SCORE,
    PRODUCTION_CUES,
)
from constants import SENIORITY_LADDER, CV_SPEECH_SKILLS, PRODUCT_INDUSTRIES
from utils import (
    load_candidates, save_pickle, is_consulting_company,
    cosine_similarity_vectors,
)


TODAY = datetime(2026, 6, 2)

# ─────────────────────────────────────────────────────────────────
# CAREER EVENT CLASSIFIER
# Maps each job role to one of 8 career event types.
# This is the "event annotation" step from the paper.
# ─────────────────────────────────────────────────────────────────

CORE_ML_TITLE_KEYWORDS = {
    "ml engineer", "machine learning", "ai engineer", "nlp engineer",
    "data scientist", "research engineer", "applied scientist",
    "recommendation", "search engineer", "ranking", "retrieval",
    "information retrieval", "applied ml", "applied ai",
    "ai research", "deep learning engineer", "embedding",
}
DATA_ENG_TITLE_KEYWORDS = {
    "data engineer", "analytics engineer", "etl", "data pipeline",
    "data analyst", "bi engineer", "data architect",
    "big data", "data platform",
}
SWE_TITLE_KEYWORDS = {
    "software engineer", "backend engineer", "full stack",
    "platform engineer", "api engineer", "developer",
    "sde", "swe", "systems engineer", "infrastructure",
}
LEADERSHIP_TITLE_KEYWORDS = {
    "lead", "manager", "director", "head of", "vp ", "chief",
    "principal", "architect", "staff engineer",
}
RESEARCH_TITLE_KEYWORDS = {
    "research", "scientist", "phd", "postdoc", "intern",
    "fellow", "academic", "university",
}


def classify_career_event(role: dict) -> str:
    """
    Classify a single job role into one of CAREER_EVENT_TYPES.
    Uses title keywords first, then industry, then company name.
    """
    title    = (role.get("title", "") or "").lower()
    industry = role.get("industry", "") or ""
    company  = role.get("company", "") or ""
    desc     = (role.get("description", "") or "").lower()

    # Consulting check (hard — overrides title)
    if is_consulting_company(company, CONSULTING_FIRMS):
        return "CONSULTING_STINT"

    # Leadership (check early — 'ML Lead' should be LEADERSHIP)
    if any(kw in title for kw in LEADERSHIP_TITLE_KEYWORDS):
        # but if also ML keywords → CORE_ML_ROLE (engineer > manager)
        if any(kw in title for kw in CORE_ML_TITLE_KEYWORDS):
            return "CORE_ML_ROLE"
        return "LEADERSHIP_EVENT"

    # Core ML/AI
    if any(kw in title for kw in CORE_ML_TITLE_KEYWORDS):
        return "CORE_ML_ROLE"

    # Research
    if any(kw in title for kw in RESEARCH_TITLE_KEYWORDS):
        return "RESEARCH_WORK"

    # Data Engineering
    if any(kw in title for kw in DATA_ENG_TITLE_KEYWORDS):
        return "DATA_ENGINEERING"

    # Software Engineering
    if any(kw in title for kw in SWE_TITLE_KEYWORDS):
        return "SOFTWARE_ENGINEERING"

    # Product domain (by industry)
    if industry in PRODUCT_INDUSTRIES:
        return "PRODUCT_DOMAIN"

    return "OTHER_ROLE"


# ─────────────────────────────────────────────────────────────────
# TEMPORAL EDGE BUILDER
# Connects consecutive roles in chronological order.
# Paper: temporal edges capture narrative chronology.
# Here: career chronology.
# ─────────────────────────────────────────────────────────────────

def build_temporal_edges(event_sequence: list) -> list:
    """
    Build temporal edges between consecutive career events.
    Returns list of (src_type, dst_type, edge_weight) tuples.
    Temporal edges always have weight 1.0 — they capture order,
    not quality. Quality is in the causal edges.
    """
    edges = []
    for i in range(len(event_sequence) - 1):
        src = event_sequence[i]
        dst = event_sequence[i + 1]
        edges.append({
            "type":   "temporal",
            "src":    src,
            "dst":    dst,
            "weight": 1.0,
        })
    return edges


# ─────────────────────────────────────────────────────────────────
# CAUSAL EDGE BUILDER
# Paper finding: causal edges matter more than temporal for coherence.
# Career analogy: logical career progression (did each role enable
# the next?) matters more than just chronological order.
# ─────────────────────────────────────────────────────────────────

def compute_causal_edge_weight(
    src_type: str,
    dst_type: str,
    src_role: dict,
    dst_role: dict,
) -> float:
    """
    Compute causal edge weight between two adjacent roles.
    Combines:
      1. Pre-defined transition score (from config.CAUSAL_TRANSITION_SCORES)
      2. Domain continuity bonus (same industry = causal continuity)
      3. Tenure depth bonus (longer tenure = more genuine expertise)
      4. Seniority progression bonus

    Returns float in [-1.0, +1.0].
    Positive = good progression, Negative = regression / bad pattern.
    """
    # Base transition score
    base = CAUSAL_TRANSITION_SCORES.get(
        (src_type, dst_type),
        DEFAULT_TRANSITION_SCORE
    )

    # Domain continuity: same industry suggests causality
    src_industry = src_role.get("industry", "")
    dst_industry = dst_role.get("industry", "")
    domain_bonus = 0.10 if (src_industry and src_industry == dst_industry) else 0.0

    # Tenure depth: src role tenure ≥ 18 months → genuine expertise acquired
    src_tenure = src_role.get("duration_months", 0) or 0
    tenure_bonus = 0.10 if src_tenure >= 18 else (-0.05 if src_tenure < 10 else 0.0)

    # Seniority progression
    src_title = (src_role.get("title", "") or "").lower()
    dst_title = (dst_role.get("title", "") or "").lower()
    src_level = max(
        (i for i, kw in enumerate(SENIORITY_LADDER) if kw in src_title),
        default=0
    )
    dst_level = max(
        (i for i, kw in enumerate(SENIORITY_LADDER) if kw in dst_title),
        default=0
    )
    seniority_bonus = 0.10 if dst_level > src_level else 0.0

    total = base + domain_bonus + tenure_bonus + seniority_bonus
    return max(-1.0, min(1.0, total))


def build_causal_edges(event_sequence: list, sorted_hist: list) -> list:
    """
    Build causal edges for the career graph.
    Each edge has a causal weight reflecting career progression quality.
    """
    edges = []
    for i in range(len(event_sequence) - 1):
        src_type = event_sequence[i]
        dst_type = event_sequence[i + 1]
        src_role = sorted_hist[i]
        dst_role = sorted_hist[i + 1]

        weight = compute_causal_edge_weight(
            src_type, dst_type, src_role, dst_role
        )
        edges.append({
            "type":   "causal",
            "src":    src_type,
            "dst":    dst_type,
            "weight": weight,
        })
    return edges


# ─────────────────────────────────────────────────────────────────
# ARC ALIGNMENT SCORER
# Paper: KL-divergence between generated and real event distributions.
# Here: cosine similarity between candidate arc and JD ideal arc.
# We use cosine (not KL) because candidate arcs are sparse —
# KL divergence requires the same support.
# ─────────────────────────────────────────────────────────────────

def compute_arc_alignment(event_sequence: list) -> float:
    """
    Compute cosine similarity between candidate's career event
    distribution and the JD ideal arc distribution.

    Returns float in [0, 1]. 1.0 = perfect alignment with ideal arc.
    """
    if not event_sequence:
        return 0.0

    # Candidate's arc distribution
    counts  = Counter(event_sequence)
    total   = len(event_sequence)
    cand_vec = [
        counts.get(et, 0) / total
        for et in CAREER_EVENT_TYPES
    ]

    # JD ideal arc distribution (from config)
    ideal_vec = [
        IDEAL_ARC_DISTRIBUTION.get(et, 0.0)
        for et in CAREER_EVENT_TYPES
    ]

    return cosine_similarity_vectors(cand_vec, ideal_vec)


# ─────────────────────────────────────────────────────────────────
# CAUSAL COHERENCE SCORER
# Paper finding: causal edges > temporal edges for coherence.
# Causal coherence = average causal edge weight along career path.
# A career where every transition makes logical sense scores +1.
# A career of consulting → consulting → consulting scores near -1.
# ─────────────────────────────────────────────────────────────────

def compute_causal_coherence(causal_edges: list) -> float:
    """
    Average causal edge weight, normalized to [0, 1].
    Raw weights are in [-1, 1], so (avg + 1) / 2 maps to [0, 1].
    """
    if not causal_edges:
        return 0.5   # no transitions — neutral
    avg_weight = sum(e["weight"] for e in causal_edges) / len(causal_edges)
    return (avg_weight + 1.0) / 2.0   # normalize to [0, 1]


# ─────────────────────────────────────────────────────────────────
# MILESTONE FIDELITY
# Paper: template fidelity recall — how many required events appear?
# Here: did the candidate hit the JD-required career milestones?
# ─────────────────────────────────────────────────────────────────

REQUIRED_MILESTONES = {
    "CORE_ML_ROLE",          # must have ML/AI/NLP work experience
    "SOFTWARE_ENGINEERING",  # must have built real systems
}
BONUS_MILESTONES = {
    "LEADERSHIP_EVENT",      # nice — senior IC or lead
    "PRODUCT_DOMAIN",        # nice — product company experience
}

def compute_milestone_fidelity(event_sequence: list) -> float:
    """
    Fraction of required milestones present in career.
    Bonus milestones add a small bonus above 1.0.
    """
    if not event_sequence:
        return 0.0

    event_set = set(event_sequence)
    required_hit = len(event_set & REQUIRED_MILESTONES) / len(REQUIRED_MILESTONES)
    bonus_hit    = len(event_set & BONUS_MILESTONES) / len(BONUS_MILESTONES)

    # Required contributes 80%, bonus 20%
    return min(1.0, required_hit * 0.80 + bonus_hit * 0.20)


# ─────────────────────────────────────────────────────────────────
# MAIN KG BUILDER
# ─────────────────────────────────────────────────────────────────

def build_candidate_kg(cand: dict) -> dict:
    """
    Build the full KG representation for one candidate.
    Returns a dict with all KG-derived features.
    """
    hist = cand.get("career_history", [])

    # Sort career history chronologically
    def safe_date(r):
        try:
            return datetime.strptime(r.get("start_date", "2000-01-01"), "%Y-%m-%d")
        except Exception:
            return datetime(2000, 1, 1)

    sorted_hist = sorted(hist, key=safe_date)

    # Classify each role into career event type (event nodes V)
    event_sequence = [classify_career_event(r) for r in sorted_hist]

    # Build temporal and causal edges (E with relation types R)
    temporal_edges = build_temporal_edges(event_sequence)
    causal_edges   = build_causal_edges(event_sequence, sorted_hist)

    # Compute KG-derived scores
    arc_alignment  = compute_arc_alignment(event_sequence)
    causal_score   = compute_causal_coherence(causal_edges)
    fidelity       = compute_milestone_fidelity(event_sequence)

    # Arc distribution vector (for downstream use)
    counts  = Counter(event_sequence)
    total   = max(len(event_sequence), 1)
    arc_vec = {et: counts.get(et, 0) / total for et in CAREER_EVENT_TYPES}

    # Composite KG score
    # Weights from paper insight: causal (0.50) > temporal arc (0.30) > fidelity (0.20)
    kg_score = (
        arc_alignment * 0.30 +
        causal_score  * 0.50 +
        fidelity      * 0.20
    )

    return {
        "candidate_id":   cand.get("candidate_id", ""),
        "event_sequence": event_sequence,
        "arc_vector":     arc_vec,
        "arc_alignment":  round(arc_alignment, 4),
        "causal_score":   round(causal_score,  4),
        "fidelity":       round(fidelity,       4),
        "kg_score":       round(kg_score,       4),
        "n_temporal":     len(temporal_edges),
        "n_causal":       len(causal_edges),
        "temporal_edges": temporal_edges,
        "causal_edges":   causal_edges,
    }


def build_all_kg_features(
    candidates: list,
    verbose: bool = True,
) -> dict:
    """
    Build KG features for all candidates.
    Returns dict: candidate_id → kg_feature_dict
    """
    kg_features = {}
    iterator = tqdm(candidates, desc="  Building career KGs", unit="cand") \
               if verbose else candidates

    for cand in iterator:
        cid = cand.get("candidate_id", "")
        kg_features[cid] = build_candidate_kg(cand)

    return kg_features


# ─────────────────────────────────────────────────────────────────
# STATS REPORTER
# ─────────────────────────────────────────────────────────────────

def report_kg_stats(kg_features: dict) -> None:
    N = len(kg_features)
    print(f"\n  KG features built for {N:,} candidates")

    arc_scores    = [v["arc_alignment"] for v in kg_features.values()]
    causal_scores = [v["causal_score"]  for v in kg_features.values()]
    kg_scores     = [v["kg_score"]      for v in kg_features.values()]
    fidelities    = [v["fidelity"]      for v in kg_features.values()]

    def stats(lst, label):
        lst = sorted(lst)
        print(f"    {label:<25s}  mean={sum(lst)/len(lst):.3f}  "
              f"median={lst[len(lst)//2]:.3f}  "
              f"min={min(lst):.3f}  max={max(lst):.3f}")

    print(f"\n  Score distributions:")
    stats(arc_scores,    "arc_alignment")
    stats(causal_scores, "causal_score")
    stats(fidelities,    "fidelity")
    stats(kg_scores,     "kg_total_score")

    # Event type distribution across all careers
    all_events = []
    for v in kg_features.values():
        all_events.extend(v["event_sequence"])

    total_events = len(all_events)
    counts = Counter(all_events)
    print(f"\n  Career event type distribution ({total_events:,} total events):")
    for et in CAREER_EVENT_TYPES:
        n = counts.get(et, 0)
        bar = "█" * round(n / max(counts.values()) * 20)
        print(f"    {et:<28s}  {n:>7,}  ({n/total_events*100:.1f}%)  {bar}")

    # Top KG scorers
    top = sorted(kg_features.items(), key=lambda x: -x[1]["kg_score"])[:5]
    print(f"\n  Top 5 by KG score:")
    for cid, feat in top:
        print(f"    {cid}  kg={feat['kg_score']:.3f}  "
              f"arc={feat['arc_alignment']:.3f}  "
              f"causal={feat['causal_score']:.3f}  "
              f"arc={feat['event_sequence'][:4]}")


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 60)
    print("  Career KG Builder — event nodes + causal edges")
    print("=" * 60 + "\n")

    candidates  = load_candidates(CANDIDATES_FILE, SAMPLE_FILE, verbose=True)
    kg_features = build_all_kg_features(candidates, verbose=True)

    report_kg_stats(kg_features)

    save_pickle(kg_features, KG_FEATURES_FILE)
    print(f"\n  Saved → {KG_FEATURES_FILE}")


if __name__ == "__main__":
    main()
