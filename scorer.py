"""
scorer.py
=========
Layer 3 of the online ranking pipeline.
Computes the 5-dimensional multiplicative score for each candidate.

Architecture:
  final_score = gate_score × (
      semantic_score    × W_semantic    +
      structural_score  × W_structural  +
      skill_idf_score   × W_skill       +
      behavioral_score  × W_behavioral
  ) × availability_multiplier

Multiplicative composition: zero in any mandatory dimension
propagates to zero final score. This mirrors how a good recruiter
thinks — disqualifiers and unavailability collapse a candidate
regardless of skill quality.
"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import SCORING, AVAILABILITY, FAISS
from utils import availability_score

# Feature matrix column indices (must match feature_extractor.py)
IDX_TITLE_REL    = 0
IDX_YOE          = 2
IDX_AI_IDF       = 3
IDX_RETR_IDF     = 4
IDX_JD_HARD      = 5
IDX_JD_IDEAL     = 6
IDX_JD_NEG       = 7
IDX_PROD_PROVEN  = 8
IDX_CTX_VER      = 9
IDX_STUFFER      = 10
IDX_CV_SPEECH    = 12   # cv/speech dominant gate (Issue B)
IDX_PROD_RATIO   = 16
IDX_LOC          = 18
IDX_AVAIL        = 19
IDX_OTW          = 20
IDX_NOTICE       = 21
IDX_RESPONSE     = 22
IDX_GITHUB       = 23
IDX_EDUCATION    = 24

EV_MULT = SCORING["skill_evidence_multipliers"]

# Issue 1: outside-India multiplicative location gates
# location_score thresholds: ≥1.0=india_target, 0.65=india_non_metro,
# 0.35=india_unwilling, 0.30=outside_willing, 0.05=outside_unwilling
OUTSIDE_INDIA_WILLING_MULT  = 0.55  # outside India, willing to relocate
OUTSIDE_INDIA_NO_RELOC_MULT = 0.30  # outside India, NOT willing — strong penalty

# Issue 6: saved_by_recruiters threshold for availability boost
# ≥25 saves in 30d = market interest signal, partially offsets inactivity
SAVED_RECRUITERS_BOOST_THRESHOLD = 25
SAVED_RECRUITERS_AVAIL_BONUS     = 0.08


def compute_semantic_score(
    faiss_cosine: float,
    jd_hard_score: float,
    jd_ideal_score: float,
    jd_neg_score: float,
) -> float:
    """
    Semantic score combines FAISS cosine similarity with
    JD section match scores.

    FAISS gives semantic proximity of career narrative to JD.
    JD section scores give precision (hard requirements matched).
    Negative section match penalises.

    Returns float [0, 1].
    """
    # FAISS cosine is already 0-1 after L2 normalization
    faiss_component  = float(faiss_cosine)

    # JD match components
    jd_component = (
        jd_hard_score  * 0.60 +
        jd_ideal_score * 0.30 -
        jd_neg_score   * 0.50    # negative section penalty
    )
    jd_component = max(0.0, jd_component)

    # Weighted combination
    semantic = faiss_component * 0.50 + jd_component * 0.50
    return float(np.clip(semantic, 0.0, 1.0))


def compute_skill_idf_score(
    ai_idf: float,
    retrieval_idf: float,
    prod_proven: float,
    ctx_verified: float,
    stuffer_ratio: float,
) -> float:
    """
    IDF-weighted skill score with evidence level multipliers.

    Retrieval skills get 1.5× boost (JD primary requirement).
    Production-proven skills get 3.0× the weight of self-reported.
    High stuffer ratio (≥0.75) halves the total skill score.

    Returns float [0, 1].
    """
    # Retrieval skills are the JD's primary technical requirement
    base_skill = ai_idf * 0.40 + retrieval_idf * 0.60

    # Evidence quality multiplier
    # prod_proven and ctx_verified are already normalized 0-1
    evidence_quality = (
        prod_proven   * EV_MULT["PRODUCTION_PROVEN"] / 3.0 +
        ctx_verified  * EV_MULT["CONTEXT_VERIFIED"]  / 3.0
    )
    # If no evidence quality, fall back to base
    evidence_boost = max(0.0, min(0.5, evidence_quality))

    # Stuffer penalty: ratio close to 1.0 = most skills unverified
    stuffer_penalty = 1.0 - (stuffer_ratio * 0.50)

    skill_score = (base_skill + evidence_boost) * stuffer_penalty
    return float(np.clip(skill_score, 0.0, 1.0))


# Issue B: CV-primary titles that should be penalised structurally
CV_PRIMARY_TITLES_LOWER = {
    'computer vision engineer', 'cv engineer', 'vision engineer',
    'computer vision researcher', 'computer vision scientist',
}

def compute_structural_score(
    title_relevance: float,
    yoe_score: float,
    product_ratio: float,
    kg_score: float,
    education_score: float,
    current_title: str = "",
) -> float:
    """
    Structural score: career arc quality + role relevance.
    Issue B fix: CV-primary titles get a 0.30 cap on structural score
    regardless of KG arc or education — the JD explicitly rejects them.
    """
    structural = (
        title_relevance  * 0.20 +
        yoe_score        * 0.10 +
        product_ratio    * 0.10 +
        kg_score         * 0.50 +
        education_score  * 0.10
    )
    result = float(np.clip(structural, 0.0, 1.0))

    # Issue B: hard cap for CV-primary titles
    if current_title.lower() in CV_PRIMARY_TITLES_LOWER:
        result = min(result, 0.30)

    return result


def compute_availability_multiplier(
    avail_decay: float,
    open_to_work: float,
    notice_score: float,
    response_rate: float,
    saved_by_recruiters: int = 0,
) -> float:
    """
    Availability multiplier — applied multiplicatively to final score.

    Issue 6 fix: saved_by_recruiters_30d is now factored in.
    A candidate saved by 25+ recruiters in 30 days is clearly active
    in the market even if their platform login is 47 days ago.
    High saved count partially offsets inactivity penalty.
    """
    mult = avail_decay   # already e^(-λt)

    # Open to work boost (35.3% set this — meaningful signal)
    if open_to_work < 0.5:
        mult *= AVAILABILITY["otw_penalty"]

    # Notice period — high notice reduces effective availability
    # notice_score is already inverse (1 = immediate, 0 = 150d)
    if notice_score < 0.40:   # notice > 90 days
        mult *= AVAILABILITY["notice_penalty_mult"]

    # Low response rate — candidate unlikely to engage
    if response_rate < AVAILABILITY["low_response_threshold"]:
        mult *= AVAILABILITY["low_response_mult"]

    # Issue 6: saved by many recruiters = market interest signal
    # partially offsets inactivity (cap boost at +8%)
    if saved_by_recruiters >= SAVED_RECRUITERS_BOOST_THRESHOLD:
        mult = min(1.0, mult + SAVED_RECRUITERS_AVAIL_BONUS)

    # Apply floor
    return float(np.clip(mult, AVAILABILITY["min_availability"], 1.0))


def compute_behavioral_score(
    github_score: float,
    response_rate: float,
    location_score: float,
) -> float:
    """
    Behavioral quality score from platform signals.
    Sparse (64.6% have no GitHub) so kept at low weight.
    """
    behavioral = (
        github_score    * 0.40 +
        response_rate   * 0.30 +
        location_score  * 0.30
    )
    return float(np.clip(behavioral, 0.0, 1.0))


def score_candidates(
    feature_matrix: np.ndarray,
    faiss_cosine_scores: np.ndarray,
    kg_scores: np.ndarray,
    candidate_indices: np.ndarray,
    saved_by_recruiters: np.ndarray = None,
) -> np.ndarray:
    """
    Compute final scores for a set of candidates.

    Args:
        feature_matrix:      shape (N_total, 25) — full matrix
        faiss_cosine_scores: shape (M,) — cosine sims from FAISS for M candidates
        kg_scores:           shape (M,) — KG arc scores for M candidates
        candidate_indices:   shape (M,) — row indices into feature_matrix
        saved_by_recruiters: shape (M,) — saved_by_recruiters_30d per candidate
                             (optional; zeros if not provided)

    Returns:
        scores: float32 array shape (M,) — final composite scores
    """
    M      = len(candidate_indices)
    scores = np.zeros(M, dtype=np.float32)
    if saved_by_recruiters is None:
        saved_by_recruiters = np.zeros(M, dtype=np.int32)

    W = SCORING

    for j, idx in enumerate(candidate_indices):
        row = feature_matrix[idx]

        # Extract features
        semantic = compute_semantic_score(
            faiss_cosine      = float(faiss_cosine_scores[j]),
            jd_hard_score     = float(row[IDX_JD_HARD]),
            jd_ideal_score    = float(row[IDX_JD_IDEAL]),
            jd_neg_score      = float(row[IDX_JD_NEG]),
        )

        skill = compute_skill_idf_score(
            ai_idf         = float(row[IDX_AI_IDF]),
            retrieval_idf  = float(row[IDX_RETR_IDF]),
            prod_proven    = float(row[IDX_PROD_PROVEN]),
            ctx_verified   = float(row[IDX_CTX_VER]),
            stuffer_ratio  = float(row[IDX_STUFFER]),
        )

        # Issue B: cv_speech_dominant flag caps structural score
        raw_structural = compute_structural_score(
            title_relevance = float(row[IDX_TITLE_REL]),
            yoe_score       = float(row[IDX_YOE]),
            product_ratio   = float(row[IDX_PROD_RATIO]),
            kg_score        = float(kg_scores[j]),
            education_score = float(row[IDX_EDUCATION]),
        )
        structural = min(raw_structural, 0.30) if row[IDX_CV_SPEECH] > 0.5 else raw_structural

        avail_mult = compute_availability_multiplier(
            avail_decay         = float(row[IDX_AVAIL]),
            open_to_work        = float(row[IDX_OTW]),
            notice_score        = float(row[IDX_NOTICE]),
            response_rate       = float(row[IDX_RESPONSE]),
            saved_by_recruiters = int(saved_by_recruiters[j]),
        )

        behavioral = compute_behavioral_score(
            github_score  = float(row[IDX_GITHUB]),
            response_rate = float(row[IDX_RESPONSE]),
            location_score= float(row[IDX_LOC]),
        )

        # Weighted additive combination of the four score components
        composite = (
            semantic    * W["semantic_weight"]    +
            structural  * W["structural_weight"]  +
            skill       * W["skill_idf_weight"]   +
            behavioral  * W["behavioral_weight"]
        )

        # Multiplicative availability — punishes inaccessible candidates
        final = composite * avail_mult

        # Issue 1: location gate — outside India candidates get a hard
        # multiplicative penalty regardless of technical strength.
        # location_score: 1.0=india_target, 0.65=india_non_metro,
        # 0.35=india_unwilling, 0.30=outside+willing, 0.05=outside+unwilling
        loc = float(row[IDX_LOC])
        if loc <= 0.10:   # outside India, not willing to relocate
            final *= OUTSIDE_INDIA_NO_RELOC_MULT
        elif loc <= 0.32:  # outside India, willing to relocate
            final *= OUTSIDE_INDIA_WILLING_MULT

        scores[j] = float(np.clip(final, 0.0, 1.0))

    return scores


def score_breakdown(
    feature_matrix: np.ndarray,
    idx: int,
    faiss_cosine: float,
    kg_score: float,
) -> dict:
    """
    Return detailed score breakdown for a single candidate.
    Used by reasoning.py to generate grounded explanations.
    """
    row = feature_matrix[idx]

    semantic = compute_semantic_score(
        float(faiss_cosine), float(row[IDX_JD_HARD]),
        float(row[IDX_JD_IDEAL]), float(row[IDX_JD_NEG]),
    )
    skill = compute_skill_idf_score(
        float(row[IDX_AI_IDF]), float(row[IDX_RETR_IDF]),
        float(row[IDX_PROD_PROVEN]), float(row[IDX_CTX_VER]),
        float(row[IDX_STUFFER]),
    )
    raw_structural = compute_structural_score(
        float(row[IDX_TITLE_REL]), float(row[IDX_YOE]),
        float(row[IDX_PROD_RATIO]), kg_score,
        float(row[IDX_EDUCATION]),
    )
    structural = min(raw_structural, 0.30) if row[IDX_CV_SPEECH] > 0.5 else raw_structural
    avail_mult = compute_availability_multiplier(
        float(row[IDX_AVAIL]), float(row[IDX_OTW]),
        float(row[IDX_NOTICE]), float(row[IDX_RESPONSE]),
    )
    behavioral = compute_behavioral_score(
        float(row[IDX_GITHUB]), float(row[IDX_RESPONSE]),
        float(row[IDX_LOC]),
    )

    W = SCORING
    composite = (
        semantic   * W["semantic_weight"]   +
        structural * W["structural_weight"] +
        skill      * W["skill_idf_weight"]  +
        behavioral * W["behavioral_weight"]
    )
    final = float(np.clip(composite * avail_mult, 0.0, 1.0))

    return {
        "semantic_score":    round(semantic,    4),
        "structural_score":  round(structural,  4),
        "skill_idf_score":   round(skill,       4),
        "behavioral_score":  round(behavioral,  4),
        "availability_mult": round(avail_mult,  4),
        "composite_score":   round(composite,   4),
        "final_score":       round(final,       4),
        "detail": {
            "title_relevance":   round(float(row[IDX_TITLE_REL]), 3),
            "yoe_score":         round(float(row[IDX_YOE]), 3),
            "jd_hard_matches":   round(float(row[IDX_JD_HARD]), 3),
            "jd_neg_matches":    round(float(row[IDX_JD_NEG]), 3),
            "retrieval_idf":     round(float(row[IDX_RETR_IDF]), 3),
            "prod_proven_skills":round(float(row[IDX_PROD_PROVEN]), 3),
            "stuffer_ratio":     round(float(row[IDX_STUFFER]), 3),
            "kg_score":          round(kg_score, 3),
            "location_score":    round(float(row[IDX_LOC]), 3),
            "notice_score":      round(float(row[IDX_NOTICE]), 3),
            "github_score":      round(float(row[IDX_GITHUB]), 3),
            "response_rate":     round(float(row[IDX_RESPONSE]), 3),
        }
    }
