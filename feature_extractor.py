"""
feature_extractor.py
====================
Builds a 25-dimensional feature vector for every candidate.
Each dimension has a principled derivation — no arbitrary numbers.

Feature dimensions (indices 0-24):
  [0]  title_relevance_score     — from TITLE_RELEVANCE map (0-1)
  [1]  is_disqualified_title     — binary gate (1=disqualified)
  [2]  yoe_score                 — Gaussian peak at JD sweet spot 6-8yr
  [3]  ai_skill_idf_score        — IDF-weighted AI/ML skills (normalized)
  [4]  retrieval_skill_idf_score — IDF-weighted retrieval-specific skills
  [5]  jd_hard_req_score         — hard requirement phrase matches (career text)
  [6]  jd_ideal_score            — ideal profile phrase matches
  [7]  jd_negative_score         — negative phrase matches (PENALISES)
  [8]  skill_production_proven   — count of production-proven skills
  [9]  skill_context_verified    — count of context-verified skills
  [10] skill_self_reported_ratio — ratio of unverified skills (stuffer signal)
  [11] is_pure_consulting        — binary (1=entire career at consulting firms)
  [12] is_cv_speech_dominant     — binary (1=CV/speech with no NLP/IR)
  [13] is_framework_enthusiast   — binary (1=LangChain only, no depth)
  [14] is_title_chaser           — binary (1=short tenures + escalating titles)
  [15] is_honeypot               — binary (1=data anomaly detected)
  [16] product_company_ratio     — fraction of career at product companies
  [17] consulting_career_ratio   — fraction of career at consulting firms
  [18] location_score            — target city match (0/0.5/1.0)
  [19] availability_raw          — e^(-λt) decay score
  [20] open_to_work              — binary (1=flag set)
  [21] notice_score              — inverse notice period score (0-1)
  [22] response_rate             — recruiter response rate (0-1)
  [23] github_score              — normalized (0-1, 0 if no account)
  [24] education_score           — tier-based (tier_1=1.0 .. tier_4=0.25)

Run:
    python feature_extractor.py
Produces:
    artifacts/feature_matrix.pkl   (numpy float32 array, shape N×25)
    artifacts/candidate_ids.pkl    (list of candidate_id strings)
"""

import sys
import math
import numpy as np
from pathlib import Path
from datetime import datetime
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs): return x

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    CANDIDATES_FILE, SAMPLE_FILE, FEATURE_MATRIX_FILE,
    CANDIDATE_IDS_FILE, SKILL_VOCAB_FILE,
    CONSULTING_FIRMS, TARGET_LOCATIONS, PRODUCTION_CUES,
    HARD_FILTER, AVAILABILITY, CAREER_EVENT_TYPES,
)
from constants import (
    TITLE_RELEVANCE, DISQUALIFIED_TITLES,
    SENIORITY_LADDER, PRODUCT_INDUSTRIES, CV_SPEECH_SKILLS,
    JD_SECTIONS,
)
from utils import (
    load_candidates, load_skill_vocab, save_pickle,
    get_career_text, get_full_text, get_skill_names,
    get_skill_names_lower, days_since, availability_score,
    get_skill_idf, get_skill_jd_label, is_consulting_company,
    normalize_text,
)
from jd_parser import parse_jd, score_text_against_jd

TODAY    = datetime(2026, 6, 2)
N_FEATS  = 25

# ─────────────────────────────────────────────────────────────────
# FEATURE DIMENSION NAMES (for debugging / reporting)
# ─────────────────────────────────────────────────────────────────
FEATURE_NAMES = [
    "title_relevance",        # 0
    "is_disqualified_title",  # 1
    "yoe_score",              # 2
    "ai_skill_idf",           # 3
    "retrieval_skill_idf",    # 4
    "jd_hard_req",            # 5
    "jd_ideal",               # 6
    "jd_negative",            # 7
    "skill_prod_proven",      # 8
    "skill_ctx_verified",     # 9
    "skill_stuffer_ratio",    # 10
    "is_pure_consulting",     # 11
    "is_cv_speech_dominant",  # 12
    "is_framework_enthusiast",# 13
    "is_title_chaser",        # 14
    "is_honeypot",            # 15
    "product_company_ratio",  # 16
    "consulting_career_ratio",# 17
    "location_score",         # 18
    "availability_raw",       # 19
    "open_to_work",           # 20
    "notice_score",           # 21
    "response_rate",          # 22
    "github_score",           # 23
    "education_score",        # 24
]


# ─────────────────────────────────────────────────────────────────
# INDIVIDUAL FEATURE COMPUTERS
# Each returns a float in [0, 1] unless noted as binary gate.
# ─────────────────────────────────────────────────────────────────

def feat_title_relevance(cand: dict) -> float:
    """
    [0] Title relevance: 0-1 from TITLE_RELEVANCE map.
    Justification: title directly reflects career domain.
    Only ~14% of corpus has relevant titles — high discriminating power.
    """
    title = cand["profile"].get("current_title", "")
    raw   = TITLE_RELEVANCE.get(title, 0)
    return raw / 5.0   # normalize 0-5 → 0-1


def feat_is_disqualified_title(cand: dict) -> float:
    """
    [1] Binary gate: 1.0 if title is in disqualified set.
    These candidates should score 0 regardless of skills.
    """
    title = cand["profile"].get("current_title", "")
    return 1.0 if title in DISQUALIFIED_TITLES else 0.0


def feat_yoe_score(cand: dict) -> float:
    """
    [2] YoE score: Gaussian centered at 7yr (JD sweet spot 6-8yr).
    σ=3 — candidates at 4yr or 10yr get ~0.6, <2yr or >14yr get <0.2.
    Justification: JD says 6-8yr, but judgment > years.
    Using soft Gaussian prevents hard cutoff artifacts.
    """
    yoe = cand["profile"].get("years_of_experience", 0) or 0
    mu, sigma = 7.0, 3.0
    return math.exp(-((yoe - mu) ** 2) / (2 * sigma ** 2))


def feat_ai_skill_idf(cand: dict, vocab: dict) -> float:
    """
    [3] IDF-weighted AI/ML skill score.
    Sum of IDF values for all AI/ML JD-positive skills,
    divided by a normalization constant.
    Justification: IDF from 100k corpus — no arbitrary weights.
    """
    skill_names = get_skill_names(cand)
    total_idf   = 0.0
    for sk in skill_names:
        label = get_skill_jd_label(sk, vocab)
        if label in ("JD_HARD_REQ", "JD_POSITIVE"):
            total_idf += get_skill_idf(sk, vocab, fallback_idf=2.0)
    # normalize: max reasonable sum ~50 (10 skills × IDF 5 avg)
    return min(total_idf / 50.0, 1.0)


def feat_retrieval_skill_idf(cand: dict, vocab: dict) -> float:
    """
    [4] Retrieval-specific skill IDF score.
    Only skills with IDF > 3.5 and JD_HARD_REQ label count.
    These are the rarest and most valuable: FAISS, Pinecone,
    Qdrant, Milvus, Weaviate, Elasticsearch, OpenSearch.
    EDA: only 3.2% of all skill mentions are retrieval skills.
    """
    RETRIEVAL_NAMES_LOWER = {
        "faiss", "pinecone", "weaviate", "qdrant", "milvus",
        "elasticsearch", "opensearch", "pgvector", "vector search",
        "hybrid search", "bm25", "information retrieval",
        "semantic search", "dense retrieval", "sparse retrieval",
    }
    skill_names_low = get_skill_names_lower(cand)
    total_idf = 0.0
    for sk_low in skill_names_low:
        if sk_low in RETRIEVAL_NAMES_LOWER:
            # Look up IDF using original casing
            for sk in get_skill_names(cand):
                if sk.lower() == sk_low:
                    total_idf += get_skill_idf(sk, vocab, fallback_idf=3.0)
                    break
    # normalize: max ~30 (5 retrieval skills × IDF 6 avg)
    return min(total_idf / 30.0, 1.0)


def feat_jd_section_scores(cand: dict, jd: dict) -> tuple:
    """
    [5, 6, 7] JD section match scores for career + summary text.
    Returns (hard_req_score, ideal_score, negative_score).
    Signed: negative_score is the raw penalty (positive float,
    applied as subtraction in scorer.py).
    """
    full_text = get_full_text(cand)
    result    = score_text_against_jd(full_text, jd)

    # normalize by expected max matches per section
    hard_norm = min(result["hard_req_matches"] / 8.0, 1.0)
    ideal_norm= min(result["ideal_matches"]    / 4.0, 1.0)
    neg_norm  = min(result["negative_matches"] / 3.0, 1.0)

    return hard_norm, ideal_norm, neg_norm


def feat_skill_evidence(cand: dict) -> tuple:
    """
    [8, 9, 10] Skill evidence quality features.
    EDA: 96.2% of candidates are keyword stuffers (self_reported > 75%).
    Production-proven: skill in career text + production cue words.
    Context-verified: skill in career text only.
    Self-reported: skills[] array only.
    """
    career_text = get_career_text(cand)
    prod_text   = " ".join(
        (r.get("description", "") or "").lower()
        for r in cand.get("career_history", [])
        if any(cue in (r.get("description","") or "").lower()
               for cue in PRODUCTION_CUES)
    )

    prod_proven  = 0
    ctx_verified = 0
    self_only    = 0

    for s in cand.get("skills", []):
        name_low = (s.get("name", "") or "").lower()
        if not name_low:
            continue
        if name_low in prod_text:
            prod_proven += 1
        elif name_low in career_text:
            ctx_verified += 1
        else:
            self_only += 1

    total = prod_proven + ctx_verified + self_only
    stuffer_ratio = (self_only / total) if total > 0 else 1.0

    # normalize counts (max 5 production-proven is excellent)
    prod_score = min(prod_proven / 5.0, 1.0)
    ctx_score  = min(ctx_verified / 5.0, 1.0)

    return prod_score, ctx_score, stuffer_ratio


def feat_is_pure_consulting(cand: dict) -> float:
    """
    [11] Binary: 1.0 if entire career at consulting firms.
    JD explicitly: 'If you've only worked at TCS/Infosys/Wipro...'
    """
    hist = cand.get("career_history", [])
    if len(hist) < 2:
        return 0.0
    consulting_count = sum(
        1 for r in hist
        if is_consulting_company(r.get("company", ""), CONSULTING_FIRMS)
    )
    return 1.0 if consulting_count == len(hist) else 0.0


def feat_is_cv_speech_dominant(cand: dict) -> float:
    """
    [12] Binary: 1.0 if CV/speech/robotics skills dominate with no NLP/IR.
    JD: 'Primary expertise is computer vision, speech, or robotics
    without significant NLP/IR exposure.'
    """
    skill_names = get_skill_names(cand)
    cv_count    = len(skill_names & CV_SPEECH_SKILLS)
    # NLP/IR counter-signals
    nlp_ir_skills = {
        "NLP", "Information Retrieval", "Semantic Search",
        "Embeddings", "Sentence Transformers", "BERT",
        "Transformers", "RAG", "Vector Search", "FAISS",
    }
    nlp_count = len(skill_names & nlp_ir_skills)
    return 1.0 if (cv_count >= 3 and cv_count > nlp_count) else 0.0


def feat_is_framework_enthusiast(cand: dict) -> float:
    """
    [13] Binary: 1.0 if wrapper tools present but no depth in career text.
    JD: 'LangChain tutorials and demo blogs — not what we need.'
    """
    skill_names_low = get_skill_names_lower(cand)
    wrapper_tools   = HARD_FILTER["wrapper_tools"]
    depth_cues      = HARD_FILTER["depth_skill_cues"]

    has_wrappers = bool(skill_names_low & wrapper_tools)
    career_text  = get_career_text(cand)
    has_depth    = any(d in career_text for d in depth_cues)

    return 1.0 if (has_wrappers and not has_depth) else 0.0


def feat_is_title_chaser(cand: dict) -> float:
    """
    [14] Binary: 1.0 if ≥3 roles with tenure < 20mo AND titles escalating.
    JD: 'Optimizing for Senior → Staff → Principal titles by switching
    companies every 1.5 years.'
    """
    hist = cand.get("career_history", [])
    if len(hist) < 3:
        return 0.0

    threshold     = HARD_FILTER["title_chaser_tenure_months"]
    min_flagged   = HARD_FILTER["title_chaser_min_flagged"]
    short_tenures = sum(
        1 for r in hist
        if 0 < (r.get("duration_months") or 0) < threshold
    )

    # check if title seniority increased over career
    title_levels = []
    for r in hist:
        t = (r.get("title") or "").lower()
        for i, level in enumerate(SENIORITY_LADDER):
            if level in t:
                title_levels.append(i)
                break

    escalating = (
        len(title_levels) >= 2 and
        title_levels[-1] > title_levels[0]
    )
    return 1.0 if (short_tenures >= min_flagged and escalating) else 0.0


def feat_is_honeypot(cand: dict) -> float:
    """
    [15] Binary: 1.0 if data anomalies detected (honeypot signals).
    Checks: skill duration > career length, impossible tenure.
    """
    yoe    = cand["profile"].get("years_of_experience", 0) or 0
    hist   = cand.get("career_history", [])
    skills = cand.get("skills", [])
    slack  = HARD_FILTER["honeypot_duration_slack_months"]

    # skill duration impossibility
    for s in skills:
        dur = s.get("duration_months", 0) or 0
        if dur > (yoe * 12) + slack:
            return 1.0

    # tenure impossibility
    for r in hist:
        try:
            start      = datetime.strptime(r.get("start_date",""), "%Y-%m-%d")
            years_back = (TODAY - start).days / 365
            dur        = r.get("duration_months", 0) or 0
            if dur / 12 > years_back + 0.5:
                return 1.0
        except Exception:
            pass

    # inflated proficiency: ≥5 'advanced' with 0 endorsements
    adv_zero = sum(
        1 for s in skills
        if s.get("proficiency") == "advanced"
        and s.get("endorsements", 0) == 0
    )
    if adv_zero >= 5:
        return 1.0

    return 0.0


def feat_company_type_ratios(cand: dict) -> tuple:
    """
    [16, 17] Product company ratio and consulting career ratio.
    Based on industry field in career history.
    EDA: IT Services = 29.3% of all job records.
    """
    hist = cand.get("career_history", [])
    if not hist:
        return 0.0, 0.0

    product_count   = sum(
        1 for r in hist
        if r.get("industry", "") in PRODUCT_INDUSTRIES
    )
    consulting_count = sum(
        1 for r in hist
        if is_consulting_company(r.get("company", ""), CONSULTING_FIRMS)
    )
    n = len(hist)
    return product_count / n, consulting_count / n


def feat_location_score(cand: dict) -> float:
    """
    [18] Location fit: 1.0 = in target city, 0.5 = willing to relocate,
    0.0 = outside India with no relocation.
    JD: Pune/Noida preferred. Hyderabad, Bangalore, Mumbai, Delhi NCR welcome.
    Outside India: case-by-case, no visa sponsorship.
    """
    p       = cand["profile"]
    sig     = cand["redrob_signals"]
    loc_str = (p.get("location", "") + " " + p.get("country", "")).lower()

    in_target = any(t in loc_str for t in TARGET_LOCATIONS)
    if in_target:
        return 1.0
    if sig.get("willing_to_relocate"):
        return 0.5
    # Outside India entirely — low but not zero
    if "india" not in loc_str:
        return 0.1
    return 0.3


def feat_availability(cand: dict) -> float:
    """
    [19] Exponential decay availability.
    EDA: 56% of candidates inactive >90 days.
    JD: explicitly states inactive candidates are not available.
    """
    sig = cand["redrob_signals"]
    d   = days_since(sig.get("last_active_date", "2020-01-01"))
    return availability_score(d, lambda_decay=AVAILABILITY["decay_lambda"])


def feat_open_to_work(cand: dict) -> float:
    """[20] Binary: open_to_work_flag. EDA: only 35.3% set this."""
    return 1.0 if cand["redrob_signals"].get("open_to_work_flag") else 0.0


def feat_notice_score(cand: dict) -> float:
    """
    [21] Inverse notice period score.
    JD: 'We'd love sub-30-day notice. Can buy out up to 30 days.
    30+ day candidates still in scope but bar gets higher.'
    EDA: mean=87d, median=90d. Only 13.8% have ≤30d.
    Formula: score = max(0, 1 - notice_days/150)
    150 = max notice in dataset.
    """
    notice = cand["redrob_signals"].get("notice_period_days", 90) or 90
    return max(0.0, 1.0 - notice / 150.0)


def feat_response_rate(cand: dict) -> float:
    """[22] Recruiter response rate (already 0-1 in dataset)."""
    rr = cand["redrob_signals"].get("recruiter_response_rate", 0.44) or 0.0
    return float(rr)


def feat_github_score(cand: dict) -> float:
    """
    [23] GitHub activity normalized 0-1. -1 = no account → 0.
    EDA: 64.6% have no GitHub. JD values open-source contributions.
    """
    gh = cand["redrob_signals"].get("github_activity_score", -1)
    if gh is None or gh < 0:
        return 0.0
    return min(gh / 100.0, 1.0)


def feat_education_score(cand: dict) -> float:
    """
    [24] Education tier score.
    tier_1 (IIT/IISc) = 1.0, tier_2 = 0.75, tier_3 = 0.50,
    tier_4 = 0.25, no_edu = 0.0.
    EDA: education field is not a strong signal (uniform distribution).
    Tier matters more than field for this role.
    """
    edu = cand.get("education", [])
    if not edu:
        return 0.0
    tier_scores = {"tier_1": 1.0, "tier_2": 0.75, "tier_3": 0.50,
                   "tier_4": 0.25, "tier_5": 0.10}
    best = 0.0
    for e in edu:
        t = e.get("tier", "")
        best = max(best, tier_scores.get(t, 0.0))
    return best


# ─────────────────────────────────────────────────────────────────
# MAIN EXTRACTOR
# ─────────────────────────────────────────────────────────────────

def extract_features(cand: dict, vocab: dict, jd: dict) -> np.ndarray:
    """
    Extract 25-dim feature vector for a single candidate.
    Returns float32 numpy array of shape (25,).
    """
    vec = np.zeros(N_FEATS, dtype=np.float32)

    vec[0]  = feat_title_relevance(cand)
    vec[1]  = feat_is_disqualified_title(cand)
    vec[2]  = feat_yoe_score(cand)
    vec[3]  = feat_ai_skill_idf(cand, vocab)
    vec[4]  = feat_retrieval_skill_idf(cand, vocab)

    hard, ideal, neg = feat_jd_section_scores(cand, jd)
    vec[5]  = hard
    vec[6]  = ideal
    vec[7]  = neg

    prod, ctx, stuffer = feat_skill_evidence(cand)
    vec[8]  = prod
    vec[9]  = ctx
    vec[10] = stuffer

    vec[11] = feat_is_pure_consulting(cand)
    vec[12] = feat_is_cv_speech_dominant(cand)
    vec[13] = feat_is_framework_enthusiast(cand)
    vec[14] = feat_is_title_chaser(cand)
    vec[15] = feat_is_honeypot(cand)

    prod_ratio, cons_ratio = feat_company_type_ratios(cand)
    vec[16] = prod_ratio
    vec[17] = cons_ratio

    vec[18] = feat_location_score(cand)
    vec[19] = feat_availability(cand)
    vec[20] = feat_open_to_work(cand)
    vec[21] = feat_notice_score(cand)
    vec[22] = feat_response_rate(cand)
    vec[23] = feat_github_score(cand)
    vec[24] = feat_education_score(cand)

    return vec


def build_feature_matrix(
    candidates: list,
    vocab: dict,
    jd: dict,
    verbose: bool = True,
) -> tuple:
    """
    Build feature matrix for all candidates.
    Returns:
      matrix    — numpy float32 array, shape (N, 25)
      cand_ids  — list of candidate_id strings (row index → id)
    """
    N      = len(candidates)
    matrix = np.zeros((N, N_FEATS), dtype=np.float32)
    ids    = []

    iterator = tqdm(candidates, desc="  Extracting features", unit="cand") \
               if verbose else candidates

    for i, cand in enumerate(iterator):
        matrix[i] = extract_features(cand, vocab, jd)
        ids.append(cand.get("candidate_id", f"CAND_{i:07d}"))

    return matrix, ids


# ─────────────────────────────────────────────────────────────────
# STATS REPORTER
# ─────────────────────────────────────────────────────────────────

def report_matrix_stats(matrix: np.ndarray, ids: list) -> None:
    N = len(ids)
    print(f"\n  Feature matrix: {matrix.shape}  dtype={matrix.dtype}")
    print(f"\n  {'Feature':<30s}  {'Mean':>7s}  {'Std':>7s}  {'Min':>7s}  {'Max':>7s}")
    print("  " + "-" * 65)
    for i, name in enumerate(FEATURE_NAMES):
        col = matrix[:, i]
        print(f"  {name:<30s}  {col.mean():>7.3f}  {col.std():>7.3f}  "
              f"{col.min():>7.3f}  {col.max():>7.3f}")

    # Gate statistics
    disq_title = int(matrix[:, 1].sum())
    honeypot   = int(matrix[:, 15].sum())
    consulting = int(matrix[:, 11].sum())
    cv_speech  = int(matrix[:, 12].sum())
    fw_enth    = int(matrix[:, 13].sum())

    print(f"\n  Gate statistics:")
    print(f"    Disqualified titles   : {disq_title:>7,}  ({disq_title/N*100:.1f}%)")
    print(f"    Honeypot signals      : {honeypot:>7,}  ({honeypot/N*100:.1f}%)")
    print(f"    Pure consulting       : {consulting:>7,}  ({consulting/N*100:.1f}%)")
    print(f"    CV/Speech dominant    : {cv_speech:>7,}  ({cv_speech/N*100:.1f}%)")
    print(f"    Framework enthusiast  : {fw_enth:>7,}  ({fw_enth/N*100:.1f}%)")

    # Any gate = eliminated
    gate_mask = (
        (matrix[:, 1] > 0.5) |   # disqualified title
        (matrix[:, 15] > 0.5) |  # honeypot
        (matrix[:, 11] > 0.5) |  # pure consulting
        (matrix[:, 12] > 0.5)    # cv/speech
    )
    eliminated = int(gate_mask.sum())
    print(f"    Total eliminated      : {eliminated:>7,}  ({eliminated/N*100:.1f}%)")
    print(f"    Eligible for ranking  : {N-eliminated:>7,}  ({(N-eliminated)/N*100:.1f}%)")


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 60)
    print("  Feature Extractor — building 25-dim candidate matrix")
    print("=" * 60 + "\n")

    # Load data
    candidates = load_candidates(CANDIDATES_FILE, SAMPLE_FILE, verbose=True)
    vocab      = load_skill_vocab(SKILL_VOCAB_FILE)
    jd         = parse_jd()

    print(f"\n  Vocab loaded: {len(vocab)} skills")
    print(f"  JD parsed: {len(jd['positive_terms'])} positive terms, "
          f"{len(jd['negative_terms'])} negative terms")

    # Build matrix
    print(f"\n  Building feature matrix for {len(candidates):,} candidates ...")
    matrix, ids = build_feature_matrix(candidates, vocab, jd, verbose=True)

    # Report
    report_matrix_stats(matrix, ids)

    # Save
    save_pickle(matrix, FEATURE_MATRIX_FILE)
    save_pickle(ids,    CANDIDATE_IDS_FILE)

    print(f"\n  Done.")
    print(f"  feature_matrix : {FEATURE_MATRIX_FILE}")
    print(f"  candidate_ids  : {CANDIDATE_IDS_FILE}")


if __name__ == "__main__":
    main()
