"""
config.py
=========
Single source of truth for all paths, scoring weights,
thresholds, and flags.

Design principle: nothing is hardcoded elsewhere.
Every tunable number lives here with a justification comment.
"""

from pathlib import Path

# ─────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────
ROOT         = Path(__file__).parent
ASSETS_DIR   = ROOT / "INDIA_RUNS_Assets"
EDA_DIR      = ROOT / "eda_outputs"
MODELS_DIR   = ROOT / "models"
ARTIFACTS_DIR = ROOT / "artifacts"

ARTIFACTS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# Input data
CANDIDATES_FILE = ASSETS_DIR / "candidates.jsonl"
SAMPLE_FILE     = ASSETS_DIR / "sample_candidates.json"
JD_FILE         = ASSETS_DIR / "job_description.docx"
SCHEMA_FILE     = ASSETS_DIR / "candidate_schema.json"

# EDA outputs (produced by eda_01)
SKILL_VOCAB_FILE = EDA_DIR / "skill_vocabulary.json"

# Offline artifacts (produced by offline pipeline)
FEATURE_MATRIX_FILE  = ARTIFACTS_DIR / "feature_matrix.pkl"
CANDIDATE_IDS_FILE   = ARTIFACTS_DIR / "candidate_ids.pkl"
FAISS_INDEX_FILE     = ARTIFACTS_DIR / "faiss_index.bin"
KG_FEATURES_FILE     = ARTIFACTS_DIR / "kg_features.pkl"
JD_EMBEDDING_FILE    = ARTIFACTS_DIR / "jd_embedding.npy"
ROLE_EMBEDDINGS_FILE = ARTIFACTS_DIR / "role_embeddings.pkl"

# ONNX model files
MINILM_ONNX   = MODELS_DIR / "minilm_l6.onnx"
T5_ONNX       = MODELS_DIR / "t5_small_int8.onnx"
MINILM_TOKDIR = MODELS_DIR / "minilm_tokenizer"   # saved tokenizer dir
T5_TOKDIR     = MODELS_DIR / "t5_tokenizer"

# Output
SUBMISSION_FILE = ROOT / "submission.csv"

# ─────────────────────────────────────────────────────────────────
# EMBEDDING CONFIG
# ─────────────────────────────────────────────────────────────────
# all-MiniLM-L6-v2: 384-dim, 22MB ONNX, ~8ms/batch CPU
EMBEDDING_MODEL_HF  = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM       = 384
EMBEDDING_BATCH_SIZE = 512   # tune down if RAM is tight

# T5-small for reasoning (int8 quantized ~60MB)
REASONING_MODEL_HF  = "google/t5-small"
REASONING_MAX_INPUT = 256    # tokens
REASONING_MAX_OUTPUT = 80    # tokens — one dense sentence

# ─────────────────────────────────────────────────────────────────
# SCORING WEIGHTS
# Justification for each weight documented inline.
# These compose MULTIPLICATIVELY, not additively:
#   final = gate × semantic × structural × availability × behavioral
# A zero in gate propagates to zero final score.
# ─────────────────────────────────────────────────────────────────
SCORING = {
    # ── Semantic score (FAISS cosine similarity)
    # JD-critical skills appear in only 3-16% of profiles.
    # Embedding over full career narrative captures semantic
    # equivalence even when exact keywords are missing.
    "semantic_weight": 0.25,

    # ── Structural score (KG arc alignment)
    # EDA showed career arc > skill inventory as signal.
    # Paper result: causal > temporal for coherence.
    # Highest weight — the novel contribution.
    "structural_weight": 0.30,

    # ── Skill IDF score (corpus-derived, no hardcoding)
    # IDF from 100k corpus gives mathematically justified
    # rarity-based weights. Context-verified skills get
    # 3× boost over self-reported skills.
    "skill_idf_weight": 0.20,

    # ── Availability multiplier (applied last, multiplicative)
    # JD explicitly: "inactive for 6 months = not available."
    # 56% of corpus inactive >90 days — not a tie-breaker,
    # a primary signal. Applied as multiplier not additive.
    "availability_weight": 0.15,   # relative contribution

    # ── Behavioral quality boost
    # GitHub score, assessment scores, response rate.
    # Sparse (only 35% have GitHub) so low weight.
    "behavioral_weight": 0.10,

    # Evidence level multipliers for skill scoring
    "skill_evidence_multipliers": {
        "PRODUCTION_PROVEN":  3.0,  # in career text + production cues
        "CONTEXT_VERIFIED":   1.5,  # in career text only
        "SELF_REPORTED":      0.4,  # skills[] array only — low trust
    },
}

# ─────────────────────────────────────────────────────────────────
# AVAILABILITY DECAY
# e^(-λt) where t = days inactive
# Calibrated so: 30d→0.95, 90d→0.80, 180d→0.40, 365d→0.16
# This matches JD's stated concern precisely.
# ─────────────────────────────────────────────────────────────────
AVAILABILITY = {
    "decay_lambda":         0.005,  # λ in e^(-λt)
    "otw_penalty":          0.88,   # open_to_work=False → ×0.88
    "notice_penalty_days":  90,     # notice >90d → apply penalty
    "notice_penalty_mult":  0.88,
    "low_response_threshold": 0.20, # response_rate < 0.20 → ×0.85
    "low_response_mult":    0.85,
    "min_availability":     0.10,   # floor — never fully zero
}

# ─────────────────────────────────────────────────────────────────
# HARD FILTER THRESHOLDS
# Binary gates — candidates failing any gate get score=0.
# ─────────────────────────────────────────────────────────────────
HARD_FILTER = {
    # Minimum skills to bother scoring
    "min_skills":        3,
    # Wrapper-only: has wrapper tools AND no depth skills in career
    "wrapper_tools":     {"langchain", "llamaindex", "flowise", "n8n", "langflow"},
    "depth_skill_cues":  {
        "pytorch", "tensorflow", "sentence-transformers",
        "sentence transformers", "hugging face", "transformers",
        "fine-tuning", "finetuning", "faiss", "embeddings",
        "vector", "custom model", "from scratch", "pre-training",
    },
    # Title-chaser: ≥3 roles with tenure < threshold months
    "title_chaser_tenure_months": 20,
    "title_chaser_min_flagged":   3,
    # Closed-source: 5yr+ career, no GitHub, no open-source text
    "closed_source_min_yoe":      5,
    # Honeypot: skill duration exceeds career length by this margin
    "honeypot_duration_slack_months": 6,
}

# ─────────────────────────────────────────────────────────────────
# CAREER EVENT TYPES
# Used by career_kg.py. These are the "event taxonomy" from the
# paper, adapted to career domain.
# ─────────────────────────────────────────────────────────────────
CAREER_EVENT_TYPES = [
    "CORE_ML_ROLE",           # ML/NLP/AI/RecSys/Search engineer
    "DATA_ENGINEERING",       # Data eng, analytics, ETL
    "SOFTWARE_ENGINEERING",   # Backend/fullstack/platform
    "LEADERSHIP_EVENT",       # Tech lead, manager, architect
    "RESEARCH_WORK",          # Research eng, scientist, intern
    "CONSULTING_STINT",       # TCS/Infosys/Wipro/Accenture etc
    "PRODUCT_DOMAIN",         # Fintech/SaaS/EdTech product co
    "OTHER_ROLE",             # Everything else
]

# Ideal career arc for this JD (derived from JD Section B)
# Ordered by desirability — used to compute arc alignment score.
IDEAL_ARC_DISTRIBUTION = {
    "CORE_ML_ROLE":        0.45,  # majority of career in ML/AI
    "DATA_ENGINEERING":    0.15,  # data background helps
    "SOFTWARE_ENGINEERING":0.20,  # product engineer history
    "LEADERSHIP_EVENT":    0.10,  # some lead experience
    "RESEARCH_WORK":       0.05,  # small research exposure ok
    "PRODUCT_DOMAIN":      0.05,  # product-co context
    "CONSULTING_STINT":    0.00,  # JD explicitly says no
    "OTHER_ROLE":          0.00,  # no credit
}

# Causal arc transition scores — how much does moving from
# role A to role B represent a coherent career progression?
# Values from -1.0 (regression) to +1.0 (ideal progression)
# Derived from JD Section B narrative logic.
CAUSAL_TRANSITION_SCORES = {
    ("DATA_ENGINEERING",    "CORE_ML_ROLE"):        +0.90,
    ("SOFTWARE_ENGINEERING","CORE_ML_ROLE"):        +0.85,
    ("RESEARCH_WORK",       "CORE_ML_ROLE"):        +0.80,
    ("CORE_ML_ROLE",        "CORE_ML_ROLE"):        +0.70,
    ("CORE_ML_ROLE",        "LEADERSHIP_EVENT"):    +0.95,
    ("SOFTWARE_ENGINEERING","LEADERSHIP_EVENT"):    +0.60,
    ("CONSULTING_STINT",    "CORE_ML_ROLE"):        +0.40,  # possible
    ("CONSULTING_STINT",    "CONSULTING_STINT"):    -0.80,  # JD DQ
    ("OTHER_ROLE",          "CONSULTING_STINT"):    -0.60,
    ("LEADERSHIP_EVENT",    "LEADERSHIP_EVENT"):    -0.20,  # no coding
    ("OTHER_ROLE",          "OTHER_ROLE"):          -0.50,
}
DEFAULT_TRANSITION_SCORE = 0.10  # unknown transition

# ─────────────────────────────────────────────────────────────────
# FAISS CONFIG
# ─────────────────────────────────────────────────────────────────
FAISS = {
    "top_k_retrieval":  5000,   # candidates passed to full scorer
    "top_k_kg":          500,   # candidates for deep KG analysis
    "top_k_output":      100,   # final submission size
    "index_type":       "flat", # IndexFlatIP — exact, no approx error
}

# ─────────────────────────────────────────────────────────────────
# CONSULTING FIRMS  (JD explicit disqualifier)
# ─────────────────────────────────────────────────────────────────
CONSULTING_FIRMS = {
    "tcs", "infosys", "wipro", "accenture", "cognizant",
    "capgemini", "hcl", "tech mahindra", "mphasis", "hexaware",
    "ltimindtree", "mindtree", "tata consultancy", "deloitte",
    "ibm global", "pwc", "kpmg", "ernst", "ey ",
}

# ─────────────────────────────────────────────────────────────────
# LOCATION CONFIG
# ─────────────────────────────────────────────────────────────────
TARGET_LOCATIONS = {
    "noida", "pune", "delhi", "hyderabad", "bangalore",
    "bengaluru", "mumbai", "gurgaon", "gurugram", "ncr",
}

# ─────────────────────────────────────────────────────────────────
# PRODUCTION CUE WORDS
# Used to detect production-proven skills in career text.
# ─────────────────────────────────────────────────────────────────
PRODUCTION_CUES = {
    "production", "deployed", "deployment", "serving", "inference",
    "latency", "throughput", "scale", "real users", "real-time",
    "a/b", "experiment", "billion", "million", "queries", "requests",
    "monitoring", "drift", "regression", "pipeline", "online",
    "shipped", "launched", "live", "customers", "traffic",
}
