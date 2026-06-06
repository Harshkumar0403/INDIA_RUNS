"""
eda_01_corpus_analysis.py  (v2 — data-driven, JD-aware)
=========================================================
Redrob Hackathon — Corpus & Profile Statistics

KEY CHANGES FROM v1:
  1. NO hardcoded skill taxonomies.
     Skills are extracted from the corpus itself, frequency-ranked,
     and tiered by IDF-style rarity into GENERIC / DOMAIN / SPECIALIST.
     Labels (AI/ML, Retrieval, etc.) are applied AFTER frequency analysis
     using lightweight substring matching — not as fixed membership sets.

  2. JD is parsed into FOUR signed sections with explicit polarity:
       Section A — Hard requirements      (+++ weight)
       Section B — Ideal profile          (++  weight)
       Section C — Nice to have           (+   weight)
       Section D — Explicit DO NOT WANT   (--- heavy penalty / disqualifier)
     Keywords from Section D are treated as RED FLAGS, not positive signals.

  3. Skill-in-context scoring:
     Every skill is scored differently depending on where it appears:
       — skill listed in skills[] array only          → "self-reported"
       — skill also appears in career description     → "context-verified"
       — skill appears in description WITH production cues → "production-proven"
     This kills the keyword-stuffer pattern at analysis time.

Run from: INDIA_RUNS/
Usage:    python eda_01_corpus_analysis.py
Outputs:  eda_outputs/corpus_stats.txt
"""

import json
import re
import sys
import math
import collections
from datetime import datetime
from pathlib import Path

# ── paths ─────────────────────────────────────────────────────────────────────
ASSETS          = Path("INDIA_RUNS_Assets")
CANDIDATES_FILE = ASSETS / "candidates.jsonl"
SAMPLE_FILE     = ASSETS / "sample_candidates.json"
OUT_DIR         = Path("eda_outputs")
OUT_DIR.mkdir(exist_ok=True)

TODAY = datetime(2026, 6, 2)

# ─────────────────────────────────────────────────────────────────────────────
# JD STRUCTURED PARSE
# Four sections with explicit polarity.
# These are conceptual anchors — keywords derived FROM the JD text,
# NOT manually invented taxonomy.
# ─────────────────────────────────────────────────────────────────────────────
JD_SECTIONS = {

    # ── A: HARD REQUIREMENTS — must have, high weight ────────────────────────
    "HARD_REQUIREMENT": {
        "raw_phrases": [
            "production experience with embeddings",
            "embeddings-based retrieval",
            "sentence-transformers", "openai embeddings", "bge", "e5",
            "embedding drift", "index refresh", "retrieval-quality regression",
            "vector databases", "hybrid search infrastructure",
            "pinecone", "weaviate", "qdrant", "milvus", "opensearch",
            "elasticsearch", "faiss",
            "operational experience",
            "strong python", "code quality",
            "evaluation frameworks for ranking",
            "ndcg", "mrr", "map",
            "offline-to-online correlation", "a/b test interpretation",
            "ranking system",
        ],
        "weight": 1.0,
        "polarity": "positive",
    },

    # ── B: IDEAL PROFILE — strong positive but not hard gate ─────────────────
    "IDEAL_PROFILE": {
        "raw_phrases": [
            "applied ml", "applied ai",
            "product companies",           # NOT pure services
            "end-to-end ranking",
            "end-to-end search",
            "end-to-end recommendation",
            "real users", "meaningful scale",
            "hybrid vs dense",
            "offline vs online",
            "when to fine-tune", "when to prompt",
            "systems they actually built",
            "6-8 years", "4-5 years in applied",
            "noida", "pune",
        ],
        "weight": 0.6,
        "polarity": "positive",
    },

    # ── C: NICE TO HAVE — soft bonus, zero penalty if absent ─────────────────
    "NICE_TO_HAVE": {
        "raw_phrases": [
            "lora", "qlora", "peft", "llm fine-tuning",
            "learning-to-rank", "xgboost-based",
            "hr-tech", "recruiting tech", "marketplace products",
            "distributed systems", "large-scale inference",
            "open-source contributions",
        ],
        "weight": 0.25,
        "polarity": "positive",
    },

    # ── D: EXPLICIT DO NOT WANT — presence is a RED FLAG / disqualifier ──────
    # CRITICAL: these appear in the JD but carry NEGATIVE polarity.
    # A naive keyword extractor would treat "LangChain" as a positive signal
    # because it's in the JD. It is NOT — it appears in the rejection section.
    "EXPLICIT_NEGATIVE": {
        "raw_phrases": [
            # title-chaser pattern
            "senior staff principal",
            "switching companies every 1.5 years",
            "optimizing for title",
            # framework enthusiast pattern
            "langchain tutorials",
            "how i used langchain",
            "how i used llamaindex",
            "framework demo",
            "framework enthusiast",
            # pure consulting
            "tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini",
            "hcl", "tech mahindra", "mphasis", "hexaware",
            # wrong domain
            "computer vision without nlp",
            "speech without nlp",
            "robotics without nlp",
            "primary expertise is computer vision",
            "primary expertise is speech",
            # closed-source isolation
            "entirely on closed-source",
            "closed-source proprietary systems",
            "no external validation",
            # wrapper-only LLM (no depth)
            "langchain",      # in skills-only context = framework enthusiast flag
            "llamaindex",     # same — wrapper without foundational skills
            "flowise",
            "n8n",
        ],
        "weight": -1.0,   # negative weight — PENALISES score
        "polarity": "negative",
    },
}

# Flatten for fast lookup — preserving polarity
JD_POSITIVE_TERMS  = set()
JD_NEGATIVE_TERMS  = set()
JD_HARD_REQ_TERMS  = set()

for section_name, section_data in JD_SECTIONS.items():
    for phrase in section_data["raw_phrases"]:
        tokens = phrase.lower().split()
        # use first meaningful token as the lookup key
        key = tokens[0] if tokens else phrase.lower()
        if section_data["polarity"] == "positive":
            JD_POSITIVE_TERMS.add(key)
            if section_name == "HARD_REQUIREMENT":
                JD_HARD_REQ_TERMS.add(key)
        else:
            JD_NEGATIVE_TERMS.add(key)

# Production context cue words — used for context-verification
PRODUCTION_CUES = {
    "production", "deployed", "deployment", "serving", "inference",
    "latency", "throughput", "scale", "real users", "real-time",
    "a/b", "experiment", "billion", "million", "queries", "requests",
    "monitoring", "drift", "regression", "pipeline", "online",
}

# Title-chaser pattern: short tenures + escalating titles
SENIORITY_LADDER = ["junior","mid","senior","staff","principal","director","vp","head"]

# Consulting firms for arc analysis
CONSULTING_FIRMS_SET = {
    "tcs","infosys","wipro","accenture","cognizant","capgemini","hcl",
    "tech mahindra","mphasis","hexaware","ltimindtree","mindtree",
    "deloitte","ibm","pwc","kpmg","tata consultancy",
}

# Wrapper-only tools — presence without depth skills = framework enthusiast flag
WRAPPER_TOOLS = {"langchain","llamaindex","flowise","n8n","langflow","dspy"}
DEPTH_SKILLS  = {
    "pytorch","tensorflow","sentence-transformers","sentence transformers",
    "hugging face","transformers","fine-tuning","finetuning","faiss",
    "embeddings","vector","custom","from scratch","pre-training",
}


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
STOP_WORDS = {
    "the","a","an","and","or","in","of","to","for","with","at","by","from",
    "as","is","was","were","are","be","been","being","have","has","had",
    "this","that","these","those","their","they","i","my","our","we","it",
    "its","on","up","also","which","who","what","when","where","how","all",
    "more","into","than","s","team","work","worked","working","including",
    "across","within","through","between","multiple","various","using","used",
    "use","part","while","over","each","new","key","well","based",
    "experience","years","built","build","building","developed","helped",
    "support","led","managed","manage","worked","strong","good","great",
    "responsible","responsibilities","role","company","join","joined",
}

def tokenize(text):
    return re.findall(r"\b\w+\b", (text or "").lower())

def clean_tokens(text):
    return [w for w in re.findall(r"\b[a-z]{3,}\b", (text or "").lower())
            if w not in STOP_WORDS]

def days_inactive(date_str):
    try:
        return (TODAY - datetime.strptime(date_str, "%Y-%m-%d")).days
    except Exception:
        return 9999

def pct(n, total):
    return f"{n/total*100:.1f}%" if total else "0.0%"

def bar(n, max_n, width=25):
    filled = round((n / max_n) * width) if max_n else 0
    return "█" * filled + "░" * (width - filled)

def stats_line(lst, label):
    lst = [x for x in lst if x is not None]
    if not lst:
        return f"  {label}: no data"
    s    = sorted(lst)
    mean = sum(lst) / len(lst)
    med  = s[len(s) // 2]
    return (f"  {label:<48s}  "
            f"mean={mean:8.1f}  median={med:8.1f}  "
            f"min={min(lst):8.1f}  max={max(lst):8.1f}")


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────
def load_candidates(max_n=None):
    candidates = []
    if CANDIDATES_FILE.exists():
        print(f"  Loading {CANDIDATES_FILE}  (~30s for 100k) ...")
        with open(CANDIDATES_FILE, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    candidates.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
                if max_n and i + 1 >= max_n:
                    break
        print(f"  Loaded {len(candidates):,} candidates from jsonl")
    elif SAMPLE_FILE.exists():
        print(f"  Falling back to {SAMPLE_FILE}")
        with open(SAMPLE_FILE) as f:
            candidates = json.load(f)
        print(f"  Loaded {len(candidates)} candidates from sample json")
    else:
        print("ERROR: no candidate data found.")
        sys.exit(1)
    return candidates


# ─────────────────────────────────────────────────────────────────────────────
# CHANGE 1: DATA-DRIVEN SKILL VOCABULARY BUILDER
# Reads all skills from the corpus itself.
# Returns tiered frequency buckets — no hardcoded taxonomy.
# ─────────────────────────────────────────────────────────────────────────────
def build_skill_vocabulary(cands):
    """
    Extract every skill name from the corpus.
    Compute:
      - raw frequency (how many times does this skill appear across all candidates)
      - document frequency (how many candidates list this skill)
      - IDF score (log(N/df)) — rare skills = high IDF = more discriminating
    
    Tier assignment is PURELY data-driven:
      TIER_1 GENERIC    : top 10% by doc-frequency  — everyone has it (Python, SQL)
      TIER_2 DOMAIN     : 10–40% doc-frequency      — domain indicators (PyTorch, Spark)
      TIER_3 SPECIALIST : bottom 60%                — rare, high-signal (FAISS, BM25)
    
    Then apply lightweight JD-polarity labeling:
      'JD_POSITIVE' if the skill name substring-matches JD hard/ideal requirements
      'JD_NEGATIVE' if it substring-matches JD explicit negatives
      'NEUTRAL'     otherwise
    """
    N  = len(cands)
    skill_doc_freq  = collections.Counter()   # df: # candidates with this skill
    skill_raw_freq  = collections.Counter()   # tf: total occurrences
    skill_avg_dur   = collections.defaultdict(list)
    skill_avg_end   = collections.defaultdict(list)
    skill_prof_dist = collections.defaultdict(collections.Counter)

    for c in cands:
        seen_in_this_doc = set()
        for s in c.get("skills", []):
            name = (s.get("name") or "").strip()
            if not name:
                continue
            skill_raw_freq[name] += 1
            if name not in seen_in_this_doc:
                skill_doc_freq[name] += 1
                seen_in_this_doc.add(name)
            if s.get("duration_months"):
                skill_avg_dur[name].append(s["duration_months"])
            if s.get("endorsements") is not None:
                skill_avg_end[name].append(s["endorsements"])
            prof = s.get("proficiency", "unknown") or "unknown"
            skill_prof_dist[name][prof] += 1

    # IDF for each skill
    skill_idf = {
        sk: math.log(N / df) if df > 0 else 0
        for sk, df in skill_doc_freq.items()
    }

    # tier boundaries — purely by doc-frequency percentile
    sorted_by_df = sorted(skill_doc_freq.items(), key=lambda x: -x[1])
    total_unique = len(sorted_by_df)
    tier1_cutoff = max(1, int(total_unique * 0.10))   # top 10% by df
    tier2_cutoff = max(1, int(total_unique * 0.40))   # top 10-40%

    skill_tier = {}
    for rank, (sk, _) in enumerate(sorted_by_df):
        if rank < tier1_cutoff:
            skill_tier[sk] = "TIER_1_GENERIC"
        elif rank < tier2_cutoff:
            skill_tier[sk] = "TIER_2_DOMAIN"
        else:
            skill_tier[sk] = "TIER_3_SPECIALIST"

    # JD polarity labeling — substring match against JD phrases
    def jd_label(skill_name):
        sk_low = skill_name.lower()
        # check negatives first (higher priority)
        for neg_term in JD_NEGATIVE_TERMS:
            if neg_term in sk_low or sk_low in neg_term:
                return "JD_NEGATIVE"
        for pos_term in JD_POSITIVE_TERMS:
            if pos_term in sk_low or sk_low in pos_term:
                # further check: is this a hard requirement?
                for hard_term in JD_HARD_REQ_TERMS:
                    if hard_term in sk_low or sk_low in hard_term:
                        return "JD_HARD_REQ"
                return "JD_POSITIVE"
        return "NEUTRAL"

    # Assemble vocabulary dict
    vocab = {}
    for sk in skill_doc_freq:
        dur_list = skill_avg_dur[sk]
        end_list = skill_avg_end[sk]
        vocab[sk] = {
            "doc_freq":   skill_doc_freq[sk],
            "raw_freq":   skill_raw_freq[sk],
            "idf":        round(skill_idf[sk], 4),
            "tier":       skill_tier[sk],
            "jd_label":   jd_label(sk),
            "avg_dur_mo": round(sum(dur_list)/len(dur_list), 1) if dur_list else None,
            "avg_end":    round(sum(end_list)/len(end_list), 1) if end_list else None,
            "prof_dist":  dict(skill_prof_dist[sk]),
        }

    return vocab, N


# ─────────────────────────────────────────────────────────────────────────────
# CHANGE 2: JD-AWARE KEYWORD ANALYSIS WITH POLARITY
# ─────────────────────────────────────────────────────────────────────────────
def build_jd_keyword_profile(cands):
    """
    For each JD section independently, measure how many candidates
    match that section's terms — and with what polarity.
    Returns structured match counts per section.
    """
    N = len(cands)
    section_matches = {sec: collections.Counter() for sec in JD_SECTIONS}

    for c in cands:
        p      = c.get("profile", {})
        hist   = c.get("career_history", [])
        skills = c.get("skills", [])

        # full text — but distinguish sources
        summary_text  = (p.get("summary", "") or "").lower()
        career_text   = " ".join(
            (r.get("description", "") or "").lower() for r in hist
        )
        skills_text   = " ".join(
            (s.get("name", "") or "").lower() for s in skills
        )
        full_text = f"{summary_text} {career_text} {skills_text}"

        for sec_name, sec_data in JD_SECTIONS.items():
            for phrase in sec_data["raw_phrases"]:
                if phrase.lower() in full_text:
                    section_matches[sec_name][phrase] += 1

    return section_matches, N


# ─────────────────────────────────────────────────────────────────────────────
# CHANGE 3: SKILL-IN-CONTEXT SCORER
# For every candidate, verify each skill against career descriptions.
# Returns three levels of evidence for each skill mention.
# ─────────────────────────────────────────────────────────────────────────────
def classify_skill_evidence(cand):
    """
    For a single candidate, classify each skill by evidence level:
      PRODUCTION_PROVEN  — skill in description + production cue words nearby
      CONTEXT_VERIFIED   — skill mentioned in career description text
      SELF_REPORTED      — skill only in skills[] array, not in descriptions
    
    Returns:
      dict mapping skill_name → evidence_level
    """
    hist       = cand.get("career_history", [])
    career_text = " ".join(
        (r.get("description", "") or "").lower() for r in hist
    )
    prod_text   = " ".join(
        (r.get("description", "") or "").lower()
        for r in hist
        if any(cue in (r.get("description", "") or "").lower()
               for cue in PRODUCTION_CUES)
    )

    evidence = {}
    for s in cand.get("skills", []):
        name    = (s.get("name") or "").strip()
        name_lw = name.lower()
        if not name:
            continue

        if name_lw in prod_text:
            evidence[name] = "PRODUCTION_PROVEN"
        elif name_lw in career_text:
            evidence[name] = "CONTEXT_VERIFIED"
        else:
            evidence[name] = "SELF_REPORTED"

    return evidence


def aggregate_skill_evidence(cands):
    """
    Across all candidates, count how many have each skill at each evidence level.
    This directly shows us keyword stuffers: skills with high SELF_REPORTED
    but low CONTEXT_VERIFIED indicate self-reporting inflation.
    """
    evidence_counts = collections.defaultdict(collections.Counter)

    for c in cands:
        ev = classify_skill_evidence(c)
        for skill, level in ev.items():
            evidence_counts[skill][level] += 1

    return evidence_counts


# ─────────────────────────────────────────────────────────────────────────────
# NEGATIVE PATTERN DETECTORS  (from JD Section D)
# ─────────────────────────────────────────────────────────────────────────────
def detect_title_chaser(cand):
    """
    JD says: switching companies every 1.5 years to chase titles.
    Signal: short tenures (<20 months) + title seniority increases + ≥3 jobs
    """
    hist = cand.get("career_history", [])
    if len(hist) < 3:
        return False, ""

    short_tenures = sum(1 for r in hist
                        if 0 < (r.get("duration_months") or 0) < 20)
    title_levels  = []
    for r in hist:
        t = (r.get("title") or "").lower()
        for i, level in enumerate(SENIORITY_LADDER):
            if level in t:
                title_levels.append(i)
                break

    if short_tenures >= 3 and len(title_levels) >= 2:
        if title_levels[-1] > title_levels[0]:   # title level increased
            avg_tenure = sum(
                r.get("duration_months", 0) or 0 for r in hist
            ) / len(hist)
            return True, f"{short_tenures}/{len(hist)} roles <20mo, avg={avg_tenure:.0f}mo, titles escalating"
    return False, ""


def detect_framework_enthusiast(cand):
    """
    JD says: LangChain tutorials, demo blogs, framework without systems thinking.
    Signal: wrapper tools present in skills, no depth skills in career text.
    """
    skill_names_lw = {
        (s.get("name") or "").lower()
        for s in cand.get("skills", [])
    }
    career_text = " ".join(
        (r.get("description", "") or "").lower()
        for r in cand.get("career_history", [])
    )

    has_wrappers = skill_names_lw & WRAPPER_TOOLS
    has_depth    = any(d in career_text for d in DEPTH_SKILLS)

    if has_wrappers and not has_depth:
        return True, f"Wrapper tools: {has_wrappers} | No depth skill evidence in career text"
    return False, ""


def detect_closed_source_isolation(cand):
    """
    JD says: 5+ years entirely on closed-source with no external validation.
    Signal: long career, github_score=-1, no open-source in descriptions.
    """
    yoe    = cand["profile"].get("years_of_experience", 0) or 0
    gh     = cand["redrob_signals"].get("github_activity_score", -1)
    career = " ".join(
        (r.get("description","") or "").lower()
        for r in cand.get("career_history", [])
    )
    open_signals = any(
        kw in career
        for kw in ["open source","open-source","github","arxiv","paper","publication","talk"]
    )

    if yoe >= 5 and gh == -1 and not open_signals:
        return True, f"YoE={yoe:.1f}, no GitHub, no open-source signals in career text"
    return False, ""


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 : CORPUS OVERVIEW  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────
def section1_corpus_overview(cands, out):
    N = len(cands)
    out.append("\n" + "=" * 68)
    out.append("SECTION 1 : CORPUS OVERVIEW")
    out.append("=" * 68)

    summary_lens, career_lens, job_counts = [], [], []
    skill_counts, cert_counts             = [], []
    total_tok_summary = total_tok_career  = 0

    for c in cands:
        p    = c.get("profile", {})
        hist = c.get("career_history", [])
        skls = c.get("skills", [])

        st = tokenize(p.get("summary", ""))
        ct = tokenize(" ".join(r.get("description","") or "" for r in hist))

        summary_lens.append(len(st))
        career_lens.append(len(ct))
        job_counts.append(len(hist))
        skill_counts.append(len(skls))
        cert_counts.append(len(c.get("certifications", [])))
        total_tok_summary += len(st)
        total_tok_career  += len(ct)

    out.append(f"\n  Total candidates               : {N:,}")
    out.append(f"  Total summary tokens           : {total_tok_summary:,}")
    out.append(f"  Total career-desc tokens       : {total_tok_career:,}")
    out.append("")
    out.append(stats_line(summary_lens,  "Summary length (tokens)"))
    out.append(stats_line(career_lens,   "Career desc length (tokens)"))
    out.append(stats_line(job_counts,    "Jobs per candidate"))
    out.append(stats_line(skill_counts,  "Skills per candidate"))
    out.append(stats_line(cert_counts,   "Certifications per candidate"))


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 : TITLE & EXPERIENCE DISTRIBUTION  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────
def section2_titles_experience(cands, out):
    N = len(cands)
    out.append("\n" + "=" * 68)
    out.append("SECTION 2 : TITLE & EXPERIENCE DISTRIBUTION")
    out.append("=" * 68)

    titles  = collections.Counter(
        c["profile"].get("current_title","Unknown") for c in cands
    )
    max_t   = titles.most_common(1)[0][1]
    out.append(f"\n  Top 30 current titles  (N={N:,}):")
    out.append(f"  {'Title':<45s} {'Count':>7s}  {'%':>6s}  Bar")
    out.append("  " + "-" * 78)
    for t, n in titles.most_common(30):
        out.append(f"  {t:<45s} {n:>7,}  {pct(n,N):>6s}  {bar(n,max_t,20)}")

    exp_buckets = collections.Counter()
    exp_vals    = []
    for c in cands:
        yoe = c["profile"].get("years_of_experience", 0) or 0
        exp_vals.append(yoe)
        if   yoe < 2:  exp_buckets["< 2 yrs"]               += 1
        elif yoe < 4:  exp_buckets["2-4 yrs"]               += 1
        elif yoe < 6:  exp_buckets["4-6 yrs"]               += 1
        elif yoe < 9:  exp_buckets["6-9 yrs ← JD sweet spot"] += 1
        elif yoe < 12: exp_buckets["9-12 yrs"]              += 1
        else:          exp_buckets["12+ yrs"]               += 1

    out.append(f"\n  Experience distribution:")
    max_e = max(exp_buckets.values())
    for k in ["< 2 yrs","2-4 yrs","4-6 yrs","6-9 yrs ← JD sweet spot",
              "9-12 yrs","12+ yrs"]:
        n = exp_buckets[k]
        out.append(f"  {k:<38s} {n:>7,}  {pct(n,N):>6s}  {bar(n,max_e,20)}")
    out.append(stats_line(exp_vals, "Experience (years)"))


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 : DATA-DRIVEN SKILL VOCABULARY  [CHANGE 1]
# ─────────────────────────────────────────────────────────────────────────────
def section3_skills_data_driven(cands, vocab, out):
    N = len(cands)
    out.append("\n" + "=" * 68)
    out.append("SECTION 3 : SKILLS ANALYSIS  (data-driven vocabulary, no hardcoding)")
    out.append("=" * 68)

    # ── 3a. Tier distribution ─────────────────────────────────────────────────
    tier_counts = collections.Counter(v["tier"] for v in vocab.values())
    jd_label_counts = collections.Counter(v["jd_label"] for v in vocab.values())

    out.append(f"\n  Unique skills in corpus : {len(vocab):,}")
    out.append(f"\n  Frequency-tier distribution  (derived from corpus IDF):")
    out.append(f"  {'Tier':<30s} {'Unique skills':>14s}  Meaning")
    out.append("  " + "-" * 70)
    tier_meanings = {
        "TIER_1_GENERIC":    "Top 10% by doc-freq — everyone has it (low signal)",
        "TIER_2_DOMAIN":     "10-40% doc-freq — domain indicator (medium signal)",
        "TIER_3_SPECIALIST": "Bottom 60% doc-freq — rare, high-signal skill",
    }
    for tier in ["TIER_1_GENERIC","TIER_2_DOMAIN","TIER_3_SPECIALIST"]:
        n = tier_counts[tier]
        out.append(f"  {tier:<30s} {n:>14,}  {tier_meanings[tier]}")

    # ── 3b. JD polarity distribution ─────────────────────────────────────────
    out.append(f"\n  JD polarity labeling  (substring match against JD sections):")
    label_meanings = {
        "JD_HARD_REQ": "Matches JD 'Must have' section  → high positive weight",
        "JD_POSITIVE": "Matches JD 'Ideal' or 'Nice to have' → moderate boost",
        "JD_NEGATIVE": "Matches JD 'DO NOT WANT' section → RED FLAG / penalty",
        "NEUTRAL":     "Not explicitly mentioned in JD",
    }
    for label in ["JD_HARD_REQ","JD_POSITIVE","JD_NEGATIVE","NEUTRAL"]:
        n = jd_label_counts[label]
        out.append(f"  {label:<18s} {n:>8,} skills   {label_meanings[label]}")

    # ── 3c. Top skills by frequency — with tier & JD label ───────────────────
    out.append(f"\n  Top 50 skills by doc-frequency  (with tier, IDF, JD-label):")
    out.append(f"  {'Skill':<35s} {'DocFreq':>8s}  {'IDF':>6s}  {'AvgDur':>7s}  {'AvgEnd':>7s}  Tier / JD-label")
    out.append("  " + "-" * 95)

    sorted_vocab = sorted(vocab.items(), key=lambda x: -x[1]["doc_freq"])
    for sk, v in sorted_vocab[:50]:
        dur = f"{v['avg_dur_mo']:.0f}mo" if v["avg_dur_mo"] else "  — "
        end = f"{v['avg_end']:.1f}" if v["avg_end"] else "  —"
        jd_marker = {
            "JD_HARD_REQ": " ★★ HARD REQ",
            "JD_POSITIVE": " ★  JD+",
            "JD_NEGATIVE": " ✗  JD-NEG",
            "NEUTRAL":     "",
        }[v["jd_label"]]
        out.append(
            f"  {sk:<35s} {v['doc_freq']:>8,}  {v['idf']:>6.3f}  "
            f"{dur:>7s}  {end:>7s}  {v['tier']}{jd_marker}"
        )

    # ── 3d. JD-NEGATIVE skills found in corpus ───────────────────────────────
    neg_skills = [(sk, v) for sk, v in vocab.items() if v["jd_label"] == "JD_NEGATIVE"]
    out.append(f"\n  JD-NEGATIVE skills in corpus  (these PENALISE score when present):")
    out.append(f"  {'Skill':<35s} {'DocFreq':>8s}  {'% Cands':>8s}")
    out.append("  " + "-" * 55)
    for sk, v in sorted(neg_skills, key=lambda x: -x[1]["doc_freq"]):
        out.append(f"  {sk:<35s} {v['doc_freq']:>8,}  {pct(v['doc_freq'],N):>8s}")

    # ── 3e. TIER_3_SPECIALIST + JD_HARD_REQ  (the golden skills) ─────────────
    golden = [
        (sk, v) for sk, v in vocab.items()
        if v["tier"] == "TIER_3_SPECIALIST" and v["jd_label"] == "JD_HARD_REQ"
    ]
    out.append(f"\n  GOLDEN SKILLS  (TIER_3_SPECIALIST + JD_HARD_REQ):")
    out.append(f"  Rare in corpus AND required by JD — highest discriminating power")
    out.append(f"  {'Skill':<35s} {'DocFreq':>8s}  {'IDF':>6s}")
    out.append("  " + "-" * 55)
    for sk, v in sorted(golden, key=lambda x: -x[1]["idf"]):
        out.append(f"  {sk:<35s} {v['doc_freq']:>8,}  {v['idf']:>6.3f}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 : SKILL-IN-CONTEXT ANALYSIS  [CHANGE 3]
# ─────────────────────────────────────────────────────────────────────────────
def section4_skill_evidence(cands, vocab, out):
    N = len(cands)
    out.append("\n" + "=" * 68)
    out.append("SECTION 4 : SKILL-IN-CONTEXT ANALYSIS  (keyword stuffer detector)")
    out.append("=" * 68)
    out.append("""
  Three evidence levels:
    PRODUCTION_PROVEN  — skill in description + production cue words nearby
    CONTEXT_VERIFIED   — skill name appears in career history text
    SELF_REPORTED      — skill only in skills[] array, not in descriptions
  
  High ratio of SELF_REPORTED → CONTEXT_VERIFIED is the keyword-stuffer signal.
""")

    evidence_counts = aggregate_skill_evidence(cands)

    # Candidate-level evidence quality distribution
    prod_proven_counts  = []
    ctx_verified_counts = []
    self_reported_counts = []

    for c in cands:
        ev   = classify_skill_evidence(c)
        prod_proven_counts.append(sum(1 for l in ev.values() if l == "PRODUCTION_PROVEN"))
        ctx_verified_counts.append(sum(1 for l in ev.values() if l == "CONTEXT_VERIFIED"))
        self_reported_counts.append(sum(1 for l in ev.values() if l == "SELF_REPORTED"))

    out.append(stats_line(prod_proven_counts,   "Production-proven skills per candidate"))
    out.append(stats_line(ctx_verified_counts,  "Context-verified skills per candidate"))
    out.append(stats_line(self_reported_counts, "Self-reported only  skills per candidate"))

    # Stuffer ratio: self-reported / total skills
    stuffer_ratios = []
    for pp, cv, sr in zip(prod_proven_counts, ctx_verified_counts, self_reported_counts):
        total = pp + cv + sr
        if total > 0:
            stuffer_ratios.append(sr / total)

    out.append(stats_line(stuffer_ratios, "Self-reported ratio (0=verified, 1=all stuffed)"))

    # Bucket by stuffer ratio
    buckets = collections.Counter()
    for r in stuffer_ratios:
        if   r <= 0.25: buckets["0-25%  (mostly verified)"]   += 1
        elif r <= 0.50: buckets["25-50% (mixed)"]             += 1
        elif r <= 0.75: buckets["50-75% (mostly unverified)"] += 1
        else:           buckets["75-100% (keyword stuffer!)"] += 1

    out.append(f"\n  Candidate stuffer-ratio distribution:")
    max_b = max(buckets.values()) if buckets else 1
    for k in ["0-25%  (mostly verified)","25-50% (mixed)",
              "50-75% (mostly unverified)","75-100% (keyword stuffer!)"]:
        n = buckets[k]
        out.append(f"  {k:<35s} {n:>7,}  {pct(n,N):>6s}  {bar(n,max_b,20)}")

    # Top JD-relevant skills — compare evidence levels
    jd_relevant_skills = [
        sk for sk, v in vocab.items()
        if v["jd_label"] in ("JD_HARD_REQ","JD_POSITIVE")
        and v["doc_freq"] >= 2
    ]
    jd_relevant_skills.sort(key=lambda sk: -vocab[sk]["doc_freq"])

    out.append(f"\n  Evidence quality for JD-relevant skills:")
    out.append(f"  {'Skill':<35s} {'TotalCands':>10s}  {'ProdProven':>10s}  {'CtxVer':>8s}  {'SelfOnly':>8s}  StufferRate")
    out.append("  " + "-" * 95)
    for sk in jd_relevant_skills[:30]:
        ev = evidence_counts.get(sk, {})
        pp  = ev.get("PRODUCTION_PROVEN", 0)
        cv  = ev.get("CONTEXT_VERIFIED",  0)
        sr  = ev.get("SELF_REPORTED",     0)
        tot = pp + cv + sr
        stuffer_rate = f"{sr/tot*100:.0f}%" if tot > 0 else " — "
        out.append(
            f"  {sk:<35s} {tot:>10,}  {pp:>10,}  {cv:>8,}  {sr:>8,}  {stuffer_rate}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 : JD POLARITY ANALYSIS  [CHANGE 2]
# ─────────────────────────────────────────────────────────────────────────────
def section5_jd_polarity(cands, out):
    N = len(cands)
    out.append("\n" + "=" * 68)
    out.append("SECTION 5 : JD POLARITY ANALYSIS  (signed sections)")
    out.append("=" * 68)
    out.append("""
  The JD has FOUR sections with different polarities.
  Simple keyword extraction treats ALL sections as positive — this is WRONG.
  Section D (DO NOT WANT) terms should PENALISE, not reward.
""")

    section_matches, _ = build_jd_keyword_profile(cands)

    for sec_name, sec_data in JD_SECTIONS.items():
        polarity_marker = {
            "HARD_REQUIREMENT": "  [+++ MUST HAVE]",
            "IDEAL_PROFILE":    "  [++  IDEAL]",
            "NICE_TO_HAVE":     "  [+   BONUS]",
            "EXPLICIT_NEGATIVE":"  [--- PENALISE/DISQUALIFY]",
        }[sec_name]

        out.append(f"\n  {'─'*65}")
        out.append(f"  {sec_name}{polarity_marker}")
        out.append(f"  Weight={sec_data['weight']:+.2f}  Polarity={sec_data['polarity']}")
        out.append(f"  {'─'*65}")
        out.append(f"  {'Phrase':<40s} {'Profiles with match':>20s}  {'%':>7s}")
        out.append("  " + "-" * 72)

        phrase_counts = section_matches[sec_name]
        sorted_phrases = sorted(phrase_counts.items(), key=lambda x: -x[1])

        if sorted_phrases:
            for phrase, cnt in sorted_phrases[:20]:
                marker = "  ⚠ RED FLAG" if sec_data["polarity"] == "negative" else ""
                out.append(f"  {phrase:<40s} {cnt:>20,}  {pct(cnt,N):>7s}{marker}")
        else:
            out.append(f"  (no matches found — terms may need adjustment)")

    # coverage summary
    out.append(f"\n  POLARITY COVERAGE SUMMARY:")
    out.append(f"  {'Section':<25s} {'Weight':>8s}  {'Total Matches':>14s}  Notes")
    out.append("  " + "-" * 65)
    coverage_note = {
        "HARD_REQUIREMENT": "Candidates must satisfy these",
        "IDEAL_PROFILE":    "Strong positive signal",
        "NICE_TO_HAVE":     "Small bonus only",
        "EXPLICIT_NEGATIVE":"Match here REDUCES score",
    }
    for sec_name, sec_data in JD_SECTIONS.items():
        total = sum(section_matches[sec_name].values())
        out.append(
            f"  {sec_name:<25s} {sec_data['weight']:>+8.2f}  {total:>14,}  {coverage_note[sec_name]}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 : NEGATIVE PATTERN DETECTION  (JD Section D patterns)
# ─────────────────────────────────────────────────────────────────────────────
def section6_negative_patterns(cands, out):
    N = len(cands)
    out.append("\n" + "=" * 68)
    out.append("SECTION 6 : NEGATIVE PATTERN DETECTION  (JD Section D)")
    out.append("=" * 68)
    out.append("""
  Three behavioral anti-patterns explicitly called out by the JD:
    1. Title-chaser        — short tenures, escalating titles, company-hopping
    2. Framework enthusiast — wrapper tools (LangChain) without depth skills
    3. Closed-source isolation — 5yr+ career, no GitHub, no external validation
""")

    tc_flagged  = []
    fe_flagged  = []
    cs_flagged  = []

    for c in cands:
        cid   = c["candidate_id"]
        title = c["profile"].get("current_title","?")

        is_tc, tc_reason = detect_title_chaser(c)
        is_fe, fe_reason = detect_framework_enthusiast(c)
        is_cs, cs_reason = detect_closed_source_isolation(c)

        if is_tc: tc_flagged.append((cid, title, tc_reason))
        if is_fe: fe_flagged.append((cid, title, fe_reason))
        if is_cs: cs_flagged.append((cid, title, cs_reason))

    patterns = [
        ("TITLE_CHASER",          tc_flagged,  "Short tenures + escalating title levels"),
        ("FRAMEWORK_ENTHUSIAST",  fe_flagged,  "Wrapper tools only, no depth in career text"),
        ("CLOSED_SOURCE_ISOLATED",cs_flagged,  "5yr+ career, no GitHub, no open-source signals"),
    ]

    for pattern_name, flagged, description in patterns:
        out.append(f"\n  {pattern_name}  ({len(flagged):,} candidates)  —  {description}")
        out.append(f"  Prevalence: {pct(len(flagged),N)}")
        out.append(f"  {'CandidateID':<16s} {'Title':<38s} Evidence")
        out.append("  " + "-" * 85)
        for cid, title, reason in flagged[:20]:
            out.append(f"  {cid:<16s} {title:<38s} {reason[:60]}")
        if len(flagged) > 20:
            out.append(f"  ... and {len(flagged)-20} more")

    out.append(f"\n  SUMMARY — candidates with at least one negative pattern:")
    any_negative = len({
        c["candidate_id"] for c in cands
        if detect_title_chaser(c)[0]
        or detect_framework_enthusiast(c)[0]
        or detect_closed_source_isolation(c)[0]
    })
    out.append(f"  {any_negative:,}  ({pct(any_negative,N)} of corpus)")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 : CAREER PATTERNS  (unchanged logic, improved labels)
# ─────────────────────────────────────────────────────────────────────────────
def section7_career(cands, out):
    N = len(cands)
    out.append("\n" + "=" * 68)
    out.append("SECTION 7 : CAREER HISTORY & COMPANY PATTERNS")
    out.append("=" * 68)

    industries    = collections.Counter()
    company_sizes = collections.Counter()
    tenures       = []
    consulting_jobs = total_jobs = pure_consulting = 0

    for c in cands:
        hist = c.get("career_history", [])
        total_jobs += len(hist)
        c_con = sum(1 for r in hist
                    if any(cf in (r.get("company","") or "").lower()
                           for cf in CONSULTING_FIRMS_SET))
        consulting_jobs += c_con
        if len(hist) >= 2 and c_con == len(hist):
            pure_consulting += 1

        for r in hist:
            industries[r.get("industry","Unknown")]     += 1
            company_sizes[r.get("company_size","Unknown")] += 1
            dur = r.get("duration_months", 0) or 0
            if dur > 0:
                tenures.append(dur)

    out.append(f"\n  Total job records              : {total_jobs:,}")
    out.append(f"  Consulting-firm roles          : {consulting_jobs:,}  ({pct(consulting_jobs,total_jobs)})")
    out.append(f"  Pure-consulting careers        : {pure_consulting:,}  ({pct(pure_consulting,N)})  ← JD disqualifier")
    out.append(stats_line(tenures, "Tenure per role (months)"))

    out.append(f"\n  Top 20 industries:")
    max_i = industries.most_common(1)[0][1]
    for ind, n in industries.most_common(20):
        out.append(f"  {ind:<42s} {n:>7,}  {pct(n,total_jobs):>6s}  {bar(n,max_i,20)}")

    out.append(f"\n  Company size distribution:")
    size_order = ["1-10","11-50","51-200","201-500","501-1000",
                  "1001-5000","5001-10000","10001+","Unknown"]
    max_s = max(company_sizes.values()) if company_sizes else 1
    for s in size_order:
        n = company_sizes.get(s, 0)
        if n:
            out.append(f"  {s:<16s} {n:>7,}  {pct(n,total_jobs):>6s}  {bar(n,max_s,20)}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8 : LOCATION & AVAILABILITY  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────
TARGET_LOCS = {
    "noida","pune","delhi","hyderabad","bangalore","bengaluru",
    "mumbai","gurgaon","gurugram","ncr","india",
}

def section8_location(cands, out):
    N = len(cands)
    out.append("\n" + "=" * 68)
    out.append("SECTION 8 : LOCATION & AVAILABILITY")
    out.append("=" * 68)

    countries  = collections.Counter()
    work_modes = collections.Counter()
    notices    = []
    inactive   = collections.Counter()
    in_target  = 0

    for c in cands:
        p   = c["profile"]
        sig = c["redrob_signals"]

        countries[p.get("country","Unknown")] += 1
        work_modes[sig.get("preferred_work_mode","Unknown")] += 1
        notices.append(sig.get("notice_period_days", 90))

        loc_str = (p.get("location","") + " " + p.get("country","")).lower()
        if any(t in loc_str for t in TARGET_LOCS):
            in_target += 1

        d = days_inactive(sig.get("last_active_date","2020-01-01"))
        if   d <= 30:  inactive["Active  (≤30d)"]    += 1
        elif d <= 90:  inactive["Recent  (31-90d)"]  += 1
        elif d <= 180: inactive["Cold    (91-180d)"] += 1
        else:          inactive["Inactive(180d+)"]   += 1

    relocate = sum(1 for c in cands if c["redrob_signals"].get("willing_to_relocate"))
    otw      = sum(1 for c in cands if c["redrob_signals"].get("open_to_work_flag"))

    out.append(f"\n  Country distribution  (top 10):")
    max_c = countries.most_common(1)[0][1]
    for co, n in countries.most_common(10):
        out.append(f"  {co:<28s} {n:>7,}  {pct(n,N):>6s}  {bar(n,max_c,20)}")

    out.append(f"\n  In JD target region (India metros)  : {in_target:,}  ({pct(in_target,N)})")
    out.append(f"  Willing to relocate                 : {relocate:,}  ({pct(relocate,N)})")
    out.append(f"  Open to work flag = True            : {otw:,}  ({pct(otw,N)})")

    out.append(f"\n  Preferred work mode:")
    max_w = max(work_modes.values())
    for m, n in work_modes.most_common():
        out.append(f"  {m:<18s} {n:>7,}  {pct(n,N):>6s}  {bar(n,max_w,20)}")

    out.append(f"\n  Activity buckets:")
    max_a = max(inactive.values())
    for k in ["Active  (≤30d)","Recent  (31-90d)","Cold    (91-180d)","Inactive(180d+)"]:
        n = inactive[k]
        out.append(f"  {k:<24s} {n:>7,}  {pct(n,N):>6s}  {bar(n,max_a,20)}")
    out.append(stats_line(notices, "Notice period (days)"))

    # Notice period bucketed
    np_buckets = collections.Counter()
    for n in notices:
        if   n <= 30:  np_buckets["≤30d  (JD preferred)"]   += 1
        elif n <= 60:  np_buckets["31-60d"]                  += 1
        elif n <= 90:  np_buckets["61-90d"]                  += 1
        else:          np_buckets["90+d  (JD penalised)"]    += 1

    out.append(f"\n  Notice period buckets:")
    max_np = max(np_buckets.values())
    for k in ["≤30d  (JD preferred)","31-60d","61-90d","90+d  (JD penalised)"]:
        n = np_buckets[k]
        out.append(f"  {k:<28s} {n:>7,}  {pct(n,N):>6s}  {bar(n,max_np,20)}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9 : BEHAVIORAL SIGNALS  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────
def section9_behavioral(cands, out):
    N = len(cands)
    out.append("\n" + "=" * 68)
    out.append("SECTION 9 : BEHAVIORAL SIGNALS (Redrob Platform)")
    out.append("=" * 68)

    fields = [
        ("profile_completeness_score",  "Profile completeness (0-100)"),
        ("profile_views_received_30d",  "Profile views  (30d)"),
        ("applications_submitted_30d",  "Applications submitted (30d)"),
        ("recruiter_response_rate",     "Recruiter response rate (0-1)"),
        ("avg_response_time_hours",     "Avg response time (hours)"),
        ("connection_count",            "Connection count"),
        ("endorsements_received",       "Endorsements received"),
        ("github_activity_score",       "GitHub score  (-1=no account)"),
        ("search_appearance_30d",       "Search appearances (30d)"),
        ("saved_by_recruiters_30d",     "Saved by recruiters (30d)"),
        ("interview_completion_rate",   "Interview completion rate"),
        ("offer_acceptance_rate",       "Offer acceptance rate (-1=none)"),
        ("notice_period_days",          "Notice period (days)"),
    ]

    out.append(f"\n  {'Signal':<48s} {'Mean':>8s}  {'Median':>8s}  {'Min':>8s}  {'Max':>8s}")
    out.append("  " + "-" * 88)
    for field, label in fields:
        vals = [c["redrob_signals"].get(field) for c in cands]
        vals = [v for v in vals if v is not None and v != -1]
        if not vals:
            continue
        s    = sorted(vals)
        mean = sum(vals) / len(vals)
        med  = s[len(s) // 2]
        out.append(
            f"  {label:<48s} {mean:>8.2f}  {med:>8.2f}  "
            f"{min(vals):>8.2f}  {max(vals):>8.2f}"
        )

    gh_none = sum(1 for c in cands
                  if c["redrob_signals"].get("github_activity_score") == -1)
    gh_has  = N - gh_none
    ver_em  = sum(1 for c in cands if c["redrob_signals"].get("verified_email"))
    ver_ph  = sum(1 for c in cands if c["redrob_signals"].get("verified_phone"))
    ver_li  = sum(1 for c in cands if c["redrob_signals"].get("linkedin_connected"))

    out.append(f"\n  GitHub linked       : {gh_has:,}  ({pct(gh_has,N)})  |  No GitHub: {gh_none:,}  ({pct(gh_none,N)})")
    out.append(f"  Verified email      : {ver_em:,}  ({pct(ver_em,N)})")
    out.append(f"  Verified phone      : {ver_ph:,}  ({pct(ver_ph,N)})")
    out.append(f"  LinkedIn connected  : {ver_li:,}  ({pct(ver_li,N)})")

    sal_min = [c["redrob_signals"]["expected_salary_range_inr_lpa"]["min"]
               for c in cands
               if "expected_salary_range_inr_lpa" in c["redrob_signals"]]
    sal_max = [c["redrob_signals"]["expected_salary_range_inr_lpa"]["max"]
               for c in cands
               if "expected_salary_range_inr_lpa" in c["redrob_signals"]]
    if sal_min:
        out.append(f"\n  Salary min INR LPA  : mean={sum(sal_min)/len(sal_min):.1f}  "
                   f"range=[{min(sal_min):.0f}, {max(sal_min):.0f}]")
    if sal_max:
        out.append(f"  Salary max INR LPA  : mean={sum(sal_max)/len(sal_max):.1f}  "
                   f"range=[{min(sal_max):.0f}, {max(sal_max):.0f}]")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 10 : EDUCATION  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────
def section10_education(cands, out):
    N = len(cands)
    out.append("\n" + "=" * 68)
    out.append("SECTION 10 : EDUCATION ANALYSIS")
    out.append("=" * 68)

    tiers   = collections.Counter()
    degrees = collections.Counter()
    fields  = collections.Counter()

    for c in cands:
        for e in c.get("education", []):
            tiers[e.get("tier","unknown")]           += 1
            degrees[e.get("degree","unknown")]        += 1
            fields[e.get("field_of_study","unknown")] += 1

    out.append(f"\n  Education tier  (tier_1 = IIT/IISc/IIM level):")
    max_t = max(tiers.values()) if tiers else 1
    for t in ["tier_1","tier_2","tier_3","tier_4","tier_5","unknown"]:
        n = tiers.get(t, 0)
        out.append(f"  {t:<12s} {n:>7,}  {pct(n,N):>6s}  {bar(n,max_t,20)}")

    out.append(f"\n  Top 15 fields of study:")
    max_f = fields.most_common(1)[0][1] if fields else 1
    for fd, n in fields.most_common(15):
        out.append(f"  {fd:<42s} {n:>7,}  {pct(n,N):>6s}  {bar(n,max_f,20)}")

    out.append(f"\n  Degree types:")
    for d, n in degrees.most_common(15):
        out.append(f"  {d:<22s} {n:>7,}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("\n" + "=" * 68)
    print("  Redrob Hackathon — EDA Script 1 v2 : Corpus Analysis")
    print("  Changes: data-driven vocab | JD polarity | context verification")
    print("=" * 68 + "\n")

    cands = load_candidates()   # use max_n=5000 for a quick test run
    N     = len(cands)

    # ── Build data-driven skill vocabulary FIRST (no hardcoding) ──────────────
    print("  Building data-driven skill vocabulary ...")
    vocab, _ = build_skill_vocabulary(cands)
    print(f"  Vocabulary built: {len(vocab):,} unique skills found in corpus")
    print(f"  Tier breakdown: "
          f"GENERIC={sum(1 for v in vocab.values() if v['tier']=='TIER_1_GENERIC')}, "
          f"DOMAIN={sum(1 for v in vocab.values() if v['tier']=='TIER_2_DOMAIN')}, "
          f"SPECIALIST={sum(1 for v in vocab.values() if v['tier']=='TIER_3_SPECIALIST')}")

    output_lines = [
        "=" * 68,
        "REDROB HACKATHON  —  EDA REPORT  (v2: data-driven + JD-aware)",
        f"Generated : {TODAY.strftime('%Y-%m-%d')}",
        f"Dataset   : {N:,} candidates",
        f"Vocab size: {len(vocab):,} unique skills (corpus-derived, no hardcoding)",
        "=" * 68,
    ]

    steps = [
        ("Section 1:  Corpus overview",           lambda: section1_corpus_overview(cands, output_lines)),
        ("Section 2:  Titles & experience",        lambda: section2_titles_experience(cands, output_lines)),
        ("Section 3:  Data-driven skill vocab",    lambda: section3_skills_data_driven(cands, vocab, output_lines)),
        ("Section 4:  Skill-in-context evidence",  lambda: section4_skill_evidence(cands, vocab, output_lines)),
        ("Section 5:  JD polarity analysis",       lambda: section5_jd_polarity(cands, output_lines)),
        ("Section 6:  Negative pattern detection", lambda: section6_negative_patterns(cands, output_lines)),
        ("Section 7:  Career patterns",            lambda: section7_career(cands, output_lines)),
        ("Section 8:  Location & availability",    lambda: section8_location(cands, output_lines)),
        ("Section 9:  Behavioral signals",         lambda: section9_behavioral(cands, output_lines)),
        ("Section 10: Education",                  lambda: section10_education(cands, output_lines)),
    ]

    for label, fn in steps:
        print(f"  Running {label} ...")
        fn()

    # write report
    report_path = OUT_DIR / "corpus_stats.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines))

    # print to console
    print("\n")
    for line in output_lines:
        print(line)

    print(f"\n\n  ✓  Report saved → {report_path}")
    print(f"  Vocab saved separately for use by eda_02 and ranker ...")

    # also save vocabulary as JSON for downstream use
    vocab_path = OUT_DIR / "skill_vocabulary.json"
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, indent=2)
    print(f"  ✓  Skill vocab  saved → {vocab_path}")


if __name__ == "__main__":
    main()
