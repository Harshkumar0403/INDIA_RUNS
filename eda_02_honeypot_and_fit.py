"""
eda_02_honeypot_and_fit.py  (v2 — data-driven + JD-aware)
===========================================================
Redrob Hackathon — Honeypot Detection, Disqualifier Analysis,
                   JD Fit Simulation & Scoring Calibration

KEY CHANGES FROM v1:
  1. NO hardcoded skill sets (AI_CORE_SKILLS, CV_SPEECH_SKILLS,
     RETRIEVAL_SKILLS, wrapper_only_skills, depth_skills).
     All replaced by loading eda_outputs/skill_vocabulary.json
     built data-driven by eda_01. Skills classified by JD label:
       JD_HARD_REQ  required (retrieval, embeddings, vector DBs)
       JD_NEGATIVE  red-flag (LangChain-only, CV, Speech)
       JD_POSITIVE  nice-to-have
       NEUTRAL      no signal either way

  2. NO hardcoded DISQUALIFIED_TITLES / RELEVANT_TITLES.
     Title relevance scored via corpus-derived avg JD skill
     coverage per title. Titles with <15% coverage are
     non-relevant. Scores are continuous, not binned.

  3. NO hardcoded product_industries list.
     Product-company detection uses corpus-derived industry
     JD skill coverage (same logic as titles).

  4. IDF-weighted skill scoring.
     score = sum(idf[sk] for sk in candidate_skills
                 if vocab[sk]['jd_label'] in HARD_REQ|POSITIVE)
     normalised to [0,1] at 95th-percentile.

  5. Negative skill penalty multiplier.
     JD_NEGATIVE skills reduce final score multiplicatively.

  DOCUMENTED EXCEPTION: consulting firm name-patterns remain
  hardcoded — company names cannot be derived from skill vocab.

Run from: INDIA_RUNS/
Usage:    python eda_02_honeypot_and_fit.py
Outputs:  eda_outputs/{honeypot_flags,fit_simulation,disqualifier_stats}.txt
"""

import json
import re
import math
import collections
from datetime import datetime
from pathlib import Path

# ── paths ─────────────────────────────────────────────────────────────────────
ASSETS          = Path("INDIA_RUNS_Assets")
CANDIDATES_FILE = ASSETS / "candidates.jsonl"
SAMPLE_FILE     = ASSETS / "sample_candidates.json"
VOCAB_FILE      = Path("eda_outputs") / "skill_vocabulary.json"
OUT_DIR         = Path("eda_outputs")
OUT_DIR.mkdir(exist_ok=True)

TODAY = datetime(2026, 6, 2)

def days_inactive(date_str):
    try:
        return (TODAY - datetime.strptime(date_str, "%Y-%m-%d")).days
    except Exception:
        return 9999

def pct(n, total):
    return f"{n/total*100:.1f}%" if total else "0%"

def bar(n, max_n, width=20):
    filled = round((n / max_n) * width) if max_n else 0
    return "█" * filled + "░" * (width - filled)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 0 — LOAD SKILL VOCABULARY
# ─────────────────────────────────────────────────────────────────────────────
def load_vocab():
    if not VOCAB_FILE.exists():
        print(f"  WARNING: {VOCAB_FILE} not found — run eda_01 first.")
        return {}, set(), set(), set(), {}
    with open(VOCAB_FILE, encoding="utf-8") as f:
        vocab = json.load(f)
    jd_hard_req = {sk for sk, v in vocab.items() if v["jd_label"] == "JD_HARD_REQ"}
    jd_positive = {sk for sk, v in vocab.items() if v["jd_label"] == "JD_POSITIVE"}
    jd_negative = {sk for sk, v in vocab.items() if v["jd_label"] == "JD_NEGATIVE"}
    idf_lookup  = {sk: v["idf"] for sk, v in vocab.items()}
    print(f"  Vocab loaded: {len(vocab)} skills | "
          f"hard_req={len(jd_hard_req)} | "
          f"positive={len(jd_positive)} | "
          f"negative={len(jd_negative)}")
    return vocab, jd_hard_req, jd_positive, jd_negative, idf_lookup


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — CORPUS-DERIVED TITLE INTELLIGENCE
# ─────────────────────────────────────────────────────────────────────────────
def build_title_intelligence(cands, jd_hard_req, jd_positive):
    """
    Per title: compute average number of JD_HARD_REQ+JD_POSITIVE skills
    across all candidates with that title.
    Normalise to [0,1]. Titles below 15% of max = non-relevant.
    """
    title_jd_counts = collections.defaultdict(list)
    title_frequency = collections.Counter()

    for c in cands:
        title       = c["profile"].get("current_title", "Unknown")
        skill_names = {s["name"] for s in c.get("skills", [])}
        jd_match    = len(skill_names & (jd_hard_req | jd_positive))
        title_frequency[title]         += 1
        title_jd_counts[title].append(jd_match)

    title_avg_jd = {
        t: sum(v) / len(v) for t, v in title_jd_counts.items()
    }
    max_avg = max(title_avg_jd.values()) if title_avg_jd else 1
    title_relevance_score = {
        t: round(avg / max_avg, 4) for t, avg in title_avg_jd.items()
    }

    NON_RELEVANT_THRESHOLD = 0.15
    non_relevant_titles = {
        t for t, s in title_relevance_score.items()
        if s < NON_RELEVANT_THRESHOLD
    }
    top_relevant = sorted(
        [(t, s) for t, s in title_relevance_score.items()
         if s >= NON_RELEVANT_THRESHOLD],
        key=lambda x: -x[1]
    )
    return title_relevance_score, title_frequency, non_relevant_titles, top_relevant


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — CORPUS-DERIVED INDUSTRY INTELLIGENCE
# ─────────────────────────────────────────────────────────────────────────────
def build_industry_intelligence(cands, jd_hard_req, jd_positive):
    """
    Per industry: compute average JD skill coverage across all
    candidates who held roles in that industry.
    product_like >= 0.40 | services_like < 0.15
    """
    ind_jd_counts = collections.defaultdict(list)
    for c in cands:
        skill_names = {s["name"] for s in c.get("skills", [])}
        jd_match    = len(skill_names & (jd_hard_req | jd_positive))
        for r in c.get("career_history", []):
            ind = r.get("industry", "Unknown") or "Unknown"
            ind_jd_counts[ind].append(jd_match)

    ind_avg    = {i: sum(v)/len(v) for i, v in ind_jd_counts.items()}
    max_avg    = max(ind_avg.values()) if ind_avg else 1
    ind_score  = {i: round(v/max_avg, 4) for i, v in ind_avg.items()}

    product_like  = {i for i, s in ind_score.items() if s >= 0.40}
    services_like = {i for i, s in ind_score.items() if s <  0.15}
    return ind_score, product_like, services_like


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — IDF NORMALISATION CONSTANT
# ─────────────────────────────────────────────────────────────────────────────
def compute_idf_normalisation(cands, idf_lookup, jd_hard_req, jd_positive):
    sums = []
    for c in cands:
        skill_names = {s["name"] for s in c.get("skills", [])}
        total = sum(
            idf_lookup.get(sk, 0)
            for sk in skill_names
            if sk in (jd_hard_req | jd_positive)
        )
        sums.append(total)
    if not sums:
        return 1.0
    sums.sort()
    p95 = sums[int(len(sums) * 0.95)]
    return max(p95, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# CONSULTING FIRM PATTERNS — documented exception to no-hardcoding rule
# Company names cannot be derived from skill vocabulary.
# ─────────────────────────────────────────────────────────────────────────────
CONSULTING_NAME_PATTERNS = {
    "tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini",
    "hcl", "tech mahindra", "mphasis", "hexaware", "ltimindtree",
    "mindtree", "cts", "deloitte", "ibm consulting", "pwc", "kpmg",
    "tata consultancy", "ntt data", "dxc technology", "unisys",
}

def is_consulting_company(company_name):
    low = (company_name or "").lower()
    return any(pat in low for pat in CONSULTING_NAME_PATTERNS)

# Semantic production cues from JD text — not skill names
PRODUCTION_CUES = {
    "production", "deployed", "deployment", "serving", "inference",
    "latency", "throughput", "scale", "real users", "real-time",
    "realtime", "a/b", "ab test", "experiment", "benchmark",
    "billion", "million", "queries", "requests", "monitoring",
    "drift", "regression", "pipeline", "online serving",
}

SENIORITY_LADDER = [
    "intern", "junior", "associate", "mid", "senior",
    "staff", "principal", "director", "vp", "head", "chief",
]

TARGET_LOCS = {
    "noida", "pune", "delhi", "hyderabad", "bangalore",
    "bengaluru", "mumbai", "gurgaon", "gurugram", "ncr",
}


# ─────────────────────────────────────────────────────────────────────────────
# HONEYPOT DETECTION — pure arithmetic, no skill taxonomy
# ─────────────────────────────────────────────────────────────────────────────
def detect_honeypots(c):
    flags  = []
    p      = c.get("profile", {})
    hist   = c.get("career_history", [])
    skills = c.get("skills", [])
    sig    = c.get("redrob_signals", {})
    yoe    = p.get("years_of_experience", 0) or 0

    # FLAG 1: skill duration > career length
    for s in skills:
        dur = s.get("duration_months", 0) or 0
        if dur > (yoe * 12) + 6:
            flags.append((
                "SKILL_DURATION_IMPOSSIBLE",
                f"'{s.get('name','?')}' claimed {dur}mo "
                f"but career is only {yoe*12:.0f}mo"
            ))

    # FLAG 2: role tenure longer than time since start_date
    for r in hist:
        dur = r.get("duration_months", 0) or 0
        try:
            start      = datetime.strptime(r.get("start_date",""), "%Y-%m-%d")
            years_back = (TODAY - start).days / 365
            if dur / 12 > years_back + 0.5:
                flags.append((
                    "TENURE_LONGER_THAN_POSSIBLE",
                    f"'{r.get('company','?')}': {dur}mo tenure "
                    f"but only {years_back:.1f}yr since start"
                ))
        except Exception:
            pass

    # FLAG 3: ≥4 'advanced' skills with 0 endorsements
    adv_zero = [s["name"] for s in skills
                if s.get("proficiency") == "advanced"
                and (s.get("endorsements") or 0) == 0]
    if len(adv_zero) >= 4:
        flags.append((
            "INFLATED_PROFICIENCY",
            f"{len(adv_zero)} 'advanced' + 0 endorsements: {adv_zero[:5]}"
        ))

    # FLAG 4: all behavioral signals simultaneously at maximum
    rr  = sig.get("recruiter_response_rate", 0) or 0
    icr = sig.get("interview_completion_rate", 0) or 0
    oar = sig.get("offer_acceptance_rate", 0) or 0
    pc  = sig.get("profile_completeness_score", 0) or 0
    if rr >= 0.99 and icr >= 0.99 and oar >= 0.99 and pc >= 99:
        flags.append((
            "ALL_SIGNALS_PERFECT",
            f"rr={rr:.2f} icr={icr:.2f} oar={oar:.2f} completeness={pc:.0f}"
        ))

    # FLAG 5: claimed YoE inconsistent with career history
    if hist:
        earliest = None
        for r in hist:
            try:
                sd = datetime.strptime(r.get("start_date",""), "%Y-%m-%d")
                if earliest is None or sd < earliest:
                    earliest = sd
            except Exception:
                pass
        if earliest:
            implied_yoe = (TODAY - earliest).days / 365
            if abs(implied_yoe - yoe) > 3.0:
                flags.append((
                    "YOE_CAREER_MISMATCH",
                    f"profile={yoe:.1f}yr but career history implies {implied_yoe:.1f}yr"
                ))

    # FLAG 6: all assessment scores are round multiples of 5
    assessments = sig.get("skill_assessment_scores") or {}
    if len(assessments) >= 3:
        rounded = sum(
            1 for v in assessments.values()
            if isinstance(v, (int, float)) and v % 5 == 0
        )
        if rounded == len(assessments):
            flags.append((
                "ROUND_ASSESSMENT_SCORES",
                f"all {len(assessments)} scores are exact multiples of 5"
            ))

    # FLAG 7: suspiciously perfect career continuity
    if len(hist) >= 4:
        sorted_hist = []
        for r in hist:
            try:
                sd = datetime.strptime(r.get("start_date",""), "%Y-%m-%d")
                sorted_hist.append((sd, r.get("duration_months", 0) or 0))
            except Exception:
                pass
        sorted_hist.sort()
        zero_gaps = 0
        for i in range(len(sorted_hist) - 1):
            start_i, dur_i = sorted_hist[i]
            start_next     = sorted_hist[i + 1][0]
            gap_months = (
                (start_next.year - start_i.year) * 12
                + (start_next.month - start_i.month)
                - dur_i
            )
            if abs(gap_months) <= 1:
                zero_gaps += 1
        if zero_gaps >= len(sorted_hist) - 1 and len(sorted_hist) >= 4:
            flags.append((
                "PERFECTLY_CONTIGUOUS_CAREER",
                f"all {len(sorted_hist)} roles have 0-1mo gap — fabricated dates?"
            ))

    return flags


# ─────────────────────────────────────────────────────────────────────────────
# DISQUALIFIER DETECTION — data-driven (vocab-derived sets)
# ─────────────────────────────────────────────────────────────────────────────
def detect_disqualifiers(c, non_relevant_titles,
                         jd_negative, jd_hard_req, jd_positive):
    reasons     = []
    p           = c.get("profile", {})
    hist        = c.get("career_history", [])
    skills      = c.get("skills", [])
    title       = p.get("current_title", "")
    skill_names = {s["name"] for s in skills}
    yoe         = p.get("years_of_experience", 0) or 0

    # DQ1: corpus-derived non-relevant title
    if title in non_relevant_titles:
        reasons.append(
            f"DQ_NON_RELEVANT_TITLE: '{title}' "
            f"(corpus-derived: <15% JD skill coverage)"
        )

    # DQ2: entire career at consulting firms (name-pattern — documented exception)
    if hist:
        c_con = sum(
            1 for r in hist
            if is_consulting_company(r.get("company", ""))
        )
        if c_con == len(hist) and len(hist) >= 2:
            firms = [r.get("company","?") for r in hist]
            reasons.append(
                f"DQ_PURE_CONSULTING: all {len(hist)} roles at "
                f"services firms: {firms}"
            )

    # DQ3: vocab-derived JD_NEGATIVE skills dominate JD_POSITIVE
    neg_count      = len(skill_names & jd_negative)
    hard_count     = len(skill_names & jd_hard_req)
    pos_count      = len(skill_names & jd_positive)
    total_positive = hard_count + pos_count
    if neg_count >= 3 and neg_count > total_positive:
        reasons.append(
            f"DQ_NEGATIVE_SKILLS_DOMINANT: {neg_count} JD-negative "
            f"({skill_names & jd_negative}) > {total_positive} JD-positive"
        )

    # DQ4: wrapper-only + no hard-req skills + low experience
    has_wrapper = bool(skill_names & jd_negative)
    has_depth   = bool(skill_names & jd_hard_req)
    if has_wrapper and not has_depth and yoe < 3:
        reasons.append(
            f"DQ_WRAPPER_ONLY_LOW_EXP: JD-negative tools "
            f"{skill_names & jd_negative} with no hard-req skills, "
            f"only {yoe:.1f}yr exp"
        )

    return bool(reasons), reasons


# ─────────────────────────────────────────────────────────────────────────────
# JD FIT SCORER
# ─────────────────────────────────────────────────────────────────────────────
def score_candidate(c, title_relevance_score, non_relevant_titles,
                    jd_hard_req, jd_positive, jd_negative,
                    idf_lookup, idf_norm_constant, product_like_industries):

    p           = c.get("profile", {})
    hist        = c.get("career_history", [])
    skills      = c.get("skills", [])
    sig         = c.get("redrob_signals", {})
    edu         = c.get("education", [])
    title       = p.get("current_title", "")
    yoe         = p.get("years_of_experience", 0) or 0
    skill_names = {s["name"] for s in skills}

    # Gate: honeypot / disqualifier
    hp_flags          = detect_honeypots(c)
    is_dq, dq_reasons = detect_disqualifiers(
        c, non_relevant_titles, jd_negative, jd_hard_req, jd_positive
    )
    if hp_flags or is_dq:
        return {
            "score": 0.0, "disqualified": is_dq,
            "honeypot": bool(hp_flags),
            "reasons": dq_reasons + [f[0]+": "+f[1] for f in hp_flags],
            "details": {},
        }

    score   = 0.0
    reasons = []
    details = {}

    # A. Title relevance  (0–0.20)  corpus-derived
    t_score     = title_relevance_score.get(title, 0.0)
    title_score = t_score * 0.20
    score      += title_score
    details["title_score"] = round(title_score, 4)
    if title_score > 0.02:
        reasons.append(f"+title('{title}' rel={t_score:.3f})")

    # B. IDF-weighted JD skill score  (0–0.30)
    jd_pos_idf_sum = sum(
        idf_lookup.get(sk, 0)
        for sk in skill_names
        if sk in (jd_hard_req | jd_positive)
    )
    idf_score = min(jd_pos_idf_sum / idf_norm_constant, 1.0) * 0.30
    score    += idf_score
    details["idf_skill_score"]  = round(idf_score, 4)
    details["jd_pos_idf_sum"]   = round(jd_pos_idf_sum, 3)
    matched_jd = skill_names & (jd_hard_req | jd_positive)
    if matched_jd:
        reasons.append(
            f"+idf_skills({len(matched_jd)}, sum={jd_pos_idf_sum:.2f})"
        )

    # C. Hard-requirement depth  (0–0.15)
    hard_matches = skill_names & jd_hard_req
    hard_score   = min(len(hard_matches) / 3, 1.0) * 0.15
    score       += hard_score
    details["hard_req_score"] = round(hard_score, 4)
    if hard_matches:
        reasons.append(f"+hard_req({len(hard_matches)}: {hard_matches})")

    # D. Production cues in career text  (0–0.10)
    career_text = " ".join(
        (r.get("description","") or "").lower() for r in hist
    )
    prod_hits   = sum(1 for cue in PRODUCTION_CUES if cue in career_text)
    prod_score  = min(prod_hits / 4, 1.0) * 0.10
    score      += prod_score
    details["prod_evidence_score"] = round(prod_score, 4)
    if prod_hits:
        reasons.append(f"+prod_evidence({prod_hits} cues)")

    # E. Context-verified skills bonus  (0–0.08)
    ctx_verified = sum(1 for sk in matched_jd if sk.lower() in career_text)
    ctx_score    = min(ctx_verified / 3, 1.0) * 0.08
    score       += ctx_score
    details["context_verified_score"] = round(ctx_score, 4)
    if ctx_verified:
        reasons.append(f"+ctx_verified({ctx_verified} JD skills in text)")

    # F. Experience in JD range  (0–0.07)
    if   6 <= yoe <= 8:  exp_score = 0.07
    elif 5 <= yoe < 6:   exp_score = 0.055
    elif 8 < yoe <= 10:  exp_score = 0.055
    elif 4 <= yoe < 5:   exp_score = 0.035
    elif 10 < yoe <= 12: exp_score = 0.035
    else:                exp_score = 0.0
    score += exp_score
    details["exp_score"] = round(exp_score, 4)
    if exp_score > 0:
        reasons.append(f"+yoe({yoe:.1f}yr)")

    # G. Product-company industry  (0–0.05)  corpus-derived
    product_roles = sum(
        1 for r in hist
        if r.get("industry","") in product_like_industries
    )
    prod_co_score = min(product_roles / 2, 1.0) * 0.05
    score        += prod_co_score
    details["product_co_score"] = round(prod_co_score, 4)
    if product_roles:
        reasons.append(f"+product_co({product_roles})")

    # H. Location  (0–0.03)
    loc_str   = (p.get("location","") + " " + p.get("country","")).lower()
    if any(t in loc_str for t in TARGET_LOCS):
        loc_score = 0.03
        reasons.append("+location_metro")
    elif sig.get("willing_to_relocate"):
        loc_score = 0.015
        reasons.append("+willing_relocate")
    else:
        loc_score = 0.0
    score += loc_score
    details["location_score"] = round(loc_score, 4)

    # I. Education tier  (0–0.02)
    edu_bonus = 0.0
    if edu:
        try:
            best_tier = min(
                int(e.get("tier","tier_5").split("_")[-1])
                for e in edu if "tier" in e.get("tier","")
            )
            edu_bonus = {1:0.02, 2:0.015, 3:0.01, 4:0.005}.get(best_tier, 0.0)
        except Exception:
            pass
    score += edu_bonus
    details["edu_bonus"] = round(edu_bonus, 4)

    # J. GitHub  (0–0.02)
    gh       = sig.get("github_activity_score", -1)
    gh_score = (gh / 100 * 0.02) if gh and gh > 0 else 0.0
    score   += gh_score
    details["github_score"] = round(gh_score, 4)
    if gh and gh > 0:
        reasons.append(f"+github({gh:.0f})")

    # K. Platform assessments  (0–0.02)  vocab-filtered
    assessments = sig.get("skill_assessment_scores") or {}
    if assessments:
        rel_assess = {
            k: v for k, v in assessments.items()
            if k in (jd_hard_req | jd_positive)
        }
        if rel_assess:
            avg_a        = sum(rel_assess.values()) / len(rel_assess)
            assess_score = (avg_a / 100) * 0.02
            score       += assess_score
            details["assessment_score"] = round(assess_score, 4)
            reasons.append(f"+assessments({len(rel_assess)}, avg={avg_a:.1f})")

    # Negative skill penalty  (vocab-derived JD_NEGATIVE set)
    neg_count = len(skill_names & jd_negative)
    neg_penalty = (
        0.50 if neg_count >= 3 else
        0.70 if neg_count == 2 else
        0.85 if neg_count == 1 else
        1.00
    )
    if neg_count > 0:
        reasons.append(
            f"-neg_skills({neg_count}: {skill_names & jd_negative}, "
            f"mult={neg_penalty:.2f})"
        )
    score = score * neg_penalty
    details["neg_skill_penalty"] = round(neg_penalty, 3)

    # Availability multiplier
    d_inactive = days_inactive(sig.get("last_active_date","2020-01-01"))
    if   d_inactive <= 14:  avail = 1.00
    elif d_inactive <= 30:  avail = 0.95
    elif d_inactive <= 60:  avail = 0.88
    elif d_inactive <= 90:  avail = 0.80
    elif d_inactive <= 120: avail = 0.70
    elif d_inactive <= 180: avail = 0.58
    else:                   avail = 0.40

    if not sig.get("open_to_work_flag"):
        avail *= 0.88
    notice = sig.get("notice_period_days", 60) or 60
    if notice > 90:
        avail *= 0.88
    rr = sig.get("recruiter_response_rate", 0.5) or 0
    if rr < 0.20:
        avail *= 0.85

    details["raw_score"]        = round(score, 4)
    details["avail_multiplier"] = round(avail, 3)
    details["neg_penalty"]      = round(neg_penalty, 3)

    final_score = round(score * avail, 6)
    details["final_score"] = final_score

    return {
        "score": final_score, "disqualified": False, "honeypot": False,
        "reasons": reasons, "details": details,
    }


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────
def load_candidates(max_n=None):
    candidates = []
    if CANDIDATES_FILE.exists():
        print(f"  Loading {CANDIDATES_FILE} ...")
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
        print(f"  Loaded {len(candidates):,} candidates")
    elif SAMPLE_FILE.exists():
        print(f"  Falling back to {SAMPLE_FILE}")
        with open(SAMPLE_FILE) as f:
            candidates = json.load(f)
        print(f"  Loaded {len(candidates)} candidates (sample)")
    return candidates


# ─────────────────────────────────────────────────────────────────────────────
# ANALYSIS RUNNER
# ─────────────────────────────────────────────────────────────────────────────
def run_analysis(cands, title_relevance_score, non_relevant_titles,
                 jd_hard_req, jd_positive, jd_negative,
                 idf_lookup, idf_norm_constant, product_like_industries):
    N = len(cands)
    print(f"\n  Scoring {N:,} candidates ...")

    results = []
    hp_flagged = []
    dq_flagged = []
    hp_type_counts = collections.Counter()
    dq_type_counts = collections.Counter()

    for c in cands:
        r     = score_candidate(
            c, title_relevance_score, non_relevant_titles,
            jd_hard_req, jd_positive, jd_negative,
            idf_lookup, idf_norm_constant, product_like_industries
        )
        cid   = c["candidate_id"]
        p     = c["profile"]
        sig   = c["redrob_signals"]

        entry = {
            "id":            cid,
            "title":         p.get("current_title","?"),
            "yoe":           p.get("years_of_experience",0) or 0,
            "location":      p.get("location","?"),
            "score":         r["score"],
            "disqualified":  r["disqualified"],
            "honeypot":      r["honeypot"],
            "reasons":       r["reasons"],
            "details":       r["details"],
            "days_inactive": days_inactive(sig.get("last_active_date","")),
            "notice":        sig.get("notice_period_days",0) or 0,
            "open_to_work":  sig.get("open_to_work_flag", False),
            "response_rate": sig.get("recruiter_response_rate",0) or 0,
        }
        results.append(entry)

        hp_flags  = detect_honeypots(c)
        is_dq, dr = detect_disqualifiers(
            c, non_relevant_titles, jd_negative, jd_hard_req, jd_positive
        )
        if hp_flags:
            hp_flagged.append({"id":cid,"title":entry["title"],"flags":hp_flags})
            for f_type,_ in hp_flags:
                hp_type_counts[f_type] += 1
        if is_dq:
            dq_flagged.append({"id":cid,"title":entry["title"],"reasons":dr})
            for r_str in dr:
                dq_type_counts[r_str.split(":")[0]] += 1

    results.sort(key=lambda x: -x["score"])
    return results, hp_flagged, dq_flagged, hp_type_counts, dq_type_counts


# ─────────────────────────────────────────────────────────────────────────────
# REPORT WRITERS
# ─────────────────────────────────────────────────────────────────────────────
def write_honeypot_report(hp_flagged, hp_type_counts,
                          dq_flagged, dq_type_counts, N):
    lines = [
        "="*68,
        "HONEYPOT & DISQUALIFIER DETECTION  (v2: data-driven)",
        f"Dataset: {N:,}  |  {TODAY.date()}",
        "="*68,"","HONEYPOT FLAG TYPES:",
    ]
    for ft, cnt in hp_type_counts.most_common():
        lines.append(f"  {ft:<48s} {cnt:>7,}  ({pct(cnt,N)})")
    lines += [
        "",f"  Total with ≥1 honeypot flag : {len(hp_flagged):,}  ({pct(len(hp_flagged),N)})",
        "","DISQUALIFIER FLAG TYPES:",
    ]
    for dt, cnt in dq_type_counts.most_common():
        lines.append(f"  {dt:<48s} {cnt:>7,}  ({pct(cnt,N)})")
    lines += [
        "",f"  Total disqualified : {len(dq_flagged):,}  ({pct(len(dq_flagged),N)})",
        "","─"*68,"HONEYPOT DETAILS (first 50)","─"*68,
    ]
    for item in hp_flagged[:50]:
        lines.append(f"\n  {item['id']}  |  {item['title']}")
        for ftype,detail in item["flags"]:
            lines.append(f"    [{ftype}]  {detail}")
    lines += ["","─"*68,"DISQUALIFIER DETAILS (first 100)","─"*68]
    for item in dq_flagged[:100]:
        lines.append(f"\n  {item['id']}  |  {item['title']}")
        for r in item["reasons"]:
            lines.append(f"    {r}")
    out_path = OUT_DIR/"honeypot_flags.txt"
    with open(out_path,"w",encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  ✓  Honeypot report       →  {out_path}")


def write_fit_simulation(results, N):
    valid = [r for r in results if not r["disqualified"] and not r["honeypot"]]
    lines = [
        "="*68,
        "JD FIT SIMULATION  (v2: IDF-weighted + JD-aware)",
        f"Dataset: {N:,}  |  eligible: {len(valid):,}  |  {TODAY.date()}",
        "="*68,"",
        f"{'Rk':>3s}  {'ID':>15s}  {'Title':<36s}  "
        f"{'YoE':>4s}  {'Score':>9s}  {'DaysAgo':>7s}  {'Notice':>6s}  Location",
        "─"*120,
    ]
    for i, r in enumerate(valid[:100], 1):
        d = r["details"]
        lines.append(
            f"{i:>3d}  {r['id']:>15s}  {r['title']:<36s}  "
            f"{r['yoe']:>4.1f}  {r['score']:>9.5f}  "
            f"{r['days_inactive']:>7d}d  {r['notice']:>6d}d  {r['location']}"
        )
        lines.append(
            f"     breakdown: title={d.get('title_score',0):.3f}  "
            f"idf={d.get('idf_skill_score',0):.3f}  "
            f"hard_req={d.get('hard_req_score',0):.3f}  "
            f"prod={d.get('prod_evidence_score',0):.3f}  "
            f"ctx={d.get('context_verified_score',0):.3f}  "
            f"exp={d.get('exp_score',0):.3f}  "
            f"avail={d.get('avail_multiplier',1):.3f}  "
            f"neg={d.get('neg_penalty',1):.3f}"
        )
        if r["reasons"]:
            lines.append(f"     signals: {' | '.join(r['reasons'][:7])}")
        lines.append("")
    lines += ["","─"*68,"SCORE DISTRIBUTION:","─"*68]
    buckets = collections.Counter()
    for r in results:
        s = r["score"]
        if   s >= 0.80: buckets["0.80–1.00"] += 1
        elif s >= 0.60: buckets["0.60–0.79"] += 1
        elif s >= 0.40: buckets["0.40–0.59"] += 1
        elif s >= 0.20: buckets["0.20–0.39"] += 1
        elif s >  0.00: buckets["0.01–0.19"] += 1
        else:           buckets["0.00 (disq/hp)"] += 1
    max_b = max(buckets.values()) if buckets else 1
    for bucket in ["0.80–1.00","0.60–0.79","0.40–0.59",
                   "0.20–0.39","0.01–0.19","0.00 (disq/hp)"]:
        n = buckets.get(bucket,0)
        lines.append(f"  {bucket:<20s} {n:>8,}  ({pct(n,N)})  {bar(n,max_b,25)}")
    out_path = OUT_DIR/"fit_simulation.txt"
    with open(out_path,"w",encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  ✓  Fit simulation        →  {out_path}")


def write_disqualifier_stats(results, dq_flagged, hp_flagged,
                             non_relevant_titles, top_relevant_titles,
                             industry_score, product_like_industries, N):
    hp_in_top10 = sum(1 for r in results[:10] if r["honeypot"])
    eligible    = [r for r in results
                   if not r["disqualified"] and not r["honeypot"]]
    lines = [
        "="*68,"DISQUALIFIER STATISTICS  (v2: data-driven)","="*68,"",
        f"  Total candidates          : {N:,}",
        f"  Disqualified              : {len(dq_flagged):,}  ({pct(len(dq_flagged),N)})",
        f"  Honeypot flags            : {len(hp_flagged):,}  ({pct(len(hp_flagged),N)})",
        f"  Eligible for ranking      : {len(eligible):,}  ({pct(len(eligible),N)})",
        "",
        f"  Honeypot rate in top 10   : {hp_in_top10}/10",
        f"  (Hackathon limit: ≤1/10 to avoid disqualification)",
        "","─"*68,
        "CORPUS-DERIVED NON-RELEVANT TITLES  (<15% JD skill coverage):","─"*68,
    ]
    for t in sorted(non_relevant_titles)[:40]:
        lines.append(f"  {t}")
    lines += ["","─"*68,
              "TOP RELEVANT TITLES  (by JD skill coverage, data-driven):","─"*68,
              f"  {'Title':<48s} {'Coverage':>10s}","  "+"-"*60]
    for t, s in top_relevant_titles[:25]:
        lines.append(f"  {t:<48s} {s:>10.4f}")
    lines += ["","─"*68,
              "INDUSTRY SCORES  (corpus-derived, product_like >= 0.40):","─"*68,
              f"  {'Industry':<42s} {'Score':>8s}  Type","  "+"-"*62]
    for ind, s in sorted(industry_score.items(), key=lambda x: -x[1]):
        kind = (
            "PRODUCT"   if ind in product_like_industries else
            "SERVICES"  if s < 0.15 else
            "neutral"
        )
        lines.append(f"  {ind:<42s} {s:>8.4f}  {kind}")
    out_path = OUT_DIR/"disqualifier_stats.txt"
    with open(out_path,"w",encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  ✓  Disqualifier stats    →  {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("\n"+"="*68)
    print("  Redrob Hackathon — EDA Script 2 v2 : Honeypot & Fit Analysis")
    print("  Changes: vocab-driven disqualifiers | IDF scoring | neg penalty")
    print("="*68+"\n")

    vocab, jd_hard_req, jd_positive, jd_negative, idf_lookup = load_vocab()
    cands = load_candidates()
    N     = len(cands)

    print("\n  Building title intelligence ...")
    (title_relevance_score, title_frequency,
     non_relevant_titles, top_relevant_titles) = build_title_intelligence(
        cands, jd_hard_req, jd_positive
    )
    print(f"  Titles: {len(title_relevance_score)}  |  "
          f"non-relevant: {len(non_relevant_titles)}  |  "
          f"top title: {top_relevant_titles[0] if top_relevant_titles else 'none'}")

    print("\n  Building industry intelligence ...")
    ind_score, product_like, services_like = build_industry_intelligence(
        cands, jd_hard_req, jd_positive
    )
    print(f"  Industries: {len(ind_score)}  |  "
          f"product-like(≥0.40): {len(product_like)} {sorted(product_like)[:5]}  |  "
          f"services-like: {len(services_like)}")

    print("\n  Computing IDF normalisation constant ...")
    idf_norm = compute_idf_normalisation(
        cands, idf_lookup, jd_hard_req, jd_positive
    )
    print(f"  IDF norm constant (p95): {idf_norm:.3f}")

    results, hp_flagged, dq_flagged, hp_type_counts, dq_type_counts = \
        run_analysis(
            cands, title_relevance_score, non_relevant_titles,
            jd_hard_req, jd_positive, jd_negative,
            idf_lookup, idf_norm, product_like
        )

    eligible = [r for r in results
                if not r["disqualified"] and not r["honeypot"]]
    print(f"\n  {'='*60}")
    print(f"  QUICK SUMMARY  ({N:,} candidates)")
    print(f"  {'='*60}")
    print(f"  Honeypot flags   : {len(hp_flagged):,}  ({pct(len(hp_flagged),N)})")
    print(f"  Disqualified     : {len(dq_flagged):,}  ({pct(len(dq_flagged),N)})")
    print(f"  Eligible         : {len(eligible):,}  ({pct(len(eligible),N)})")
    print(f"\n  Top 15 by fit score:")
    print(f"  {'Rk':>3s}  {'ID':>15s}  {'Title':<36s}  "
          f"{'Score':>9s}  {'YoE':>4s}  Location")
    print("  "+"-"*95)
    for i, r in enumerate(eligible[:15], 1):
        print(f"  {i:>3d}  {r['id']:>15s}  {r['title']:<36s}  "
              f"{r['score']:>9.5f}  {r['yoe']:>4.1f}  {r['location']}")

    write_honeypot_report(hp_flagged, hp_type_counts,
                          dq_flagged, dq_type_counts, N)
    write_fit_simulation(results, N)
    write_disqualifier_stats(
        results, dq_flagged, hp_flagged,
        non_relevant_titles, top_relevant_titles,
        ind_score, product_like, N
    )
    print(f"\n  All outputs in: {OUT_DIR}/")


if __name__ == "__main__":
    main()
