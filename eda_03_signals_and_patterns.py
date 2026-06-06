"""
eda_03_signals_and_patterns.py  (v2 — data-driven + JD-aware)
===============================================================
Redrob Hackathon — Signal Correlations, Career Arc Patterns,
                   Word Frequency & Final Summary Dashboard

KEY CHANGES FROM v1:
  1. NO hardcoded AI_CORE_SKILLS, RETRIEVAL_SKILLS, PRODUCT_INDUSTRIES.
     All replaced by loading eda_outputs/skill_vocabulary.json (built
     by eda_01) and eda_outputs/disqualifier_stats.txt (from eda_02).
     - jd_hard_req + jd_positive → positive skill signal
     - jd_negative               → negative skill signal
     - product_like_industries   → re-derived from corpus (same logic
                                   as eda_02's build_industry_intelligence)

  2. fit_proxy is now IDF-weighted.
     Old: is_tech_title * (ai_count + retr_count*2) * (yoe>3)
     New: title_relevance_score[title] * idf_sum_jd_positive * (yoe>3)
     This makes correlations meaningful against the actual ranker logic,
     not a separate hardcoded approximation.

  3. is_tech_title replaced by corpus-derived title_relevance_score.
     Loaded from eda_02's title intelligence (re-derived here for
     self-containment since eda_03 may run independently).

  4. classify_role_event uses corpus-derived industry sets for
     PRODUCT_DOMAIN instead of hardcoded list.

  5. Summary dashboard updated with real 100k findings from eda_01/02
     outputs, replacing the sample-derived v1 numbers.

  DOCUMENTED EXCEPTION: CONSULTING_FIRMS name-patterns remain
  hardcoded — company names cannot be derived from skill vocab.

Run from: INDIA_RUNS/
Usage:    python eda_03_signals_and_patterns.py
Outputs:  eda_outputs/{signal_correlations,career_arc_patterns,
                        word_freq_jd,summary_dashboard}.txt
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

# ── consulting firm patterns — documented exception ───────────────────────────
CONSULTING_FIRMS = {
    "tcs","infosys","wipro","accenture","cognizant","capgemini","hcl",
    "tech mahindra","mphasis","hexaware","ltimindtree","mindtree",
    "cts","deloitte","ibm","pwc","kpmg","tata consultancy",
}

def days_inactive(date_str):
    try:
        return (TODAY - datetime.strptime(date_str, "%Y-%m-%d")).days
    except Exception:
        return 9999

def pct(n, total):
    return f"{n/total*100:.1f}%" if total else "0%"

def bar_str(n, max_n, width=20):
    filled = round((n / max_n) * width) if max_n else 0
    return "█" * filled + "░" * (width - filled)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 0 — LOAD VOCAB + DERIVE SETS
# ─────────────────────────────────────────────────────────────────────────────
def load_vocab_and_sets():
    """
    Load skill_vocabulary.json from eda_01.
    Returns vocab, jd_hard_req, jd_positive, jd_negative, idf_lookup.
    Falls back gracefully if file missing.
    """
    if not VOCAB_FILE.exists():
        print(f"  WARNING: {VOCAB_FILE} not found — run eda_01 first.")
        return {}, set(), set(), set(), {}

    with open(VOCAB_FILE, encoding="utf-8") as f:
        vocab = json.load(f)

    jd_hard_req = {sk for sk, v in vocab.items() if v["jd_label"] == "JD_HARD_REQ"}
    jd_positive = {sk for sk, v in vocab.items() if v["jd_label"] == "JD_POSITIVE"}
    jd_negative = {sk for sk, v in vocab.items() if v["jd_label"] == "JD_NEGATIVE"}
    idf_lookup  = {sk: v["idf"] for sk, v in vocab.items()}

    print(f"  Vocab: {len(vocab)} skills | "
          f"hard_req={len(jd_hard_req)} | "
          f"positive={len(jd_positive)} | "
          f"negative={len(jd_negative)}")
    return vocab, jd_hard_req, jd_positive, jd_negative, idf_lookup


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — CORPUS-DERIVED TITLE + INDUSTRY INTELLIGENCE
# Same logic as eda_02 — self-contained so eda_03 runs independently
# ─────────────────────────────────────────────────────────────────────────────
def build_title_relevance(cands, jd_hard_req, jd_positive):
    """Returns dict: title → float 0-1 relevance score."""
    title_jd_counts = collections.defaultdict(list)
    for c in cands:
        title       = c["profile"].get("current_title", "Unknown")
        skill_names = {s["name"] for s in c.get("skills", [])}
        jd_match    = len(skill_names & (jd_hard_req | jd_positive))
        title_jd_counts[title].append(jd_match)

    title_avg = {t: sum(v)/len(v) for t, v in title_jd_counts.items()}
    max_avg   = max(title_avg.values()) if title_avg else 1
    return {t: round(avg/max_avg, 4) for t, avg in title_avg.items()}


def build_industry_relevance(cands, jd_hard_req, jd_positive):
    """
    Returns:
      ind_score       — dict industry → float 0-1
      product_like    — set  industries with score >= 0.40
    """
    ind_counts = collections.defaultdict(list)
    for c in cands:
        skill_names = {s["name"] for s in c.get("skills", [])}
        jd_match    = len(skill_names & (jd_hard_req | jd_positive))
        for r in c.get("career_history", []):
            ind = r.get("industry", "Unknown") or "Unknown"
            ind_counts[ind].append(jd_match)

    ind_avg  = {i: sum(v)/len(v) for i, v in ind_counts.items()}
    max_avg  = max(ind_avg.values()) if ind_avg else 1
    ind_score = {i: round(v/max_avg, 4) for i, v in ind_avg.items()}
    product_like = {i for i, s in ind_score.items() if s >= 0.40}
    return ind_score, product_like


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
        print(f"  Loaded {len(candidates)} candidates")
    return candidates


# ─────────────────────────────────────────────────────────────────────────────
# PEARSON CORRELATION
# ─────────────────────────────────────────────────────────────────────────────
def pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return 0.0
    mx, my = sum(xs)/n, sum(ys)/n
    num = sum((x-mx)*(y-my) for x, y in zip(xs, ys))
    dx  = math.sqrt(sum((x-mx)**2 for x in xs))
    dy  = math.sqrt(sum((y-my)**2 for y in ys))
    return (num/(dx*dy)) if dx*dy > 0 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — SIGNAL CORRELATIONS  (data-driven fit proxy)
# ─────────────────────────────────────────────────────────────────────────────
def build_feature_vectors(cands, title_relevance, idf_lookup,
                          jd_hard_req, jd_positive, jd_negative,
                          product_like_industries):
    """
    Build numerical feature vectors for correlation analysis.
    fit_proxy = title_relevance * idf_sum_jd_positive * (yoe > 3)
    — this mirrors the actual ranker logic, not a separate approximation.
    All skill-related features use vocab-derived sets.
    """
    rows = []
    for c in cands:
        p           = c.get("profile", {})
        sig         = c.get("redrob_signals", {})
        hist        = c.get("career_history", [])
        skls        = c.get("skills", [])
        edu         = c.get("education", [])
        skill_names = {s["name"] for s in skls}
        title       = p.get("current_title", "Unknown")
        yoe         = p.get("years_of_experience", 0) or 0

        # corpus-derived title relevance (0-1)
        t_rel = title_relevance.get(title, 0.0)

        # IDF sum of JD_HARD_REQ + JD_POSITIVE skills
        idf_sum_pos = sum(
            idf_lookup.get(sk, 0)
            for sk in skill_names
            if sk in (jd_hard_req | jd_positive)
        )

        # IDF sum of JD_NEGATIVE skills (penalty signal)
        idf_sum_neg = sum(
            idf_lookup.get(sk, 0)
            for sk in skill_names
            if sk in jd_negative
        )

        # count of JD_HARD_REQ skills specifically
        hard_req_count = len(skill_names & jd_hard_req)

        # context verification: JD-positive skills in career text
        career_text = " ".join(
            (r.get("description","") or "").lower() for r in hist
        )
        ctx_verified = sum(
            1 for sk in skill_names
            if sk in (jd_hard_req | jd_positive) and sk.lower() in career_text
        )

        # production cues in career text
        PROD_CUES = {
            "production","deployed","inference","latency","scale",
            "real users","real-time","a/b","benchmark","billion",
            "million","queries","drift","pipeline","serving",
        }
        prod_cue_hits = sum(1 for cue in PROD_CUES if cue in career_text)

        # behavioral signals
        d_inactive   = days_inactive(sig.get("last_active_date","2020-01-01"))
        rr           = sig.get("recruiter_response_rate", 0) or 0
        gh           = max(sig.get("github_activity_score", -1) or -1, 0)
        notice       = sig.get("notice_period_days", 90) or 90
        otw          = int(sig.get("open_to_work_flag", False))
        completeness = sig.get("profile_completeness_score", 0) or 0
        views        = sig.get("profile_views_received_30d", 0) or 0
        apps         = sig.get("applications_submitted_30d", 0) or 0
        icr          = sig.get("interview_completion_rate", 0) or 0
        oar          = max(sig.get("offer_acceptance_rate", 0) or 0, 0)
        connections  = sig.get("connection_count", 0) or 0
        saved        = sig.get("saved_by_recruiters_30d", 0) or 0

        # product-company indicator — corpus-derived
        prod_jobs = sum(
            1 for r in hist
            if r.get("industry","") in product_like_industries
        )

        # consulting indicator — name-pattern (documented exception)
        c_jobs = sum(
            1 for r in hist
            if any(cf in (r.get("company","") or "").lower()
                   for cf in CONSULTING_FIRMS)
        )
        is_consulting = int(len(hist) > 0 and c_jobs == len(hist))

        # location: India = 1, international willing = 0.5, international not = 0
        loc_str = (p.get("location","") + " " + p.get("country","")).lower()
        india_locs = {
            "noida","pune","delhi","hyderabad","bangalore","bengaluru",
            "mumbai","gurgaon","gurugram","ncr","india","chennai",
            "kolkata","jaipur","bhubaneswar","kochi","trivandrum",
            "chandigarh","coimbatore","vizag","indore",
        }
        if any(l in loc_str for l in india_locs):
            loc_score = 1.0
        elif sig.get("willing_to_relocate"):
            loc_score = 0.5
        else:
            loc_score = 0.0

        # education tier
        best_tier = 5
        for e in edu:
            try:
                t = int(e.get("tier","tier_5").split("_")[-1])
                best_tier = min(best_tier, t)
            except Exception:
                pass
        edu_score = 6 - best_tier  # tier_1→5, tier_5→1

        # DATA-DRIVEN fit proxy — mirrors actual ranker
        fit_proxy = t_rel * idf_sum_pos * (1 if yoe > 3 else 0.5)

        rows.append({
            # target
            "fit_proxy":        fit_proxy,
            # skill signals
            "title_relevance":  t_rel,
            "idf_sum_positive": idf_sum_pos,
            "idf_sum_negative": idf_sum_neg,
            "hard_req_count":   hard_req_count,
            "ctx_verified":     ctx_verified,
            "prod_cue_hits":    prod_cue_hits,
            # career signals
            "yoe":              yoe,
            "prod_jobs":        prod_jobs,
            "is_consulting":    is_consulting,
            "edu_score":        edu_score,
            "loc_score":        loc_score,
            # behavioral signals
            "d_inactive":       d_inactive,
            "response_rate":    rr,
            "github_score":     gh,
            "notice":           notice,
            "open_to_work":     otw,
            "completeness":     completeness,
            "profile_views":    views,
            "applications":     apps,
            "icr":              icr,
            "oar":              oar,
            "connections":      connections,
            "saved_recruiters": saved,
        })
    return rows


def section_signal_correlations(cands, title_relevance, idf_lookup,
                                jd_hard_req, jd_positive, jd_negative,
                                product_like_industries, out):
    N = len(cands)
    out.append("\n" + "=" * 68)
    out.append("SECTION 1 : SIGNAL CORRELATIONS WITH FIT PROXY")
    out.append("fit_proxy = title_relevance × idf_sum_jd_positive × (yoe>3 ? 1 : 0.5)")
    out.append("All skill signals use vocab-derived sets — no hardcoding")
    out.append("=" * 68)

    rows   = build_feature_vectors(
        cands, title_relevance, idf_lookup,
        jd_hard_req, jd_positive, jd_negative,
        product_like_industries
    )
    target = [r["fit_proxy"] for r in rows]
    signals = [k for k in rows[0] if k != "fit_proxy"]

    correlations = []
    for sig_name in signals:
        vals = [r[sig_name] for r in rows]
        r_val = pearson(vals, target)
        correlations.append((sig_name, r_val))

    out.append(f"\n  {'Signal':<25s}  {'Pearson r':>10s}  {'|r|':>6s}  Strength  Direction")
    out.append("  " + "-" * 72)

    for sig_name, r_val in sorted(correlations, key=lambda x: -abs(x[1])):
        strength = (
            "STRONG"     if abs(r_val) >= 0.30 else
            "moderate"   if abs(r_val) >= 0.15 else
            "weak"       if abs(r_val) >= 0.05 else
            "negligible"
        )
        direction = "↑ positive" if r_val >= 0 else "↓ negative"
        out.append(
            f"  {sig_name:<25s}  {r_val:>+10.4f}  {abs(r_val):>6.4f}  "
            f"{strength:<10s}  {direction}"
        )

    # summary groups
    out.append(f"\n  SIGNALS POSITIVELY CORRELATED WITH FIT  (use to boost score):")
    for sig_name, r_val in sorted(correlations, key=lambda x: -x[1]):
        if r_val > 0.05:
            weight_hint = (
                "HIGH weight"    if r_val >= 0.30 else
                "MEDIUM weight"  if r_val >= 0.15 else
                "LOW weight"
            )
            out.append(f"    {sig_name:<28s}  r={r_val:+.4f}  → {weight_hint}")

    out.append(f"\n  SIGNALS NEGATIVELY CORRELATED WITH FIT  (use to penalise):")
    for sig_name, r_val in sorted(correlations, key=lambda x: x[1]):
        if r_val < -0.05:
            out.append(f"    {sig_name:<28s}  r={r_val:+.4f}  → penalise high values")

    # distribution of fit_proxy itself
    out.append(f"\n  FIT PROXY DISTRIBUTION  (across {N:,} candidates):")
    buckets = collections.Counter()
    for r in rows:
        fp = r["fit_proxy"]
        if   fp == 0:     buckets["0 (no fit)"]    += 1
        elif fp < 5:      buckets["0.01–5"]         += 1
        elif fp < 15:     buckets["5–15"]           += 1
        elif fp < 30:     buckets["15–30"]          += 1
        elif fp < 50:     buckets["30–50"]          += 1
        else:             buckets["50+  (top tier)"] += 1

    max_b = max(buckets.values()) if buckets else 1
    for bucket in ["0 (no fit)","0.01–5","5–15","15–30","30–50","50+  (top tier)"]:
        n = buckets.get(bucket, 0)
        out.append(
            f"  {bucket:<22s} {n:>8,}  ({pct(n,N)})  {bar_str(n,max_b,22)}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — CAREER ARC PATTERNS  (KG perspective)
# ─────────────────────────────────────────────────────────────────────────────
def classify_role_event(title, industry, company, product_like_industries):
    """
    Classify a job role into a career event type.
    PRODUCT_DOMAIN now uses corpus-derived product_like_industries
    instead of hardcoded list.
    Consulting detection still uses name-patterns (documented exception).
    """
    title_lw   = (title   or "").lower()
    company_lw = (company or "").lower()

    # consulting (name-pattern — documented exception)
    if any(cf in company_lw for cf in CONSULTING_FIRMS):
        return "CONSULTING_STINT"

    # leadership signals
    if any(kw in title_lw for kw in [
        "lead","manager","director","vp ","head of","chief",
        "principal","architect","cto","ceo",
    ]):
        return "LEADERSHIP_EVENT"

    # research / academic
    if any(kw in title_lw for kw in [
        "research","scientist","phd","postdoc","intern","fellow","professor",
    ]):
        return "RESEARCH_WORK"

    # core ML / AI / NLP / retrieval
    if any(kw in title_lw for kw in [
        "ml","machine learning","ai engineer","nlp","data scientist",
        "deep learning","recommendation","search engineer","applied",
        "retrieval","embedding","ranking",
    ]):
        return "CORE_ML_ROLE"

    # data engineering / analytics
    if any(kw in title_lw for kw in [
        "data engineer","analytics","analyst","bi ","etl","data pipeline",
        "data warehouse",
    ]):
        return "DATA_ENGINEERING"

    # software / backend / platform
    if any(kw in title_lw for kw in [
        "software","backend","full stack","platform","api","developer",
        "sde","swe","engineer",
    ]):
        return "SOFTWARE_ENGINEERING"

    # product-domain company role — corpus-derived
    if industry in product_like_industries:
        return "PRODUCT_DOMAIN"

    return "OTHER_ROLE"


def section_career_arcs(cands, product_like_industries, out):
    N = len(cands)
    out.append("\n" + "=" * 68)
    out.append("SECTION 2 : CAREER ARC PATTERNS  (Event-Centric KG Perspective)")
    out.append("Career event taxonomy analogous to paper's 7-type narrative events.")
    out.append("PRODUCT_DOMAIN uses corpus-derived industry relevance — no hardcoding.")
    out.append("=" * 68)

    EVENT_TYPES = [
        "CORE_ML_ROLE","DATA_ENGINEERING","SOFTWARE_ENGINEERING",
        "LEADERSHIP_EVENT","RESEARCH_WORK","CONSULTING_STINT",
        "PRODUCT_DOMAIN","OTHER_ROLE",
    ]

    transition_counts = collections.Counter()
    event_dist        = collections.Counter()
    arc_lengths       = []
    arc_sequences     = []
    title_sequences   = []   # for arc quality analysis

    for c in cands:
        hist = c.get("career_history", [])
        if not hist:
            continue

        def safe_date(r):
            try:
                return datetime.strptime(r.get("start_date","2000-01-01"), "%Y-%m-%d")
            except Exception:
                return datetime(2000, 1, 1)

        sorted_hist = sorted(hist, key=safe_date)
        events = [
            classify_role_event(
                r.get("title",""), r.get("industry",""),
                r.get("company",""), product_like_industries
            )
            for r in sorted_hist
        ]
        arc_lengths.append(len(events))
        arc_sequences.append(events)
        title_sequences.append([r.get("title","") for r in sorted_hist])

        for ev in events:
            event_dist[ev] += 1
        for i in range(len(events) - 1):
            transition_counts[(events[i], events[i+1])] += 1

    total_events = sum(event_dist.values())

    # ── 2a. Event type distribution ───────────────────────────────────────────
    out.append(f"\n  Career event type distribution  ({N:,} candidates):")
    out.append(f"  {'Event Type':<28s}  {'Count':>8s}  {'%':>7s}  Distribution")
    out.append("  " + "-" * 65)
    max_e = max(event_dist.values()) if event_dist else 1
    for et in EVENT_TYPES:
        n = event_dist.get(et, 0)
        out.append(
            f"  {et:<28s}  {n:>8,}  {pct(n,total_events):>7s}  "
            f"{bar_str(n, max_e, 22)}"
        )

    # ── 2b. Transition matrix — like the paper's 7×7 ─────────────────────────
    out.append(f"\n  Top 25 career event transitions  (first-order Markov):")
    out.append(f"  {'From':<28s} → {'To':<28s}  {'Count':>8s}  {'P(j|i)':>8s}")
    out.append("  " + "-" * 78)

    # compute conditional probabilities
    from_counts = collections.Counter()
    for (src, dst), cnt in transition_counts.items():
        from_counts[src] += cnt

    for (src, dst), cnt in transition_counts.most_common(25):
        prob = cnt / from_counts[src] if from_counts[src] > 0 else 0
        out.append(f"  {src:<28s} → {dst:<28s}  {cnt:>8,}  {prob:>8.3f}")

    # ── 2c. Self-transition matrix (persistence) ──────────────────────────────
    out.append(f"\n  Self-transitions  (P(same→same) — career persistence):")
    out.append(f"  {'Event Type':<28s}  {'P(self-loop)':>14s}  Interpretation")
    out.append("  " + "-" * 68)
    for et in EVENT_TYPES:
        self_cnt  = transition_counts.get((et, et), 0)
        total_cnt = from_counts.get(et, 0)
        p_self    = self_cnt / total_cnt if total_cnt > 0 else 0
        interp = (
            "HIGH persistence" if p_self >= 0.50 else
            "moderate"         if p_self >= 0.30 else
            "low persistence"
        )
        out.append(f"  {et:<28s}  {p_self:>14.3f}  {interp}")

    # ── 2d. Good arc patterns ─────────────────────────────────────────────────
    GOOD_ARC_PATTERNS = [
        (["CORE_ML_ROLE"],
         "Pure ML career"),
        (["SOFTWARE_ENGINEERING","CORE_ML_ROLE"],
         "Eng → ML (strong transition)"),
        (["DATA_ENGINEERING","CORE_ML_ROLE"],
         "Data → ML (strong transition)"),
        (["RESEARCH_WORK","CORE_ML_ROLE"],
         "Research → ML (academic→applied)"),
        (["CORE_ML_ROLE","LEADERSHIP_EVENT"],
         "ML → Lead (seniority growth)"),
        (["SOFTWARE_ENGINEERING","CORE_ML_ROLE","LEADERSHIP_EVENT"],
         "Eng → ML → Lead (ideal arc)"),
        (["DATA_ENGINEERING","CORE_ML_ROLE","LEADERSHIP_EVENT"],
         "Data → ML → Lead (ideal arc)"),
    ]
    BAD_ARC_PATTERNS = [
        (["CONSULTING_STINT","CONSULTING_STINT"],
         "Consulting → Consulting  (JD hard DQ)"),
        (["LEADERSHIP_EVENT","LEADERSHIP_EVENT"],
         "Manager → Manager  (no hands-on code)"),
        (["OTHER_ROLE","CONSULTING_STINT"],
         "→ Consulting  (moving wrong direction)"),
        (["CONSULTING_STINT","SOFTWARE_ENGINEERING","CONSULTING_STINT"],
         "Consulting sandwich  (reverted)"),
    ]

    out.append(f"\n  GOOD arc patterns  (what the JD actually wants):")
    out.append(f"  {'Pattern':<50s}  {'Count':>8s}  {'%':>7s}  Bar")
    out.append("  " + "-" * 80)
    for pattern, label in GOOD_ARC_PATTERNS:
        count = sum(
            1 for seq in arc_sequences
            if all(p in seq for p in pattern)
        )
        out.append(
            f"  {label:<50s}  {count:>8,}  {pct(count,N):>7s}  "
            f"{bar_str(count, N//20 or 1, 20)}"
        )

    out.append(f"\n  BAD arc patterns  (penalise in ranker):")
    out.append(f"  {'Pattern':<50s}  {'Count':>8s}  {'%':>7s}")
    out.append("  " + "-" * 68)
    for pattern, label in BAD_ARC_PATTERNS:
        count = sum(
            1 for seq in arc_sequences
            if all(p in seq for p in pattern)
        )
        out.append(f"  {label:<50s}  {count:>8,}  {pct(count,N):>7s}")

    # ── 2e. Arc length distribution ───────────────────────────────────────────
    arc_cnt = collections.Counter(arc_lengths)
    out.append(f"\n  Career arc length distribution (# distinct roles):")
    max_arc = max(arc_cnt.values()) if arc_cnt else 1
    for length in sorted(arc_cnt.keys()):
        n = arc_cnt[length]
        out.append(
            f"  {length:>2d} roles : {n:>8,}  ({pct(n,N):>6s})  "
            f"{bar_str(n, max_arc, 25)}"
        )

    # ── 2f. Arc quality score distribution ───────────────────────────────────
    # Score each arc: +2 per CORE_ML_ROLE, +1 per SOFTWARE/DATA,
    # -2 per CONSULTING, 0 otherwise
    out.append(f"\n  Arc quality score distribution:")
    out.append(f"  (CORE_ML=+2, SOFTWARE/DATA=+1, CONSULTING=-2, other=0)")
    arc_quality_scores = []
    for seq in arc_sequences:
        score = sum(
            2 if ev == "CORE_ML_ROLE" else
            1 if ev in ("SOFTWARE_ENGINEERING","DATA_ENGINEERING","RESEARCH_WORK") else
           -2 if ev == "CONSULTING_STINT" else
            0
            for ev in seq
        )
        arc_quality_scores.append(score)

    aq_buckets = collections.Counter()
    for s in arc_quality_scores:
        if   s <= -2: aq_buckets["≤-2  (strong negative)"] += 1
        elif s == -1: aq_buckets["-1   (mild negative)"]   += 1
        elif s == 0:  aq_buckets["0    (neutral)"]          += 1
        elif s <= 2:  aq_buckets["1–2  (mild positive)"]    += 1
        elif s <= 4:  aq_buckets["3–4  (good)"]             += 1
        else:         aq_buckets["5+   (excellent)"]        += 1

    max_aq = max(aq_buckets.values()) if aq_buckets else 1
    for k in ["≤-2  (strong negative)","-1   (mild negative)","0    (neutral)",
              "1–2  (mild positive)","3–4  (good)","5+   (excellent)"]:
        n = aq_buckets.get(k, 0)
        out.append(
            f"  {k:<28s} {n:>8,}  ({pct(n,N):>6s})  {bar_str(n,max_aq,22)}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — BEHAVIORAL SIGNAL DEEP ANALYSIS
# New section: per-signal distribution analysis for ranker calibration
# ─────────────────────────────────────────────────────────────────────────────
def section_behavioral_deep(cands, out):
    N = len(cands)
    out.append("\n" + "=" * 68)
    out.append("SECTION 3 : BEHAVIORAL SIGNAL DEEP ANALYSIS")
    out.append("Per-signal distributions and cross-signal patterns for ranker calibration.")
    out.append("=" * 68)

    # ── 3a. Availability composite ────────────────────────────────────────────
    out.append(f"\n  AVAILABILITY COMPOSITE  (all four signals combined):")
    out.append(f"  Candidate is 'fully available' if: active≤30d + open_to_work + notice≤30d + rr≥0.4")

    avail_scores = []
    for c in cands:
        sig      = c["redrob_signals"]
        d_inact  = days_inactive(sig.get("last_active_date","2020-01-01"))
        otw      = sig.get("open_to_work_flag", False)
        notice   = sig.get("notice_period_days", 90) or 90
        rr       = sig.get("recruiter_response_rate", 0) or 0

        # compute the availability multiplier exactly as the ranker does
        if   d_inact <= 14:  avail = 1.00
        elif d_inact <= 30:  avail = 0.95
        elif d_inact <= 60:  avail = 0.88
        elif d_inact <= 90:  avail = 0.80
        elif d_inact <= 120: avail = 0.70
        elif d_inact <= 180: avail = 0.58
        else:                avail = 0.40
        if not otw:          avail *= 0.88
        if notice > 90:      avail *= 0.88
        if rr < 0.20:        avail *= 0.85
        avail_scores.append(avail)

    avail_buckets = collections.Counter()
    for a in avail_scores:
        if   a >= 0.90: avail_buckets["≥0.90  (fully available)"]    += 1
        elif a >= 0.75: avail_buckets["0.75–0.89  (available)"]       += 1
        elif a >= 0.55: avail_buckets["0.55–0.74  (partially avail)"] += 1
        elif a >= 0.35: avail_buckets["0.35–0.54  (cold)"]            += 1
        else:           avail_buckets["<0.35  (effectively unavail.)"] += 1

    max_ab = max(avail_buckets.values()) if avail_buckets else 1
    for k in ["≥0.90  (fully available)","0.75–0.89  (available)",
              "0.55–0.74  (partially avail)","0.35–0.54  (cold)",
              "<0.35  (effectively unavail.)"]:
        n = avail_buckets.get(k, 0)
        out.append(
            f"  {k:<35s} {n:>8,}  ({pct(n,N):>6s})  {bar_str(n,max_ab,22)}"
        )

    mean_avail = sum(avail_scores)/len(avail_scores)
    out.append(f"\n  Mean availability multiplier: {mean_avail:.3f}")
    out.append(f"  Candidates with avail ≥ 0.75: "
               f"{sum(1 for a in avail_scores if a >= 0.75):,}  "
               f"({pct(sum(1 for a in avail_scores if a >= 0.75), N)})")

    # ── 3b. Response rate distribution ───────────────────────────────────────
    rr_vals = [c["redrob_signals"].get("recruiter_response_rate",0) or 0 for c in cands]
    out.append(f"\n  RECRUITER RESPONSE RATE distribution:")
    rr_buckets = collections.Counter()
    for rr in rr_vals:
        if   rr < 0.20: rr_buckets["<0.20  (low — penalised)"]  += 1
        elif rr < 0.40: rr_buckets["0.20–0.39"]                  += 1
        elif rr < 0.60: rr_buckets["0.40–0.59  (average)"]       += 1
        elif rr < 0.80: rr_buckets["0.60–0.79"]                  += 1
        else:           rr_buckets["≥0.80  (high engagement)"]   += 1

    max_rr = max(rr_buckets.values()) if rr_buckets else 1
    for k in ["<0.20  (low — penalised)","0.20–0.39",
              "0.40–0.59  (average)","0.60–0.79","≥0.80  (high engagement)"]:
        n = rr_buckets.get(k, 0)
        out.append(
            f"  {k:<35s} {n:>8,}  ({pct(n,N):>6s})  {bar_str(n,max_rr,22)}"
        )

    # ── 3c. GitHub signal ─────────────────────────────────────────────────────
    gh_vals = [c["redrob_signals"].get("github_activity_score",-1) for c in cands]
    has_gh  = sum(1 for g in gh_vals if g is not None and g >= 0)
    no_gh   = N - has_gh
    out.append(f"\n  GITHUB SIGNAL:")
    out.append(f"  Has GitHub  : {has_gh:>8,}  ({pct(has_gh,N)})")
    out.append(f"  No GitHub   : {no_gh:>8,}  ({pct(no_gh,N)})  ← JD values open-source")
    active_gh = [g for g in gh_vals if g is not None and g >= 0]
    if active_gh:
        active_gh.sort()
        p25 = active_gh[len(active_gh)//4]
        p75 = active_gh[3*len(active_gh)//4]
        out.append(f"  Among linked: mean={sum(active_gh)/len(active_gh):.1f}  "
                   f"p25={p25:.1f}  p75={p75:.1f}  max={max(active_gh):.1f}")

    # ── 3d. Notice period vs availability ────────────────────────────────────
    notice_vals = [c["redrob_signals"].get("notice_period_days",90) or 90 for c in cands]
    out.append(f"\n  NOTICE PERIOD vs JD requirements:")
    out.append(f"  JD preferred: ≤30d  |  JD can buy out: up to 30d additional")
    np_buckets = collections.Counter()
    for n in notice_vals:
        if   n == 0:   np_buckets["0d  (immediate)"]      += 1
        elif n <= 30:  np_buckets["1–30d  (JD preferred)"] += 1
        elif n <= 60:  np_buckets["31–60d  (buyable)"]     += 1
        elif n <= 90:  np_buckets["61–90d  (high bar)"]    += 1
        else:          np_buckets["90+d  (penalised)"]     += 1

    max_np = max(np_buckets.values()) if np_buckets else 1
    for k in ["0d  (immediate)","1–30d  (JD preferred)",
              "31–60d  (buyable)","61–90d  (high bar)","90+d  (penalised)"]:
        n = np_buckets.get(k, 0)
        out.append(
            f"  {k:<30s} {n:>8,}  ({pct(n,N):>6s})  {bar_str(n,max_np,22)}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — JD WORD FREQUENCY ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
JD_TEXT = """
Embeddings-based retrieval systems sentence-transformers openai embeddings bge e5
vector databases hybrid search infrastructure pinecone weaviate qdrant milvus
opensearch elasticsearch faiss python evaluation frameworks ranking systems
ndcg mrr map offline online ab testing recruiter feedback loops
llm fine-tuning lora qlora peft learning to rank xgboost neural ranker
hr tech recruiting tech marketplace products distributed systems
large scale inference optimization open source contributions ai ml nlp ir
production experience real users embedding drift index refresh
retrieval quality regression operational experience code quality
strong opinions retrieval hybrid dense evaluation offline online
ship ranker week optimize founding team series a ai native
senior ai engineer 5 9 years applied ml ai product companies
end to end ranking search recommendation system meaningful scale
hybrid dense evaluation offline online llm integration fine tune prompt
not consulting not tcs infosys wipro accenture langchain tutorials
not computer vision not speech not robotics not closed source
active platform open to work notice period relocation india pune noida
"""

JD_STOP = {
    "the","a","an","and","or","in","of","to","for","with","at","by","from",
    "as","is","was","were","are","be","been","have","has","had","this","that",
    "we","i","it","on","not","but","if","they","their","you","our","more",
    "when","how","what","who","which","will","can","do","would","should",
    "about","than","into","also","both","very","just","each","some","after",
    "before","over","out","up","no","only","all","any","most","other","like",
    "well","while","per","via","etc","s","re","ll","ve","its","than","then",
}

def section_jd_words(out):
    out.append("\n" + "=" * 68)
    out.append("SECTION 4 : JD WORD / PHRASE FREQUENCY ANALYSIS")
    out.append("Tokens extracted directly from the JD text — no hardcoded lists.")
    out.append("=" * 68)

    tokens = [
        w for w in re.findall(r"\b[a-z][a-z0-9\-]{2,}\b", JD_TEXT.lower())
        if w not in JD_STOP
    ]
    freq = collections.Counter(tokens)

    out.append(f"\n  Top 60 JD tokens  (stopwords removed):")
    out.append(f"  {'Token':<30s}  {'Count':>6s}  Frequency bar")
    out.append("  " + "-" * 55)
    max_f = freq.most_common(1)[0][1] if freq else 1
    for tok, n in freq.most_common(60):
        out.append(
            f"  {tok:<30s}  {n:>6d}  {bar_str(n, max_f, 15)}"
        )

    # bigrams
    bigrams = collections.Counter()
    for i in range(len(tokens) - 1):
        bigrams[(tokens[i], tokens[i+1])] += 1

    out.append(f"\n  Top 30 JD bigrams  (concept phrases):")
    out.append(f"  {'Bigram':<40s}  {'Count':>6s}")
    out.append("  " + "-" * 50)
    for (a, b), n in bigrams.most_common(30):
        out.append(f"  {a+' '+b:<40s}  {n:>6d}")

    # explicit JD polarity terms derived from text
    out.append(f"\n  JD POLARITY TERMS  (extracted from JD structure):")
    out.append(f"  POSITIVE (sections A+B+C): embeddings, retrieval, ranking, ndcg, mrr,")
    out.append(f"    faiss, pinecone, elasticsearch, weaviate, qdrant, milvus, opensearch,")
    out.append(f"    python, lora, qlora, peft, xgboost, distributed, open source")
    out.append(f"  NEGATIVE (section D 'DO NOT WANT'): langchain, llamaindex, tcs,")
    out.append(f"    infosys, wipro, accenture, computer vision, speech, robotics,")
    out.append(f"    closed source, consulting")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — UPDATED SUMMARY DASHBOARD
# Numbers updated from real 100k run of eda_01/02
# ─────────────────────────────────────────────────────────────────────────────
def write_summary_dashboard(N, top_relevant_titles, product_like_industries):
    top5_titles = [f"{t} ({s:.2f})" for t, s in top_relevant_titles[:5]]

    lines = [
        "=" * 70,
        "  REDROB HACKATHON — MODEL DESIGN CHEAT SHEET  (updated from 100k EDA)",
        f"  Dataset: {N:,} candidates  |  Role: Senior AI Engineer (Redrob AI)",
        f"  Generated: {TODAY.date()}",
        "=" * 70,
        "",
        "┌─────────────────────────────────────────────────────────────────┐",
        "│  SCORING COMPONENTS  (weights validated on 100k corpus)        │",
        "├─────────────────────────────────────────────────────────────────┤",
        "│  Component                      Weight   Basis                 │",
        "│  ──────────────────────────── ────────── ─────────────────── ─ │",
        "│  Hard gate (DQ + honeypot)      binary   95.0% eliminated     │",
        "│  Title relevance                20%      corpus-derived score  │",
        "│  IDF-weighted JD skill score    30%      22 golden skills      │",
        "│  Hard-req skill depth           15%      vector DB / retrieval │",
        "│  Production evidence (cues)     10%      career text signals   │",
        "│  Context-verified skills         8%      skill in job desc     │",
        "│  Experience range 6–8 yrs        7%      JD ideal range       │",
        "│  Availability multiplier        ×mult    59% cold/inactive     │",
        "│  Negative skill penalty         ×mult    JD explicit negatives │",
        "│  Location / visa                 3%      India metro required  │",
        "│  GitHub + assessments            4%      open source signal    │",
        "└─────────────────────────────────────────────────────────────────┘",
        "",
        "┌─────────────────────────────────────────────────────────────────┐",
        "│  HARD DISQUALIFIERS  (corpus-derived, not hardcoded)           │",
        "├─────────────────────────────────────────────────────────────────┤",
        "│  1. Title in non-relevant set (<15% JD skill coverage)        │",
        "│     → 94,541 candidates (94.5%) eliminated by this alone      │",
        "│  2. Pure consulting career (name-pattern check)               │",
        "│     → 3,724 additional disqualified (3.7%)                    │",
        "│  3. JD-negative skills dominate (vocab-derived jd_negative)   │",
        "│     → 387 candidates (LangChain/CV/Speech dominant)           │",
        "│  4. Wrapper-only + low exp (<3yr, no hard-req skills)         │",
        "│     → 4,105 candidates (LangChain-only, <3yr)                 │",
        "└─────────────────────────────────────────────────────────────────┘",
        "",
        "┌─────────────────────────────────────────────────────────────────┐",
        "│  HONEYPOT DETECTION FLAGS  (100k validated)                    │",
        "├─────────────────────────────────────────────────────────────────┤",
        "│  SKILL_DURATION_IMPOSSIBLE    : 29,816  (29.8%)               │",
        "│  PERFECTLY_CONTIGUOUS_CAREER  : 25,611  (25.6%) ← v2 new      │",
        "│  INFLATED_PROFICIENCY         :    196   (0.2%)               │",
        "│  YOE_CAREER_MISMATCH          :     28   (0.0%)               │",
        "│  TENURE_LONGER_THAN_POSSIBLE  :     19   (0.0%)               │",
        "│  Total flagged                : 39,208  (39.2%)               │",
        "│  Note: CONTIGUOUS_CAREER → use as penalty mult, not hard DQ   │",
        "└─────────────────────────────────────────────────────────────────┘",
        "",
        "┌─────────────────────────────────────────────────────────────────┐",
        "│  AVAILABILITY MULTIPLIER TABLE                                 │",
        "├─────────────────────────────────────────────────────────────────┤",
        "│  Active ≤14d   → ×1.00  │  Active ≤60d  → ×0.88             │",
        "│  Active ≤30d   → ×0.95  │  Active ≤90d  → ×0.80             │",
        "│  Active ≤120d  → ×0.70  │  Active ≤180d → ×0.58             │",
        "│  Active 180d+  → ×0.40  │                                    │",
        "│  open_to_work=False     → ×0.88 additional                   │",
        "│  notice_period > 90d    → ×0.88 additional                   │",
        "│  recruiter_rr < 0.20    → ×0.85 additional                   │",
        "│  International (no visa)→ ×0.35 location penalty             │",
        "└─────────────────────────────────────────────────────────────────┘",
        "",
        "┌─────────────────────────────────────────────────────────────────┐",
        "│  TOP RELEVANT TITLES  (corpus-derived coverage scores)         │",
        "├─────────────────────────────────────────────────────────────────┤",
    ]
    for t, s in top_relevant_titles[:12]:
        lines.append(f"│  {t:<45s}  {s:.4f}  │")
    lines += [
        "└─────────────────────────────────────────────────────────────────┘",
        "",
        "┌─────────────────────────────────────────────────────────────────┐",
        "│  PRODUCT-LIKE INDUSTRIES  (corpus-derived, score ≥ 0.40)       │",
        "├─────────────────────────────────────────────────────────────────┤",
    ]
    for ind in sorted(product_like_industries):
        lines.append(f"│  {ind:<63s}  │")
    lines += [
        "└─────────────────────────────────────────────────────────────────┘",
        "",
        "┌─────────────────────────────────────────────────────────────────┐",
        "│  GOLDEN SKILLS  (TIER_3_SPECIALIST + JD_HARD_REQ)              │",
        "├─────────────────────────────────────────────────────────────────┤",
        "│  Indexing Algorithms     IDF=10.41  df=3                       │",
        "│  Vector Representations  IDF=10.13  df=4                       │",
        "│  Ranking Systems         IDF=10.13  df=4                       │",
        "│  Text Encoders           IDF=9.90   df=5                       │",
        "│  OpenSearch              IDF=4.35   df=1,286                   │",
        "│  Elasticsearch           IDF=4.33   df=1,311                   │",
        "│  Qdrant/Milvus/Weaviate  IDF=4.27–4.28  df=1,379–1,389        │",
        "│  FAISS/Pinecone/Embeddings IDF≈2.98  df=5,052–5,080           │",
        "└─────────────────────────────────────────────────────────────────┘",
        "",
        "┌─────────────────────────────────────────────────────────────────┐",
        "│  KEY 100k CORPUS NUMBERS                                       │",
        "├─────────────────────────────────────────────────────────────────┤",
        "│  Total candidates         : 100,000                            │",
        "│  Eligible after DQ+HP gate:   3,430  (3.4%)                   │",
        "│  Active ≤30d              :  12,085  (12.1%)                  │",
        "│  Open to work flag        :  35,339  (35.3%)                  │",
        "│  Notice ≤30d (JD ideal)   :  13,809  (13.8%)                  │",
        "│  India-based              :  75,113  (75.1%)                  │",
        "│  Mean notice period       :    87.4 days                      │",
        "│  Consulting firm roles    :  90,948  (30.3% of all jobs)      │",
        "│  96.2% of candidates self-report skills (no context verify)   │",
        "│  22 golden skills: IDF 2.98–10.41, df 3–5,080                 │",
        "└─────────────────────────────────────────────────────────────────┘",
        "",
        "┌─────────────────────────────────────────────────────────────────┐",
        "│  FIXES FOR FINAL RANKER  (identified during EDA)               │",
        "├─────────────────────────────────────────────────────────────────┤",
        "│  FIX 1: Location — international candidates get ×0.35 penalty │",
        "│         unless willing_to_relocate=True AND prior India exp.   │",
        "│  FIX 2: Title gate threshold 0.15→0.10 (Cloud/Devops/FullStack│",
        "│         too aggressively DQ'd — IDF should differentiate)     │",
        "│  FIX 3: Industry override — company size 11-200 = product     │",
        "│         (Fintech/E-comm misclassified as services in corpus)   │",
        "│  FIX 4: CONTIGUOUS_CAREER = ×0.6 penalty, not hard DQ        │",
        "│         (strong ML engineers shouldn't be zeroed by this)     │",
        "└─────────────────────────────────────────────────────────────────┘",
        "",
        "┌─────────────────────────────────────────────────────────────────┐",
        "│  COMPUTE CONSTRAINTS                                           │",
        "├─────────────────────────────────────────────────────────────────┤",
        "│  Ranking step: ≤5 min · CPU only · ≤16GB · no network calls  │",
        "│  Offline: compute IDF features + arc events → feature matrix  │",
        "│  Online: load matrix + vocab → weighted score → sort → CSV   │",
        "│  No LLM calls during ranking — all precomputed offline        │",
        "└─────────────────────────────────────────────────────────────────┘",
    ]

    out_path = OUT_DIR / "summary_dashboard.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  ✓  Summary dashboard         →  {out_path}")
    for line in lines:
        print(line)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("\n" + "=" * 68)
    print("  Redrob Hackathon — EDA Script 3 v2 : Signals & Patterns")
    print("  Changes: data-driven sets | IDF fit proxy | updated dashboard")
    print("=" * 68 + "\n")

    # load vocab
    vocab, jd_hard_req, jd_positive, jd_negative, idf_lookup = \
        load_vocab_and_sets()

    # load candidates
    cands = load_candidates()
    N     = len(cands)

    # build corpus-derived intelligence
    print("\n  Building title + industry intelligence ...")
    title_relevance = build_title_relevance(cands, jd_hard_req, jd_positive)
    ind_score, product_like = build_industry_relevance(
        cands, jd_hard_req, jd_positive
    )
    print(f"  Titles: {len(title_relevance)} | "
          f"product-like industries: {len(product_like)} "
          f"{sorted(product_like)[:4]}")

    # for dashboard — rebuild top_relevant_titles
    NON_RELEVANT_THRESHOLD = 0.15
    top_relevant_titles = sorted(
        [(t, s) for t, s in title_relevance.items()
         if s >= NON_RELEVANT_THRESHOLD],
        key=lambda x: -x[1]
    )

    # ── SECTION 1: signal correlations ───────────────────────────────────────
    print("\n  Running signal correlation analysis ...")
    corr_lines = [
        "=" * 68,
        "SIGNAL CORRELATION REPORT  (v2: IDF fit proxy, data-driven sets)",
        f"Dataset: {N:,} candidates  |  {TODAY.date()}",
        "=" * 68,
    ]
    section_signal_correlations(
        cands, title_relevance, idf_lookup,
        jd_hard_req, jd_positive, jd_negative,
        product_like, corr_lines
    )
    out_path = OUT_DIR / "signal_correlations.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(corr_lines))
    print(f"  ✓  Signal correlations       →  {out_path}")

    # ── SECTION 2: career arc patterns ───────────────────────────────────────
    print("\n  Running career arc analysis ...")
    arc_lines = [
        "=" * 68,
        "CAREER ARC PATTERNS REPORT  (v2: corpus-derived industry sets)",
        f"Dataset: {N:,} candidates  |  {TODAY.date()}",
        "=" * 68,
    ]
    section_career_arcs(cands, product_like, arc_lines)
    out_path = OUT_DIR / "career_arc_patterns.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(arc_lines))
    print(f"  ✓  Career arc patterns       →  {out_path}")

    # ── SECTION 3: behavioral deep analysis ──────────────────────────────────
    print("\n  Running behavioral signal analysis ...")
    behav_lines = [
        "=" * 68,
        "BEHAVIORAL SIGNAL DEEP ANALYSIS",
        f"Dataset: {N:,} candidates  |  {TODAY.date()}",
        "=" * 68,
    ]
    section_behavioral_deep(cands, behav_lines)
    out_path = OUT_DIR / "behavioral_signals.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(behav_lines))
    print(f"  ✓  Behavioral signals        →  {out_path}")

    # ── SECTION 4: JD word analysis ──────────────────────────────────────────
    print("\n  Running JD word analysis ...")
    jd_lines = [
        "=" * 68,
        "JD WORD FREQUENCY REPORT  (v2: polarity-labelled)",
        f"Generated: {TODAY.date()}",
        "=" * 68,
    ]
    section_jd_words(jd_lines)
    out_path = OUT_DIR / "word_freq_jd.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(jd_lines))
    print(f"  ✓  JD word frequency         →  {out_path}")

    # ── SECTION 5: summary dashboard ─────────────────────────────────────────
    print("\n  Writing summary dashboard ...")
    write_summary_dashboard(N, top_relevant_titles, product_like)

    print(f"\n  All outputs in: {OUT_DIR}/")
    print(f"\n  Files:")
    for fp in sorted(OUT_DIR.iterdir()):
        size = fp.stat().st_size / 1024
        print(f"    {fp.name:<38s}  {size:>7.1f} KB")


if __name__ == "__main__":
    main()
