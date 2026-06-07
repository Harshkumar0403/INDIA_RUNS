"""
utils.py
========
Shared helpers used across the entire pipeline.
No ML models loaded here — pure utility functions only.
"""

import json
import re
import math
import pickle
import collections
from datetime import datetime
from pathlib import Path
from typing import Optional

TODAY = datetime(2026, 6, 2)

# ─────────────────────────────────────────────────────────────────
# TEXT HELPERS
# ─────────────────────────────────────────────────────────────────
STOP_WORDS = {
    "the","a","an","and","or","in","of","to","for","with","at","by","from",
    "as","is","was","were","are","be","been","being","have","has","had",
    "this","that","these","those","their","they","i","my","our","we","it",
    "its","on","up","also","which","who","what","when","where","how","all",
    "more","into","than","s","team","work","worked","working","including",
    "across","within","through","between","multiple","various","using","used",
    "use","part","while","over","each","new","key","well","based",
    "experience","years","built","build","building","developed","helped",
    "support","led","managed","manage","strong","good","great",
    "responsible","responsibilities","role","company","join","joined",
}

def tokenize(text: str) -> list[str]:
    """Split text into lowercase word tokens."""
    return re.findall(r"\b\w+\b", (text or "").lower())

def clean_tokens(text: str) -> list[str]:
    """Tokenize, remove stop-words and short tokens."""
    return [w for w in re.findall(r"\b[a-z]{3,}\b", (text or "").lower())
            if w not in STOP_WORDS]

def normalize_text(text: str) -> str:
    """Lowercase, collapse whitespace, strip."""
    return re.sub(r"\s+", " ", (text or "").lower()).strip()

# ─────────────────────────────────────────────────────────────────
# DATE / AVAILABILITY HELPERS
# ─────────────────────────────────────────────────────────────────
def days_since(date_str: str) -> int:
    """Days between date_str (YYYY-MM-DD) and TODAY. Returns 9999 on error."""
    try:
        return (TODAY - datetime.strptime(date_str, "%Y-%m-%d")).days
    except Exception:
        return 9999

def availability_score(days_inactive: int, lambda_decay: float = 0.005) -> float:
    """
    Exponential decay availability score.
    f(t) = e^(-λt)
    Calibrated: 30d→0.86, 90d→0.64, 180d→0.41, 365d→0.16
    """
    return math.exp(-lambda_decay * days_inactive)

# ─────────────────────────────────────────────────────────────────
# CANDIDATE DATA ACCESSORS
# ─────────────────────────────────────────────────────────────────
def get_career_text(cand: dict) -> str:
    """Concatenated text of all career history descriptions."""
    return " ".join(
        (r.get("description", "") or "")
        for r in cand.get("career_history", [])
    ).lower()

def get_summary_text(cand: dict) -> str:
    return (cand.get("profile", {}).get("summary", "") or "").lower()

def get_skills_text(cand: dict) -> str:
    return " ".join(
        (s.get("name", "") or "")
        for s in cand.get("skills", [])
    ).lower()

def get_full_text(cand: dict) -> str:
    """Full concatenated text: summary + career + skills."""
    return " ".join([
        get_summary_text(cand),
        get_career_text(cand),
        get_skills_text(cand),
    ])

def get_career_narrative(cand: dict) -> str:
    """
    Career narrative for embedding — the primary semantic chunk.
    Ordered chronologically: summary → oldest role → newest role.
    """
    p    = cand.get("profile", {})
    hist = cand.get("career_history", [])
    skls = cand.get("skills", [])

    # sort history by start_date ascending (oldest first)
    def safe_date(r):
        try:
            return datetime.strptime(r.get("start_date", "2000-01-01"), "%Y-%m-%d")
        except Exception:
            return datetime(2000, 1, 1)

    sorted_hist = sorted(hist, key=safe_date)

    parts = [p.get("summary", "") or ""]
    for r in sorted_hist:
        title   = r.get("title", "") or ""
        company = r.get("company", "") or ""
        desc    = r.get("description", "") or ""
        industry = r.get("industry", "") or ""
        if title or desc:
            parts.append(f"{title} at {company} ({industry}). {desc}")

    # append top skills for extra signal
    top_skills = [s.get("name","") for s in skls[:10]]
    if top_skills:
        parts.append("Skills: " + ", ".join(top_skills))

    return " ".join(p for p in parts if p).strip()

def get_skill_names(cand: dict) -> set[str]:
    """Set of skill names (original casing)."""
    return {s.get("name", "").strip() for s in cand.get("skills", []) if s.get("name")}

def get_skill_names_lower(cand: dict) -> set[str]:
    """Set of skill names lowercased."""
    return {n.lower() for n in get_skill_names(cand)}

# ─────────────────────────────────────────────────────────────────
# SERIALIZATION
# ─────────────────────────────────────────────────────────────────
def save_pickle(obj, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=5)
    print(f"  Saved → {path}  ({path.stat().st_size/1024/1024:.1f} MB)")

def load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)

def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(obj, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

# ─────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────
def load_candidates(
    jsonl_path: Path,
    sample_path: Optional[Path] = None,
    max_n: Optional[int] = None,
    verbose: bool = True,
) -> list[dict]:
    """
    Load from full .jsonl, or fall back to sample .json.
    max_n: cap for quick test runs (e.g. max_n=1000).
    """
    candidates = []

    if Path(jsonl_path).exists():
        if verbose:
            print(f"  Loading {jsonl_path}  (~30s for 100k) ...")
        with open(jsonl_path, "r", encoding="utf-8") as f:
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
        if verbose:
            print(f"  Loaded {len(candidates):,} candidates")

    elif sample_path and Path(sample_path).exists():
        if verbose:
            print(f"  Falling back to sample: {sample_path}")
        with open(sample_path) as f:
            candidates = json.load(f)
        if max_n:
            candidates = candidates[:max_n]
        if verbose:
            print(f"  Loaded {len(candidates)} candidates (sample)")
    else:
        raise FileNotFoundError(
            f"No candidate data found at {jsonl_path} or {sample_path}"
        )

    return candidates

# ─────────────────────────────────────────────────────────────────
# SKILL VOCABULARY LOADER
# ─────────────────────────────────────────────────────────────────
def load_skill_vocab(vocab_path: Path) -> dict:
    """
    Load skill vocabulary produced by eda_01_corpus_analysis.py.
    Returns dict: skill_name → {doc_freq, idf, tier, jd_label, ...}
    """
    if not Path(vocab_path).exists():
        print(f"  WARNING: skill_vocabulary.json not found at {vocab_path}")
        print(f"  Run eda_01_corpus_analysis.py first to generate it.")
        return {}
    return load_json(vocab_path)

def get_skill_idf(skill_name: str, vocab: dict, fallback_idf: float = 2.0) -> float:
    """
    IDF for a skill. Falls back to fallback_idf (≈ generic tier value)
    if skill not in vocabulary — handles OOV skills gracefully.
    """
    entry = vocab.get(skill_name) or vocab.get(skill_name.lower())
    if entry:
        return entry.get("idf", fallback_idf)
    return fallback_idf

def get_skill_jd_label(skill_name: str, vocab: dict) -> str:
    """
    JD polarity label for a skill.
    Returns: 'JD_HARD_REQ', 'JD_POSITIVE', 'JD_NEGATIVE', or 'NEUTRAL'
    """
    entry = vocab.get(skill_name) or vocab.get(skill_name.lower())
    if entry:
        return entry.get("jd_label", "NEUTRAL")
    return "NEUTRAL"

# ─────────────────────────────────────────────────────────────────
# CONSULTING FIRM CHECKER
# ─────────────────────────────────────────────────────────────────
def is_consulting_company(company_name: str, firm_set: set) -> bool:
    """Check if a company name substring-matches any consulting firm."""
    low = (company_name or "").lower()
    return any(cf in low for cf in firm_set)

# ─────────────────────────────────────────────────────────────────
# DISPLAY HELPERS
# ─────────────────────────────────────────────────────────────────
def pct(n: int, total: int) -> str:
    return f"{n/total*100:.1f}%" if total else "0.0%"

def bar(n: float, max_n: float, width: int = 20) -> str:
    filled = round((n / max_n) * width) if max_n else 0
    return "█" * filled + "░" * (width - filled)

def stats_dict(lst: list, label: str) -> dict:
    """Return basic stats as a dict — useful for logging."""
    lst = [x for x in lst if x is not None]
    if not lst:
        return {"label": label, "n": 0}
    s = sorted(lst)
    return {
        "label":  label,
        "n":      len(lst),
        "mean":   round(sum(lst) / len(lst), 3),
        "median": s[len(s) // 2],
        "min":    min(lst),
        "max":    max(lst),
    }

def cosine_similarity_vectors(a: list, b: list) -> float:
    """
    Simple cosine similarity between two equal-length vectors.
    Used for distribution comparison without importing sklearn.
    """
    if len(a) != len(b):
        return 0.0
    dot   = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(y * y for y in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)
