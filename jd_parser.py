"""
jd_parser.py
============
Parses the job description into four signed sections,
builds lookup sets for scoring, and produces the JD
embedding input text.

This is the ONLY file that reads the JD document.
All other files consume the parsed output dict.

Output dict structure:
  {
    "sections": {
      "HARD_REQUIREMENT": {"phrases": [...], "weight": 1.0, "polarity": "positive"},
      "IDEAL_PROFILE":    {...},
      "NICE_TO_HAVE":     {...},
      "EXPLICIT_NEGATIVE":{...},
    },
    "positive_terms": set(),   # fast lookup for any positive phrase
    "negative_terms": set(),   # fast lookup for any negative phrase
    "hard_req_terms":  set(),  # subset of positive — highest weight
    "embedding_text":  str,    # text used to create JD embedding
    "disqualifier_firms": set(),  # from negative section
  }
"""

import re
from pathlib import Path
from constants import JD_SECTIONS
from config import CONSULTING_FIRMS
from utils import normalize_text


# ─────────────────────────────────────────────────────────────────
# JD EMBEDDING TEXT
# This is the text fed into MiniLM to create the JD query vector.
# Constructed from the HARD_REQUIREMENT and IDEAL_PROFILE sections
# only — we embed what we WANT, not what we don't want.
# Negative section terms are deliberately excluded from embedding.
# ─────────────────────────────────────────────────────────────────
JD_EMBEDDING_TEXT = """
Senior AI Engineer at an AI-native startup building a recruiting intelligence platform.
Production experience with embeddings-based retrieval systems using sentence-transformers,
OpenAI embeddings, BGE or E5 models deployed to real users.
Handled embedding drift, index refresh, retrieval quality regression in production.
Production experience with vector databases: Pinecone, Weaviate, Qdrant, Milvus,
OpenSearch, Elasticsearch, FAISS. Operational experience at scale.
Strong Python, code quality, production systems engineering.
Designed evaluation frameworks for ranking systems: NDCG, MRR, MAP,
offline-to-online correlation, A/B test interpretation.
Applied ML and AI roles at product companies, not consulting firms.
End-to-end ranking search recommendation system shipped to real users at meaningful scale.
Strong opinions about hybrid versus dense retrieval, offline versus online evaluation,
when to fine-tune versus prompt LLMs, grounded in systems actually built.
6 to 8 years total experience, 4 to 5 years in applied ML AI roles.
Located in or willing to relocate to Noida Pune Hyderabad Bangalore Mumbai Delhi NCR.
""".strip()


def parse_jd() -> dict:
    """
    Build the structured JD representation from constants.JD_SECTIONS.

    Returns a dict with fast-lookup sets and the embedding text.
    This function is deterministic — no file I/O, no model calls.
    """
    positive_terms  = set()
    negative_terms  = set()
    hard_req_terms  = set()

    for section_name, section_data in JD_SECTIONS.items():
        for phrase in section_data["phrases"]:
            normalized = normalize_text(phrase)
            # Use first meaningful token as the fast-lookup key
            tokens = normalized.split()
            key    = tokens[0] if tokens else normalized

            if section_data["polarity"] == "positive":
                positive_terms.add(key)
                positive_terms.add(normalized)   # also add full phrase
                if section_name == "HARD_REQUIREMENT":
                    hard_req_terms.add(key)
                    hard_req_terms.add(normalized)
            else:
                negative_terms.add(key)
                negative_terms.add(normalized)

    # Extract consulting firm names from negative section
    disqualifier_firms = set(CONSULTING_FIRMS)

    return {
        "sections":          JD_SECTIONS,
        "positive_terms":    positive_terms,
        "negative_terms":    negative_terms,
        "hard_req_terms":    hard_req_terms,
        "embedding_text":    JD_EMBEDDING_TEXT,
        "disqualifier_firms": disqualifier_firms,
    }


def score_text_against_jd(text: str, jd: dict) -> dict:
    """
    Score a piece of text against the JD sections.
    Returns per-section match counts and signed total score.

    This is used in feature_extractor.py to score career text
    and summary text separately.

    Args:
        text: lowercased candidate text
        jd:   parsed JD dict from parse_jd()

    Returns:
        {
          "hard_req_matches":    int,
          "ideal_matches":       int,
          "nice_to_have_matches":int,
          "negative_matches":    int,
          "signed_score":        float,  # weighted sum with polarity
          "matched_hard_terms":  list,
          "matched_neg_terms":   list,
        }
    """
    text_low = text.lower()

    hard_req_matches    = 0
    ideal_matches       = 0
    nice_matches        = 0
    negative_matches    = 0
    matched_hard        = []
    matched_neg         = []

    for section_name, section_data in jd["sections"].items():
        weight   = section_data["weight"]
        polarity = section_data["polarity"]

        for phrase in section_data["phrases"]:
            phrase_low = phrase.lower()
            if phrase_low in text_low:
                if section_name == "HARD_REQUIREMENT":
                    hard_req_matches += 1
                    matched_hard.append(phrase)
                elif section_name == "IDEAL_PROFILE":
                    ideal_matches += 1
                elif section_name == "NICE_TO_HAVE":
                    nice_matches += 1
                elif section_name == "EXPLICIT_NEGATIVE":
                    negative_matches += 1
                    matched_neg.append(phrase)

    # Signed score: positive sections add, negative section subtracts
    signed_score = (
        hard_req_matches  * JD_SECTIONS["HARD_REQUIREMENT"]["weight"] +
        ideal_matches     * JD_SECTIONS["IDEAL_PROFILE"]["weight"] +
        nice_matches      * JD_SECTIONS["NICE_TO_HAVE"]["weight"] +
        negative_matches  * JD_SECTIONS["EXPLICIT_NEGATIVE"]["weight"]  # weight is -1.0
    )

    return {
        "hard_req_matches":     hard_req_matches,
        "ideal_matches":        ideal_matches,
        "nice_to_have_matches": nice_matches,
        "negative_matches":     negative_matches,
        "signed_score":         signed_score,
        "matched_hard_terms":   matched_hard,
        "matched_neg_terms":    matched_neg,
    }


def get_jd_context_for_reasoning(jd: dict) -> str:
    """
    Compact JD summary for conditioning reasoning generation.
    Used by reasoning.py to ground explanation strings.
    """
    hard_req = jd["sections"]["HARD_REQUIREMENT"]["phrases"][:8]
    ideal    = jd["sections"]["IDEAL_PROFILE"]["phrases"][:5]
    neg      = jd["sections"]["EXPLICIT_NEGATIVE"]["phrases"][:5]

    lines = [
        "Role: Senior AI Engineer, AI-native startup, Pune/Noida.",
        "Must have: " + ", ".join(hard_req[:5]) + ".",
        "Ideal: "    + ", ".join(ideal[:3]) + ".",
        "Reject if: "+ ", ".join(neg[:3]) + ".",
    ]
    return " ".join(lines)


if __name__ == "__main__":
    jd = parse_jd()
    print(f"JD parsed successfully.")
    print(f"  Positive terms (fast lookup): {len(jd['positive_terms'])}")
    print(f"  Hard req terms:               {len(jd['hard_req_terms'])}")
    print(f"  Negative terms:               {len(jd['negative_terms'])}")
    print(f"  Disqualifier firms:           {len(jd['disqualifier_firms'])}")
    print(f"\n  Embedding text ({len(jd['embedding_text'].split())} tokens):")
    print(f"  {jd['embedding_text'][:200]} ...")

    # test scoring
    test_text = """
    built faiss-based retrieval pipeline deployed to 5M users.
    handled embedding drift and index refresh weekly.
    evaluated with ndcg and mrr metrics.
    worked at tcs for 3 years before moving to product company.
    """
    result = score_text_against_jd(test_text, jd)
    print(f"\n  Test scoring on sample text:")
    for k, v in result.items():
        print(f"    {k}: {v}")
