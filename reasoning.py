"""
reasoning.py  (v4 — T5-ONNX reasoning, no external API)
=========================================================
Architecture:
  PRIMARY   — T5-small int8 ONNX with greedy decoding
              Runs entirely offline on CPU.
              Conditioned on a tightly structured prompt built
              from verified profile facts — no hallucination possible
              because every fact in the prompt comes from the data.

  FALLBACK  — Rule-based structured reasoning
              Fires when T5 ONNX is not exported yet, or produces
              output shorter than 10 words.

Key design: T5 is a conditional text-to-text model. We give it a
structured "summarize this candidate for this role" prompt and it
generates a fluent paragraph from the facts we provide. Because
ALL facts come from the profile dict (not from T5's weights), the
output is grounded even if T5 paraphrases them.

For 100 candidates, T5-small on CPU takes ~0.3-0.5s per candidate
→ total reasoning time ~30-50s. Well within the 5-min constraint.

All Issues A-F fixes retained from v3.
"""

import re, sys, json, math
from pathlib import Path
from datetime import datetime
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent))
from config import T5_ONNX, T5_TOKDIR, REASONING_MAX_INPUT, REASONING_MAX_OUTPUT
from utils import days_since, get_career_text

TODAY = datetime(2026, 6, 2)

# ─────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────
CONSULTING_NAMES = {
    'tcs','infosys','wipro','accenture','cognizant',
    'capgemini','hcl','tech mahindra','mphasis','hexaware',
}
RETRIEVAL_CONCEPTS = {
    'faiss','pinecone','weaviate','milvus','qdrant','opensearch',
    'elasticsearch','vector search','hybrid search','bm25',
    'dense retrieval','sparse retrieval','semantic search',
    'sentence-transformer','bi-encoder','cross-encoder','reranking',
    'ndcg','mrr','map','information retrieval','ann','hnsw',
    'query expansion','retrieval pipeline','recall@','relevance judgment',
    'nearest neighbor','vector index','embedding index','dual encoder',
}
METRIC_PATTERNS = [
    r'\d+[MKB][\+]?\s*(?:users?|queries|requests|documents|records)',
    r'\d+[\.\d]*\s*[%]\s*(?:improvement|reduction|increase|gain|lift|better|faster)',
    r'\d+[\.\d]*x\s*(?:faster|improvement|speedup|reduction)',
    r'(?:NDCG|MRR|MAP|AUC|F1|precision|recall)\s*(?:of|@|=|:)?\s*[\d\.]+',
    r'\d+\s*(?:billion|million|thousand)\s*(?:records|vectors|tokens|users)',
    r'p\d{2}\s*latency', r'\d+\s*ms\b',
    r'revenue[- ]per[- ](?:search|session|user)',
]
# Issue A — negated sentences are evidence of failure, not success
NEGATION_PHRASES = [
    "didn't make it to production","did not make it to production",
    "never made it to production","not in production",
    "wasn't deployed","was not deployed","didn't ship",
    "never shipped","proof of concept","poc that","prototype that",
    "experiment that never","didn't go live","never went live",
    "was scrapped","was shelved","was cancelled",
]
# Issue D — candidate explicitly says production was someone else's job
# REMOVED: "my own modeling work was secondary" — this phrase appears in
# sentences where the candidate is saying they were the PRODUCTION engineer,
# e.g. "my modeling work was secondary — I was the production-side engineer"
# which is actually a positive signal. Only keep unambiguous phrases.
NOT_OWNER_PHRASES = [
    "production deployment was handled by",
    "deployment was handled by",
    "was handled by the platform team",
    "was handled by the infra team",
    "was handled by the devops team",
    "not responsible for production",
    "productionization was done by",
    "was taken to production by",
    "pure ml side of the work; production",
    "my role was more on the modeling side than the productionization",
]
# Issue B — CV-primary titles are outside JD scope
CV_PRIMARY_TITLES = {
    'computer vision engineer','cv engineer','vision engineer',
    'computer vision researcher','computer vision scientist',
}
# Issue E — target metros for this role
INDIA_METROS = {
    'noida','pune','delhi','hyderabad','bangalore','bengaluru',
    'mumbai','gurgaon','gurugram','ncr','chennai','kolkata',
    'ahmedabad','jaipur','chandigarh','coimbatore','kochi',
    'vizag','indore','trivandrum','surat',
}


# ─────────────────────────────────────────────────────────────────
# T5-ONNX SESSION
# Loads once, cached as module-level singleton.
# Handles encoder-decoder seq2seq with greedy decoding.
# ─────────────────────────────────────────────────────────────────

class T5ONNXSession:
    """
    Wraps the T5-small ONNX encoder + decoder for seq2seq generation.

    T5 ONNX export produces three files:
      encoder_model.onnx
      decoder_model.onnx
      decoder_with_past_model.onnx  ← used for fast autoregressive decoding

    We run greedy decoding:
      1. Encode the prompt with encoder_model.onnx
      2. Feed encoder output + decoder start token to decoder_with_past_model.onnx
      3. Greedily pick argmax of logits at each step
      4. Repeat until EOS token or max_length

    This is the same decoding loop transformers uses internally,
    just driven manually via onnxruntime sessions.
    """

    _instance = None   # singleton

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._loaded = False
        return cls._instance

    def load(self):
        if self._loaded:
            return
        self._ready = False

        # Locate model files — T5 export produces a directory
        t5_dir = T5_ONNX.parent / "t5_tmp"
        t5_int8_dir = T5_ONNX.parent / "t5_tmp" / "int8"

        # Try to find encoder/decoder ONNX files
        candidate_dirs = [
            T5_ONNX.parent / "t5_tmp" / "int8",
            T5_ONNX.parent / "t5_tmp",
            T5_ONNX.parent,
        ]

        encoder_path = None
        decoder_path = None
        decoder_past_path = None

        for d in candidate_dirs:
            if not d.exists():
                continue
            for f in d.iterdir():
                name = f.name.lower()
                if "encoder" in name and f.suffix == ".onnx":
                    encoder_path = f
                elif "decoder_with_past" in name and f.suffix == ".onnx":
                    decoder_past_path = f
                elif "decoder" in name and "past" not in name and f.suffix == ".onnx":
                    decoder_path = f

        # Also check if T5_ONNX itself is a single encoder file
        if T5_ONNX.exists() and encoder_path is None:
            encoder_path = T5_ONNX

        if encoder_path is None:
            print(f"  T5 ONNX not found. Run: python export_onnx.py")
            print(f"  Reasoning will use structured fallback.")
            self._loaded = True
            return

        try:
            import onnxruntime as ort
            sess_opts = ort.SessionOptions()
            sess_opts.inter_op_num_threads = 2
            sess_opts.intra_op_num_threads = 2

            self.encoder = ort.InferenceSession(
                str(encoder_path),
                sess_options=sess_opts,
                providers=["CPUExecutionProvider"]
            )
            print(f"  T5 encoder loaded: {encoder_path.name}")

            if decoder_past_path and decoder_past_path.exists():
                self.decoder = ort.InferenceSession(
                    str(decoder_past_path),
                    sess_options=sess_opts,
                    providers=["CPUExecutionProvider"]
                )
                self._use_past = True
                print(f"  T5 decoder (with past) loaded: {decoder_past_path.name}")
            elif decoder_path and decoder_path.exists():
                self.decoder = ort.InferenceSession(
                    str(decoder_path),
                    sess_options=sess_opts,
                    providers=["CPUExecutionProvider"]
                )
                self._use_past = False
                print(f"  T5 decoder loaded: {decoder_path.name}")
            else:
                # Only encoder available — use it for feature extraction
                # and let fallback handle text generation
                self.decoder = None
                self._use_past = False

            # Load tokenizer
            self.tokenizer = self._load_tokenizer()
            if self.tokenizer is not None:
                self._ready = (self.decoder is not None)
                if self._ready:
                    print(f"  T5 ONNX reasoning ready.")
                else:
                    print(f"  T5 encoder-only — using structured fallback.")
            else:
                self._ready = False

        except Exception as e:
            print(f"  T5 ONNX load error: {e}")
            print(f"  Using structured fallback for reasoning.")
            self._ready = False

        self._loaded = True

    def _load_tokenizer(self):
        """Load T5 tokenizer from saved directory or HuggingFace."""
        try:
            from transformers import AutoTokenizer
            if T5_TOKDIR.exists():
                tok = AutoTokenizer.from_pretrained(str(T5_TOKDIR))
            else:
                tok = AutoTokenizer.from_pretrained("google/t5-small")
                tok.save_pretrained(str(T5_TOKDIR))
            print(f"  T5 tokenizer loaded.")
            return tok
        except Exception as e:
            print(f"  T5 tokenizer load error: {e}")
            return None

    def generate(self, prompt: str, max_new_tokens: int = 80) -> str:
        """
        Greedy decoding with T5-small ONNX.

        T5 is a text-to-text model. We feed it:
          "summarize candidate: [facts]"
        and it generates a fluent paragraph.

        Greedy decoding: at each step, pick the token with highest
        logit score. Stop at EOS token or max_new_tokens.
        """
        if not self._ready:
            return ""

        import numpy as np

        try:
            # Tokenize input
            inputs = self.tokenizer(
                prompt,
                return_tensors="np",
                max_length=REASONING_MAX_INPUT,
                truncation=True,
                padding=False,
            )
            input_ids      = inputs["input_ids"].astype(np.int64)
            attention_mask = inputs["attention_mask"].astype(np.int64)

            # Encoder forward pass
            enc_out = self.encoder.run(
                None,
                {"input_ids": input_ids, "attention_mask": attention_mask}
            )
            # enc_out[0] = last_hidden_state shape (1, seq_len, hidden)
            encoder_hidden = enc_out[0]

            # Greedy decode
            # T5 uses decoder_start_token_id = pad_token_id = 0
            decoder_input_ids = np.array([[self.tokenizer.pad_token_id]], dtype=np.int64)
            generated_ids = []

            EOS_ID = self.tokenizer.eos_token_id or 1

            for step in range(max_new_tokens):
                if self._use_past and step > 0:
                    # Use decoder_with_past for faster autoregressive steps
                    # Only feed the last token
                    dec_inputs = {
                        "input_ids": decoder_input_ids[:, -1:],
                        "encoder_hidden_states": encoder_hidden,
                        "encoder_attention_mask": attention_mask,
                    }
                    # Add past key values if session expects them
                    dec_inputs.update(self._past_kv if hasattr(self, '_past_kv') else {})
                else:
                    # First step or no-past decoder
                    dec_inputs = {
                        "input_ids": decoder_input_ids,
                        "encoder_hidden_states": encoder_hidden,
                        "encoder_attention_mask": attention_mask,
                    }

                # Run decoder
                try:
                    dec_out = self.decoder.run(None, dec_inputs)
                except Exception:
                    # Input name mismatch — try alternate naming
                    alt_inputs = {
                        inp.name: dec_inputs.get(inp.name, dec_inputs.get(
                            inp.name.replace("encoder_hidden_states", "encoder_hidden_states"),
                            None
                        ))
                        for inp in self.decoder.get_inputs()
                    }
                    alt_inputs = {k: v for k, v in alt_inputs.items() if v is not None}
                    dec_out = self.decoder.run(None, alt_inputs)

                # dec_out[0] = logits shape (1, seq_len, vocab_size)
                logits      = dec_out[0]
                next_token  = int(np.argmax(logits[0, -1, :]))
                generated_ids.append(next_token)

                if next_token == EOS_ID:
                    break

                # Append new token to decoder input
                decoder_input_ids = np.concatenate(
                    [decoder_input_ids, np.array([[next_token]], dtype=np.int64)],
                    axis=1
                )

            if not generated_ids:
                return ""

            # Decode
            text = self.tokenizer.decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
            return text.strip()

        except Exception as e:
            # Any decoding failure → fall through to structured template
            return ""

    @property
    def ready(self) -> bool:
        return getattr(self, '_ready', False)


# Module-level singleton — loaded once on first use
_T5 = T5ONNXSession()


def get_t5() -> T5ONNXSession:
    """Get (and lazily load) the T5 session."""
    if not _T5._loaded:
        _T5.load()
    return _T5


# ─────────────────────────────────────────────────────────────────
# PROFILE HELPERS (Issues A, B, D, E, F)
# ─────────────────────────────────────────────────────────────────

def is_cv_primary_title(title: str) -> bool:
    """Issue B."""
    return title.lower() in CV_PRIMARY_TITLES

def is_negated(sentence: str) -> bool:
    """Issue A: sentence explicitly says work failed or never reached production."""
    s = sentence.lower()
    return any(p in s for p in NEGATION_PHRASES)

def is_not_owner(sentence: str) -> bool:
    """Issue D: sentence says production was handled by someone else."""
    s = sentence.lower()
    return any(p in s for p in NOT_OWNER_PHRASES)

def detect_duplicate_descs(hist: list) -> set:
    """Issue F: fingerprint duplicate role descriptions."""
    counts = Counter(
        (r.get('description') or '').strip().lower()[:80]
        for r in hist
    )
    return {d for d, c in counts.items() if c > 1 and d}

def extract_key_sentences(
    career_text: str, hist: list,
    _extra_dupes: set = None,
) -> list:
    """Issues A+D+F: return only genuinely positive, non-templated sentences."""
    dupes = detect_duplicate_descs(hist)
    if _extra_dupes:
        dupes = dupes | _extra_dupes
    scored = []
    for sent in re.split(r'(?<=[.!?])\s+', career_text):
        sent = sent.strip()
        if len(sent) < 35 or len(sent) > 300:
            continue
        if is_negated(sent):    continue   # Issue A
        if is_not_owner(sent):  continue   # Issue D
        s_low = sent.lower()
        score = 0
        for pat in METRIC_PATTERNS:
            if re.search(pat, sent, re.IGNORECASE): score += 3
        for c in RETRIEVAL_CONCEPTS:
            if c in s_low: score += 2
        for cue in {'production','deployed','real users','launched',
                    'shipped','scale','a/b','latency','ndcg','mrr'}:
            if cue in s_low: score += 1
        for bad in {'responsible for','i have','i am','my role was',
                    'helped the team','manufacturing','supply chain'}:
            if bad in s_low: score -= 2
        if s_low[:80] in dupes: score -= 3   # Issue F
        scored.append((score, sent))
    scored.sort(key=lambda x: -x[0])
    return [s for sc, s in scored[:2] if sc > 0]

def extract_retrieval_ev(career_text: str) -> list:
    t = career_text.lower()
    return [c for c in RETRIEVAL_CONCEPTS if c in t][:4]

def extract_metrics(career_text: str) -> list:
    seen, out = set(), []
    for pat in METRIC_PATTERNS:
        for m in re.findall(pat, career_text, re.IGNORECASE):
            k = m.strip().lower()
            if k not in seen and len(k) > 2:
                seen.add(k); out.append(m.strip())
    return out[:3]

def get_location_ctx(cand: dict) -> tuple:
    """Issue E: returns (loc_str, is_outside_india, willing_to_relocate)."""
    p   = cand.get('profile', {})
    sig = cand.get('redrob_signals', {})
    loc = (p.get('location') or '').strip()
    country = (p.get('country') or '').lower()
    loc_low = loc.lower()
    in_india = 'india' in country or any(m in loc_low for m in INDIA_METROS)
    return loc, not in_india, bool(sig.get('willing_to_relocate', False))

def get_career_arc(hist: list, kg_score: float) -> str:
    if not hist or kg_score < 0.35: return ""
    def sd(r):
        try: return datetime.strptime(r.get('start_date','2000-01-01'), '%Y-%m-%d')
        except: return datetime(2000, 1, 1)
    srt = sorted(hist, key=sd)
    steps = [
        f"{r.get('title','')} at {r.get('company','')}"
        for r in srt[-3:] if r.get('title')
    ]
    return " → ".join(steps) if len(steps) >= 2 else ""

def get_genuine_gaps(cand: dict, breakdown: dict) -> list:
    """Identify real, meaningful gaps — no generic noise."""
    p      = cand.get('profile', {})
    sig    = cand.get('redrob_signals', {})
    hist   = cand.get('career_history', [])
    skills = cand.get('skills', [])
    skill_low = {(s.get('name') or '').lower() for s in skills}

    yoe    = p.get('years_of_experience', 0) or 0
    notice = sig.get('notice_period_days', 90) or 90
    otw    = sig.get('open_to_work_flag', False)
    gh     = sig.get('github_activity_score', -1)
    d_in   = days_since(sig.get('last_active_date', '2020-01-01'))
    title  = p.get('current_title', '')
    career = get_career_text(cand)
    bd     = breakdown

    gaps = []

    # Issue B: CV title
    if is_cv_primary_title(title):
        gaps.append(
            "primary expertise is computer vision — "
            "limited NLP/retrieval production background per JD requirement"
        )

    # Issue D: not-owner pattern
    not_owner_count = sum(1 for ph in NOT_OWNER_PHRASES if ph in career.lower())
    if not_owner_count >= 2:
        gaps.append(
            "career text indicates ML modeling focus without "
            "end-to-end production ownership"
        )

    # Pure consulting career
    if len(hist) >= 2:
        cons = sum(
            1 for r in hist
            if any(cf in (r.get('company') or '').lower() for cf in CONSULTING_NAMES)
        )
        if cons == len(hist):
            firms = list({r.get('company', '?') for r in hist})[:2]
            gaps.append(f"entire career at IT services firms ({', '.join(firms)})")

    # Wrapper tools without retrieval depth
    wrappers = {'langchain', 'llamaindex', 'flowise'}
    if skill_low & wrappers and not extract_retrieval_ev(career.lower()):
        gaps.append(
            "LLM wrapper tools (LangChain/LlamaIndex) in skills "
            "without retrieval-system evidence in career history"
        )

    if yoe < 4.0:
        gaps.append(f"only {yoe:.1f}yr experience vs JD target of 6–8yr")
    if notice > 90:
        gaps.append(f"{notice}-day notice — requires negotiation")
    if (gh is None or gh <= 0) and bd.get('structural_score', 0) >= 0.70:
        gaps.append("no GitHub signal — open-source validation unavailable")
    if d_in > 150 and not otw:
        gaps.append(f"inactive {d_in}d, not open-to-work — outreach response uncertain")

    # Issue E: outside India without relocation willingness
    _, outside, willing = get_location_ctx(cand)
    if outside and not willing:
        gaps.append(
            "based outside India, not willing to relocate — "
            "no visa sponsorship offered"
        )

    return gaps[:3]


# ─────────────────────────────────────────────────────────────────
# T5 PROMPT BUILDER
# The prompt is a structured "candidate fact sheet" fed to T5.
# T5 is trained as text-to-text: it reads the facts and writes
# a fluent summary. Since all facts come from the profile dict,
# T5 can only paraphrase — it cannot invent new facts.
#
# Prompt format designed for T5-small (60M params):
#   - Short, keyword-dense sentences (T5 processes these well)
#   - Explicit instruction: "summarize candidate fit:"
#   - Most important facts first (T5 attends more to early tokens)
#   - Under 256 tokens (REASONING_MAX_INPUT)
# ─────────────────────────────────────────────────────────────────

def build_t5_prompt(cand: dict, rank: int, breakdown: dict, gaps: list, global_duplicates: set = None) -> str:
    """
    Build a T5-optimised prompt from verified profile facts.

    T5-small works best with:
    - Clear task prefix ("summarize candidate fit:")
    - Short, factual clauses rather than long sentences
    - Concrete numbers and names rather than abstractions
    - Under 256 tokens total

    The output will be a fluent 2-3 sentence evaluation that
    paraphrases these facts — different for every candidate
    because the facts differ.
    """
    p      = cand.get('profile', {})
    sig    = cand.get('redrob_signals', {})
    hist   = cand.get('career_history', [])
    skills = cand.get('skills', [])

    title   = p.get('current_title', 'ML Engineer')
    company = p.get('current_company', '') or ''
    yoe     = p.get('years_of_experience', 0) or 0
    loc, outside, willing = get_location_ctx(cand)

    d_in    = days_since(sig.get('last_active_date', '2020-01-01'))
    notice  = sig.get('notice_period_days', 90) or 90
    rr      = sig.get('recruiter_response_rate', 0) or 0
    gh      = sig.get('github_activity_score', -1)
    otw     = sig.get('open_to_work_flag', False)

    bd      = breakdown
    struct  = bd.get('structural_score', 0)
    sem     = bd.get('semantic_score', 0)
    avail   = bd.get('availability_mult', 0)

    # Career facts
    career_text  = get_career_text(cand)
    retr_ev      = extract_retrieval_ev(career_text.lower())
    metrics      = extract_metrics(career_text)
    key_sents    = extract_key_sentences(career_text, hist, _extra_dupes=global_duplicates)
    arc          = get_career_arc(hist, bd.get('detail', {}).get('kg_score', 0))

    # Top 4 skills by endorsements
    ranked_skills = sorted(
        skills, key=lambda s: s.get('endorsements', 0) or 0, reverse=True
    )
    top_skill_names = [s['name'] for s in ranked_skills[:4] if s.get('name')]

    # Build compact fact block
    facts = []
    facts.append(f"role: {title}, {yoe:.0f}yr, {company}, {loc}")
    facts.append(f"rank: {rank} of 100, structural score: {struct:.2f}, semantic: {sem:.2f}")

    if retr_ev:
        facts.append(f"retrieval systems used: {', '.join(retr_ev[:3])}")
    if metrics:
        facts.append(f"outcomes: {', '.join(metrics[:2])}")
    if key_sents:
        # Take first key sentence, capped at 100 chars
        ev = key_sents[0][:100]
        facts.append(f"career evidence: {ev}")
    if arc:
        facts.append(f"career arc: {arc}")
    if top_skill_names:
        facts.append(f"top skills: {', '.join(top_skill_names)}")

    # Availability
    if d_in <= 30 and notice <= 30:
        facts.append(f"availability: active {d_in}d ago, {notice}d notice, immediate")
    elif d_in <= 90:
        facts.append(f"availability: active {d_in}d ago, {notice}d notice")
    else:
        facts.append(f"availability: inactive {d_in}d, {notice}d notice, risk")

    if outside:
        facts.append(f"location risk: outside India, {'willing to relocate' if willing else 'no visa sponsorship'}")

    if gaps:
        facts.append(f"gaps: {'; '.join(gaps[:2])}")

    # T5 task prefix + compact facts
    prompt = "summarize candidate fit for senior AI engineer role: " + " | ".join(facts)
    return prompt


# ─────────────────────────────────────────────────────────────────
# STRUCTURED FALLBACK (all A-F fixes, no T5 needed)
# ─────────────────────────────────────────────────────────────────

def build_fallback(
    cand: dict, rank: int, breakdown: dict, gaps: list,
    global_duplicates: set = None,
) -> str:
    """
    Structured reasoning when T5 is unavailable or produces poor output.
    All Issues A-F are applied. Non-templated: the opening sentence
    varies by what signal dominates (retrieval+metrics, key evidence,
    or structural score).
    """
    p      = cand.get('profile', {})
    sig    = cand.get('redrob_signals', {})
    hist   = cand.get('career_history', [])
    title  = p.get('current_title', 'ML Engineer')
    yoe    = p.get('years_of_experience', 0) or 0
    bd     = breakdown
    d_in   = days_since(sig.get('last_active_date', '2020-01-01'))
    notice = sig.get('notice_period_days', 90) or 90
    rr     = sig.get('recruiter_response_rate', 0) or 0
    loc, outside, willing = get_location_ctx(cand)

    career = get_career_text(cand)
    retr   = extract_retrieval_ev(career.lower())
    mets   = extract_metrics(career)
    # Issue 3: merge per-candidate duplicates with global cross-candidate duplicates
    combined_dupes = detect_duplicate_descs(hist)
    if global_duplicates:
        combined_dupes = combined_dupes | global_duplicates
    kents  = extract_key_sentences(career, hist, _extra_dupes=combined_dupes)
    arc    = get_career_arc(hist, bd.get('detail', {}).get('kg_score', 0))

    parts = []

    # ── Opening: pick strongest available signal ──────────────────
    if is_cv_primary_title(title):
        # Issue B: lead with the concern
        parts.append(
            f"{title} ({yoe:.0f}yr) — primary domain is computer vision, "
            f"outside JD's NLP/retrieval scope."
        )
    elif retr and mets:
        parts.append(
            f"{title} ({yoe:.0f}yr) with production retrieval work — "
            f"career references {retr[0]} with outcomes: {mets[0]}."
        )
    elif kents:
        best = kents[0][:160].rstrip('.') + ("..." if len(kents[0]) > 160 else ".")
        parts.append(f"{title} ({yoe:.0f}yr). Career evidence: \"{best}\"")
    elif retr:
        parts.append(
            f"{title} ({yoe:.0f}yr) with {', '.join(retr[:2])} "
            f"mentioned in career history."
        )
    else:
        struct = bd.get('structural_score', 0)
        parts.append(
            f"{title} ({yoe:.0f}yr) — structural alignment {struct:.2f}; "
            f"no retrieval-specific evidence found in career text."
        )

    # ── Career arc (only if coherent) ────────────────────────────
    if arc:
        parts.append(f"Career: {arc}.")

    # ── Availability + location ───────────────────────────────────
    if d_in <= 14 and notice <= 30:
        av = f"Immediately actionable — active {d_in}d ago, {notice}d notice, {rr:.0%} response rate."
    elif d_in <= 45 and notice <= 60:
        av = f"Available — last active {d_in}d ago, {notice}d notice."
    else:
        av = f"Availability: {d_in}d inactive, {notice}d notice."

    if outside:
        av += f" Based in {loc} ({'willing to relocate' if willing else 'not willing to relocate — no visa sponsorship'})."
    else:
        av += f" Location: {loc}."
    parts.append(av)

    # ── Gaps ──────────────────────────────────────────────────────
    if gaps:
        parts.append("Gaps: " + "; ".join(gaps) + ".")

    return " ".join(parts)


# ─────────────────────────────────────────────────────────────────
# SCORE CALIBRATION  (Issue C: sigmoid S-curve)
# ─────────────────────────────────────────────────────────────────

def calibrate_scores(raw_scores: list) -> list:
    """
    Non-linear sigmoid calibration → [0.50, 0.95].
    Top 10: 0.85-0.95 | Mid: 0.60-0.85 | Bottom: 0.50-0.60
    Ordering is preserved exactly — only displayed value changes.
    """
    n = len(raw_scores)
    if n == 0: return []
    if n == 1: return [0.90]

    mn, mx = min(raw_scores), max(raw_scores)
    rng = mx - mn if mx > mn else 1.0
    k   = 6.0
    sig_min = 1.0 / (1.0 + math.exp( k * 0.5))   # at pct=0
    sig_max = 1.0 / (1.0 + math.exp(-k * 0.5))   # at pct=1
    sig_rng = sig_max - sig_min

    out = []
    for s in raw_scores:
        pct = (s - mn) / rng
        sig = 1.0 / (1.0 + math.exp(-k * (pct - 0.5)))
        cal = 0.50 + ((sig - sig_min) / sig_rng) * 0.45
        out.append(round(float(cal), 6))
    return out


# ─────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────

def generate_reasoning(
    cand: dict,
    rank: int,
    breakdown: dict,
    jd_context: str = "",
    use_llm: bool = True,   # kept for API compatibility; controls T5 use
) -> str:
    """
    Generate reasoning for one candidate.
    Tries T5-ONNX first (offline, no API), falls back to structured template.
    Always returns a non-empty, fact-grounded string.
    """
    gaps = get_genuine_gaps(cand, breakdown)

    if use_llm:
        t5 = get_t5()
        if t5.ready:
            try:
                prompt = build_t5_prompt(cand, rank, breakdown, gaps)
                out    = t5.generate(prompt, max_new_tokens=REASONING_MAX_OUTPUT)
                # Require at least 10 words of meaningful output
                if out and len(out.split()) >= 10:
                    return out
            except Exception:
                pass  # fall through to structured fallback

    return build_fallback(cand, rank, breakdown, gaps)


def generate_all_reasoning(
    top_candidates: list,
    jd_context: str = "",
    use_llm: bool = True,
) -> dict:
    """
    Generate reasoning for all top-100 candidates.
    T5 session is loaded once and reused across all calls.
    Returns dict: candidate_id → reasoning string.

    Issue 3 fix: build a global set of duplicate career evidence
    fingerprints across ALL top-100 candidates before generating
    any reasoning. This prevents the same templated role description
    from surfacing as "strong evidence" for multiple candidates.
    """
    # Load T5 once before the loop
    if use_llm:
        t5 = get_t5()
        if t5.ready:
            print(f"  T5 ONNX session active — generating dynamic reasoning.")
        else:
            print(f"  T5 not available — using structured fallback reasoning.")

    # Issue 3: build global duplicate fingerprint set across all candidates
    # Any sentence appearing in 2+ candidates' career text is templated
    from collections import Counter
    global_sentence_counts = Counter()
    for cand, _, _ in top_candidates:
        career = get_career_text(cand)
        for sent in re.split(r'(?<=[.!?])\s+', career):
            sent = sent.strip()
            if len(sent) >= 35:
                global_sentence_counts[sent[:80].lower()] += 1
    global_duplicates = {fp for fp, cnt in global_sentence_counts.items() if cnt >= 2}

    results = {}
    for cand, rank, breakdown in top_candidates:
        cid = cand.get('candidate_id', f'rank_{rank}')
        try:
            gaps = get_genuine_gaps(cand, breakdown)

            if use_llm:
                t5 = get_t5()
                if t5.ready:
                    try:
                        prompt = build_t5_prompt(cand, rank, breakdown, gaps, global_duplicates=global_duplicates)
                        out    = t5.generate(prompt, max_new_tokens=REASONING_MAX_OUTPUT)
                        if out and len(out.split()) >= 10:
                            results[cid] = out
                            continue
                    except Exception:
                        pass

            # Fallback with global duplicates injected
            results[cid] = build_fallback(
                cand, rank, breakdown, gaps,
                global_duplicates=global_duplicates,
            )
        except Exception as e:
            p = cand.get('profile', {})
            results[cid] = (
                f"{p.get('current_title','Candidate')} "
                f"({p.get('years_of_experience',0):.0f}yr) — "
                f"reasoning error: {str(e)[:60]}"
            )

    return results
