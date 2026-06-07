"""
embedder.py
===========
ONNX-based embedding inference using all-MiniLM-L6-v2.
No PyTorch. No transformers at inference time.
Only onnxruntime + tokenizers (lightweight).

Produces three types of embeddings per candidate:
  1. career_narrative  — full career text (primary FAISS chunk)
  2. role_level        — per-role embeddings (KG domain continuity)
  3. jd_embedding      — single JD query vector

All stored to disk for online ranking phase.

Run:
    python embedder.py
Produces:
    artifacts/faiss_index.bin       — FAISS flat index (100k × 384)
    artifacts/jd_embedding.npy      — JD query vector (384,)
    artifacts/role_embeddings.pkl   — dict: cand_id → list of role vecs
"""

import sys
import json
import numpy as np
import faiss
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    CANDIDATES_FILE, SAMPLE_FILE,
    FAISS_INDEX_FILE, JD_EMBEDDING_FILE, ROLE_EMBEDDINGS_FILE,
    MINILM_ONNX, MINILM_TOKDIR,
    EMBEDDING_DIM, EMBEDDING_BATCH_SIZE,
)
from utils import (
    load_candidates, save_pickle, load_pickle, get_career_narrative,
)
from jd_parser import parse_jd


# ─────────────────────────────────────────────────────────────────
# ONNX EMBEDDING SESSION
# Loads once, reused for all inference calls.
# ─────────────────────────────────────────────────────────────────

class ONNXEmbedder:
    """
    Thin wrapper around the ONNX MiniLM session.
    Handles tokenization and mean-pooling internally.
    Falls back to a simple TF-IDF bag-of-words vector if
    ONNX model is not yet exported (for development use).
    """

    def __init__(self, model_path: Path, tokenizer_dir: Path):
        self.model_path    = model_path
        self.tokenizer_dir = tokenizer_dir
        self.session       = None
        self.tokenizer     = None
        self._load()

    def _load(self):
        if not self.model_path.exists():
            print(f"  WARNING: ONNX model not found at {self.model_path}")
            print(f"  Run: python export_onnx.py  to export models first.")
            print(f"  Falling back to random unit vectors (dev mode only).")
            self._fallback = True
            return

        try:
            import onnxruntime as ort
            self.session = ort.InferenceSession(
                str(self.model_path),
                providers=["CPUExecutionProvider"],
            )
            print(f"  ONNX session loaded: {self.model_path.name}")

            # Load tokenizer
            if self.tokenizer_dir.exists():
                from tokenizers import Tokenizer
                self.tokenizer = Tokenizer.from_file(
                    str(self.tokenizer_dir / "tokenizer.json")
                )
                self.tokenizer.enable_padding(
                    pad_id=0, pad_token="[PAD]", length=128
                )
                self.tokenizer.enable_truncation(max_length=128)
                print(f"  Tokenizer loaded from {self.tokenizer_dir.name}/")
            else:
                # Fallback: use transformers tokenizer if available
                try:
                    from transformers import AutoTokenizer
                    self._hf_tokenizer = AutoTokenizer.from_pretrained(
                        "sentence-transformers/all-MiniLM-L6-v2"
                    )
                    print(f"  Using HuggingFace tokenizer (tokenizer dir not found).")
                    self._use_hf = True
                except Exception:
                    print(f"  WARNING: No tokenizer found. Using dev fallback.")
                    self._fallback = True
                    return

            self._fallback = False
            print(f"  Embedder ready (ONNX + tokenizers).")

        except Exception as e:
            print(f"  WARNING: Failed to load ONNX session: {e}")
            self._fallback = True

    def _tokenize_batch(self, texts: list[str]) -> dict:
        """Tokenize a batch of texts. Returns numpy arrays."""
        if hasattr(self, "_use_hf") and self._use_hf:
            enc = self._hf_tokenizer(
                texts, padding=True, truncation=True,
                max_length=128, return_tensors="np"
            )
            return {
                "input_ids":      enc["input_ids"].astype(np.int64),
                "attention_mask": enc["attention_mask"].astype(np.int64),
            }

        encodings    = self.tokenizer.encode_batch(texts)
        input_ids    = np.array([e.ids      for e in encodings], dtype=np.int64)
        attention_mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def _mean_pool(self, token_embeddings: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
        """
        Mean pooling — average token embeddings weighted by attention mask.
        Exactly what sentence-transformers uses for MiniLM.
        """
        mask_expanded = attention_mask[:, :, np.newaxis].astype(np.float32)
        sum_embeddings = (token_embeddings * mask_expanded).sum(axis=1)
        sum_mask       = mask_expanded.sum(axis=1).clip(min=1e-9)
        pooled         = sum_embeddings / sum_mask

        # L2 normalize — makes FAISS inner product == cosine similarity
        norms = np.linalg.norm(pooled, axis=1, keepdims=True).clip(min=1e-9)
        return pooled / norms

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """
        Embed a batch of texts.
        Returns float32 array of shape (len(texts), 384).
        """
        if self._fallback or self.session is None:
            # Dev fallback: deterministic pseudo-embeddings from text hash
            vecs = []
            for t in texts:
                np.random.seed(abs(hash(t[:50])) % (2**31))
                v = np.random.randn(EMBEDDING_DIM).astype(np.float32)
                v = v / np.linalg.norm(v)
                vecs.append(v)
            return np.stack(vecs)

        inputs = self._tokenize_batch(texts)
        input_names = [i.name for i in self.session.get_inputs()]

        feed = {}
        if "input_ids" in input_names:
            feed["input_ids"] = inputs["input_ids"]
        if "attention_mask" in input_names:
            feed["attention_mask"] = inputs["attention_mask"]
        if "token_type_ids" in input_names:
            feed["token_type_ids"] = np.zeros_like(inputs["input_ids"])

        outputs = self.session.run(None, feed)
        # outputs[0] = last_hidden_state or pooler_output
        token_embs = outputs[0]   # shape (B, seq_len, 384)

        if token_embs.ndim == 3:
            return self._mean_pool(token_embs, inputs["attention_mask"])
        else:
            # Already pooled (some ONNX exports give pooler_output)
            norms = np.linalg.norm(token_embs, axis=1, keepdims=True).clip(min=1e-9)
            return (token_embs / norms).astype(np.float32)

    def embed_texts(self, texts: list[str], batch_size: int = 512,
                    verbose: bool = False) -> np.ndarray:
        """
        Embed a list of texts in batches.
        Returns float32 array of shape (N, 384).
        """
        all_vecs = []
        batches  = [texts[i:i+batch_size] for i in range(0, len(texts), batch_size)]
        iterator = tqdm(batches, desc="  Embedding batches") if verbose else batches

        for batch in iterator:
            vecs = self.embed_batch(batch)
            all_vecs.append(vecs)

        return np.vstack(all_vecs).astype(np.float32)

    def embed_single(self, text: str) -> np.ndarray:
        """Embed a single text. Returns (384,) float32 array."""
        return self.embed_batch([text])[0]


# ─────────────────────────────────────────────────────────────────
# FAISS INDEX BUILDER
# ─────────────────────────────────────────────────────────────────

def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """
    Build a FAISS flat inner-product index.
    IndexFlatIP with L2-normalized vectors = cosine similarity.
    Exact (no approximation) — correct for 100k vectors on CPU.

    Timing: ~0.5s search over 100k 384-dim vectors.
    Memory: 100k × 384 × 4 bytes = ~147 MB.
    """
    N, D = embeddings.shape
    assert D == EMBEDDING_DIM, f"Expected {EMBEDDING_DIM}-dim, got {D}"

    # Ensure float32 and L2 normalized
    emb_f32 = embeddings.astype(np.float32)
    faiss.normalize_L2(emb_f32)

    index = faiss.IndexFlatIP(D)    # Inner product (= cosine after L2 norm)
    index.add(emb_f32)

    print(f"  FAISS index built: {index.ntotal:,} vectors, {D}-dim")
    return index


def save_faiss_index(index: faiss.Index, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(path))
    size_mb = path.stat().st_size / 1024 / 1024
    print(f"  FAISS index saved → {path}  ({size_mb:.1f} MB)")


def load_faiss_index(path: Path) -> faiss.Index:
    index = faiss.read_index(str(path))
    print(f"  FAISS index loaded: {index.ntotal:,} vectors")
    return index


# ─────────────────────────────────────────────────────────────────
# ROLE-LEVEL EMBEDDINGS
# One embedding per job role — used by KG layer for domain
# continuity scoring between adjacent roles.
# ─────────────────────────────────────────────────────────────────

def build_role_embeddings(
    candidates: list,
    embedder: ONNXEmbedder,
    batch_size: int = 512,
) -> dict:
    """
    Build role-level embeddings for all candidates.
    Returns dict: candidate_id → list of (384,) arrays (one per role).
    """
    # Flatten all roles into a single batch for efficiency
    all_role_texts = []
    role_index     = []   # (candidate_id, role_idx)

    for cand in candidates:
        cid  = cand.get("candidate_id", "")
        hist = cand.get("career_history", [])
        for j, role in enumerate(hist):
            title    = role.get("title", "") or ""
            company  = role.get("company", "") or ""
            industry = role.get("industry", "") or ""
            desc     = (role.get("description", "") or "")[:300]  # cap length
            text     = f"{title} at {company} ({industry}). {desc}".strip()
            all_role_texts.append(text)
            role_index.append((cid, j))

    print(f"  Embedding {len(all_role_texts):,} role texts ...")
    all_vecs = embedder.embed_texts(
        all_role_texts, batch_size=batch_size, verbose=True
    )

    # Reconstruct per-candidate lists
    role_embeddings = {}
    for (cid, _), vec in zip(role_index, all_vecs):
        if cid not in role_embeddings:
            role_embeddings[cid] = []
        role_embeddings[cid].append(vec)

    print(f"  Role embeddings built for {len(role_embeddings):,} candidates")
    return role_embeddings


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 60)
    print("  Embedder — ONNX MiniLM → FAISS index + role embeddings")
    print("=" * 60 + "\n")

    # Load embedder
    embedder = ONNXEmbedder(MINILM_ONNX, MINILM_TOKDIR)

    # Load candidates
    candidates = load_candidates(CANDIDATES_FILE, SAMPLE_FILE, verbose=True)
    N          = len(candidates)

    # ── 1. JD embedding (single vector, query for FAISS)
    print(f"\n  [1/3] Embedding JD ...")
    jd          = parse_jd()
    jd_vec      = embedder.embed_single(jd["embedding_text"])
    np.save(str(JD_EMBEDDING_FILE), jd_vec)
    print(f"  JD embedding saved → {JD_EMBEDDING_FILE}  shape={jd_vec.shape}")

    # ── 2. Career narrative embeddings → FAISS index
    print(f"\n  [2/3] Building career narrative embeddings ({N:,} candidates) ...")
    narratives = [get_career_narrative(c) for c in candidates]
    print(f"  Mean narrative length: {sum(len(t.split()) for t in narratives)//N} tokens")

    career_vecs = embedder.embed_texts(
        narratives, batch_size=EMBEDDING_BATCH_SIZE, verbose=True
    )
    print(f"  Career embeddings shape: {career_vecs.shape}")

    # Build and save FAISS index
    index = build_faiss_index(career_vecs)
    save_faiss_index(index, FAISS_INDEX_FILE)

    # ── 3. Role-level embeddings → for KG domain continuity
    print(f"\n  [3/3] Building role-level embeddings ...")
    role_embs = build_role_embeddings(candidates, embedder, EMBEDDING_BATCH_SIZE)
    save_pickle(role_embs, ROLE_EMBEDDINGS_FILE)

    # ── Verification
    print(f"\n  Verification — FAISS search with JD vector:")
    jd_query = jd_vec.reshape(1, -1).astype(np.float32)
    faiss.normalize_L2(jd_query)
    scores, indices = index.search(jd_query, 10)

    from utils import load_pickle
    from config import CANDIDATE_IDS_FILE
    if CANDIDATE_IDS_FILE.exists():
        cand_ids = load_pickle(CANDIDATE_IDS_FILE)
        print(f"  {'Rank':>4s}  {'CandidateID':>18s}  {'CosineSim':>10s}")
        print("  " + "-" * 40)
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), 1):
            cid = cand_ids[idx] if idx < len(cand_ids) else f"idx_{idx}"
            print(f"  {rank:>4d}  {cid:>18s}  {score:>10.4f}")
    else:
        print(f"  Top-10 indices: {indices[0].tolist()}")
        print(f"  Top-10 scores:  {[round(float(s),4) for s in scores[0]]}")

    print(f"\n  Done. Artifacts saved:")
    print(f"    {FAISS_INDEX_FILE}")
    print(f"    {JD_EMBEDDING_FILE}")
    print(f"    {ROLE_EMBEDDINGS_FILE}")


if __name__ == "__main__":
    main()
