# 🕸️ ArcRank — Intelligent Hiring via Career Arc Analysis

> Ranks candidates by understanding career narratives — not matching keywords.

**Live Demo:** [https://indiaruns-88tch9omkj6zmuc3ew6pq9.streamlit.app](https://indiaruns-88tch9omkj6zmuc3ew6pq9.streamlit.app)

---

## Architecture

```
candidates.jsonl (100k)
        │
        ▼
┌──────────────────┐
│   Hard Filter    │  Binary gates — DQ titles, pure consulting, CV/speech,
│   (6 gates)      │  framework enthusiasts, honeypots → eliminates ~80%
└────────┬─────────┘
         │ ~19,500 eligible
         ▼
┌──────────────────┐
│  FAISS Retrieval │  ONNX MiniLM-L6-v2 embeddings → sub-index search
│  (top-5,000)     │  Career narrative vs JD semantic similarity
└────────┬─────────┘
         │
         ▼
┌──────────────────────────────────────────────────────┐
│              Scoring Matrix (25 features)             │
│                                                       │
│  Semantic     ×  Career KG Arc  ×  Skill IDF         │
│  (FAISS cos)     (causal edges)    (corpus-derived)  │
│                                                       │
│  × Availability Multiplier  × Location Gate          │
│    (e^-λt decay)               (outside-India cap)   │
└────────┬─────────────────────────────────────────────┘
         │
         ▼
┌──────────────────┐
│  Top-100 Output  │  Calibrated scores [0.50–0.95]
│  + Reasoning     │  Grounded, fact-extracted reasoning
└──────────────────┘
```

**Career KG** (novel contribution) — adapted from event-centric narrative KG research.
Career roles classified into 8 event types (`CORE_ML_ROLE`, `DATA_ENGINEERING`, `LEADERSHIP_EVENT`…).
Causal edges between adjacent roles score career progression logic.
A `Data Eng → ML Eng → Tech Lead at product companies` arc scores higher than
the same keywords in a disconnected or consulting-heavy career.

**Key design choices:**
- IDF-weighted skills from 100k corpus (no hardcoded taxonomy)
- JD parsed into 4 signed sections — Section D terms (LangChain, TCS, CV-only) **penalise**, not reward
- Multiplicative score composition — zero availability = near-zero final score regardless of skills

---

## Files

```
├── rank.py               # Main ranking entry point → Harsh_0403.csv
├── mini_ranker.py        # Standalone ranker for small inputs (sandbox)
├── build_index.py        # Offline pipeline — builds artifacts/ (run once)
├── export_onnx.py        # Exports MiniLM + T5 to ONNX (run once)
├── feature_extractor.py  # 25-dim candidate feature matrix
├── career_kg.py          # Career event KG + causal arc scoring
├── embedder.py           # ONNX MiniLM inference + FAISS index
├── hard_filter.py        # Binary elimination gates
├── scorer.py             # Multiplicative score fusion
├── reasoning.py          # Grounded reasoning generation
├── jd_parser.py          # 4-section polarity-aware JD parser
├── sandbox/app.py        # Streamlit UI (ArcRank)
├── Dockerfile
└── requirements.txt
```

---

## Run Without Docker

**Prerequisites:** Python 3.11+, 16 GB RAM, CPU only.

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Export ONNX models (one time, needs internet, ~5 min)
python export_onnx.py

# 3. Build index artifacts (one time, ~50 min on 100k candidates)
python build_index.py

# 4. Run ranking → produces Harsh_0403.csv
python rank.py
```

**Quick test on sample data (no artifacts needed):**
```bash
python mini_ranker.py
# → produces sample_rank.csv from sample_candidates.json
```

**Custom input:**
```bash
python mini_ranker.py --candidates your_candidates.json --out output.csv
```

---

## Run With Docker

```bash
# Build
docker build -t redrob-ranker .

# Run → produces Harsh_0403.csv inside the container
docker run --name ranker redrob-ranker

# Copy CSV to your current directory
docker cp ranker:/app/Harsh_0403.csv .

# Clean up
docker rm ranker
```

One-liner:
```bash
docker run --name ranker redrob-ranker && \
docker cp ranker:/app/Harsh_0403.csv . && \
docker rm ranker
```

---

## Constraints

| Constraint | Limit | ArcRank |
|---|---|---|
| Runtime (ranking step) | ≤ 5 min | ~22s |
| Memory | ≤ 16 GB | ~4 GB peak |
| Compute | CPU only | ✅ ONNX CPUExecutionProvider |
| Network | Off | ✅ Zero API calls |

---

## AI Tools

Claude (Anthropic) used as development assistant.
Architecture, EDA methodology, KG design and engineering were done by the team
with iterative debugging and improvement cycles.
