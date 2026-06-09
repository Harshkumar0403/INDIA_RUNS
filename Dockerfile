# ─────────────────────────────────────────────────────────────────
# Dockerfile — Redrob Hackathon Ranking System
#
# Build (after running export_onnx.py and build_index.py):
#   docker build -t redrob-ranker .
#
# Run:
#   docker run --name ranker redrob-ranker
#   docker cp ranker:/app/Harsh_0403.csv .
#   docker rm ranker
#
# One-liner:
#   docker run --name ranker redrob-ranker && \
#   docker cp ranker:/app/Harsh_0403.csv . && \
#   docker rm ranker
# ─────────────────────────────────────────────────────────────────

FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY  requirements.txt . 
RUN pip install -r requirements.txt --no-cache-dir 

# Source code
COPY config.py constants.py utils.py jd_parser.py ./
COPY feature_extractor.py career_kg.py embedder.py ./
COPY hard_filter.py scorer.py reasoning.py ./
COPY mini_ranker.py rank.py build_index.py ./

# MiniLM embedding model (22 MB)
COPY models/minilm_l6.onnx          models/minilm_l6.onnx
COPY models/minilm_tokenizer/        models/minilm_tokenizer/

# T5 reasoning model (encoder + decoder, ~250 MB total)
COPY models/t5_tmp/encoder_model.onnx              models/t5_tmp/encoder_model.onnx
COPY models/t5_tmp/decoder_model.onnx              models/t5_tmp/decoder_model.onnx
COPY models/t5_tmp/decoder_with_past_model.onnx    models/t5_tmp/decoder_with_past_model.onnx
COPY models/t5_tokenizer/                          models/t5_tokenizer/
COPY models/t5_small_int8.onnx			   models/t5_small_int8.onnx
COPY models/t5_tmp/                                models/t5_tmp/




# Skill vocabulary from EDA
COPY eda_outputs/skill_vocabulary.json eda_outputs/skill_vocabulary.json

# Prebuilt ranking artifacts (~2 GB)
COPY artifacts/feature_matrix.pkl  artifacts/feature_matrix.pkl
COPY artifacts/candidate_ids.pkl   artifacts/candidate_ids.pkl
COPY artifacts/faiss_index.bin     artifacts/faiss_index.bin
COPY artifacts/kg_features.pkl     artifacts/kg_features.pkl
COPY artifacts/jd_embedding.npy    artifacts/jd_embedding.npy

# Full dataset (487 MB)
COPY INDIA_RUNS_Assets/candidates.jsonl       INDIA_RUNS_Assets/candidates.jsonl
COPY INDIA_RUNS_Assets/sample_candidates.json INDIA_RUNS_Assets/sample_candidates.json

RUN mkdir -p output

# -u = unbuffered output so logs print immediately
CMD ["python", "-u", "rank.py"]
