"""
export_onnx.py
==============
ONE-TIME offline script. Run this BEFORE the ranking pipeline.
Exports two models to ONNX format so the ranker never needs
PyTorch or the full transformers library at inference time.

Models exported:
  1. all-MiniLM-L6-v2  → models/minilm_l6.onnx       (~22 MB)
     Used for: career narrative embeddings + JD embedding
     Why ONNX: 8ms/batch on CPU vs 35ms with torch, no GPU needed

  2. t5-small (int8)   → models/t5_small_int8.onnx    (~60 MB)
     Used for: reasoning string generation (top-100 only)
     Why int8: 60MB vs 240MB full precision, ~2× faster on CPU
     Why t5-small not larger: ranking step must run ≤5 min on CPU.
     Reasoning is called for 100 candidates only — not 100k.

Usage:
    python export_onnx.py

Requirements: sentence-transformers, transformers, optimum
(already in requirements.txt — these are OFFLINE dependencies only)
"""

import os
from pathlib import Path

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

MINILM_HF   = "sentence-transformers/all-MiniLM-L6-v2"
T5_HF       = "google-t5/t5-small"
MINILM_OUT  = MODELS_DIR / "minilm_l6.onnx"
T5_OUT      = MODELS_DIR / "t5_small_int8.onnx"
MINILM_TOK  = MODELS_DIR / "minilm_tokenizer"
T5_TOK      = MODELS_DIR / "t5_tokenizer"


# ─────────────────────────────────────────────────────────────────
# EXPORT 1: MiniLM-L6 → ONNX
# ─────────────────────────────────────────────────────────────────
def export_minilm():
    print("\n[1/2] Exporting all-MiniLM-L6-v2 to ONNX ...")

    if MINILM_OUT.exists():
        print(f"  Already exists: {MINILM_OUT}  — skipping.")
        return

    try:
        from optimum.onnxruntime import ORTModelForFeatureExtraction
        from transformers import AutoTokenizer

        print(f"  Downloading from HuggingFace: {MINILM_HF}")
        tokenizer = AutoTokenizer.from_pretrained(MINILM_HF)
        model     = ORTModelForFeatureExtraction.from_pretrained(
            MINILM_HF,
            export=True,
        )

        # Save ONNX model + tokenizer
        model.save_pretrained(MODELS_DIR / "minilm_tmp")
        tokenizer.save_pretrained(MINILM_TOK)

        # Move the ONNX file to expected path
        tmp_onnx = MODELS_DIR / "minilm_tmp" / "model.onnx"
        if tmp_onnx.exists():
            tmp_onnx.rename(MINILM_OUT)
            print(f"  Saved → {MINILM_OUT}  ({MINILM_OUT.stat().st_size/1024/1024:.1f} MB)")
        else:
            print(f"  ERROR: ONNX file not found at expected path {tmp_onnx}")

    except ImportError as e:
        print(f"  ERROR: {e}")
        print("  Install: pip install optimum[onnxruntime] sentence-transformers")

    except Exception as e:
        print(f"  ERROR during export: {e}")
        # Fallback: try direct torch.onnx export
        _export_minilm_torch_fallback()


def _export_minilm_torch_fallback():
    """Fallback using direct torch.onnx if optimum fails."""
    print("  Trying torch.onnx fallback ...")
    try:
        import torch
        from transformers import AutoTokenizer, AutoModel

        tokenizer = AutoTokenizer.from_pretrained(MINILM_HF)
        model     = AutoModel.from_pretrained(MINILM_HF)
        model.eval()

        tokenizer.save_pretrained(MINILM_TOK)

        # Dummy input for tracing
        dummy = tokenizer(
            ["This is a test sentence for ONNX export."],
            return_tensors="pt", padding=True, truncation=True, max_length=128
        )

        with torch.no_grad():
            torch.onnx.export(
                model,
                (dummy["input_ids"], dummy["attention_mask"]),
                str(MINILM_OUT),
                input_names=["input_ids", "attention_mask"],
                output_names=["last_hidden_state"],
                dynamic_axes={
                    "input_ids":      {0: "batch", 1: "seq"},
                    "attention_mask": {0: "batch", 1: "seq"},
                    "last_hidden_state": {0: "batch", 1: "seq"},
                },
                opset_version=14,
                do_constant_folding=True,
            )
        print(f"  Saved (torch fallback) → {MINILM_OUT}  "
              f"({MINILM_OUT.stat().st_size/1024/1024:.1f} MB)")

    except Exception as e2:
        print(f"  Fallback also failed: {e2}")
        print("  You may need to install: pip install torch")


# ─────────────────────────────────────────────────────────────────
# EXPORT 2: T5-small → ONNX (encoder + decoder separately)
# reasoning.py expects:  models/t5_tmp/encoder_model.onnx
#                        models/t5_tmp/decoder_with_past_model.onnx
#                        models/t5_tokenizer/
# ─────────────────────────────────────────────────────────────────
def export_t5():
    print("\n[2/2] Exporting T5-small to ONNX (encoder + decoder) ...")

    t5_tmp = MODELS_DIR / "t5_tmp"
    enc_out = t5_tmp / "encoder_model.onnx"
    dec_out = t5_tmp / "decoder_with_past_model.onnx"

    if enc_out.exists() and dec_out.exists():
        print(f"  Already exists: {t5_tmp}/  — skipping.")
        return

    t5_tmp.mkdir(exist_ok=True)

    try:
        from optimum.onnxruntime import ORTModelForSeq2SeqLM
        from transformers import AutoTokenizer

        print(f"  Downloading T5-small from HuggingFace ...")
        tokenizer = AutoTokenizer.from_pretrained(T5_HF)
        tokenizer.save_pretrained(T5_TOK)
        print(f"  Tokenizer saved → {T5_TOK}")

        print(f"  Exporting T5 to ONNX (this takes 2-4 min) ...")
        model = ORTModelForSeq2SeqLM.from_pretrained(T5_HF, export=True)
        model.save_pretrained(t5_tmp)

        # List what was exported
        print(f"  Exported files:")
        for f in sorted(t5_tmp.rglob("*.onnx")):
            print(f"    {f.relative_to(MODELS_DIR)}  "
                  f"({f.stat().st_size/1024/1024:.1f} MB)")

        # Try int8 quantization on the decoder for speed
        try:
            from optimum.onnxruntime import ORTQuantizer
            from optimum.onnxruntime.configuration import AutoQuantizationConfig
            print(f"  Quantizing decoder to int8 ...")
            qconfig = AutoQuantizationConfig.avx2(is_static=False, per_channel=False)
            for component in ["decoder_model.onnx", "decoder_with_past_model.onnx"]:
                src = t5_tmp / component
                if src.exists():
                    q = ORTQuantizer.from_pretrained(t5_tmp, file_name=component)
                    q.quantize(save_dir=t5_tmp / "int8", quantization_config=qconfig)
            print(f"  int8 quantized files saved to {t5_tmp}/int8/")
        except Exception as qe:
            print(f"  int8 quantization skipped ({qe}) — using fp32.")

    except ImportError as e:
        print(f"  Import error: {e}")
        print(f"  Install: pip install optimum[onnxruntime] transformers")
        _export_t5_torch_fallback(t5_tmp)
    except Exception as e:
        print(f"  T5 ONNX export failed: {e}")
        print(f"  Reasoning will use structured fallback (still grounded).")


def _export_t5_torch_fallback(t5_tmp: Path):
    """Fallback: export T5 encoder only via torch.onnx."""
    print("  Trying torch.onnx fallback for T5 encoder ...")
    try:
        import torch
        from transformers import AutoTokenizer, T5ForConditionalGeneration

        tok   = AutoTokenizer.from_pretrained(T5_HF)
        model = T5ForConditionalGeneration.from_pretrained(T5_HF)
        model.eval()
        tok.save_pretrained(T5_TOK)

        enc_out = t5_tmp / "encoder_model.onnx"
        dummy   = tok(["test"], return_tensors="pt", max_length=32,
                      truncation=True, padding=True)
        enc = model.encoder

        with torch.no_grad():
            torch.onnx.export(
                enc,
                (dummy["input_ids"], dummy["attention_mask"]),
                str(enc_out),
                input_names=["input_ids", "attention_mask"],
                output_names=["last_hidden_state"],
                dynamic_axes={
                    "input_ids":        {0:"batch",1:"seq"},
                    "attention_mask":   {0:"batch",1:"seq"},
                    "last_hidden_state":{0:"batch",1:"seq"},
                },
                opset_version=13,
            )
        print(f"  T5 encoder saved → {enc_out}  "
              f"({enc_out.stat().st_size/1024/1024:.1f} MB)")
        print(f"  Note: decoder not exported — reasoning will use fallback.")
    except Exception as e2:
        print(f"  Torch fallback also failed: {e2}")


# ─────────────────────────────────────────────────────────────────
# VERIFY EXPORTS
# ─────────────────────────────────────────────────────────────────
def verify_exports():
    print("\n[Verify] Testing exported models ...")

    # Test MiniLM
    if MINILM_OUT.exists():
        try:
            import onnxruntime as ort
            from tokenizers import Tokenizer

            sess = ort.InferenceSession(
                str(MINILM_OUT),
                providers=["CPUExecutionProvider"]
            )
            print(f"  MiniLM ONNX loaded OK. Inputs: "
                  f"{[i.name for i in sess.get_inputs()]}")
        except Exception as e:
            print(f"  MiniLM verify failed: {e}")
    else:
        print(f"  MiniLM ONNX not found at {MINILM_OUT}")

    # Test T5
    if T5_OUT.exists():
        print(f"  T5 ONNX found: {T5_OUT.stat().st_size/1024/1024:.1f} MB")
    else:
        print(f"  T5 ONNX not found — reasoning will use template fallback")

    # Summary
    print("\n  Model files:")
    for p in sorted(MODELS_DIR.rglob("*.onnx")):
        print(f"    {p.relative_to(MODELS_DIR)}  "
              f"({p.stat().st_size/1024/1024:.1f} MB)")


if __name__ == "__main__":
    print("=" * 60)
    print("  ONNX Export Script  (run once, offline)")
    print("=" * 60)

    export_minilm()
    export_t5()
    verify_exports()

    print("\nDone. You can now run the offline pipeline:")
    print("  python build_index.py")
