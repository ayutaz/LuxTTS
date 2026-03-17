#!/usr/bin/env bash
#
# train_japanese_distill.sh -- Japanese fine-tuning using ZipVoiceDistill architecture
#
# This is the CORRECTED training script that uses train_zipvoice_distill.py
# (ZipVoiceDistill model) instead of train_zipvoice.py (ZipVoice base model).
# The pretrained LuxTTS checkpoint is a distilled 4-step model, so fine-tuning
# must use the matching ZipVoiceDistill architecture.
#
# Approach:
#   1. Expand the embedding layer to accommodate Japanese tokens (401 vocab).
#   2. Run first-stage distillation with the original model as teacher.
#      - The teacher (ZipVoice) generates intermediate targets via multi-step
#        flow-matching sampling.
#      - The student (ZipVoiceDistill) learns to reach the same targets in
#        fewer steps, while also adapting its embed + text_encoder to Japanese.
#      - --freeze-decoder 0 unlocks embed and text_encoder so that the new
#        Japanese token embeddings can actually be learned (the default
#        distillation mode only trains fm_decoder).
#      - --checkpoint points to the embedding-expanded model so the student
#        starts with the correct vocab size, while --teacher-model points to
#        the original checkpoint for the teacher.
#
# Prerequisites:
#   1. Prepare Japanese fbank features and token manifests under data/fbank/
#      (see zipvoice/bin/prepare_dataset.py and zipvoice/bin/compute_fbank.py)
#   2. Generate the Japanese token file at data/tokens_ja.txt
#      (see scripts/generate_tokens.py)
#   3. Place the pretrained checkpoint at data/pretrained/model.pt
#      and config at data/pretrained/config.json
#
# Usage:
#   bash scripts/train_japanese_distill.sh
#

set -euo pipefail

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

# Force UTF-8 for correct Japanese text processing in Python
export PYTHONUTF8=1

# Reduce CUDA memory fragmentation (PyTorch >= 2.0)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ---------------------------------------------------------------------------
# Step 1: Expand embedding layer for Japanese tokens
# ---------------------------------------------------------------------------
# The pretrained model has ~350 tokens (English + Chinese phonemes).
# Japanese g2p via pyopenjtalk-plus adds new phoneme tokens.  We extend the
# embedding matrix to 401 rows, initialising new rows near the mean of the
# existing embeddings so they start in a reasonable region of weight space.

echo "=== Step 1: Expanding embedding layer ==="
python -m scripts.expand_distill_model \
    --input  data/pretrained/model.pt \
    --output data/pretrained/model_ja_distill.pt \
    --new-vocab-size 401

# ---------------------------------------------------------------------------
# Step 2: First-stage distillation fine-tuning
# ---------------------------------------------------------------------------
# Key design decisions:
#
# --teacher-model   : original pretrained checkpoint (teacher = ZipVoice base)
# --checkpoint      : embedding-expanded checkpoint (student = ZipVoiceDistill)
#                     The student is loaded with strict=False so the expanded
#                     embedding is kept while other weights come from the
#                     pretrained model.
#
# --freeze-decoder 0: unlocks embed + text_encoder in addition to fm_decoder.
#                     Standard distillation only trains fm_decoder, but for
#                     language adaptation the embedding and encoder must also
#                     learn the new phoneme representations.
#
# --finetune 1      : enables NaN/Inf loss skipping and graceful grad_scale
#                     recovery -- essential for stability when training with
#                     newly-initialised embeddings.
#
# --distill-stage first: teacher generates multi-step flow-matching targets;
#                        student learns to approximate them in fewer steps.
#
# --base-lr 0.00005 : conservative LR to avoid catastrophic forgetting of
#                     English/Chinese capabilities while learning Japanese.
#
# --num-iters 5000  : sufficient for language adaptation on a modest dataset.
#                     Monitor exp/zipvoice_ja_distill/tensorboard/ and increase
#                     if the loss is still decreasing at 5000 iterations.
#
# --max-duration 80 : total seconds of audio per batch. Reduce to 40-60 if
#                     you encounter OOM on GPUs with <16 GB VRAM.

echo "=== Step 2: First-stage distillation fine-tuning ==="

args=()

# --- Distributed / device ---------------------------------------------------
args+=(--world-size 1)
args+=(--use-fp16 1)

# --- Fine-tuning mode -------------------------------------------------------
args+=(--finetune 1)
args+=(--freeze-decoder 0)

# --- Model architecture -----------------------------------------------------
args+=(--model-config data/pretrained/config.json)

# --- Training schedule -------------------------------------------------------
args+=(--num-iters 5000)
args+=(--max-duration 80)
args+=(--base-lr 0.00005)
args+=(--save-every-n 1000)

# --- Tokenizer / language ----------------------------------------------------
args+=(--tokenizer emilia)
args+=(--lang ja)
args+=(--token-file data/tokens_ja.txt)

# --- Data --------------------------------------------------------------------
args+=(--dataset custom)
args+=(--train-manifest data/fbank/custom_cuts_train_tokens.jsonl.gz)
args+=(--dev-manifest   data/fbank/custom_cuts_dev_tokens.jsonl.gz)
args+=(--manifest-dir   data/fbank)

# --- Model checkpoints -------------------------------------------------------
# Teacher: original pretrained model (ZipVoice base architecture internally)
# Student: embedding-expanded model (ZipVoiceDistill architecture)
args+=(--teacher-model data/pretrained/model.pt)
args+=(--checkpoint    data/pretrained/model_ja_distill.pt)

# --- Distillation config -----------------------------------------------------
args+=(--distill-stage first)

# --- Output ------------------------------------------------------------------
args+=(--exp-dir exp/zipvoice_ja_distill)

# --- DataLoader performance --------------------------------------------------
args+=(--num-workers 8)
args+=(--prefetch-factor 8)

# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------
python -m zipvoice.bin.train_zipvoice_distill "${args[@]}"

echo "=== Training complete ==="
echo "Checkpoints saved to: exp/zipvoice_ja_distill/"
echo "Monitor progress:     tensorboard --logdir exp/zipvoice_ja_distill/tensorboard"
