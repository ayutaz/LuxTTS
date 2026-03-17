#!/usr/bin/env bash
#
# train_japanese.sh -- Optimized Japanese fine-tuning for LuxTTS (ZipVoice)
#
# This script fine-tunes the pretrained model on Japanese data using all
# available optimizations for maximum training speed and stability:
#   - BF16 mixed precision
#   - Frozen decoder with staged text-encoder unfreezing
#   - Gradient checkpointing to reduce VRAM usage
#   - Gradient accumulation for larger effective batch size
#   - Curriculum learning with progressive max-length scheduling
#   - Early stopping to prevent overfitting
#   - Aggressive data prefetching for GPU saturation
#
# Prerequisites:
#   1. Prepare Japanese fbank features and token manifests under data/fbank/
#      (see zipvoice/bin/prepare_dataset.py and zipvoice/bin/compute_fbank.py)
#   2. Generate the Japanese token file at data/tokens_ja.txt
#      (see scripts/generate_tokens.py)
#   3. Place the pretrained checkpoint at data/pretrained/model_ja.pt
#      and config at data/pretrained/config.json
#
# Usage:
#   bash scripts/train_japanese.sh
#

set -euo pipefail

# ---------------------------------------------------------------------------
# Environment variables
# ---------------------------------------------------------------------------

# Force UTF-8 encoding for Python (important for Japanese text processing)
export PYTHONUTF8=1

# Use expandable CUDA memory segments to reduce fragmentation and avoid OOM
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ---------------------------------------------------------------------------
# Build the argument list
# ---------------------------------------------------------------------------
args=()

# --- Distributed / device ---------------------------------------------------
args+=(--world-size 1)
args+=(--use-bf16 1)

# --- Fine-tuning mode -------------------------------------------------------
# Enable fine-tuning with frozen decoder; unfreeze the text encoder after
# 2 warm-up epochs so the model first adapts its output heads to the new
# language before updating the heavier encoder weights.
args+=(--finetune 1)
args+=(--freeze-decoder 1)
args+=(--freeze-text-encoder-epochs 2)

# --- Memory optimizations ---------------------------------------------------
# Gradient checkpointing trades ~20% slower forward pass for a large
# reduction in activation memory, allowing bigger batches or longer
# sequences to fit in VRAM.
args+=(--gradient-checkpointing 1)

# --- Model architecture -----------------------------------------------------
args+=(--model-name zipvoice)
args+=(--model-config data/pretrained/config.json)

# --- Training schedule -------------------------------------------------------
# 10 epochs with a conservative LR, 500-batch linear warm-up, and gradient
# accumulation over 4 steps (effective batch = 4 * 200 = 800s of audio per
# parameter update). Early stopping halts training if the validation loss
# does not improve for 3 consecutive epochs.
args+=(--num-epochs 10)
args+=(--max-duration 200)
args+=(--base-lr 0.0001)
args+=(--warmup-batches 500)
args+=(--grad-accum-steps 4)
args+=(--early-stopping-patience 3)

# --- Curriculum learning -----------------------------------------------------
# Gradually increase the maximum utterance length across the first 3 phases
# of training (10s -> 20s -> 30s). This stabilises early optimisation steps
# and improves convergence on longer utterances.
args+=(--curriculum-max-lens "10,20,30")

# --- Tokenizer / language ----------------------------------------------------
# Emilia tokenizer with Japanese g2p (pyopenjtalk-plus).
args+=(--tokenizer emilia)
args+=(--lang ja)
args+=(--token-file data/tokens_ja.txt)

# --- Data --------------------------------------------------------------------
args+=(--dataset custom)
args+=(--train-manifest data/fbank/custom_cuts_train_tokens.jsonl.gz)
args+=(--dev-manifest data/fbank/custom_cuts_dev_tokens.jsonl.gz)
args+=(--manifest-dir data/fbank)

# --- Checkpoint / output -----------------------------------------------------
args+=(--checkpoint data/pretrained/model_ja.pt)
args+=(--exp-dir exp/zipvoice_ja_v2)
args+=(--save-every-n 1000)

# --- DataLoader performance --------------------------------------------------
# 8 workers with prefetch_factor=8 keeps the GPU fed by pre-loading the
# next 8 batches per worker while the current batch is being processed.
args+=(--num-workers 8)
args+=(--prefetch-factor 8)

# ---------------------------------------------------------------------------
# Launch training
# ---------------------------------------------------------------------------
python -m zipvoice.bin.train_zipvoice "${args[@]}"
