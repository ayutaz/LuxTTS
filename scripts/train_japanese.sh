#!/usr/bin/env bash
#
# train_japanese.sh -- Optimized Japanese fine-tuning for LuxTTS (ZipVoice)
#
# This script fine-tunes the pretrained distilled model on Japanese data
# with the fm_decoder frozen, using BF16 mixed precision for stability.
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
# Training
# ---------------------------------------------------------------------------

python3 -m zipvoice.bin.train_zipvoice \
    --world-size 1 \
    --use-bf16 1 \
    --finetune 1 \
    --freeze-decoder 1 \
    --model-name zipvoice_distill \
    --num-epochs 5 \
    --max-duration 150 \
    --base-lr 0.0001 \
    --grad-accum-steps 2 \
    --model-config data/pretrained/config.json \
    --tokenizer emilia \
    --lang ja \
    --token-file data/tokens_ja.txt \
    --dataset custom \
    --train-manifest data/fbank/custom_cuts_train_tokens.jsonl.gz \
    --dev-manifest data/fbank/custom_cuts_dev_tokens.jsonl.gz \
    --manifest-dir data/fbank \
    --checkpoint data/pretrained/model_ja.pt \
    --exp-dir exp/zipvoice_ja_optimized \
    --save-every-n 1000 \
    --num-workers 8 \
    --log-interval 100

# ---------------------------------------------------------------------------
# Option reference
# ---------------------------------------------------------------------------
#
# --world-size 1           Single-GPU training
# --use-bf16 1             BF16 mixed precision (more stable than FP16,
#                          recommended for RTX 3090/4090 fine-tuning)
# --finetune 1             Enable fine-tuning mode: uses a fixed learning
#                          rate schedule and skips the large dropout phase
# --freeze-decoder 1       Freeze the fm_decoder (flow-matching decoder)
#                          so only the text encoder is updated -- saves
#                          VRAM and speeds up training
# --model-name             Use the distilled model architecture (4-step
#   zipvoice_distill       Euler solver) as the starting point
# --num-epochs 5           Number of training epochs
# --max-duration 150       Max total audio duration (seconds) per batch;
#                          can be larger than usual because the decoder
#                          is frozen, reducing memory usage
# --base-lr 0.0001         Conservative learning rate for fine-tuning
# --grad-accum-steps 2     Accumulate gradients over 2 steps; effective
#                          batch size = 2 * 150 = 300s of audio per update
# --model-config           Model architecture config from pretrained dir
#   data/pretrained/config.json
# --tokenizer emilia       Use the Emilia tokenizer (supports Japanese
#                          via pyopenjtalk-plus g2p when --lang ja)
# --lang ja                Japanese language -- triggers Japanese g2p
#                          pipeline in the Emilia tokenizer
# --token-file             Token vocabulary file for Japanese
#   data/tokens_ja.txt
# --dataset custom         Use custom dataset manifests (not built-in
#                          Emilia/LibriTTS splits)
# --train-manifest         Path to training cut manifest
#   data/fbank/custom_cuts_train_tokens.jsonl.gz
# --dev-manifest           Path to validation cut manifest
#   data/fbank/custom_cuts_dev_tokens.jsonl.gz
# --manifest-dir           Base directory for lhotse cut manifests
#   data/fbank
# --checkpoint             Pretrained model checkpoint to fine-tune from
#   data/pretrained/model_ja.pt
# --exp-dir                Output directory for checkpoints, logs, and
#   exp/zipvoice_ja_optimized   tensorboard events
# --save-every-n 1000      Save a checkpoint every 1000 training steps
# --num-workers 8          DataLoader worker processes for parallel I/O
# --log-interval 100       Print training loss every 100 steps
