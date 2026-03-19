#!/bin/bash
# Japanese fine-tuning for LuxTTS
# IMPORTANT: Use ZipVoice (base) for both training AND inference
# Inference: LuxTTS(model_path, device, lang='ja', model_name='zipvoice')
# The base model uses EulerSolver which needs num_steps=16 (not 4)

export PYTHONUTF8=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Step 1: Initialize Japanese embeddings from English phoneme mapping
python scripts/init_japanese_embeds.py

# Step 2: Train
python -m zipvoice.bin.train_zipvoice \
    --world-size 1 \
    --use-fp16 1 \
    --finetune 1 \
    --model-name zipvoice \
    --num-epochs 50 \
    --max-duration 80 \
    --base-lr 0.0005 \
    --grad-accum-steps 2 \
    --early-stopping-patience 5 \
    --model-config data/pretrained/config.json \
    --tokenizer emilia \
    --lang ja \
    --token-file data/tokens_ja.txt \
    --dataset custom \
    --train-manifest data/fbank/custom_cuts_train_tokens.jsonl.gz \
    --dev-manifest data/fbank/custom_cuts_dev_tokens.jsonl.gz \
    --manifest-dir data/fbank \
    --checkpoint data/pretrained/model_ja_distill_v3.pt \
    --exp-dir exp/zipvoice_ja_v3 \
    --save-every-n 5000 \
    --num-workers 8 \
    --prefetch-factor 8
