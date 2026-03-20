#!/bin/bash
# ==============================================================================
# Japanese Fine-tuning for LuxTTS
# ==============================================================================
#
# WHY ZipVoice BASE (not Distill):
#   Training calls ZipVoice.forward(), which computes the flow-matching loss
#   directly. The distilled model (ZipVoiceDistill) wraps the base model and
#   only provides a .sample() shortcut for 4-step inference -- it does NOT
#   implement .forward(). Therefore we MUST use model_name=zipvoice (the base
#   model) for training. This also means inference after training must use the
#   base model with num_steps=16 (not the distill shortcut with num_steps=4).
#
# TRAINING PIPELINE:
#   Step 1 - Embedding initialization
#     Run init_japanese_embeds.py to create Japanese phoneme embeddings.
#     This script maps each Japanese phoneme (from pyopenjtalk-plus) to the
#     closest English phoneme embedding in the pretrained model, giving the
#     model a reasonable starting point instead of random weights.
#
#   Step 2 - Fine-tuning
#     Fine-tune the base ZipVoice model on Japanese speech data.
#     Training uses FP16 mixed precision with gradient accumulation.
#
# KEY HYPERPARAMETERS:
#   --base-lr 0.0001    Learning rate. 0.0005 causes loss divergence after
#                        ~10 epochs; 0.0001 is stable throughout training.
#   --num-epochs 50      Maximum epochs. With early-stopping-patience=5,
#                        training typically stops around epoch 30.
#   --early-stopping-patience 5
#                        Stop if dev loss does not improve for 5 consecutive
#                        evaluation rounds. Prevents overfitting.
#
# EXPECTED TRAINING TIME:
#   ~24 hours on a single NVIDIA RTX 4090 (24GB VRAM).
#   VRAM usage peaks at ~18GB with max-duration=80 and FP16 enabled.
#
# INFERENCE AFTER TRAINING:
#   The trained checkpoint must be loaded with the BASE model, not distill.
#   Use num_steps=16 (the base model's EulerSolver, not the 4-step distill).
#
#   Python API:
#     from zipvoice import LuxTTS
#     lux_tts = LuxTTS(
#         'exp/zipvoice_ja_v3',
#         device='cuda',
#         lang='ja',
#         model_name='zipvoice'       # BASE model, not 'zipvoice_distill'
#     )
#     encoded_prompt = lux_tts.encode_prompt('reference.wav', duration=5, rms=0.01)
#     audio = lux_tts.generate_speech(
#         "こんにちは",
#         encoded_prompt,
#         num_steps=16,                # Base model needs 16 steps (not 4)
#         t_shift=0.9
#     )
#
#   CLI:
#     python -m zipvoice.bin.infer_zipvoice \
#         --model-dir exp/zipvoice_ja_v3 \
#         --model-name zipvoice \
#         --lang ja \
#         --num-steps 16 \
#         --text "こんにちは" \
#         --prompt reference.wav
#
# ==============================================================================

export PYTHONUTF8=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Step 1: Initialize Japanese embeddings from English phoneme mapping
# Maps Japanese phonemes (pyopenjtalk-plus) to their closest English phoneme
# embeddings in the pretrained model, providing a warm start for fine-tuning.
python scripts/init_japanese_embeds.py

# Step 2: Fine-tune the base ZipVoice model on Japanese data
python -m zipvoice.bin.train_zipvoice \
    --world-size 1 \
    --use-fp16 1 \
    --finetune 1 \
    --model-name zipvoice \
    --num-epochs 50 \
    --max-duration 80 \
    --base-lr 0.0001 \
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
