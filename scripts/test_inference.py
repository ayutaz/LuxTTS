#!/usr/bin/env python3
"""Test inference with fine-tuned Japanese model."""
import os, sys, json, torch, numpy as np, soundfile as sf
os.environ['PYTHONUTF8'] = '1'

from zipvoice.luxvoice import LuxTTS

def test_inference(model_dir='data/pretrained', checkpoint=None, token_file=None, lang=None):
    """Test inference with optional fine-tuned weights."""

    # Load model using official API
    lux = LuxTTS(model_dir, device='cuda', lang=lang)

    # Override with fine-tuned weights if provided
    if checkpoint and token_file:
        from zipvoice.tokenizer.tokenizer import EmiliaTokenizer
        # Replace tokenizer with expanded one
        lux.tokenizer = EmiliaTokenizer(token_file=token_file, lang=lang)

        # Load fine-tuned weights
        ckpt = torch.load(checkpoint, map_location='cuda', weights_only=False)
        if 'model' in ckpt:
            lux.model.load_state_dict(ckpt['model'], strict=False)
            print(f'Loaded fine-tuned weights from {checkpoint}')

    # Reference audio
    wavs = sorted(os.listdir('data/moe-speech-resampled/wavs/'))
    ref_wav = f'data/moe-speech-resampled/wavs/{wavs[0]}'
    print(f'Reference: {ref_wav}')

    prompt = lux.encode_prompt(ref_wav, duration=5, rms=0.01)

    # Test texts
    texts = {
        'english': 'Hello, how are you today? The weather is nice.',
        'japanese_hiragana': 'こんにちは、きょうはいいてんきですね。',
        'japanese_kanji': 'こんにちは、今日はいい天気ですね。',
        'japanese_long': '東京タワーは日本で一番有名な観光地です。毎年多くの人が訪れます。',
    }

    os.makedirs('exp/inference_test', exist_ok=True)
    for name, text in texts.items():
        try:
            audio = lux.generate_speech(text, prompt, num_steps=4, t_shift=0.9)
            audio_np = audio.numpy().astype(np.float32).flatten()
            path = f'exp/inference_test/{name}.wav'
            sf.write(path, audio_np, 48000, subtype='FLOAT')
            print(f'  [{name}] {len(audio_np)/48000:.1f}s -> {path}')
        except Exception as e:
            print(f'  [{name}] FAILED: {e}')

if __name__ == '__main__':
    print("=== Test 1: Original pretrained model (English) ===")
    test_inference()

    print("\n=== Test 2: Original model with lang=ja ===")
    test_inference(lang='ja')

    # Test 3 only if fine-tuned model exists
    ft_ckpt = 'exp/zipvoice_ja_v2/best-valid-loss.pt'
    if os.path.exists(ft_ckpt):
        print(f"\n=== Test 3: Fine-tuned model ===")
        test_inference(checkpoint=ft_ckpt, token_file='data/tokens_ja.txt', lang='ja')
