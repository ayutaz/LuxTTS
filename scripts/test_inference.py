#!/usr/bin/env python3
"""Test Japanese TTS inference with fine-tuned model."""
import os, torch, numpy as np, soundfile as sf
os.environ['PYTHONUTF8'] = '1'

from zipvoice.luxvoice import LuxTTS

def main():
    # Use ZipVoice BASE model (matches training architecture)
    lux = LuxTTS('data/pretrained', device='cuda', lang='ja', model_name='zipvoice')

    # Load fine-tuned Japanese weights
    ckpt_path = 'exp/zipvoice_ja_v4/best-valid-loss.pt'
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location='cuda', weights_only=False)
        lux.model.load_state_dict(ckpt['model'], strict=False)
        print(f'Loaded: {ckpt_path}')

    # Reference audio
    ref_wav = 'data/moe-speech-resampled/wavs/' + sorted(os.listdir('data/moe-speech-resampled/wavs/'))[0]
    prompt = lux.encode_prompt(ref_wav, duration=5, rms=0.01)

    # Generate
    texts = [
        ('ja1', 'こんにちは、今日はいい天気ですね。'),
        ('ja2', 'はじめまして、私の名前は太郎です。'),
        ('ja3', '東京タワーは日本で一番有名な観光地です。'),
        ('en1', 'Hello, how are you today?'),
    ]
    os.makedirs('exp/test_output', exist_ok=True)
    for name, text in texts:
        # num_steps=16 for base EulerSolver (auto-selected when model_name='zipvoice')
        audio = lux.generate_speech(text, prompt, t_shift=0.9, speed=1.0)
        a = audio.numpy().astype(np.float32).flatten()
        sf.write(f'exp/test_output/{name}.wav', a, 48000, subtype='FLOAT')
        print(f'[{name}] "{text}" -> {len(a)/48000:.1f}s')
    print('Output: exp/test_output/')

if __name__ == '__main__':
    main()
