# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## プロジェクト概要

LuxTTS（パッケージ名: `zipvoice`）は、音声クローニングに特化した軽量テキスト音声合成モデル。Flow-matching拡散と蒸留済み4ステップEulerソルバーによる効率的な推論を実現。48kHz音声を出力。CUDA、MPS（Mac）、CPU（ONNX経由）に対応。

## ビルド・インストール

```bash
# uvでインストール（推奨）
uv pip install -e .

# またはpipで
pip install -e .
```

ビルドバックエンドは `uv_build`。`piper_phonemize` の依存解決には特別なfind-links URLが必要（pyproject.tomlに設定済み）。

## 推論の実行

```python
from zipvoice import LuxTTS

lux_tts = LuxTTS('YatharthS/LuxTTS', device='cuda')  # 'cpu' または 'mps' も可
encoded_prompt = lux_tts.encode_prompt('reference_audio.wav', duration=5, rms=0.01)
audio = lux_tts.generate_speech("Text to speak", encoded_prompt, num_steps=4, t_shift=0.9)
```

CLI推論スクリプト: `zipvoice/bin/infer_zipvoice.py`（GPU用）、`zipvoice/bin/infer_zipvoice_onnx.py`（CPU/ONNX用）。

## テスト

自動テストスイートは存在しない。推論スクリプトやHuggingFace Spacesデモによる手動テストで検証。

## アーキテクチャ

### データフロー

```
テキスト → Tokenizer → Text Encoder (TTSZipformer) → テキスト条件付け
音声プロンプト → Librosa/Whisper → VocosFbank特徴量 + 書き起こしトークン
ノイズ → Flow-Matching Decoder (TTSZipformer, 5層) + Euler ODEソルバー → 音声特徴量
音声特徴量 → Vocos vocoder (linacodec) → 48kHz波形
```

### 主要モジュール

- **`zipvoice/luxvoice.py`** — メインの `LuxTTS` クラス。encode_prompt() と generate_speech() のユーザー向けAPI。全推論のエントリーポイント。
- **`zipvoice/modeling_utils.py`** — 推論のコアロジック。`generate()`（GPU用）と `generate_cpu()`（ONNX用）がパイプライン全体を制御。
- **`zipvoice/models/zipvoice_distill.py`** — 蒸留モデル（4ステップ）。本番用の主要モデル。`zipvoice.py` はベースモデル（16ステップ以上）。
- **`zipvoice/models/modules/zipformer.py`** — TTSZipformer: テキストエンコーディングとflow-matchingデコーディングの両方に使用されるコアニューラルアーキテクチャ。約1700行。
- **`zipvoice/models/modules/solver.py`** — Flow-matchingサンプリング用のEuler ODEソルバー。
- **`zipvoice/tokenizer/tokenizer.py`** — 複数のTokenizer実装（EmiliaTokenizerが主要）。音素レベルのトークン化を処理。
- **`zipvoice/tokenizer/normalizer.py`** — 英語・中国語のテキスト正規化。
- **`zipvoice/utils/feature.py`** — VocosFbankメルスペクトログラム抽出。
- **`zipvoice/onnx_modeling.py`** — CPUデプロイ用のONNXランタイム推論。

### 推論パス

- **GPU**: PyTorchモデル → `model.sample()` → Vocosデコード
- **CPU**: ONNXモデル (`text_encoder.onnx` + `fm_decoder.onnx`) → Vocosデコード
- **MPS**: GPUパスと同様、PyTorchのMPSデバイスを使用

### 学習スクリプト

`zipvoice/bin/` 配下: `train_zipvoice.py`、`train_zipvoice_distill.py`、`train_zipvoice_dialog.py`。データ準備は `prepare_dataset.py`、`prepare_tokens.py`、`compute_fbank.py`。

### 主要パラメータ

- `num_steps`: 4（蒸留）または16以上（ベース） — 品質と速度のトレードオフ
- `t_shift`: 0.5–0.9 — 発音の明瞭さを制御
- `guidance_scale`: デフォルト3.0 — プロンプト忠実度とテキスト忠実度のバランス
- `speed`: デフォルト1.3倍 — 時間方向のスケーリング

### 日本語対応

#### 推論
```python
from zipvoice.luxvoice import LuxTTS

# 日本語モデルの読み込み（ZipVoice BASEモデルを使用）
lux_tts = LuxTTS('path/to/model', device='cuda', lang='ja', model_name='zipvoice')

# ファインチューニング済み重みの読み込み
import torch
ckpt = torch.load('exp/zipvoice_ja_v4/best-valid-loss.pt', map_location='cuda', weights_only=False)
lux_tts.model.load_state_dict(ckpt['model'], strict=False)

# 音声生成（BASEモデルはnum_steps=16が推奨）
encoded_prompt = lux_tts.encode_prompt('reference_audio.wav', duration=5, rms=0.01)
audio = lux_tts.generate_speech("こんにちは、今日はいい天気ですね。", encoded_prompt, num_steps=16, t_shift=0.9)
```

#### 日本語学習パイプライン
1. データ準備: `scripts/convert_moe_speech.py` → TSV変換 + リサンプリング
2. マニフェスト: `prepare_dataset.py` → Lhotseマニフェスト生成
3. トークン化: `prepare_tokens.py --lang ja` → pyopenjtalk G2P
4. 特徴量: `compute_fbank.py` → VocosFbank (100次元, 24kHz)
5. 埋め込み初期化: `scripts/init_japanese_embeds.py` → 英語音素マッピング
6. 学習: `scripts/train_japanese.sh` → ZipVoice BASE + AdamW + FP16

#### 重要な注意事項
- 学習にはZipVoice（ベース）を使用、推論時も`model_name='zipvoice'`を指定
- `model_name='zipvoice_distill'`は英語/中国語の事前学習モデル用
- BASEモデルのEulerSolverは16ステップ以上が推奨（DistillEulerSolverの4ステップとは異なる）
- J_トークンの埋め込みは英語音素からマッピング初期化する（ランダム初期化は不可）
- `scaling.py`のSwooshLForward/SwooshRForwardは`torch.logaddexp`を使用（exp()オーバーフロー対策）

#### トークン体系
- 英語: espeak IPA音素（ID 0-159）
- 中国語: pypinyin（ID 160-359）
- 日本語: pyopenjtalk + J_プレフィックス（ID 360-400、41トークン）
- 例: こんにちは → J_k J_o J_N J_n J_i J_ch J_i J_w J_a
