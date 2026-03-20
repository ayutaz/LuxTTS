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

#### 環境構築

```bash
# 1. uvで仮想環境を作成（Python 3.10推奨）
uv venv --python 3.10
uv pip install -e ".[train]"

# onnxruntimeはPython 3.10では<=1.23.2が必要
# pyproject.tomlに設定済み

# 2. テストの実行
uv run pytest tests/test_japanese_tokenizer.py -v
```

**主要な依存関係:**
- `pyopenjtalk-plus` — 日本語G2P（grapheme-to-phoneme）
- `torch` + `torchaudio` — 学習・推論
- `lhotse` — データ準備・読み込み
- `transformers` — Whisper（音声プロンプトの書き起こし）
- `einops` — 学習時のみ（`[train]`オプション）

**Windows固有の注意:**
- `PYTHONUTF8=1`環境変数が必須（日本語テキストのエンコーディング）
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`推奨（CUDAメモリ断片化対策）
- `scaling.py`のSwooshLForward/SwooshRForwardは`torch.logaddexp`を使用（k2ライブラリなし環境でのexp()オーバーフロー対策）

#### データ準備

```bash
# 1. データセットのダウンロード（例: moe-speech-20speakers-ljspeech）
huggingface-cli download --repo-type dataset ayousanz/moe-speech-20speakers-ljspeech \
    --local-dir data/moe-speech-20speakers-ljspeech
cd data/moe-speech-20speakers-ljspeech && unzip wavs.zip

# 2. TSV変換 + リサンプリング（22kHz→24kHz）
python scripts/convert_moe_speech.py --num-workers 8

# 3. Lhotseマニフェスト生成
PYTHONUTF8=1 python -m zipvoice.bin.prepare_dataset \
    --tsv-path data/custom_train.tsv --prefix custom --subset train \
    --output-dir data/manifests --sampling-rate 24000

PYTHONUTF8=1 python -m zipvoice.bin.prepare_dataset \
    --tsv-path data/custom_dev.tsv --prefix custom --subset dev \
    --output-dir data/manifests --sampling-rate 24000

# 4. 日本語トークン化（pyopenjtalk G2P）
PYTHONUTF8=1 python -m zipvoice.bin.prepare_tokens \
    --input-file data/manifests/custom_cuts_train.jsonl.gz \
    --output-file data/manifests/custom_cuts_train_tokens.jsonl.gz \
    --tokenizer emilia --lang ja

PYTHONUTF8=1 python -m zipvoice.bin.prepare_tokens \
    --input-file data/manifests/custom_cuts_dev.jsonl.gz \
    --output-file data/manifests/custom_cuts_dev_tokens.jsonl.gz \
    --tokenizer emilia --lang ja

# 5. VocosFbank特徴量抽出（100次元メルスペクトログラム、24kHz）
PYTHONUTF8=1 python -m zipvoice.bin.compute_fbank \
    --source-dir data/manifests --dest-dir data/fbank \
    --dataset custom --subset train_tokens --sampling-rate 24000 --type vocos

PYTHONUTF8=1 python -m zipvoice.bin.compute_fbank \
    --source-dir data/manifests --dest-dir data/fbank \
    --dataset custom --subset dev_tokens --sampling-rate 24000 --type vocos
```

#### 事前学習モデルの準備

```bash
# 1. HuggingFaceからダウンロード
python -c "from huggingface_hub import snapshot_download; snapshot_download('YatharthS/LuxTTS', local_dir='data/pretrained')"

# 2. 日本語トークンファイル生成（英語360 + 日本語41 = 401トークン）
python scripts/generate_tokens.py

# 3. 埋め込み層の初期化（英語音素→日本語音素マッピング）
python scripts/init_japanese_embeds.py
```

#### 学習の実行

```bash
# 一括実行
bash scripts/train_japanese.sh

# または手動で実行（パラメータ調整可能）
PYTHONUTF8=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -m zipvoice.bin.train_zipvoice \
    --world-size 1 --use-fp16 1 --finetune 1 \
    --model-name zipvoice --num-epochs 50 --max-duration 80 \
    --base-lr 0.0001 --grad-accum-steps 2 --early-stopping-patience 5 \
    --model-config data/pretrained/config.json \
    --tokenizer emilia --lang ja --token-file data/tokens_ja.txt \
    --dataset custom \
    --train-manifest data/fbank/custom_cuts_train_tokens.jsonl.gz \
    --dev-manifest data/fbank/custom_cuts_dev_tokens.jsonl.gz \
    --manifest-dir data/fbank \
    --checkpoint data/pretrained/model_ja_distill_v3.pt \
    --exp-dir exp/zipvoice_ja --save-every-n 5000 --num-workers 8
```

**学習時間:** RTX 4090で約24時間（Early stoppingにより30エポック前後で停止）

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
