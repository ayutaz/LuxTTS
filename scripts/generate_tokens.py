#!/usr/bin/env python3
"""Generate a merged tokens.txt for Japanese TTS training with LuxTTS.

This script:
1. Reads the existing pretrained tokens file (data/pretrained/tokens.txt)
2. Generates all possible J_-prefixed Japanese phonemes via pyopenjtalk.g2p()
3. Scans training data manifests to find tokens actually used in supervisions
4. Merges everything: existing tokens keep their IDs, new J_ tokens get
   sequential IDs starting after the last existing one
5. Saves the result to data/tokens_ja.txt
"""

import gzip
import json
import logging
import os
import sys
import time
from collections import OrderedDict
from pathlib import Path

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    level=logging.INFO,
    force=True,
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PRETRAINED_TOKENS = PROJECT_ROOT / "data" / "pretrained" / "tokens.txt"
TRAIN_MANIFEST = PROJECT_ROOT / "data" / "fbank" / "custom_cuts_train_tokens.jsonl.gz"
OUTPUT_TOKENS = PROJECT_ROOT / "data" / "tokens_ja.txt"


# ---------------------------------------------------------------------------
# 1. Read existing pretrained tokens
# ---------------------------------------------------------------------------
def read_existing_tokens(path: Path) -> OrderedDict:
    """Read a tokens.txt file and return an OrderedDict of token -> id."""
    token2id: OrderedDict = OrderedDict()
    if not path.exists():
        logger.warning("Pretrained tokens file not found at %s", path)
        return token2id
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 2:
                logger.warning("Skipping malformed line: %r", line)
                continue
            token, tid = parts[0], int(parts[1])
            if token in token2id:
                logger.warning("Duplicate token in pretrained file: %r", token)
                continue
            token2id[token] = tid
    logger.info(
        "Read %d existing tokens from %s (max id=%d)",
        len(token2id),
        path,
        max(token2id.values()) if token2id else -1,
    )
    return token2id


# ---------------------------------------------------------------------------
# 2. Generate all possible Japanese phonemes via pyopenjtalk
# ---------------------------------------------------------------------------
def generate_japanese_phonemes() -> set:
    """Run pyopenjtalk.g2p on comprehensive Japanese text to collect all
    possible phonemes, then return them as a set of J_-prefixed strings."""
    try:
        import pyopenjtalk
    except ImportError:
        logger.error(
            "pyopenjtalk-plus is required. Install with: pip install pyopenjtalk-plus"
        )
        sys.exit(1)

    # Comprehensive Japanese text samples covering all phoneme categories
    texts = [
        # --- All basic hiragana (gojuuon) ---
        "あいうえお",
        "かきくけこ",
        "さしすせそ",
        "たちつてと",
        "なにぬねの",
        "はひふへほ",
        "まみむめも",
        "やゆよ",
        "らりるれろ",
        "わをん",
        # --- Voiced (dakuon) ---
        "がぎぐげご",
        "ざじずぜぞ",
        "だぢづでど",
        "ばびぶべぼ",
        # --- Semi-voiced (handakuon) ---
        "ぱぴぷぺぽ",
        # --- Combination sounds (youon) ---
        "きゃきゅきょ",
        "しゃしゅしょ",
        "ちゃちゅちょ",
        "にゃにゅにょ",
        "ひゃひゅひょ",
        "みゃみゅみょ",
        "りゃりゅりょ",
        "ぎゃぎゅぎょ",
        "じゃじゅじょ",
        "びゃびゅびょ",
        "ぴゃぴゅぴょ",
        # --- Double consonants (sokuon) ---
        "がっこう",
        "きって",
        "さっき",
        "はっぱ",
        "いっぱい",
        "こっち",
        "まっすぐ",
        "ざっし",
        # --- Long vowels (chouon) ---
        "おかあさん",
        "おにいさん",
        "くうき",
        "おねえさん",
        "おおきい",
        "とうきょう",
        "おおさか",
        # --- N before various consonants (moraic nasal) ---
        "しんぶん",
        "えんぴつ",
        "かんたん",
        "おんがく",
        "さんぽ",
        "でんわ",
        "にんげん",
        "うんどう",
        "ぜんぶ",
        "しんかんせん",
        # --- Numbers ---
        "一二三四五六七八九十百千万",
        "零",
        "一つ二つ三つ四つ五つ六つ七つ八つ九つ十",
        "一月二月三月四月五月六月七月八月九月十月十一月十二月",
        # --- Katakana (foreign loanwords with special phonemes) ---
        "ティーパーティー",
        "ファイル",
        "フォーク",
        "ディスク",
        "ウォーター",
        "チェック",
        "シェフ",
        "ジェット",
        "フィルム",
        "ヴァイオリン",
        "ヴィジョン",
        "ツァー",
        "トゥーン",
        "デュエット",
        "テュートリアル",
        "フュージョン",
        "ビュッフェ",
        "ミュージック",
        "ニュース",
        "キュート",
        "クォーター",
        "グァム",
        "ウィンドウ",
        "ウェブ",
        "スウェーデン",
        # --- Common words and sentences ---
        "こんにちは",
        "ありがとうございます",
        "おはようございます",
        "こんばんは",
        "さようなら",
        "すみません",
        "わたしはにほんごをべんきょうしています",
        "きょうはいいてんきですね",
        "おなまえはなんですか",
        "どうぞよろしくおねがいします",
        # --- Kanji-heavy sentences (for broader phoneme coverage) ---
        "東京都千代田区",
        "日本語の発音練習",
        "新幹線で大阪まで行きます",
        "図書館で本を読みました",
        "美しい花が咲いている",
        "山の上から海が見える",
        "学校の先生は優しいです",
        "電車に乗って会社に行く",
        "春夏秋冬の四季がある",
        "食事の前に手を洗う",
        "赤い靴を履いている少女",
        "空港に飛行機が着陸した",
        "科学技術の発展は目覚ましい",
        "政治経済の問題について議論する",
        "芸術は心を豊かにする",
        "医者に薬をもらった",
        "彼女は音楽が好きです",
        "犬と猫は仲良しだ",
        "北海道の冬は寒い",
        "沖縄の海は美しい",
        # --- Sentences with diverse phonetic patterns ---
        "ぱっとひらめいた",
        "ぷるぷるのゼリー",
        "ぺんぎんは南極に住む",
        "ぽかぽかと暖かい日",
        "ぴかぴかに光っている",
        "ざわざわと音がする",
        "ずるずるとすべる",
        "ぜんぜん分からない",
        "ぞうは大きい動物です",
        "づらい仕事をしている",
        "ぢめんにすわる",
        # --- Pitch accent variations ---
        "箸と橋と端",
        "雨と飴",
        "花と鼻",
        "酒と鮭",
        "海と膿",
        # --- Onomatopoeia (rich in phoneme variety) ---
        "ごろごろ",
        "ぱちぱち",
        "どきどき",
        "わくわく",
        "きらきら",
        "ふわふわ",
        "もぐもぐ",
        "ぺらぺら",
        "ちくちく",
        "つるつる",
        "ばたばた",
        "ひそひそ",
        "ぬるぬる",
        "すやすや",
        "にこにこ",
        # --- Special phoneme contexts ---
        "ん",  # standalone N
        "んー",  # lengthened N
        "っ",  # standalone sokuon
        # --- Additional combo sounds (less common) ---
        "ふぁふぃふぇふぉ",
        "つぁつぃつぇつぉ",
        "てぃでぃ",
        "とぅどぅ",
        "うぃうぇうぉ",
        # --- Vowel devoicing contexts ---
        "くすり",
        "しつもん",
        "ふくろう",
        "つくえ",
        "ひとつ",
        "ちかてつ",
        # --- Longer passages ---
        "日本は四季のある美しい国です。春には桜が咲き、夏は海水浴を楽しみ、"
        "秋は紅葉が美しく、冬はスキーができます。",
        "東京は世界でも有数の大都市であり、伝統的な文化と現代的な技術が"
        "共存する魅力的な場所です。",
        "富士山は日本で最も高い山で、その美しい姿は多くの芸術作品に"
        "描かれてきました。",
    ]

    all_phonemes = set()

    for text in texts:
        try:
            phonemes = pyopenjtalk.g2p(text, join=False)
            for p in phonemes:
                if p == "sil" or p == "pau":
                    # sil and pau are silence markers; we include pau as it
                    # appears mid-utterance and is meaningful for prosody
                    if p == "pau":
                        all_phonemes.add(f"J_{p}")
                    continue
                all_phonemes.add(f"J_{p}")
        except Exception as ex:
            logger.warning("g2p failed for text %r: %s", text[:30], ex)

    logger.info("Generated %d unique J_-prefixed phonemes from g2p", len(all_phonemes))
    return all_phonemes


# ---------------------------------------------------------------------------
# 3. Scan training manifests for tokens actually used
# ---------------------------------------------------------------------------
def scan_manifest_tokens(path: Path) -> set:
    """Read a lhotse-format cuts JSONL(.gz) file and collect all tokens
    from supervisions."""
    tokens = set()
    if not path.exists():
        logger.warning("Manifest not found at %s, skipping scan", path)
        return tokens

    open_fn = gzip.open if str(path).endswith(".gz") else open

    count = 0
    try:
        with open_fn(path, "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue

                # lhotse CutSet format: supervisions is a list, each has 'tokens'
                supervisions = obj.get("supervisions", [])
                for sup in supervisions:
                    sup_tokens = sup.get("tokens", [])
                    if isinstance(sup_tokens, list):
                        for t in sup_tokens:
                            tokens.add(t)
                count += 1
    except Exception as ex:
        logger.warning("Error reading manifest %s: %s", path, ex)

    logger.info(
        "Scanned %d cuts from %s, found %d unique tokens", count, path, len(tokens)
    )
    return tokens


# ---------------------------------------------------------------------------
# 4. Merge and save
# ---------------------------------------------------------------------------
def merge_and_save(
    existing: OrderedDict,
    ja_phonemes: set,
    manifest_tokens: set,
    output_path: Path,
):
    """Merge existing tokens with new Japanese phonemes and manifest tokens.
    Existing tokens keep their IDs. New tokens get sequential IDs starting
    after the max existing ID."""

    # Start with a copy of existing tokens
    merged: OrderedDict = OrderedDict(existing)

    # Determine next available ID
    if merged:
        next_id = max(merged.values()) + 1
    else:
        # If no existing tokens, start with standard special tokens
        merged["_"] = 0  # padding
        merged["^"] = 1  # start of sentence
        merged["$"] = 2  # end of sentence
        merged[" "] = 3  # space
        next_id = 4

    # Collect all new tokens (not in existing)
    new_tokens = set()

    # Add J_ phonemes from g2p
    for t in sorted(ja_phonemes):
        if t not in merged:
            new_tokens.add(t)

    # Add manifest tokens that are missing
    for t in sorted(manifest_tokens):
        if t not in merged:
            new_tokens.add(t)

    # Sort new tokens for deterministic output
    # J_ tokens sorted together, others sorted separately
    j_tokens = sorted(t for t in new_tokens if t.startswith("J_"))
    other_tokens = sorted(t for t in new_tokens if not t.startswith("J_"))

    # Add other tokens first, then J_ tokens
    for t in other_tokens:
        merged[t] = next_id
        next_id += 1

    for t in j_tokens:
        merged[t] = next_id
        next_id += 1

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for token, tid in merged.items():
            f.write(f"{token}\t{tid}\n")

    # Summary
    n_existing = len(existing)
    n_new_j = len(j_tokens)
    n_new_other = len(other_tokens)
    logger.info(
        "Saved %d tokens to %s (existing: %d, new J_: %d, new other: %d)",
        len(merged),
        output_path,
        n_existing,
        n_new_j,
        n_new_other,
    )

    # Print the J_ tokens for inspection
    logger.info("--- J_ tokens (%d) ---", n_new_j + sum(1 for t in existing if t.startswith("J_")))
    all_j = sorted(t for t in merged if t.startswith("J_"))
    for t in all_j:
        logger.info("  %s\t%d", t, merged[t])

    return merged


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    logger.info("=== LuxTTS Japanese Token Generator ===")
    logger.info("Project root: %s", PROJECT_ROOT)

    # 1. Read existing tokens (wait briefly if file might still be downloading)
    if not PRETRAINED_TOKENS.exists():
        logger.info(
            "Pretrained tokens not found at %s, will generate standalone file",
            PRETRAINED_TOKENS,
        )
    existing_tokens = read_existing_tokens(PRETRAINED_TOKENS)

    # 2. Generate Japanese phonemes
    logger.info("Generating Japanese phonemes via pyopenjtalk.g2p()...")
    ja_phonemes = generate_japanese_phonemes()

    # 3. Scan training manifests
    logger.info("Scanning training manifest for used tokens...")
    manifest_tokens = set()
    if TRAIN_MANIFEST.exists():
        manifest_tokens = scan_manifest_tokens(TRAIN_MANIFEST)
    else:
        logger.info("Training manifest not found at %s, skipping", TRAIN_MANIFEST)

    # Also check dev manifest
    dev_manifest = PROJECT_ROOT / "data" / "fbank" / "custom_cuts_dev_tokens.jsonl.gz"
    if dev_manifest.exists():
        dev_tokens = scan_manifest_tokens(dev_manifest)
        manifest_tokens |= dev_tokens

    # 4. Merge and save
    merged = merge_and_save(existing_tokens, ja_phonemes, manifest_tokens, OUTPUT_TOKENS)

    logger.info("=== Done! Total tokens: %d ===", len(merged))
    logger.info("Output: %s", OUTPUT_TOKENS)


if __name__ == "__main__":
    main()
