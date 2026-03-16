"""Tests for Japanese g2p support in LuxTTS tokenizer.

Covers:
- JapaneseTextNormalizer (full-width to half-width conversion)
- Character classification helpers (is_hiragana, is_katakana, is_japanese)
- get_segment() language detection with and without lang="ja"
- tokenize_JA() phoneme generation (requires pyopenjtalk-plus)
- texts_to_tokens() integration
"""

import pytest

from zipvoice.tokenizer.normalizer import JapaneseTextNormalizer
from zipvoice.tokenizer.tokenizer import EmiliaTokenizer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def normalizer():
    return JapaneseTextNormalizer()


@pytest.fixture
def tokenizer():
    """EmiliaTokenizer without token_file (token-to-id mapping not needed)."""
    return EmiliaTokenizer(token_file=None)


@pytest.fixture
def tokenizer_ja():
    """EmiliaTokenizer with lang='ja' so CJK chars are classified as Japanese."""
    return EmiliaTokenizer(token_file=None, lang="ja")


# ---------------------------------------------------------------------------
# 1. JapaneseTextNormalizer
# ---------------------------------------------------------------------------

class TestJapaneseTextNormalizer:
    def test_fullwidth_digits_to_halfwidth(self, normalizer):
        assert normalizer.normalize("０１２") == "012"

    def test_fullwidth_alpha_to_halfwidth(self, normalizer):
        assert normalizer.normalize("ＡＢＣ") == "ABC"

    def test_fullwidth_lower_alpha_to_halfwidth(self, normalizer):
        assert normalizer.normalize("ａｂｃ") == "abc"

    def test_plain_text_passthrough(self, normalizer):
        assert normalizer.normalize("こんにちは") == "こんにちは"

    def test_mixed_fullwidth_and_normal(self, normalizer):
        assert normalizer.normalize("テスト１２３abc") == "テスト123abc"

    def test_empty_string(self, normalizer):
        assert normalizer.normalize("") == ""

    def test_halfwidth_katakana_to_fullwidth(self, normalizer):
        assert normalizer.normalize("ｱｲｳｴｵ") == "アイウエオ"

    def test_fullwidth_space(self, normalizer):
        assert normalizer.normalize("テスト\u3000テスト") == "テスト テスト"

    def test_idempotent(self, normalizer):
        assert normalizer.normalize("abc123") == "abc123"


# ---------------------------------------------------------------------------
# 2. is_hiragana()
# ---------------------------------------------------------------------------

class TestIsHiragana:
    @pytest.mark.parametrize("char", ["あ", "い", "う", "ん", "ゔ"])
    def test_hiragana_returns_true(self, tokenizer, char):
        assert tokenizer.is_hiragana(char) is True

    @pytest.mark.parametrize("char", ["ア", "漢", "A", "1"])
    def test_non_hiragana_returns_false(self, tokenizer, char):
        assert tokenizer.is_hiragana(char) is False


# ---------------------------------------------------------------------------
# 3. is_katakana()
# ---------------------------------------------------------------------------

class TestIsKatakana:
    @pytest.mark.parametrize("char", ["ア", "イ", "ウ", "ン", "ヴ"])
    def test_katakana_returns_true(self, tokenizer, char):
        assert tokenizer.is_katakana(char) is True

    @pytest.mark.parametrize("char", ["あ", "漢", "A", "1"])
    def test_non_katakana_returns_false(self, tokenizer, char):
        assert tokenizer.is_katakana(char) is False


# ---------------------------------------------------------------------------
# 4. is_japanese()
# ---------------------------------------------------------------------------

class TestIsJapanese:
    @pytest.mark.parametrize("char", ["あ", "い", "ア", "カ"])
    def test_hiragana_and_katakana_return_true(self, tokenizer, char):
        assert tokenizer.is_japanese(char) is True

    @pytest.mark.parametrize("char", ["漢", "A", "1", ","])
    def test_kanji_and_ascii_return_false(self, tokenizer, char):
        assert tokenizer.is_japanese(char) is False

    @pytest.mark.parametrize("char", ["ｱ", "ｶ", "ﾝ"])
    def test_halfwidth_katakana_returns_true(self, tokenizer, char):
        assert tokenizer.is_japanese(char) is True

    @pytest.mark.parametrize("char", ["\u31f0", "\u31f5", "\u31ff"])
    def test_katakana_extension_returns_true(self, tokenizer, char):
        assert tokenizer.is_japanese(char) is True


# ---------------------------------------------------------------------------
# 5. get_segment() -- default lang=None
# ---------------------------------------------------------------------------

class TestGetSegmentDefault:
    def test_pure_hiragana(self, tokenizer):
        segments = tokenizer.get_segment("こんにちは")
        langs = [seg[1] for seg in segments]
        assert "ja" in langs
        # All content should be classified as Japanese
        for seg_text, seg_lang in segments:
            if seg_lang != "other":
                assert seg_lang == "ja"

    def test_english_text(self, tokenizer):
        segments = tokenizer.get_segment("Hello")
        langs = [seg[1] for seg in segments]
        assert "en" in langs

    def test_kanji_defaults_to_zh(self, tokenizer):
        """Without lang='ja', CJK ideographs default to 'zh'."""
        segments = tokenizer.get_segment("漢字")
        langs = [seg[1] for seg in segments]
        assert "zh" in langs
        assert "ja" not in langs


# ---------------------------------------------------------------------------
# 6. get_segment() -- lang="ja"
# ---------------------------------------------------------------------------

class TestGetSegmentLangJa:
    def test_kanji_becomes_ja(self, tokenizer_ja):
        """With lang='ja', CJK ideographs are classified as 'ja'."""
        segments = tokenizer_ja.get_segment("漢字")
        langs = [seg[1] for seg in segments]
        assert "ja" in langs
        assert "zh" not in langs

    def test_kanji_hiragana_mixed(self, tokenizer_ja):
        """Kanji + hiragana should stay in a single 'ja' segment."""
        segments = tokenizer_ja.get_segment("東京はいい")
        langs = [seg[1] for seg in segments]
        assert all(l in ("ja", "other") for l in langs)

    def test_three_way_mixed(self, tokenizer_ja):
        """Japanese + English + CJK text should produce ja and en segments."""
        segments = tokenizer_ja.get_segment("こんにちはHello漢字")
        langs = [seg[1] for seg in segments]
        assert "ja" in langs
        assert "en" in langs

    def test_numbers_with_japanese(self, tokenizer_ja):
        """Numbers mixed with Japanese text like '3月15日'."""
        segments = tokenizer_ja.get_segment("3月15日")
        langs = [seg[1] for seg in segments]
        assert "ja" in langs


# ---------------------------------------------------------------------------
# 7. tokenize_JA() -- requires pyopenjtalk-plus
# ---------------------------------------------------------------------------

class TestTokenizeJA:
    @pytest.fixture(autouse=True)
    def _require_pyopenjtalk(self):
        pytest.importorskip(
            "pyopenjtalk",
            reason="pyopenjtalk-plus is not installed; skipping Japanese g2p tests",
        )

    def test_hiragana_produces_j_prefix_phonemes(self, tokenizer):
        phonemes = tokenizer.tokenize_JA("こんにちは")
        assert len(phonemes) > 0
        assert all(p.startswith("J_") for p in phonemes)

    def test_kanji_with_lang_ja(self, tokenizer_ja):
        phonemes = tokenizer_ja.tokenize_JA("東京は大きい")
        assert len(phonemes) > 0
        assert all(p.startswith("J_") for p in phonemes)

    def test_empty_string(self, tokenizer):
        phonemes = tokenizer.tokenize_JA("")
        assert phonemes == []

    def test_sil_tokens_removed(self, tokenizer):
        """pyopenjtalk often inserts 'sil' at boundaries; tokenize_JA should drop them."""
        phonemes = tokenizer.tokenize_JA("こんにちは")
        assert "J_sil" not in phonemes
        assert "sil" not in phonemes

    def test_katakana_input(self, tokenizer):
        phonemes = tokenizer.tokenize_JA("カタカナ")
        assert len(phonemes) > 0
        assert all(p.startswith("J_") for p in phonemes)

    def test_mixed_hiragana_katakana(self, tokenizer):
        phonemes = tokenizer.tokenize_JA("こんにちはカタカナ")
        assert len(phonemes) > 0
        assert all(p.startswith("J_") for p in phonemes)

    def test_numbers_in_japanese(self, tokenizer):
        phonemes = tokenizer.tokenize_JA("3つのりんご")
        assert len(phonemes) > 0

    def test_japanese_punctuation(self, tokenizer):
        phonemes = tokenizer.tokenize_JA("こんにちは。")
        assert len(phonemes) > 0
        assert all(p.startswith("J_") for p in phonemes)


# ---------------------------------------------------------------------------
# 8. Half-width katakana segmentation
# ---------------------------------------------------------------------------

class TestHalfwidthKatakana:
    def test_halfwidth_katakana_segmented_as_japanese(self, tokenizer_ja):
        segments = tokenizer_ja.get_segment("ｶﾀｶﾅ")
        langs = [seg[1] for seg in segments]
        assert "ja" in langs


# ---------------------------------------------------------------------------
# 9. texts_to_tokens() -- integration
# ---------------------------------------------------------------------------

class TestTextsToTokensIntegration:
    @pytest.fixture(autouse=True)
    def _require_pyopenjtalk(self):
        pytest.importorskip(
            "pyopenjtalk",
            reason="pyopenjtalk-plus is not installed; skipping integration tests",
        )

    def test_japanese_text_routes_to_ja(self, tokenizer_ja):
        """Japanese text with lang='ja' should produce J_-prefixed phonemes."""
        tokens = tokenizer_ja.texts_to_tokens(["こんにちは"])
        assert len(tokens) == 1
        assert len(tokens[0]) > 0
        assert all(t.startswith("J_") for t in tokens[0])

    def test_mixed_japanese_english(self, tokenizer_ja):
        """Mixed text should produce both J_-prefixed and non-prefixed phonemes."""
        tokens = tokenizer_ja.texts_to_tokens(["こんにちは,Hello"])
        assert len(tokens) == 1
        phonemes = tokens[0]
        assert len(phonemes) > 0
        has_ja = any(t.startswith("J_") for t in phonemes)
        has_en = any(not t.startswith("J_") for t in phonemes)
        assert has_ja, "Expected Japanese phonemes (J_ prefix) in output"
        assert has_en, "Expected English phonemes in output"
