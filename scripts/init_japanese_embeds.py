#!/usr/bin/env python3
"""Initialize Japanese (J_) token embeddings from similar English IPA phonemes.

Instead of random noise (as expand_distill_model.py does), this script copies
embedding vectors from acoustically similar English/IPA phonemes in the
pretrained model, giving the fine-tuning a much better starting point.

Mapping strategy:
  - Direct mappings: J_a -> 'ɑ' (AA), J_i -> 'iː' (IY), etc.
  - Palatalized consonants (ky, gy, ...): base consonant embed + small noise.
  - Unmapped tokens: base consonant embed + small perturbation.

Usage:
    python -m scripts.init_japanese_embeds \
        --input  data/pretrained/model.pt \
        --output data/pretrained/model_ja_distill_v3.pt \
        --tokens data/pretrained/tokens.txt
"""

import argparse
import logging
import sys
from pathlib import Path

import torch

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    level=logging.INFO,
    force=True,
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Token file utilities
# ---------------------------------------------------------------------------

def load_token2id(path: str) -> dict:
    """Load a tokens.txt file and return a token -> id mapping."""
    token2id = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 2:
                continue
            token, tid = parts[0], int(parts[1])
            token2id[token] = tid
    return token2id


# ---------------------------------------------------------------------------
# Japanese-to-IPA phoneme mapping
# ---------------------------------------------------------------------------
# The keys are J_-prefixed Japanese phoneme tokens (as produced by pyopenjtalk).
# The values are espeak IPA tokens that exist in the pretrained vocabulary.
#
# Phonetic rationale for each mapping:
#   Vowels:
#     J_a  -> ɑ   (open back unrounded, closest to Japanese /a/)
#     J_i  -> iː  simulated as 'i' (close front unrounded)
#     J_u  -> ɯ   (close back unrounded, Japanese /u/ is unrounded)
#     J_e  -> ɛ   (open-mid front unrounded, close to Japanese /e/)
#     J_o  -> o   (close-mid back rounded)
#   Consonants:
#     J_k  -> k   (voiceless velar plosive)
#     J_g  -> ɡ   (voiced velar plosive -- note: IPA ɡ, not ASCII g)
#     J_s  -> s   (voiceless alveolar fricative)
#     J_z  -> z   (voiced alveolar fricative)
#     J_t  -> t   (voiceless alveolar plosive)
#     J_d  -> d   (voiced alveolar plosive)
#     J_n  -> n   (alveolar nasal)
#     J_h  -> h   (voiceless glottal fricative)
#     J_b  -> b   (voiced bilabial plosive)
#     J_p  -> p   (voiceless bilabial plosive)
#     J_m  -> m   (bilabial nasal)
#     J_r  -> ɾ   (alveolar tap -- Japanese /r/ is a tap, not English /ɹ/)
#     J_w  -> w   (labial-velar approximant)
#     J_y  -> j   (palatal approximant)
#   Affricates / Fricatives:
#     J_sh -> ʃ   (voiceless postalveolar fricative)
#     J_ch -> tʃ  simulated as 'ʃ' (closest single token)
#     J_ts -> t   (voiceless alveolar plosive -- first element of /ts/)
#     J_j  -> ʒ   (voiced postalveolar fricative -- closest to Japanese /dʒ/)
#     J_f  -> ɸ   (voiceless bilabial fricative -- Japanese /f/ is bilabial)
#   Special:
#     J_N  -> ŋ   (velar nasal -- moraic nasal often surfaces as [ŋ])
#     J_cl -> ʔ   (glottal stop -- geminate closure)
#     J_pau -> _  (padding/silence)
#   Devoiced vowels:
#     J_I  -> ɪ   (near-close near-front unrounded -- devoiced /i/)
#     J_U  -> ʊ   (near-close near-back rounded -- devoiced /u/)

DIRECT_MAPPING = {
    # Vowels
    "J_a":   "ɑ",
    "J_i":   "i",
    "J_u":   "ɯ",
    "J_e":   "ɛ",
    "J_o":   "o",
    # Consonants
    "J_k":   "k",
    "J_g":   "ɡ",   # IPA ɡ (U+0261), not ASCII g
    "J_s":   "s",
    "J_z":   "z",
    "J_t":   "t",
    "J_d":   "d",
    "J_n":   "n",
    "J_h":   "h",
    "J_b":   "b",
    "J_p":   "p",
    "J_m":   "m",
    "J_r":   "ɾ",   # alveolar tap
    "J_w":   "w",
    "J_y":   "j",   # palatal approximant
    # Affricates and fricatives
    "J_sh":  "ʃ",
    "J_ch":  "ʃ",   # tʃ approximated by ʃ
    "J_ts":  "t",   # ts approximated by t
    "J_j":   "ʒ",   # voiced postalveolar fricative
    "J_f":   "ɸ",   # bilabial fricative (Japanese fu)
    # Special
    "J_N":   "ŋ",   # moraic nasal
    "J_cl":  "ʔ",   # geminate closure ~ glottal stop
    "J_pau": "_",   # pause ~ padding
    # Devoiced vowels
    "J_I":   "ɪ",   # devoiced /i/
    "J_U":   "ʊ",   # devoiced /u/
}

# Palatalized consonants: use the base consonant + small perturbation.
# Format: J_Xy -> base consonant token for X
PALATALIZED_BASE = {
    "J_ky":  "k",
    "J_gy":  "ɡ",
    "J_hy":  "h",
    "J_by":  "b",
    "J_my":  "m",
    "J_ny":  "n",
    "J_ry":  "ɾ",
    "J_py":  "p",
    "J_ty":  "t",
    "J_dy":  "d",
    "J_fy":  "ɸ",
    # J_kw: labialized velar
    "J_kw":  "k",
}


def get_source_id(token2id: dict, ipa_token: str) -> int:
    """Look up the source IPA token in the vocabulary.

    Returns the token id, or -1 if not found.
    """
    if ipa_token in token2id:
        return token2id[ipa_token]
    # Fallback: try ASCII 'g' for ɡ (some token files use ASCII g)
    if ipa_token == "ɡ" and "g" in token2id:
        return token2id["g"]
    return -1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Initialize J_ embeddings from similar English IPA phonemes."
    )
    parser.add_argument(
        "--input",
        default="data/pretrained/model.pt",
        help="Path to the pretrained checkpoint (original vocab).",
    )
    parser.add_argument(
        "--output",
        default="data/pretrained/model_ja_distill_v3.pt",
        help="Path to save the new checkpoint with initialized J_ embeddings.",
    )
    parser.add_argument(
        "--tokens",
        default="data/pretrained/tokens.txt",
        help="Path to the (expanded) tokens.txt with J_ tokens.",
    )
    parser.add_argument(
        "--new-vocab-size",
        type=int,
        default=401,
        help="New vocabulary size (must cover all J_ token indices).",
    )
    parser.add_argument(
        "--noise-scale",
        type=float,
        default=0.01,
        help="Scale of Gaussian noise added to palatalized/fallback embeddings.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    # Load token vocabulary
    token2id = load_token2id(args.tokens)
    logger.info("Loaded %d tokens from %s", len(token2id), args.tokens)

    # Identify J_ tokens
    ja_tokens = {t: tid for t, tid in token2id.items() if t.startswith("J_")}
    logger.info("Found %d J_ tokens (indices %d-%d)",
                len(ja_tokens),
                min(ja_tokens.values()) if ja_tokens else -1,
                max(ja_tokens.values()) if ja_tokens else -1)

    # Load checkpoint
    logger.info("Loading checkpoint from %s ...", args.input)
    ckpt = torch.load(args.input, map_location="cpu", weights_only=False)

    old_embed = ckpt["model"]["embed.weight"]
    old_vocab, embed_dim = old_embed.shape
    new_vocab = args.new_vocab_size

    logger.info("Old vocab: %d, New vocab: %d, Embed dim: %d",
                old_vocab, new_vocab, embed_dim)

    if new_vocab <= old_vocab:
        logger.warning("New vocab size (%d) <= old vocab size (%d). "
                       "No expansion needed, but will still initialize J_ tokens "
                       "if they fall within existing range.", new_vocab, old_vocab)

    # Create expanded embedding
    new_embed = torch.zeros(new_vocab, embed_dim)
    new_embed[:old_vocab] = old_embed

    # Compute mean embedding as ultimate fallback
    mean_embed = old_embed.mean(dim=0)

    # Also get the palatal approximant embedding for blending with palatalized consonants
    j_id = get_source_id(token2id, "j")  # IPA /j/ = palatal approximant
    j_embed = old_embed[j_id] if j_id >= 0 and j_id < old_vocab else mean_embed

    # Statistics
    stats = {"direct": 0, "palatalized": 0, "fallback_mean": 0}

    for ja_tok, ja_id in sorted(ja_tokens.items(), key=lambda x: x[1]):
        if ja_id >= new_vocab:
            logger.error("J_ token '%s' has id %d >= new_vocab %d, skipping!",
                         ja_tok, ja_id, new_vocab)
            continue

        # Strategy 1: Direct mapping
        if ja_tok in DIRECT_MAPPING:
            source_ipa = DIRECT_MAPPING[ja_tok]
            source_id = get_source_id(token2id, source_ipa)
            if source_id >= 0 and source_id < old_vocab:
                new_embed[ja_id] = old_embed[source_id].clone()
                stats["direct"] += 1
                logger.info("  %s (id=%d) <- '%s' (id=%d) [direct]",
                            ja_tok, ja_id, source_ipa, source_id)
                continue
            else:
                logger.warning("  %s: source IPA '%s' not found in vocab (id=%d), "
                               "falling through to palatalized/fallback.",
                               ja_tok, source_ipa, source_id)

        # Strategy 2: Palatalized consonant -> base consonant + palatal blend + noise
        if ja_tok in PALATALIZED_BASE:
            base_ipa = PALATALIZED_BASE[ja_tok]
            base_id = get_source_id(token2id, base_ipa)
            if base_id >= 0 and base_id < old_vocab:
                # Blend: 70% base consonant + 30% palatal approximant /j/
                blended = 0.7 * old_embed[base_id] + 0.3 * j_embed
                new_embed[ja_id] = blended + torch.randn(embed_dim) * args.noise_scale
                stats["palatalized"] += 1
                logger.info("  %s (id=%d) <- '%s' (id=%d) + /j/ blend [palatalized]",
                            ja_tok, ja_id, base_ipa, base_id)
                continue
            else:
                logger.warning("  %s: base IPA '%s' not found for palatalized init.",
                               ja_tok, base_ipa)

        # Strategy 3: Fallback to mean embedding + noise
        new_embed[ja_id] = mean_embed + torch.randn(embed_dim) * args.noise_scale
        stats["fallback_mean"] += 1
        logger.info("  %s (id=%d) <- mean + noise [fallback]",
                    ja_tok, ja_id)

    # Fill any remaining new indices (between old_vocab and new_vocab) that
    # are NOT J_ tokens with mean + noise (safety net)
    ja_ids = set(ja_tokens.values())
    for idx in range(old_vocab, new_vocab):
        if idx not in ja_ids:
            new_embed[idx] = mean_embed + torch.randn(embed_dim) * args.noise_scale

    # Save
    ckpt["model"]["embed.weight"] = new_embed
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, str(output_path))

    logger.info("=== Summary ===")
    logger.info("  Direct IPA mapping:     %d tokens", stats["direct"])
    logger.info("  Palatalized (blended):  %d tokens", stats["palatalized"])
    logger.info("  Fallback (mean+noise):  %d tokens", stats["fallback_mean"])
    logger.info("  Total J_ tokens:        %d", len(ja_tokens))
    logger.info("Saved to %s", output_path)
    logger.info("Embedding shape: %s", list(new_embed.shape))


if __name__ == "__main__":
    main()
