#!/usr/bin/env python3
"""Expand ZipVoiceDistill embedding layer for Japanese tokens."""
import torch
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/pretrained/model.pt")
    parser.add_argument("--output", default="data/pretrained/model_ja_distill.pt")
    parser.add_argument("--new-vocab-size", type=int, default=401)
    args = parser.parse_args()

    ckpt = torch.load(args.input, map_location="cpu", weights_only=False)

    old_embed = ckpt["model"]["embed.weight"]
    old_vocab, embed_dim = old_embed.shape
    new_vocab = args.new_vocab_size

    print(f"Old vocab: {old_vocab}, New vocab: {new_vocab}, Embed dim: {embed_dim}")

    new_embed = torch.zeros(new_vocab, embed_dim)
    new_embed[:old_vocab] = old_embed
    mean_embed = old_embed.mean(dim=0)
    for i in range(old_vocab, new_vocab):
        new_embed[i] = mean_embed + torch.randn(embed_dim) * 0.01

    ckpt["model"]["embed.weight"] = new_embed
    torch.save(ckpt, args.output)
    print(f"Saved expanded model to {args.output}")

if __name__ == "__main__":
    main()
