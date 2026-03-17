# Copyright    2024-2025  Xiaomi Corp.        (authors: Wei Kang, Han Zhu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
TtsDataModule: data loading and batching for TTS training.

Provides lazy CutSet loading, dynamic bucketing for training, and
simple sequential sampling for validation.
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Union

import torch
from lhotse import CutSet
from lhotse.dataset import DynamicBucketingSampler, SimpleCutSampler
from torch.utils.data import DataLoader, Dataset


class TtsDataset(Dataset):
    """A simple map-style dataset that converts lhotse CutSets into
    batch dicts expected by the TTS training loop.

    Each batch contains:
        - "text": list of raw text strings (one per cut).
        - "tokens": list of token-id lists (one per cut, from
          ``supervision.tokens`` when available).
        - "features": zero-padded feature tensor of shape ``(B, T, F)``.
        - "features_lens": integer tensor of shape ``(B,)`` with the
          original (unpadded) frame counts.
    """

    def __getitem__(self, cuts: CutSet) -> Dict[str, Union[List, torch.Tensor]]:
        cuts = sorted(cuts, key=lambda c: c.num_frames, reverse=True)

        texts: List[str] = []
        tokens: List[List[int]] = []
        features_list: List[torch.Tensor] = []

        for cut in cuts:
            # Load pre-computed features (e.g. VocosFbank, shape (T, 100))
            feats = torch.from_numpy(cut.load_features())
            features_list.append(feats)

            sup = cut.supervisions[0]
            texts.append(sup.text)

            if hasattr(sup, "tokens") and sup.tokens is not None:
                tokens.append(sup.tokens)
            else:
                # Fallback: empty list; the training script will
                # tokenize on-the-fly via CutSet.map(tokenize_text).
                tokens.append([])

        # Pad features to the longest sequence in the batch
        max_frames = max(f.shape[0] for f in features_list)
        feat_dim = features_list[0].shape[1]

        padded = torch.zeros(len(features_list), max_frames, feat_dim)
        lengths = torch.zeros(len(features_list), dtype=torch.int64)

        for i, feats in enumerate(features_list):
            t = feats.shape[0]
            padded[i, :t, :] = feats
            lengths[i] = t

        return {
            "text": texts,
            "tokens": tokens,
            "features": padded,
            "features_lens": lengths,
        }


class TtsDataModule:
    """Data module used by all ZipVoice training scripts.

    Handles:
        - CLI argument registration (``add_arguments``).
        - Lazy loading of lhotse CutSets from ``manifest_dir``.
        - Construction of training / validation ``DataLoader`` instances
          with appropriate samplers.
    """

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        group = parser.add_argument_group(
            title="TtsDataModule options",
            description="Options related to the TTS data pipeline.",
        )
        group.add_argument(
            "--manifest-dir",
            type=str,
            default="data/fbank",
            help="Directory containing lhotse cut manifests (.jsonl.gz).",
        )
        group.add_argument(
            "--max-duration",
            type=float,
            default=500.0,
            help="Maximum total audio duration (in seconds) per batch.",
        )
        group.add_argument(
            "--num-workers",
            type=int,
            default=8,
            help="Number of DataLoader worker processes.",
        )

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.manifest_dir = Path(args.manifest_dir)

    # ------------------------------------------------------------------
    # Emilia cuts
    # ------------------------------------------------------------------

    def train_emilia_EN_cuts(self) -> CutSet:
        logging.info("Loading train emilia EN cuts (lazy)")
        return CutSet.from_jsonl_lazy(
            self.manifest_dir / "emilia_cuts_train_en.jsonl.gz"
        )

    def train_emilia_ZH_cuts(self) -> CutSet:
        logging.info("Loading train emilia ZH cuts (lazy)")
        return CutSet.from_jsonl_lazy(
            self.manifest_dir / "emilia_cuts_train_zh.jsonl.gz"
        )

    def dev_emilia_EN_cuts(self) -> CutSet:
        logging.info("Loading dev emilia EN cuts (lazy)")
        return CutSet.from_jsonl_lazy(
            self.manifest_dir / "emilia_cuts_dev_en.jsonl.gz"
        )

    def dev_emilia_ZH_cuts(self) -> CutSet:
        logging.info("Loading dev emilia ZH cuts (lazy)")
        return CutSet.from_jsonl_lazy(
            self.manifest_dir / "emilia_cuts_dev_zh.jsonl.gz"
        )

    # ------------------------------------------------------------------
    # LibriTTS cuts
    # ------------------------------------------------------------------

    def train_libritts_cuts(self) -> CutSet:
        logging.info("Loading train LibriTTS cuts (lazy)")
        return CutSet.from_jsonl_lazy(
            self.manifest_dir / "libritts_cuts_train.jsonl.gz"
        )

    def dev_libritts_cuts(self) -> CutSet:
        logging.info("Loading dev LibriTTS cuts (lazy)")
        return CutSet.from_jsonl_lazy(
            self.manifest_dir / "libritts_cuts_dev.jsonl.gz"
        )

    # ------------------------------------------------------------------
    # OpenDialog cuts
    # ------------------------------------------------------------------

    def train_opendialog_en_cuts(self) -> CutSet:
        logging.info("Loading train OpenDialog EN cuts (lazy)")
        return CutSet.from_jsonl_lazy(
            self.manifest_dir / "opendialog_cuts_train_en.jsonl.gz"
        )

    def train_opendialog_zh_cuts(self) -> CutSet:
        logging.info("Loading train OpenDialog ZH cuts (lazy)")
        return CutSet.from_jsonl_lazy(
            self.manifest_dir / "opendialog_cuts_train_zh.jsonl.gz"
        )

    def dev_opendialog_en_cuts(self) -> CutSet:
        logging.info("Loading dev OpenDialog EN cuts (lazy)")
        return CutSet.from_jsonl_lazy(
            self.manifest_dir / "opendialog_cuts_dev_en.jsonl.gz"
        )

    def dev_opendialog_zh_cuts(self) -> CutSet:
        logging.info("Loading dev OpenDialog ZH cuts (lazy)")
        return CutSet.from_jsonl_lazy(
            self.manifest_dir / "opendialog_cuts_dev_zh.jsonl.gz"
        )

    # ------------------------------------------------------------------
    # Custom cuts
    # ------------------------------------------------------------------

    def train_custom_cuts(self, manifest_path: str) -> CutSet:
        logging.info(f"Loading train custom cuts from {manifest_path} (lazy)")
        return CutSet.from_jsonl_lazy(manifest_path)

    def dev_custom_cuts(self, manifest_path: str) -> CutSet:
        logging.info(f"Loading dev custom cuts from {manifest_path} (lazy)")
        return CutSet.from_jsonl_lazy(manifest_path)

    # ------------------------------------------------------------------
    # DataLoaders
    # ------------------------------------------------------------------

    def train_dataloaders(self, cuts: CutSet) -> DataLoader:
        """Build a training DataLoader with dynamic bucketing."""
        sampler = DynamicBucketingSampler(
            cuts,
            max_duration=self.args.max_duration,
            shuffle=True,
            drop_last=True,
        )
        dataset = TtsDataset()
        dl = DataLoader(
            dataset,
            sampler=sampler,
            batch_size=None,
            num_workers=self.args.num_workers,
            persistent_workers=(self.args.num_workers > 0),
            pin_memory=True,
            prefetch_factor=4 if self.args.num_workers > 0 else None,
        )
        return dl

    def dev_dataloaders(self, cuts: CutSet) -> DataLoader:
        """Build a validation DataLoader with simple sequential sampling."""
        sampler = SimpleCutSampler(
            cuts,
            max_duration=self.args.max_duration,
            shuffle=False,
        )
        dataset = TtsDataset()
        dl = DataLoader(
            dataset,
            sampler=sampler,
            batch_size=None,
            num_workers=self.args.num_workers,
            persistent_workers=(self.args.num_workers > 0),
            pin_memory=True,
            prefetch_factor=4 if self.args.num_workers > 0 else None,
        )
        return dl
