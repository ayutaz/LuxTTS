#!/usr/bin/env python3
"""
Convert MoeSpeech LJSpeech-format dataset to LuxTTS TSV format.

MoeSpeech metadata.csv format (pipe-delimited):
    {speaker_id}_{utterance_id}|{speaker_id}|{transcription_text}

LuxTTS TSV format (tab-delimited):
    {uniq_id}\t{text}\t{wav_path}

This script reads the metadata, verifies WAV files exist, resamples audio
from the source sample rate (default 22050 Hz) to a target rate (default
24000 Hz), splits into train/dev sets stratified by speaker, and writes
the output TSV files.

Usage:
    python scripts/convert_moe_speech.py \
        --input-dir data/moe-speech-20speakers-ljspeech \
        --output-dir data/moe-speech-resampled \
        --train-tsv data/custom_train.tsv \
        --dev-tsv data/custom_dev.tsv \
        --sampling-rate 24000 \
        --source-rate 22050 \
        --dev-ratio 0.05 \
        --num-workers 4
"""

import argparse
import logging
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import soundfile as sf
import torch
import torchaudio
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert MoeSpeech LJSpeech-format dataset to LuxTTS TSV format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/moe-speech-20speakers-ljspeech"),
        help="Path to the MoeSpeech dataset directory containing metadata.csv and wavs/.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/moe-speech-resampled"),
        help="Directory where resampled WAV files will be saved.",
    )
    parser.add_argument(
        "--train-tsv",
        type=Path,
        default=Path("data/custom_train.tsv"),
        help="Output path for the training TSV file.",
    )
    parser.add_argument(
        "--dev-tsv",
        type=Path,
        default=Path("data/custom_dev.tsv"),
        help="Output path for the dev/validation TSV file.",
    )
    parser.add_argument(
        "--sampling-rate",
        type=int,
        default=24000,
        help="Target sampling rate for resampled audio (default: 24000).",
    )
    parser.add_argument(
        "--source-rate",
        type=int,
        default=22050,
        help="Source sampling rate of the input audio (default: 22050).",
    )
    parser.add_argument(
        "--dev-ratio",
        type=float,
        default=0.05,
        help="Fraction of data to hold out for dev set (default: 0.05).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of parallel workers for resampling (default: 4).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible train/dev split (default: 42).",
    )
    return parser.parse_args()


def parse_metadata(metadata_path: Path) -> List[Dict[str, str]]:
    """
    Parse the pipe-delimited metadata.csv file.

    Expected format per line:
        {speaker_id}_{utterance_id}|{speaker_id}|{transcription_text}

    Returns a list of dicts with keys: filename, speaker_id, text.
    """
    entries = []
    with open(metadata_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.split("|")
            if len(parts) < 3:
                logger.warning(
                    "Skipping malformed line %d: expected 3 pipe-delimited fields, "
                    "got %d: %r",
                    line_num,
                    len(parts),
                    line,
                )
                continue
            # Text field may itself contain pipes, so rejoin everything after
            # the second delimiter.
            filename = parts[0].strip()
            speaker_id = parts[1].strip()
            text = "|".join(parts[2:]).strip()
            if not filename or not text:
                logger.warning(
                    "Skipping line %d with empty filename or text: %r",
                    line_num,
                    line,
                )
                continue
            entries.append(
                {
                    "filename": filename,
                    "speaker_id": speaker_id,
                    "text": text,
                }
            )
    return entries


def resample_single(
    src_path: Path,
    dst_path: Path,
    source_rate: int,
    target_rate: int,
) -> Optional[Tuple[str, float]]:
    """
    Resample a single WAV file and save to dst_path.

    Returns (dst_path_str, duration_seconds) on success, or None on failure.
    """
    try:
        data, sr = sf.read(str(src_path), dtype="float32")

        if sr != source_rate:
            logger.warning(
                "File %s has sample rate %d (expected %d); resampling from actual rate.",
                src_path.name,
                sr,
                source_rate,
            )
            source_rate = sr

        # Convert to torch tensor (channels, samples)
        waveform = torch.from_numpy(data)
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        else:
            waveform = waveform.T  # (samples, channels) -> (channels, samples)

        # Convert to mono if multi-channel
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        if source_rate != target_rate:
            resampler = torchaudio.transforms.Resample(
                orig_freq=source_rate, new_freq=target_rate
            )
            waveform = resampler(waveform)

        dst_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(dst_path), waveform.squeeze(0).numpy(), target_rate)

        duration = waveform.shape[1] / target_rate
        return str(dst_path), duration

    except Exception as e:
        logger.warning("Failed to process %s: %s", src_path, e)
        return None


def stratified_split(
    entries: List[Dict],
    dev_ratio: float,
    seed: int,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Split entries into train and dev sets, stratified by speaker_id.

    For each speaker, dev_ratio of their utterances go to dev, the rest to
    train. Each speaker contributes at least 1 utterance to dev if they have
    more than 1 utterance total.
    """
    import random

    rng = random.Random(seed)

    by_speaker: Dict[str, List[Dict]] = defaultdict(list)
    for entry in entries:
        by_speaker[entry["speaker_id"]].append(entry)

    train_entries = []
    dev_entries = []

    for speaker_id in sorted(by_speaker.keys()):
        speaker_utts = by_speaker[speaker_id]
        rng.shuffle(speaker_utts)

        n_dev = max(1, int(len(speaker_utts) * dev_ratio))
        # If the speaker has only 1 utterance, put it in train
        if len(speaker_utts) == 1:
            train_entries.extend(speaker_utts)
        else:
            dev_entries.extend(speaker_utts[:n_dev])
            train_entries.extend(speaker_utts[n_dev:])

    return train_entries, dev_entries


def write_tsv(entries: List[Dict], tsv_path: Path) -> None:
    """
    Write entries to a TSV file in LuxTTS format.

    Format: {uniq_id}\t{text}\t{wav_path}
    """
    tsv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(tsv_path, "w", encoding="utf-8") as f:
        for entry in entries:
            uniq_id = entry["filename"]
            text = entry["text"]
            wav_path = entry["resampled_path"]
            f.write(f"{uniq_id}\t{text}\t{wav_path}\n")
    logger.info("Wrote %d entries to %s", len(entries), tsv_path)


def print_statistics(
    all_entries: List[Dict],
    train_entries: List[Dict],
    dev_entries: List[Dict],
) -> None:
    """Print summary statistics about the dataset."""
    by_speaker: Dict[str, List[Dict]] = defaultdict(list)
    for entry in all_entries:
        by_speaker[entry["speaker_id"]].append(entry)

    total_duration = sum(e.get("duration", 0.0) for e in all_entries)
    train_duration = sum(e.get("duration", 0.0) for e in train_entries)
    dev_duration = sum(e.get("duration", 0.0) for e in dev_entries)

    print("\n" + "=" * 60)
    print("  MoeSpeech -> LuxTTS Conversion Statistics")
    print("=" * 60)
    print(f"  Total utterances:     {len(all_entries)}")
    print(f"  Train utterances:     {len(train_entries)}")
    print(f"  Dev utterances:       {len(dev_entries)}")
    print(f"  Total duration:       {total_duration:.1f}s ({total_duration / 3600:.2f}h)")
    print(f"  Train duration:       {train_duration:.1f}s ({train_duration / 3600:.2f}h)")
    print(f"  Dev duration:         {dev_duration:.1f}s ({dev_duration / 3600:.2f}h)")
    print(f"  Number of speakers:   {len(by_speaker)}")
    print("-" * 60)
    print(f"  {'Speaker':<20} {'Utterances':>12} {'Duration (s)':>14}")
    print("-" * 60)
    for speaker_id in sorted(by_speaker.keys()):
        utts = by_speaker[speaker_id]
        spk_dur = sum(e.get("duration", 0.0) for e in utts)
        print(f"  {speaker_id:<20} {len(utts):>12} {spk_dur:>14.1f}")
    print("=" * 60 + "\n")


def main():
    args = get_args()

    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO, force=True)

    input_dir: Path = args.input_dir
    output_dir: Path = args.output_dir
    train_tsv: Path = args.train_tsv
    dev_tsv: Path = args.dev_tsv
    target_rate: int = args.sampling_rate
    source_rate: int = args.source_rate
    dev_ratio: float = args.dev_ratio
    num_workers: int = args.num_workers
    seed: int = args.seed

    # Validate input directory
    metadata_path = input_dir / "metadata.csv"
    wavs_dir = input_dir / "wavs"

    if not metadata_path.is_file():
        logger.error("metadata.csv not found at %s", metadata_path)
        sys.exit(1)
    if not wavs_dir.is_dir():
        logger.error("wavs/ directory not found at %s", wavs_dir)
        sys.exit(1)

    # Step 1: Parse metadata
    logger.info("Parsing metadata from %s", metadata_path)
    entries = parse_metadata(metadata_path)
    logger.info("Found %d entries in metadata.csv", len(entries))

    if not entries:
        logger.error("No valid entries found in metadata.csv")
        sys.exit(1)

    # Step 2: Verify WAV files exist and collect resample tasks
    logger.info("Verifying WAV files exist...")
    valid_entries = []
    missing_count = 0
    for entry in entries:
        src_wav = wavs_dir / f"{entry['filename']}.wav"
        if not src_wav.is_file():
            logger.warning("WAV file not found, skipping: %s", src_wav)
            missing_count += 1
            continue
        entry["src_path"] = src_wav
        entry["dst_path"] = output_dir / "wavs" / f"{entry['filename']}.wav"
        valid_entries.append(entry)

    if missing_count > 0:
        logger.warning("Skipped %d entries with missing WAV files", missing_count)
    logger.info("Verified %d WAV files", len(valid_entries))

    if not valid_entries:
        logger.error("No valid WAV files found")
        sys.exit(1)

    # Step 3: Resample audio files in parallel
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "wavs").mkdir(parents=True, exist_ok=True)

    logger.info(
        "Resampling %d files from %d Hz to %d Hz using %d workers...",
        len(valid_entries),
        source_rate,
        target_rate,
        num_workers,
    )

    successful_entries = []
    failed_count = 0

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        future_to_entry = {}
        for entry in valid_entries:
            future = executor.submit(
                resample_single,
                entry["src_path"],
                entry["dst_path"],
                source_rate,
                target_rate,
            )
            future_to_entry[future] = entry

        with tqdm(total=len(future_to_entry), desc="Resampling audio") as pbar:
            for future in as_completed(future_to_entry):
                entry = future_to_entry[future]
                try:
                    result = future.result()
                except Exception as e:
                    logger.warning(
                        "Worker exception for %s: %s", entry["filename"], e
                    )
                    failed_count += 1
                    pbar.update(1)
                    continue

                if result is not None:
                    resampled_path, duration = result
                    entry["resampled_path"] = resampled_path
                    entry["duration"] = duration
                    successful_entries.append(entry)
                else:
                    failed_count += 1
                pbar.update(1)

    if failed_count > 0:
        logger.warning("Failed to resample %d files", failed_count)
    logger.info("Successfully resampled %d files", len(successful_entries))

    if not successful_entries:
        logger.error("No files were successfully resampled")
        sys.exit(1)

    # Step 4: Stratified train/dev split
    logger.info(
        "Splitting into train/dev with %.0f%%/%.0f%% ratio, stratified by speaker...",
        (1 - dev_ratio) * 100,
        dev_ratio * 100,
    )
    train_entries, dev_entries = stratified_split(
        successful_entries, dev_ratio, seed
    )

    # Step 5: Write TSV files
    logger.info("Writing TSV files...")
    write_tsv(train_entries, train_tsv)
    write_tsv(dev_entries, dev_tsv)

    # Step 6: Print statistics
    print_statistics(successful_entries, train_entries, dev_entries)

    logger.info("Conversion complete!")
    logger.info("  Train TSV: %s (%d utterances)", train_tsv, len(train_entries))
    logger.info("  Dev TSV:   %s (%d utterances)", dev_tsv, len(dev_entries))
    logger.info(
        "Next steps:\n"
        "  1. python -m zipvoice.bin.prepare_dataset "
        "--tsv-path %s --prefix custom --subset train --output-dir data/manifests\n"
        "  2. python -m zipvoice.bin.prepare_dataset "
        "--tsv-path %s --prefix custom --subset dev --output-dir data/manifests",
        train_tsv,
        dev_tsv,
    )


if __name__ == "__main__":
    main()
