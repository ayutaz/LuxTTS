"""
This file reads the texts in given manifest and save the new cuts with prepared tokens.
"""

import argparse
import logging
from functools import partial
from pathlib import Path

from lhotse import load_manifest, split_parallelize_combine

from zipvoice.tokenizer.tokenizer import add_tokens, add_token_ids


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-file",
        type=str,
        help="Input manifest without tokens",
    )

    parser.add_argument(
        "--output-file",
        type=str,
        help="Output manifest with tokens.",
    )

    parser.add_argument(
        "--num-jobs",
        type=int,
        default=20,
        help="Number of jobs to run in parallel.",
    )

    parser.add_argument(
        "--tokenizer",
        type=str,
        default="emilia",
        choices=["emilia", "espeak", "dialog", "libritts", "simple"],
        help="The destination directory of manifest files.",
    )

    parser.add_argument(
        "--lang",
        type=str,
        default="en-us",
        help="Language identifier. For espeak tokenizer, see "
        "https://github.com/rhasspy/espeak-ng/blob/master/docs/languages.md . "
        "For emilia/dialog tokenizer, set to 'ja' to enable Japanese g2p "
        "so that CJK characters are treated as Japanese instead of Chinese.",
    )

    parser.add_argument(
        "--token-file",
        type=str,
        default=None,
        help="Path to the token file for converting tokens to IDs. "
        "Required when --store-ids is set.",
    )

    parser.add_argument(
        "--store-ids",
        action="store_true",
        default=False,
        help="When set, store pre-computed token IDs instead of string tokens.",
    )

    return parser.parse_args()


def prepare_tokens(
    input_file: Path,
    output_file: Path,
    num_jobs: int,
    tokenizer: str,
    lang: str = "en-us",
    token_file: str = None,
    store_ids: bool = False,
):
    logging.info(f"Processing {input_file}")
    if output_file.is_file():
        logging.info(f"{output_file} exists, skipping.")
        return
    logging.info(f"loading manifest from {input_file}")
    cut_set = load_manifest(input_file)

    if store_ids:
        logging.info("Adding pre-computed token IDs")
        _add_fn = partial(
            add_token_ids, tokenizer=tokenizer, lang=lang, token_file=token_file
        )
    else:
        _add_fn = partial(add_tokens, tokenizer=tokenizer, lang=lang)
        logging.info("Adding tokens")

    cut_set = split_parallelize_combine(
        num_jobs=num_jobs, manifest=cut_set, fn=_add_fn
    )

    logging.info(f"Saving file to {output_file}")
    cut_set.to_file(output_file)


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO, force=True)

    args = get_args()
    input_file = Path(args.input_file)
    output_file = Path(args.output_file)
    num_jobs = args.num_jobs
    tokenizer = args.tokenizer
    lang = args.lang
    token_file = args.token_file
    store_ids = args.store_ids

    output_file.parent.mkdir(parents=True, exist_ok=True)

    prepare_tokens(
        input_file=input_file,
        output_file=output_file,
        num_jobs=num_jobs,
        tokenizer=tokenizer,
        lang=lang,
        token_file=token_file,
        store_ids=store_ids,
    )

    logging.info("Done!")
