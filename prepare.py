"""Create a small, reproducible English-to-French OPUS Books experiment."""
import argparse
import json
import random
import re
from pathlib import Path

from datasets import load_dataset

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=2000)
    ap.add_argument("--validation", type=int, default=100)
    ap.add_argument("--test", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    DATA.mkdir(exist_ok=True)
    raw = load_dataset("Helsinki-NLP/opus_books", "en-fr", split="train")
    pairs, seen = [], set()
    for row in raw:
        en = re.sub(r"\s+", " ", row["translation"]["en"]).strip()
        fr = re.sub(r"\s+", " ", row["translation"]["fr"]).strip()
        if not (3 <= len(en.split()) <= 30 and 3 <= len(fr.split()) <= 35):
            continue
        key = (en.casefold(), fr.casefold())
        if key in seen:
            continue
        seen.add(key)
        pairs.append({"source": en, "reference": fr})
    rng = random.Random(args.seed)
    rng.shuffle(pairs)
    n = args.train + args.validation + args.test
    if len(pairs) < n:
        raise RuntimeError(f"Only {len(pairs)} pairs after filtering; requested {n}")
    selected = pairs[:n]
    boundaries = [0, args.train, args.train + args.validation, n]
    for name, start, stop in zip(("train", "validation", "test"), boundaries[:-1], boundaries[1:]):
        with (DATA / f"{name}.jsonl").open("w", encoding="utf-8") as f:
            for row in selected[start:stop]:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    meta = {"source": "https://huggingface.co/datasets/Helsinki-NLP/opus_books", "config": "en-fr", "seed": args.seed,
            "filter": "3-30 English words; 3-35 French words; exact case-insensitive pair deduplication",
            "available_after_filter": len(pairs), "train": args.train, "validation": args.validation, "test": args.test}
    (DATA / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
