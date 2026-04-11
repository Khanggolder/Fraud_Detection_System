import os
import sys
import glob
import argparse
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.c_features import extract_c_features
from src.cpp_features import extract_cpp_ast_features


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASETS = {
    "c": {
        "ai_dir": os.path.join(BASE_DIR, "data_C", "AI"),
        "human_dir": os.path.join(BASE_DIR, "data_C", "Human"),
        "extensions": ["*.c", "*.h"],
        "extractor": extract_c_features,
        "output": os.path.join(BASE_DIR, "features_c.csv"),
    },
    "cpp": {
        "ai_dir": os.path.join(BASE_DIR, "dataset_CPP", "ai"),
        "human_dir": os.path.join(BASE_DIR, "dataset_CPP", "human"),
        "extensions": ["*.cpp", "*.cc", "*.cxx", "*.c", "*.h", "*.hpp", "*.hxx"],
        "extractor": extract_cpp_ast_features,
        "output": os.path.join(BASE_DIR, "features_cpp.csv"),
    },
}


def _collect_files(directory: str, extensions: list) -> list:
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(directory, "**", ext), recursive=True))
    return sorted(set(files))


def _process_single_file(args):
    filepath, label, extractor_name = args

    if extractor_name == "c":
        from src.c_features import extract_c_features as extractor
    else:
        from src.cpp_features import extract_cpp_ast_features as extractor

    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            code = f.read()

        if len(code.strip()) < 10:
            return None

        features = extractor(code)

        features["_file"] = os.path.basename(filepath)
        features["_path"] = filepath
        features["_label"] = label
        features["_chars"] = len(code)

        return features
    except Exception as e:
        return {"_file": os.path.basename(filepath), "_error": str(e), "_label": label}


def _extract_dataset(lang: str, max_workers: int = 8, sample_size: int = 0):
    config = DATASETS[lang]
    ai_dir = config["ai_dir"]
    human_dir = config["human_dir"]
    extensions = config["extensions"]
    output_path = config["output"]
    extractor_name = lang

    print(f"\n{'='*60}")
    print(f" Extracting features for: {lang.upper()}")
    print(f"{'='*60}")

    ai_files = _collect_files(ai_dir, extensions)
    human_files = _collect_files(human_dir, extensions)
    print(f"  AI files found:    {len(ai_files):,}")
    print(f"  Human files found: {len(human_files):,}")

    if not ai_files and not human_files:
        print(f"  [WARN] No files found, skipping {lang}.")
        return

    if sample_size > 0:
        import random
        random.seed(42)
        if len(ai_files) > sample_size:
            ai_files = random.sample(ai_files, sample_size)
            print(f"  Sampled AI files:  {len(ai_files):,}")
        if len(human_files) > sample_size:
            human_files = random.sample(human_files, sample_size)
            print(f"  Sampled Human files: {len(human_files):,}")

    work_items = []
    for fp in ai_files:
        work_items.append((fp, 1, extractor_name))
    for fp in human_files:
        work_items.append((fp, 0, extractor_name))

    total = len(work_items)
    print(f"  Total to process:  {total:,}")
    print(f"  Workers:           {max_workers}")
    print()

    checkpoint_path = output_path + ".checkpoint"
    processed_files = set()
    existing_rows = []

    if os.path.exists(checkpoint_path):
        try:
            existing_df = pd.read_csv(checkpoint_path)
            processed_files = set(existing_df["_path"].dropna().tolist())
            existing_rows = existing_df.to_dict("records")
            print(f"  Resuming from checkpoint: {len(processed_files):,} files already done")
        except Exception:
            processed_files = set()

    work_items = [w for w in work_items if w[0] not in processed_files]
    remaining = len(work_items)
    print(f"  Remaining to process: {remaining:,}")

    if remaining == 0:
        print("  All files already processed! Generating final CSV...")
        df = pd.DataFrame(existing_rows)
        clean = df[~df.columns.str.startswith("_error")].copy()
        clean.to_csv(output_path, index=False)
        print(f"  Saved: {output_path} ({len(clean):,} rows)")
        return

    results = list(existing_rows)
    done = len(existing_rows)
    errors = 0
    start_time = time.time()

    CHECKPOINT_EVERY = 2000

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_process_single_file, item): item for item in work_items}

        for future in as_completed(futures):
            done += 1
            result = future.result()

            if result is None:
                continue
            if "_error" in result:
                errors += 1
                continue

            results.append(result)

            if done % 500 == 0 or done == total:
                elapsed = time.time() - start_time
                rate = (done - len(existing_rows)) / max(elapsed, 0.01)
                eta = (remaining - (done - len(existing_rows))) / max(rate, 0.01)
                print(
                    f"  [{done:>7,}/{total:,}]  "
                    f"{rate:.0f} files/sec  "
                    f"ETA: {eta/60:.1f} min  "
                    f"Errors: {errors}"
                )

            if done % CHECKPOINT_EVERY == 0:
                _save_checkpoint(results, checkpoint_path)

    df = pd.DataFrame(results)

    _save_checkpoint(results, checkpoint_path)

    meta_cols = ["_file", "_path", "_label", "_chars"]
    feature_cols = [c for c in df.columns if not c.startswith("_error")]
    df_clean = df[feature_cols].copy()

    df_clean.to_csv(output_path, index=False)

    elapsed = time.time() - start_time
    print(f"\n  Done! Saved {output_path}")
    print(f"  Total rows: {len(df_clean):,}  |  Errors: {errors}  |  Time: {elapsed:.1f}s")

    label_counts = df_clean["_label"].value_counts()
    print(f"  AI samples:    {label_counts.get(1, 0):,}")
    print(f"  Human samples: {label_counts.get(0, 0):,}")

    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)


def _save_checkpoint(results: list, path: str):
    try:
        df = pd.DataFrame(results)
        df.to_csv(path, index=False)
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(
        description="Extract AST-based features from C/C++ datasets into CSV for ML training."
    )
    parser.add_argument(
        "--lang",
        choices=["c", "cpp", "both"],
        default="both",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=0,
    )
    args = parser.parse_args()

    print("=" * 60)
    print(" Feature Extraction Pipeline")
    print(f" Language: {args.lang.upper()}")
    print(f" Workers:  {args.workers}")
    if args.sample > 0:
        print(f" Sample:   {args.sample:,} per class")
    print("=" * 60)

    if args.lang in ("c", "both"):
        _extract_dataset("c", max_workers=args.workers, sample_size=args.sample)

    if args.lang in ("cpp", "both"):
        _extract_dataset("cpp", max_workers=args.workers, sample_size=args.sample)

    print("\n All feature extraction complete!")


if __name__ == "__main__":
    main()
