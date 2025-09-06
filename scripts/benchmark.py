#!/usr/bin/env python
"""
Benchmark ingestion + sentiment labeling throughput.

Usage examples:
  # VADER, full file
  python scripts/benchmark.py --csv data/news_perf_test_100k.csv --model vader

  # FinBERT on CPU, batch 32, cap 20k rows
  python scripts/benchmark.py --csv data/news_perf_test_100k.csv --model ProsusAI/finbert --batch-size 32 --limit 20000

  # CardiffNLP RoBERTa, write labeled parquet and results CSV
  python scripts/benchmark.py --csv data/news_perf_test.csv --model cardiffnlp/twitter-roberta-base-sentiment-latest --results out/bench.csv --save-labeled out/labeled.parquet
"""
import argparse
import csv
import hashlib
import os
import sys
import time
from pathlib import Path

import pandas as pd

from app.sentiment import BaselineVader, HFClassifier

# Ensure repo root is on sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def md5(s: str) -> str:
    return hashlib.md5(s.encode("utf-8", errors="ignore")).hexdigest()


def load_csv(path: str, limit: int | None):
    """Load CSV and normalize schema for benchmark."""
    t0 = time.perf_counter()
    df = pd.read_csv(path)
    if limit:
        df = df.head(limit)

    cols = [c.lower() for c in df.columns]
    text_idx = next(
        (i for i, c in enumerate(cols) if "headline" in c or "title" in c or "text" in c),
        None,
    )
    if text_idx is None:
        raise ValueError("No text-like column (headline/title/text) found.")
    date_idx = next((i for i, c in enumerate(cols) if "date" in c or "time" in c), None)
    tick_idx = next((i for i, c in enumerate(cols) if "ticker" in c or "symbol" in c), None)

    norm = pd.DataFrame(
        {
            "date": (pd.to_datetime(df.iloc[:, date_idx], errors="coerce").dt.date if date_idx is not None else None),
            "ticker": df.iloc[:, tick_idx] if tick_idx is not None else None,
            "source": Path(path).name,
            "headline": df.iloc[:, text_idx],
            "text": df.iloc[:, text_idx],
        }
    )
    dt = (time.perf_counter() - t0) * 1000
    return norm, dt


def get_model(model_str: str, batch_size: int, max_len: int):
    """Load model wrapper (VADER or HF)."""
    if model_str.lower() == "vader":
        return BaselineVader()
    os.environ.setdefault("SENT_BATCH_SIZE", str(batch_size))
    os.environ.setdefault("SENT_MAX_LEN", str(max_len))
    return HFClassifier(model_str)


def label_with_dedupe(df: pd.DataFrame, model):
    """Hash-dedupe identical texts; label unique only."""
    t0 = time.perf_counter()
    work = df.copy()
    work["text"] = work["text"].fillna("").astype(str)
    work["__h"] = work["text"].map(md5)

    # unique texts
    uniq = work.drop_duplicates("__h", keep="first")[["__h", "text"]].copy()
    t_load = (time.perf_counter() - t0) * 1000

    # label
    t1 = time.perf_counter()
    texts = uniq["text"].tolist()
    try:
        labels, conf = model.predict_with_scores(texts)
        uniq["sentiment"], uniq["confidence"] = labels, conf
    except Exception:
        uniq["sentiment"] = model.predict(texts)
    t_label = (time.perf_counter() - t1) * 1000

    # merge back
    t2 = time.perf_counter()
    out = work.merge(uniq.drop(columns=["text"]), on="__h", how="left").drop(columns="__h")
    t_merge = (time.perf_counter() - t2) * 1000

    stats = {
        "rows_total": len(work),
        "rows_unique": len(uniq),
        "dedupe_ratio": 1.0 - (len(uniq) / max(1, len(work))),
        "t_load_ms": round(t_load, 1),
        "t_label_ms": round(t_label, 1),
        "t_merge_ms": round(t_merge, 1),
        "t_total_ms": round((time.perf_counter() - t0) * 1000, 1),
        "throughput_rows_per_s": round(len(work) / max(0.001, (time.perf_counter() - t1)), 1),
    }
    return out, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to input CSV")
    ap.add_argument("--model", required=True, help="'vader' or HF model id (e.g., ProsusAI/finbert)")
    ap.add_argument("--batch-size", type=int, default=32, help="HF batch size")
    ap.add_argument("--max-len", type=int, default=96, help="HF max tokens")
    ap.add_argument("--limit", type=int, default=None, help="Cap rows")
    ap.add_argument("--save-labeled", default=None, help="Optional path to save labeled parquet")
    ap.add_argument("--results", default=None, help="Optional CSV to append benchmark results")
    args = ap.parse_args()

    # Step 1: load + normalize
    df, t_ingest = load_csv(args.csv, args.limit)

    # Step 2: load model
    model = get_model(args.model, args.batch_size, args.max_len)

    # Step 3: label
    labeled, stats = label_with_dedupe(df, model)

    # Optional save
    if args.save_labeled:
        Path(args.save_labeled).parent.mkdir(parents=True, exist_ok=True)
        labeled.to_parquet(args.save_labeled, index=False)

    # Report
    report = {
        "csv": args.csv,
        "model": args.model,
        "batch_size": args.batch_size,
        "max_len": args.max_len,
        "limit": args.limit or -1,
        "rows_total": stats["rows_total"],
        "rows_unique": stats["rows_unique"],
        "dedupe_ratio": stats["dedupe_ratio"],
        "ingest_ms": round(t_ingest, 1),
        "label_ms": stats["t_label_ms"],
        "merge_ms": stats["t_merge_ms"],
        "total_ms": stats["t_total_ms"],
        "rows_per_s": stats["throughput_rows_per_s"],
    }

    print("\n=== Benchmark Report ===")
    for k, v in report.items():
        print(f"{k:>14}: {v}")

    # Append results CSV
    if args.results:
        Path(args.results).parent.mkdir(parents=True, exist_ok=True)
        write_header = not Path(args.results).exists()
        with open(args.results, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(report.keys()))
            if write_header:
                w.writeheader()
            w.writerow(report)


if __name__ == "__main__":
    main()
