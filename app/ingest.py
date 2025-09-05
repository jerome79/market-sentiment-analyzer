"""
Module for ingesting and normalizing financial news CSV data.
Handles loading data from folders, merging sector information, and writing to Parquet.
"""
import os, glob, pandas as pd
from dotenv import load_dotenv
from pathlib import Path
load_dotenv()

SCHEMA = ["date","ticker","source","headline","text"]

def _resolve_dir(csv_dir: str | None) -> Path:
    """
    Resolve a CSV directory to an absolute path, using the repo root as base if necessary.

    Parameters:
        csv_dir (str | None): Directory containing CSV files. If None, uses NEWS_CSV_DIR env variable or defaults to 'data'.

    Returns:
        Path: Absolute path to directory.

    Example:
        If csv_dir is relative, it is resolved against the repo root.
    """
    repo_root = Path(__file__).resolve().parents[1]
    # env or default
    raw = csv_dir or os.getenv("NEWS_CSV_DIR", "data")
    p = Path(os.path.expanduser(raw))
    if not p.is_absolute():
        p = (repo_root / p).resolve()
    return p


def load_csv_dir(csv_dir: str | None = None) -> pd.DataFrame:
    """
    Load and concatenate all CSV files in a directory into a single DataFrame.

    Parameters:
        csv_dir (str | None): Directory path. Uses NEWS_CSV_DIR environment variable if not provided.

    Returns:
        pd.DataFrame: DataFrame containing all rows from all CSV files.

    Raises:
        FileNotFoundError: If no CSV files found.
        ValueError: If no headline/title/text column found in a file.

    Example:
        load_csv_dir('data/news')
    """
    rows = []
    dir_path = _resolve_dir(csv_dir)
    pattern = str(dir_path / "*.csv")
    print(f"[ingest] looking for CSVs in: {dir_path}")
    print(f"[ingest] glob pattern: {pattern}")

    paths = sorted(glob.glob(pattern))
    if not paths:
        print(f"[ingest] ⚠️ no CSV files found. Ensure folder exists and has *.csv")
    for fp in paths:
        try:
            df = pd.read_csv(fp)
        except Exception as e:
            print(f"[ingest] skip {fp}: {e}")
            continue
        # best-effort column names
        date_col  = next((c for c in df.columns if pd.Series([c]).str.contains("date|time", case=False).any()), None)
        text_col  = next((c for c in df.columns if pd.Series([c]).str.contains("headline|title|text", case=False).any()), None)
        tick_col  = next((c for c in df.columns if pd.Series([c]).str.contains("ticker|symbol", case=False).any()), None)
        if text_col is None:
            print(f"[ingest] skip {fp}: no headline/title/text column")
            continue

        for _, r in df.iterrows():
            rows.append({
                "date": pd.to_datetime(r.get(date_col), errors="coerce").date() if date_col else None,
                "ticker": (r.get(tick_col) if tick_col else None),
                "source": os.path.basename(fp),
                "headline": r.get(text_col),
                "text": r.get(text_col)
            })

    out = pd.DataFrame(rows, columns=SCHEMA)
    out.dropna(subset=["text"], inplace=True)
    print(f"[ingest] loaded rows: {len(out)}")
    return out

def normalize_and_save(df: pd.DataFrame, out_path: str)-> pd.DataFrame:
    """
    Normalize financial news DataFrame and save to Parquet.

    - Converts date to standard format.
    - Merges sector information from sector map, if available.
    - Writes DataFrame to Parquet.

    Parameters:
        df (pd.DataFrame): Input DataFrame with columns ['date', 'ticker', 'headline', 'text', ...].
        out_path (str): Output path for Parquet file.

    Returns:
        pd.DataFrame: Normalized DataFrame.

    Example:
        normalize_and_save(df, 'data/news.parquet')
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.date

    # --- merge sector map if available ---
    sector_path = _resolve_dir(os.getenv("SECTOR_MAP_CSV"))
    if sector_path.exists():
        sector_map = pd.read_csv(sector_path)
        df = df.merge(sector_map[["ticker","sector"]], on="ticker", how="left")

    df.to_parquet(out_path, index=False)
    return df


if __name__ == "__main__":
    df = load_csv_dir(os.getenv("NEWS_CSV_DIR"))
    normalize_and_save(df, Path(__file__).resolve().parents[1] / "data/news.parquet")
    print("Saved:", "data/news.parquet", "| rows:", len(df))