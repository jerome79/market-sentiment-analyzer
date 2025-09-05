import os, glob, pandas as pd
from dotenv import load_dotenv
from pathlib import Path
load_dotenv()

SCHEMA = ["date","ticker","source","headline","text"]

def _resolve_dir(csv_dir: str | None) -> Path:
    """
    Resolves the CSV directory path to an absolute path.

    Args:
        csv_dir (str | None): The directory containing CSV files. If None, uses the NEWS_CSV_DIR environment variable or defaults to 'data'.

    Returns:
        Path: Absolute path to the CSV directory, relative to the repository root if needed.
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
    Loads all CSV files from the specified directory into a single DataFrame.

    Args:
        csv_dir (str | None): Path to the directory containing CSV files. If None, uses the NEWS_CSV_DIR environment variable or defaults to 'data'.

    Returns:
        pd.DataFrame: Combined DataFrame containing all rows from the CSV files.
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

def normalize_and_save(df: pd.DataFrame, out_path: str):
    """
       Normalizes the input DataFrame (e.g., date formatting, sector merge) and saves it to Parquet.

       Args:
           df (pd.DataFrame): Input data.
           out_path (str): Output path for Parquet file.
       Returns:
           pd.DataFrame: Normalized DataFrame.
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