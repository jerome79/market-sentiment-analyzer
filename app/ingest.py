"""
Module for ingesting and normalizing financial news CSV data.

This module handles loading CSV data from folders, normalizing column names,
merging sector information, and writing processed data to Parquet format.
"""
import glob
import os
from pathlib import Path
from typing import Optional

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# Standard schema for normalized news data
SCHEMA = ["date", "ticker", "source", "headline", "text"]


def _resolve_directory(csv_dir: Optional[str]) -> Path:
    """
    Resolve a CSV directory to an absolute path using repo root as base.

    Args:
        csv_dir: Directory containing CSV files. If None, uses NEWS_CSV_DIR
                environment variable or defaults to 'data'.

    Returns:
        Absolute path to directory.

    Example:
        If csv_dir is relative, it is resolved against the repo root.
    """
    repo_root = Path(__file__).resolve().parents[1]
    # Use environment variable or default
    raw_path = csv_dir or os.getenv("NEWS_CSV_DIR", "data")
    directory_path = Path(os.path.expanduser(raw_path))

    if not directory_path.is_absolute():
        directory_path = (repo_root / directory_path).resolve()

    return directory_path


def load_csv_directory(csv_dir: Optional[str] = None) -> pd.DataFrame:
    """
    Load and concatenate all CSV files in a directory into a single DataFrame.

    Args:
        csv_dir: Directory path. Uses NEWS_CSV_DIR environment variable
                if not provided.

    Returns:
        DataFrame containing all rows from all CSV files.

    Raises:
        FileNotFoundError: If no CSV files found in directory.
        ValueError: If no headline/title/text column found in a file.

    Example:
        >>> df = load_csv_directory('data/news')
        >>> print(f"Loaded {len(df)} rows")
    """
    rows = []
    directory_path = _resolve_directory(csv_dir)
    pattern = str(directory_path / "*.csv")
    print(f"[ingest] Looking for CSVs in: {directory_path}")
    print(f"[ingest] Glob pattern: {pattern}")

    csv_paths = sorted(glob.glob(pattern))
    if not csv_paths:
        print(
            "[ingest] ⚠️  No CSV files found. "
            "Ensure folder exists and contains *.csv files"
        )
        return pd.DataFrame(columns=SCHEMA)

    for file_path in csv_paths:
        try:
            dataframe = pd.read_csv(file_path)
        except Exception as error:
            print(f"[ingest] Skipping {file_path}: {error}")
            continue

        # Best-effort column name detection
        date_column = _find_column(dataframe, ["date", "time"])
        text_column = _find_column(dataframe, ["headline", "title", "text"])
        ticker_column = _find_column(dataframe, ["ticker", "symbol"])

        if text_column is None:
            print(f"[ingest] Skipping {file_path}: no headline/title/text column")
            continue

        # Process each row
        for _, row in dataframe.iterrows():
            processed_row = {
                "date": _safe_parse_date(row.get(date_column))
                if date_column
                else None,
                "ticker": row.get(ticker_column) if ticker_column else None,
                "source": os.path.basename(file_path),
                "headline": row.get(text_column),
                "text": row.get(text_column),
            }
            rows.append(processed_row)

    result_dataframe = pd.DataFrame(rows, columns=SCHEMA)
    result_dataframe.dropna(subset=["text"], inplace=True)
    print(f"[ingest] Loaded {len(result_dataframe)} rows")
    return result_dataframe


def _find_column(dataframe: pd.DataFrame, patterns: list) -> Optional[str]:
    """
    Find a column matching any of the given patterns (case-insensitive).

    Args:
        dataframe: DataFrame to search.
        patterns: List of patterns to match against column names.

    Returns:
        First matching column name, or None if no match found.
    """
    for column in dataframe.columns:
        for pattern in patterns:
            if pattern.lower() in column.lower():
                return column
    return None


def _safe_parse_date(date_value):
    """
    Safely parse a date value, returning date object or None.

    Args:
        date_value: Value to parse as date.

    Returns:
        Date object or None if parsing fails.
    """
    if pd.isna(date_value):
        return None
    try:
        return pd.to_datetime(date_value, errors="coerce").date()
    except Exception:
        return None


def normalize_and_save(dataframe: pd.DataFrame, output_path: str) -> pd.DataFrame:
    """
    Normalize financial news DataFrame and save to Parquet.

    This function:
    - Converts date column to standard format
    - Merges sector information from sector map, if available
    - Writes DataFrame to Parquet format

    Args:
        dataframe: Input DataFrame with columns
                  ['date', 'ticker', 'headline', 'text', ...].
        output_path: Output path for Parquet file.

    Returns:
        Normalized DataFrame.

    Raises:
        ValueError: If required columns are missing.
        IOError: If output path is not writable.

    Example:
        >>> df = load_csv_directory('data/news')
        >>> normalized_df = normalize_and_save(df, 'data/news.parquet')
    """
    if dataframe.empty:
        raise ValueError("Cannot normalize empty DataFrame")

    normalized_df = dataframe.copy()

    # Normalize date column
    try:
        normalized_df["date"] = pd.to_datetime(normalized_df["date"]).dt.date
    except Exception as error:
        print(f"[ingest] Warning: Could not normalize dates: {error}")

    # Merge sector information if available
    try:
        sector_path = _resolve_directory(os.getenv("SECTOR_MAP_CSV"))
        if sector_path.exists():
            print(f"[ingest] Loading sector map from: {sector_path}")
            sector_map = pd.read_csv(sector_path)
            if {"ticker", "sector"}.issubset(sector_map.columns):
                normalized_df = normalized_df.merge(
                    sector_map[["ticker", "sector"]], on="ticker", how="left"
                )
                print("[ingest] Merged sector information for tickers")
            else:
                print("[ingest] Warning: Sector map missing required columns")
    except Exception as error:
        print(f"[ingest] Warning: Could not load sector map: {error}")

    # Save to Parquet
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        normalized_df.to_parquet(output_path, index=False)
        print(f"[ingest] Saved normalized data to: {output_path}")
    except Exception as error:
        raise IOError(f"Failed to save to {output_path}: {error}")

    return normalized_df


# Maintain backward compatibility
load_csv_dir = load_csv_directory


if __name__ == "__main__":
    news_dataframe = load_csv_directory(os.getenv("NEWS_CSV_DIR"))
    output_file = Path(__file__).resolve().parents[1] / "data/news.parquet"
    normalize_and_save(news_dataframe, output_file)
    print(f"Saved: data/news.parquet | rows: {len(news_dataframe)}")
