"""
Module for ingesting and normalizing financial news CSV data.

This module handles the loading and preprocessing of financial news data from
CSV files, including automatic column detection, data normalization, and
sector information merging.

Functions:
    load_csv_dir: Load and concatenate CSV files from a directory.
    normalize_and_save: Normalize data format and save to Parquet.

Typical usage example:
    # Load raw news data from CSV files
    raw_data = load_csv_dir("data/news_csvs/")

    # Normalize and save for further processing
    clean_data = normalize_and_save(raw_data, "data/news.parquet")
"""

import glob
import os
from pathlib import Path
from typing import Optional

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# Expected schema for normalized news data
SCHEMA = ["date", "ticker", "source", "headline", "text"]


def _resolve_dir(csv_dir: Optional[str]) -> Path:
    """
    Resolve a CSV directory path to an absolute Path object.

    This function handles both absolute and relative paths, resolving relative
    paths against the repository root directory. It also supports environment
    variable fallbacks for flexible configuration.

    Args:
        csv_dir: Directory path containing CSV files. If None, uses the
            NEWS_CSV_DIR environment variable or defaults to 'data'.

    Returns:
        Absolute Path object pointing to the resolved directory.

    Example:
        >>> path = _resolve_dir("news_data")  # Relative to repo root
        >>> path = _resolve_dir("/absolute/path/to/data")  # Absolute path
        >>> path = _resolve_dir(None)  # Uses environment variable or 'data'
    """
    # Get repository root (parent of app directory)
    repo_root = Path(__file__).resolve().parents[1]

    # Use provided path, environment variable, or default
    raw_path = csv_dir or os.getenv("NEWS_CSV_DIR", "data")
    directory_path = Path(os.path.expanduser(raw_path))

    # Convert relative paths to absolute using repo root
    if not directory_path.is_absolute():
        directory_path = (repo_root / directory_path).resolve()

    return directory_path


def load_csv_dir(csv_dir: Optional[str] = None) -> pd.DataFrame:
    """
    Load and concatenate all CSV files from a directory into a DataFrame.

    This function automatically detects CSV files in the specified directory,
    attempts to identify relevant columns (date, ticker, text content), and
    normalizes them into a consistent schema. It handles various column naming
    conventions and provides informative error messages for debugging.

    Args:
        csv_dir: Directory path containing CSV files. If None, uses the
            NEWS_CSV_DIR environment variable.

    Returns:
        DataFrame with normalized columns following the SCHEMA format.
        Empty DataFrame if no valid CSV files or data found.

    Raises:
        Warning: Logged for files that cannot be processed, but execution
            continues with remaining files.

    Example:
        >>> df = load_csv_dir("data/news/")
        >>> print(df.columns.tolist())
        ['date', 'ticker', 'source', 'headline', 'text']
        >>> print(len(df))
        1500
    """
    data_rows = []
    directory_path = _resolve_dir(csv_dir)
    file_pattern = str(directory_path / "*.csv")

    print(f"[ingest] Looking for CSV files in: {directory_path}")
    print(f"[ingest] Using glob pattern: {file_pattern}")

    # Find all CSV files matching the pattern
    csv_files = sorted(glob.glob(file_pattern))
    if not csv_files:
        print(
            "[ingest] ⚠️  No CSV files found. "
            "Please ensure the directory exists and contains *.csv files."
        )
        return pd.DataFrame(columns=SCHEMA)

    # Process each CSV file
    for file_path in csv_files:
        try:
            current_df = pd.read_csv(file_path)
        except Exception as file_error:
            print(f"[ingest] Skipping {file_path}: {file_error}")
            continue

        # Attempt to identify relevant columns using pattern matching
        column_names = current_df.columns.tolist()

        # Find date column (date, time, timestamp, etc.)
        date_column = next(
            (
                col
                for col in column_names
                if pd.Series([col]).str.contains("date|time", case=False).any()
            ),
            None,
        )

        # Find text content column (headline, title, text, etc.)
        text_column = next(
            (
                col
                for col in column_names
                if pd.Series([col])
                .str.contains("headline|title|text", case=False)
                .any()
            ),
            None,
        )

        # Find ticker/symbol column
        ticker_column = next(
            (
                col
                for col in column_names
                if pd.Series([col]).str.contains("ticker|symbol", case=False).any()
            ),
            None,
        )

        # Skip files without text content
        if text_column is None:
            print(
                f"[ingest] Skipping {file_path}: " "no headline/title/text column found"
            )
            continue

        # Extract and normalize data from each row
        for _, row in current_df.iterrows():
            # Parse date with error handling
            date_value = None
            if date_column:
                date_value = pd.to_datetime(
                    row.get(date_column), errors="coerce"
                ).date()

            # Extract ticker if available
            ticker_value = row.get(ticker_column) if ticker_column else None

            # Create normalized row
            normalized_row = {
                "date": date_value,
                "ticker": ticker_value,
                "source": os.path.basename(file_path),  # Use filename as source
                "headline": row.get(text_column),
                "text": row.get(text_column),  # Use same content for both fields
            }
            data_rows.append(normalized_row)

    # Create final DataFrame and clean up
    result_df = pd.DataFrame(data_rows, columns=SCHEMA)
    result_df.dropna(subset=["text"], inplace=True)

    print(f"[ingest] Successfully loaded {len(result_df)} rows")
    return result_df


def normalize_and_save(dataframe: pd.DataFrame, output_path: str) -> pd.DataFrame:
    """
    Normalize financial news DataFrame and save to Parquet format.

    This function performs several normalization steps:
    - Standardizes date format to date objects
    - Merges sector information if available via environment configuration
    - Saves the result in efficient Parquet format for downstream processing

    Args:
        dataframe: Input DataFrame containing news data with required columns.
        output_path: File path where the normalized data will be saved.

    Returns:
        The normalized DataFrame with standardized formatting and optional
        sector information merged.

    Example:
        >>> raw_df = load_csv_dir("data/news/")
        >>> clean_df = normalize_and_save(raw_df, "data/news_normalized.parquet")
        >>> print("sector" in clean_df.columns)  # True if sector map available
    """
    # Create a copy to avoid modifying the original
    normalized_df = dataframe.copy()

    # Standardize date format
    normalized_df["date"] = pd.to_datetime(normalized_df["date"]).dt.date

    # Merge sector information if available
    sector_map_path = _resolve_dir(os.getenv("SECTOR_MAP_CSV"))
    if sector_map_path.exists():
        try:
            sector_mapping = pd.read_csv(sector_map_path)
            if (
                "ticker" in sector_mapping.columns
                and "sector" in sector_mapping.columns
            ):
                normalized_df = normalized_df.merge(
                    sector_mapping[["ticker", "sector"]], on="ticker", how="left"
                )
                print(f"[ingest] Merged sector information from {sector_map_path}")
            else:
                print(
                    f"[ingest] Warning: Sector map at {sector_map_path} "
                    "missing required 'ticker' or 'sector' columns"
                )
        except Exception as sector_error:
            print(
                f"[ingest] Warning: Could not load sector map "
                f"from {sector_map_path}: {sector_error}"
            )

    # Save to Parquet format for efficient storage and loading
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    normalized_df.to_parquet(output_path, index=False)

    return normalized_df


if __name__ == "__main__":
    # Command-line execution for batch processing
    news_df = load_csv_dir(os.getenv("NEWS_CSV_DIR"))
    output_file = Path(__file__).resolve().parents[1] / "data/news.parquet"
    normalize_and_save(news_df, output_file)
    print(f"Saved: {output_file} | rows: {len(news_df)}")
