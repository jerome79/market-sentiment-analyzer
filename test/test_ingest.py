import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from app.ingest import SCHEMA, load_csv_dir, normalize_and_save


def test_load_csv_dir_empty_returns_empty_df_with_schema(tmp_path):
    df = load_csv_dir(str(tmp_path))
    assert list(df.columns) == SCHEMA
    assert df.empty


def test_load_csv_dir_varied_columns_and_source_names(tmp_path):
    # File 1: columns time + headline
    f1 = tmp_path / "a.csv"
    pd.DataFrame(
        {
            "time": ["2024-01-01", "2024-01-02"],
            "headline": ["Markets rally", "Shares slip"],
        }
    ).to_csv(f1, index=False)

    # File 2: columns date + title + symbol
    f2 = tmp_path / "b.csv"
    pd.DataFrame(
        {
            "date": ["2024-01-03"],
            "title": ["Tech stocks surge"],
            "symbol": ["AAPL"],
        }
    ).to_csv(f2, index=False)

    df = load_csv_dir(str(tmp_path))
    # Expect 3 rows and standard schema
    assert set(df.columns) == set(SCHEMA)
    assert len(df) == 3
    # Sources should be file basenames
    assert set(df["source"].unique()) == {f1.name, f2.name}
    # Text is copied into both headline and text fields
    assert df["headline"].notna().all()
    assert df["text"].notna().all()


def test_normalize_and_save_merges_sector_and_writes_parquet(tmp_path, monkeypatch):
    # Prepare input df
    df = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-02"],
            "ticker": ["AAPL", "MSFT"],
            "headline": ["h1", "h2"],
            "text": ["t1", "t2"],
            "source": ["x", "y"],
        }
    )

    # Sector map file (env var may be relative; function resolves it)
    sector_csv = tmp_path / "sector_map.csv"
    pd.DataFrame({"ticker": ["AAPL", "MSFT"], "sector": ["Tech", "Tech"]}).to_csv(sector_csv, index=False)

    monkeypatch.setenv("SECTOR_MAP_CSV", str(sector_csv))

    out_path = tmp_path / "out.parquet"
    normalize_and_save(df, str(out_path))

    assert out_path.exists()
    # Parquet content should include sector
    read_back = pd.read_parquet(out_path)
    assert "sector" in read_back.columns
    assert set(read_back["sector"].dropna().unique()) == {"Tech"}


def test_load_csv_dir_empty_returns_empty_df_with_schema(tmp_path: Path) -> None:
    """
    Test that load_csv_dir returns an empty DataFrame with the correct schema when the directory is empty.

    Args:
        tmp_path (Path): Temporary directory provided by pytest.
    """
    df = load_csv_dir(str(tmp_path))
    assert list(df.columns) == SCHEMA
    assert df.empty


def test_load_csv_dir_varied_columns_and_source_names(tmp_path: Path) -> None:
    """
    Test that load_csv_dir correctly loads CSV files with varied columns and source names.
    """
    f1 = tmp_path / "a.csv"
    pd.DataFrame({"time": ["2024-01-01"], "headline": ["Markets rally"]}).to_csv(f1, index=False)
    f2 = tmp_path / "b.csv"
    pd.DataFrame({"date": ["2024-01-03"], "title": ["Tech stocks surge"], "symbol": ["AAPL"]}).to_csv(f2, index=False)

    df = load_csv_dir(str(tmp_path))
    # Should have standard schema
    assert set(df.columns) == set(SCHEMA)
    # Should have 2 rows (one per file)
    assert len(df) == 2
    # Sources should match file names
    assert set(df["source"].unique()) == {f1.name, f2.name}
    # Headline and text columns should not be empty
    assert df["headline"].notna().all()
    assert df["text"].notna().all()
