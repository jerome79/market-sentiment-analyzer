"""
Unit tests for the data ingestion module.

Tests CSV loading, data normalization, and file handling functionality
to ensure robust data processing with various input formats.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from app.ingest import _resolve_dir, load_csv_dir, normalize_and_save


class TestResolveDir:
    """Test cases for directory path resolution."""

    def test_resolve_absolute_path(self):
        """Test handling of absolute paths."""
        absolute_path = "/tmp/test_data"
        result = _resolve_dir(absolute_path)

        assert result.is_absolute()
        assert str(result) == absolute_path

    def test_resolve_relative_path(self):
        """Test resolution of relative paths against repo root."""
        relative_path = "data/news"
        result = _resolve_dir(relative_path)

        assert result.is_absolute()
        assert result.name == "news"
        assert "data" in str(result)

    @patch.dict(os.environ, {"NEWS_CSV_DIR": "/env/test/path"})
    def test_resolve_from_environment(self):
        """Test path resolution from environment variable."""
        result = _resolve_dir(None)

        assert str(result) == "/env/test/path"

    @patch.dict(os.environ, {}, clear=True)
    def test_resolve_default_path(self):
        """Test default path when no environment variable is set."""
        result = _resolve_dir(None)

        assert result.is_absolute()
        assert result.name == "data"

    def test_resolve_expanduser(self):
        """Test expansion of user home directory."""
        home_path = "~/test_data"
        result = _resolve_dir(home_path)

        assert result.is_absolute()
        assert "~" not in str(result)  # Should be expanded


class TestLoadCsvDir:
    """Test cases for CSV directory loading."""

    def create_test_csv(self, file_path: Path, data: dict, columns: list = None):
        """Helper method to create test CSV files."""
        df = pd.DataFrame(data)
        if columns:
            df.columns = columns
        df.to_csv(file_path, index=False)

    def test_load_single_csv_standard_columns(self):
        """Test loading a single CSV with standard column names."""
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "test.csv"

            # Create test data with standard columns
            test_data = {
                "date": ["2024-01-01", "2024-01-02"],
                "ticker": ["AAPL", "TSLA"],
                "headline": ["Apple rises", "Tesla falls"],
                "text": ["Apple stock up", "Tesla stock down"],
            }
            self.create_test_csv(csv_path, test_data)

            result = load_csv_dir(temp_dir)

            assert len(result) == 2
            assert list(result.columns) == [
                "date",
                "ticker",
                "source",
                "headline",
                "text",
            ]
            assert result["headline"].iloc[0] == "Apple rises"
            assert result["ticker"].iloc[1] == "TSLA"

    def test_load_csv_variant_column_names(self):
        """Test loading CSV with variant column names (case-insensitive matching)."""
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "variant.csv"

            # Create test data with variant column names
            test_data = {
                "Date": ["2024-01-01"],
                "Symbol": ["AAPL"],
                "Title": ["Apple news"],
            }
            self.create_test_csv(csv_path, test_data)

            result = load_csv_dir(temp_dir)

            assert len(result) == 1
            assert result["ticker"].iloc[0] == "AAPL"
            assert result["headline"].iloc[0] == "Apple news"

    def test_load_multiple_csv_files(self):
        """Test loading and combining multiple CSV files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create first CSV
            csv1_path = Path(temp_dir) / "file1.csv"
            data1 = {"headline": ["News 1"], "date": ["2024-01-01"]}
            self.create_test_csv(csv1_path, data1)

            # Create second CSV
            csv2_path = Path(temp_dir) / "file2.csv"
            data2 = {"headline": ["News 2"], "date": ["2024-01-02"]}
            self.create_test_csv(csv2_path, data2)

            result = load_csv_dir(temp_dir)

            assert len(result) == 2
            assert "News 1" in result["headline"].values
            assert "News 2" in result["headline"].values

    def test_load_csv_missing_text_column(self):
        """Test handling of CSV files without text content columns."""
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "no_text.csv"

            # Create CSV without text columns
            test_data = {"date": ["2024-01-01"], "volume": [1000]}
            self.create_test_csv(csv_path, test_data)

            result = load_csv_dir(temp_dir)

            # Should return empty DataFrame when no valid text columns found
            assert len(result) == 0

    def test_load_csv_malformed_file(self):
        """Test handling of malformed CSV files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a malformed CSV file
            bad_csv_path = Path(temp_dir) / "bad.csv"
            with open(bad_csv_path, "w") as f:
                f.write("This is not a valid CSV\nfile content")

            # Create a good CSV file
            good_csv_path = Path(temp_dir) / "good.csv"
            good_data = {"headline": ["Good news"]}
            self.create_test_csv(good_csv_path, good_data)

            result = load_csv_dir(temp_dir)

            # Should process the good file and skip the bad one
            assert len(result) == 1
            assert result["headline"].iloc[0] == "Good news"

    def test_load_empty_directory(self):
        """Test loading from directory with no CSV files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            result = load_csv_dir(temp_dir)

            assert len(result) == 0
            assert list(result.columns) == [
                "date",
                "ticker",
                "source",
                "headline",
                "text",
            ]

    def test_load_csv_with_nan_values(self):
        """Test handling of missing/NaN values in CSV data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "with_nans.csv"

            # Create data with missing values
            test_data = {
                "headline": ["Good news", None, "Bad news"],
                "ticker": ["AAPL", "TSLA", None],
                "date": ["2024-01-01", None, "2024-01-03"],
            }
            self.create_test_csv(csv_path, test_data)

            result = load_csv_dir(temp_dir)

            # Should drop rows where text (headline) is NaN
            assert len(result) == 2  # One row dropped due to NaN headline
            assert "Good news" in result["headline"].values
            assert "Bad news" in result["headline"].values


class TestNormalizeAndSave:
    """Test cases for data normalization and saving."""

    def test_normalize_basic_dataframe(self):
        """Test basic normalization without sector mapping."""
        test_data = pd.DataFrame(
            {
                "date": ["2024-01-01", "2024-01-02"],
                "ticker": ["AAPL", "TSLA"],
                "source": ["test.csv", "test.csv"],
                "headline": ["Apple news", "Tesla news"],
                "text": ["Apple text", "Tesla text"],
            }
        )

        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as temp_file:
            try:
                result = normalize_and_save(test_data, temp_file.name)

                # Check that data is normalized
                assert len(result) == 2
                assert result["date"].dtype == "object"  # date objects
                assert not result.isna().any().any()  # No NaN values

                # Check that file was saved
                assert os.path.exists(temp_file.name)

                # Verify saved data can be read back
                loaded_data = pd.read_parquet(temp_file.name)
                assert len(loaded_data) == 2

            finally:
                # Clean up
                if os.path.exists(temp_file.name):
                    os.unlink(temp_file.name)

    @patch("app.ingest._resolve_dir")
    def test_normalize_with_sector_mapping(self, mock_resolve):
        """Test normalization with sector mapping file."""
        # Mock sector mapping file
        sector_data = pd.DataFrame(
            {"ticker": ["AAPL", "TSLA"], "sector": ["Technology", "Automotive"]}
        )

        # Create a temporary sector mapping file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ) as sector_file:
            sector_data.to_csv(sector_file.name, index=False)

            # Mock the resolve function to return the temp file path
            mock_resolve.return_value = Path(sector_file.name)

            try:
                test_data = pd.DataFrame(
                    {
                        "date": ["2024-01-01", "2024-01-02"],
                        "ticker": ["AAPL", "TSLA"],
                        "source": ["test.csv", "test.csv"],
                        "headline": ["Apple news", "Tesla news"],
                        "text": ["Apple text", "Tesla text"],
                    }
                )

                with tempfile.NamedTemporaryFile(
                    suffix=".parquet", delete=False
                ) as output_file:
                    try:
                        result = normalize_and_save(test_data, output_file.name)

                        # Check that sector information was merged
                        assert "sector" in result.columns
                        assert result["sector"].iloc[0] == "Technology"
                        assert result["sector"].iloc[1] == "Automotive"

                    finally:
                        if os.path.exists(output_file.name):
                            os.unlink(output_file.name)

            finally:
                if os.path.exists(sector_file.name):
                    os.unlink(sector_file.name)

    def test_normalize_date_conversion(self):
        """Test proper conversion of date formats."""
        test_data = pd.DataFrame(
            {
                "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
                "ticker": ["AAPL", "TSLA", "MSFT"],
                "source": ["test", "test", "test"],
                "headline": ["News 1", "News 2", "News 3"],
                "text": ["Text 1", "Text 2", "Text 3"],
            }
        )

        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as temp_file:
            try:
                result = normalize_and_save(test_data, temp_file.name)

                # All dates should be converted to date objects
                assert all(
                    isinstance(
                        d, (type(pd.to_datetime("2024-01-01").date()), type(None))
                    )
                    for d in result["date"]
                )

            finally:
                if os.path.exists(temp_file.name):
                    os.unlink(temp_file.name)

    def test_normalize_preserves_original_data(self):
        """Test that normalization doesn't modify the original DataFrame."""
        original_data = pd.DataFrame(
            {
                "date": ["2024-01-01"],
                "ticker": ["AAPL"],
                "source": ["test"],
                "headline": ["News"],
                "text": ["Text"],
            }
        )

        original_copy = original_data.copy()

        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as temp_file:
            try:
                normalize_and_save(original_data, temp_file.name)

                # Original data should be unchanged
                pd.testing.assert_frame_equal(original_data, original_copy)

            finally:
                if os.path.exists(temp_file.name):
                    os.unlink(temp_file.name)
