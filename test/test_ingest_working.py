"""Test ingest functionality with working tests."""
import os
import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
from market_sentiment_analyzer.ingest import SCHEMA


def test_schema_defined():
    """Test that SCHEMA is properly defined."""
    expected_columns = ["date", "ticker", "source", "headline", "text"]
    assert SCHEMA == expected_columns


def test_schema_has_required_columns():
    """Test that SCHEMA contains the required columns."""
    assert "date" in SCHEMA
    assert "ticker" in SCHEMA
    assert "text" in SCHEMA
    assert "headline" in SCHEMA
    assert "source" in SCHEMA


@patch('market_sentiment_analyzer.ingest.os.getenv')
def test_resolve_sector_map_env_var(mock_getenv):
    """Test sector map resolution from environment variable."""
    from market_sentiment_analyzer.ingest import _resolve_sector_map
    
    mock_getenv.return_value = "/test/path/sector.csv"
    result = _resolve_sector_map()
    
    expected = Path("/test/path/sector.csv")
    assert result == expected


@patch('market_sentiment_analyzer.ingest.os.getenv')
def test_resolve_sector_map_default(mock_getenv):
    """Test sector map resolution with default value."""
    from market_sentiment_analyzer.ingest import _resolve_sector_map
    
    mock_getenv.return_value = None
    result = _resolve_sector_map()
    
    # Should use default relative to ROOT
    assert result.name == "sector_map.csv"
    assert result.is_absolute()


def test_empty_dataframe_creation():
    """Test creating empty dataframe with SCHEMA."""
    df = pd.DataFrame(columns=SCHEMA)
    assert list(df.columns) == SCHEMA
    assert len(df) == 0


def test_ingest_text_col_mapping():
    """Test the text column mapping logic."""
    from market_sentiment_analyzer.ingest import _ingest_text_col
    
    # Test with headline column
    result = _ingest_text_col({"headline": "test"}, None, None)
    assert result == "test"
    
    # Test with title column  
    result = _ingest_text_col({"title": "test"}, None, None)
    assert result == "test"
    
    # Test with summary column
    result = _ingest_text_col({"summary": "test"}, None, None)
    assert result == "test"
    
    # Test with description column
    result = _ingest_text_col({"description": "test"}, None, None)
    assert result == "test"


def test_ingest_date_col_mapping():
    """Test the date column mapping logic."""
    from market_sentiment_analyzer.ingest import _ingest_date_col
    
    # Test with date column
    result = _ingest_date_col({"date": "2023-01-01"}, None, None)
    expected = pd.to_datetime("2023-01-01").date()
    assert result == expected
    
    # Test with time column
    result = _ingest_date_col({"time": "2023-01-01"}, None, None)
    expected = pd.to_datetime("2023-01-01").date()
    assert result == expected


def test_ingest_ticker_col_mapping():
    """Test the ticker column mapping logic."""
    from market_sentiment_analyzer.ingest import _ingest_ticker_col
    
    # Test with ticker column
    result = _ingest_ticker_col({"ticker": "AAPL"}, None, None)
    assert result == "AAPL"
    
    # Test with symbol column
    result = _ingest_ticker_col({"symbol": "MSFT"}, None, None)
    assert result == "MSFT"