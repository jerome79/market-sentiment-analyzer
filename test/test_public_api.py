# test/test_public_api.py

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from app import public_api


@pytest.fixture
def sample_panel() -> pd.DataFrame:
    """
    Fixture that returns a sample pandas DataFrame representing panel data
    with columns: date, ticker, and avg_sentiment.
    """
    return pd.DataFrame({"date": ["2024-06-01", "2024-06-02", "2024-06-03"], "ticker": ["AAPL", "GOOG", "AAPL"], "avg_sentiment": [0.5, -0.2, 0.7]})


def test_load_panel(monkeypatch, sample_panel) -> None:
    """
    Test the load_panel function to ensure it loads a DataFrame from a parquet file.

    Args:
        monkeypatch: pytest fixture to mock pandas.read_parquet.
        sample_panel: pandas DataFrame fixture with sample panel data.
    """
    monkeypatch.setattr(pd, "read_parquet", lambda path: sample_panel.copy())
    df = public_api.load_panel("dummy_path")
    assert isinstance(df, pd.DataFrame)


def test_panel_stats(monkeypatch, sample_panel) -> None:
    """
    Test the panel_stats function to ensure it returns correct stats and series
    for a given ticker and date range.
    """
    monkeypatch.setattr(pd, "read_parquet", lambda path: sample_panel.copy())
    result = public_api.panel_stats(["AAPL"], "2024-06-01", "2024-06-03", "dummy_path")
    assert isinstance(result, dict)
    assert "stats" in result and "series" in result
    stats = result["stats"]
    assert stats["tickers"] == ["AAPL"]
    assert stats["n_news"] == 2
    assert isinstance(stats["avg_sentiment"], float)
    assert all(item["ticker"] == "AAPL" for item in result["series"])
