import pandas as pd

from market_sentiment_analyzer.ui_streamlit import load_labeled_parquet, trend_market, trend_ticker


def test_load_labeled_parquet_missing_file():
    assert load_labeled_parquet("not_found.parquet") is None


def test_trend_market_empty():
    df = pd.DataFrame({"date": [], "sentiment": []})
    out = trend_market(df)
    assert "avg_sentiment" in out.columns


def test_trend_ticker_nonexistent():
    df = pd.DataFrame({"date": ["2024-01-01"], "ticker": ["AAPL"], "sentiment": [1]})
    out = trend_ticker(df, "MSFT")
    assert out.empty
