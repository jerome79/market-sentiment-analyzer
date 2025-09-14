import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from market_sentiment_analyzer.plots import (
    sentiment_trend_by_date,
    sentiment_trend_by_sector_date,
    sentiment_trend_by_ticker_date,
)


def test_plot_market_by_date_returns_figure():
    df = pd.DataFrame({"date": ["2024-01-01", "2024-01-02"], "sentiment": [1, -1]})
    fig = sentiment_trend_by_date(df)
    assert hasattr(fig, "axes")
    plt.close(fig)


def test_plot_ticker_by_date_returns_figure():
    df = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-02", "2024-01-02"],
            "ticker": ["AAPL", "AAPL", "MSFT"],
            "sentiment": [1, 0, -1],
        }
    )
    fig = sentiment_trend_by_ticker_date(df, "AAPL")
    assert hasattr(fig, "axes")
    plt.close(fig)


def test_plot_sector_by_date_handles_missing_sector_column():
    df = pd.DataFrame({"date": ["2024-01-01"], "sentiment": [0]})
    fig = sentiment_trend_by_sector_date(df, "Tech")
    assert hasattr(fig, "axes")
    plt.close(fig)
