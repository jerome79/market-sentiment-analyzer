"""
Plotting utilities for market sentiment visualizations.

This module provides functions for creating sentiment trend plots by date,
ticker, and sector for market analysis visualization.
"""
import matplotlib.pyplot as plt
import pandas as pd


def sentiment_trend_by_date(dataframe: pd.DataFrame) -> plt.Figure:
    """
    Plot average sentiment by date for the overall market.

    Args:
        dataframe: DataFrame with 'date' and 'sentiment' columns.

    Returns:
        Figure object containing the sentiment trend plot.

    Raises:
        ValueError: If required columns are missing.

    Example:
        >>> fig = sentiment_trend_by_date(df)
        >>> fig.show()
    """
    if dataframe.empty:
        raise ValueError("Cannot plot empty DataFrame")

    required_columns = {"date", "sentiment"}
    if not required_columns.issubset(dataframe.columns):
        missing = required_columns - set(dataframe.columns)
        raise ValueError(f"Missing required columns: {missing}")

    grouped_sentiment = dataframe.groupby("date")["sentiment"].mean()
    figure, axis = plt.subplots(figsize=(10, 6))
    grouped_sentiment.plot(ax=axis)
    axis.set_title("Average Sentiment — Market (by date)")
    axis.set_xlabel("Date")
    axis.set_ylabel("Average Sentiment (-1..1)")
    axis.grid(True, alpha=0.3)
    return figure


def sentiment_trend_by_ticker_date(
    dataframe: pd.DataFrame, ticker: str
) -> plt.Figure:
    """
    Plot average sentiment by date for a specific ticker.

    Args:
        dataframe: DataFrame with 'date', 'ticker', and 'sentiment' columns.
        ticker: Ticker symbol to filter by.

    Returns:
        Figure object containing the ticker sentiment trend plot.

    Raises:
        ValueError: If required columns are missing or ticker not found.

    Example:
        >>> fig = sentiment_trend_by_ticker_date(df, 'AAPL')
        >>> fig.show()
    """
    if dataframe.empty:
        raise ValueError("Cannot plot empty DataFrame")

    required_columns = {"date", "ticker", "sentiment"}
    if not required_columns.issubset(dataframe.columns):
        missing = required_columns - set(dataframe.columns)
        raise ValueError(f"Missing required columns: {missing}")

    ticker_data = dataframe[dataframe["ticker"] == ticker]
    if ticker_data.empty:
        raise ValueError(f"No data found for ticker: {ticker}")

    grouped_sentiment = ticker_data.groupby("date")["sentiment"].mean()
    figure, axis = plt.subplots(figsize=(10, 6))
    grouped_sentiment.plot(ax=axis)
    axis.set_title(f"Average Sentiment — {ticker} (by date)")
    axis.set_xlabel("Date")
    axis.set_ylabel("Average Sentiment (-1..1)")
    axis.grid(True, alpha=0.3)
    return figure


def sentiment_trend_by_sector_date(
    dataframe: pd.DataFrame, sector: str
) -> plt.Figure:
    """
    Plot average sentiment by date for a specific sector.

    Args:
        dataframe: DataFrame with 'date', 'sector', and 'sentiment' columns.
        sector: Sector name to filter by.

    Returns:
        Figure object containing the sector sentiment trend plot.

    Raises:
        ValueError: If required columns are missing or sector not found.

    Example:
        >>> fig = sentiment_trend_by_sector_date(df, 'Technology')
        >>> fig.show()
    """
    if dataframe.empty:
        raise ValueError("Cannot plot empty DataFrame")

    required_columns = {"date", "sentiment"}
    if not required_columns.issubset(dataframe.columns):
        missing = required_columns - set(dataframe.columns)
        raise ValueError(f"Missing required columns: {missing}")

    # Handle missing sector column gracefully
    if "sector" not in dataframe.columns:
        sector_data = pd.DataFrame(columns=dataframe.columns)
    else:
        sector_data = dataframe[dataframe["sector"] == sector]

    if sector_data.empty:
        # Create empty plot with appropriate message
        figure, axis = plt.subplots(figsize=(10, 6))
        axis.text(
            0.5,
            0.5,
            f"No data available for sector: {sector}",
            transform=axis.transAxes,
            ha="center",
            va="center",
            fontsize=12,
        )
        axis.set_title(f"Average Sentiment — {sector} (by date)")
        return figure

    grouped_sentiment = sector_data.groupby("date")["sentiment"].mean()
    figure, axis = plt.subplots(figsize=(10, 6))
    grouped_sentiment.plot(ax=axis)
    axis.set_title(f"Average Sentiment — {sector} (by date)")
    axis.set_xlabel("Date")
    axis.set_ylabel("Average Sentiment (-1..1)")
    axis.grid(True, alpha=0.3)
    return figure
