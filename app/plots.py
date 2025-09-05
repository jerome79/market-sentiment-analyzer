"""
Plotting utilities for market sentiment visualizations.

This module provides functions for creating visualizations of sentiment trends
across different dimensions: market-wide, ticker-specific, and sector-specific.
All functions return matplotlib Figure objects for easy integration with
Streamlit and other UI frameworks.

Functions:
    sentiment_trend_by_date: Plot market-wide sentiment trends over time.
    sentiment_trend_by_ticker_date: Plot sentiment trends for specific tickers.
    sentiment_trend_by_sector_date: Plot sentiment trends for specific sectors.

Typical usage example:
    import pandas as pd
    from app.plots import sentiment_trend_by_date

    # Load your labeled data
    df = pd.read_parquet("data/news_labeled.parquet")

    # Create market trend plot
    fig = sentiment_trend_by_date(df)
    fig.savefig("market_sentiment_trend.png")
"""

import matplotlib.pyplot as plt
import pandas as pd


def sentiment_trend_by_date(df: pd.DataFrame) -> plt.Figure:
    """
    Plot average sentiment by date for the entire market.

    This function aggregates sentiment scores across all news items for each
    date and displays the market-wide sentiment trend over time. Useful for
    identifying overall market mood and significant sentiment shifts.

    Args:
        df: DataFrame containing at least 'date' and 'sentiment' columns.
            Sentiment values should be numeric (typically -1 to 1).

    Returns:
        matplotlib Figure object with the sentiment trend plot.

    Example:
        >>> import pandas as pd
        >>> df = pd.DataFrame({
        ...     'date': ['2024-01-01', '2024-01-02', '2024-01-03'],
        ...     'sentiment': [0.2, -0.1, 0.5]
        ... })
        >>> fig = sentiment_trend_by_date(df)
        >>> fig.savefig('market_trend.png')
    """
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Handle empty DataFrame
    if df.empty or len(df) == 0:
        ax.text(
            0.5,
            0.5,
            "No data available to plot",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=12,
        )
    else:
        # Group by date and calculate mean sentiment
        daily_sentiment = df.groupby("date")["sentiment"].mean()

        if not daily_sentiment.empty:
            daily_sentiment.plot(ax=ax, linewidth=2, marker="o", markersize=4)

    # Customize the plot
    ax.set_title("Average Sentiment — Market (by date)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Average Sentiment (-1 to 1)", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color="red", linestyle="--", alpha=0.5, label="Neutral")

    # Improve date formatting on x-axis
    plt.xticks(rotation=45)
    plt.tight_layout()

    return fig


def sentiment_trend_by_ticker_date(df: pd.DataFrame, ticker: str) -> plt.Figure:
    """
    Plot average sentiment by date for a specific ticker symbol.

    This function filters the data to a specific ticker and shows how sentiment
    for that ticker has evolved over time. Useful for tracking investor
    sentiment toward individual companies or assets.

    Args:
        df: DataFrame containing 'date', 'ticker', and 'sentiment' columns.
        ticker: Stock ticker symbol to filter and analyze (e.g., 'AAPL', 'TSLA').

    Returns:
        matplotlib Figure object with the ticker-specific sentiment trend.

    Example:
        >>> df = pd.DataFrame({
        ...     'date': ['2024-01-01', '2024-01-01', '2024-01-02'],
        ...     'ticker': ['AAPL', 'TSLA', 'AAPL'],
        ...     'sentiment': [0.3, -0.2, 0.1]
        ... })
        >>> fig = sentiment_trend_by_ticker_date(df, 'AAPL')
        >>> fig.savefig('aapl_sentiment.png')
    """
    # Filter data for the specific ticker
    ticker_data = df[df["ticker"] == ticker]

    if ticker_data.empty:
        # Create empty plot with informative message
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(
            0.5,
            0.5,
            f"No data available for ticker: {ticker}",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=12,
        )
        ax.set_title(f"Average Sentiment — {ticker} (by date)")
        return fig

    # Group by date and calculate mean sentiment
    daily_sentiment = ticker_data.groupby("date")["sentiment"].mean()

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    daily_sentiment.plot(ax=ax, linewidth=2, marker="o", markersize=4, color="green")

    # Customize the plot
    ax.set_title(
        f"Average Sentiment — {ticker} (by date)", fontsize=14, fontweight="bold"
    )
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Average Sentiment (-1 to 1)", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color="red", linestyle="--", alpha=0.5, label="Neutral")

    # Improve date formatting on x-axis
    plt.xticks(rotation=45)
    plt.tight_layout()

    return fig


def sentiment_trend_by_sector_date(df: pd.DataFrame, sector: str) -> plt.Figure:
    """
    Plot average sentiment by date for a specific sector.

    This function filters the data to a specific sector and shows sentiment
    trends for that sector over time. Requires sector information to be
    available in the DataFrame (typically merged from a sector mapping file).

    Args:
        df: DataFrame containing 'date', 'sector', and 'sentiment' columns.
        sector: Sector name to filter and analyze (e.g., 'Technology', 'Healthcare').

    Returns:
        matplotlib Figure object with the sector-specific sentiment trend.

    Example:
        >>> df = pd.DataFrame({
        ...     'date': ['2024-01-01', '2024-01-01', '2024-01-02'],
        ...     'sector': ['Technology', 'Healthcare', 'Technology'],
        ...     'sentiment': [0.2, -0.1, 0.4]
        ... })
        >>> fig = sentiment_trend_by_sector_date(df, 'Technology')
        >>> fig.savefig('tech_sentiment.png')
    """
    # Filter data for the specific sector (handle missing sector column)
    if "sector" in df.columns:
        sector_data = df[df["sector"] == sector]
    else:
        sector_data = pd.DataFrame()  # Empty DataFrame if no sector column

    if sector_data.empty:
        # Create empty plot with informative message
        fig, ax = plt.subplots(figsize=(10, 6))
        message = (
            f"No data available for sector: {sector}"
            if "sector" in df.columns
            else "No sector information available"
        )
        ax.text(
            0.5,
            0.5,
            message,
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=12,
        )
        ax.set_title(f"Average Sentiment — {sector} (by date)")
        return fig

    # Group by date and calculate mean sentiment
    daily_sentiment = sector_data.groupby("date")["sentiment"].mean()

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    daily_sentiment.plot(ax=ax, linewidth=2, marker="o", markersize=4, color="blue")

    # Customize the plot
    ax.set_title(
        f"Average Sentiment — {sector} (by date)", fontsize=14, fontweight="bold"
    )
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Average Sentiment (-1 to 1)", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color="red", linestyle="--", alpha=0.5, label="Neutral")

    # Improve date formatting on x-axis
    plt.xticks(rotation=45)
    plt.tight_layout()

    return fig
