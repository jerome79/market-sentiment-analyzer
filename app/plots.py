"""
Plotting utilities for market sentiment visualizations.
Includes functions for plotting sentiment trends by date, ticker, and sector.
"""

import matplotlib.pyplot as plt
import pandas as pd


def sentiment_trend_by_date(df: pd.DataFrame):
    """
    Plot average sentiment by date for the market.

    Parameters:
        df (pd.DataFrame): DataFrame with 'date' and 'sentiment'.

    Returns:
        plt.Figure: Figure of average sentiment by date.
    """
    g = df.groupby("date")["sentiment"].mean()
    fig, ax = plt.subplots()
    g.plot(ax=ax)
    ax.set_title("Average Sentiment — Market (by date)")
    ax.set_xlabel("date")
    ax.set_ylabel("avg sentiment (-1..1)")
    return fig


def sentiment_trend_by_ticker_date(df: pd.DataFrame, ticker: str):
    """
    Plot average sentiment by date for a specific ticker.

    Parameters:
        df (pd.DataFrame): DataFrame with 'date', 'ticker', 'sentiment'.
        ticker (str): Ticker to filter by.

    Returns:
        plt.Figure: Figure of average sentiment for ticker by date.
    """
    sdf = df[df["ticker"] == ticker]
    g = sdf.groupby("date")["sentiment"].mean()
    fig, ax = plt.subplots()
    g.plot(ax=ax)
    ax.set_title(f"Average Sentiment — {ticker} (by date)")
    ax.set_xlabel("date")
    ax.set_ylabel("avg sentiment (-1..1)")
    return fig


def sentiment_trend_by_sector_date(df: pd.DataFrame, sector: str):
    """
    Plot average sentiment by date for a specific sector.

    Parameters:
        df (pd.DataFrame): DataFrame with 'date', 'sector', 'sentiment'.
        sector (str): Sector to filter by.

    Returns:
        plt.Figure: Figure of average sentiment for sector by date.
    """
    sdf = df[df.get("sector").eq(sector)] if "sector" in df.columns else df.iloc[0:0]
    g = sdf.groupby("date")["sentiment"].mean()
    fig, ax = plt.subplots()
    g.plot(ax=ax)
    ax.set_title(f"Average Sentiment — {sector} (by date)")
    ax.set_xlabel("date")
    ax.set_ylabel("avg sentiment (-1..1)")
    return fig
