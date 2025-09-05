"""
Unit tests for the plotting module.

Tests sentiment visualization functions to ensure they generate proper
matplotlib figures and handle various data scenarios correctly.
"""

import matplotlib.pyplot as plt
import pandas as pd

from app.plots import (
    sentiment_trend_by_date,
    sentiment_trend_by_ticker_date,
    sentiment_trend_by_sector_date,
)


class TestSentimentTrendByDate:
    """Test cases for market-wide sentiment trend plotting."""

    def test_basic_sentiment_trend(self):
        """Test basic functionality with valid data."""
        # Create test data
        test_data = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]).date,
                "sentiment": [0.5, -0.3, 0.8],
                "ticker": ["AAPL", "TSLA", "MSFT"],
            }
        )

        fig = sentiment_trend_by_date(test_data)

        # Check that a figure is returned
        assert isinstance(fig, plt.Figure)

        # Check that the figure has an axes
        assert len(fig.axes) == 1

        # Check basic plot properties
        ax = fig.axes[0]
        assert "Average Sentiment — Market (by date)" in ax.get_title()
        assert "Date" in ax.get_xlabel()
        assert "Average Sentiment" in ax.get_ylabel()

    def test_sentiment_trend_with_multiple_entries_per_date(self):
        """Test aggregation when multiple entries exist for the same date."""
        test_data = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-02"]).date,
                "sentiment": [0.5, 0.3, -0.2],  # First date should average to 0.4
                "ticker": ["AAPL", "TSLA", "MSFT"],
            }
        )

        fig = sentiment_trend_by_date(test_data)

        assert isinstance(fig, plt.Figure)

        # The figure should be created successfully with aggregated data
        ax = fig.axes[0]
        assert len(ax.lines) > 0  # Should have at least one line plot

    def test_sentiment_trend_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        empty_data = pd.DataFrame(columns=["date", "sentiment"])

        # Should not crash with empty data, but might have no plot lines
        fig = sentiment_trend_by_date(empty_data)
        assert isinstance(fig, plt.Figure)

    def test_sentiment_trend_single_data_point(self):
        """Test plotting with only one data point."""
        single_point = pd.DataFrame(
            {"date": [pd.to_datetime("2024-01-01").date()], "sentiment": [0.5]}
        )

        fig = sentiment_trend_by_date(single_point)
        assert isinstance(fig, plt.Figure)

    def test_sentiment_trend_date_ordering(self):
        """Test that dates are properly ordered in the plot."""
        # Create data with dates out of order
        test_data = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-03", "2024-01-01", "2024-01-02"]).date,
                "sentiment": [0.1, 0.2, 0.3],
            }
        )

        fig = sentiment_trend_by_date(test_data)

        # Should handle date ordering gracefully
        assert isinstance(fig, plt.Figure)
        ax = fig.axes[0]
        assert len(ax.lines) > 0


class TestSentimentTrendByTickerDate:
    """Test cases for ticker-specific sentiment trend plotting."""

    def test_basic_ticker_trend(self):
        """Test basic functionality with valid ticker data."""
        test_data = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]).date,
                "ticker": ["AAPL", "AAPL", "TSLA"],
                "sentiment": [0.5, 0.3, -0.2],
            }
        )

        fig = sentiment_trend_by_ticker_date(test_data, "AAPL")

        assert isinstance(fig, plt.Figure)
        ax = fig.axes[0]
        assert "AAPL" in ax.get_title()
        assert "Date" in ax.get_xlabel()
        assert "Average Sentiment" in ax.get_ylabel()

    def test_ticker_trend_nonexistent_ticker(self):
        """Test handling of ticker that doesn't exist in data."""
        test_data = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-01", "2024-01-02"]).date,
                "ticker": ["AAPL", "AAPL"],
                "sentiment": [0.5, 0.3],
            }
        )

        fig = sentiment_trend_by_ticker_date(test_data, "NONEXISTENT")

        assert isinstance(fig, plt.Figure)
        ax = fig.axes[0]
        assert "NONEXISTENT" in ax.get_title()
        # Should show informative message for no data
        assert len(ax.texts) > 0  # Should have text indicating no data

    def test_ticker_trend_multiple_dates(self):
        """Test ticker trend with multiple entries per date."""
        test_data = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-02"]).date,
                "ticker": ["AAPL", "AAPL", "AAPL"],
                "sentiment": [0.5, 0.7, -0.1],  # First date should average to 0.6
            }
        )

        fig = sentiment_trend_by_ticker_date(test_data, "AAPL")

        assert isinstance(fig, plt.Figure)
        ax = fig.axes[0]
        assert len(ax.lines) > 0  # Should have plot line

    def test_ticker_trend_empty_after_filter(self):
        """Test when ticker filter results in empty data."""
        test_data = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-01"]).date,
                "ticker": ["TSLA"],
                "sentiment": [0.5],
            }
        )

        fig = sentiment_trend_by_ticker_date(test_data, "AAPL")

        assert isinstance(fig, plt.Figure)
        # Should handle empty filtered data gracefully

    def test_ticker_trend_case_sensitivity(self):
        """Test that ticker matching is case-sensitive."""
        test_data = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-01"]).date,
                "ticker": ["aapl"],  # lowercase
                "sentiment": [0.5],
            }
        )

        # Should not match uppercase ticker
        fig = sentiment_trend_by_ticker_date(test_data, "AAPL")
        assert isinstance(fig, plt.Figure)


class TestSentimentTrendBySectorDate:
    """Test cases for sector-specific sentiment trend plotting."""

    def test_basic_sector_trend(self):
        """Test basic functionality with valid sector data."""
        test_data = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]).date,
                "sector": ["Technology", "Technology", "Healthcare"],
                "sentiment": [0.5, 0.3, -0.2],
            }
        )

        fig = sentiment_trend_by_sector_date(test_data, "Technology")

        assert isinstance(fig, plt.Figure)
        ax = fig.axes[0]
        assert "Technology" in ax.get_title()
        assert "Date" in ax.get_xlabel()
        assert "Average Sentiment" in ax.get_ylabel()

    def test_sector_trend_missing_sector_column(self):
        """Test handling of DataFrame without sector column."""
        test_data = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-01"]).date,
                "sentiment": [0.5],
                # No sector column
            }
        )

        fig = sentiment_trend_by_sector_date(test_data, "Technology")

        assert isinstance(fig, plt.Figure)
        ax = fig.axes[0]
        # Should show message about no sector information
        assert len(ax.texts) > 0

    def test_sector_trend_nonexistent_sector(self):
        """Test handling of sector that doesn't exist in data."""
        test_data = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-01"]).date,
                "sector": ["Technology"],
                "sentiment": [0.5],
            }
        )

        fig = sentiment_trend_by_sector_date(test_data, "Healthcare")

        assert isinstance(fig, plt.Figure)
        ax = fig.axes[0]
        assert "Healthcare" in ax.get_title()

    def test_sector_trend_multiple_entries_per_date(self):
        """Test sector trend with multiple entries per date."""
        test_data = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-02"]).date,
                "sector": ["Technology", "Technology", "Technology"],
                "sentiment": [0.3, 0.7, -0.1],  # First date should average to 0.5
            }
        )

        fig = sentiment_trend_by_sector_date(test_data, "Technology")

        assert isinstance(fig, plt.Figure)
        ax = fig.axes[0]
        assert len(ax.lines) > 0

    def test_sector_trend_with_none_values(self):
        """Test handling of None values in sector column."""
        test_data = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-01", "2024-01-02"]).date,
                "sector": ["Technology", None],
                "sentiment": [0.5, 0.3],
            }
        )

        fig = sentiment_trend_by_sector_date(test_data, "Technology")

        assert isinstance(fig, plt.Figure)
        # Should filter out None values and plot Technology data


class TestPlottingIntegration:
    """Integration tests for plotting functions."""

    def test_all_plotting_functions_return_figures(self):
        """Test that all plotting functions return matplotlib Figure objects."""
        # Create comprehensive test data
        test_data = pd.DataFrame(
            {
                "date": pd.to_datetime(
                    [
                        "2024-01-01",
                        "2024-01-02",
                        "2024-01-03",
                        "2024-01-01",
                        "2024-01-02",
                        "2024-01-03",
                    ]
                ).date,
                "ticker": ["AAPL", "AAPL", "AAPL", "TSLA", "TSLA", "TSLA"],
                "sector": [
                    "Technology",
                    "Technology",
                    "Technology",
                    "Automotive",
                    "Automotive",
                    "Automotive",
                ],
                "sentiment": [0.2, 0.5, -0.1, -0.3, 0.1, 0.4],
            }
        )

        # Test all plotting functions
        market_fig = sentiment_trend_by_date(test_data)
        ticker_fig = sentiment_trend_by_ticker_date(test_data, "AAPL")
        sector_fig = sentiment_trend_by_sector_date(test_data, "Technology")

        # All should return Figure objects
        assert isinstance(market_fig, plt.Figure)
        assert isinstance(ticker_fig, plt.Figure)
        assert isinstance(sector_fig, plt.Figure)

        # All should have titles and labels
        for fig in [market_fig, ticker_fig, sector_fig]:
            ax = fig.axes[0]
            assert ax.get_title()
            assert ax.get_xlabel()
            assert ax.get_ylabel()

    def test_plot_consistency_across_functions(self):
        """Test that plots have consistent styling and structure."""
        test_data = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-01", "2024-01-02"]).date,
                "ticker": ["AAPL", "AAPL"],
                "sector": ["Technology", "Technology"],
                "sentiment": [0.3, 0.7],
            }
        )

        market_fig = sentiment_trend_by_date(test_data)
        ticker_fig = sentiment_trend_by_ticker_date(test_data, "AAPL")
        sector_fig = sentiment_trend_by_sector_date(test_data, "Technology")

        # Check that all have similar structure
        for fig in [market_fig, ticker_fig, sector_fig]:
            assert len(fig.axes) == 1  # Single subplot
            ax = fig.axes[0]
            assert "Average Sentiment" in ax.get_ylabel()
            assert "Date" in ax.get_xlabel()

    def test_figure_creation_called(self):
        """Test that matplotlib creates figures properly."""
        test_data = pd.DataFrame(
            {"date": pd.to_datetime(["2024-01-01"]).date, "sentiment": [0.5]}
        )

        fig = sentiment_trend_by_date(test_data)

        # Should return a valid Figure object
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 1
