"""
Streamlit user interface for the Market Sentiment Analyzer.

This module provides a web-based dashboard for ingesting financial news data,
applying sentiment analysis, and visualizing sentiment trends. The interface
includes two main tabs: data ingestion/labeling and sentiment analytics dashboard.

Main components:
- File upload and CSV folder ingestion
- Model selection for sentiment analysis (VADER, RoBERTa, FinBERT)
- Interactive sentiment trend visualizations
- Data export functionality

Usage:
    streamlit run app/ui_streamlit.py
"""

import io
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# Repository root path setup for module imports
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.ingest import load_csv_dir, normalize_and_save


# Application configuration
load_dotenv()
st.set_page_config(page_title="Market Sentiment Analyzer", layout="wide")
st.set_option("client.showErrorDetails", True)


@st.cache_resource
def _get_hf_model(model_id: str) -> Any:
    """
    Lazily load and cache a HuggingFace sentiment model.

    Uses Streamlit's caching to avoid reloading models on every interaction.
    This significantly improves performance for transformer-based models.

    Args:
        model_id: HuggingFace model identifier (e.g., 'ProsusAI/finbert').

    Returns:
        Instantiated HFClassifier for sentiment prediction.

    Example:
        >>> model = _get_hf_model("ProsusAI/finbert")
        >>> sentiments = model.predict(["Great earnings", "Market volatility"])
    """
    from app.sentiment import HFClassifier

    return HFClassifier(model_id)


def _select_sentiment_model(model_choice: str, hf_model_id: str) -> Any:
    """
    Select and instantiate the appropriate sentiment model.

    Args:
        model_choice: User-selected model name from the UI dropdown.
        hf_model_id: HuggingFace model ID for transformer-based models.

    Returns:
        Instantiated sentiment model with predict() method.

    Example:
        >>> model = _select_sentiment_model("VADER (fast)", "")
        >>> model = _select_sentiment_model("FinBERT (ProsusAI)", "ProsusAI/finbert")
    """
    from app.sentiment import BaselineVader

    if model_choice == "VADER (fast)":
        return BaselineVader()
    else:
        return _get_hf_model(hf_model_id)


def _apply_sentiment_labeling(
    dataframe: pd.DataFrame, sentiment_model: Any
) -> pd.DataFrame:
    """
    Apply sentiment analysis to a DataFrame containing text data.

    This function handles text column detection, applies the sentiment model,
    and adds sentiment scores (and confidence if available) to the DataFrame.

    Args:
        dataframe: Input DataFrame with text content.
        sentiment_model: Model instance with predict() or predict_with_scores().

    Returns:
        DataFrame with added 'sentiment' and optionally 'confidence' columns.

    Raises:
        ValueError: If no text-like column is found in the DataFrame.

    Example:
        >>> df = pd.DataFrame({'headline': ['Good news', 'Bad news']})
        >>> model = BaselineVader()
        >>> labeled_df = _apply_sentiment_labeling(df, model)
        >>> print(labeled_df['sentiment'].tolist())  # [1, -1]
    """
    # Create copy to avoid modifying original data
    labeled_df = dataframe.copy()

    # Auto-detect text column if not explicitly named 'text'
    if "text" not in labeled_df.columns:
        # Look for columns that likely contain text content
        text_candidates = [
            col
            for col in labeled_df.columns
            if any(keyword in col.lower() for keyword in ("headline", "title", "text"))
        ]
        if not text_candidates:
            raise ValueError(
                "No text-like column found in DataFrame. "
                "Expected columns containing 'headline', 'title', or 'text'."
            )
        # Use the first candidate as the text column
        labeled_df["text"] = labeled_df[text_candidates[0]].astype(str)

    # Prepare text data for model input
    text_content = labeled_df["text"].fillna("").astype(str).tolist()

    # Apply sentiment analysis
    if hasattr(sentiment_model, "predict_with_scores"):
        # Model supports confidence scores
        sentiment_labels, confidence_scores = sentiment_model.predict_with_scores(
            text_content
        )
        labeled_df["sentiment"] = sentiment_labels
        labeled_df["confidence"] = confidence_scores
    else:
        # Model only provides labels
        labeled_df["sentiment"] = sentiment_model.predict(text_content)

    return labeled_df


def resolve_data_directory(env_variable: str = "NEWS_CSV_DIR") -> Path:
    """
    Resolve data directory path from environment variables.

    Converts relative paths to absolute paths using the repository root,
    and handles environment variable defaults gracefully.

    Args:
        env_variable: Environment variable name containing the directory path.

    Returns:
        Absolute Path object pointing to the resolved directory.

    Example:
        >>> data_dir = resolve_data_directory("NEWS_CSV_DIR")
        >>> print(data_dir.exists())  # True if directory exists
    """
    # Get directory from environment or use default
    directory_value = os.getenv(env_variable) or "data"
    directory_path = Path(directory_value)

    # Convert relative paths to absolute using repository root
    if not directory_path.is_absolute():
        directory_path = REPO_ROOT / directory_path

    return directory_path.resolve()


def show_debug_information() -> None:
    """
    Display debug information in the sidebar.

    Shows current working directory, environment variables, and system info
    to help users troubleshoot configuration issues.
    """
    st.header("🔧 Debug Information")
    st.caption("System and environment details for troubleshooting.")

    st.write("**Current Working Directory:**", os.getcwd())
    st.write("**Python Version:**", sys.version.split()[0])

    st.subheader("Environment Variables")
    env_vars = ["NEWS_CSV_DIR", "SECTOR_MAP_CSV", "SENTIMENT_MODEL"]
    for var in env_vars:
        value = os.getenv(var, "(not set)")
        st.write(f"**{var}:**", value)


# Main UI Layout
st.title("📈 Market Sentiment Analyzer")

# Sidebar with debug information
with st.sidebar:
    show_debug_information()

# Main application tabs
tab_ingest, tab_dashboard = st.tabs(["📁 Ingest & Label", "📊 Dashboard"])

# Data Ingestion and Labeling Tab
with tab_ingest:
    st.subheader("Data Ingestion and Sentiment Labeling")
    st.markdown(
        "Upload CSV files or specify a folder containing news data. "
        "The system will automatically detect relevant columns and apply "
        "sentiment analysis using your selected model."
    )

    with st.form("data_ingestion_form", clear_on_submit=False):
        # File upload option
        uploaded_files = st.file_uploader(
            "Upload CSV files (with headline/text + optional ticker/date columns)",
            type=["csv"],
            accept_multiple_files=True,
            key="csv_uploader",
            help="Select one or more CSV files containing financial news data",
        )

        # Folder input option
        folder_path = st.text_input(
            "Or specify a folder path containing CSV files",
            value=str(resolve_data_directory("NEWS_CSV_DIR")),
            key="folder_path_input",
            help="Path will be resolved relative to repository root if not absolute",
        )

        # Model selection
        model_selection = st.selectbox(
            "Choose sentiment analysis model",
            [
                "VADER (fast)",
                "RoBERTa (CardiffNLP)",
                "FinBERT (ProsusAI)",
                "FinBERT (Tone)",
            ],
            index=2,
            key="model_selection",
            help=(
                "VADER is fastest, FinBERT models are most accurate for "
                "financial text"
            ),
        )

        # Map model choices to HuggingFace model IDs
        model_id_mapping = {
            "RoBERTa (CardiffNLP)": "cardiffnlp/twitter-roberta-base-sentiment-latest",
            "FinBERT (ProsusAI)": "ProsusAI/finbert",
            "FinBERT (Tone)": "yiyanghkust/finbert-tone",
        }
        selected_model_id = model_id_mapping.get(
            model_selection, os.getenv("SENTIMENT_MODEL", "ProsusAI/finbert")
        )

        # Submit button
        submit_button = st.form_submit_button(
            "🚀 Process Data",
            use_container_width=True,
            help="Load data and apply sentiment analysis",
        )

    # Status information
    file_count = 0 if not uploaded_files else len(uploaded_files)
    st.caption(
        f"📋 Status: {file_count} uploaded files | "
        f"Folder: {folder_path} | Model: {model_selection}"
    )

    # Process data when form is submitted
    if submit_button:
        try:
            with st.spinner("Processing data..."):
                # Load data from either uploads or folder
                if uploaded_files:
                    st.info("Loading data from uploaded files...")
                    data_frames = []

                    for uploaded_file in uploaded_files:
                        try:
                            current_df = pd.read_csv(uploaded_file)
                            data_frames.append(current_df)
                        except Exception as file_error:
                            file_name = getattr(uploaded_file, "name", "unknown")
                            st.warning(f"Could not read file {file_name}: {file_error}")

                    if not data_frames:
                        st.error(
                            "❌ No valid CSV files could be processed. "
                            "Please verify file format, encoding (UTF-8 recommended), "
                            "and column headers."
                        )
                        st.stop()

                    # Combine all uploaded files
                    combined_raw_data = pd.concat(data_frames, ignore_index=True)

                    # Auto-detect columns
                    column_names_lower = [
                        col.lower() for col in combined_raw_data.columns
                    ]

                    # Find text content column
                    text_column_index = next(
                        (
                            i
                            for i, col in enumerate(column_names_lower)
                            if any(
                                keyword in col
                                for keyword in ["headline", "title", "text"]
                            )
                        ),
                        None,
                    )

                    if text_column_index is None:
                        st.error(
                            "❌ No text content column found. "
                            "Please ensure your CSV has a column containing "
                            "'headline', 'title', or 'text' in the name."
                        )
                        st.stop()

                    # Find optional date and ticker columns
                    date_column_index = next(
                        (
                            i
                            for i, col in enumerate(column_names_lower)
                            if any(keyword in col for keyword in ["date", "time"])
                        ),
                        None,
                    )

                    ticker_column_index = next(
                        (
                            i
                            for i, col in enumerate(column_names_lower)
                            if any(keyword in col for keyword in ["ticker", "symbol"])
                        ),
                        None,
                    )

                    # Create standardized DataFrame
                    standardized_data = pd.DataFrame(
                        {
                            "date": (
                                pd.to_datetime(
                                    combined_raw_data.iloc[:, date_column_index],
                                    errors="coerce",
                                ).dt.date
                                if date_column_index is not None
                                else None
                            ),
                            "ticker": (
                                combined_raw_data.iloc[:, ticker_column_index]
                                if ticker_column_index is not None
                                else None
                            ),
                            "source": "upload",
                            "headline": combined_raw_data.iloc[:, text_column_index],
                            "text": combined_raw_data.iloc[:, text_column_index],
                        }
                    )

                else:
                    # Load from folder
                    st.info(f"Loading data from folder: {folder_path}")
                    standardized_data = load_csv_dir(folder_path)

                    if standardized_data.empty:
                        st.error(
                            "❌ No data loaded from the specified folder. "
                            "Please check the path and ensure CSV files are present."
                        )
                        st.stop()

                # Save normalized version
                os.makedirs("data", exist_ok=True)
                normalized_data = normalize_and_save(
                    standardized_data, "data/news.parquet"
                )

                # Apply sentiment analysis
                st.info(f"Applying sentiment analysis using {model_selection}...")
                sentiment_model = _select_sentiment_model(
                    model_selection, selected_model_id
                )
                labeled_data = _apply_sentiment_labeling(
                    normalized_data, sentiment_model
                )

                # Save labeled dataset
                labeled_data.to_parquet("data/news_labeled.parquet", index=False)

                # Show success message and preview
                st.success(
                    f"✅ Successfully processed {len(labeled_data)} rows! "
                    f"Data saved to: data/news_labeled.parquet"
                )

                st.subheader("Data Preview")
                st.dataframe(
                    labeled_data.head(20), use_container_width=True, height=400
                )

                # Show summary statistics
                if "sentiment" in labeled_data.columns:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        positive_count = (labeled_data["sentiment"] == 1).sum()
                        st.metric("Positive Sentiment", positive_count)
                    with col2:
                        neutral_count = (labeled_data["sentiment"] == 0).sum()
                        st.metric("Neutral Sentiment", neutral_count)
                    with col3:
                        negative_count = (labeled_data["sentiment"] == -1).sum()
                        st.metric("Negative Sentiment", negative_count)

        except Exception as processing_error:
            st.error(f"❌ Error during processing: {processing_error}")
            st.exception(processing_error)
            st.stop()

# Dashboard Tab
with tab_dashboard:
    st.subheader("📊 Sentiment Analytics Dashboard")
    st.markdown(
        "Visualize sentiment trends across different dimensions. "
        "Data must be processed in the Ingest & Label tab first."
    )

    try:
        # Check if labeled data exists
        labeled_data_path = "data/news_labeled.parquet"
        if os.path.exists(labeled_data_path):
            # Load labeled data
            dashboard_df = pd.read_parquet(labeled_data_path)

            # Ensure sector column exists for sector analysis
            if "sector" not in dashboard_df.columns:
                dashboard_df["sector"] = None

            # Aggregation mode selection
            aggregation_mode = st.selectbox(
                "📈 Choose aggregation method",
                ["Market (Date)", "Ticker + Date", "Sector + Date"],
                index=0,
                key="aggregation_mode",
                help="Select how to group and visualize sentiment data",
            )

            # Import plotting functions (lazy import for better performance)
            from app.plots import (
                sentiment_trend_by_date,
                sentiment_trend_by_ticker_date,
                sentiment_trend_by_sector_date,
            )

            # Generate appropriate visualization
            if aggregation_mode == "Market (Date)":
                st.subheader("Market-Wide Sentiment Trend")
                market_chart = sentiment_trend_by_date(dashboard_df)
                st.pyplot(market_chart)

            elif aggregation_mode == "Ticker + Date":
                st.subheader("Ticker-Specific Sentiment Trends")

                # Get available tickers
                available_tickers = sorted(
                    [
                        ticker
                        for ticker in dashboard_df["ticker"].dropna().unique()
                        if ticker
                    ]
                )

                if available_tickers:
                    selected_ticker = st.selectbox(
                        "Choose ticker symbol", available_tickers, key="ticker_selector"
                    )
                    ticker_chart = sentiment_trend_by_ticker_date(
                        dashboard_df, selected_ticker
                    )
                    st.pyplot(ticker_chart)
                else:
                    st.info(
                        "ℹ️ No ticker information found in the labeled data. "
                        "Upload data with ticker/symbol columns to enable "
                        "ticker analysis."
                    )

            else:  # Sector + Date
                st.subheader("Sector-Specific Sentiment Trends")

                # Get available sectors
                available_sectors = sorted(
                    [
                        sector
                        for sector in dashboard_df["sector"].dropna().unique()
                        if sector
                    ]
                )

                if available_sectors:
                    selected_sector = st.selectbox(
                        "Choose sector", available_sectors, key="sector_selector"
                    )
                    sector_chart = sentiment_trend_by_sector_date(
                        dashboard_df, selected_sector
                    )
                    st.pyplot(sector_chart)
                else:
                    st.info(
                        "ℹ️ No sector information available. "
                        "Provide a sector mapping CSV file (see .env.example) "
                        "to enable sector-based analysis."
                    )

            # Data export functionality
            st.subheader("📤 Export Data")
            csv_buffer = io.StringIO()
            dashboard_df.to_csv(csv_buffer, index=False)

            st.download_button(
                label="⬇️ Download Labeled Data as CSV",
                data=csv_buffer.getvalue(),
                file_name="news_sentiment_labeled.csv",
                mime="text/csv",
                key="export_csv_button",
                help="Download the processed data with sentiment labels",
            )

        else:
            # No labeled data available
            st.info(
                "ℹ️ No labeled data available yet. "
                "Please process some data in the **Ingest & Label** tab first."
            )

    except Exception as dashboard_error:
        st.error(f"❌ Dashboard error: {dashboard_error}")
        st.exception(dashboard_error)
