"""
Streamlit UI for Market Sentiment Analyzer.

This module provides a web interface for uploading CSV files, analyzing sentiment,
and visualizing sentiment trends across different aggregation levels.
"""
import io
import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# -------- Add repo root to path for local imports --------
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from app.ingest import load_csv_dir, normalize_and_save  # noqa: E402


# -------------------- App config --------------------
load_dotenv()
st.set_page_config(page_title="Market Sentiment Analyzer", layout="wide")
st.set_option("client.showErrorDetails", True)

# -------------------- Helpers --------------------


@st.cache_resource
def _get_huggingface_model(model_id: str):
    """
    Lazily load and cache a HuggingFace sentiment model by model ID.

    Args:
        model_id: HuggingFace model identifier.

    Returns:
        HuggingFaceClassifier: Instantiated classifier.
    """
    from app.sentiment import HuggingFaceClassifier

    # Cache only the HF model (VADER is lightweight)
    return HuggingFaceClassifier(model_id)


def _choose_sentiment_model(model_choice: str, huggingface_id: str):
    """
    Select the sentiment model based on user choice.

    Args:
        model_choice: Model name selected by user.
        huggingface_id: HuggingFace model ID for non-VADER models.

    Returns:
        Model instance for sentiment prediction.
    """
    from app.sentiment import BaselineVader

    if model_choice == "VADER (fast)":
        return BaselineVader()
    else:
        return _get_huggingface_model(huggingface_id)


def _label_dataframe(dataframe: pd.DataFrame, model) -> pd.DataFrame:
    """
    Apply sentiment labeling to a DataFrame using the provided model.

    This function:
    - Infers 'text' column if missing
    - Uses model's predict or predict_with_scores for sentiment and confidence
    - Adds 'sentiment' and optionally 'confidence' columns

    Args:
        dataframe: DataFrame containing at least a text-like column.
        model: Model with .predict or .predict_with_scores method.

    Returns:
        Labeled DataFrame with 'sentiment' and optionally 'confidence'.

    Raises:
        ValueError: If no text-like column found in input DataFrame.
    """
    result_df = dataframe.copy()

    if "text" not in result_df.columns:
        # Try to infer a text-like column
        text_columns = [
            col
            for col in result_df.columns
            if any(keyword in col.lower() for keyword in ("headline", "title", "text"))
        ]
        if not text_columns:
            raise ValueError("No text-like column found in input DataFrame.")

        result_df["text"] = (
            result_df[text_columns[0]].astype(str) if text_columns else ""
        )
    texts = result_df["text"].fillna("").astype(str).tolist()

    # Try confidences if available
    if hasattr(model, "predict_with_scores"):
        labels, confidence_scores = model.predict_with_scores(texts)
        result_df["sentiment"] = labels
        result_df["confidence"] = confidence_scores
    else:
        result_df["sentiment"] = model.predict(texts)

    return result_df


def resolve_data_directory(env_var: str = "NEWS_CSV_DIR") -> Path:
    """
    Resolve the data directory from an environment variable to absolute Path.

    Args:
        env_var: Environment variable name for the directory.

    Returns:
        Absolute Path object for the data directory.

    Example:
        >>> resolve_data_directory("NEWS_CSV_DIR")
        PosixPath('/path/to/data')
    """
    env_directory = os.getenv(env_var) or "data"
    directory_path = Path(env_directory)
    if not directory_path.is_absolute():
        directory_path = REPO_ROOT / directory_path
    return directory_path.resolve()


def show_debug_sidebar():
    """Show debug information in sidebar for troubleshooting."""
    st.header("🔧 Debug")
    st.caption("Use this to verify inputs and code path.")
    st.write("CWD:", os.getcwd())
    st.write("NEWS_CSV_DIR:", os.getenv("NEWS_CSV_DIR", "(unset)"))
    st.write("SECTOR_MAP_CSV:", os.getenv("SECTOR_MAP_CSV", "(unset)"))
    st.write("SENTIMENT_MODEL:", os.getenv("SENTIMENT_MODEL", "(unset)"))
    st.write("Python:", sys.version.split()[0])


# -------------------- UI --------------------


# -------------------- Sidebar debug --------------------
with st.sidebar:
    show_debug_sidebar()


st.title("📈 Market Sentiment Analyzer")

tab_ingest, tab_dashboard = st.tabs(["Ingest/Label", "Dashboard"])

# ===== Ingest / Label =====
with tab_ingest:
    st.subheader("Upload CSVs or use a folder")

    with st.form("ingest_form", clear_on_submit=False):
        uploaded_files = st.file_uploader(
            "Upload CSV (headline/text + optional ticker/date)",
            type=["csv"],
            accept_multiple_files=True,
            key="uploader",
        )
        folder_path = st.text_input(
            "Or CSV folder (resolved from repo root if relative)",
            value=resolve_data_directory("NEWS_CSV_DIR"),
            key="folder_input",
        )
        model_choice = st.selectbox(
            "Sentiment model",
            [
                "VADER (fast)",
                "RoBERTa (CardiffNLP)",
                "FinBERT (ProsusAI)",
                "FinBERT (Tone)",
            ],
            index=2,
            key="model_choice",
        )
        huggingface_model_mapping = {
            "RoBERTa (CardiffNLP)": "cardiffnlp/twitter-roberta-base-sentiment-latest",
            "FinBERT (ProsusAI)": "ProsusAI/finbert",
            "FinBERT (Tone)": "yiyanghkust/finbert-tone",
        }
        huggingface_id = huggingface_model_mapping.get(
            model_choice, os.getenv("SENTIMENT_MODEL", "ProsusAI/finbert")
        )

        submit_button = st.form_submit_button(
            "Ingest & Label", use_container_width=True
        )

    st.caption(
        f"🔎 Debug — uploaded files: "
        f"{0 if not uploaded_files else len(uploaded_files)} | "
        f"folder: {folder_path} | model: {model_choice}"
    )

    if submit_button:
        try:
            # 1) Load raw data (either uploads or folder)
            if uploaded_files:
                dataframes = []
                for uploaded_file in uploaded_files:
                    try:
                        dataframes.append(pd.read_csv(uploaded_file))
                    except Exception as error:
                        file_name = getattr(uploaded_file, "name", "<upload>")
                        st.warning(f"Error reading {file_name}: {error}")
                if not dataframes:
                    st.error(
                        "No valid CSV uploads detected. "
                        "Please verify your file format (CSV), "
                        "encoding (UTF-8 recommended), and column headers."
                    )
                    st.stop()
                combined_raw_data = pd.concat(dataframes, ignore_index=True)

                # Best-effort column mapping
                column_names = [col.lower() for col in combined_raw_data.columns]

                # Find the first column likely to contain text features
                text_column_index = next(
                    (
                        index
                        for index, column_name in enumerate(column_names)
                        if any(
                            keyword in column_name
                            for keyword in ("headline", "title", "text")
                        )
                    ),
                    None,
                )
                if text_column_index is None:
                    st.error("No text-like column found (need headline/title/text).")
                    st.stop()

                # Optional date/ticker columns
                date_column_index = next(
                    (
                        index
                        for index, column_name in enumerate(column_names)
                        if any(keyword in column_name for keyword in ("date", "time"))
                    ),
                    None,
                )
                ticker_column_index = next(
                    (
                        index
                        for index, column_name in enumerate(column_names)
                        if any(
                            keyword in column_name
                            for keyword in ("ticker", "symbol")
                        )
                    ),
                    None,
                )

                # Create normalized dataframe structure
                normalized_data = pd.DataFrame(
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
                normalized_data = load_csv_dir(folder_path)
                if normalized_data.empty:
                    st.error(
                        "No rows loaded from folder. Check path and CSV presence."
                    )
                    st.stop()

            # 2) Save normalized snapshot
            os.makedirs("data", exist_ok=True)
            normalized_data = normalize_and_save(normalized_data, "data/news.parquet")

            # 3) Model selection (lazy load HF)
            selected_model = _choose_sentiment_model(model_choice, huggingface_id)

            # 4) Apply sentiment labeling
            labeled_data = _label_dataframe(normalized_data, selected_model)

            # 5) Persist labeled dataset
            labeled_data.to_parquet("data/news_labeled.parquet", index=False)
            st.success(
                f"Ingested & labeled: {len(labeled_data)} rows → "
                "data/news_labeled.parquet"
            )
            st.dataframe(labeled_data.head(20))

        except Exception as error:
            st.exception(error)
            st.stop()


# ===== Dashboard =====
with tab_dashboard:
    st.subheader("Sentiment Trends")
    from app.plots import (
        sentiment_trend_by_date,
        sentiment_trend_by_ticker_date,
        sentiment_trend_by_sector_date,
    )

    try:
        if os.path.exists("data/news_labeled.parquet"):
            labeled_dataframe = pd.read_parquet("data/news_labeled.parquet")
            if "sector" not in labeled_dataframe.columns:
                labeled_dataframe["sector"] = None

            aggregation_mode = st.selectbox(
                "Aggregate by",
                ["Market (Date)", "Ticker + Date", "Sector + Date"],
                index=0,
                key="group_mode",
            )

            if aggregation_mode == "Market (Date)":
                st.pyplot(sentiment_trend_by_date(labeled_dataframe))

            elif aggregation_mode == "Ticker + Date":
                available_tickers = sorted(
                    [
                        ticker
                        for ticker in labeled_dataframe["ticker"].dropna().unique()
                        if ticker
                    ]
                )
                selected_ticker = st.selectbox(
                    "Ticker", available_tickers or ["(none)"], key="ticker_select"
                )
                if available_tickers:
                    st.pyplot(
                        sentiment_trend_by_ticker_date(
                            labeled_dataframe, selected_ticker
                        )
                    )
                else:
                    st.info("No tickers found in labeled data.")

            else:  # Sector + Date
                available_sectors = sorted(
                    [
                        sector
                        for sector in labeled_dataframe["sector"].dropna().unique()
                        if sector
                    ]
                )
                selected_sector = st.selectbox(
                    "Sector", available_sectors or ["(none)"], key="sector_select"
                )
                if available_sectors:
                    st.pyplot(
                        sentiment_trend_by_sector_date(
                            labeled_dataframe, selected_sector
                        )
                    )
                else:
                    st.info(
                        "No sectors found. "
                        "Provide a sector map CSV in .env (SECTOR_MAP_CSV)."
                    )

            # Export current labeled dataset
            export_buffer = io.StringIO()
            labeled_dataframe.to_csv(export_buffer, index=False)
            st.download_button(
                "⬇️ Export labeled CSV",
                export_buffer.getvalue(),
                "news_labeled.csv",
                key="export_btn",
            )
        else:
            st.info("No labeled data yet. Go to Ingest/Label first.")
    except Exception as error:
        st.exception(error)
