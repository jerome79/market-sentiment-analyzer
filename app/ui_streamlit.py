import hashlib
import io
import os
import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from app.ingest import load_csv_dir, normalize_and_save  # resolves paths internally

# -------- repo-root import fix --------
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# -------------------- App config --------------------
load_dotenv()
st.set_page_config(page_title="Market Sentiment Analyzer", layout="wide")
st.set_option("client.showErrorDetails", True)


# -------------------- Helpers --------------------
@st.cache_data(show_spinner=False)
def load_labeled_parquet(path="data/news_labeled.parquet"):
    if not os.path.exists(path):
        return None
    df = pd.read_parquet(path)
    if "ticker" in df:
        df["ticker"] = df["ticker"].astype("category")
    if "sector" in df:
        df["sector"] = df["sector"].astype("category")
    return df


@st.cache_data(show_spinner=False)
def trend_market(df: pd.DataFrame):
    return (
        df.groupby("date", as_index=False)["sentiment"]
        .mean()
        .rename(columns={"sentiment": "avg_sentiment"})
    )


@st.cache_data(show_spinner=False)
def trend_ticker(df: pd.DataFrame, ticker: str):
    sdf = df[df["ticker"] == ticker]
    return (
        sdf.groupby("date", as_index=False)["sentiment"]
        .mean()
        .rename(columns={"sentiment": "avg_sentiment"})
    )


@st.cache_data(show_spinner=False)
def trend_sector(df: pd.DataFrame, sector: str):
    sdf = df[df.get("sector").eq(sector)]
    return (
        sdf.groupby("date", as_index=False)["sentiment"]
        .mean()
        .rename(columns={"sentiment": "avg_sentiment"})
    )


@st.cache_resource
def _get_hf(model_id: str):
    """
    Lazily load and cache a HuggingFace sentiment model by model ID.

    Parameters:
        model_id (str): HuggingFace model identifier.

    Returns:
        HFClassifier: Instantiated classifier.
    """
    from app.sentiment import HFClassifier

    # cache only the HF model (VADER is tiny)
    return HFClassifier(model_id)


def _choose_model(model_choice: str, hf_id: str):
    """
    Select the sentiment model based on user choice.

    Parameters:
        model_choice (str): Model name selected by user.
        hf_id (str): HuggingFace model ID for non-VADER models.

    Returns:
        Model instance for sentiment prediction.
    """
    from app.sentiment import BaselineVader

    return _get_hf(hf_id) if model_choice != "VADER (fast)" else BaselineVader()


def _hash(s: str) -> str:
    """
    Generate an MD5 hash for the given string.

    Parameters:
        s (str): Input string to hash.

    Returns:
        str: MD5 hexadecimal digest of the input string.
    """
    return hashlib.md5(s.encode("utf-8", "ignore")).hexdigest()


def _label_df(df: pd.DataFrame, model) -> pd.DataFrame:
    """
    Apply sentiment labeling to a DataFrame using the provided model.

    - Infers 'text' column if missing.
    - Uses model's predict or predict_with_scores for sentiment and confidence.
    - Adds 'sentiment' and optionally 'confidence' columns.

    Parameters:
        df (pd.DataFrame): DataFrame containing at least a text-like column.
        model: Model with .predict or .predict_with_scores method.

    Returns:
        pd.DataFrame: Labeled DataFrame with 'sentiment' and optionally 'confidence'.
    """
    out = df.copy()
    if "text" not in out.columns:
        # try to infer a text-like column
        cand = [
            c for c in out.columns if any(k in c.lower() for k in ("headline", "title", "text"))
        ]
        if not cand:
            raise ValueError("No text-like column found in input DataFrame.")
        out["text"] = out[cand[0]].astype(str) if cand else ""
    out["text"] = out["text"].fillna("").astype(str)
    out["__h"] = out["text"].map(_hash)

    uniq = out.drop_duplicates("__h", keep="first")[["__h", "text"]].copy()
    tx = uniq["text"].tolist()
    # try confidences if available
    try:
        labels, conf = model.predict_with_scores(tx)
        uniq["sentiment"], uniq["confidence"] = labels, conf
    except Exception:
        uniq["sentiment"] = model.predict(tx)
    out = out.merge(uniq.drop(columns=["text"]), on="__h", how="left").drop(columns="__h")
    return out


def resolve_data_dir(env_var: str = "NEWS_CSV_DIR") -> Path:
    """
    Resolve the data directory from an environment variable to an absolute Path.

    Parameters:
        env_var (str): Environment variable name for the directory (default: "NEWS_CSV_DIR").

    Returns:
        Path: Absolute path to the directory.

    Example:
        resolve_data_dir("NEWS_CSV_DIR")
    """
    env_dir = os.getenv(env_var) or "data"
    p = Path(env_dir)
    if not p.is_absolute():
        p = ROOT / p
    return p.resolve()


def show_debug_sidebar():
    """Show debug info in sidebar."""
    st.header("🔧 Debug")
    st.caption("Use this to verify inputs and code path.")
    st.write("CWD:", os.getcwd())
    st.write("NEWS_CSV_DIR:", os.getenv("NEWS_CSV_DIR", "(unset)"))
    st.write("SECTOR_MAP_CSV:", os.getenv("SECTOR_MAP_CSV", "(unset)"))
    st.write("SENTIMENT_MODEL:", os.getenv("SENTIMENT_MODEL", "(unset)"))
    st.write("Python:", sys.version.split()[0])


def _t():
    """
    Context manager for timing code execution in milliseconds.

    Usage:
        with _t() as timer:
            # code block
        print(timer.dt)  # elapsed time in ms
    """

    class T:
        def __enter__(self):
            self.s = time.perf_counter()
            return self

        def __exit__(self, *_):
            self.dt = (time.perf_counter() - self.s) * 1000  # ms

    return T()


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
        up = st.file_uploader(
            "Upload CSV (headline/text + optional ticker/date)",
            type=["csv"],
            accept_multiple_files=True,
            key="uploader",
        )
        folder = st.text_input(
            "Or CSV folder (resolved from repo root if relative)",
            value=resolve_data_dir("NEWS_CSV_DIR"),
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
        hf_id = {
            "RoBERTa (CardiffNLP)": "cardiffnlp/twitter-roberta-base-sentiment-latest",
            "FinBERT (ProsusAI)": "ProsusAI/finbert",
            "FinBERT (Tone)": "yiyanghkust/finbert-tone",
        }.get(model_choice, os.getenv("SENTIMENT_MODEL", "ProsusAI/finbert"))

        submit = st.form_submit_button("Ingest & Label", use_container_width=True)

    st.caption(
        f"🔎 Debug — uploaded files: {0 if not up else len(up)} | folder: {folder} | model: {model_choice}"
    )

    if submit:
        try:
            # 1) Load raw (either uploads or folder)
            if up:
                frames = []
                for f in up:
                    try:
                        frames.append(pd.read_csv(f))
                    except Exception as e:
                        st.warning(f"Error reading {getattr(f, 'name', '<upload>')}: {e}")
                if not frames:
                    st.error(
                        "No valid CSV uploads detected. Please verify your file format (CSV), encoding (UTF-8 recommended), and column headers."
                    )
                    st.stop()
                raw_all = pd.concat(frames, ignore_index=True)

                # best-effort column mapping
                cols = [c.lower() for c in raw_all.columns]

                # Find the first column likely to contain text features (headline/title/text)
                text_col_idx = next(
                    (
                        i
                        for i, c in enumerate(cols)
                        if ("headline" in c or "title" in c or "text" in c)
                    ),
                    None,
                )
                if text_col_idx is None:
                    st.error("No text-like column found (need headline/title/text).")
                    st.stop()
                # optional date/ticker
                date_idx = next(
                    (i for i, c in enumerate(cols) if ("date" in c or "time" in c)),
                    None,
                )
                tick_idx = next(
                    (i for i, c in enumerate(cols) if ("ticker" in c or "symbol" in c)),
                    None,
                )

                raw = pd.DataFrame(
                    {
                        "date": (
                            pd.to_datetime(raw_all.iloc[:, date_idx], errors="coerce").dt.date
                            if date_idx is not None
                            else None
                        ),
                        "ticker": (raw_all.iloc[:, tick_idx] if tick_idx is not None else None),
                        "source": "upload",
                        "headline": raw_all.iloc[:, text_col_idx],
                        "text": raw_all.iloc[:, text_col_idx],
                    }
                )
            else:
                raw = load_csv_dir(folder)
                if raw.empty:
                    st.error("No rows loaded from folder. Check path and CSV presence.")
                    st.stop()

            # 2) Save normalized snapshot
            os.makedirs("data", exist_ok=True)
            raw = normalize_and_save(raw, "data/news.parquet")

            # 3) Model selection (lazy load HF)
            model = _choose_model(model_choice, hf_id)

            # 4) Label
            labeled = _label_df(raw, model)

            # 5) Persist labeled dataset
            labeled.to_parquet("data/news_labeled.parquet", index=False)
            st.success(f"Ingested & labeled: {len(labeled)} rows → data/news_labeled.parquet")
            st.dataframe(labeled.head(20))

        except Exception as e:
            st.exception(e)
            st.stop()

# ===== Dashboard =====
with tab_dashboard:
    st.subheader("Sentiment Trends")

    try:
        if os.path.exists("data/news_labeled.parquet"):
            df = load_labeled_parquet()
            if df is None or df.empty:
                st.info("No labeled data yet. Go to Ingest/Label first.")
            else:
                # --- Date range filter ---
                min_d, max_d = df["date"].min(), df["date"].max()
                dsel = st.date_input("Date range", (min_d, max_d), key="date_range")
                if isinstance(dsel, tuple) and len(dsel) == 2:
                    df = df[(df["date"] >= dsel[0]) & (df["date"] <= dsel[1])]

            group_mode = st.radio(
                "Group by:",
                ["Market (Date)", "Ticker + Date", "Sector + Date"],
                key="group_mode",
            )

            if group_mode == "Market (Date)":
                tr = trend_market(df)
                st.line_chart(tr.set_index("date")["avg_sentiment"])
            elif group_mode == "Ticker + Date":
                tickers = sorted([t for t in df["ticker"].dropna().unique().tolist() if t])
                sel_t = st.selectbox("Select ticker", tickers, key="sel_t")
                if sel_t:
                    tr = trend_ticker(df, sel_t)
                    st.line_chart(tr.set_index("date")["avg_sentiment"])
                else:
                    st.warning("No ticker available in dataset.")

            else:  # Sector + Date
                if "sector" not in df:
                    st.warning(
                        "No sector column available. Please ensure sector_map.csv was applied during ingest."
                    )
                else:
                    sectors = sorted([s for s in df["sector"].dropna().unique().tolist() if s])
                    sel_s = st.selectbox("Select sector", sectors, key="sel_s")
                    if sel_s:
                        tr = trend_sector(df, sel_s)
                        st.line_chart(tr.set_index("date")["avg_sentiment"])
                    else:
                        st.warning("No sector available in dataset.")

            # Export current labeled dataset
            cap = st.slider("Max rows to display", 100, 5000, 1000, 100, key="row_cap")
            st.dataframe(df.head(cap))
            buf = io.StringIO()
            df.to_csv(buf, index=False)
            st.download_button(
                "⬇️ Export labeled CSV",
                buf.getvalue(),
                "news_labeled.csv",
                key="export_btn",
            )
        else:
            st.info("No labeled data yet. Go to Ingest/Label first.")
    except Exception as e:
        st.exception(e)
