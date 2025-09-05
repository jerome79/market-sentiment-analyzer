import os
import io
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# -------- repo-root import fix --------
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.ingest import load_csv_dir, normalize_and_save  # resolves paths internally

# -------------------- App config --------------------
load_dotenv()
st.set_page_config(page_title="Market Sentiment Analyzer", layout="wide")
st.set_option("client.showErrorDetails", True)

# -------------------- Helpers --------------------
@st.cache_resource
def _get_hf(model_id: str):
    from app.sentiment import HFClassifier
    # cache only the HF model (VADER is tiny)
    return HFClassifier(model_id)

def _choose_model(model_choice: str, hf_id: str):
    from app.sentiment import BaselineVader
    return (_get_hf(hf_id) if model_choice != "VADER (fast)" else BaselineVader())

def _label_df(df: pd.DataFrame, model) -> pd.DataFrame:
    out = df.copy()
    if "text" not in out.columns:
        # try to infer a text-like column
        cand = [c for c in out.columns if any(k in c.lower() for k in ("headline", "title", "text"))]
        out["text"] = out[cand[0]].astype(str) if cand else ""
    texts = out["text"].fillna("").astype(str).tolist()
    # try confidences if available
    if hasattr(model, "predict_with_scores"):
        labels, conf = model.predict_with_scores(texts)
        out["sentiment"] = labels
        out["confidence"] = conf
    else:
        out["sentiment"] = model.predict(texts)
    return out

def resolve_data_dir(env_var: str = "NEWS_CSV_DIR") -> Path:
    """
    Resolve CHROMA_PERSIST_DIR to an absolute path.
    """
    env_dir = os.getenv(env_var) or "data"
    p = Path(env_dir)
    if not p.is_absolute():
        p = (ROOT / p)
    return p.resolve()

# -------------------- UI --------------------

# -------------------- Sidebar debug --------------------
with st.sidebar:
    st.header("🔧 Debug")
    st.caption("Use this to verify inputs and code path.")
    st.write("CWD:", os.getcwd())
    st.write("NEWS_CSV_DIR:", os.getenv("NEWS_CSV_DIR", "(unset)"))
    st.write("SECTOR_MAP_CSV:", os.getenv("SECTOR_MAP_CSV", "(unset)"))
    st.write("SENTIMENT_MODEL:", os.getenv("SENTIMENT_MODEL", "(unset)"))
    st.write("Python:", sys.version.split()[0])

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
            ["VADER (fast)", "RoBERTa (CardiffNLP)", "FinBERT (ProsusAI)", "FinBERT (Tone)"],
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
                        st.warning(f"Skipping {getattr(f, 'name', '<upload>')}: {e}")
                if not frames:
                    st.error("No valid CSV uploads were read.")
                    st.stop()
                raw_all = pd.concat(frames, ignore_index=True)

                # best-effort column mapping
                cols = [c.lower() for c in raw_all.columns]
                # text column
                text_idx = next((i for i, c in enumerate(cols) if ("headline" in c or "title" in c or "text" in c)),
                                None)
                if text_idx is None:
                    st.error("No text-like column found (need headline/title/text).")
                    st.stop()
                # optional date/ticker
                date_idx = next((i for i, c in enumerate(cols) if ("date" in c or "time" in c)), None)
                tick_idx = next((i for i, c in enumerate(cols) if ("ticker" in c or "symbol" in c)), None)

                raw = pd.DataFrame({
                    "date": pd.to_datetime(raw_all.iloc[:, date_idx],
                                           errors="coerce").dt.date if date_idx is not None else None,
                    "ticker": raw_all.iloc[:, tick_idx] if tick_idx is not None else None,
                    "source": "upload",
                    "headline": raw_all.iloc[:, text_idx],
                    "text": raw_all.iloc[:, text_idx]
                })
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
    from app.plots import sentiment_trend, sentiment_trend_by_date, sentiment_trend_by_ticker_date, sentiment_trend_by_sector_date
    try:
        if os.path.exists("data/news_labeled.parquet"):
            df = pd.read_parquet("data/news_labeled.parquet")
            if "sector" not in df.columns:
                df["sector"] = None

            group_mode = st.selectbox(
                "Aggregate by",
                ["Market (Date)", "Ticker + Date", "Sector + Date"],
                index=0,
                key="group_mode",
            )

            if group_mode == "Market (Date)":
                st.pyplot(sentiment_trend_by_date(df))

            elif group_mode == "Ticker + Date":
                tickers = sorted([t for t in df["ticker"].dropna().unique().tolist() if t])
                sel_t = st.selectbox("Ticker", tickers or ["(none)"], key="ticker_select")
                if tickers:
                    st.pyplot(sentiment_trend_by_ticker_date(df, sel_t))
                else:
                    st.info("No tickers found in labeled data.")

            else:  # Sector + Date
                sectors = sorted([s for s in df["sector"].dropna().unique().tolist() if s])
                sel_s = st.selectbox("Sector", sectors or ["(none)"], key="sector_select")
                if sectors:
                    st.pyplot(sentiment_trend_by_sector_date(df, sel_s))
                else:
                    st.info("No sectors found. Provide a sector map CSV in .env (SECTOR_MAP_CSV).")

            # Export current labeled dataset
            buf = io.StringIO()
            df.to_csv(buf, index=False)
            st.download_button("⬇️ Export labeled CSV", buf.getvalue(), "news_labeled.csv", key="export_btn")
        else:
            st.info("No labeled data yet. Go to Ingest/Label first.")
    except Exception as e:
        st.exception(e)
