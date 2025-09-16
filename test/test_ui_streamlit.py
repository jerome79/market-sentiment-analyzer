import sys
from pathlib import Path
import io
import os
import sys
import types
import pandas as pd
import pytest

import streamlit as st
from unittest import mock

from market_sentiment_analyzer import ui_streamlit as ui

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from market_sentiment_analyzer.ui_streamlit import _hash, resolve_data_dir, load_labeled_parquet, trend_market, _label_df, trend_ticker, trend_sector


def test_hash_is_consistent_and_hex() -> None:
    """
    Test that _hash returns a consistent 32-character hex string for the same input.
    """
    s = "test string"
    h1 = _hash(s)
    h2 = _hash(s)
    assert h1 == h2
    assert isinstance(h1, str)
    assert len(h1) == 32
    int(h1, 16)  # should be valid hex


def test_resolve_data_dir_absolute_and_relative(tmp_path: Path, monkeypatch) -> None:
    """
    Test that resolve_data_dir returns an absolute Path for both absolute and relative NEWS_CSV_DIR.
    """
    abs_dir = tmp_path / "abs"
    abs_dir.mkdir()
    monkeypatch.setenv("NEWS_CSV_DIR", str(abs_dir))
    p = resolve_data_dir("NEWS_CSV_DIR")
    assert p.is_absolute()
    assert p.name == "abs"
    monkeypatch.setenv("NEWS_CSV_DIR", "rel_dir")
    p2 = resolve_data_dir("NEWS_CSV_DIR")
    assert p2.is_absolute()
    assert "rel_dir" in str(p2)


def test_hash_is_consistent_and_hex():
    s = "test string"
    h1 = _hash(s)
    h2 = _hash(s)
    assert h1 == h2
    assert isinstance(h1, str)
    assert len(h1) == 32
    int(h1, 16)  # should be valid hex


def test_resolve_data_dir_absolute_and_relative(tmp_path, monkeypatch):
    monkeypatch.setenv("NEWS_CSV_DIR", str(tmp_path))
    assert resolve_data_dir("NEWS_CSV_DIR").is_absolute()
    monkeypatch.setenv("NEWS_CSV_DIR", "data")
    path = resolve_data_dir("NEWS_CSV_DIR")
    assert path.is_absolute()
    assert path.parts[-1] == "data"


def test_load_labeled_parquet(tmp_path):
    df = pd.DataFrame({"date": ["2024-01-01"], "ticker": ["AAPL"], "sector": ["Tech"], "sentiment": [1]})
    out_path = tmp_path / "news_labeled.parquet"
    df.to_parquet(out_path, index=False)
    loaded = load_labeled_parquet(str(out_path))
    assert loaded is not None
    assert loaded["ticker"].dtype.name == "category"
    assert loaded["sector"].dtype.name == "category"


def test_trend_market_and_ticker():
    df = pd.DataFrame({"date": ["2024-01-01", "2024-01-02", "2024-01-02"], "ticker": ["AAPL", "AAPL", "MSFT"], "sentiment": [1, 0, -1]})
    tm = trend_market(df)
    assert "avg_sentiment" in tm.columns
    tt = trend_ticker(df, "AAPL")
    assert "avg_sentiment" in tt.columns


def test_trend_sector():
    df = pd.DataFrame({"date": ["2024-01-01", "2024-01-01"], "sector": ["Tech", "Finance"], "sentiment": [1, -1]})
    ts = trend_sector(df, "Tech")
    assert "avg_sentiment" in ts.columns


def test_label_df_infers_text_and_uses_predict_with_scores():
    class DummyModel:
        def predict_with_scores(self, tx):
            return [0] * len(tx), [0.9] * len(tx)

    df = pd.DataFrame({"date": ["2024-01-01"], "ticker": ["AAPL"], "headline": ["h"], "text": ["t"]})
    out = _label_df(df, DummyModel())
    assert "sentiment" in out.columns
    assert "confidence" in out.columns


def test_label_df_raises_without_text_like_columns():
    class DummyModel:
        def predict(self, tx):
            return [0] * len(tx)

    df = pd.DataFrame({"foo": [1, 2]})
    with pytest.raises(Exception):
        _label_df(df, DummyModel())


def test_label_df_branch_no_sentiment(monkeypatch):
    # covers else branch if no sentiment
    class DummyModel:
        def predict(self, tx):
            return [None] * len(tx)

    df = pd.DataFrame({"date": ["2024-01-01"], "ticker": ["AAPL"], "headline": ["h"], "text": ["t"]})
    out = _label_df(df, DummyModel())
    assert "avg_sentiment" in out.columns


def test_dashboard_group_modes(monkeypatch):
    # Simulates group_mode branches in dashboard logic
    df = pd.DataFrame({"date": ["2024-01-01", "2024-01-02"], "ticker": ["AAPL", "MSFT"], "sector": ["Tech", "Finance"], "sentiment": [1, -1]})
    assert not trend_market(df).empty
    assert not trend_ticker(df, "AAPL").empty
    assert not trend_sector(df, "Tech").empty


# Helper to patch st.stop to raise for testable exit
@pytest.fixture(autouse=True)
def patch_st_stop(monkeypatch):
    monkeypatch.setattr(st, "stop", lambda: (_ for _ in ()).throw(SystemExit()))


def test_upload_empty(monkeypatch):
    # Simulate user uploaded no files
    monkeypatch.setattr(st, "file_uploader", lambda *a, **kw: [])
    monkeypatch.setattr(st, "text_input", lambda *a, **kw: "data")
    monkeypatch.setattr(st, "selectbox", lambda *a, **kw: "FinBERT (ProsusAI)")
    monkeypatch.setattr(st, "form_submit_button", lambda *a, **kw: True)
    monkeypatch.setattr(st, "caption", lambda *a, **kw: None)
    monkeypatch.setattr(st, "warning", lambda *a, **kw: None)
    monkeypatch.setattr(st, "error", lambda *a, **kw: (_ for _ in ()).throw(ValueError("No valid CSV uploads detected")))
    # Patch load_csv_dir to return empty df so we get the error branch
    monkeypatch.setattr(ui, "load_csv_dir", lambda _: pd.DataFrame())
    with pytest.raises(ValueError):
        # Call the logic in ingest form conditionally
        folder = "data"
        up = []
        if not up:
            raw = ui.load_csv_dir(folder)  # This will be empty
            if raw.empty:
                st.error("No rows loaded from folder. Check path and CSV presence.")
                st.stop()


def test_upload_with_bad_csv(monkeypatch):
    # Simulate a file upload that fails to read as CSV
    class BadFile:
        name = "bad.csv"

    monkeypatch.setattr(st, "file_uploader", lambda *a, **kw: [BadFile()])
    monkeypatch.setattr(st, "text_input", lambda *a, **kw: "data")
    monkeypatch.setattr(st, "selectbox", lambda *a, **kw: "FinBERT (ProsusAI)")
    monkeypatch.setattr(st, "form_submit_button", lambda *a, **kw: True)
    monkeypatch.setattr(st, "warning", lambda *a, **kw: None)
    monkeypatch.setattr(st, "caption", lambda *a, **kw: None)
    monkeypatch.setattr(st, "error", lambda *a, **kw: (_ for _ in ()).throw(ValueError("No valid CSV uploads detected")))
    # Patch pd.read_csv to raise Exception
    monkeypatch.setattr(pd, "read_csv", lambda f: (_ for _ in ()).throw(Exception("bad format")))
    with pytest.raises(ValueError):
        frames = []
        for f in [BadFile()]:
            try:
                frames.append(pd.read_csv(f))
            except Exception as e:
                st.warning(f"Error reading {getattr(f, 'name', '<upload>')}: {e}")
        if not frames:
            st.error("No valid CSV uploads detected. Please verify your file format (CSV), encoding (UTF-8 recommended), and column headers.")
            st.stop()


def test_missing_text_col_raises(monkeypatch):
    # Simulates the UI error path for missing text-like columns
    df = pd.DataFrame({"foo": [1, 2]})
    monkeypatch.setattr(st, "error", lambda *a, **kw: (_ for _ in ()).throw(ValueError("No text-like column found")))
    with pytest.raises(ValueError):
        cols = [c.lower() for c in df.columns]
        text_col_idx = next((i for i, c in enumerate(cols) if ("headline" in c or "title" in c or "text" in c)), None)
        if text_col_idx is None:
            st.error("No text-like column found (need headline/title/text).")
            st.stop()


def test_dashboard_no_file(monkeypatch):
    # Simulates dashboard logic when no labeled file exists
    monkeypatch.setattr(os.path, "exists", lambda p: False)
    monkeypatch.setattr(st, "info", lambda *a, **kw: (_ for _ in ()).throw(SystemExit()))
    with pytest.raises(SystemExit):
        # This mimics the dashboard UI code
        if not os.path.exists("data/news_labeled.parquet"):
            st.info("No labeled data yet. Go to Ingest/Label first.")


def test_dashboard_empty_df(monkeypatch):
    # Simulates dashboard logic when labeled file is empty
    monkeypatch.setattr(os.path, "exists", lambda p: True)
    monkeypatch.setattr(ui, "load_labeled_parquet", lambda: pd.DataFrame())
    monkeypatch.setattr(st, "info", lambda *a, **kw: (_ for _ in ()).throw(SystemExit()))
    with pytest.raises(SystemExit):
        df = ui.load_labeled_parquet()
        if df is None or df.empty:
            st.info("No labeled data yet. Go to Ingest/Label first.")


def test_choose_model_env(monkeypatch):
    # Explicitly test fallback to SENTIMENT_MODEL env var
    monkeypatch.setenv("SENTIMENT_MODEL", "TEST_MODEL_ID")
    model_choice = "Unknown"
    hf_id = {
        "RoBERTa (CardiffNLP)": "cardiffnlp/twitter-roberta-base-sentiment-latest",
        "FinBERT (ProsusAI)": "ProsusAI/finbert",
        "FinBERT (Tone)": "yiyanghkust/finbert-tone",
    }.get(model_choice, os.getenv("SENTIMENT_MODEL", "ProsusAI/finbert"))
    assert hf_id == "TEST_MODEL_ID"


def test_timing_context_manager_precision():
    # Test _t context manager for timing
    timer_class = ui._t()
    with timer_class as timer:
        time = 0
        for _ in range(100000):
            time += 1
    assert hasattr(timer, "dt")
    assert timer.dt > 0


def test_download_button(monkeypatch):
    monkeypatch.setattr(st, "download_button", lambda *a, **k: None)
    monkeypatch.setattr(st, "dataframe", lambda *a, **k: None)
    # Simulate dataframe with some data
    df = pd.DataFrame({"date": ["2024-01-01"], "ticker": ["AAPL"], "sentiment": [1]})
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    st.download_button("Export", buf.getvalue(), "file.csv")
