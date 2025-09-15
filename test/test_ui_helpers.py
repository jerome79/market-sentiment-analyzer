import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from market_sentiment_analyzer.ui_streamlit import _choose_model, _hash, _label_df, resolve_data_dir


class DummyModel:
    def __init__(self, labels=None, conf=None):
        # default: all zeros
        self._labels = labels
        self._conf = conf

    def predict_with_scores(self, texts):
        if self._labels is None:
            raise RuntimeError("no scores provided")
        return self._labels[: len(texts)], self._conf[: len(texts)]

    def predict(self, texts):
        if self._labels is not None:
            return self._labels[: len(texts)]
        # simple deterministic fallback
        return [0 for _ in texts]


def test_hash_stability_and_difference():
    a = _hash("abc")
    b = _hash("abc")
    c = _hash("abcd")
    assert a == b and a != c and len(a) == 32


def test_label_df_infers_text_and_uses_predict_with_scores():
    df = pd.DataFrame({"headline": ["up day", "down day", "up day"]})
    model = DummyModel(labels=[1, -1, 1], conf=[0.9, 0.8, 0.95])
    out = _label_df(df, model)
    assert "sentiment" in out.columns
    assert "confidence" in out.columns
    # should preserve row count and assign labels to duplicates via dedup/merge
    assert len(out) == len(df)
    assert set(out["sentiment"].unique()) <= {-1, 0, 1}


def test_label_df_raises_without_text_like_columns():
    df = pd.DataFrame({"foo": [1, 2], "bar": [3, 4]})
    with pytest.raises(ValueError):
        _label_df(df, DummyModel())


def test_resolve_data_dir_returns_absolute(tmp_path, monkeypatch):
    # relative path in env should resolve to absolute
    monkeypatch.setenv("NEWS_CSV_DIR", tmp_path)
    p = resolve_data_dir("NEWS_CSV_DIR")
    assert p.is_absolute()


def test_choose_model(monkeypatch):
    sentinel = object()
    # replace _get_hf to return sentinel
    import market_sentiment_analyzer.ui_streamlit as ui

    monkeypatch.setattr(ui, "_get_hf", lambda mid: sentinel)

    m1 = _choose_model("VADER (fast)", "anything")
    # The class name check avoids importing VADER here
    assert m1.__class__.__name__ == "BaselineVader"

    m2 = _choose_model("FinBERT (ProsusAI)", "ProsusAI/finbert")
    assert m2 is sentinel


import pandas as pd
from market_sentiment_analyzer.ui_streamlit import load_labeled_parquet, trend_market, trend_ticker, trend_sector


def test_load_labeled_parquet(tmp_path):
    # Create a parquet file
    df = pd.DataFrame({"date": ["2024-01-01"], "ticker": ["AAPL"], "sector": ["Tech"], "sentiment": [0.8]})
    file = tmp_path / "news_labeled.parquet"
    df.to_parquet(file, index=False)
    loaded = load_labeled_parquet(str(file))
    assert loaded is not None
    assert "ticker" in loaded.columns
    assert "sector" in loaded.columns


def test_trend_market_and_ticker():
    df = pd.DataFrame({"date": ["2024-01-01", "2024-01-01"], "ticker": ["AAPL", "GOOG"], "sentiment": [1, -1]})
    tm = trend_market(df)
    assert "avg_sentiment" in tm.columns
    tt = trend_ticker(df, "AAPL")
    assert "avg_sentiment" in tt.columns


def test_trend_sector():
    df = pd.DataFrame({"date": ["2024-01-01", "2024-01-01"], "sector": ["Tech", "Finance"], "sentiment": [1, -1]})
    ts = trend_sector(df, "Tech")
    assert "avg_sentiment" in ts.columns


def test_upload_missing_text_column(monkeypatch):
    import pandas as pd
    import streamlit as st
    from market_sentiment_analyzer.ui_streamlit import _label_df

    # Patch st.error and st.stop to raise exceptions
    monkeypatch.setattr(st, "error", lambda msg: (_ for _ in ()).throw(ValueError(msg)))
    monkeypatch.setattr(st, "stop", lambda: (_ for _ in ()).throw(SystemExit()))
    df = pd.DataFrame({"foo": [1, 2]})
    try:
        _label_df(df, DummyModel())
    except Exception as e:
        assert "No text-like column" in str(e) or isinstance(e, ValueError)


from market_sentiment_analyzer.ui_streamlit import show_debug_sidebar, _t


def test_show_debug_sidebar_runs(monkeypatch):
    # Patch st methods to no-ops
    import streamlit as st

    monkeypatch.setattr(st, "header", lambda *a, **k: None)
    monkeypatch.setattr(st, "caption", lambda *a, **k: None)
    monkeypatch.setattr(st, "write", lambda *a, **k: None)
    show_debug_sidebar()


def test_timing_context_manager():
    import time

    with _t() as timer:
        time.sleep(0.01)
    assert timer.dt >= 0
