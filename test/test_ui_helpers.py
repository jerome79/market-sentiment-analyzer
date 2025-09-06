import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.ui_streamlit import _choose_model, _hash, _label_df, resolve_data_dir


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
    import app.ui_streamlit as ui

    monkeypatch.setattr(ui, "_get_hf", lambda mid: sentinel)

    m1 = _choose_model("VADER (fast)", "anything")
    # The class name check avoids importing VADER here
    assert m1.__class__.__name__ == "BaselineVader"

    m2 = _choose_model("FinBERT (ProsusAI)", "ProsusAI/finbert")
    assert m2 is sentinel
