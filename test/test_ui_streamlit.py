import sys
from pathlib import Path
import pandas as pd
import pytest

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
