import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from market_sentiment_analyzer.ui_streamlit import _hash, resolve_data_dir


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
