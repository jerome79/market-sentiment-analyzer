import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from app.sentiment import BaselineVader


def test_vader_predict_returns_int_labels_and_handles_none():
    model = BaselineVader()
    texts = ["Great rally in markets", "Horrible loss reported", "", None]
    out = model.predict(texts)
    assert len(out) == len(texts)
    assert all(o in (-1, 0, 1) for o in out)


def test_vader_predict_is_stateless():
    model = BaselineVader()
    texts = ["Stocks up", "Stocks down"]
    out1 = model.predict(texts)
    out2 = model.predict(texts)
    assert out1 == out2
