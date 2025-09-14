import sys
from pathlib import Path


def test_imports():
    ROOT = Path(__file__).resolve().parents[1]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    import market_sentiment_analyzer.ingest as i
    import market_sentiment_analyzer.sentiment as s

    assert hasattr(s, "BaselineVader")
    assert hasattr(i, "load_csv_dir")
