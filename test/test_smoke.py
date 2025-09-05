"""Smoke tests to verify basic imports and module functionality."""
import sys
from pathlib import Path

# Add parent directory to path for imports
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))


def test_sentiment_imports():
    """Test that sentiment module imports correctly."""
    import app.sentiment as sentiment_module

    # Test class exists
    assert hasattr(sentiment_module, "BaselineVader")
    assert hasattr(sentiment_module, "HuggingFaceClassifier")
    
    # Test backward compatibility
    assert hasattr(sentiment_module, "HFClassifier")


def test_ingest_imports():
    """Test that ingest module imports correctly."""
    import app.ingest as ingest_module

    # Test function exists
    assert hasattr(ingest_module, "load_csv_directory")
    assert hasattr(ingest_module, "normalize_and_save")
    
    # Test backward compatibility
    assert hasattr(ingest_module, "load_csv_dir")


def test_plots_imports():
    """Test that plots module imports correctly."""
    import app.plots as plots_module

    # Test functions exist
    assert hasattr(plots_module, "sentiment_trend_by_date")
    assert hasattr(plots_module, "sentiment_trend_by_ticker_date")
    assert hasattr(plots_module, "sentiment_trend_by_sector_date")


def test_basic_functionality():
    """Test basic functionality of core components."""
    from app.sentiment import BaselineVader
    
    # Test VADER can be instantiated and make predictions
    vader = BaselineVader()
    test_texts = ["Good news!", "Bad news!", "Neutral text"]
    predictions = vader.predict(test_texts)
    
    assert len(predictions) == 3
    assert all(pred in [-1, 0, 1] for pred in predictions)
