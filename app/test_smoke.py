"""
Smoke tests for basic import functionality.

These tests ensure that core modules can be imported successfully and contain
the expected classes and functions. This serves as a basic sanity check for
the package structure.
"""


def test_imports():
    """Test that core modules import correctly and contain expected components."""
    # Test sentiment module imports
    import app.sentiment as sentiment_module

    assert hasattr(sentiment_module, "BaselineVader")
    assert hasattr(sentiment_module, "HFClassifier")

    # Test ingest module imports
    import app.ingest as ingest_module

    assert hasattr(ingest_module, "load_csv_dir")
    assert hasattr(ingest_module, "normalize_and_save")

    # Test plots module imports
    import app.plots as plots_module

    assert hasattr(plots_module, "sentiment_trend_by_date")
    assert hasattr(plots_module, "sentiment_trend_by_ticker_date")
    assert hasattr(plots_module, "sentiment_trend_by_sector_date")
