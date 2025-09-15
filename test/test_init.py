"""Test the __init__.py module functionality."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import market_sentiment_analyzer


def test_version_available():
    """Test that __version__ is available."""
    assert hasattr(market_sentiment_analyzer, '__version__')
    assert isinstance(market_sentiment_analyzer.__version__, str)


def test_eager_imports_available():
    """Test that eager imports are available."""
    assert hasattr(market_sentiment_analyzer, 'SCHEMA')
    assert hasattr(market_sentiment_analyzer, 'load_csv_dir')
    assert hasattr(market_sentiment_analyzer, 'normalize_and_save')


def test_all_contains_expected():
    """Test that __all__ contains expected public API."""
    expected = [
        "SCHEMA",
        "__version__",
        "load_csv_dir", 
        "load_panel",
        "normalize_and_save",
        "panel_stats",
    ]
    for item in expected:
        assert item in market_sentiment_analyzer.__all__


def test_lazy_attribute_load_panel():
    """Test that lazy loading works for load_panel."""
    # This should trigger the __getattr__ mechanism
    load_panel = market_sentiment_analyzer.load_panel
    assert callable(load_panel)
    
    # After first access, it should be cached in globals
    assert hasattr(market_sentiment_analyzer, 'load_panel')


def test_lazy_attribute_panel_stats():
    """Test that lazy loading works for panel_stats."""
    # This should trigger the __getattr__ mechanism
    panel_stats = market_sentiment_analyzer.panel_stats
    assert callable(panel_stats)
    
    # After first access, it should be cached in globals
    assert hasattr(market_sentiment_analyzer, 'panel_stats')


def test_getattr_unknown_attribute():
    """Test that unknown attributes raise AttributeError."""
    try:
        _ = market_sentiment_analyzer.nonexistent_attribute
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert "has no attribute 'nonexistent_attribute'" in str(e)


def test_dir_functionality():
    """Test that __dir__ returns expected attributes."""
    dir_result = dir(market_sentiment_analyzer)
    
    # Should include eager imports
    assert 'SCHEMA' in dir_result
    assert 'load_csv_dir' in dir_result
    assert 'normalize_and_save' in dir_result
    
    # Should include lazy attributes
    assert 'load_panel' in dir_result
    assert 'panel_stats' in dir_result
    
    # Should be sorted
    assert dir_result == sorted(dir_result)


def test_lazy_attrs_dict():
    """Test that _LAZY_ATTRS is properly configured."""
    expected_lazy = {
        "load_panel": ".public_api",
        "panel_stats": ".public_api",
    }
    
    for key, value in expected_lazy.items():
        assert key in market_sentiment_analyzer._LAZY_ATTRS
        assert market_sentiment_analyzer._LAZY_ATTRS[key] == value


def test_caching_behavior():
    """Test that lazy attributes are cached after first access."""
    # Clear cache if it exists
    if hasattr(market_sentiment_analyzer, 'load_panel'):
        delattr(market_sentiment_analyzer, 'load_panel')
    
    # First access should trigger lazy loading
    load_panel1 = market_sentiment_analyzer.load_panel
    
    # Second access should use cached version
    load_panel2 = market_sentiment_analyzer.load_panel
    
    # Should be the same object (cached)
    assert load_panel1 is load_panel2