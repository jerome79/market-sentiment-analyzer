"""
Market Sentiment Analyzer
=========================

Public interface (stable for now):
    - SCHEMA
    - load_csv_dir
    - normalize_and_save
    - load_panel        (lazy)
    - panel_stats       (lazy)

Lazy loading is used so that importing the core package stays fast and only
brings in the lightweight ingestion pieces immediately. Heavier / future
submodules (analytics, models, plotting, etc.) can be added to the lazy loader
section without impacting import time.

Design notes:
    1. Core constants + ingestion helpers are imported eagerly (cheap & commonly used).
    2. Analytical helpers (currently load_panel, panel_stats) are resolved on first access
       via module-level __getattr__ (Python 3.7+).
    3. After first resolution the attribute is cached in globals() to avoid repeated
       import overhead.
    4. Type checkers get the real symbols through a TYPE_CHECKING block.
    5. A clear error is raised if an unknown attribute is requested.
"""

import importlib
from importlib import metadata as _metadata
from typing import TYPE_CHECKING

# --- Version metadata --------------------------------------------------------
try:
    __version__ = _metadata.version("market-sentiment-analyzer")
except _metadata.PackageNotFoundError:  # Fallback for editable/local use
    __version__ = "0.0.0+local"

# --- Eager (lightweight) imports ---------------------------------------------
from .ingest import SCHEMA, load_csv_dir, normalize_and_save

# --- Public API declaration --------------------------------------------------
# Include lazy names even though they are not yet bound.
__all__ = [
    "SCHEMA",
    "__version__",
    "load_csv_dir",
    "load_panel",
    "normalize_and_save",
    "panel_stats",
]

# --- Optional: static typing visibility --------------------------------------
if TYPE_CHECKING:  # These will let IDEs / mypy see the symbols.
    from .public_api import load_panel, panel_stats


# --- Lazy attribute loader ---------------------------------------------------
_LAZY_ATTRS = {
    "load_panel": ".public_api",
    "panel_stats": ".public_api",
    # Future additions example:
    # "analyze_sentiment": ".sentiment",
    # "plot_sector_trends": ".plots",
}


def __getattr__(name: str):
    """
    Lazily resolve selected attributes.

    On first access:
        - Import the target module.
        - Retrieve the attribute.
        - Cache it in globals() for subsequent direct hits.

    Raises:
        AttributeError: If the name is not part of the public lazy map.
    """
    if name in _LAZY_ATTRS:
        module_path = _LAZY_ATTRS[name]
        module = importlib.import_module(module_path, __name__)
        attr = getattr(module, name)
        # Cache to avoid repeated import / getattr cost
        globals()[name] = attr
        return attr
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


# --- (Optional) explicit dir() friendliness ----------------------------------
def __dir__():
    # Combine normal globals with lazy keys
    return sorted(set(list(globals().keys()) + list(_LAZY_ATTRS.keys())))
