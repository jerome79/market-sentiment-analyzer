import os


def test_configure():
    # Keep PyTorch/transformers quiet and deterministic in tests
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("TORCH_NUM_THREADS", "1")


# Use a headless backend for matplotlib so tests can run in CI/servers
def test_sessionstart():
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
    except Exception:
        # If matplotlib is not installed for some reason, ignore
        pass
