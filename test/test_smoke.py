def test_imports():
    import app.ingest as i
    import app.sentiment as s

    assert hasattr(s, "BaselineVader")
    assert hasattr(i, "load_csv_dir")
