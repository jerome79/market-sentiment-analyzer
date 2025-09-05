def test_imports():
    import app.sentiment as s
    import app.ingest as i
    assert hasattr(s, "BaselineVader")
    assert hasattr(i, "load_csv_dir")
