import pandas as pd


def load_panel(panel_path: str | None = None) -> pd.DataFrame:
    """Load daily sentiment panel with columns at least: date, ticker, avg_sentiment."""
    panel_path = panel_path or "data/sentiment_panel.parquet"
    df = pd.read_parquet(panel_path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def panel_stats(tickers: list[str], date_from: str, date_to: str, panel_path: str | None = None) -> dict:
    df = load_panel(panel_path)
    mask = (df["date"] >= date_from) & (df["date"] <= date_to) & (df["ticker"].isin(tickers))
    sel = df.loc[mask].copy()
    stats = {
        "tickers": tickers,
        "date_from": date_from,
        "date_to": date_to,
        "avg_sentiment": float(sel["avg_sentiment"].mean()) if not sel.empty else None,
        "n_news": int(sel.shape[0]),
    }
    series = sel[["date", "ticker", "avg_sentiment"]].sort_values(["ticker", "date"])
    series["date"] = series["date"].dt.strftime("%Y-%m-%d")
    return {"stats": stats, "series": series.to_dict(orient="records")}
