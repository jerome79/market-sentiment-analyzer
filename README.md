# Market Sentiment Analyzer

## Overview

A comprehensive financial news sentiment analysis platform that ingests CSV data, applies state-of-the-art sentiment analysis models, and provides interactive visualizations for alpha generation and risk management. The system supports both rule-based (VADER) and transformer-based models (FinBERT, RoBERTa) for accurate financial sentiment classification.

## 🚀 Features

- **Multi-format Data Ingestion**: CSV uploads and folder-based processing with automatic column detection
- **Advanced Sentiment Models**: 
  - VADER (ultra-fast, rule-based)
  - RoBERTa (CardiffNLP) - social media optimized
  - FinBERT (ProsusAI) - financial domain specific
  - FinBERT (Tone) - financial tone analysis
- **Interactive Dashboard**: Real-time sentiment trends by market, ticker, and sector
- **Robust Data Processing**: Automatic normalization, sector mapping, and export capabilities
- **Production-Ready**: Comprehensive test coverage, type hints, and PEP8 compliance

## 📦 Installation

### Quick Start

```bash
# Clone the repository
git clone https://github.com/jerome79/market-sentiment-analyzer.git
cd market-sentiment-analyzer

# Copy environment configuration
cp .env.example .env

# Install dependencies
pip install -r requirements.txt

# Launch the dashboard
streamlit run app/ui_streamlit.py
```

### System Requirements

- Python 3.8+
- 4GB+ RAM (for transformer models)
- Modern web browser for dashboard

## 🎯 Quickstart Guide

### 1. Prepare Your Data

Ensure your CSV files contain at least one text column with headlines or news content:

```csv
date,ticker,headline,text
2025-01-01,AAPL,Apple rises,Apple stock surges after strong earnings
2025-01-01,TSLA,Tesla falls,Tesla shares dip after recall announcement
```

**Supported column names** (case-insensitive):
- **Text content**: `headline`, `title`, `text`, `content`
- **Date**: `date`, `time`, `timestamp`
- **Ticker**: `ticker`, `symbol`, `stock`

### 2. Configure Environment (Optional)

Edit `.env` to customize data paths:

```bash
# Data source configuration
NEWS_CSV_DIR=data/news_csvs/           # Path to CSV files
SECTOR_MAP_CSV=data/sector_map.csv     # Optional: ticker-to-sector mapping
SENTIMENT_MODEL=ProsusAI/finbert       # Default model for HF classifiers
```

### 3. Run the Application

```bash
# Start the Streamlit dashboard
streamlit run app/ui_streamlit.py

# Alternative: Use Makefile shortcuts
make ui      # Launch dashboard
make ingest  # Command-line batch processing
make test    # Run test suite
```

### 4. Process Data

1. **Upload Files**: Drag and drop CSV files in the dashboard
2. **Select Model**: Choose between VADER (fast) or FinBERT (accurate)
3. **Process**: Click "🚀 Process Data" to apply sentiment analysis
4. **Visualize**: Switch to Dashboard tab for interactive charts

## 🛠 API Usage

### Programmatic Access

```python
from app.sentiment import BaselineVader, HFClassifier
from app.ingest import load_csv_dir, normalize_and_save
from app.plots import sentiment_trend_by_date

# Load and process data
raw_data = load_csv_dir("data/news/")
clean_data = normalize_and_save(raw_data, "data/processed.parquet")

# Apply sentiment analysis
vader_model = BaselineVader()
sentiments = vader_model.predict(["Great earnings!", "Market volatility"])
print(sentiments)  # [1, 0] for positive, neutral

# Use transformer model for higher accuracy
finbert = HFClassifier("ProsusAI/finbert")
advanced_sentiments = finbert.predict(["Strong Q4 results", "Regulatory concerns"])

# Generate visualizations
import pandas as pd
labeled_df = pd.read_parquet("data/news_labeled.parquet")
fig = sentiment_trend_by_date(labeled_df)
fig.savefig("market_sentiment.png")
```

### Batch Processing

```python
import os
from pathlib import Path
from app.ingest import load_csv_dir, normalize_and_save
from app.sentiment import HFClassifier

# Batch process news data
def process_news_batch(input_dir: str, output_path: str, model_name: str = "ProsusAI/finbert"):
    """Process a directory of CSV files with sentiment analysis."""
    
    # Load raw data
    print(f"Loading CSV files from {input_dir}...")
    raw_data = load_csv_dir(input_dir)
    
    if raw_data.empty:
        print("No data found!")
        return
    
    # Normalize data
    normalized_data = normalize_and_save(raw_data, output_path.replace('.parquet', '_raw.parquet'))
    
    # Apply sentiment analysis
    print(f"Applying sentiment analysis with {model_name}...")
    model = HFClassifier(model_name)
    
    texts = normalized_data['text'].fillna('').tolist()
    sentiments = model.predict(texts)
    
    # Add sentiment scores
    normalized_data['sentiment'] = sentiments
    
    # Save final results
    normalized_data.to_parquet(output_path, index=False)
    print(f"Processed {len(normalized_data)} records → {output_path}")

# Usage
process_news_batch("data/news_csvs/", "data/sentiment_results.parquet")
```

## 📊 Dashboard Screenshot

![Market Sentiment Dashboard](dashboard_screenshot.png)

*The interactive dashboard provides real-time sentiment visualization across market, ticker, and sector dimensions with export capabilities.*

## 🎛 Available Models

| Model | Speed | Accuracy | Use Case |
|-------|-------|----------|----------|
| **VADER** | ⚡ Fastest | ⭐⭐⭐ Good | Real-time processing, social media |
| **RoBERTa (CardiffNLP)** | ⚡⚡ Fast | ⭐⭐⭐⭐ Better | Social media, general finance |
| **FinBERT (ProsusAI)** | ⚡⚡⚡ Moderate | ⭐⭐⭐⭐⭐ Best | Financial news, earnings calls |
| **FinBERT (Tone)** | ⚡⚡⚡ Moderate | ⭐⭐⭐⭐⭐ Best | Financial tone analysis |

**Recommendations**:
- **Development/Testing**: Use VADER for fast iteration
- **Production (Real-time)**: Use RoBERTa for balanced speed/accuracy
- **Production (Batch)**: Use FinBERT for maximum accuracy

## 📈 Example Data Formats

### Input CSV Example

```csv
date,ticker,headline,text,source
2025-01-01,AAPL,Apple Q4 Earnings Beat,Apple Inc. reported strong Q4 earnings with revenue growth of 15%,reuters
2025-01-01,TSLA,Tesla Recalls 500K Vehicles,Tesla announces voluntary recall affecting Model S and Model X vehicles,bloomberg
2025-01-02,MSFT,Microsoft Cloud Growth,Microsoft Azure revenue increased 40% year-over-year in latest quarter,cnbc
```

### Labeled Output Example

```csv
date,ticker,headline,text,source,sentiment,confidence
2025-01-01,AAPL,Apple Q4 Earnings Beat,Apple Inc. reported strong Q4 earnings...,reuters,1,0.92
2025-01-01,TSLA,Tesla Recalls 500K Vehicles,Tesla announces voluntary recall...,bloomberg,-1,0.87
2025-01-02,MSFT,Microsoft Cloud Growth,Microsoft Azure revenue increased...,cnbc,1,0.89
```

**Sentiment Labels**:
- `1`: Positive sentiment (bullish)
- `0`: Neutral sentiment
- `-1`: Negative sentiment (bearish)

## 🗂 Project Structure

```
market-sentiment-analyzer/
├── app/
│   ├── __init__.py
│   ├── sentiment.py          # Sentiment analysis models
│   ├── ingest.py             # Data loading and normalization
│   ├── plots.py              # Visualization functions
│   ├── ui_streamlit.py       # Web dashboard
│   ├── test_sentiment.py     # Sentiment model tests
│   ├── test_ingest.py        # Data processing tests
│   ├── test_plots.py         # Plotting tests
│   └── test_smoke.py         # Basic import tests
├── data/                     # Data storage directory
├── requirements.txt          # Python dependencies
├── .env.example             # Environment configuration template
├── .flake8                  # Code style configuration
├── .gitignore              # Git ignore rules
├── MakeFile                # Build shortcuts
└── README.md               # This file
```

## 🛣 Roadmap

### Short Term (v1.1)
- [ ] **Enhanced Error Handling**: Graceful handling of malformed data and network issues
- [ ] **Additional Models**: Support for custom transformer models and ensemble methods
- [ ] **Real-time Processing**: WebSocket integration for live news feeds
- [ ] **Advanced Metrics**: Confidence intervals, sentiment volatility, and trend strength

### Medium Term (v1.2)
- [ ] **Multi-language Support**: Sentiment analysis for non-English financial news
- [ ] **Advanced Visualizations**: Candlestick charts with sentiment overlay, heatmaps
- [ ] **API Endpoints**: REST API for programmatic access and integration
- [ ] **Database Integration**: PostgreSQL/MongoDB support for large-scale data storage

### Long Term (v2.0)
- [ ] **Machine Learning Pipeline**: Automated model retraining and drift detection
- [ ] **Advanced NLP Features**: 
  - Named entity recognition for company/person extraction
  - Topic modeling for thematic analysis
  - Sarcasm and context-aware sentiment detection
- [ ] **Trading Integration**: Integration with broker APIs for automated trading signals
- [ ] **Enterprise Features**: 
  - User authentication and role-based access
  - Multi-tenant support
  - Audit logging and compliance tools

### Research & Development
- [ ] **Domain Adaptation**: Fine-tuning models on proprietary financial datasets
- [ ] **Causal Analysis**: Understanding sentiment impact on stock price movements
- [ ] **Risk Modeling**: Sentiment-based risk metrics and portfolio optimization
- [ ] **Alternative Data**: Integration with social media, earnings call transcripts, SEC filings

## 🧪 Testing & Development

### Running Tests

```bash
# Run all tests
make test

# Run specific test modules
python -m pytest app/test_sentiment.py -v
python -m pytest app/test_ingest.py -v
python -m pytest app/test_plots.py -v

# Run with coverage
python -m pytest app/ --cov=app --cov-report=html
```

### Code Quality

```bash
# Check PEP8 compliance
flake8 app/

# Format code
black app/ --line-length=88

# Type checking (if mypy configured)
mypy app/
```

### Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes with tests
4. Ensure code quality: `make test && flake8 app/`
5. Commit changes: `git commit -m 'Add amazing feature'`
6. Push to branch: `git push origin feature/amazing-feature`
7. Open a Pull Request

## ⚠️ Limitations & Considerations

### Current Limitations
- **Model Drift**: Sentiment models may become less accurate over time as language evolves
- **Sarcasm Detection**: Current models struggle with sarcastic or ironic financial commentary
- **Domain-Specific Terms**: May misinterpret technical financial jargon or new terminology
- **Context Sensitivity**: Single headlines lack broader market context for nuanced analysis

### Performance Considerations
- **Memory Usage**: Transformer models require significant RAM (2-4GB+ per model)
- **Processing Speed**: Batch processing recommended for large datasets (>10k records)
- **Model Loading**: First prediction with transformer models includes ~30s initialization time

### Data Quality
- **Encoding**: Ensure CSV files use UTF-8 encoding for international characters
- **Date Formats**: Supports ISO format (YYYY-MM-DD) and common variants
- **Missing Data**: Robust handling of missing tickers, dates, or text content

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Acknowledgments

- **Sentiment Models**: Thanks to HuggingFace, CardiffNLP, ProsusAI for pre-trained models
- **VADER**: [VADER Sentiment Analysis](https://github.com/cjhutto/vaderSentiment)
- **Streamlit**: For the excellent web framework enabling rapid dashboard development
- **Financial Community**: For feedback and feature requests driving development priorities

---

## 📞 Support

- **Documentation**: [GitHub Wiki](https://github.com/jerome79/market-sentiment-analyzer/wiki)
- **Issues**: [GitHub Issues](https://github.com/jerome79/market-sentiment-analyzer/issues)
- **Discussions**: [GitHub Discussions](https://github.com/jerome79/market-sentiment-analyzer/discussions)

**Built with ❤️ for the financial analysis community**