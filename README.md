# Market Sentiment Analyzer

A comprehensive tool for analyzing financial news sentiment to generate trading insights and risk alerts.

## 🚀 Features

- **Multi-source Data Ingestion**: Load financial news from CSV files or directories
- **Advanced Sentiment Analysis**: Multiple models including VADER, RoBERTa, and FinBERT
- **Interactive Dashboard**: Streamlit-based web interface for data visualization
- **Sector Analysis**: Automatic sector mapping and sentiment trends
- **Export Capabilities**: Download labeled datasets in CSV format
- **Production Ready**: Comprehensive error handling, logging, and testing

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Sentiment Models](#sentiment-models)
- [Data Format](#data-format)
- [Dashboard Features](#dashboard-features)
- [Development](#development)
- [Configuration](#configuration)
- [Testing](#testing)
- [Contributing](#contributing)
- [License](#license)

## 🔧 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/jerome79/market-sentiment-analyzer.git
   cd market-sentiment-analyzer
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env file with your configuration
   ```

## 🚀 Quick Start

### Start the Web Interface

```bash
make ui
# or
streamlit run app/ui_streamlit.py
```

### Command Line Usage

```bash
# Ingest and process data
make ingest
# or
python app/ingest.py

# Run tests
make run-tests
# or
pytest test/
```

## 📊 Usage

### 1. Data Upload

The application supports two methods for data input:

**File Upload**: Use the web interface to upload CSV files directly
**Directory Processing**: Configure a directory path containing CSV files

### 2. Sentiment Analysis

Choose from multiple sentiment analysis models:
- **VADER**: Fast rule-based analyzer optimized for social media text
- **RoBERTa (CardiffNLP)**: Transformer model trained on Twitter data
- **FinBERT (ProsusAI)**: Financial domain-specific BERT model
- **FinBERT (Tone)**: Alternative financial BERT model

### 3. Dashboard Analysis

View sentiment trends across three aggregation levels:
- **Market Overview**: Overall sentiment trends by date
- **Ticker Analysis**: Individual stock sentiment tracking
- **Sector Analysis**: Industry-wide sentiment patterns

## 🤖 Sentiment Models

### VADER (Valence Aware Dictionary and sEntiment Reasoner)
- **Best for**: Real-time processing, social media text
- **Speed**: Very fast
- **Accuracy**: Good for general sentiment
- **Use case**: Quick analysis of large datasets

### RoBERTa (Cardiff NLP)
- **Best for**: Social media and informal text
- **Speed**: Moderate
- **Accuracy**: High for general sentiment
- **Use case**: Balanced speed and accuracy

### FinBERT Models
- **Best for**: Financial news and formal business text
- **Speed**: Slower but more accurate
- **Accuracy**: Highest for financial content
- **Use case**: Professional financial analysis

## 📁 Data Format

### Input CSV Requirements

Your CSV files should contain at least one text column. The system automatically detects columns with these patterns:

**Text Content** (required):
- `headline`, `title`, `text`, or similar

**Optional Columns**:
- `date`, `time` - for temporal analysis
- `ticker`, `symbol` - for stock-specific analysis
- Additional columns are preserved

### Example Input CSV

```csv
date,ticker,headline,text,source
2025-01-01,AAPL,Apple Reports Strong Q4,Apple Inc. reported better than expected earnings...,financial_news
2025-01-01,TSLA,Tesla Faces Challenges,Tesla stock declined following production issues...,market_watch
2025-01-02,MSFT,Microsoft Cloud Growth,Microsoft's cloud division shows continued growth...,tech_times
```

### Output Format

The processed data includes sentiment scores and confidence levels:

```csv
date,ticker,source,headline,text,sentiment,confidence,sector
2025-01-01,AAPL,financial_news,Apple Reports Strong Q4,Apple Inc. reported...,1,0.95,Technology
2025-01-01,TSLA,market_watch,Tesla Faces Challenges,Tesla stock declined...,-1,0.87,Automotive
2025-01-02,MSFT,tech_times,Microsoft Cloud Growth,Microsoft's cloud...,1,0.92,Technology
```

**Sentiment Scale**:
- `1`: Positive sentiment
- `0`: Neutral sentiment  
- `-1`: Negative sentiment

## 📈 Dashboard Features

### Market Overview
- Time series visualization of overall market sentiment
- Aggregated sentiment trends across all news sources
- Filtering by date ranges

### Ticker Analysis
- Individual stock sentiment tracking
- Compare sentiment across different time periods
- Identify sentiment-driven price movements

### Sector Analysis
- Industry-wide sentiment patterns
- Sector rotation insights
- Cross-sector sentiment comparison

### Export Functionality
- Download processed datasets as CSV
- Maintain data lineage and processing metadata
- Integration with external analysis tools

## 🛠 Development

### Code Quality

This project maintains high code quality standards:

```bash
# Run linting
make lint

# Format code
make format

# Run tests
make run-tests

# Clean cache files
make clean
```

### Project Structure

```
market-sentiment-analyzer/
├── app/
│   ├── __init__.py
│   ├── sentiment.py        # Sentiment analysis models
│   ├── ingest.py          # Data ingestion and processing
│   ├── plots.py           # Visualization utilities
│   └── ui_streamlit.py    # Web interface
├── test/
│   ├── __init__.py
│   ├── test_smoke.py      # Basic functionality tests
│   └── test_ui.py         # UI component tests
├── data/                  # Data storage directory
├── requirements.txt       # Python dependencies
├── Makefile              # Development commands
├── .env.example          # Environment configuration template
└── README.md             # This file
```

### Adding New Features

1. Follow PEP 8 style guidelines
2. Add comprehensive docstrings
3. Include error handling
4. Write tests for new functionality
5. Update documentation

## ⚙️ Configuration

### Environment Variables

Create a `.env` file with the following configurations:

```bash
# Data source configuration
NEWS_CSV_DIR=data/news          # Directory containing CSV files
SECTOR_MAP_CSV=data/sectors.csv # Optional sector mapping file

# Model configuration
SENTIMENT_MODEL=ProsusAI/finbert # Default HuggingFace model

# Optional API keys for future features
# ALPHA_VANTAGE_API_KEY=your_key_here
# NEWS_API_KEY=your_key_here
```

### Sector Mapping

To enable sector analysis, provide a CSV file with ticker-to-sector mappings:

```csv
ticker,sector,industry
AAPL,Technology,Consumer Electronics
MSFT,Technology,Software
TSLA,Automotive,Electric Vehicles
JPM,Financial,Banking
```

## 🧪 Testing

### Running Tests

```bash
# Run all tests
make run-tests

# Run specific test files
pytest test/test_smoke.py -v

# Run with coverage
pytest --cov=app test/
```

### Test Categories

- **Smoke Tests**: Basic import and functionality verification
- **Unit Tests**: Individual component testing
- **Integration Tests**: End-to-end workflow testing

### Writing Tests

Follow these guidelines for test development:
- Use descriptive test names
- Include docstrings explaining test purpose
- Test both success and failure scenarios
- Mock external dependencies

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Make your changes**
   - Follow code quality standards
   - Add tests for new functionality
   - Update documentation
4. **Run tests and linting**
   ```bash
   make lint
   make run-tests
   ```
5. **Submit a pull request**

### Code Review Process

- All changes require review
- Tests must pass
- Code coverage should not decrease
- Documentation must be updated

## 📋 Limitations & Future Roadmap

### Current Limitations

- **Language Support**: Currently optimized for English text
- **Real-time Processing**: Batch processing only
- **Data Sources**: CSV files only (no direct API integration)
- **Deployment**: Local deployment only

### Planned Features

- [ ] Real-time news feed integration
- [ ] Multi-language sentiment analysis
- [ ] Advanced ML models and ensemble methods
- [ ] Cloud deployment options
- [ ] REST API for programmatic access
- [ ] Database integration for large datasets
- [ ] Email/Slack alerts for sentiment changes
- [ ] Backtesting capabilities with price data

### Performance Improvements

- [ ] Parallel processing for large datasets
- [ ] Model caching and optimization
- [ ] Incremental data processing
- [ ] Memory usage optimization

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **VADER**: [Hutto & Gilbert (2014)](https://github.com/cjhutto/vaderSentiment)
- **Transformers**: [Hugging Face](https://huggingface.co/transformers/)
- **FinBERT**: [ProsusAI](https://huggingface.co/ProsusAI/finbert)
- **Streamlit**: [Streamlit Team](https://streamlit.io/)

## 📞 Support

For questions, issues, or contributions:

- **GitHub Issues**: [Create an issue](https://github.com/jerome79/market-sentiment-analyzer/issues)
- **Documentation**: Check this README and inline code documentation
- **Community**: Join discussions in GitHub Discussions

---

**Happy Analyzing! 📈✨**