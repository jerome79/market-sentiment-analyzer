# Dashboard Documentation

This document provides detailed information about using the Market Sentiment Analyzer dashboard.

## Overview

The dashboard is a Streamlit-based web interface that provides an intuitive way to:
- Upload and process financial news data
- Analyze sentiment using multiple models
- Visualize sentiment trends across different dimensions
- Export processed data for further analysis

## Getting Started

### Launching the Dashboard

```bash
# Using Makefile
make ui

# Or directly
streamlit run app/ui_streamlit.py
```

The dashboard will open in your default web browser at `http://localhost:8501`.

## Dashboard Components

### 1. Sidebar - Debug Information

The sidebar displays helpful debugging information:

- **Current Working Directory**: Shows where the application is running
- **Environment Variables**: Displays configuration settings
  - `NEWS_CSV_DIR`: Directory path for CSV files
  - `SECTOR_MAP_CSV`: Path to sector mapping file
  - `SENTIMENT_MODEL`: Default HuggingFace model
- **Python Version**: Currently installed Python version

### 2. Main Interface Tabs

#### Tab 1: Ingest/Label

This tab handles data upload and sentiment analysis.

**Data Input Options:**

1. **File Upload**
   - Click "Browse files" to select CSV files
   - Multiple files can be uploaded simultaneously
   - Supports drag-and-drop functionality
   - Files are processed in the browser

2. **Directory Path**
   - Specify a folder containing CSV files
   - Path can be absolute or relative to repository root
   - All `.csv` files in the directory are processed

**Model Selection:**

Choose from four sentiment analysis models:

- **VADER (fast)**: Rule-based sentiment analyzer
  - Fastest processing speed
  - Good for general sentiment analysis
  - No GPU required

- **RoBERTa (CardiffNLP)**: Transformer-based model
  - Balanced speed and accuracy
  - Trained on Twitter data
  - Good for social media-style text

- **FinBERT (ProsusAI)**: Financial domain-specific model
  - Optimized for financial news
  - Higher accuracy for financial content
  - Slower processing but more precise

- **FinBERT (Tone)**: Alternative financial model
  - Another financial domain option
  - Different training approach
  - Good for formal financial documents

**Processing Flow:**

1. Click "Ingest & Label" to start processing
2. The system will:
   - Load and validate your data
   - Auto-detect column types (date, ticker, text)
   - Apply the selected sentiment model
   - Save results to `data/news_labeled.parquet`
3. Preview the first 20 rows of processed data
4. Success message shows total rows processed

**Error Handling:**

- Invalid CSV files are skipped with warnings
- Missing required columns trigger error messages
- Processing errors are displayed with full stack traces

#### Tab 2: Dashboard

This tab provides visualization and analysis tools.

**Aggregation Options:**

1. **Market (Date)**
   - Shows overall market sentiment trends
   - Aggregates all news sources by date
   - Useful for identifying market-wide sentiment shifts

2. **Ticker + Date**
   - Individual stock sentiment analysis
   - Select specific ticker symbols from dropdown
   - Track sentiment for particular companies over time
   - Helpful for stock-specific analysis

3. **Sector + Date**
   - Industry-wide sentiment patterns
   - Requires sector mapping configuration
   - Compare sentiment across different sectors
   - Identify sector rotation opportunities

**Visualization Features:**

- **Interactive Plots**: Built with matplotlib
- **Time Series Charts**: Show sentiment trends over time
- **Grid Lines**: Improve readability
- **Clear Labels**: Date on X-axis, sentiment score on Y-axis
- **Automatic Scaling**: Adjusts to data range

**Export Functionality:**

- **Download Button**: Export processed data as CSV
- **Data Preservation**: Maintains all columns and metadata
- **Integration Ready**: Compatible with Excel, R, Python, etc.

## Data Requirements

### Input Data Format

**Required Columns:**
- At least one text column (headline, title, text, etc.)

**Optional Columns:**
- `date` or `time`: For temporal analysis
- `ticker` or `symbol`: For stock-specific analysis
- `source`: To track data origins

**Supported Patterns:**
The system automatically detects columns using case-insensitive matching:
- Text: headline, title, text, content, message
- Date: date, time, timestamp, datetime
- Ticker: ticker, symbol, stock, company

### Output Data Structure

Processed data includes:
- **Original columns**: All input data is preserved
- **sentiment**: Integer score (-1, 0, 1)
- **confidence**: Probability score (0.0 to 1.0) [if available]
- **sector**: Industry classification [if sector mapping provided]

## Configuration

### Environment Variables

Configure the dashboard behavior through environment variables:

```bash
# Data paths
NEWS_CSV_DIR=data/news              # Default CSV directory
SECTOR_MAP_CSV=data/sectors.csv     # Sector mapping file

# Model settings
SENTIMENT_MODEL=ProsusAI/finbert    # Default HuggingFace model
```

### Sector Mapping

To enable sector analysis, create a CSV file with ticker-to-sector mappings:

```csv
ticker,sector,industry
AAPL,Technology,Consumer Electronics
GOOGL,Technology,Internet Services
JPM,Financial,Banking
XOM,Energy,Oil & Gas
JNJ,Healthcare,Pharmaceuticals
```

## Performance Considerations

### Processing Speed

- **VADER**: Processes thousands of texts per second
- **RoBERTa**: Moderate speed, ~16 texts per batch
- **FinBERT**: Slower but more accurate, ~16 texts per batch

### Memory Usage

- **Batch Processing**: Models process data in chunks
- **Caching**: HuggingFace models are cached after first load
- **Streamlit Caching**: UI elements are cached for performance

### Large Datasets

For datasets with >10,000 rows:
- Use VADER for initial analysis
- Consider processing subsets with FinBERT
- Monitor memory usage during processing

## Troubleshooting

### Common Issues

**1. "No text-like column found"**
- Ensure your CSV has columns like 'headline', 'title', or 'text'
- Check column names for typos
- Verify CSV format and encoding (UTF-8 recommended)

**2. "No rows loaded from folder"**
- Check the directory path is correct
- Ensure the directory contains .csv files
- Verify file permissions

**3. Model loading errors**
- Check internet connection (models download on first use)
- Verify sufficient disk space for model cache
- Try switching to VADER if HuggingFace models fail

**4. Empty visualizations**
- Ensure data has been processed in the Ingest/Label tab
- Check that sentiment analysis completed successfully
- Verify date columns are properly formatted

### Debug Information

Use the sidebar debug panel to verify:
- Correct working directory
- Environment variable settings
- Python version compatibility

### Performance Issues

If processing is slow:
- Use VADER for large datasets
- Process data in smaller batches
- Close other browser tabs to free memory
- Restart the Streamlit app if memory usage is high

## Best Practices

### Data Preparation

1. **Clean Your Data**
   - Remove empty rows and columns
   - Ensure consistent date formats
   - Standardize ticker symbols

2. **Column Naming**
   - Use clear, descriptive column names
   - Follow common patterns (date, ticker, headline)
   - Avoid special characters and spaces

3. **File Organization**
   - Organize CSV files by date or source
   - Use consistent naming conventions
   - Keep raw and processed data separate

### Analysis Workflow

1. **Start with VADER**
   - Quick overview of sentiment distribution
   - Identify interesting patterns
   - Validate data quality

2. **Deep Dive with FinBERT**
   - Focus on specific time periods or stocks
   - Use for final analysis and reporting
   - Higher confidence in financial context

3. **Export and Integrate**
   - Download processed data for external analysis
   - Combine with price data for correlation analysis
   - Use in trading algorithms or risk models

### Visualization Tips

- **Market Overview**: Use for broad market sentiment trends
- **Ticker Analysis**: Focus on specific investment opportunities
- **Sector Analysis**: Identify rotation and relative strength
- **Time Ranges**: Filter by date ranges for specific periods

## Advanced Features

### Custom Models

To use custom HuggingFace models:
1. Set `SENTIMENT_MODEL` environment variable
2. Ensure model follows expected output format
3. Test with small dataset first

### Batch Processing

For automated workflows:
1. Use command line ingestion: `python app/ingest.py`
2. Configure data directories in `.env`
3. Integrate with data pipelines

### API Integration

Future versions will support:
- REST API endpoints
- Real-time news feeds
- Database connections
- Cloud deployment

## Support

For dashboard-specific issues:
- Check the debug sidebar for configuration
- Verify data format requirements
- Test with sample data first
- Report issues with screenshots and error messages

---

*Last updated: Current version*