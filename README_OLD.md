# Market Sentiment Analyzer

## Overview
Analyze financial news/headlines to label sentiment and track trends for alpha generation and risk alerts.

## Features
- Ingest CSV data
- Label sentiment using VADER/FinBERT/RoBERTa
- Dashboard for market/ticker/sector sentiment

## Installation
```bash
git clone https://github.com/jerome79/market-sentiment-analyzer.git
cd market-sentiment-analyzer
cp .env.example .env
pip install -r requirements.txt
```

## Usage
1. Upload CSVs or specify a folder
2. Select sentiment model
3. View dashboard for insights

## Models
- VADER (fast)
- RoBERTa (CardiffNLP)
- FinBERT (ProsusAI, Tone)

## Limitations & Roadmap
- Handles drift, sarcasm, domain terms (future improvements)
- See `issues/` for next steps

## License
MIT
## Example Input CSV

```csv
date,ticker,headline,text
2025-01-01,AAPL,Apple rises,Apple stock surges after earnings
2025-01-01,TSLA,Tesla falls,Tesla shares dip after recall news
```

## Example Labeled Output

```csv
date,ticker,headline,text,sentiment,confidence
2025-01-01,AAPL,Apple rises,Apple stock surges after earnings,1,0.95
2025-01-01,TSLA,Tesla falls,Tesla shares dip after recall news,-1,0.90
```
...