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

## Screenshots
![Dashboard demo](screenshots/dashboard.png)