FROM python:3.11-slim

ENV PIP_DEFAULT_TIMEOUT=1200 \
    PIP_RETRIES=20 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Upgrade pip first (fewer resolver quirks)
COPY requirements.txt requirements.txt
COPY constraints.txt constraints.txt
RUN python -m pip install --upgrade pip setuptools wheel

# Install a pinned CPU-only torch from the official index (faster/reliable)
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu -c constraints.txt torch==2.4.1

# Now install the rest, respecting the same constraints (prevents pip from upgrading torch to 2.8.0)
RUN pip install -r requirements.txt -c constraints.txt --prefer-binary

COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "market_sentiment_analyzer/ui_streamlit.py", "--server.port=8501", "--server.headless=true"]
