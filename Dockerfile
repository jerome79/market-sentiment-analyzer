FROM python:3.11-slim

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Optional dev tools
COPY requirements-dev.txt requirements-dev.txt
RUN pip install --no-cache-dir -r requirements-dev.txt || true

COPY . .
ENV PYTHONUNBUFFERED=1
ENV PORT=8501

EXPOSE 8501
CMD ["streamlit", "run", "market_sentiment_analyzer/ui_streamlit.py", "--server.port=8501", "--server.headless=true"]
