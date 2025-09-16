.PHONY: demo setup run test fmt lint bench docker-up docker-down

PY=python

setup:
    pip install -r requirements.txt
    pip install -r requirements-dev.txt || true
    cp -n .env.example .env || true

run:
    streamlit run market_sentiment_analyzer/ui_streamlit.py --server.port 8501

demo: setup
    $(PY) scripts/bootstrap_sample.py
    $(MAKE) run

bench:
    $(PY) scripts/benchmark.py --csv data/news_tiny.csv --model vader --results out/bench_demo.csv

fmt:
    ruff check --fix || true
    black .

lint:
    ruff check
    black --check .

test:
    pytest -q

docker-up:
    docker compose up --build

docker-down:
    docker compose down -v
