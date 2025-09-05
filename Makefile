# Market Sentiment Analyzer Makefile

.PHONY: ui ingest run-tests lint format clean help

# Default target
help:
	@echo "Available commands:"
	@echo "  ui          - Start Streamlit UI"
	@echo "  ingest      - Run data ingestion"
	@echo "  run-tests   - Run all tests"
	@echo "  lint        - Run code linting with flake8"
	@echo "  format      - Format code with black and isort"
	@echo "  clean       - Clean up cache files"

# Start Streamlit UI
ui: 
	streamlit run app/ui_streamlit.py

# Run data ingestion
ingest: 
	python app/ingest.py

# Run tests
run-tests: 
	pytest -q test/

# For backward compatibility
test: run-tests

# Run linting
lint:
	flake8 app/ test/

# Format code
format:
	black app/ test/
	isort app/ test/

# Clean cache files
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	rm -rf .coverage htmlcov/
