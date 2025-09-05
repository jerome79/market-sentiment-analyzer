"""
Sentiment model wrappers for financial news analysis.

This module provides sentiment analysis models for financial text including
VADER (rule-based) and transformer-based models from HuggingFace.

Classes:
    BaselineVader: Fast rule-based sentiment analyzer using VADER.
    HFClassifier: Deep learning sentiment classifier using transformers.

Typical usage example:
    # For quick analysis
    vader_model = BaselineVader()
    sentiments = vader_model.predict(["Stock prices soar", "Market crashes"])

    # For higher accuracy
    bert_model = HFClassifier("ProsusAI/finbert")
    sentiments = bert_model.predict(["Stock prices soar", "Market crashes"])
"""

import os
from typing import List, Optional

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


class BaselineVader:
    """
    VADER-based sentiment analyzer for financial headlines and text.

    VADER (Valence Aware Dictionary and sEntiment Reasoner) is a lexicon and
    rule-based sentiment analysis tool specifically designed for social media
    text. It works well for financial news without requiring training data.

    Attributes:
        vader_analyzer: The VADER sentiment intensity analyzer instance.

    Methods:
        predict: Analyze sentiment of text and return integer labels.

    Example:
        >>> model = BaselineVader()
        >>> texts = ["Apple stock surges after earnings", "Tesla shares plummet"]
        >>> sentiments = model.predict(texts)
        >>> print(sentiments)  # [1, -1] for positive, negative
    """

    def __init__(self) -> None:
        """Initialize the VADER sentiment analyzer."""
        self.vader_analyzer = SentimentIntensityAnalyzer()

    def predict(self, texts: List[str]) -> List[int]:
        """
        Predict sentiment labels for a list of texts.

        Args:
            texts: List of text strings to analyze.

        Returns:
            List of sentiment labels: -1 (negative), 0 (neutral), 1 (positive).

        Example:
            >>> model = BaselineVader()
            >>> sentiments = model.predict(["Great earnings!", "Stock drops"])
            >>> print(sentiments)  # [1, -1]
        """
        sentiment_labels = []
        for text in texts:
            # Get compound score from VADER (ranges from -1 to 1)
            scores = self.vader_analyzer.polarity_scores(text or "")
            compound_score = scores["compound"]

            # Convert to discrete labels using thresholds
            if compound_score > 0.05:
                label = 1  # Positive
            elif compound_score < -0.05:
                label = -1  # Negative
            else:
                label = 0  # Neutral

            sentiment_labels.append(label)

        return sentiment_labels


class HFClassifier:
    """
    HuggingFace transformer-based sentiment classifier for financial text.

    This class wraps pre-trained transformer models from HuggingFace that are
    specifically fine-tuned for financial sentiment analysis, providing higher
    accuracy than rule-based approaches at the cost of computational complexity.

    Attributes:
        model_id: HuggingFace model identifier.
        tokenizer: Model tokenizer for text preprocessing.
        model: Pre-trained sentiment classification model.
        label_mapping: Maps model outputs to standardized labels.

    Example:
        >>> classifier = HFClassifier("ProsusAI/finbert")
        >>> texts = ["Strong quarterly results", "Regulatory concerns"]
        >>> sentiments = classifier.predict(texts)
        >>> print(sentiments)  # [1, -1] for positive, negative
    """

    def __init__(self, model_id: Optional[str] = None) -> None:
        """
        Initialize the HuggingFace sentiment classifier.

        Args:
            model_id: HuggingFace model identifier. If None, uses environment
                variable SENTIMENT_MODEL or defaults to RoBERTa model.
        """
        default_model = "cardiffnlp/twitter-roberta-base-sentiment-latest"
        self.model_id = model_id or os.getenv("SENTIMENT_MODEL", default_model)

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_id)
        self.model.eval()  # Set to evaluation mode

        # Standard mapping for 3-class models: negative=0, neutral=1, positive=2
        self.label_mapping = {0: -1, 1: 0, 2: 1}

    @torch.inference_mode()
    def predict(self, texts: List[str]) -> List[int]:
        """
        Predict sentiment labels for a list of texts using batch processing.

        Args:
            texts: List of text strings to analyze.

        Returns:
            List of sentiment labels: -1 (negative), 0 (neutral), 1 (positive).

        Example:
            >>> classifier = HFClassifier()
            >>> texts = ["Revenue beats expectations", "Lawsuit filed"]
            >>> sentiments = classifier.predict(texts)
            >>> print(sentiments)  # [1, -1]
        """
        predictions = []
        batch_size = 16  # Process in batches to manage memory

        for batch_start in range(0, len(texts), batch_size):
            # Extract current batch
            batch_texts = texts[batch_start:batch_start + batch_size]

            # Tokenize batch with padding and truncation
            encoded_batch = self.tokenizer(
                batch_texts,
                truncation=True,
                padding=True,
                max_length=128,
                return_tensors="pt",
            )

            # Get model predictions
            model_outputs = self.model(**encoded_batch)
            logits = model_outputs.logits

            # Convert logits to predicted class indices
            predicted_classes = logits.argmax(dim=-1).cpu().tolist()

            # Map model outputs to standardized labels
            batch_predictions = [
                self.label_mapping.get(pred, 0) for pred in predicted_classes
            ]
            predictions.extend(batch_predictions)

        return predictions
