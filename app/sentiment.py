"""
Sentiment model wrappers for financial news analysis.

Classes:
    BaselineVader: Wrapper for VADER sentiment analysis.
    HuggingFaceClassifier: Wrapper for HuggingFace transformer-based models.
"""
import os
from typing import List

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


class BaselineVader:
    """
    VADER-based sentiment analyzer for financial headlines/text.

    This class provides a simple interface to VADER sentiment analysis,
    specifically tuned for financial text with appropriate thresholds.

    Attributes:
        analyzer (SentimentIntensityAnalyzer): VADER sentiment analyzer instance.

    Methods:
        predict(texts: List[str]) -> List[int]:
            Assigns sentiment scores (-1 negative, 0 neutral, 1 positive).

    Example:
        >>> model = BaselineVader()
        >>> sentiments = model.predict(["Stocks rally", "Market crashes"])
        >>> print(sentiments)  # [1, -1]
    """

    def __init__(self):
        """Initialize the VADER sentiment analyzer."""
        self.analyzer = SentimentIntensityAnalyzer()

    def predict(self, texts: List[str]) -> List[int]:
        """
        Predict sentiment for a list of texts.

        Args:
            texts: List of text strings to analyze.

        Returns:
            List of sentiment scores: -1 (negative), 0 (neutral), 1 (positive).

        Raises:
            TypeError: If texts is not a list.
        """
        if not isinstance(texts, list):
            raise TypeError("Input texts must be a list")

        results = []
        for text in texts:
            # Handle empty or None texts
            text = text or ""
            scores = self.analyzer.polarity_scores(text)
            compound_score = scores["compound"]

            # Apply thresholds optimized for financial sentiment
            if compound_score > 0.05:
                sentiment = 1  # Positive
            elif compound_score < -0.05:
                sentiment = -1  # Negative
            else:
                sentiment = 0  # Neutral

            results.append(sentiment)

        return results


class HuggingFaceClassifier:
    """
    HuggingFace transformer-based sentiment classifier.

    This class provides an interface to HuggingFace transformer models
    for sentiment analysis, with batch processing support.

    Attributes:
        model_id (str): HuggingFace model identifier.
        tokenizer: Tokenizer for the model.
        model: Pre-trained sentiment classification model.
        label_mapping (dict): Maps model outputs to sentiment scores.
    """

    def __init__(self, model_id: str = None):
        """
        Initialize the HuggingFace classifier.

        Args:
            model_id: HuggingFace model identifier. If None, uses environment
                     variable SENTIMENT_MODEL or defaults to CardiffNLP RoBERTa.
        """
        default_model = "cardiffnlp/twitter-roberta-base-sentiment-latest"
        self.model_id = model_id or os.getenv("SENTIMENT_MODEL", default_model)

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_id
            )
            self.model.eval()
        except Exception as e:
            raise RuntimeError(f"Failed to load model {self.model_id}: {e}")

        # Map model outputs to sentiment scores: neg/neu/pos -> -1/0/1
        self.label_mapping = {0: -1, 1: 0, 2: 1}

    @torch.inference_mode()
    def predict(self, texts: List[str]) -> List[int]:
        """
        Predict sentiment for a list of texts using batch processing.

        Args:
            texts: List of text strings to analyze.

        Returns:
            List of sentiment scores: -1 (negative), 0 (neutral), 1 (positive).

        Raises:
            TypeError: If texts is not a list.
            RuntimeError: If prediction fails.
        """
        if not isinstance(texts, list):
            raise TypeError("Input texts must be a list")

        if not texts:
            return []

        results = []
        batch_size = 16  # Process in batches to manage memory

        try:
            for batch_start in range(0, len(texts), batch_size):
                batch_end = batch_start + batch_size
                batch_texts = texts[batch_start:batch_end]

                # Tokenize batch
                encoded = self.tokenizer(
                    batch_texts,
                    truncation=True,
                    padding=True,
                    max_length=128,
                    return_tensors="pt",
                )

                # Get predictions
                logits = self.model(**encoded).logits
                predictions = logits.argmax(-1).cpu().tolist()

                # Map to sentiment scores
                batch_results = [
                    self.label_mapping.get(pred, 0) for pred in predictions
                ]
                results.extend(batch_results)

        except Exception as e:
            raise RuntimeError(f"Sentiment prediction failed: {e}")

        return results


# Keep backward compatibility
HFClassifier = HuggingFaceClassifier
