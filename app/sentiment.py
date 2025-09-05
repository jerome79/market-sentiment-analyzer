"""
Sentiment model wrappers for financial news analysis.

Classes:
    BaselineVader: Wrapper for VADER sentiment analysis.
    HFClassifier: Wrapper for HuggingFace transformer-based models.
"""
import os
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

class BaselineVader:
    """
    VADER-based sentiment analyzer for financial headlines/text.

    Methods:
        predict(texts: list[str]) -> list[int]:
            Assigns sentiment scores (-1 negative, 0 neutral, 1 positive).

    Example:
        model = BaselineVader()
        sentiments = model.predict(["Stocks rally", "Market crashes"])
    """
    def __init__(self): self.v = SentimentIntensityAnalyzer()
    def predict(self, texts):
        """
                Predicts sentiment scores for a list of texts.

                Args:
                    texts (List[str]): List of input texts.

                Returns:
                    List[int]: List of sentiment scores (1 for positive, -1 for negative, 0 for neutral).
                """
        # returns -1/0/1
        out = []
        for t in texts:
            vs = self.v.polarity_scores(t or "")
            sc = vs["compound"]
            out.append(1 if sc > 0.05 else (-1 if sc < -0.05 else 0))
        return out

class HFClassifier:
    """HuggingFace transformer-based sentiment classifier."""
    def __init__(self, model_id=None):
        self.model_id = model_id or os.getenv("SENTIMENT_MODEL","cardiffnlp/twitter-roberta-base-sentiment-latest")
        self.tok = AutoTokenizer.from_pretrained(self.model_id)
        self.mdl = AutoModelForSequenceClassification.from_pretrained(self.model_id)
        self.mdl.eval()
        self.lbl = {0:-1, 1:0, 2:1}  # most 3-class financial models: neg/neu/pos
    @torch.inference_mode()
    def predict(self, texts):
        out = []
        for i in range(0, len(texts), 16):
            batch = texts[i:i+16]
            enc = self.tok(batch, truncation=True, padding=True, max_length=128, return_tensors="pt")
            logits = self.mdl(**enc).logits
            preds = logits.argmax(-1).cpu().tolist()
            out.extend([self.lbl.get(p,0) for p in preds])
        return out
