import types
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from app.sentiment import HFClassifier


class DummyTok:
    def __call__(self, batch, truncation=True, padding=True, max_length=96, return_tensors="pt"):
        # simulate tokenizer output: only input_ids is sufficient for our dummy model
        bsz = len(batch)
        return {"input_ids": torch.zeros((bsz, 2), dtype=torch.long)}


class DummyModel:
    def to(self, *_, **__):
        return self

    def eval(self):
        return self

    # Simulate logits where argmax is class 2 (mapped to +1)
    def __call__(self, **enc):
        bsz = enc["input_ids"].shape[0]
        logits = torch.tensor([[0.1, 0.2, 0.7]]).repeat(bsz, 1)
        return types.SimpleNamespace(logits=logits)


def test_hfclassifier_predict_is_mocked_and_maps_to_int_labels(monkeypatch):
    # Fully mock transformers to avoid network/model downloads
    import app.sentiment as sent

    monkeypatch.setattr(sent, "AutoTokenizer", types.SimpleNamespace(from_pretrained=lambda *a, **k: DummyTok()))
    monkeypatch.setattr(sent, "AutoModelForSequenceClassification", types.SimpleNamespace(from_pretrained=lambda *a, **k: DummyModel()))

    model = HFClassifier(model_id="dummy/model")
    texts = ["foo", "bar", "baz"]
    out = model.predict(texts)
    assert out == [1, 1, 1]  # class index 2 -> label +1 via mapping
