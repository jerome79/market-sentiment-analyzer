"""
Unit tests for the sentiment analysis module.

Tests both VADER and HuggingFace-based sentiment classifiers to ensure
they provide consistent output formats and handle edge cases properly.
"""

from unittest.mock import Mock, patch

from app.sentiment import BaselineVader, HFClassifier


class TestBaselineVader:
    """Test cases for the VADER sentiment analyzer."""

    def test_init(self):
        """Test that BaselineVader initializes correctly."""
        model = BaselineVader()
        assert hasattr(model, "vader_analyzer")
        assert model.vader_analyzer is not None

    def test_predict_basic_sentiments(self):
        """Test prediction of basic positive, negative, and neutral sentiments."""
        model = BaselineVader()

        # Test texts with clear sentiment
        test_texts = [
            "Stock prices are soaring! Excellent performance!",  # Positive
            "Market crashed badly today, terrible losses",  # Negative
            "The company reported quarterly results",  # Neutral
        ]

        predictions = model.predict(test_texts)

        # Check return format
        assert isinstance(predictions, list)
        assert len(predictions) == len(test_texts)
        assert all(pred in [-1, 0, 1] for pred in predictions)

        # Check that we get expected sentiment directions
        assert predictions[0] == 1  # Positive
        assert predictions[1] == -1  # Negative
        # Note: neutral prediction might vary, so we don't assert a specific value

    def test_predict_empty_and_none_inputs(self):
        """Test handling of empty strings and None values."""
        model = BaselineVader()

        # Test edge cases
        edge_cases = ["", None, "   ", "neutral text"]
        predictions = model.predict(edge_cases)

        assert len(predictions) == len(edge_cases)
        assert all(pred in [-1, 0, 1] for pred in predictions)

    def test_predict_single_vs_batch(self):
        """Test that single and batch predictions are consistent."""
        model = BaselineVader()

        test_text = "Great earnings report"

        # Single prediction
        single_pred = model.predict([test_text])

        # Batch prediction with same text repeated
        batch_pred = model.predict([test_text, test_text])

        assert single_pred[0] == batch_pred[0] == batch_pred[1]


class TestHFClassifier:
    """Test cases for the HuggingFace sentiment classifier."""

    @patch("app.sentiment.AutoTokenizer")
    @patch("app.sentiment.AutoModelForSequenceClassification")
    def test_init_default_model(self, mock_model_class, mock_tokenizer_class):
        """Test initialization with default model."""
        # Mock the model and tokenizer
        mock_tokenizer = Mock()
        mock_model = Mock()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        mock_model_class.from_pretrained.return_value = mock_model

        classifier = HFClassifier()

        # Check that model ID is set correctly
        assert classifier.model_id is not None
        assert isinstance(classifier.model_id, str)

        # Check that tokenizer and model were loaded
        mock_tokenizer_class.from_pretrained.assert_called_once()
        mock_model_class.from_pretrained.assert_called_once()

    @patch("app.sentiment.AutoTokenizer")
    @patch("app.sentiment.AutoModelForSequenceClassification")
    def test_init_custom_model(self, mock_model_class, mock_tokenizer_class):
        """Test initialization with custom model."""
        custom_model_id = "test/custom-model"

        # Mock the model and tokenizer
        mock_tokenizer = Mock()
        mock_model = Mock()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        mock_model_class.from_pretrained.return_value = mock_model

        classifier = HFClassifier(custom_model_id)

        assert classifier.model_id == custom_model_id
        mock_tokenizer_class.from_pretrained.assert_called_with(custom_model_id)
        mock_model_class.from_pretrained.assert_called_with(custom_model_id)

    @patch("app.sentiment.AutoTokenizer")
    @patch("app.sentiment.AutoModelForSequenceClassification")
    @patch("torch.inference_mode")
    def test_predict_output_format(
        self, mock_inference, mock_model_class, mock_tokenizer_class
    ):
        """Test that predict returns the correct output format."""
        # Setup mocks
        mock_tokenizer = Mock()
        mock_model = Mock()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        mock_model_class.from_pretrained.return_value = mock_model

        # Mock model output
        mock_logits = Mock()
        mock_logits.argmax.return_value.cpu.return_value.tolist.return_value = [0, 1, 2]
        mock_model.return_value.logits = mock_logits

        # Mock tokenizer output
        mock_tokenizer.return_value = {"input_ids": Mock(), "attention_mask": Mock()}

        classifier = HFClassifier("test/model")

        test_texts = ["positive text", "neutral text", "negative text"]
        predictions = classifier.predict(test_texts)

        # Check output format
        assert isinstance(predictions, list)
        assert len(predictions) == len(test_texts)
        assert all(pred in [-1, 0, 1] for pred in predictions)

        # Check that predictions follow label mapping
        expected = [-1, 0, 1]  # Based on label_mapping {0: -1, 1: 0, 2: 1}
        assert predictions == expected

    @patch("app.sentiment.AutoTokenizer")
    @patch("app.sentiment.AutoModelForSequenceClassification")
    def test_predict_batch_processing(self, mock_model_class, mock_tokenizer_class):
        """Test that large inputs are properly batched."""
        # Setup mocks
        mock_tokenizer = Mock()
        mock_model = Mock()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        mock_model_class.from_pretrained.return_value = mock_model

        # Mock model output for multiple batches
        # Need to return appropriate number of predictions per batch
        mock_logits = Mock()
        # First call returns 16 predictions, second call returns 4
        mock_logits.argmax.return_value.cpu.return_value.tolist.side_effect = [
            [1] * 16,  # First batch of 16
            [1] * 4,  # Second batch of 4
        ]
        mock_model.return_value.logits = mock_logits

        # Mock tokenizer output
        mock_tokenizer.return_value = {"input_ids": Mock(), "attention_mask": Mock()}

        classifier = HFClassifier("test/model")

        # Create a large input that should be split into batches
        large_input = ["test text"] * 20  # Should create 2 batches of 16 and 4

        with patch("torch.inference_mode"):
            predictions = classifier.predict(large_input)

        assert len(predictions) == len(large_input)
        assert all(pred in [-1, 0, 1] for pred in predictions)


def test_sentiment_models_integration():
    """Integration test comparing VADER and mocked HF outputs."""
    # Test VADER on real data
    vader_model = BaselineVader()
    test_texts = [
        "Amazing profits and growth!",
        "Terrible losses and bankruptcy",
        "Company reports quarterly results",
    ]

    vader_predictions = vader_model.predict(test_texts)

    # Basic sanity checks
    assert len(vader_predictions) == len(test_texts)
    assert all(pred in [-1, 0, 1] for pred in vader_predictions)

    # Check that clearly positive and negative texts get appropriate labels
    assert vader_predictions[0] == 1  # Positive
    assert vader_predictions[1] == -1  # Negative
