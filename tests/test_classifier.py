"""
Test suite for BERT Text Classification
"""

import pytest
import tempfile
import os
from bert_classifier import BERTTextClassifier, create_mock_database


class TestBERTTextClassifier:
    """Test cases for BERTTextClassifier class."""
    
    def test_init(self):
        """Test classifier initialization."""
        classifier = BERTTextClassifier()
        assert classifier.model_name == "bert-base-uncased"
        assert classifier.num_labels == 2
        assert classifier.tokenizer is None
        assert classifier.model is None
    
    def test_init_custom_params(self):
        """Test classifier initialization with custom parameters."""
        classifier = BERTTextClassifier(
            model_name="bert-base-cased",
            num_labels=3
        )
        assert classifier.model_name == "bert-base-cased"
        assert classifier.num_labels == 3
    
    def test_create_label_mappings(self):
        """Test label mapping creation."""
        classifier = BERTTextClassifier()
        labels = ["positive", "negative", "neutral"]
        classifier.create_label_mappings(labels)
        
        assert classifier.label2id == {"negative": 0, "neutral": 1, "positive": 2}
        assert classifier.id2label == {0: "negative", 1: "neutral", 2: "positive"}
    
    def test_prepare_dataset(self):
        """Test dataset preparation."""
        classifier = BERTTextClassifier()
        texts = ["Text 1", "Text 2", "Text 3", "Text 4"]
        labels = ["positive", "negative", "positive", "negative"]
        
        train_dataset, test_dataset = classifier.prepare_dataset(texts, labels)
        
        assert len(train_dataset) + len(test_dataset) == 4
        assert len(train_dataset) > 0
        assert len(test_dataset) > 0
        assert "input_ids" in train_dataset.features
        assert "attention_mask" in train_dataset.features
        assert "label" in train_dataset.features


class TestMockDatabase:
    """Test cases for mock database creation."""
    
    def test_create_mock_database(self):
        """Test mock database creation."""
        texts, labels = create_mock_database()
        
        assert len(texts) == 20
        assert len(labels) == 20
        assert len(set(labels)) == 2
        assert "positive" in labels
        assert "negative" in labels
    
    def test_mock_database_content(self):
        """Test mock database content quality."""
        texts, labels = create_mock_database()
        
        # Check that positive texts contain positive words
        positive_texts = [text for text, label in zip(texts, labels) if label == "positive"]
        negative_texts = [text for text, label in zip(texts, labels) if label == "negative"]
        
        assert len(positive_texts) == 10
        assert len(negative_texts) == 10
        
        # Check for positive sentiment words
        positive_words = ["fantastic", "love", "amazing", "excellent", "perfect"]
        negative_words = ["terrible", "hate", "horrible", "awful", "poor"]
        
        positive_contains_good = any(
            any(word in text.lower() for word in positive_words)
            for text in positive_texts
        )
        negative_contains_bad = any(
            any(word in text.lower() for word in negative_words)
            for text in negative_texts
        )
        
        assert positive_contains_good
        assert negative_contains_bad


class TestIntegration:
    """Integration tests."""
    
    def test_full_pipeline_small_dataset(self):
        """Test full pipeline with small dataset."""
        # Create small dataset
        texts = ["This is great!", "This is terrible!"]
        labels = ["positive", "negative"]
        
        classifier = BERTTextClassifier()
        
        # This test doesn't actually train the model to avoid long execution time
        # Instead, we test the data preparation
        train_dataset, test_dataset = classifier.prepare_dataset(texts, labels)
        
        assert len(train_dataset) + len(test_dataset) == 2
        assert classifier.label2id == {"negative": 0, "positive": 1}
        assert classifier.id2label == {0: "negative", 1: "positive"}


if __name__ == "__main__":
    pytest.main([__file__])
