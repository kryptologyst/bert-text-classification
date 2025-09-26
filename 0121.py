"""
Project 121: BERT Fine-tuning for Text Classification
====================================================

A modern implementation of BERT fine-tuning for text classification using the latest
Hugging Face Transformers library and best practices.

Features:
- Modern transformers library with latest techniques
- Comprehensive evaluation metrics
- Model saving and loading
- Data preprocessing pipeline
- Mock database for demonstration
- Streamlit web interface

This is the main script that demonstrates the complete pipeline.
"""

import logging
from bert_classifier import BERTTextClassifier, create_mock_database

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Main function to run the BERT text classification pipeline."""
    logger.info("Starting BERT Text Classification Pipeline")
    
    # Create mock database
    texts, labels = create_mock_database()
    
    # Initialize classifier
    classifier = BERTTextClassifier(num_labels=2)
    classifier.load_tokenizer_and_model()
    
    # Prepare dataset
    train_dataset, test_dataset = classifier.prepare_dataset(texts, labels)
    
    # Train model
    classifier.train(train_dataset, test_dataset, num_epochs=3)
    
    # Evaluate model
    evaluation_report = classifier.evaluate_model(test_dataset)
    
    # Plot confusion matrix
    classifier.plot_confusion_matrix(evaluation_report)
    
    # Test predictions
    test_texts = [
        "This is an amazing product!",
        "I don't like this at all.",
        "The quality is excellent and I'm very satisfied."
    ]
    
    predictions = classifier.predict(test_texts)
    
    logger.info("Sample predictions:")
    for pred in predictions:
        logger.info(f"Text: {pred['text']}")
        logger.info(f"Prediction: {pred['predicted_label']} (confidence: {pred['confidence']:.3f})")
        logger.info("---")
    
    logger.info("Pipeline completed successfully!")


if __name__ == "__main__":
    main()