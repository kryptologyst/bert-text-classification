"""
Basic usage example for BERT Text Classification
"""

from bert_classifier import BERTTextClassifier, create_mock_database


def main():
    """Demonstrate basic usage of the BERT classifier."""
    print("🤖 BERT Text Classification - Basic Usage Example")
    print("=" * 50)
    
    # Create mock database
    print("\n1. Creating mock database...")
    texts, labels = create_mock_database()
    print(f"   ✅ Created {len(texts)} samples")
    print(f"   📊 Labels: {set(labels)}")
    
    # Initialize classifier
    print("\n2. Initializing BERT classifier...")
    classifier = BERTTextClassifier()
    classifier.load_tokenizer_and_model()
    print("   ✅ Model and tokenizer loaded")
    
    # Prepare dataset
    print("\n3. Preparing dataset...")
    train_dataset, test_dataset = classifier.prepare_dataset(texts, labels)
    print(f"   ✅ Dataset prepared: {len(train_dataset)} train, {len(test_dataset)} test")
    
    # Train model (with minimal epochs for demo)
    print("\n4. Training model...")
    classifier.train(
        train_dataset, 
        test_dataset, 
        num_epochs=1,  # Minimal for demo
        batch_size=8
    )
    print("   ✅ Training completed")
    
    # Evaluate model
    print("\n5. Evaluating model...")
    evaluation_report = classifier.evaluate_model(test_dataset)
    print(f"   📈 Accuracy: {evaluation_report['accuracy']:.3f}")
    print(f"   📈 F1 Score: {evaluation_report['f1_score']:.3f}")
    
    # Make predictions
    print("\n6. Making predictions...")
    test_texts = [
        "This product is absolutely amazing!",
        "I hate this service, it's terrible.",
        "The quality is excellent and I'm very satisfied."
    ]
    
    predictions = classifier.predict(test_texts)
    
    print("\n📝 Prediction Results:")
    for i, pred in enumerate(predictions, 1):
        print(f"   {i}. Text: {pred['text']}")
        print(f"      Prediction: {pred['predicted_label']}")
        print(f"      Confidence: {pred['confidence']:.3f}")
        print()
    
    print("🎉 Example completed successfully!")


if __name__ == "__main__":
    main()
