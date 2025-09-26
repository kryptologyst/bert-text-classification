"""
Custom dataset example for BERT Text Classification
"""

import pandas as pd
from bert_classifier import BERTTextClassifier


def create_custom_dataset():
    """Create a custom dataset for demonstration."""
    # Sample data for product review classification
    data = {
        'text': [
            "This laptop is fantastic! Great performance and battery life.",
            "The camera quality is poor and the battery drains quickly.",
            "Amazing customer service! They resolved my issue immediately.",
            "Terrible product! Broke after just one week of use.",
            "Excellent value for money. Highly recommended!",
            "The delivery was late and the packaging was damaged.",
            "Outstanding quality! Exceeded all my expectations.",
            "Waste of money. The product doesn't work as advertised.",
            "Perfect! Everything was exactly as described.",
            "Disappointing experience. The quality is very poor.",
            "Great product! Fast shipping and excellent packaging.",
            "The worst purchase I've ever made. Complete garbage.",
            "Love it! The design is beautiful and functional.",
            "Horrible customer support. They were completely unhelpful.",
            "Fantastic! This product changed my life for the better.",
            "Bad quality materials. Feels cheap and flimsy.",
            "Wonderful experience! The team was very professional.",
            "Terrible! The product arrived broken and unusable.",
            "Brilliant! The innovation and quality are top-notch.",
            "Poor value. The price doesn't match the quality."
        ],
        'label': [
            'positive', 'negative', 'positive', 'negative', 'positive',
            'negative', 'positive', 'negative', 'positive', 'negative',
            'positive', 'negative', 'positive', 'negative', 'positive',
            'negative', 'positive', 'negative', 'positive', 'negative'
        ]
    }
    
    return pd.DataFrame(data)


def main():
    """Demonstrate custom dataset usage."""
    print("🤖 BERT Text Classification - Custom Dataset Example")
    print("=" * 55)
    
    # Create custom dataset
    print("\n1. Creating custom dataset...")
    df = create_custom_dataset()
    print(f"   ✅ Created dataset with {len(df)} samples")
    print(f"   📊 Label distribution:")
    print(df['label'].value_counts().to_string())
    
    # Extract texts and labels
    texts = df['text'].tolist()
    labels = df['label'].tolist()
    
    # Initialize classifier
    print("\n2. Initializing BERT classifier...")
    classifier = BERTTextClassifier()
    classifier.load_tokenizer_and_model()
    print("   ✅ Model and tokenizer loaded")
    
    # Prepare dataset
    print("\n3. Preparing dataset...")
    train_dataset, test_dataset = classifier.prepare_dataset(texts, labels, test_size=0.3)
    print(f"   ✅ Dataset prepared: {len(train_dataset)} train, {len(test_dataset)} test")
    
    # Train model
    print("\n4. Training model...")
    classifier.train(
        train_dataset, 
        test_dataset, 
        num_epochs=2,  # More epochs for better performance
        batch_size=8,
        learning_rate=2e-5
    )
    print("   ✅ Training completed")
    
    # Evaluate model
    print("\n5. Evaluating model...")
    evaluation_report = classifier.evaluate_model(test_dataset)
    print(f"   📈 Accuracy: {evaluation_report['accuracy']:.3f}")
    print(f"   📈 Precision: {evaluation_report['precision']:.3f}")
    print(f"   📈 Recall: {evaluation_report['recall']:.3f}")
    print(f"   📈 F1 Score: {evaluation_report['f1_score']:.3f}")
    
    # Save model
    print("\n6. Saving model...")
    classifier.save_model("./custom_model")
    print("   ✅ Model saved to ./custom_model")
    
    # Test predictions
    print("\n7. Testing predictions...")
    test_texts = [
        "This is the best product I've ever used!",
        "Completely disappointed with this purchase.",
        "Good quality but could be better.",
        "Absolutely love it! Five stars!"
    ]
    
    predictions = classifier.predict(test_texts)
    
    print("\n📝 Prediction Results:")
    for i, pred in enumerate(predictions, 1):
        print(f"   {i}. Text: {pred['text']}")
        print(f"      Prediction: {pred['predicted_label']}")
        print(f"      Confidence: {pred['confidence']:.3f}")
        print(f"      Probabilities: {pred['all_probabilities']}")
        print()
    
    print("🎉 Custom dataset example completed successfully!")


if __name__ == "__main__":
    main()
