"""
BERT Text Classifier Module
==========================

A modular implementation of BERT-based text classification.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings("ignore")

import torch
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from transformers import (
    BertTokenizerFast, 
    BertForSequenceClassification, 
    Trainer, 
    TrainingArguments,
    EarlyStoppingCallback
)
from datasets import Dataset, load_dataset
import evaluate

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BERTTextClassifier:
    """
    A comprehensive BERT-based text classifier with modern features.
    """
    
    def __init__(self, model_name: str = "bert-base-uncased", num_labels: int = 2):
        """
        Initialize the BERT text classifier.
        
        Args:
            model_name: Pre-trained BERT model name
            num_labels: Number of classification labels
        """
        self.model_name = model_name
        self.num_labels = num_labels
        self.tokenizer = None
        self.model = None
        self.trainer = None
        self.label2id = {}
        self.id2label = {}
        
    def load_tokenizer_and_model(self):
        """Load tokenizer and model."""
        logger.info(f"Loading tokenizer and model: {self.model_name}")
        self.tokenizer = BertTokenizerFast.from_pretrained(self.model_name)
        self.model = BertForSequenceClassification.from_pretrained(
            self.model_name, 
            num_labels=self.num_labels
        )
        
    def create_label_mappings(self, labels: List[str]):
        """Create label to ID mappings."""
        unique_labels = sorted(list(set(labels)))
        self.label2id = {label: idx for idx, label in enumerate(unique_labels)}
        self.id2label = {idx: label for label, idx in self.label2id.items()}
        logger.info(f"Label mappings created: {self.label2id}")
        
    def tokenize_function(self, examples):
        """Tokenize text examples."""
        return self.tokenizer(
            examples["text"], 
            padding=True, 
            truncation=True, 
            max_length=512
        )
        
    def prepare_dataset(self, texts: List[str], labels: List[str], 
                       test_size: float = 0.2) -> Tuple[Dataset, Dataset]:
        """
        Prepare dataset for training.
        
        Args:
            texts: List of text samples
            labels: List of corresponding labels
            test_size: Fraction of data to use for testing
            
        Returns:
            Tuple of (train_dataset, test_dataset)
        """
        logger.info(f"Preparing dataset with {len(texts)} samples")
        
        # Create label mappings
        self.create_label_mappings(labels)
        
        # Convert labels to IDs
        label_ids = [self.label2id[label] for label in labels]
        
        # Create dataset
        dataset = Dataset.from_dict({
            "text": texts,
            "label": label_ids
        })
        
        # Split dataset
        split_dataset = dataset.train_test_split(test_size=test_size, seed=42)
        
        # Tokenize
        train_dataset = split_dataset["train"].map(
            self.tokenize_function, 
            batched=True,
            desc="Tokenizing train data"
        )
        test_dataset = split_dataset["test"].map(
            self.tokenize_function, 
            batched=True,
            desc="Tokenizing test data"
        )
        
        # Set format
        train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
        test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
        
        logger.info(f"Dataset prepared: {len(train_dataset)} train, {len(test_dataset)} test")
        return train_dataset, test_dataset
        
    def compute_metrics(self, eval_pred):
        """Compute evaluation metrics."""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
        accuracy = accuracy_score(labels, predictions)
        
        return {
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }
        
    def train(self, train_dataset: Dataset, test_dataset: Dataset, 
              output_dir: str = "./bert_model", num_epochs: int = 3,
              batch_size: int = 16, learning_rate: float = 2e-5):
        """
        Train the BERT model.
        
        Args:
            train_dataset: Training dataset
            test_dataset: Test dataset
            output_dir: Directory to save the model
            num_epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate
        """
        logger.info("Starting training...")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir=f"{output_dir}/logs",
            logging_steps=100,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            learning_rate=learning_rate,
            save_total_limit=2,
            report_to=None,  # Disable wandb/tensorboard
        )
        
        # Initialize trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
        )
        
        # Train
        self.trainer.train()
        
        # Save model and tokenizer
        self.save_model(output_dir)
        
        logger.info("Training completed!")
        
    def save_model(self, output_dir: str):
        """Save the trained model and tokenizer."""
        logger.info(f"Saving model to {output_dir}")
        
        # Save model and tokenizer
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        # Save label mappings
        with open(f"{output_dir}/label_mappings.json", "w") as f:
            json.dump({
                "label2id": self.label2id,
                "id2label": self.id2label
            }, f, indent=2)
            
    def load_model(self, model_path: str):
        """Load a trained model."""
        logger.info(f"Loading model from {model_path}")
        
        self.model = BertForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = BertTokenizerFast.from_pretrained(model_path)
        
        # Load label mappings
        with open(f"{model_path}/label_mappings.json", "r") as f:
            mappings = json.load(f)
            self.label2id = mappings["label2id"]
            self.id2label = {int(k): v for k, v in mappings["id2label"].items()}
            
    def predict(self, texts: List[str]) -> List[Dict]:
        """
        Make predictions on new texts.
        
        Args:
            texts: List of texts to classify
            
        Returns:
            List of prediction dictionaries
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not loaded. Call load_model() first.")
            
        # Tokenize
        inputs = self.tokenizer(
            texts, 
            padding=True, 
            truncation=True, 
            max_length=512,
            return_tensors="pt"
        )
        
        # Predict
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_ids = torch.argmax(predictions, dim=-1)
            
        # Format results
        results = []
        for i, text in enumerate(texts):
            pred_id = predicted_ids[i].item()
            confidence = predictions[i][pred_id].item()
            
            results.append({
                "text": text,
                "predicted_label": self.id2label[pred_id],
                "confidence": confidence,
                "all_probabilities": {
                    self.id2label[j]: predictions[i][j].item() 
                    for j in range(len(self.id2label))
                }
            })
            
        return results
        
    def evaluate_model(self, test_dataset: Dataset) -> Dict:
        """Evaluate the model and return comprehensive metrics."""
        logger.info("Evaluating model...")
        
        # Get predictions
        predictions = self.trainer.predict(test_dataset)
        pred_ids = np.argmax(predictions.predictions, axis=1)
        true_ids = predictions.label_ids
        
        # Calculate metrics
        accuracy = accuracy_score(true_ids, pred_ids)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_ids, pred_ids, average='weighted'
        )
        
        # Confusion matrix
        cm = confusion_matrix(true_ids, pred_ids)
        
        # Create evaluation report
        evaluation_report = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "confusion_matrix": cm.tolist(),
            "label_mappings": self.id2label
        }
        
        # Save evaluation report
        with open("evaluation_report.json", "w") as f:
            json.dump(evaluation_report, f, indent=2)
            
        logger.info(f"Evaluation completed. Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
        return evaluation_report
        
    def plot_confusion_matrix(self, evaluation_report: Dict, save_path: str = "confusion_matrix.png"):
        """Plot and save confusion matrix."""
        cm = np.array(evaluation_report["confusion_matrix"])
        labels = list(evaluation_report["label_mappings"].values())
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=labels, yticklabels=labels)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Confusion matrix saved to {save_path}")


def create_mock_database() -> Tuple[List[str], List[str]]:
    """
    Create a mock database with sample text classification data.
    
    Returns:
        Tuple of (texts, labels) for sentiment analysis
    """
    logger.info("Creating mock database...")
    
    # Sample positive and negative reviews
    positive_texts = [
        "This movie is absolutely fantastic! The acting was superb and the plot was engaging.",
        "I love this product! It works perfectly and exceeded my expectations.",
        "Amazing experience! The service was excellent and the staff was very friendly.",
        "Outstanding quality! I would definitely recommend this to anyone.",
        "Perfect! Everything was exactly as described and arrived on time.",
        "Excellent product! Great value for money and fast delivery.",
        "Wonderful experience! The customer service was top-notch.",
        "Fantastic! This exceeded all my expectations and I'm very satisfied.",
        "Brilliant! The quality is outstanding and the design is beautiful.",
        "Superb! I couldn't be happier with this purchase."
    ]
    
    negative_texts = [
        "This movie was terrible! The acting was awful and the plot made no sense.",
        "I hate this product! It doesn't work at all and was a complete waste of money.",
        "Horrible experience! The service was poor and the staff was rude.",
        "Terrible quality! I would never recommend this to anyone.",
        "Awful! Nothing was as described and it arrived late.",
        "Poor product! Bad value for money and slow delivery.",
        "Disappointing experience! The customer service was terrible.",
        "Terrible! This didn't meet any of my expectations and I'm very unsatisfied.",
        "Bad! The quality is poor and the design is ugly.",
        "Disappointing! I couldn't be more unhappy with this purchase."
    ]
    
    # Combine and shuffle
    all_texts = positive_texts + negative_texts
    all_labels = ["positive"] * len(positive_texts) + ["negative"] * len(negative_texts)
    
    # Shuffle data
    import random
    random.seed(42)
    combined = list(zip(all_texts, all_labels))
    random.shuffle(combined)
    texts, labels = zip(*combined)
    
    logger.info(f"Mock database created with {len(texts)} samples")
    return list(texts), list(labels)
