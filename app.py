"""
Streamlit Web Interface for BERT Text Classification
==================================================

A modern web interface for training and using BERT models for text classification.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os
import tempfile
from pathlib import Path
import logging

# Import our BERT classifier
from bert_classifier import BERTTextClassifier, create_mock_database

# Configure page
st.set_page_config(
    page_title="BERT Text Classification",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .prediction-result {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #1f77b4;
    }
    .negative-prediction {
        background-color: #fdeaea;
        border-left-color: #d62728;
    }
    .positive-prediction {
        background-color: #eafaf1;
        border-left-color: #2ca02c;
    }
</style>
""", unsafe_allow_html=True)

def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<h1 class="main-header">🤖 BERT Text Classification</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["🏠 Home", "📊 Train Model", "🔮 Make Predictions", "📈 Model Evaluation", "📁 Model Management"]
    )
    
    if page == "🏠 Home":
        show_home_page()
    elif page == "📊 Train Model":
        show_training_page()
    elif page == "🔮 Make Predictions":
        show_prediction_page()
    elif page == "📈 Model Evaluation":
        show_evaluation_page()
    elif page == "📁 Model Management":
        show_model_management_page()

def show_home_page():
    """Display the home page."""
    st.markdown("## Welcome to BERT Text Classification! 🎉")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### What is this project?
        
        This is a modern implementation of BERT fine-tuning for text classification using:
        
        - **Latest Transformers Library**: Uses the most recent Hugging Face transformers
        - **Comprehensive Evaluation**: Multiple metrics and visualizations
        - **Web Interface**: Easy-to-use Streamlit interface
        - **Model Management**: Save, load, and manage trained models
        - **Mock Database**: Sample data for demonstration
        
        ### Features:
        ✅ Modern BERT implementation  
        ✅ Comprehensive evaluation metrics  
        ✅ Model saving and loading  
        ✅ Interactive web interface  
        ✅ Confusion matrix visualization  
        ✅ Real-time predictions  
        """)
    
    with col2:
        st.markdown("""
        ### Quick Start:
        
        1. **Train Model**: Go to the "Train Model" page to train a new BERT classifier
        2. **Make Predictions**: Use the "Make Predictions" page to classify new text
        3. **View Results**: Check the "Model Evaluation" page for detailed metrics
        4. **Manage Models**: Use "Model Management" to save/load models
        
        ### Sample Data:
        The application comes with a mock database containing:
        - 20 sample texts (10 positive, 10 negative)
        - Ready-to-use sentiment analysis data
        - Perfect for testing and demonstration
        """)
    
    # Show sample data
    st.markdown("### Sample Data Preview:")
    texts, labels = create_mock_database()
    sample_df = pd.DataFrame({
        'Text': texts[:5],
        'Label': labels[:5]
    })
    st.dataframe(sample_df, use_container_width=True)

def show_training_page():
    """Display the training page."""
    st.markdown("## 📊 Train BERT Model")
    
    # Training parameters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### Training Parameters")
        num_epochs = st.slider("Number of Epochs", 1, 10, 3)
        batch_size = st.selectbox("Batch Size", [8, 16, 32], index=1)
        learning_rate = st.selectbox("Learning Rate", [1e-5, 2e-5, 5e-5], index=1)
    
    with col2:
        st.markdown("### Model Settings")
        model_name = st.selectbox(
            "Pre-trained Model",
            ["bert-base-uncased", "bert-base-cased", "distilbert-base-uncased"],
            index=0
        )
        test_size = st.slider("Test Size", 0.1, 0.4, 0.2)
    
    with col3:
        st.markdown("### Data Source")
        data_source = st.radio(
            "Choose data source:",
            ["Mock Database", "Upload CSV", "Manual Input"]
        )
    
    # Data preparation
    if data_source == "Mock Database":
        texts, labels = create_mock_database()
        st.success(f"✅ Loaded {len(texts)} samples from mock database")
    elif data_source == "Upload CSV":
        uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            if 'text' in df.columns and 'label' in df.columns:
                texts = df['text'].tolist()
                labels = df['label'].tolist()
                st.success(f"✅ Loaded {len(texts)} samples from uploaded file")
            else:
                st.error("❌ CSV must contain 'text' and 'label' columns")
                return
        else:
            st.info("Please upload a CSV file with 'text' and 'label' columns")
            return
    else:  # Manual Input
        st.markdown("### Manual Data Input")
        num_samples = st.number_input("Number of samples", min_value=2, max_value=50, value=4)
        
        texts = []
        labels = []
        
        for i in range(num_samples):
            col1, col2 = st.columns(2)
            with col1:
                text = st.text_area(f"Text {i+1}", key=f"text_{i}")
            with col2:
                label = st.selectbox(f"Label {i+1}", ["positive", "negative"], key=f"label_{i}")
            
            if text.strip():
                texts.append(text.strip())
                labels.append(label)
        
        if len(texts) < 2:
            st.warning("⚠️ Please provide at least 2 samples")
            return
    
    # Show data preview
    if texts:
        st.markdown("### Data Preview")
        preview_df = pd.DataFrame({
            'Text': texts,
            'Label': labels
        })
        st.dataframe(preview_df, use_container_width=True)
        
        # Label distribution
        label_counts = pd.Series(labels).value_counts()
        fig = px.pie(
            values=label_counts.values,
            names=label_counts.index,
            title="Label Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Training button
    if st.button("🚀 Start Training", type="primary"):
        if len(texts) < 2:
            st.error("❌ Need at least 2 samples to train")
            return
        
        # Initialize progress
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Initialize classifier
            status_text.text("Initializing BERT classifier...")
            progress_bar.progress(10)
            
            classifier = BERTTextClassifier(model_name=model_name, num_labels=2)
            classifier.load_tokenizer_and_model()
            
            # Prepare dataset
            status_text.text("Preparing dataset...")
            progress_bar.progress(30)
            
            train_dataset, test_dataset = classifier.prepare_dataset(
                texts, labels, test_size=test_size
            )
            
            # Train model
            status_text.text("Training model...")
            progress_bar.progress(50)
            
            with tempfile.TemporaryDirectory() as temp_dir:
                classifier.train(
                    train_dataset, 
                    test_dataset, 
                    output_dir=temp_dir,
                    num_epochs=num_epochs,
                    batch_size=batch_size,
                    learning_rate=learning_rate
                )
                
                # Evaluate model
                status_text.text("Evaluating model...")
                progress_bar.progress(80)
                
                evaluation_report = classifier.evaluate_model(test_dataset)
                
                # Save model to session state
                st.session_state['trained_model'] = classifier
                st.session_state['evaluation_report'] = evaluation_report
                st.session_state['model_path'] = temp_dir
                
                progress_bar.progress(100)
                status_text.text("✅ Training completed!")
                
                # Show results
                st.success("🎉 Model trained successfully!")
                
                # Display metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Accuracy", f"{evaluation_report['accuracy']:.3f}")
                with col2:
                    st.metric("F1 Score", f"{evaluation_report['f1_score']:.3f}")
                with col3:
                    st.metric("Precision", f"{evaluation_report['precision']:.3f}")
                with col4:
                    st.metric("Recall", f"{evaluation_report['recall']:.3f}")
                
                # Confusion matrix
                st.markdown("### Confusion Matrix")
                cm = evaluation_report['confusion_matrix']
                labels_list = list(evaluation_report['label_mappings'].values())
                
                fig = px.imshow(
                    cm,
                    text_auto=True,
                    aspect="auto",
                    x=labels_list,
                    y=labels_list,
                    title="Confusion Matrix"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        except Exception as e:
            st.error(f"❌ Training failed: {str(e)}")
            logging.error(f"Training error: {str(e)}")

def show_prediction_page():
    """Display the prediction page."""
    st.markdown("## 🔮 Make Predictions")
    
    # Check if model is trained
    if 'trained_model' not in st.session_state:
        st.warning("⚠️ No trained model found. Please train a model first.")
        return
    
    classifier = st.session_state['trained_model']
    
    # Input methods
    input_method = st.radio(
        "Choose input method:",
        ["Single Text", "Multiple Texts", "Upload File"]
    )
    
    if input_method == "Single Text":
        text_input = st.text_area(
            "Enter text to classify:",
            placeholder="Type your text here...",
            height=100
        )
        
        if st.button("🔮 Predict", type="primary"):
            if text_input.strip():
                predictions = classifier.predict([text_input.strip()])
                pred = predictions[0]
                
                # Display result
                confidence = pred['confidence']
                predicted_label = pred['predicted_label']
                
                # Style based on prediction
                if predicted_label == "positive":
                    css_class = "positive-prediction"
                    emoji = "😊"
                else:
                    css_class = "negative-prediction"
                    emoji = "😞"
                
                st.markdown(f"""
                <div class="prediction-result {css_class}">
                    <h3>{emoji} Prediction: {predicted_label.title()}</h3>
                    <p><strong>Confidence:</strong> {confidence:.3f}</p>
                    <p><strong>Text:</strong> {pred['text']}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Show all probabilities
                st.markdown("### All Probabilities")
                prob_df = pd.DataFrame([
                    {"Label": label, "Probability": prob}
                    for label, prob in pred['all_probabilities'].items()
                ])
                
                fig = px.bar(
                    prob_df,
                    x="Label",
                    y="Probability",
                    title="Prediction Probabilities"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Please enter some text to classify.")
    
    elif input_method == "Multiple Texts":
        st.markdown("### Enter multiple texts (one per line):")
        texts_input = st.text_area(
            "Enter texts:",
            placeholder="Text 1\nText 2\nText 3",
            height=200
        )
        
        if st.button("🔮 Predict All", type="primary"):
            if texts_input.strip():
                texts = [text.strip() for text in texts_input.split('\n') if text.strip()]
                predictions = classifier.predict(texts)
                
                # Display results
                results_data = []
                for pred in predictions:
                    results_data.append({
                        'Text': pred['text'],
                        'Prediction': pred['predicted_label'],
                        'Confidence': f"{pred['confidence']:.3f}"
                    })
                
                results_df = pd.DataFrame(results_data)
                st.dataframe(results_df, use_container_width=True)
                
                # Summary
                pred_counts = pd.Series([p['predicted_label'] for p in predictions]).value_counts()
                fig = px.pie(
                    values=pred_counts.values,
                    names=pred_counts.index,
                    title="Prediction Summary"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Please enter some texts to classify.")
    
    else:  # Upload File
        uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            if 'text' in df.columns:
                texts = df['text'].dropna().tolist()
                
                if st.button("🔮 Predict Uploaded Texts", type="primary"):
                    predictions = classifier.predict(texts)
                    
                    # Add predictions to dataframe
                    df['prediction'] = [p['predicted_label'] for p in predictions]
                    df['confidence'] = [f"{p['confidence']:.3f}" for p in predictions]
                    
                    st.dataframe(df, use_container_width=True)
                    
                    # Download results
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results",
                        data=csv,
                        file_name="predictions.csv",
                        mime="text/csv"
                    )
            else:
                st.error("❌ CSV must contain a 'text' column")

def show_evaluation_page():
    """Display the evaluation page."""
    st.markdown("## 📈 Model Evaluation")
    
    if 'evaluation_report' not in st.session_state:
        st.warning("⚠️ No evaluation report found. Please train a model first.")
        return
    
    report = st.session_state['evaluation_report']
    
    # Metrics overview
    st.markdown("### Performance Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Accuracy", f"{report['accuracy']:.3f}")
    with col2:
        st.metric("F1 Score", f"{report['f1_score']:.3f}")
    with col3:
        st.metric("Precision", f"{report['precision']:.3f}")
    with col4:
        st.metric("Recall", f"{report['recall']:.3f}")
    
    # Detailed metrics
    st.markdown("### Detailed Analysis")
    
    metrics_data = {
        'Metric': ['Accuracy', 'F1 Score', 'Precision', 'Recall'],
        'Value': [report['accuracy'], report['f1_score'], report['precision'], report['recall']]
    }
    
    fig = px.bar(
        metrics_data,
        x='Metric',
        y='Value',
        title="Model Performance Metrics",
        color='Value',
        color_continuous_scale='Blues'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Confusion matrix
    st.markdown("### Confusion Matrix")
    cm = report['confusion_matrix']
    labels_list = list(report['label_mappings'].values())
    
    fig = px.imshow(
        cm,
        text_auto=True,
        aspect="auto",
        x=labels_list,
        y=labels_list,
        title="Confusion Matrix",
        color_continuous_scale='Blues'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Model info
    st.markdown("### Model Information")
    st.json({
        "Labels": report['label_mappings'],
        "Total Samples": sum(sum(cm)),
        "Correct Predictions": sum(cm[i][i] for i in range(len(cm))),
        "Incorrect Predictions": sum(sum(cm)) - sum(cm[i][i] for i in range(len(cm)))
    })

def show_model_management_page():
    """Display the model management page."""
    st.markdown("## 📁 Model Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Save Model")
        if 'trained_model' in st.session_state:
            model_name = st.text_input("Model name:", value="my_bert_model")
            
            if st.button("💾 Save Model"):
                try:
                    save_path = f"./saved_models/{model_name}"
                    os.makedirs(save_path, exist_ok=True)
                    
                    classifier = st.session_state['trained_model']
                    classifier.save_model(save_path)
                    
                    st.success(f"✅ Model saved to {save_path}")
                except Exception as e:
                    st.error(f"❌ Failed to save model: {str(e)}")
        else:
            st.info("No trained model to save. Train a model first.")
    
    with col2:
        st.markdown("### Load Model")
        
        # List available models
        models_dir = Path("./saved_models")
        if models_dir.exists():
            available_models = [d.name for d in models_dir.iterdir() if d.is_dir()]
            
            if available_models:
                selected_model = st.selectbox("Select model to load:", available_models)
                
                if st.button("📂 Load Model"):
                    try:
                        model_path = f"./saved_models/{selected_model}"
                        classifier = BERTTextClassifier()
                        classifier.load_model(model_path)
                        
                        st.session_state['trained_model'] = classifier
                        st.success(f"✅ Model loaded from {model_path}")
                    except Exception as e:
                        st.error(f"❌ Failed to load model: {str(e)}")
            else:
                st.info("No saved models found.")
        else:
            st.info("No saved models directory found.")
    
    # Model status
    st.markdown("### Current Model Status")
    if 'trained_model' in st.session_state:
        classifier = st.session_state['trained_model']
        st.success("✅ Model is loaded and ready for predictions")
        
        # Show model info
        st.json({
            "Model Name": classifier.model_name,
            "Number of Labels": classifier.num_labels,
            "Label Mappings": classifier.label2id
        })
    else:
        st.warning("⚠️ No model loaded")

if __name__ == "__main__":
    main()
