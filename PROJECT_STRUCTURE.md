# BERT Text Classification Project Structure

```
bert-text-classification/
├── README.md                 # Project documentation
├── LICENSE                   # MIT License
├── requirements.txt         # Production dependencies
├── requirements-dev.txt     # Development dependencies
├── 0121.py                  # Main script (command line)
├── app.py                   # Streamlit web interface
├── bert_classifier.py       # Core BERT classifier module
├── .gitignore              # Git ignore rules
├── .github/                # GitHub workflows and templates
│   ├── workflows/
│   │   └── ci.yml          # Continuous Integration
│   └── ISSUE_TEMPLATE/     # Issue templates
├── tests/                  # Test suite
│   ├── __init__.py
│   ├── test_classifier.py
│   └── test_utils.py
├── docs/                   # Documentation
│   ├── conf.py
│   ├── index.rst
│   └── api.rst
├── examples/               # Usage examples
│   ├── basic_usage.py
│   ├── custom_dataset.py
│   └── model_evaluation.py
├── saved_models/           # Saved model directory (created at runtime)
├── logs/                   # Training logs (created at runtime)
└── results/                # Training results (created at runtime)
```

## File Descriptions

### Core Files
- **0121.py**: Main command-line script demonstrating the complete pipeline
- **app.py**: Streamlit web interface for interactive model training and inference
- **bert_classifier.py**: Core BERT classifier implementation with all methods

### Configuration Files
- **requirements.txt**: Production dependencies for running the application
- **requirements-dev.txt**: Development dependencies for testing and code quality
- **README.md**: Comprehensive project documentation
- **LICENSE**: MIT License file

### Development Files
- **tests/**: Unit tests for the classifier and utilities
- **docs/**: Sphinx documentation
- **examples/**: Usage examples and tutorials
- **.github/**: GitHub Actions workflows and issue templates

### Runtime Directories
- **saved_models/**: Directory for saved trained models
- **logs/**: Training logs and metrics
- **results/**: Training results and evaluation reports

## Usage

### Quick Start
```bash
# Install dependencies
pip install -r requirements.txt

# Run web interface
streamlit run app.py

# Run command line version
python 0121.py
```

### Development
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Run linting
black .
flake8 .
```
