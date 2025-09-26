"""
Setup script for BERT Text Classification package
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="bert-text-classification",
    version="1.0.0",
    author="AI Projects",
    author_email="ai.projects@example.com",
    description="A modern implementation of BERT fine-tuning for text classification",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/bert-text-classification",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
        "ui": [
            "streamlit>=1.25.0",
            "plotly>=5.15.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "bert-classify=0121:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.md", "*.txt", "*.yml"],
    },
    keywords="bert, text-classification, nlp, machine-learning, transformers, huggingface",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/bert-text-classification/issues",
        "Source": "https://github.com/yourusername/bert-text-classification",
        "Documentation": "https://bert-text-classification.readthedocs.io/",
    },
)
