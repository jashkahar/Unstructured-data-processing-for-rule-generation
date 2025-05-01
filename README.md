# Pharmaceutical Compliance Analysis System

A system for analyzing pharmaceutical promotional materials and generating brand-specific compliance rules that extend beyond baseline FDA guidelines.

## Overview

This system automates the extraction of text from PDF documents, checks content against FDA regulations, identifies unique brand preferences through semantic analysis, and generates structured compliance rules using advanced NLP techniques.

## Components

### 1. Document Processing

- **Document Processor**: Handles various file formats with specialized parsers
- **PDF Parser**: Extracts text with layout preservation, detects tables and sections
- **Image Parser**: Performs OCR on image-based content
- **Chunking Strategy**: Segments documents into semantic chunks for analysis

### 2. Pattern Analysis

- **Pattern Analyzer**: Identifies linguistic patterns, tones, and aesthetic preferences
- **Feature Extractor**: Analyzes sentiment, benefit-risk balance, and promotional tone
- **Embedding Manager**: Generates semantic embeddings using transformer models
- **Pattern Clusterer**: Uses unsupervised learning to discover content patterns

### 3. Rule Generation

- **Rule Generator**: Transforms patterns into explicit compliance rules
- **FDA Checker**: Verifies compliance against FDA regulations
- **Rule Validation**: Ensures rule quality and applicability

### 4. Storage & Evaluation

- **Document Store**: Maintains document repository with multi-dimensional indexing
- **Output System**: Manages persistence of results in JSON format
- **Evaluation Controller**: Assesses system performance and rule effectiveness

## Usage

```python
from pharma_compliance import ComplianceAnalyzer

# Initialize the analyzer
analyzer = ComplianceAnalyzer("config.yaml")

# Process a document
results = analyzer.process_document("marketing_material.pdf")

# Access results
documents = results["documents"]
clusters = results["clusters"]
rules = results["rules"]
fda_results = results["fda_compliance"]
evaluation = results["evaluation"]
```

## Requirements

- Python 3.9+
- OpenAI API key (for rule generation)
- See requirements.txt for package dependencies

## Project Structure

```
pharma_compliance/
├── src/                    # Source code
│   ├── document_processing/ # PDF parsing and chunking
│   ├── pattern_analysis/   # Embedding and clustering
│   ├── rule_generation/    # Rule synthesis
│   ├── fda_checker/        # FDA compliance verification
│   ├── evaluation/         # System evaluation
│   └── utils/              # Utility functions
├── tests/                  # Test files
├── config.yaml             # Configuration file
└── requirements.txt        # Project dependencies
```

## Installation

1. Clone the repository
2. Create a virtual environment
3. Install dependencies: `pip install -r requirements.txt`
4. Configure API keys in config.yaml
