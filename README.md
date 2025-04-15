# Pharmaceutical Compliance Analysis System

A system for analyzing pharmaceutical promotional materials and generating brand-specific compliance rules that extend beyond baseline FDA guidelines.

## Overview

This system automates the extraction of text from PDF documents, checks content against FDA regulations, identifies unique brand preferences via semantic analysis, and generates structured compliance rules. The system uses a combination of advanced NLP techniques, including:

- Layout-aware PDF parsing using Unstructured
- Semantic embeddings with Sentence Transformers
- Unsupervised clustering with FAISS
- LLM-assisted rule generation with GPT-3.5/GPT-4
- Simulated FDA compliance checking

## Features

- PDF document processing with layout preservation
- Semantic chunking and embedding generation
- Unsupervised content clustering
- Pattern mining and analysis
- LLM-assisted rule generation
- FDA compliance verification
- Comprehensive evaluation and reporting

## Project Structure

```
pharma_compliance/
├── docs/                    # Documentation files
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
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Configure the system:
   - Copy `config.yaml.example` to `config.yaml`
   - Set your OpenAI API key in the configuration
   - Adjust other parameters as needed

## Usage

1. Place PDF promotional materials in your working directory
2. Run the analysis:

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

## Input/Output

### Input

- PDF documents containing promotional materials
- Configuration file with model settings and parameters

### Output

The system produces a structured JSON output containing:

- Processed documents with chunks and metadata
- Content clusters and patterns
- Generated compliance rules
- FDA compliance results
- System evaluation metrics

## Requirements

- Python 3.9+
- OpenAI API key (for rule generation)
- See requirements.txt for package dependencies

## Configuration

The system is highly configurable through `config.yaml`:

- Model settings (embedding, tokenizer, sentiment)
- Document processing parameters
- Pattern analysis thresholds
- Rule generation settings
- FDA compliance parameters

## License

[Your License Here]

## Contributing

[Your Contributing Guidelines Here]
