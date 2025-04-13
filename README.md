# Pharmaceutical Compliance Analysis System

A system for analyzing pharmaceutical promotional materials and generating brand-specific compliance rules that extend beyond baseline FDA guidelines.

## Overview

This system automates the extraction of text from multi-format documents (PDFs, images, and videos), checks content against FDA regulations, identifies unique brand preferences via natural language analysis, and generates structured compliance rules.

## Features

- Multi-format document processing (PDFs, images, videos)
- FDA compliance verification
- Brand-specific pattern identification
- Automated rule generation
- Compliance testing and reporting

## Project Structure

```
pharma_compliance/
├── docs/                    # Documentation files
├── promotional_materials/   # Sample promotional materials
├── src/                    # Source code
│   ├── document_processing/ # Document parsing modules
│   ├── fda_checker/        # FDA compliance verification
│   ├── pattern_analysis/   # Brand pattern identification
│   ├── rule_generation/    # Rule creation and testing
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

## Usage

1. Place promotional materials in the `promotional_materials/` directory
2. Configure settings in `config.yaml`
3. Run the analysis:
   ```bash
   python src/main.py
   ```

## Requirements

- Python 3.9+
- See requirements.txt for package dependencies

## License

[Your License Here]

## Contributing

[Your Contributing Guidelines Here]
