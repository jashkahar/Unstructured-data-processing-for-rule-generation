# Pharmaceutical Compliance Analysis System – Product Requirements Document

## 1. Introduction

The Pharmaceutical Compliance Analysis System is designed to analyze promotional materials for a pharmaceutical brand and derive brand-specific compliance rules that extend beyond standard FDA guidelines. In the initial phase, the system will focus solely on processing PDF documents. In future iterations, image and video parsing functionality may be added.

## 2. Objectives

- **Automate PDF Ingestion:** Extract text, structure, and metadata from PDF promotional materials using libraries such as PyPDF2 or Unstructured.io.
- **Baseline Compliance Verification:** Evaluate the PDF content against predefined FDA regulatory guidelines.
- **Brand-Specific Pattern Identification:** Analyze the extracted PDF content to detect patterns in language, tone, and layout that reflect the brand’s unique approach.
- **Rule Generation:** Merge baseline FDA rules with the identified brand-specific patterns to produce detailed, granular compliance rules.
- **Validation:** Apply the generated rules to sample PDF promotional materials and output a detailed compliance report.
- **Output Structured Data:** Produce a machine-readable JSON file that encapsulates each derived rule along with detailed descriptions, supporting reasoning, and evaluation methods.

## 3. Scope

### In Scope
- Parsing PDF promotional materials (text extraction and metadata).
- Evaluating parsed PDF content against standard FDA guidelines (using the FDA Fact Checker).
- Extracting brand-specific patterns from the PDF content via NLP/LLM analysis.
- Merging baseline FDA rules with the extracted brand patterns to generate refined compliance rules.
- Testing these rules against new PDF-based promotional content.
- Outputting a structured JSON file containing the generated rules and a compliance report.

### Out of Scope (for now)
- Parsing images via OCR.
- Video content transcription/extraction.
- Persistent storage across processing steps (the system is a direct pipeline).

## 4. Functional Requirements

- **PDF Document Processing**
  - Extract text, headings, and metadata from PDFs.
  - Output a structured JSON/dictionary with document ID, segmented content, and metadata.
  
- **FDA Fact Checker**
  - Evaluate the extracted text against FDA guidelines.
  - Output a JSON structure listing individual FDA guideline compliance results (e.g., compliant/non-compliant).

- **Pattern Analysis**
  - Analyze the aggregated text from PDFs to identify:
    - **Keyword-based Patterns:** Frequency of specific terms (e.g., “real relief,” avoidance of “miracle cure”).
    - **Semantic Tone:** Overall sentiment and language style (e.g., an emphasis on risk information).
    - **Aesthetic/Formatting Cues:** Detect consistent formatting or layout styles present in the PDFs.
  - Output a structured summary of these patterns.

- **Evaluation**
  - Merge baseline FDA rules and brand-specific patterns.
  - Identify where brand practices differ from FDA guidelines.
  - Output combined evaluation insights.

- **Rule Generation**
  - Use the evaluation insights to generate brand-specific compliance rules.
  - Each rule should include a description, the underlying reasoning, and an evaluation method.
  - Produce a detailed JSON file with these rules.

- **Rule Testing**
  - Apply the generated rules to new PDF promotional content.
  - Generate a compliance report indicating whether each rule is met or violated.

## 5. Non-Functional Requirements

- **Performance:**  
  Process PDF documents and complete the entire compliance pipeline in near-real time.
- **Modularity:**  
  Each component (parsing, FDA checking, pattern analysis, rule generation, and evaluation) must be developed and tested as an independent module.
- **Maintainability:**  
  Code should adhere to Python best practices (PEP8) and be well-documented, to facilitate future enhancements (e.g., adding image/video parsing).
- **Scalability:**  
  The pipeline is designed to work with PDF data now and can be extended to include additional formats later.
- **Usability:**  
  The final JSON output must be structured, clearly annotating each rule with its description, rationale, and evaluation method.

## 6. Acceptance Criteria

- The system correctly extracts text and metadata from PDFs.
- The FDA Fact Checker returns baseline compliance rules for each PDF (via mock data based on FDA guidelines).
- The Pattern Analysis Module identifies key brand-specific patterns from PDF content.
- The Rule Generation module produces a detailed JSON file with refined rules that incorporate both FDA baseline and brand-specific insights.
- A compliance report is generated after testing new PDF content against the generated rules.
- The system demonstrates modular functionality with well-defined unit tests for PDF parsing, analysis, and rule generation.

## 7. Assumptions

- The provided PDF promotional materials are of adequate quality and representative of the brand’s approved content.
- The FDA Fact Checker module is considered reliable and returns structured baseline rules.
- The NLP/LLM used for pattern analysis is capable of detecting both keyword-based and semantic patterns from text.
- The merged evaluation process can reconcile any differences between FDA guidelines and observed brand practices into granular rules.
- Future enhancements (image/video parsing) will follow a similar modular approach.
