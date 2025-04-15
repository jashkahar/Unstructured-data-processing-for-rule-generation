# Pharmaceutical Compliance Analysis System – Design Document

## 1. Introduction

The Pharmaceutical Compliance Analysis System is designed to automate the evaluation of promotional materials for compliance with FDA guidelines while capturing brand-specific preferences. The system parses PDF promotional content, checks the content against FDA regulations, and uses pattern analysis to extract both textual and semantic signals that represent a brand's unique approach. Finally, it merges these insights to generate detailed, granular rules presented as a machine-readable JSON output. These rules can be used to assess new promotional material, ensuring both regulatory compliance and consistency with the brand's identity.

---

## 2. System Architecture Overview

The system is built as a linear pipeline that processes data through several key components:

1. **Document Processing Layer:**

   - Extracts text and metadata from PDFs using Unstructured
   - Preserves layout information and document structure
   - Implements adaptive chunking with overlap

2. **Pattern Analysis Module:**

   - Generates embeddings using Sentence Transformers
   - Performs clustering using FAISS
   - Identifies patterns in language, tone, and structure

3. **Rule Generation Module:**

   - Uses GPT-3.5/GPT-4 to synthesize rules from patterns
   - Generates structured JSON output
   - Includes rule validation and testing

4. **FDA Compliance Checker:**

   - Simulates FDA compliance checking
   - Validates content against standard guidelines
   - Generates compliance reports

5. **Evaluation Component:**

   - Evaluates system performance
   - Provides comprehensive metrics
   - Validates rule effectiveness

6. **Main Controller/Orchestrator:**
   - Coordinates the pipeline flow
   - Manages component interactions
   - Handles error recovery

### Data Flow Diagram (Text Representation)

```plaintext
[PDF Documents]
        │
        ▼
[Document Processing Layer]
  ├─> pdf_parser.py (Extracts text, metadata)
  └─> chunking.py (Adaptive chunking)
        │
        ▼
[Pattern Analysis Module]
  ├─> embeddings.py (Generate embeddings)
  └─> clustering.py (FAISS clustering)
        │
        ▼
[Rule Generation Module]
  ├─> generator.py (LLM rule synthesis)
  └─> tester.py (Rule validation)
        │
        ▼
[FDA Compliance Checker]
  └─> checker.py (Compliance verification)
        │
        ▼
[Evaluation Component]
  └─> evaluator.py (System evaluation)
        │
        ▼
[Main Controller]
  └─> main_controller.py (Pipeline orchestration)
```

---

## 3. Component-Level Design

### 3.1 Document Processing Layer

- **pdf_parser.py:**

  - Uses Unstructured for PDF parsing
  - Preserves layout and structure
  - Extracts metadata and sections
  - Outputs structured JSON:
    ```json
    {
      "document_id": "DOC_001",
      "sections": [
        {
          "title": "Indications",
          "content": "...",
          "page_number": 1
        }
      ],
      "metadata": {
        "format": "PDF",
        "page_count": 10
      }
    }
    ```

- **chunking.py:**
  - Implements adaptive chunking
  - Preserves semantic coherence
  - Maintains overlap between chunks
  - Outputs chunks with metadata:
    ```json
    {
      "chunk_id": "CHK_001",
      "section_title": "Indications",
      "content": "...",
      "page_number": 1,
      "token_count": 150,
      "section_type": "body"
    }
    ```

### 3.2 Pattern Analysis Module

- **embeddings.py:**

  - Uses Sentence Transformers
  - Generates 384-dimensional vectors
  - Maintains FAISS index
  - Outputs embeddings array

- **clustering.py:**
  - Implements K-Means clustering
  - Uses silhouette scores for optimal k
  - Analyzes cluster characteristics
  - Outputs cluster information:
    ```json
    {
      "cluster_id": "CLS_001",
      "size": 5,
      "characteristics": {
        "most_common_section": "Safety",
        "top_words": { "risk": 10, "warning": 5 }
      },
      "representative": {
        "content": "..."
      }
    }
    ```

### 3.3 Rule Generation Module

- **generator.py:**

  - Uses GPT-3.5/GPT-4 for rule synthesis
  - Generates structured rules
  - Outputs JSON rules:
    ```json
    {
      "title": "Balance Benefits and Risks",
      "description": "Promotional content must present a fair balance...",
      "category": "Balance",
      "severity": "HIGH",
      "examples": ["Content with only benefits", "Minimizing risk information"],
      "rationale": "FDA requires fair balance in promotional materials"
    }
    ```

- **tester.py:**
  - Validates rules against content
  - Tests rule effectiveness
  - Generates test reports

### 3.4 FDA Compliance Checker

- **checker.py:**
  - Simulates FDA compliance checking
  - Validates against guidelines
  - Outputs compliance results:
    ```json
    {
      "summary": {
        "total_documents": 1,
        "violations": 2,
        "warnings": 1
      },
      "violations": [
        {
          "document_id": "DOC_001",
          "section": "Claims",
          "rule": "FAIR_BALANCE",
          "description": "Content mentions benefits without risks",
          "severity": "HIGH"
        }
      ]
    }
    ```

### 3.5 Evaluation Component

- **evaluator.py:**
  - Evaluates system performance
  - Generates comprehensive metrics
  - Outputs evaluation results:
    ```json
    {
      "document_processing": {
        "total_documents": 1,
        "total_chunks": 15,
        "avg_chunks_per_doc": 15
      },
      "pattern_analysis": {
        "total_clusters": 3,
        "avg_cluster_size": 5
      },
      "rule_generation": {
        "total_rules": 5,
        "categories": { "Balance": 2, "Claims": 3 }
      },
      "fda_compliance": {
        "total_violations": 2,
        "total_warnings": 1
      }
    }
    ```

---

## 4. Technology Stack

- **Programming Language:** Python 3.9+
- **PDF Processing:** Unstructured
- **NLP/Embeddings:** Sentence Transformers (all-MiniLM-L6-v2)
- **Vector Storage:** FAISS
- **Rule Generation:** OpenAI GPT-3.5/GPT-4
- **Configuration:** PyYAML
- **Testing:** pytest
- **Logging:** loguru

---

## 5. Non-Functional Considerations

- **Modularity:**  
  Each component operates independently with clear interfaces.

- **Performance:**  
  Optimized for PDF processing with layout preservation.

- **Scalability:**  
  Efficient vector operations with FAISS.

- **Maintainability:**  
  Well-documented code with clear component separation.

- **Extensibility:**  
  Easy to add new features or modify existing ones.

---

## 6. Future Enhancements

- **Enhanced Pattern Analysis:**  
  Add more sophisticated pattern detection algorithms.

- **Real FDA API Integration:**  
  Replace simulated FDA checker with real API.

- **Multi-format Support:**  
  Add support for images and videos.

- **User Interface:**  
  Develop web interface for rule management.

---

## 7. Conclusion

This design document details a modular, pipeline-based system that processes pharmaceutical promotional materials, checks them against FDA guidelines, and extracts brand-specific patterns. The system generates detailed compliance rules in JSON format, which can be used to verify future content. The design emphasizes real-time processing, modularity, and maintainability, ensuring that the system can be extended as needed.

---

This detailed design document encapsulates our thought process and clarifications, ensuring that our assumptions and approaches are well documented before we begin coding the system.
