# Pharmaceutical Compliance Analysis System - Design Document

## 1. System Architecture Overview

The system is designed as a linear pipeline with direct data flows, without persistent storage between components. Raw promotional materials are parsed and immediately passed through each module in sequence. The primary components include:

1. **Document Processing Layer:**  
   - Parses PDFs, images (via OCR), and videos (via transcription) to extract textual content and metadata.

2. **FDA Fact Checker:**  
   - Receives parsed content directly from document processing.
   - Checks content against FDA regulatory requirements and outputs baseline compliance rules.

3. **Pattern Analysis Module:**  
   - Accepts aggregated parsed content (from document processing) and baseline results.
   - Uses an LLM/NLP to identify recurring brand-specific language and design patterns.
   - Outputs structured findings describing these patterns.

4. **Evaluation Component:**  
   - Combines baseline FDA rules and brand-specific patterns.
   - Determines areas where the brand’s standards exceed baseline requirements.
   - Forwards the combined insights for final rule generation.

5. **Rule Generation Module:**  
   - Takes evaluation insights and passes them to an LLM to create brand-specific compliance rules.
   - Each rule includes a description, reasoning (from the analysis), and a testing methodology.
   - Outputs a structured JSON file.

6. **Rule Testing Component:** (Optional)  
   - Applies the generated rules to new promotional materials.
   - Produces a compliance report indicating pass/fail status for each rule.

7. **Main Controller/Orchestrator:**  
   - Manages the overall workflow, directing data sequentially from one module to the next.
   - Triggers the rule testing phase and collects the final JSON output.

## 2. Data Flow Diagram

```plaintext
[Promotional Materials]
        │
        ▼
[Document Processing Layer]
        │  -- Parsed Text, Metadata -->
        │
        ├──> [FDA Fact Checker] --- Baseline FDA Rules -->
        │
        └──> [Pattern Analysis Module] --- Brand-Specific Patterns -->
                │
                ▼
         [Evaluation Component] 
                │ (merge baseline + patterns)
                ▼
         [Rule Generation Module] --- Final Rules JSON -->
                │
                ▼
         [Optional: Rule Testing Component] --- Compliance Report
                │
                ▼
         [Main Controller gathers outputs]
