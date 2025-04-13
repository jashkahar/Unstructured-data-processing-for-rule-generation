"""
Type definitions and data classes for the compliance analyzer.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Union
from pathlib import Path

class DocumentType(Enum):
    """Supported document types."""
    PDF = "pdf"
    DOCX = "docx"
    IMAGE = "image"
    VIDEO = "video"

class ComplianceLevel(Enum):
    """Compliance level classifications."""
    PASS = "pass"
    WARNING = "warning"
    FAIL = "fail"

@dataclass
class Document:
    """Represents a processed document with its content and metadata."""
    id: str
    path: Path
    doc_type: DocumentType
    content: str
    metadata: Dict
    processed_date: datetime
    file_size: int
    page_count: Optional[int] = None
    duration: Optional[float] = None  # For videos

@dataclass
class FDAComplianceResult:
    """Results from FDA compliance checking."""
    document_id: str
    compliance_level: ComplianceLevel
    violations: List[str]
    warnings: List[str]
    recommendations: List[str]
    checked_date: datetime

@dataclass
class BrandPattern:
    """Identified brand-specific pattern."""
    category: str
    pattern: str
    confidence: float
    occurrences: int
    examples: List[str]
    context: Dict

@dataclass
class EvaluationResult:
    """Combined evaluation of FDA compliance and brand patterns."""
    document_id: str
    fda_compliance: FDAComplianceResult
    brand_patterns: List[BrandPattern]
    overall_score: float
    recommendations: List[str]

@dataclass
class ComplianceRule:
    """Generated compliance rule."""
    id: str
    description: str
    category: str
    rationale: str
    test_methodology: str
    severity: str
    examples: List[str]
    created_date: datetime

@dataclass
class AnalysisResult:
    """Complete analysis results."""
    fda_compliance: List[FDAComplianceResult]
    brand_patterns: List[BrandPattern]
    evaluation: List[EvaluationResult]
    rules: List[ComplianceRule]

@dataclass
class TestResult:
    """Results from testing compliance rules."""
    rule_id: str
    document_id: str
    passed: bool
    details: Dict
    timestamp: datetime

# Type aliases
ConfigDict = Dict[str, Union[str, int, float, bool, List, Dict]]
JSONDict = Dict[str, Union[str, int, float, bool, List, Dict, None]] 