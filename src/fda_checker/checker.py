"""
FDA compliance checker module for the Pharmaceutical Compliance Analysis System.
Simulates FDA compliance checking for promotional content.
"""

import re
from typing import Dict, List, Optional
from loguru import logger

from document_processing.chunking import Chunk

class FDAChecker:
    """Simulates FDA compliance checking for promotional content."""
    
    def __init__(self, config: Dict = None):
        """Initialize the FDA checker with configuration.
        
        Args:
            config: Configuration dictionary for FDA checking parameters
        """
        self.config = config or {}
        
        # Define FDA rules (simulated)
        self.fda_rules = {
            "FAIR_BALANCE": {
                "description": "Promotional content must present a fair balance of benefits and risks",
                "keywords": {
                    "benefits": ["effective", "improves", "treats", "helps", "benefits"],
                    "risks": ["side effects", "risks", "warnings", "precautions", "adverse"]
                },
                "severity": "HIGH"
            },
            "EVIDENCE_BASED": {
                "description": "Claims must be supported by substantial evidence",
                "keywords": {
                    "evidence": ["study", "clinical", "research", "data", "evidence", "trial"],
                    "unsupported": ["best", "most", "greatest", "premier", "leading"]
                },
                "severity": "HIGH"
            },
            "APPROPRIATE_TONE": {
                "description": "Content must maintain appropriate professional tone",
                "keywords": {
                    "inappropriate": ["!", "!!", "!!!", "amazing", "revolutionary", "breakthrough"]
                },
                "severity": "MEDIUM"
            },
            "DISCLAIMERS": {
                "description": "Required disclaimers must be present",
                "keywords": {
                    "disclaimers": ["prescription only", "consult your doctor", "not for everyone"]
                },
                "severity": "HIGH"
            }
        }
    
    def check_documents(self, documents: List[Dict]) -> Dict:
        """Check documents for FDA compliance.
        
        Args:
            documents: List of document objects to check
            
        Returns:
            Dictionary containing compliance results
        """
        results = {
            "violations": [],
            "warnings": [],
            "summary": {
                "total_documents": len(documents),
                "violations": 0,
                "warnings": 0
            }
        }
        
        for doc in documents:
            # Get document ID
            doc_id = doc.id
            
            # Get chunks from document
            chunks = doc.chunks
            if not chunks:
                logger.warning(f"No chunks found in document {doc_id}")
                continue
                
            # Check each chunk
            for chunk in chunks:
                # Check compliance for this chunk
                chunk_results = self._check_chunk(chunk.content)
                
                # Add violations
                for violation in chunk_results["violations"]:
                    results["violations"].append({
                        "document_id": doc_id,
                        "section": chunk.section_title,
                        "chunk_id": chunk.chunk_id,
                        "rule": violation["rule"],
                        "description": violation["description"],
                        "severity": violation["severity"]
                    })
                    results["summary"]["violations"] += 1
                
                # Add warnings
                for warning in chunk_results["warnings"]:
                    results["warnings"].append({
                        "document_id": doc_id,
                        "section": chunk.section_title,
                        "chunk_id": chunk.chunk_id,
                        "rule": warning["rule"],
                        "description": warning["description"],
                        "severity": warning["severity"]
                    })
                    results["summary"]["warnings"] += 1
        
        return results
    
    def _check_chunk(self, content: str) -> Dict:
        """Check a single chunk for FDA compliance.
        
        Args:
            content: Text content to check
            
        Returns:
            Dictionary containing violations and warnings
        """
        results = {
            "violations": [],
            "warnings": []
        }
        
        # Check each FDA rule
        for rule_id, rule in self.fda_rules.items():
            # Check fair balance
            if rule_id == "FAIR_BALANCE":
                benefit_count = sum(1 for word in rule["keywords"]["benefits"] 
                                  if word.lower() in content.lower())
                risk_count = sum(1 for word in rule["keywords"]["risks"] 
                               if word.lower() in content.lower())
                
                if benefit_count > 0 and risk_count == 0:
                    results["violations"].append({
                        "rule": rule_id,
                        "description": "Content mentions benefits without corresponding risks",
                        "severity": rule["severity"]
                    })
                elif benefit_count > risk_count * 2:
                    results["warnings"].append({
                        "rule": rule_id,
                        "description": "Content may have imbalanced benefit-risk presentation",
                        "severity": rule["severity"]
                    })
            
            # Check evidence-based claims
            elif rule_id == "EVIDENCE_BASED":
                evidence_count = sum(1 for word in rule["keywords"]["evidence"] 
                                   if word.lower() in content.lower())
                unsupported_count = sum(1 for word in rule["keywords"]["unsupported"] 
                                      if word.lower() in content.lower())
                
                if unsupported_count > 0 and evidence_count == 0:
                    results["violations"].append({
                        "rule": rule_id,
                        "description": "Content makes claims without supporting evidence",
                        "severity": rule["severity"]
                    })
            
            # Check appropriate tone
            elif rule_id == "APPROPRIATE_TONE":
                inappropriate_count = sum(1 for word in rule["keywords"]["inappropriate"] 
                                        if word.lower() in content.lower())
                
                if inappropriate_count > 2:
                    results["warnings"].append({
                        "rule": rule_id,
                        "description": "Content may have inappropriate promotional tone",
                        "severity": rule["severity"]
                    })
            
            # Check disclaimers
            elif rule_id == "DISCLAIMERS":
                disclaimer_count = sum(1 for word in rule["keywords"]["disclaimers"] 
                                     if word.lower() in content.lower())
                
                if disclaimer_count == 0:
                    results["warnings"].append({
                        "rule": rule_id,
                        "description": "Content may be missing required disclaimers",
                        "severity": rule["severity"]
                    })
        
        return results 