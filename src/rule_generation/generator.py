"""
Rule generation module for the Pharmaceutical Compliance Analysis System.
Uses LLM to synthesize compliance rules from discovered patterns.
"""

import json
import os
from typing import Dict, List, Optional, Union
from openai import OpenAI
from loguru import logger
from pydantic import BaseModel, Field, validator
from dotenv import load_dotenv
from enum import Enum
import re

# Load environment variables
load_dotenv()

class RuleCategory(str, Enum):
    """Enumeration of valid rule categories."""
    TONE = "Tone"
    BALANCE = "Balance"
    CLAIMS = "Claims"
    STRUCTURE = "Structure"
    LANGUAGE = "Language"
    VISUAL = "Visual"
    DISCLAIMERS = "Disclaimers"
    EVIDENCE = "Evidence"

class SeverityLevel(str, Enum):
    """Enumeration of valid severity levels."""
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"

class ComplianceRule(BaseModel):
    """Model for a compliance rule with enhanced validation."""
    title: str = Field(..., min_length=5, max_length=100, description="Short title describing the rule")
    description: str = Field(..., min_length=20, max_length=500, description="Detailed description of the rule")
    category: RuleCategory = Field(..., description="Category of the rule")
    severity: SeverityLevel = Field(..., description="Severity level")
    examples: List[str] = Field(default_factory=list, min_items=1, max_items=5, description="Example violations of the rule")
    rationale: str = Field(..., min_length=20, max_length=500, description="Explanation of why this rule is important")
    supporting_evidence: List[str] = Field(default_factory=list, description="Evidence from cluster analysis supporting this rule")
    
    # Additional fields for source tracking and rule merging
    source: Optional[str] = Field(None, description="Source of the rule: 'textual', 'visual', or 'merged'")
    inference_confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Confidence score for the rule inference")
    similarity_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Similarity score when rules are merged")
    related_rule_ids: Optional[List[str]] = Field(default_factory=list, description="IDs of related rules")
    test_methodology: Optional[str] = Field(None, description="Method for testing compliance with this rule")
    
    @validator('examples')
    def validate_examples(cls, v):
        """Ensure examples are non-empty strings."""
        return [ex.strip() for ex in v if ex.strip()]
    
    @validator('supporting_evidence')
    def validate_evidence(cls, v):
        """Ensure evidence items are non-empty strings."""
        return [ev.strip() for ev in v if ev.strip()]
    
    @validator('description', 'rationale')
    def validate_text_fields(cls, v):
        """Additional validation for text fields."""
        if not v or not v.strip():
            raise ValueError("Text field cannot be empty or just whitespace")
        return v.strip()
    
    def merge_with(self, other: 'ComplianceRule', similarity_score: float = 0.0) -> 'ComplianceRule':
        """Create a new rule by merging this rule with another.
        
        Args:
            other: Another ComplianceRule to merge with
            similarity_score: Similarity score between the rules
            
        Returns:
            A new ComplianceRule representing the merged rules
        """
        # Determine which rule to use as the base for category and severity
        severity_values = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}
        self_severity_value = severity_values.get(str(self.severity), 2)
        other_severity_value = severity_values.get(str(other.severity), 2)
        
        # Use the highest severity
        severity = self.severity if self_severity_value >= other_severity_value else other.severity
        
        # Combine title with indication if visual aspects are merged
        title = self.title
        if getattr(other, 'source', '') == 'visual' and 'visual' in other.title.lower() and 'visual' not in self.title.lower():
            title = f"{self.title} (Visual Aspects)"
        
        # Combine examples, ensuring uniqueness
        combined_examples = list(set(self.examples + other.examples))
        
        # Combine supporting evidence, ensuring uniqueness
        combined_evidence = list(set(
            getattr(self, 'supporting_evidence', []) + getattr(other, 'supporting_evidence', [])
        ))
        
        # Create merged rule with enhanced description and rationale
        merged = ComplianceRule(
            title=title,
            description=f"{self.description}\n\nVisual Considerations: {other.description}",
            category=self.category,  # Use category from first rule
            severity=severity,
            examples=combined_examples[:5],  # Limit to 5 examples
            rationale=f"{self.rationale}\n\nVisual Rationale: {other.rationale}",
            supporting_evidence=combined_evidence,
            source="merged",
            similarity_score=similarity_score,
            related_rule_ids=[getattr(self, 'id', ''), getattr(other, 'id', '')]
        )
        
        return merged

class RuleGenerator:
    """Generates compliance rules from discovered patterns using LLM."""

    def __init__(self, config: Dict = None):
        """Initialize the rule generator with configuration.

        Args:
            config: Configuration dictionary for rule generation parameters
        """
        self.config = config or {}
        self.model = self.config.get("model", "gpt-4")
        self.temperature = self.config.get("temperature", 0.2)
        self.max_retries = self.config.get("max_retries", 3)
        
        # Initialize OpenAI client
        api_key = os.getenv("OPENAI_API")
        if not api_key:
            raise ValueError("OpenAI API key is required for rule generation. Please set the OPENAI_API_KEY environment variable.")
        
        # Check if using a project-based API key
        # if api_key.startswith("sk-proj-"):
        #     logger.info("Detected project-based API key, configuring client accordingly")
        #     # For project-based API keys, we need to set the base URL
        #     self.client = OpenAI(
        #         api_key=api_key,
        #         base_url="https://api.openai.com/v1"
        #     )
        # else:
        #     # Standard API key
        self.client = OpenAI()
        self.client.api_key = api_key
        
        # Test the API key with a simple request
        try:
            # Make a simple API call to verify the key
            self.client.models.list()
            logger.info("OpenAI API key validated successfully")
        except Exception as e:
            error_message = str(e)
            logger.error(f"Error validating OpenAI API key: {error_message}")
            raise ValueError(f"Invalid OpenAI API key: {error_message}")
    
    def generate_rules(self, patterns: Dict) -> List[ComplianceRule]:
        """Generate compliance rules from discovered patterns.
        
        Args:
            patterns: Dictionary containing cluster analysis and patterns
            
        Returns:
            List of ComplianceRule objects
        """
        try:
            # Log pattern statistics
            logger.info(f"Generating rules from {len(patterns.get('patterns', []))} clusters")
            
            # Prepare comprehensive prompt
            prompt = self._create_rule_generation_prompt(patterns)
            
            # Generate rules using LLM
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature
            )
            
            # Parse response - using the new response format
            rules_text = response.choices[0].message.content
            logger.info(f"LLM response received: {len(rules_text)} characters")
            
            # Save raw response to file for debugging
            os.makedirs("logs", exist_ok=True)
            with open("logs/llm_response.txt", "w", encoding="utf-8") as f:
                f.write(rules_text)
            logger.info("Saved raw LLM response to logs/llm_response.txt")
            
            # Parse and validate rules
            rules = self._parse_and_validate_rules(rules_text)
            if rules:
                logger.info(f"Successfully generated {len(rules)} rules")
                return rules
            
            # If parsing failed, return a fallback rule with the error
            logger.warning("Failed to parse LLM response as valid JSON")
            return [self._create_fallback_rule("Failed to parse LLM response as valid JSON. Check logs/llm_response.txt for the raw response.")]
            
        except Exception as e:
            logger.error(f"Error in rule generation: {str(e)}")
            return [self._create_fallback_rule(str(e))]
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for the LLM."""
        return """You are a pharmaceutical compliance expert specializing in analyzing promotional content patterns and generating comprehensive compliance rules.
        Your task is to analyze patterns across multiple clusters of promotional content and derive both EXPLICIT and IMPLICIT compliance rules.
        
        Focus on identifying:
        1. Explicit patterns that directly indicate compliance issues or strengths
        2. Implicit guidelines that emerge from the data but aren't explicitly stated
        3. Industry best practices that should be followed based on these patterns
        
        Ensure your rules are:
        1. Evidence-based and supported by the cluster analysis
        2. Clear and actionable
        3. Properly categorized and severity-rated
        4. Include specific examples from the data
        5. Provide clear rationale for each rule
        
        IMPORTANT: Your response MUST be a valid JSON array containing rule objects. Do not include any explanatory text before or after the JSON array.
        The JSON array should start with '[' and end with ']' and contain one or more rule objects.
        Each rule object should have the following fields: title, description, category, severity, examples, rationale, and supporting_evidence.
        
        Example format:
        [
          {
            "title": "Rule Title",
            "description": "Rule description",
            "category": "Tone",
            "severity": "HIGH",
            "examples": ["Example 1", "Example 2"],
            "rationale": "Rationale for the rule",
            "supporting_evidence": ["Evidence 1", "Evidence 2"]
          }
        ]"""
    
    def _create_rule_generation_prompt(self, patterns: Dict) -> str:
        """Create comprehensive prompt for rule generation.
        
        Args:
            patterns: Dictionary containing cluster analysis and patterns
            
        Returns:
            Formatted prompt string
        """
        # Start with cluster summary
        prompt = "Based on the following comprehensive cluster analysis of pharmaceutical promotional content, generate a set of compliance rules.\n\n"
        prompt += "CLUSTER SUMMARY:\n"
        
        # Add cluster summaries
        for cluster in patterns.get("patterns", []):
            prompt += self._format_cluster_summary(cluster)
        
        # Add cross-cluster analysis
        prompt += "\nCROSS-CLUSTER ANALYSIS:\n"
        prompt += self._analyze_cross_cluster_patterns(patterns)
        
        # Add rule generation instructions
        prompt += """
        Based on this textual pattern analysis from pharmaceutical marketing materials, I need you to:
        
        1. Infer both EXPLICIT and IMPLICIT compliance and style rules
        2. Consider what these patterns reveal about both stated and unstated guidelines
        3. Identify rules that might not be directly visible but are suggested by the patterns
        
        Focus on extracting rules related to:
        - Regulatory compliance
        - Balance of benefits and risks
        - Appropriate tone and language
        - Evidence-based claims
        - Clear disclaimers
        - Structural patterns
        - Content organization
        - Visual references in text
        
        IMPORTANT: Your response MUST be a valid JSON array containing rule objects. Do not include any explanatory text before or after the JSON array.
        The JSON array should start with '[' and end with ']' and contain one or more rule objects.
        
        Generate rules in the following JSON format:
        [
            {
                "title": "Clear, concise rule title",
                "description": "Detailed rule description with specific guidance",
                "category": "One of: Tone, Balance, Claims, Structure, Language, Visual, Disclaimers, Evidence",
                "severity": "One of: HIGH, MEDIUM, LOW",
                "examples": ["Specific example from the data", "Another example"],
                "rationale": "Clear explanation of why this rule is important and the reasoning behind inferring this rule",
                "supporting_evidence": ["Evidence from cluster analysis", "Additional supporting data"]
            }
        ]
        
        Ensure each rule:
        1. Is supported by evidence from the cluster analysis
        2. Has clear, actionable guidance
        3. Includes specific examples from the data
        4. Provides a clear rationale explaining your reasoning for inferring this rule
        5. Is properly categorized and severity-rated
        
        Return ONLY the JSON array, with no additional text. Do not include any explanations, notes, or other content outside the JSON array."""
        
        return prompt
    
    def _format_cluster_summary(self, cluster: Dict) -> str:
        """Format a single cluster's summary for the prompt.
        
        Args:
            cluster: Cluster data dictionary
            
        Returns:
            Formatted cluster summary string
        """
        summary = f"\nCluster {cluster.get('cluster_id', 'unknown')}:\n"
        summary += f"Size: {cluster.get('size', 0)} chunks\n"
        
        # Add characteristics
        characteristics = cluster.get('characteristics', {})
        if characteristics:
            summary += "Key Characteristics:\n"
            
            # Most common section
            summary += f"- Primary section type: {characteristics.get('most_common_section', 'unknown')}\n"
            
            # Top words
            top_words = characteristics.get('top_words', {})
            if top_words:
                summary += "- Key terms:\n"
                for word, freq in sorted(top_words.items(), key=lambda x: x[1], reverse=True)[:5]:
                    summary += f"  * {word}: {freq} occurrences\n"
            
            # Length statistics
            avg_length = characteristics.get('avg_length', 0)
            summary += f"- Content length: {avg_length:.1f} words average\n"
        
        # Add features
        features = cluster.get('features', {})
        if features:
            summary += "Feature Analysis:\n"
            for feature, value in features.items():
                summary += f"- {feature}: {value}\n"
        
        # Add representative content
        representative = cluster.get('representative', {})
        if representative and 'content' in representative:
            summary += f"Representative Content: {representative['content'][:150]}...\n"
        
        return summary
    
    def _analyze_cross_cluster_patterns(self, patterns: Dict) -> str:
        """Analyze patterns across multiple clusters.
        
        Args:
            patterns: Dictionary containing cluster analysis
            
        Returns:
            Formatted cross-cluster analysis string
        """
        analysis = ""
        
        # Analyze common themes
        all_words = {}
        section_types = {}
        feature_patterns = {}
        
        for cluster in patterns.get("patterns", []):
            # Aggregate word frequencies
            characteristics = cluster.get('characteristics', {})
            for word, freq in characteristics.get('top_words', {}).items():
                all_words[word] = all_words.get(word, 0) + freq
            
            # Aggregate section types
            for section, count in characteristics.get('section_types', {}).items():
                section_types[section] = section_types.get(section, 0) + count
            
            # Aggregate features
            features = cluster.get('features', {})
            for feature, value in features.items():
                if feature not in feature_patterns:
                    feature_patterns[feature] = []
                feature_patterns[feature].append(value)
        
        # Add common themes
        if all_words:
            analysis += "Common Themes:\n"
            for word, freq in sorted(all_words.items(), key=lambda x: x[1], reverse=True)[:10]:
                analysis += f"- {word}: {freq} total occurrences\n"
        
        # Add section distribution
        if section_types:
            analysis += "\nSection Distribution:\n"
            for section, count in sorted(section_types.items(), key=lambda x: x[1], reverse=True):
                analysis += f"- {section}: {count} total chunks\n"
        
        # Add feature patterns
        if feature_patterns:
            analysis += "\nFeature Patterns:\n"
            for feature, values in feature_patterns.items():
                analysis += f"- {feature}: {len(values)} clusters share this feature\n"
        
        return analysis
    
    def _parse_and_validate_rules(self, rules_text: str) -> List[ComplianceRule]:
        """Parse and validate rules from LLM response.
        
        Args:
            rules_text: Raw rules text from LLM
            
        Returns:
            List of validated ComplianceRule objects
        """
        try:
            # Try to extract JSON from the text if it's not pure JSON
            json_text = self._extract_json_from_text(rules_text)
            
            # Parse JSON
            rules_data = json.loads(json_text)
            
            # Ensure rules_data is a list
            if not isinstance(rules_data, list):
                if isinstance(rules_data, dict):
                    rules_data = [rules_data]
                else:
                    return []
            
            # Convert to ComplianceRule objects
            rules = []
            for rule in rules_data:
                try:
                    # Add source attribute if not present
                    if 'source' not in rule:
                        rule['source'] = 'textual'
                    
                    # Set default confidence if not present
                    if 'inference_confidence' not in rule:
                        rule['inference_confidence'] = 0.8
                    
                    # Create ComplianceRule object
                    compliance_rule = ComplianceRule(**rule)
                    rules.append(compliance_rule)
                    logger.info(f"Created rule: {compliance_rule.title}")
                except Exception as e:
                    logger.error(f"Error creating rule from {rule}: {str(e)}")
                    continue
            
            return rules
            
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing rules as JSON: {str(e)}")
            return []
    
    def _extract_json_from_text(self, text: str) -> str:
        """Extract JSON from text that might contain additional content.
        
        Args:
            text: Text that might contain JSON
            
        Returns:
            Extracted JSON string
        """
        # Look for JSON array pattern
        json_array_pattern = r'\[\s*\{.*\}\s*\]'
        match = re.search(json_array_pattern, text, re.DOTALL)
        
        if match:
            return match.group(0)
        
        # Try to find JSON object
        json_object_pattern = r'\{.*\}'
        match = re.search(json_object_pattern, text, re.DOTALL)
        
        if match:
            return f"[{match.group(0)}]"
        
        # If no JSON found, return the original text
        return text
    
    def _create_fallback_rule(self, error_message: str) -> ComplianceRule:
        """Create a fallback rule when rule generation fails.
        
        Args:
            error_message: Error message to include in the rule
            
        Returns:
            ComplianceRule object
        """
        return ComplianceRule(
            title="Rule Generation Error",
            description="There was an error generating compliance rules. Please check the logs for details.",
            category=RuleCategory.STRUCTURE,
            severity=SeverityLevel.HIGH,
            examples=["Error in rule generation process"],
            rationale=f"Rule generation failed: {error_message}",
            supporting_evidence=["System logs", "Error reports"]
        ) 