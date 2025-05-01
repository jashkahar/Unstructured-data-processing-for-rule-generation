"""
Visual rule generator module for the Pharmaceutical Compliance Analysis System.
Generates compliance rules from visual patterns using LLM.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Any
from loguru import logger
from openai import OpenAI
import re

from rule_generation.generator import ComplianceRule, RuleCategory, SeverityLevel

class VisualRuleGenerator:
    """Generates compliance rules from visual patterns using LLM."""
    
    def __init__(self, config: Dict = None):
        """Initialize the visual rule generator with configuration.
        
        Args:
            config: Configuration dictionary for rule generation parameters
        """
        self.config = config or {}
        
        # LLM configuration
        self.model = self.config.get("model", "gpt-4")
        self.temperature = self.config.get("temperature", 0.2)
        self.max_retries = self.config.get("max_retries", 3)
        
        # Output configuration
        self.rules_output = Path(self.config.get("visual_rules_output", "output/intermediate_results/visual_rules.json"))
        self.rules_output.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize OpenAI client
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key is required for rule generation. Please set the OPENAI_API_KEY environment variable.")
        
        self.client = OpenAI()
        self.client.api_key = api_key
        
        logger.info("Initialized VisualRuleGenerator")
    
    def generate_rules(self, visual_patterns: Dict) -> List[ComplianceRule]:
        """Generate compliance rules from visual patterns.
        
        Args:
            visual_patterns: Dictionary containing visual patterns
            
        Returns:
            List of ComplianceRule objects
        """
        try:
            patterns = visual_patterns.get("patterns", [])
            if not patterns:
                logger.warning("No visual patterns to generate rules from")
                return []
            
            logger.info(f"Generating rules from {len(patterns)} visual patterns")
            
            # Prepare prompt
            prompt = self._create_prompt(visual_patterns)
            
            # Generate rules using LLM
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature
            )
            
            # Extract rules from response
            rules_text = response.choices[0].message.content
            
            # Save raw LLM response for debugging
            os.makedirs("logs", exist_ok=True)
            with open("logs/visual_llm_response.txt", "w", encoding="utf-8") as f:
                f.write(rules_text)
            
            # Parse rules
            rules = self._parse_rules(rules_text)
            
            # Save rules to file
            self._save_rules(rules)
            
            logger.info(f"Generated {len(rules)} visual compliance rules")
            return rules
            
        except Exception as e:
            logger.error(f"Error generating visual rules: {str(e)}")
            return []
    
    def update_prompt_for_cross_document(self):
        """Update the system prompt to emphasize cross-document patterns."""
        self.cross_document_prompt = """You are a pharmaceutical advertising and compliance expert specializing in cross-document visual analysis.
        
Your task is to analyze visual patterns discovered ACROSS MULTIPLE pharmaceutical marketing materials and create compliance rules that apply to MULTIPLE DOCUMENTS.

PRIORITIZE patterns that appear consistently in MULTIPLE DOCUMENTS. These cross-document patterns are especially important as they reveal consistent design approaches or compliance strategies.

When creating rules:
1. EMPHASIZE patterns that appear in multiple documents (these are marked with "is_cross_document": true)
2. FOCUS on creating rules that apply broadly across promotional materials
3. HIGHLIGHT when a rule is derived from patterns seen across multiple documents
4. MENTION specific document names that demonstrate the pattern
5. INCORPORATE cross-document insights into the rule rationale

Remember that pharmaceutical promotional materials must adhere to strict FDA regulations, including:
1. Fair balance of risk and benefit information
2. Accurate and non-misleading claims
3. Clear presentation of important safety information
4. Appropriate font sizes and readability
5. Proper use of imagery and visual elements

For COMBINED categories, use a slash format like "Visual/Claims" for rules that span multiple categories.

Each rule should include:
1. A concise title (prefixed with "Cross-Document:" if it applies to multiple documents)
2. A detailed description mentioning how the pattern appears across materials
3. A clear category or combined category (e.g., "Visual/Claims") 
4. Severity level (HIGH, MEDIUM, or LOW)
5. Example violations from DIFFERENT documents when possible
6. Rationale explaining regulatory importance across the document set
7. The names of documents where this pattern appears

Focus on creating actionable rules that address both regulatory requirements and effective pharmaceutical marketing practices that span multiple promotional materials."""
        
        # Store original prompt for restoring if needed
        if not hasattr(self, 'original_system_prompt'):
            self.original_system_prompt = self._get_system_prompt()
        
        # Set flag to use cross-document prompt
        self.use_cross_document_prompt = True
        
        logger.info("Updated system prompt to emphasize cross-document patterns")
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for the LLM."""
        # Use cross-document prompt if set
        if hasattr(self, 'use_cross_document_prompt') and self.use_cross_document_prompt and hasattr(self, 'cross_document_prompt'):
            return self.cross_document_prompt
        
        # Otherwise use the standard prompt
        return """You are a pharmaceutical advertising and compliance expert specializing in visual analysis.
        
Your task is to analyze visual patterns discovered in pharmaceutical marketing materials and create specific compliance rules.

Remember that pharmaceutical promotional materials must adhere to strict FDA regulations, including:
1. Fair balance of risk and benefit information
2. Accurate and non-misleading claims
3. Clear presentation of important safety information
4. Appropriate font sizes and readability
5. Proper use of imagery and visual elements

When creating compliance rules, you will categorize them using one or more of these categories (you can combine multiple with a slash):
- Visual: Rules specifically about visual elements, imagery, or design
- Claims: Rules about claims made in visuals or imagery
- Balance: Rules about balancing risk and benefit information in visuals
- Structure: Rules about the layout, organization or placement of visual elements
- Tone: Rules about the emotional tone conveyed by visual elements
- Disclaimers: Rules about visual presentation of disclaimers or important safety information
- Evidence: Rules about providing visual evidence for claims
- Language: Rules about text used within visuals

For COMBINED categories, use a slash format like "Visual/Claims" for rules that span multiple categories.

Each rule should include:
1. A concise title
2. A detailed description
3. A clear category or combined category (e.g., "Visual/Claims")
4. Severity level (HIGH, MEDIUM, or LOW)
5. Example violations based on the pattern analysis
6. Rationale explaining regulatory importance
7. Any supporting evidence from the pattern analysis

Focus on creating actionable rules that address both regulatory requirements and effective pharmaceutical marketing."""
    
    def _create_prompt(self, visual_patterns: Dict) -> str:
        """Create a prompt for rule generation.
        
        Args:
            visual_patterns: Dictionary containing visual patterns
            
        Returns:
            Formatted prompt string
        """
        patterns = visual_patterns.get("patterns", [])
        
        prompt = """Based on the following visual descriptions of pharmaceutical marketing materials, I need you to infer the set of design and layout rules that these materials appear to follow.

VISUAL DATA FOR MARKETING DOCUMENTS:
"""
        
        # Add pattern summaries
        for pattern in patterns:
            cluster_id = pattern.get("cluster_id", "unknown")
            size = pattern.get("size", 0)
            
            prompt += f"\nVisual Cluster {cluster_id}:\n"
            prompt += f"Size: {size} instances\n"
            
            # Add examples
            examples = pattern.get("representative_examples", [])
            prompt += "Examples:\n"
            for i, example in enumerate(examples[:3]):  # Limit to 3 examples
                prompt += f"  - {example}\n"
            
            # Add common patterns if available
            common_patterns = pattern.get("common_patterns", {})
            if common_patterns:
                prompt += "Common visual elements:\n"
                
                # Layout patterns
                layout = common_patterns.get("layout", "")
                if layout:
                    prompt += f"  - Layout: {layout}\n"
                
                # Color patterns
                colors = common_patterns.get("color", "")
                if colors:
                    prompt += f"  - Colors: {colors}\n"
                
                # Design element patterns
                design = common_patterns.get("design", "")
                if design:
                    prompt += f"  - Design elements: {design}\n"
        
        prompt += """
Based on these visual descriptions, please INFER a set of LATENT DESIGN AND LAYOUT RULES that these pharmaceutical marketing materials appear to follow. 

I'm looking for both:
1. Explicitly visible rules based on recurring patterns
2. Implicit/unstated guidelines that seem to guide the design decisions
3. Rules that might not be directly stated but are suggested by the patterns

Consider how these materials balance:
- Visual impact vs regulatory compliance
- Brand prominence vs safety information
- Creative design vs industry standards
- Attention-grabbing elements vs information clarity

Generate at least 5 specific rules in JSON format:
[
  {
    "title": "Rule title",
    "description": "Detailed description of the inferred rule",
    "category": "Visual",
    "severity": "HIGH/MEDIUM/LOW",
    "examples": ["Example 1", "Example 2"],
    "rationale": "Why this rule matters and how you inferred it from the visual patterns"
  }
]

IMPORTANT: Focus only on visual aspects of the marketing materials. Your response must be a valid JSON array containing rule objects with no explanatory text outside the array."""
        
        return prompt
    
    def _parse_rules(self, rules_text: str) -> List[ComplianceRule]:
        """Parse and validate rules from LLM response.
        
        Args:
            rules_text: Text containing JSON rules
            
        Returns:
            List of ComplianceRule objects
        """
        try:
            # Extract JSON array from response
            json_text = self._extract_json(rules_text)
            
            # Parse JSON
            rules_data = json.loads(json_text)
            
            # Validate and create ComplianceRule objects
            rules = []
            for rule_data in rules_data:
                try:
                    # Ensure category is valid
                    category_str = rule_data.get("category", "Visual")
                    
                    # For combined categories like "Visual/Claims"
                    if '/' in category_str:
                        # Verify each part is valid
                        parts = [part.strip() for part in category_str.split('/')]
                        valid_parts = []
                        for part in parts:
                            if hasattr(RuleCategory, part.upper()):
                                # Use exact case from enum
                                valid_parts.append(getattr(RuleCategory, part.upper()).value)
                            else:
                                # If not valid, skip this part
                                logger.warning(f"Invalid category part '{part}' in combined category")
                        
                        if valid_parts:
                            category_str = '/'.join(valid_parts)
                        else:
                            # Fallback if no parts are valid
                            category_str = "Visual"
                    else:
                        # For single categories, validate and get correct case
                        if hasattr(RuleCategory, category_str.upper()):
                            category_str = getattr(RuleCategory, category_str.upper()).value
                        else:
                            category_str = "Visual"  # Default
                    
                    # Ensure severity is valid
                    severity_str = rule_data.get("severity", "MEDIUM")
                    if not hasattr(SeverityLevel, severity_str.upper()):
                        severity_str = "MEDIUM"
                    
                    # Get supporting evidence (if any)
                    supporting_evidence = rule_data.get("supporting_evidence", [])
                    if isinstance(supporting_evidence, str):
                        supporting_evidence = [supporting_evidence]
                    
                    # Create ComplianceRule
                    rule = ComplianceRule(
                        title=rule_data.get("title", "Missing title"),
                        description=rule_data.get("description", "Missing description"),
                        category=category_str,  # Use string value instead of enum
                        severity=getattr(SeverityLevel, severity_str.upper()),
                        examples=rule_data.get("examples", ["No examples provided"]),
                        rationale=rule_data.get("rationale", "No rationale provided"),
                        supporting_evidence=supporting_evidence,
                        source="visual",  # Mark as coming from visual analysis
                        inference_confidence=rule_data.get("confidence", 0.8)  # Default confidence
                    )
                    
                    rules.append(rule)
                except Exception as e:
                    logger.error(f"Error parsing rule: {str(e)}")
            
            return rules
            
        except Exception as e:
            logger.error(f"Error parsing rules: {str(e)}")
            return []
    
    def _extract_json(self, text: str) -> str:
        """Extract JSON array from text.
        
        Args:
            text: Text containing JSON
            
        Returns:
            Extracted JSON string
        """
        # Try to find JSON array using regex
        match = re.search(r'\[\s*\{.*\}\s*\]', text, re.DOTALL)
        if match:
            return match.group(0)
        
        # If regex fails, check for start and end markers
        if '[' in text and ']' in text:
            start = text.find('[')
            end = text.rfind(']') + 1
            return text[start:end]
        
        # If all else fails, return original text
        return text
    
    def _save_rules(self, rules: List[ComplianceRule]) -> None:
        """Save rules to a JSON file.
        
        Args:
            rules: List of ComplianceRule objects
        """
        try:
            # Convert rules to dictionaries
            rules_data = []
            for rule in rules:
                rules_data.append({
                    "title": rule.title,
                    "description": rule.description,
                    "category": rule.category,  # Already a string with our changes
                    "severity": rule.severity.value,
                    "examples": rule.examples,
                    "rationale": rule.rationale,
                    "supporting_evidence": rule.supporting_evidence,
                    "source": getattr(rule, "source", "visual"),
                    "inference_confidence": getattr(rule, "inference_confidence", None),
                    "similarity_score": getattr(rule, "similarity_score", None),
                    "related_rule_ids": getattr(rule, "related_rule_ids", [])
                })
            
            # Write to file
            with open(self.rules_output, 'w', encoding='utf-8') as f:
                json.dump(rules_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved {len(rules)} visual rules to {self.rules_output}")
            
        except Exception as e:
            logger.error(f"Error saving rules to {self.rules_output}: {str(e)}")
    
    def load_rules(self) -> List[ComplianceRule]:
        """Load rules from a previously saved JSON file.
        
        Returns:
            List of ComplianceRule objects
        """
        try:
            if not self.rules_output.exists():
                logger.warning(f"Rules file not found: {self.rules_output}")
                return []
            
            with open(self.rules_output, 'r', encoding='utf-8') as f:
                rules_data = json.load(f)
            
            rules = []
            for rule_data in rules_data:
                try:
                    # Extract additional fields
                    source = rule_data.get("source", "visual")
                    inference_confidence = rule_data.get("inference_confidence", None)
                    similarity_score = rule_data.get("similarity_score", None)
                    related_rule_ids = rule_data.get("related_rule_ids", [])
                    supporting_evidence = rule_data.get("supporting_evidence", [])
                    
                    # Get category as string
                    category = rule_data.get("category", "Visual")
                    
                    # Create ComplianceRule
                    rule = ComplianceRule(
                        title=rule_data.get("title", "Missing title"),
                        description=rule_data.get("description", "Missing description"),
                        category=category,  # Use string value
                        severity=getattr(SeverityLevel, rule_data.get("severity", "MEDIUM").upper()),
                        examples=rule_data.get("examples", ["No examples provided"]),
                        rationale=rule_data.get("rationale", "No rationale provided"),
                        supporting_evidence=supporting_evidence
                    )
                    
                    # Add additional attributes
                    rule.source = source
                    if inference_confidence is not None:
                        rule.inference_confidence = inference_confidence
                    if similarity_score is not None:
                        rule.similarity_score = similarity_score
                    if related_rule_ids:
                        rule.related_rule_ids = related_rule_ids
                    
                    rules.append(rule)
                except Exception as e:
                    logger.error(f"Error parsing rule: {str(e)}")
            
            logger.info(f"Loaded {len(rules)} visual rules from {self.rules_output}")
            return rules
            
        except Exception as e:
            logger.error(f"Error loading rules from {self.rules_output}: {str(e)}")
            return [] 