"""
Main controller module for the Pharmaceutical Compliance Analysis System.
This module orchestrates the entire pipeline from document processing to rule generation.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import yaml
from loguru import logger

from document_processing.processor import DocumentProcessor
from fda_checker.checker import FDAChecker
from pattern_analysis.analyzer import PatternAnalyzer
from evaluation.evaluator import Evaluator
from rule_generation.generator import RuleGenerator
from rule_testing.tester import RuleTester
from utils.config import load_config
from utils.types import Document, ComplianceRule, AnalysisResult

class ComplianceAnalyzer:
    """Main controller class that orchestrates the compliance analysis pipeline."""

    def __init__(self, config_path: str = "config.yaml"):
        """Initialize the compliance analyzer with configuration.

        Args:
            config_path: Path to the configuration file
        """
        self.config = load_config(config_path)
        self._setup_logging()
        
        # Initialize components
        self.doc_processor = DocumentProcessor(self.config["document_processing"])
        self.fda_checker = FDAChecker(self.config["fda_checker"])
        self.pattern_analyzer = PatternAnalyzer(self.config["pattern_analysis"])
        self.evaluator = Evaluator()
        self.rule_generator = RuleGenerator(self.config["rule_generation"])
        self.rule_tester = RuleTester()

    def _setup_logging(self) -> None:
        """Configure logging based on the configuration."""
        log_config = self.config["logging"]
        logger.add(
            log_config["file"],
            rotation=f"{log_config['max_file_size_mb']} MB",
            retention=log_config["backup_count"],
            level=log_config["level"]
        )

    def process_documents(self, input_path: str) -> List[Document]:
        """Process input documents and extract content.

        Args:
            input_path: Path to the input documents

        Returns:
            List of processed documents
        """
        logger.info(f"Processing documents from {input_path}")
        return self.doc_processor.process_directory(input_path)

    def analyze_compliance(self, documents: List[Document]) -> AnalysisResult:
        """Analyze documents for compliance and generate rules.

        Args:
            documents: List of processed documents

        Returns:
            Analysis results including FDA compliance and brand patterns
        """
        # Check FDA compliance
        fda_results = self.fda_checker.check_documents(documents)
        
        # Analyze patterns
        patterns = self.pattern_analyzer.analyze(documents)
        
        # Evaluate combined results
        evaluation = self.evaluator.evaluate(fda_results, patterns)
        
        # Generate rules
        rules = self.rule_generator.generate_rules(evaluation)
        
        return AnalysisResult(
            fda_compliance=fda_results,
            brand_patterns=patterns,
            evaluation=evaluation,
            rules=rules
        )

    def test_rules(self, rules: List[ComplianceRule], test_documents: List[Document]) -> Dict:
        """Test generated rules against new documents.

        Args:
            rules: List of generated compliance rules
            test_documents: Documents to test against

        Returns:
            Dictionary containing test results
        """
        return self.rule_tester.test_rules(rules, test_documents)

    def run_pipeline(self, input_path: str, test_path: Optional[str] = None) -> Dict:
        """Run the complete compliance analysis pipeline.

        Args:
            input_path: Path to input documents
            test_path: Optional path to test documents

        Returns:
            Dictionary containing all analysis results
        """
        try:
            # Process input documents
            documents = self.process_documents(input_path)
            
            # Analyze compliance and generate rules
            analysis_result = self.analyze_compliance(documents)
            
            results = {
                "analysis": analysis_result,
                "test_results": None
            }
            
            # Test rules if test documents are provided
            if test_path:
                test_documents = self.process_documents(test_path)
                results["test_results"] = self.test_rules(
                    analysis_result.rules,
                    test_documents
                )
            
            return results
            
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            raise

def main():
    """Main entry point for the compliance analyzer."""
    analyzer = ComplianceAnalyzer()
    
    # Example usage
    input_path = "promotional_materials/"
    test_path = "promotional_materials/test/"
    
    results = analyzer.run_pipeline(input_path, test_path)
    
    # Save results
    output_path = Path("output")
    output_path.mkdir(exist_ok=True)
    
    with open(output_path / "analysis_results.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
