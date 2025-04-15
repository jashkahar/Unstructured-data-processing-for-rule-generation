"""
Main controller for the Pharmaceutical Compliance Analysis System.
Orchestrates the compliance analysis pipeline.
"""

import yaml
import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
from dotenv import load_dotenv

from loguru import logger

from document_processing.processor import DocumentProcessor
from document_processing.chunking import ChunkingStrategy
from pattern_analysis.embeddings import EmbeddingManager
from pattern_analysis.feature_extraction import FeatureExtractor
from pattern_analysis.clustering import PatternClusterer
from pattern_analysis.pattern_validator import PatternValidator
from rule_generation.generator import RuleGenerator
from fda_checker.checker import FDAChecker

# Load environment variables from .env file
load_dotenv()

def convert_to_serializable(obj):
    """Convert an object to a JSON-serializable format.
    
    Args:
        obj: Object to convert
        
    Returns:
        JSON-serializable representation of the object
    """
    if hasattr(obj, 'model_dump'):  # Pydantic v2
        return obj.model_dump()
    elif hasattr(obj, 'dict'):  # Pydantic v1
        return obj.dict()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif hasattr(obj, '__dict__'):  # Custom objects
        # Handle Chunk dataclass specifically
        if obj.__class__.__name__ == 'Chunk':
            return {
                'chunk_id': obj.chunk_id,
                'section_title': obj.section_title,
                'content': obj.content,
                'page_number': obj.page_number,
                'metadata': obj.metadata,
                'token_count': obj.token_count,
                'section_type': obj.section_type,
                'visual_elements': obj.visual_elements
            }
        # Handle ComplianceRule from utils/types.py
        elif obj.__class__.__name__ == 'ComplianceRule' and hasattr(obj, 'id'):
            return {
                'id': obj.id,
                'description': obj.description,
                'category': obj.category,
                'rationale': obj.rationale,
                'test_methodology': obj.test_methodology,
                'severity': obj.severity,
                'examples': obj.examples,
                'created_date': obj.created_date.isoformat() if hasattr(obj.created_date, 'isoformat') else str(obj.created_date)
            }
        # Handle ComplianceRule from rule_generation/generator.py
        elif obj.__class__.__name__ == 'ComplianceRule' and hasattr(obj, 'title'):
            return {
                'title': obj.title,
                'description': obj.description,
                'category': obj.category,
                'severity': obj.severity,
                'examples': obj.examples,
                'rationale': obj.rationale
            }
        return obj.__dict__
    return obj

class ComplianceAnalyzer:
    """Main controller for compliance analysis."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize the compliance analyzer.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self._setup_logging()
        self._initialize_components()
        logger.info("Initialized ComplianceAnalyzer")
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Configuration dictionary
        """
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Load environment variables
            if 'rule_generation' in config and 'openai_api_key' in config['rule_generation']:
                # Use environment variable if available, otherwise use config value
                config['rule_generation']['openai_api_key'] = os.getenv('OPENAI_API_KEY', config['rule_generation']['openai_api_key'])
            
            logger.info(f"Loaded configuration from {config_path}")
            return config
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
            raise
            
    def _setup_logging(self):
        """Configure logging based on settings."""
        log_config = self.config.get('logging', {})
        logger.add(
            log_config.get('file', 'logs/compliance_analyzer.log'),
            level=log_config.get('level', 'INFO'),
            rotation=log_config.get('max_file_size', 10485760),
            retention=log_config.get('backup_count', 5)
        )
        
    def _initialize_components(self):
        """Initialize all system components."""
        try:
            # Document processing
            self.document_processor = DocumentProcessor(self.config.get('document_processing', {}))
            self.chunking_strategy = ChunkingStrategy(self.config.get('document_processing', {}))
            
            # Pattern analysis
            self.embedding_manager = EmbeddingManager(self.config.get('pattern_analysis', {}))
            self.feature_extractor = FeatureExtractor(self.config.get('pattern_analysis', {}))
            self.pattern_clusterer = PatternClusterer(self.config.get('pattern_analysis', {}))
            self.pattern_validator = PatternValidator(self.config.get('pattern_analysis', {}))
            
            # Rule generation
            self.rule_generator = RuleGenerator(self.config.get('rule_generation', {}))
            
            # FDA compliance
            self.fda_checker = FDAChecker(self.config.get('fda_compliance', {}))
            
            logger.info("Initialized all components")
        except Exception as e:
            logger.error(f"Error initializing components: {str(e)}")
            raise
            
    def process_document(self, document_path: str) -> Dict[str, Any]:
        """Process a single document through the pipeline.
        
        Args:
            document_path: Path to the document to process
            
        Returns:
            Dictionary containing processing results
        """
        try:
            logger.info(f"Processing document: {document_path}")
            
            # Parse document using document processor
            document = self.document_processor.process_file(document_path)
            if not document:
                raise ValueError(f"Failed to process document: {document_path}")
            
            # Save intermediate result after document parsing
            self._save_intermediate_result("01_document_parsed.json", {
                "document_id": document.id,
                "metadata": document.metadata,
                "content_preview": document.content[:500] + "..." if len(document.content) > 500 else document.content
            })
            
            # Generate chunks
            chunks = self.chunking_strategy.chunk_document(document)
            
            # Save intermediate result after chunking
            self._save_intermediate_result("02_chunks_generated.json", {
                "document_id": document.id,
                "num_chunks": len(chunks),
                "chunks_preview": [
                    {
                        "chunk_id": chunk.chunk_id,
                        "content_preview": chunk.content[:200] + "..." if len(chunk.content) > 200 else chunk.content,
                        "page_number": chunk.page_number
                    }
                    for chunk in chunks[:5]  # Save preview of first 5 chunks
                ]
            })
            
            # Extract features
            features = []
            for chunk in chunks:
                chunk_features = self.feature_extractor.extract_features(chunk.content)
                features.append(chunk_features)
                
            # Save intermediate result after feature extraction
            self._save_intermediate_result("03_features_extracted.json", {
                "document_id": document.id,
                "num_features": len(features),
                "feature_preview": features[:5]  # Save preview of first 5 feature sets
            })
                
            # Generate embeddings
            embeddings = self.embedding_manager.generate_embeddings([chunk.content for chunk in chunks])
            
            # Save intermediate result after embedding generation
            self._save_intermediate_result("04_embeddings_generated.json", {
                "document_id": document.id,
                "num_embeddings": len(embeddings),
                "embedding_dimensions": embeddings.shape[1] if len(embeddings) > 0 else 0
            })
            
            # Discover patterns
            patterns = self.pattern_clusterer.discover_patterns(chunks, embeddings, features)
            
            # Save intermediate result after pattern discovery
            self._save_intermediate_result("05_patterns_discovered.json", {
                "document_id": document.id,
                "num_patterns": len(patterns["patterns"]),
                "pattern_preview": patterns["patterns"][:5]  # Save preview of first 5 patterns
            })
            
            # Validate patterns
            validated_patterns = self.pattern_validator.validate_patterns(patterns)
            
            # Save intermediate result after pattern validation
            self._save_intermediate_result("06_patterns_validated.json", {
                "document_id": document.id,
                "num_validated_patterns": len(validated_patterns.get('patterns', [])),
                "validation_metrics": validated_patterns.get('metrics', {})
            })
            
            # Generate rules
            rules = self.rule_generator.generate_rules({'patterns': validated_patterns['patterns']})
            
            # Save intermediate result after rule generation
            self._save_intermediate_result("07_rules_generated.json", {
                "document_id": document.id,
                "num_rules": len(rules),
                "rule_preview": rules[:5]  # Save preview of first 5 rules
            })
            
            # Check FDA compliance
            # Create a document-like object with chunks for FDA checking
            document_for_fda = type('DocumentForFDA', (), {
                'id': document.id,
                'chunks': chunks
            })
            compliance_results = self.fda_checker.check_documents([document_for_fda])
            
            # Save intermediate result after FDA compliance check
            self._save_intermediate_result("08_compliance_checked.json", {
                "document_id": document.id,
                "compliance_results": compliance_results
            })
            
            # Prepare final results
            results = {
                'additional_rules': [
                    {
                        'rule': rule.title,
                        'description': rule.description,
                        'derivation_reasoning': rule.rationale,
                        'evaluation_method': rule.test_methodology if hasattr(rule, 'test_methodology') else "Content will be checked against this rule's criteria",
                        'examples': rule.examples
                    }
                    for rule in rules
                ]
            }
            
            logger.info(f"Completed processing document: {document_path}")
            return results
            
        except Exception as e:
            logger.error(f"Error processing document {document_path}: {str(e)}")
            raise
            
    def _save_intermediate_result(self, filename: str, data: Dict[str, Any]) -> None:
        """Save intermediate processing results to a JSON file.
        
        Args:
            filename: Name of the file to save results to
            data: Dictionary containing the results to save
        """
        try:
            # Create intermediate results directory if it doesn't exist
            output_dir = Path("output/intermediate_results")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Convert data to JSON-serializable format
            serializable_data = convert_to_serializable(data)
            
            # Save results to file
            output_path = output_dir / filename
            with open(output_path, 'w') as f:
                json.dump(serializable_data, f, indent=2)
                
            logger.debug(f"Saved intermediate results to {output_path}")
            
        except Exception as e:
            logger.error(f"Error saving intermediate results to {filename}: {str(e)}")
            
    def process_directory(self, directory_path: str) -> List[Dict[str, Any]]:
        """Process all documents in a directory.
        
        Args:
            directory_path: Path to directory containing documents
            
        Returns:
            List of processing results for each document
        """
        try:
            logger.info(f"Processing directory: {directory_path}")
            
            # Process all supported files in the directory
            processed_documents = self.document_processor.process_directory(directory_path)
            results = []
            
            for document in processed_documents:
                try:
                    # Process each document through the analysis pipeline
                    result = self.process_document(str(document.path))
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error processing {document.path}: {str(e)}")
                    continue
                    
            logger.info(f"Completed processing directory: {directory_path}")
            return results
            
        except Exception as e:
            logger.error(f"Error processing directory {directory_path}: {str(e)}")
            raise

def main():
    """Main entry point for the compliance analyzer."""
    parser = argparse.ArgumentParser(description='Pharmaceutical Compliance Analysis System')
    parser.add_argument('--input', '-i', help='Path to input document or directory')
    parser.add_argument('--config', '-c', default='config.yaml', help='Path to configuration file')
    parser.add_argument('--output', '-o', help='Path to output file')
    
    args = parser.parse_args()
    
    try:
        # Initialize analyzer
        analyzer = ComplianceAnalyzer(args.config)
        
        # Use config defaults if arguments not provided
        input_path = args.input or analyzer.config.get('paths', {}).get('default_input')
        output_path = args.output or analyzer.config.get('paths', {}).get('default_output')
        
        if not input_path:
            raise ValueError("No input path provided. Please specify --input or set default_input in config.yaml")
        
        # Create output directory if it doesn't exist
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Process input
        input_path = Path(input_path)
        if input_path.is_file():
            results = [analyzer.process_document(str(input_path))]
        else:
            results = analyzer.process_directory(str(input_path))
            
        # Convert results to JSON-serializable format
        serializable_results = convert_to_serializable(results)
            
        # Save results
        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
            
        logger.info(f"Analysis complete. Results saved to {output_path}")
        
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
