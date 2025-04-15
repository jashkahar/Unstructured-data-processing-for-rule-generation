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
from sentence_transformers import SentenceTransformer, util
import torch

from loguru import logger

from document_processing.processor import DocumentProcessor
from document_processing.chunking import ChunkingStrategy
from pattern_analysis.embeddings import EmbeddingManager
from pattern_analysis.feature_extraction import FeatureExtractor
from pattern_analysis.clustering import PatternClusterer
from pattern_analysis.pattern_validator import PatternValidator
from rule_generation.generator import RuleGenerator
from fda_checker.checker import FDAChecker

# Import visual processing components
from visual_processing.image_renderer import ImageRenderer
from visual_processing.visual_extractor import VisualExtractor
from visual_processing.visual_aggregator import VisualAggregator
from rule_generation.visual_rule_generator import VisualRuleGenerator

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
            
            # Initialize visual processing components if configured
            if 'visual_processing' in self.config:
                logger.info("Initializing visual processing components")
                self.image_renderer = ImageRenderer(self.config.get('visual_processing', {}))
                self.visual_extractor = VisualExtractor(self.config.get('visual_processing', {}))
                self.visual_aggregator = VisualAggregator(self.config.get('visual_processing', {}))
                self.visual_rule_generator = VisualRuleGenerator(self.config.get('rule_generation', {}))
            else:
                logger.info("Visual processing not configured, skipping initialization")
                self.image_renderer = None
                self.visual_extractor = None
                self.visual_aggregator = None
                self.visual_rule_generator = None
            
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
            
            # Text processing pipeline
            logger.info(f"Starting text processing pipeline for {document_path}")
            text_rules = self._process_text_pipeline(document)
            
            # Visual processing pipeline (if configured)
            visual_rules = []
            if hasattr(self, 'image_renderer') and self.image_renderer is not None:
                logger.info(f"Starting visual processing pipeline for {document_path}")
                visual_rules = self._process_visual_pipeline(document_path, document.id)
            else:
                logger.info("Visual processing not configured, skipping visual analysis")
            
            # Merge text and visual rules
            all_rules = self._merge_rules(text_rules, visual_rules, document.id)
            
            # Save intermediate result after merging rules
            self._save_intermediate_result("09_all_rules_merged.json", {
                "document_id": document.id,
                "total_rules": len(all_rules),
                "text_rules": len(text_rules),
                "visual_rules": len(visual_rules),
                "merged_rules": len(all_rules),
                "rules_preview": [convert_to_serializable(r) for r in all_rules[:5]]  # Save preview of first 5 rules
            })
            
            # Prepare final results
            results = {
                'additional_rules': [
                    {
                        'rule': rule.title,
                        'description': rule.description,
                        'derivation_reasoning': rule.rationale,
                        'evaluation_method': rule.test_methodology if hasattr(rule, 'test_methodology') else "Content will be checked against this rule's criteria",
                        'examples': rule.examples,
                        'source': getattr(rule, 'source', 'textual') if hasattr(rule, 'source') else 'textual'
                    }
                    for rule in all_rules
                ]
            }
            
            logger.info(f"Completed processing document: {document_path}")
            return results
            
        except Exception as e:
            logger.error(f"Error processing document {document_path}: {str(e)}")
            raise
    
    def _merge_rules(self, text_rules: List[Any], visual_rules: List[Any], document_id: str) -> List[Any]:
        """Merge text and visual rules, avoiding duplication and identifying related rules.
        
        Args:
            text_rules: List of rules from text processing
            visual_rules: List of rules from visual processing
            document_id: ID of the document being processed
            
        Returns:
            List of merged rules
        """
        if not visual_rules:
            logger.info("No visual rules to merge, returning text rules only")
            return text_rules
            
        if not text_rules:
            logger.info("No text rules to merge, returning visual rules only")
            return visual_rules
        
        try:
            logger.info(f"Merging {len(text_rules)} text rules and {len(visual_rules)} visual rules")
            
            # Step 1: Identify potentially duplicate or complementary rules
            # We'll use sentence embeddings to compare rule descriptions
            
            # Initialize sentence embedding model if needed
            if not hasattr(self, '_sentence_encoder'):
                model_name = self.config.get('models', {}).get('embedding_model', 'sentence-transformers/all-MiniLM-L6-v2')
                self._sentence_encoder = SentenceTransformer(model_name)
                logger.info(f"Initialized sentence encoder model: {model_name}")
            
            # Generate descriptions for comparison
            text_descriptions = [f"{rule.title} {rule.description}" for rule in text_rules]
            visual_descriptions = [f"{rule.title} {rule.description}" for rule in visual_rules]
            
            # Only compute embeddings if both lists have elements
            if text_descriptions and visual_descriptions:
                # Compute embeddings
                text_embeddings = self._sentence_encoder.encode(text_descriptions, convert_to_tensor=True)
                visual_embeddings = self._sentence_encoder.encode(visual_descriptions, convert_to_tensor=True)
                
                # Compute similarity matrix
                similarity_scores = util.cos_sim(text_embeddings, visual_embeddings)
                
                # Find potential matches (high similarity score)
                similarity_threshold = self.config.get('rule_merging', {}).get('similarity_threshold', 0.7)
                similar_rule_pairs = []
                
                for i in range(len(text_rules)):
                    for j in range(len(visual_rules)):
                        score = similarity_scores[i][j].item()
                        if score > similarity_threshold:
                            similar_rule_pairs.append((i, j, score))
                
                # Sort by similarity score (highest first)
                similar_rule_pairs.sort(key=lambda x: x[2], reverse=True)
                
                # Track which rules have been merged
                merged_text_rules = set()
                merged_visual_rules = set()
                
                # Create merged rules list
                merged_rules = []
                
                # Process similar rule pairs
                for text_idx, visual_idx, score in similar_rule_pairs:
                    if text_idx in merged_text_rules or visual_idx in merged_visual_rules:
                        continue  # Skip if either rule has already been merged
                    
                    # Get the rules
                    text_rule = text_rules[text_idx]
                    visual_rule = visual_rules[visual_idx]
                    
                    # Create a merged rule
                    merged_rule = self._create_merged_rule(text_rule, visual_rule, score)
                    merged_rules.append(merged_rule)
                    
                    # Mark as merged
                    merged_text_rules.add(text_idx)
                    merged_visual_rules.add(visual_idx)
                
                # Add remaining unmerged text rules
                for i, rule in enumerate(text_rules):
                    if i not in merged_text_rules:
                        # Add source attribute
                        rule.source = 'textual'
                        merged_rules.append(rule)
                
                # Add remaining unmerged visual rules
                for i, rule in enumerate(visual_rules):
                    if i not in merged_visual_rules:
                        # Add source attribute
                        rule.source = 'visual'
                        merged_rules.append(rule)
                
                logger.info(f"Created {len(merged_rules)} merged rules ({len(similar_rule_pairs)} pairs of similar rules identified)")
                
                # Save detailed merging information for debugging
                self._save_intermediate_result(f"merge_analysis_{document_id}.json", {
                    "document_id": document_id,
                    "text_rules_count": len(text_rules),
                    "visual_rules_count": len(visual_rules),
                    "merged_rules_count": len(merged_rules),
                    "similar_pairs_count": len(similar_rule_pairs),
                    "similar_pairs": [
                        {
                            "text_rule": text_rules[i].title,
                            "visual_rule": visual_rules[j].title,
                            "similarity_score": score
                        }
                        for i, j, score in similar_rule_pairs
                    ],
                    "similarity_threshold": similarity_threshold
                })
                
                return merged_rules
            else:
                # If one list is empty, just return the other with source added
                all_rules = []
                for rule in text_rules:
                    rule.source = 'textual'
                    all_rules.append(rule)
                for rule in visual_rules:
                    rule.source = 'visual'
                    all_rules.append(rule)
                return all_rules
                
        except Exception as e:
            logger.error(f"Error merging rules: {str(e)}")
            
            # Fall back to simple concatenation if merging fails
            all_rules = []
            for rule in text_rules:
                rule.source = 'textual'
                all_rules.append(rule)
            for rule in visual_rules:
                rule.source = 'visual'
                all_rules.append(rule)
            return all_rules
    
    def _create_merged_rule(self, text_rule: Any, visual_rule: Any, similarity_score: float) -> Any:
        """Create a merged rule from a text rule and a visual rule.
        
        Args:
            text_rule: Rule from text processing
            visual_rule: Rule from visual processing
            similarity_score: Similarity score between the rules
            
        Returns:
            Merged rule
        """
        # Use the new merge_with method if available
        if hasattr(text_rule, 'merge_with'):
            # Ensure source attribute is set
            if not hasattr(text_rule, 'source'):
                text_rule.source = 'textual'
            if not hasattr(visual_rule, 'source'):
                visual_rule.source = 'visual'
                
            # Use the merge_with method
            merged_rule = text_rule.merge_with(visual_rule, similarity_score)
            return merged_rule
        
        # Fall back to the original implementation if merge_with is not available
        from rule_generation.generator import ComplianceRule
        
        # Create a merged title
        if 'visual' in visual_rule.title.lower() and visual_rule.title.lower() not in text_rule.title.lower():
            title = f"{text_rule.title} (Visual Aspects)"
        else:
            title = text_rule.title
        
        # Enhance description
        description = f"{text_rule.description}\n\nVisual Considerations: {visual_rule.description}"
        
        # Combine examples
        examples = list(set(text_rule.examples + visual_rule.examples))
        
        # Enhance rationale
        rationale = f"{text_rule.rationale}\n\nVisual Rationale: {visual_rule.rationale}"
        
        # Use the highest severity
        severity_values = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}
        text_severity_value = severity_values.get(str(text_rule.severity), 2)
        visual_severity_value = severity_values.get(str(visual_rule.severity), 2)
        severity = text_rule.severity if text_severity_value >= visual_severity_value else visual_rule.severity
        
        # Create merged rule
        merged_rule = ComplianceRule(
            title=title,
            description=description,
            category=text_rule.category,  # Use category from text rule
            severity=severity,
            examples=examples,
            rationale=rationale
        )
        
        # Add additional metadata
        merged_rule.source = 'merged'
        merged_rule.similarity_score = similarity_score
        merged_rule.text_rule_title = text_rule.title
        merged_rule.visual_rule_title = visual_rule.title
        
        return merged_rule
    
    def _process_text_pipeline(self, document) -> List[Any]:
        """Process the text pipeline for a document.
        
        Args:
            document: Document object to process
            
        Returns:
            List of generated rules
        """
        try:
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
            
            return rules
            
        except Exception as e:
            logger.error(f"Error in text processing pipeline: {str(e)}")
            return []
    
    def _process_visual_pipeline(self, document_path: str, document_id: str) -> List[Any]:
        """Process the visual pipeline for a document.
        
        Args:
            document_path: Path to the document
            document_id: ID of the document
            
        Returns:
            List of generated visual rules
        """
        try:
            # Render PDF pages as images
            logger.info(f"Rendering images for document: {document_path}")
            image_paths = self.image_renderer.render_pdf(document_path)
            
            # Save intermediate result after image rendering
            self._save_intermediate_result("v01_images_rendered.json", {
                "document_id": document_id,
                "num_images": len(image_paths),
                "image_paths_preview": [img["image_path"] for img in image_paths[:3]]  # First 3 paths
            })
            
            if not image_paths:
                logger.warning(f"No images rendered for document: {document_path}")
                return []
            
            # Generate captions from images
            logger.info(f"Generating captions for {len(image_paths)} images")
            captions_data = self.visual_extractor.extract_captions(image_paths)
            
            # Save intermediate result after caption extraction
            self._save_intermediate_result("v02_captions_extracted.json", {
                "document_id": document_id,
                "num_captions": len(captions_data),
                "captions_preview": captions_data[:2]  # First 2 captions
            })
            
            if not captions_data:
                logger.warning(f"No captions generated for document: {document_path}")
                self.image_renderer.clean_up_images(image_paths)
                return []
            
            # Aggregate and cluster captions
            logger.info("Aggregating and clustering captions")
            visual_patterns = self.visual_aggregator.aggregate_captions(captions_data)
            
            # Save intermediate result after caption aggregation
            self._save_intermediate_result("v03_visual_patterns.json", {
                "document_id": document_id,
                "num_patterns": len(visual_patterns.get("patterns", [])),
                "patterns_preview": visual_patterns.get("patterns", [])[:2]  # First 2 patterns
            })
            
            # Generate visual rules
            logger.info("Generating visual compliance rules")
            visual_rules = self.visual_rule_generator.generate_rules(visual_patterns)
            
            # Save intermediate result after visual rule generation
            self._save_intermediate_result("v04_visual_rules.json", {
                "document_id": document_id,
                "num_rules": len(visual_rules),
                "rules_preview": [convert_to_serializable(r) for r in visual_rules[:3]]  # First 3 rules
            })
            
            # Clean up temporary image files
            logger.info("Cleaning up temporary image files")
            self.image_renderer.clean_up_images(image_paths)
            
            return visual_rules
            
        except Exception as e:
            logger.error(f"Error in visual processing pipeline: {str(e)}")
            return []
            
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
