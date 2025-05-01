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
from PIL import Image
import copy
from enum import Enum

from loguru import logger

from document_processing.processor import DocumentProcessor
from document_processing.chunking import ChunkingStrategy
from pattern_analysis.embeddings import EmbeddingManager
from pattern_analysis.feature_extraction import FeatureExtractor
from pattern_analysis.clustering import PatternClusterer
from pattern_analysis.pattern_validator import PatternValidator
from rule_generation.generator import RuleGenerator
from fda_checker.checker import FDAChecker
from utils.types import DocumentType

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
    # Handle Enum objects
    elif hasattr(obj, 'value') and hasattr(obj, '__class__') and issubclass(obj.__class__, Enum):
        return obj.value
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
                'severity': convert_to_serializable(obj.severity),
                'examples': obj.examples,
                'created_date': obj.created_date.isoformat() if hasattr(obj.created_date, 'isoformat') else str(obj.created_date)
            }
        # Handle ComplianceRule from rule_generation/generator.py
        elif obj.__class__.__name__ == 'ComplianceRule' and hasattr(obj, 'title'):
            return {
                'title': obj.title,
                'description': obj.description,
                'category': obj.category,
                'severity': convert_to_serializable(obj.severity),
                'examples': obj.examples,
                'rationale': obj.rationale
            }
        # Convert all attributes to serializable format
        return {k: convert_to_serializable(v) for k, v in obj.__dict__.items()}
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
                # Pass document type to visual pipeline
                visual_rules = self._process_visual_pipeline(document_path, document.id, document.doc_type)
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
            document_id: ID of the document being processed, or "cross_document_analysis" for unified processing
            
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
            is_cross_document = document_id == "cross_document_analysis"
            
            if is_cross_document:
                logger.info(f"Merging {len(text_rules)} text rules and {len(visual_rules)} visual rules for cross-document analysis")
            else:
                logger.info(f"Merging {len(text_rules)} text rules and {len(visual_rules)} visual rules for document {document_id}")
            
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
                
                # For cross-document analysis, we can use a slightly lower threshold
                if is_cross_document:
                    similarity_threshold *= 0.9  # 10% lower threshold for cross-document
                
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
                    
                    # For cross-document analysis, combine document references
                    if is_cross_document:
                        text_documents = getattr(text_rule, 'documents', [])
                        visual_documents = getattr(visual_rule, 'documents', [])
                        merged_rule.documents = list(set(text_documents + visual_documents))
                        
                        # If the merged rule applies to multiple documents, mark it
                        if len(merged_rule.documents) > 1 and not merged_rule.title.startswith("Cross-Document:"):
                            merged_rule.title = f"Cross-Document: {merged_rule.title}"
                            merged_rule.cross_document_score = min(1.0, 0.7 + 0.05 * len(merged_rule.documents))
                    
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
                
                # Sort merged rules by cross-document score if this is cross-document analysis
                if is_cross_document:
                    merged_rules.sort(key=lambda r: getattr(r, 'cross_document_score', 0.0), reverse=True)
                    
                logger.info(f"Created {len(merged_rules)} merged rules ({len(similar_rule_pairs)} pairs of similar rules identified)")
                
                # For cross-document analysis, prioritize rules that appear in multiple documents
                cross_doc_count = sum(1 for r in merged_rules if getattr(r, 'cross_document_score', 0.0) > 0)
                logger.info(f"Found {cross_doc_count} rules that apply to multiple documents")
                
                # Save detailed merging information for debugging
                self._save_intermediate_result(f"merge_analysis_{document_id}.json", {
                    "document_id": document_id,
                    "text_rules_count": len(text_rules),
                    "visual_rules_count": len(visual_rules),
                    "merged_rules_count": len(merged_rules),
                    "similar_pairs_count": len(similar_rule_pairs),
                    "is_cross_document": is_cross_document,
                    "cross_document_rules_count": sum(1 for r in merged_rules if getattr(r, 'cross_document_score', 0.0) > 0),
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
                
                # Sort by cross-document score if this is cross-document analysis
                if is_cross_document:
                    all_rules.sort(key=lambda r: getattr(r, 'cross_document_score', 0.0), reverse=True)
                    
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
        """Create a merged rule from text and visual rules.
        
        Args:
            text_rule: Rule from text processing
            visual_rule: Rule from visual processing
            similarity_score: Similarity score between the rules
            
        Returns:
            Merged rule object
        """
        try:
            # Start with a copy of the text rule
            merged_rule = copy.deepcopy(text_rule)
            
            # Add visual rule information
            merged_rule.title = f"{text_rule.title} (with visual elements)"
            merged_rule.description = f"{text_rule.description}\n\nVisual elements: {visual_rule.description}"
            merged_rule.rationale += f"\n\nVisual evidence: {visual_rule.rationale}"
            
            # Combine examples
            text_examples = set(text_rule.examples)
            visual_examples = set(visual_rule.examples)
            merged_rule.examples = list(text_examples.union(visual_examples))
            
            # Mark as merged rule
            merged_rule.source = 'merged'
            merged_rule.similarity_score = similarity_score
            
            return merged_rule
        except Exception as e:
            logger.error(f"Error creating merged rule: {str(e)}")
            # Fall back to text rule if merging fails
            text_rule.source = 'textual'
            return text_rule
    
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
    
    def _process_visual_pipeline(self, document_path: str, document_id: str, doc_type: DocumentType = None) -> List[Any]:
        """Process the visual pipeline for a document.
        
        Args:
            document_path: Path to the document
            document_id: ID of the document
            doc_type: Type of document (PDF or IMAGE)
            
        Returns:
            List of generated visual rules
        """
        try:
            # Determine if this is a PDF or an image file
            is_image = doc_type == DocumentType.IMAGE if doc_type else Path(document_path).suffix.lower() in ['.jpg', '.jpeg', '.png', '.tiff', '.bmp']
            image_paths = []
            
            if is_image:
                # For standalone images, use the image renderer's standalone image processor
                logger.info(f"Processing standalone image: {document_path}")
                image_metadata = self.image_renderer.process_standalone_image(document_path)
                if image_metadata:
                    image_paths = [image_metadata]
            else:
                # For PDFs, render pages as images
                logger.info(f"Rendering images for PDF document: {document_path}")
                image_paths = self.image_renderer.render_pdf(document_path)
            
            # Save intermediate result after image preparation
            self._save_intermediate_result("v01_images_prepared.json", {
                "document_id": document_id,
                "num_images": len(image_paths),
                "is_standalone_image": is_image,
                "image_paths_preview": [img["image_path"] for img in image_paths[:3]]  # First 3 paths
            })
            
            if not image_paths:
                logger.warning(f"No images prepared for document: {document_path}")
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
                if not is_image:  # Only clean up temporary images from PDFs
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
            
            # Clean up temporary image files only for PDFs
            if not is_image:
                logger.info("Cleaning up temporary image files from PDF")
                self.image_renderer.clean_up_images(image_paths)
            else:
                logger.info("Skipping cleanup for original image files")
            
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
        """Process all documents in a directory together for cross-document analysis.
        
        Args:
            directory_path: Path to directory containing documents
            
        Returns:
            List of processing results
        """
        try:
            logger.info(f"Processing directory: {directory_path}")
            
            # 1. First, collect all documents
            processed_documents = self.document_processor.process_directory(directory_path)
            logger.info(f"Collected {len(processed_documents)} documents for unified analysis")
            
            # 2. Process text content from all documents together
            logger.info("Starting unified text processing pipeline")
            all_text_chunks = []
            document_map = {}
            
            for doc in processed_documents:
                # Create a document map for later reference
                document_map[doc.id] = doc
                
                # Chunk the document
                chunks = self.chunking_strategy.chunk_document(doc)
                
                # Add document reference to each chunk
                for chunk in chunks:
                    chunk.document_id = doc.id
                    all_text_chunks.append(chunk)
            
            # 3. Process all chunks together to find cross-document patterns
            text_rules = self._process_unified_text_pipeline(all_text_chunks, document_map)
            
            # 4. Process all visual content together
            logger.info("Starting unified visual processing pipeline")
            all_image_paths = []
            
            for doc in processed_documents:
                doc_type = doc.doc_type
                doc_path = str(doc.path)
                
                # Collect images from all documents
                if doc_type == DocumentType.IMAGE:
                    # For standalone images
                    image_metadata = self.image_renderer.process_standalone_image(doc_path)
                    if image_metadata:
                        all_image_paths.append(image_metadata)
                elif doc_type == DocumentType.PDF:
                    # For PDFs, render pages as images
                    pdf_images = self.image_renderer.render_pdf(doc_path)
                    all_image_paths.extend(pdf_images)
            
            # 5. Process all images together
            visual_rules = self._process_unified_visual_pipeline(all_image_paths, document_map)
            
            # 6. Merge text and visual rules
            all_rules = self._merge_rules(text_rules, visual_rules, "cross_document_analysis")
            
            # 7. Save results
            results = {
                'rules': [
                    {
                        'rule': rule.title,
                        'description': rule.description,
                        'derivation_reasoning': rule.rationale,
                        'evaluation_method': rule.test_methodology if hasattr(rule, 'test_methodology') else "Content checked against criteria",
                        'examples': rule.examples,
                        'source': getattr(rule, 'source', 'textual') if hasattr(rule, 'source') else 'textual',
                        'applies_to_documents': getattr(rule, 'documents', [])
                    }
                    for rule in all_rules
                ],
                'document_count': len(processed_documents),
                'document_ids': [doc.id for doc in processed_documents]
            }
            
            # Save intermediate result after cross-document analysis
            self._save_intermediate_result("cross_document_analysis.json", {
                "document_count": len(processed_documents),
                "text_rules": len(text_rules),
                "visual_rules": len(visual_rules),
                "merged_rules": len(all_rules),
                "rules_preview": [convert_to_serializable(r) for r in all_rules[:5]]
            })
            
            logger.info(f"Completed cross-document analysis of {len(processed_documents)} documents")
            return [results]  # Return as list for compatibility with existing code
            
        except Exception as e:
            logger.error(f"Error processing directory {directory_path}: {str(e)}")
            raise

    def _process_unified_text_pipeline(self, all_chunks: List[Any], document_map: Dict[str, Any]) -> List[Any]:
        """Process text chunks across all documents to find common patterns.
        
        Args:
            all_chunks: List of all chunks from all documents
            document_map: Map of document IDs to Document objects
            
        Returns:
            List of generated text rules
        """
        try:
            logger.info(f"Processing {len(all_chunks)} chunks from all documents")
            
            # Save intermediate result after chunking
            self._save_intermediate_result("u01_all_chunks.json", {
                "chunk_count": len(all_chunks),
                "document_count": len(document_map),
                "document_ids": list(document_map.keys()),
                "chunks_preview": [convert_to_serializable(c) for c in all_chunks[:5]]
            })
            
            # Generate embeddings for all chunks
            embeddings = self.embedding_manager.generate_embeddings(all_chunks)
            
            # Extract features from all chunks
            features = self.feature_extractor.extract_features(all_chunks, embeddings)
            
            # Save intermediate result after feature extraction
            self._save_intermediate_result("u02_features_extracted.json", {
                "chunk_count": len(all_chunks),
                "feature_count": len(features) if isinstance(features, list) else "N/A",
                "features_preview": convert_to_serializable(features)[:1000] if isinstance(features, str) else {}
            })
            
            # Discover patterns across all documents
            patterns = self.pattern_clusterer.discover_patterns(all_chunks, embeddings, features)
            
            # Track which documents contain each pattern
            for pattern in patterns['patterns']:
                # Extract document IDs from chunks in this pattern
                doc_ids = []
                for chunk in pattern['chunks']:
                    if hasattr(chunk, 'document_id'):
                        doc_ids.append(chunk.document_id)
                
                pattern['documents'] = list(set(doc_ids))
                
                # Add document titles for context
                pattern['document_titles'] = [
                    os.path.basename(str(document_map[doc_id].path)) 
                    for doc_id in pattern['documents'] if doc_id in document_map
                ]
                
                # Mark as cross-document pattern if it appears in multiple documents
                if len(pattern.get('documents', [])) > 1:
                    pattern['is_cross_document'] = True
                    pattern['cross_document_score'] = min(1.0, 0.5 + 0.1 * len(pattern.get('documents', [])))
                else:
                    pattern['is_cross_document'] = False
                    pattern['cross_document_score'] = 0.0
            
            # Save intermediate result after pattern discovery
            self._save_intermediate_result("u03_cross_doc_patterns.json", {
                "pattern_count": len(patterns['patterns']),
                "patterns_preview": convert_to_serializable(patterns['patterns'][:3])
            })
            
            # Validate patterns
            validated_patterns = self.pattern_validator.validate_patterns(patterns)
            
            # Update the prompt to emphasize cross-document patterns
            self.rule_generator.update_prompt_for_cross_document()
            
            # Generate rules
            rules = self.rule_generator.generate_rules(validated_patterns)
            
            # Add document references to rules
            for rule in rules:
                if hasattr(rule, 'supporting_evidence') and rule.supporting_evidence:
                    # Extract document IDs from supporting evidence
                    doc_ids = []
                    for evidence in rule.supporting_evidence:
                        if isinstance(evidence, str) and 'document:' in evidence:
                            doc_id = evidence.split('document:')[1].strip()
                            if doc_id:
                                doc_ids.append(doc_id)
                    rule.documents = list(set(doc_ids))
                else:
                    rule.documents = []
                
                # Mark as cross-document rule if it applies to multiple documents
                if len(getattr(rule, 'documents', [])) > 1 and not rule.title.startswith("Cross-Document:"):
                    rule.title = f"Cross-Document: {rule.title}"
                    rule.cross_document_score = min(1.0, 0.6 + 0.05 * len(getattr(rule, 'documents', [])))
                else:
                    rule.cross_document_score = 0.0
            
            # Sort rules by cross-document score to prioritize patterns across documents
            rules.sort(key=lambda r: getattr(r, 'cross_document_score', 0.0), reverse=True)
            
            # Save intermediate result after rule generation
            self._save_intermediate_result("u04_cross_doc_text_rules.json", {
                "rule_count": len(rules),
                "cross_document_rules": sum(1 for r in rules if getattr(r, 'cross_document_score', 0.0) > 0),
                "rules_preview": [convert_to_serializable(r) for r in rules[:5]]
            })
            
            logger.info(f"Generated {len(rules)} text rules from cross-document analysis")
            return rules
            
        except Exception as e:
            logger.error(f"Error in unified text processing pipeline: {str(e)}")
            return []

    def _process_unified_visual_pipeline(self, all_image_paths: List[Dict], document_map: Dict[str, Any]) -> List[Any]:
        """Process visual content across all documents to find common patterns.
        
        Args:
            all_image_paths: List of all image paths from all documents
            document_map: Map of document IDs to Document objects
            
        Returns:
            List of generated visual rules
        """
        try:
            logger.info(f"Processing {len(all_image_paths)} images from all documents")
            
            # Save intermediate result after image collection
            self._save_intermediate_result("v01_all_images_collected.json", {
                "image_count": len(all_image_paths),
                "document_count": len(document_map),
                "image_paths_preview": [img["image_path"] for img in all_image_paths[:5]]
            })
            
            # Generate captions for all images
            captions_data = self.visual_extractor.extract_captions(all_image_paths)
            
            # Save intermediate result after caption extraction
            self._save_intermediate_result("v02_cross_doc_captions.json", {
                "caption_count": len(captions_data),
                "captions_preview": captions_data[:2]  # First 2 captions
            })
            
            if not captions_data:
                logger.warning("No captions generated for any documents")
                return []
            
            # Aggregate and cluster captions across all documents
            visual_patterns = self.visual_aggregator.aggregate_captions(captions_data)
            
            # Track which documents each pattern appears in
            for pattern in visual_patterns.get('patterns', []):
                # Add document titles for context
                try:
                    pattern['document_titles'] = [
                        os.path.basename(str(document_map[doc_id].path)) 
                        for doc_id in pattern.get('documents', []) if doc_id in document_map
                    ]
                except Exception as e:
                    logger.error(f"Error adding document titles to pattern: {str(e)}")
                    pattern['document_titles'] = []
                
                # Mark as cross-document pattern if it appears in multiple documents
                if len(pattern.get('documents', [])) > 1:
                    pattern['is_cross_document'] = True
                    pattern['cross_document_score'] = min(1.0, 0.5 + 0.1 * len(pattern.get('documents', [])))
                else:
                    pattern['is_cross_document'] = False
                    pattern['cross_document_score'] = 0.0
            
            # Save intermediate result after pattern aggregation
            self._save_intermediate_result("v03_cross_doc_visual_patterns.json", {
                "pattern_count": len(visual_patterns.get('patterns', [])),
                "patterns_preview": visual_patterns.get('patterns', [])[:2]  # First 2 patterns
            })
            
            # Update the prompt to emphasize cross-document patterns
            self.visual_rule_generator.update_prompt_for_cross_document()
            
            # Generate visual rules with cross-document context
            visual_rules = self.visual_rule_generator.generate_rules(visual_patterns)
            
            # Mark rules that apply to multiple documents
            for rule in visual_rules:
                # Extract document IDs from the rule
                if hasattr(rule, 'documents'):
                    # Mark as cross-document rule if it applies to multiple documents
                    if len(rule.documents) > 1 and not rule.title.startswith("Cross-Document:"):
                        rule.title = f"Cross-Document: {rule.title}"
                        rule.cross_document_score = min(1.0, 0.6 + 0.05 * len(rule.documents))
                    else:
                        rule.cross_document_score = 0.0
            
            # Save intermediate result after rule generation
            self._save_intermediate_result("v04_cross_doc_visual_rules.json", {
                "rule_count": len(visual_rules),
                "rules_preview": [convert_to_serializable(r) for r in visual_rules[:3]]  # First 3 rules
            })
            
            # Clean up temporary images
            tmp_images = [img for img in all_image_paths if str(self.image_renderer.output_dir) in img.get('image_path', '')]
            self.image_renderer.clean_up_images(tmp_images)
            
            logger.info(f"Generated {len(visual_rules)} visual rules from cross-document analysis")
            return visual_rules
            
        except Exception as e:
            logger.error(f"Error in unified visual processing pipeline: {str(e)}")
            return []

def main():
    """Main entry point for the compliance analyzer."""
    parser = argparse.ArgumentParser(description='Pharmaceutical Compliance Analysis System')
    parser.add_argument('--input', '-i', help='Path to input document or directory')
    parser.add_argument('--config', '-c', default='config.yaml', help='Path to configuration file')
    parser.add_argument('--output', '-o', help='Path to output file')
    parser.add_argument('--cross-document', '-x', action='store_true', help='Process directory as cross-document analysis')
    
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
        results = None
        
        if input_path.is_file():
            # For single file, always use per-document processing
            logger.info(f"Processing single file: {input_path}")
            results = [analyzer.process_document(str(input_path))]
        else:
            # For directory, check if cross-document flag is set
            if args.cross_document:
                logger.info(f"Processing directory with cross-document analysis: {input_path}")
                results = analyzer.process_directory(str(input_path))
            else:
                # Process directory as individual files
                logger.info(f"Processing files individually in directory: {input_path}")
                
                # Get list of files in directory
                processed_documents = analyzer.document_processor.process_directory(str(input_path))
                results = []
                
                for document in processed_documents:
                    try:
                        # Process each document individually
                        result = analyzer.process_document(str(document.path))
                        results.append(result)
                    except Exception as e:
                        logger.error(f"Error processing {document.path}: {str(e)}")
                        continue
        
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
