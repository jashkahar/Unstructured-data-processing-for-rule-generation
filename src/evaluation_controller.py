"""
Evaluation controller for the Pharmaceutical Compliance Analysis System.
Assesses documents against previously generated compliance rules.
"""

import os
import json
import yaml
from typing import Dict, List, Any, Optional
from pathlib import Path
from loguru import logger
from dotenv import load_dotenv
from openai import OpenAI
import copy

from document_processing.processor import DocumentProcessor
from document_processing.chunking import ChunkingStrategy
from visual_processing.image_renderer import ImageRenderer
from visual_processing.visual_extractor import VisualExtractor
from utils.types import DocumentType

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
    elif hasattr(obj, 'value') and hasattr(obj, '__class__'):
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
        # Handle ComplianceRule
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


class ComplianceEvaluator:
    """Evaluates documents against previously generated compliance rules."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize the compliance evaluator.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self._setup_logging()
        self._initialize_components()
        self._load_rules()
        logger.info("Initialized ComplianceEvaluator")
        
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
            log_config.get('file', 'logs/compliance_evaluator.log'),
            level=log_config.get('level', 'INFO'),
            rotation=log_config.get('max_file_size', 10485760),
            retention=log_config.get('backup_count', 5)
        )
        
    def _initialize_components(self):
        """Initialize all system components needed for evaluation."""
        try:
            # Document processing
            self.document_processor = DocumentProcessor(self.config.get('document_processing', {}))
            self.chunking_strategy = ChunkingStrategy(self.config.get('document_processing', {}))
            
            # Visual processing if configured
            if 'visual_processing' in self.config:
                logger.info("Initializing visual processing components")
                self.image_renderer = ImageRenderer(self.config.get('visual_processing', {}))
                self.visual_extractor = VisualExtractor(self.config.get('visual_processing', {}))
            else:
                self.image_renderer = None
                self.visual_extractor = None
            
            # Initialize OpenAI client for evaluation
            api_key = os.getenv("OPENAI_API")
            if not api_key:
                raise ValueError("OpenAI API key is required for evaluation. Please set the OPENAI_API_KEY environment variable.")
            
            self.client = OpenAI()
            self.client.api_key = api_key
            
            # Test the API key with a simple request
            try:
                self.client.models.list()
                logger.info("OpenAI API key validated successfully")
            except Exception as e:
                error_message = str(e)
                logger.error(f"Error validating OpenAI API key: {error_message}")
                raise ValueError(f"Invalid OpenAI API key: {error_message}")
                
            logger.info("Initialized all components for evaluation")
        except Exception as e:
            logger.error(f"Error initializing components: {str(e)}")
            raise
    
    def _load_rules(self):
        """Load previously generated rules from output directory."""
        try:
            rules_path = Path(self.config.get('output', {}).get('dir', 'output'), 'analysis_results.json')
            if not rules_path.exists():
                logger.warning(f"Rules file not found at {rules_path}. No rules will be loaded.")
                self.rules = []
                return
                
            with open(rules_path, 'r') as f:
                results = json.load(f)
                
            # Extract rules from analysis results
            self.rules = []
            for document_result in results:
                if 'compliance_rules' in document_result:
                    self.rules.extend(document_result['compliance_rules'])
                    
            logger.info(f"Loaded {len(self.rules)} rules for evaluation")
        except Exception as e:
            logger.error(f"Error loading rules: {str(e)}")
            raise
    
    def evaluate_document(self, document_path: str) -> Dict[str, Any]:
        """Evaluate a single document against the loaded rules.
        
        Args:
            document_path: Path to the document to evaluate
            
        Returns:
            Dictionary containing evaluation results
        """
        try:
            logger.info(f"Evaluating document: {document_path}")
            
            # Parse document
            document = self.document_processor.process_file(document_path)
            if not document:
                logger.error(f"Failed to process document: {document_path}")
                return {"error": f"Failed to process document: {document_path}"}
                
            # Extract and chunk document content
            chunks = self.chunking_strategy.chunk_document(document)
            logger.info(f"Generated {len(chunks)} text chunks")
            
            # Extract visual elements if available
            visual_elements = []
            if self.image_renderer and self.visual_extractor:
                image_list = self._process_document_images(document_path, document.metadata.get('doc_id', 'unknown'))
                for image_info in image_list:
                    if 'caption' in image_info and 'features' in image_info:
                        visual_elements.append({
                            'image_path': image_info.get('path', ''),
                            'caption': image_info.get('caption', ''),
                            'features': image_info.get('features', {})
                        })
                logger.info(f"Extracted {len(visual_elements)} visual elements")
            
            # Prepare data for evaluation
            evaluation_data = {
                'document_path': document_path,
                'document_id': document.metadata.get('doc_id', 'unknown'),
                'text_chunks': convert_to_serializable(chunks),
                'visual_elements': visual_elements,
                'rules': self.rules
            }
            
            # Run evaluation using LLM
            evaluation_results = self._run_evaluation(evaluation_data)
            
            # Save results
            self._save_evaluation_result(evaluation_results)
            
            return evaluation_results
            
        except Exception as e:
            logger.error(f"Error evaluating document: {str(e)}")
            raise
    
    def _process_document_images(self, document_path: str, document_id: str) -> List[Dict[str, Any]]:
        """Process images from the document.
        
        Args:
            document_path: Path to the document
            document_id: Document identifier
            
        Returns:
            List of dictionaries containing image information
        """
        try:
            # Process with image renderer - check method name
            # The method might be named differently than render_document_images
            if hasattr(self.image_renderer, 'render_document_images'):
                rendered_images = self.image_renderer.render_document_images(document_path)
            elif hasattr(self.image_renderer, 'extract_images'):
                rendered_images = self.image_renderer.extract_images(document_path)
            else:
                logger.warning("Image renderer doesn't have expected methods, skipping image processing")
                return []
                
            logger.info(f"Rendered {len(rendered_images)} images from document")
            
            # Extract features and captions
            processed_images = []
            for image_info in rendered_images:
                image_path = image_info.get('path')
                if not image_path or not os.path.exists(image_path):
                    continue
                    
                # Extract visual features
                features = self.visual_extractor.extract_features(image_path)
                
                # Generate image caption
                caption = self.visual_extractor.generate_caption(image_path)
                
                processed_images.append({
                    'path': image_path,
                    'page': image_info.get('page', 0),
                    'features': features,
                    'caption': caption
                })
                
            return processed_images
            
        except Exception as e:
            logger.error(f"Error processing document images: {str(e)}")
            return []
    
    def _run_evaluation(self, evaluation_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run the evaluation using OpenAI.
        
        Args:
            evaluation_data: Data for evaluation
            
        Returns:
            Evaluation results dictionary
        """
        try:
            # Prepare a prompt for the LLM to evaluate compliance
            system_prompt = self._get_evaluation_system_prompt()
            user_prompt = self._create_evaluation_prompt(evaluation_data)
            
            # Call OpenAI API - use evaluation config if available
            eval_config = self.config.get('evaluation', {})
            model = eval_config.get('model', self.config.get('rule_generation', {}).get('model', 'gpt-4'))
            temperature = eval_config.get('temperature', self.config.get('rule_generation', {}).get('temperature', 0.2))
            
            logger.info(f"Using model {model} with temperature {temperature} for evaluation")
            
            # Create request without response_format for compatibility with all models
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            # Only add response_format for models that support it (newer GPT models)
            supported_models = ['gpt-4-turbo', 'gpt-4-1106-preview', 'gpt-4-0125-preview', 'gpt-3.5-turbo-1106']
            if any(supported_model in model for supported_model in supported_models):
                response = self.client.chat.completions.create(
                    model=model,
                    temperature=temperature,
                    messages=messages,
                    response_format={"type": "json_object"}
                )
            else:
                # For models that don't support response_format
                response = self.client.chat.completions.create(
                    model=model,
                    temperature=temperature,
                    messages=messages
                )
            
            # Parse and structure the response
            try:
                evaluation_response = json.loads(response.choices[0].message.content)
                
                # Combine with metadata
                result = {
                    'document_path': evaluation_data['document_path'],
                    'document_id': evaluation_data['document_id'],
                    'evaluation_results': evaluation_response,
                    'rules_count': len(evaluation_data['rules']),
                    'chunks_count': len(evaluation_data['text_chunks']),
                    'visual_elements_count': len(evaluation_data['visual_elements'])
                }
                
                return result
                
            except json.JSONDecodeError:
                logger.error("Failed to parse JSON from evaluation response")
                return {
                    'document_path': evaluation_data['document_path'],
                    'document_id': evaluation_data['document_id'],
                    'error': "Failed to parse evaluation results",
                    'raw_response': response.choices[0].message.content
                }
                
        except Exception as e:
            logger.error(f"Error running evaluation: {str(e)}")
            return {
                'document_path': evaluation_data['document_path'],
                'document_id': evaluation_data['document_id'],
                'error': str(e)
            }
    
    def _get_evaluation_system_prompt(self) -> str:
        """Get the system prompt for the evaluation.
        
        Returns:
            System prompt string
        """
        return """You are a pharmaceutical compliance evaluator specializing in assessing promotional materials against FDA regulations and company policies. Your task is to evaluate a document against a set of compliance rules.

Your evaluation should:

1. Analyze each text chunk and visual element provided from the document
2. For each compliance rule, determine whether the document complies with the rule or violates it
3. Provide specific examples from the document for any violations found
4. Assign a compliance status to each rule: COMPLIANT, MINOR_VIOLATION, MAJOR_VIOLATION, or NOT_APPLICABLE
5. Provide an overall assessment of the document's compliance, including a compliance score and recommended actions

Your response must be a valid JSON object with the following structure:
{
  "overall_assessment": {
    "compliance_score": 0.0-1.0,  // Float representing percentage compliance
    "summary": "Overall assessment of compliance",
    "key_findings": ["List of key findings"],
    "recommended_actions": ["List of recommended actions"]
  },
  "rule_evaluations": [
    {
      "rule_title": "Title of the rule",
      "category": "Rule category",
      "compliance_status": "COMPLIANT|MINOR_VIOLATION|MAJOR_VIOLATION|NOT_APPLICABLE",
      "evidence": ["Specific examples from the document supporting this evaluation"],
      "recommendation": "Recommendation to address the violation (if any)"
    }
  ]
}

Be precise and detailed in your analysis. Base your evaluation solely on the content provided in the document chunks and the rules provided. Do not make assumptions about content not included in the input.
"""
    
    def _create_evaluation_prompt(self, evaluation_data: Dict[str, Any]) -> str:
        """Create the evaluation prompt from the data.
        
        Args:
            evaluation_data: Data for evaluation
            
        Returns:
            Evaluation prompt string
        """
        # Format the rules for better readability
        rules_text = ""
        for i, rule in enumerate(evaluation_data['rules']):
            rules_text += f"Rule {i+1}:\n"
            rules_text += f"Title: {rule.get('title', 'Untitled')}\n"
            rules_text += f"Description: {rule.get('description', 'No description')}\n"
            rules_text += f"Category: {rule.get('category', 'Uncategorized')}\n"
            rules_text += f"Severity: {rule.get('severity', 'MEDIUM')}\n"
            
            # Include examples if available
            if 'examples' in rule and rule['examples']:
                rules_text += "Examples of violations:\n"
                for example in rule['examples']:
                    rules_text += f"- {example}\n"
            
            rules_text += f"Rationale: {rule.get('rationale', 'No rationale provided')}\n\n"
        
        # Format text chunks
        chunks_text = ""
        for i, chunk in enumerate(evaluation_data['text_chunks']):
            chunks_text += f"--- Chunk {i+1} ---\n"
            if 'section_title' in chunk and chunk['section_title']:
                chunks_text += f"Section: {chunk['section_title']}\n"
            if 'page_number' in chunk:
                chunks_text += f"Page: {chunk['page_number']}\n"
            chunks_text += f"Content: {chunk['content']}\n\n"
        
        # Format visual elements
        visuals_text = ""
        for i, visual in enumerate(evaluation_data['visual_elements']):
            visuals_text += f"--- Visual Element {i+1} ---\n"
            if 'caption' in visual:
                visuals_text += f"Caption: {visual['caption']}\n"
            if 'features' in visual:
                visuals_text += "Features:\n"
                for feature_type, value in visual['features'].items():
                    if isinstance(value, list):
                        visuals_text += f"- {feature_type}: {', '.join(value[:5])}"
                        if len(value) > 5:
                            visuals_text += " (and more)"
                        visuals_text += "\n"
                    else:
                        visuals_text += f"- {feature_type}: {value}\n"
            visuals_text += "\n"
        
        # Construct the full prompt
        prompt = f"""Please evaluate the following document against the compliance rules provided.

Document Information:
Path: {evaluation_data['document_path']}
ID: {evaluation_data['document_id']}

COMPLIANCE RULES:
{rules_text}

DOCUMENT CONTENT:

Text Chunks:
{chunks_text}

Visual Elements:
{visuals_text}

Based on the above content, evaluate whether the document complies with each of the rules. Provide a detailed assessment including specific evidence from the document for any violations found.
"""
        
        return prompt
    
    def _save_evaluation_result(self, evaluation_result: Dict[str, Any]) -> None:
        """Save evaluation result to file.
        
        Args:
            evaluation_result: Evaluation result to save
        """
        try:
            # Get output path from config
            output_file = self.config.get('evaluation', {}).get('output_file', 'output/evaluation_results.json')
            output_path = Path(output_file)
            
            # Create output directory if it doesn't exist
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Check if file exists
            if output_path.exists():
                # Load existing results
                with open(output_path, 'r') as f:
                    try:
                        existing_results = json.load(f)
                    except json.JSONDecodeError:
                        existing_results = []
                
                # If existing results is not a list, convert it to a list
                if not isinstance(existing_results, list):
                    existing_results = [existing_results]
                
                # Append new result
                existing_results.append(evaluation_result)
                results_to_save = existing_results
            else:
                # Create new results list
                results_to_save = [evaluation_result]
            
            # Save results
            with open(output_path, 'w') as f:
                json.dump(results_to_save, f, indent=2)
                
            logger.info(f"Saved evaluation result to {output_path}")
            
        except Exception as e:
            logger.error(f"Error saving evaluation result: {str(e)}")
            
    def evaluate_directory(self, directory_path: str = None) -> List[Dict[str, Any]]:
        """Evaluate all documents in a directory.
        
        Args:
            directory_path: Path to directory containing documents.
                           If None, uses the evaluation_materials directory from config.
            
        Returns:
            List of evaluation results
        """
        try:
            # If no directory specified, use the one from config
            if directory_path is None:
                directory_path = self.config.get('evaluation', {}).get('materials_dir', 
                                 self.config.get('paths', {}).get('evaluation_materials', 'evaluation_materials'))
            
            logger.info(f"Evaluating documents in directory: {directory_path}")
            
            # Get list of supported file extensions
            supported_extensions = self.config.get('document_processing', {}).get('supported_extensions', ['.pdf', '.docx', '.pptx', '.jpg', '.png'])
            
            # Find all supported files in directory
            directory = Path(directory_path)
            if not directory.exists():
                logger.error(f"Directory {directory_path} does not exist")
                return [{"error": f"Directory {directory_path} does not exist"}]
                
            document_paths = []
            for ext in supported_extensions:
                document_paths.extend(list(directory.glob(f"**/*{ext}")))
            
            logger.info(f"Found {len(document_paths)} documents to evaluate")
            
            # Process each document
            evaluation_results = []
            for doc_path in document_paths:
                try:
                    logger.info(f"Evaluating document: {doc_path}")
                    result = self.evaluate_document(str(doc_path))
                    evaluation_results.append(result)
                except Exception as e:
                    logger.error(f"Error evaluating document {doc_path}: {str(e)}")
                    evaluation_results.append({
                        'document_path': str(doc_path),
                        'error': str(e)
                    })
            
            return evaluation_results
            
        except Exception as e:
            logger.error(f"Error evaluating directory: {str(e)}")
            raise


def main():
    """Main entry point for the Compliance Evaluator."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Pharmaceutical Compliance Evaluator")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to configuration file")
    parser.add_argument("--document", type=str, help="Path to document to evaluate")
    parser.add_argument("--directory", type=str, help="Path to directory of documents to evaluate")
    parser.add_argument("--use-config-dir", action="store_true", 
                        help="Use the evaluation_materials directory specified in the config file")
    
    args = parser.parse_args()
    
    evaluator = ComplianceEvaluator(config_path=args.config)
    
    if args.document:
        logger.info(f"Evaluating single document: {args.document}")
        result = evaluator.evaluate_document(args.document)
        logger.info(f"Evaluation completed for {args.document}")
    elif args.directory:
        logger.info(f"Evaluating directory: {args.directory}")
        results = evaluator.evaluate_directory(args.directory)
        logger.info(f"Evaluation completed for {len(results)} documents in {args.directory}")
    elif args.use_config_dir:
        logger.info("Using evaluation_materials directory from config")
        results = evaluator.evaluate_directory()
        logger.info(f"Evaluation completed for {len(results)} documents in config-specified directory")
    else:
        # Default to using config directory
        logger.info("No document or directory specified, using evaluation_materials directory from config")
        results = evaluator.evaluate_directory()
        logger.info(f"Evaluation completed for {len(results)} documents in config-specified directory")

if __name__ == "__main__":
    main() 