"""
Evaluation controller for the Pharmaceutical Compliance Analysis System.
Assesses documents against previously generated compliance rules.
"""

import os
import json
import yaml
import sys
from typing import Dict, List, Any, Optional
from pathlib import Path
from loguru import logger
from dotenv import load_dotenv
from openai import OpenAI
import copy

# Import document processing components
try:
    from document_processing.processor import DocumentProcessor
    from document_processing.chunking import ChunkingStrategy
    document_processing_available = True
except ImportError:
    logger.warning("Document processing imports failed, using fallback methods")
    document_processing_available = False

# Import visual processing components
try:
    from visual_processing.image_renderer import ImageRenderer
    from visual_processing.visual_extractor import VisualExtractor
    visual_processing_available = True
except ImportError:
    logger.warning("Visual processing imports failed, visual analysis will be disabled")
    visual_processing_available = False

# Import utils
try:
    from utils.types import DocumentType
    utils_available = True
except ImportError:
    logger.warning("Utils imports failed")
    utils_available = False

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
        """Initialize all system components."""
        try:
            # Initialize document processing if available
            if document_processing_available:
                try:
                    self.document_processor = DocumentProcessor(self.config.get('document_processing', {}))
                    self.chunking_strategy = ChunkingStrategy(self.config.get('document_processing', {}))
                    logger.info("Initialized document processing components")
                except Exception as e:
                    logger.error(f"Error initializing document processing: {str(e)}")
                    self.document_processor = None
                    self.chunking_strategy = None
            else:
                self.document_processor = None
                self.chunking_strategy = None
                logger.info("Document processing components not available")
            
            # Initialize visual processing if available
            if visual_processing_available and 'visual_processing' in self.config:
                try:
                    self.image_renderer = ImageRenderer(self.config.get('visual_processing', {}))
                    self.visual_extractor = VisualExtractor(self.config.get('visual_processing', {}))
                    logger.info("Initialized visual processing components")
                except Exception as e:
                    logger.error(f"Error initializing visual processing: {str(e)}")
                    self.image_renderer = None
                    self.visual_extractor = None
            else:
                self.image_renderer = None
                self.visual_extractor = None
                logger.info("Visual processing components not available")
            
            # Initialize OpenAI client
            self._initialize_openai()
            
        except Exception as e:
            logger.error(f"Error initializing components: {str(e)}")
            raise
     
    def _initialize_openai(self):
        """Initialize OpenAI client for evaluation."""
        try:
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
                
            logger.info("Initialized OpenAI client for evaluation")
        except Exception as e:
            logger.error(f"Error initializing OpenAI client: {str(e)}")
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
            document_id = Path(document_path).stem
            
            # Process document based on available components
            text_content = ""
            visual_elements = []
            
            # Try document processor if available
            if document_processing_available and self.document_processor and self.chunking_strategy:
                try:
                    document = self.document_processor.process_file(document_path)
                    if document:
                        # Extract document ID from metadata if available
                        if hasattr(document, 'metadata') and hasattr(document.metadata, 'get'):
                            document_id = document.metadata.get('doc_id', document_id)
                        
                        # Extract and chunk document content
                        chunks = self.chunking_strategy.chunk_document(document)
                        logger.info(f"Generated {len(chunks)} text chunks")
                        
                        # Combine chunks into text content
                        for chunk in chunks:
                            if hasattr(chunk, 'content'):
                                text_content += chunk.content + "\n\n"
                            elif isinstance(chunk, dict) and 'content' in chunk:
                                text_content += chunk['content'] + "\n\n"
                        
                        # Process images if visual processing is available
                        if visual_processing_available and self.image_renderer and self.visual_extractor:
                            try:
                                image_list = self._process_document_images(document_path, document_id)
                                visual_elements = image_list
                                logger.info(f"Extracted {len(visual_elements)} visual elements")
                            except Exception as e:
                                logger.error(f"Error processing visual elements: {str(e)}")
                                logger.info("Continuing evaluation without visual elements")
                except Exception as e:
                    logger.error(f"Error using document processor: {str(e)}")
                    logger.info("Falling back to basic text extraction")
            
            # If no text content was extracted, fall back to basic extraction
            if not text_content:
                text_content = self._extract_text_from_document(document_path)
                logger.info(f"Extracted {len(text_content)} characters using fallback method")
            
            # If document processing failed, try visual processing directly
            if not visual_elements and visual_processing_available and self.image_renderer and self.visual_extractor:
                try:
                    visual_elements = self._process_document_images(document_path, document_id)
                    logger.info(f"Extracted {len(visual_elements)} visual elements using direct method")
                except Exception as e:
                    logger.error(f"Error processing visual elements directly: {str(e)}")
            
            # Prepare data for evaluation
            evaluation_data = {
                'document_path': document_path,
                'document_id': document_id,
                'document_text': text_content,
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
            return {"error": str(e), "document_path": document_path}
    
    def _extract_text_from_document(self, document_path: str) -> str:
        """Extract text from a document.
        
        Args:
            document_path: Path to the document
            
        Returns:
            Extracted text
        """
        # For PDF files
        if document_path.lower().endswith('.pdf'):
            try:
                # Try using PyPDF2 if available
                try:
                    import PyPDF2
                    with open(document_path, 'rb') as file:
                        reader = PyPDF2.PdfReader(file)
                        text = ""
                        for page in reader.pages:
                            text += page.extract_text() + "\n\n"
                        logger.info(f"Extracted text from PDF using PyPDF2: {len(text)} characters")
                        return text
                except ImportError:
                    logger.warning("PyPDF2 not available, trying other methods")
                
                # Try using pdfplumber if available
                try:
                    import pdfplumber
                    with pdfplumber.open(document_path) as pdf:
                        text = ""
                        for page in pdf.pages:
                            text += page.extract_text() + "\n\n"
                        logger.info(f"Extracted text from PDF using pdfplumber: {len(text)} characters")
                        return text
                except ImportError:
                    logger.warning("pdfplumber not available, using fallback method")
                
                # Simple fallback that just returns metadata
                file_stats = os.stat(document_path)
                return f"PDF document: {Path(document_path).name}\nSize: {file_stats.st_size} bytes\nUnable to extract content."
                    
            except Exception as e:
                logger.error(f"Error extracting text from PDF: {str(e)}")
                return f"Error extracting text: {str(e)}"
        
        # For text files
        elif document_path.lower().endswith(('.txt', '.md')):
            try:
                with open(document_path, 'r', encoding='utf-8') as file:
                    text = file.read()
                    logger.info(f"Extracted text from text file: {len(text)} characters")
                    return text
            except Exception as e:
                logger.error(f"Error reading text file: {str(e)}")
                return f"Error reading text file: {str(e)}"
        
        # For other file types
        else:
            logger.warning(f"Unsupported file type for text extraction: {document_path}")
            return f"Document: {Path(document_path).name}\nUnsupported file type for text extraction."
    
    def _process_document_images(self, document_path: str, document_id: str) -> List[Dict[str, Any]]:
        """Process images from the document.
        
        Args:
            document_path: Path to the document
            document_id: Document identifier
            
        Returns:
            List of dictionaries containing image information
        """
        try:
            # Check if visual processing is available
            if not visual_processing_available or not self.image_renderer or not self.visual_extractor:
                logger.warning("Visual processing not available")
                return []
            
            # Process the document based on its type
            rendered_images = []
            
            # For PDF files, use render_pdf method
            if document_path.lower().endswith('.pdf'):
                try:
                    if hasattr(self.image_renderer, 'render_pdf'):
                        rendered_images = self.image_renderer.render_pdf(document_path)
                        logger.info(f"Used render_pdf method, found {len(rendered_images)} images")
                    else:
                        logger.warning("ImageRenderer doesn't have render_pdf method for PDF files")
                except Exception as e:
                    logger.error(f"Error rendering PDF: {str(e)}")
            
            # For standalone image files, use process_standalone_image method
            elif document_path.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp')):
                try:
                    if hasattr(self.image_renderer, 'process_standalone_image'):
                        single_image = self.image_renderer.process_standalone_image(document_path)
                        rendered_images = [single_image]
                        logger.info(f"Used process_standalone_image method for image file")
                    else:
                        logger.warning("ImageRenderer doesn't have process_standalone_image method for image files")
                        # Fallback: create a simple image info dictionary
                        rendered_images = [{
                            'document_id': document_id,
                            'page_number': 1,
                            'image_path': document_path
                        }]
                except Exception as e:
                    logger.error(f"Error processing image file: {str(e)}")
            
            # For other file types (like Word docs), we may not have direct support
            else:
                logger.warning(f"No direct image processing method for file type: {Path(document_path).suffix}")
                
            if not rendered_images:
                logger.info("No images found in document")
                return []
                
            # Extract features and captions from the rendered images
            processed_images = []
            for image_info in rendered_images:
                # Get the image path from the metadata
                image_path = None
                if isinstance(image_info, dict):
                    image_path = image_info.get('image_path') or image_info.get('path')
                else:
                    # Try to handle non-dictionary objects
                    if hasattr(image_info, 'image_path'):
                        image_path = image_info.image_path
                    elif hasattr(image_info, 'path'):
                        image_path = image_info.path
                
                if not image_path or not os.path.exists(image_path):
                    logger.warning(f"Image path invalid or not found: {image_path}")
                    continue
                
                # Get the page number if available
                page_num = 1
                if isinstance(image_info, dict) and 'page_number' in image_info:
                    page_num = image_info.get('page_number')
                elif hasattr(image_info, 'page_number'):
                    page_num = image_info.page_number
                
                # Extract visual features
                features = {}
                try:
                    if hasattr(self.visual_extractor, 'extract_features'):
                        features = self.visual_extractor.extract_features(image_path)
                    else:
                        logger.warning("VisualExtractor doesn't have extract_features method")
                        # Use basic image properties as features
                        try:
                            from PIL import Image
                            img = Image.open(image_path)
                            features = {
                                'width': img.width,
                                'height': img.height,
                                'format': img.format,
                                'mode': img.mode
                            }
                        except:
                            pass
                except Exception as e:
                    logger.error(f"Error extracting features from image {image_path}: {str(e)}")
                
                # Generate image caption
                caption = ""
                try:
                    if hasattr(self.visual_extractor, 'generate_caption'):
                        caption = self.visual_extractor.generate_caption(image_path)
                    else:
                        logger.warning("VisualExtractor doesn't have generate_caption method")
                        caption = f"Image from page {page_num}" 
                except Exception as e:
                    logger.error(f"Error generating caption for image {image_path}: {str(e)}")
                    caption = f"Image from page {page_num} (caption generation failed)"
                
                # Add the processed image information
                processed_images.append({
                    'path': image_path,
                    'page': page_num,
                    'features': features,
                    'caption': caption
                })
            
            logger.info(f"Successfully processed {len(processed_images)} images from document")
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
            
            # Format visual elements
            visuals_text = ""
            if evaluation_data.get('visual_elements'):
                for i, visual in enumerate(evaluation_data['visual_elements']):
                    visuals_text += f"--- Visual Element {i+1} ---\n"
                    if 'caption' in visual:
                        visuals_text += f"Caption: {visual['caption']}\n"
                    if 'features' in visual:
                        visuals_text += "Features:\n"
                        for feature_type, value in visual['features'].items():
                            if isinstance(value, list):
                                visuals_text += f"- {feature_type}: {', '.join([str(v) for v in value[:5]])}"
                                if len(value) > 5:
                                    visuals_text += " (and more)"
                                visuals_text += "\n"
                            else:
                                visuals_text += f"- {feature_type}: {value}\n"
                    visuals_text += "\n"
            
            # Prepare system prompt
            system_prompt = self._get_evaluation_system_prompt()
            
            # Create user prompt
            user_prompt = f"""Please evaluate the following document against the compliance rules provided.

Document Information:
Path: {evaluation_data['document_path']}
ID: {evaluation_data['document_id']}

COMPLIANCE RULES:
{rules_text}

DOCUMENT CONTENT:
{evaluation_data['document_text'][:15000]}

"""
            # Add visual elements if available
            if visuals_text:
                user_prompt += f"\nVISUAL ELEMENTS:\n{visuals_text}\n"
                
            user_prompt += """
Based on the above content, evaluate whether the document complies with each of the rules. Provide a detailed assessment including specific evidence from the document for any violations found.
"""

            # Call OpenAI API - use evaluation config if available
            eval_config = self.config.get('evaluation', {})
            model = eval_config.get('model', self.config.get('rule_generation', {}).get('model', 'gpt-3.5-turbo'))
            temperature = eval_config.get('temperature', self.config.get('rule_generation', {}).get('temperature', 0.2))
            
            logger.info(f"Using model {model} with temperature {temperature} for evaluation")
            
            # Create request without response_format for compatibility with all models
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            # Call OpenAI API
            response = self.client.chat.completions.create(
                model=model,
                temperature=temperature,
                messages=messages
            )
            
            # Parse and structure the response
            try:
                content = response.choices[0].message.content
                # Find JSON part in the response
                start_idx = content.find('{')
                end_idx = content.rfind('}') + 1
                
                if start_idx != -1 and end_idx != -1:
                    json_str = content[start_idx:end_idx]
                    evaluation_response = json.loads(json_str)
                else:
                    # If no JSON found, try to parse the whole content
                    evaluation_response = json.loads(content)
                
                # Combine with metadata
                result = {
                    'document_path': evaluation_data['document_path'],
                    'document_id': evaluation_data['document_id'],
                    'evaluation_results': evaluation_response,
                    'rules_count': len(evaluation_data['rules']),
                    'visual_elements_count': len(evaluation_data.get('visual_elements', []))
                }
                
                logger.info(f"Successfully evaluated document: {evaluation_data['document_path']}")
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

1. Analyze the text and visual elements provided from the document
2. For each compliance rule, determine whether the document complies with the rule or violates it
3. Provide specific examples from the document for any violations found
4. Assign a compliance status to each rule: COMPLIANT, MINOR_VIOLATION, MAJOR_VIOLATION, or NOT_APPLICABLE
5. Provide an overall assessment of the document's compliance, including a compliance score and recommended actions

Your response must be a valid JSON object with the following structure:
{
  "overall_assessment": {
    "compliance_score": 0.0-1.0,
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

Be precise and detailed in your analysis. Base your evaluation solely on the content provided in the document and the rules provided. Do not make assumptions about content not included in the input.
"""
    
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
            supported_extensions = ['.pdf', '.docx', '.pptx', '.txt', '.md', '.jpg', '.png']
            
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