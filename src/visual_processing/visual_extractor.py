"""
Visual extractor module for the Pharmaceutical Compliance Analysis System.
Uses BLIP to generate captions for images.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Any
from loguru import logger
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

class VisualExtractor:
    """Extracts visual information from images using BLIP."""
    
    def __init__(self, config: Dict = None):
        """Initialize the visual extractor with configuration.
        
        Args:
            config: Configuration dictionary for visual extraction parameters
        """
        self.config = config or {}
        
        # BLIP model configuration
        self.model_name = self.config.get("blip_model", "Salesforce/blip-image-captioning-base")
        self.max_new_tokens = self.config.get("blip_parameters", {}).get("max_new_tokens", 100)
        self.num_beams = self.config.get("blip_parameters", {}).get("num_beams", 4)
        
        # Output configuration
        self.caption_output = Path(self.config.get("caption_output", "output/intermediate_results/visual_captions.json"))
        self.caption_output.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize BLIP model
        logger.info(f"Loading BLIP model: {self.model_name}")
        self.processor = BlipProcessor.from_pretrained(self.model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(self.model_name)
        logger.info("BLIP model loaded successfully")
        
        # Define prompts for different aspects of visual analysis
        self.prompts = self.config.get("prompts", [
            "A detailed description of this pharmaceutical marketing material:",
            "The layout and design of this pharmaceutical advertisement includes:",
            "Colors, imagery, and visual elements in this pharmaceutical material:"
        ])
    
    def extract_captions(self, image_paths: List[Dict]) -> List[Dict]:
        """Generate captions for images using BLIP.
        
        Args:
            image_paths: List of dictionaries containing image information
            
        Returns:
            List of dictionaries containing image captions
        """
        captions_data = []
        
        try:
            for img_info in image_paths:
                image_path = img_info.get("image_path")
                if not image_path or not os.path.exists(image_path):
                    logger.warning(f"Image not found: {image_path}")
                    continue
                
                logger.info(f"Generating captions for image: {image_path}")
                
                try:
                    # Load image
                    image = Image.open(image_path).convert('RGB')
                    
                    # Generate captions with multiple prompts
                    captions = {}
                    for prompt in self.prompts:
                        caption = self._generate_caption(image, prompt)
                        prompt_key = prompt.split(':')[0].strip().lower().replace(' ', '_')
                        captions[prompt_key] = caption
                    
                    # Create caption data
                    caption_data = {
                        "document_id": img_info.get("document_id"),
                        "page_number": img_info.get("page_number"),
                        "image_path": image_path,
                        "image_dimensions": {
                            "width": img_info.get("width"),
                            "height": img_info.get("height")
                        },
                        "captions": captions
                    }
                    
                    captions_data.append(caption_data)
                    logger.debug(f"Generated captions for {image_path}")
                    
                except Exception as e:
                    logger.error(f"Error generating captions for {image_path}: {str(e)}")
            
            # Save captions to file
            self._save_captions(captions_data)
            
            return captions_data
            
        except Exception as e:
            logger.error(f"Error in caption extraction: {str(e)}")
            return captions_data
    
    def _generate_caption(self, image: Image.Image, prompt: str) -> str:
        """Generate a caption for an image with a specific prompt.
        
        Args:
            image: PIL Image object
            prompt: Text prompt for captioning
            
        Returns:
            Generated caption
        """
        try:
            # Prepare inputs
            inputs = self.processor(image, prompt, return_tensors="pt")
            
            # Generate caption
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                num_beams=self.num_beams
            )
            
            # Decode caption
            caption = self.processor.decode(outputs[0], skip_special_tokens=True)
            
            # Remove the prompt from the beginning of the caption if it's there
            if caption.startswith(prompt):
                caption = caption[len(prompt):].strip()
            
            return caption
            
        except Exception as e:
            logger.error(f"Error generating caption: {str(e)}")
            return "Caption generation failed"
    
    def _save_captions(self, captions_data: List[Dict]) -> None:
        """Save captions to a JSON file.
        
        Args:
            captions_data: List of dictionaries containing caption data
        """
        try:
            with open(self.caption_output, 'w', encoding='utf-8') as f:
                json.dump(captions_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved {len(captions_data)} captions to {self.caption_output}")
            
        except Exception as e:
            logger.error(f"Error saving captions to {self.caption_output}: {str(e)}")
    
    def load_captions(self) -> List[Dict]:
        """Load captions from a previously saved JSON file.
        
        Returns:
            List of dictionaries containing caption data
        """
        try:
            if not self.caption_output.exists():
                logger.warning(f"Captions file not found: {self.caption_output}")
                return []
            
            with open(self.caption_output, 'r', encoding='utf-8') as f:
                captions_data = json.load(f)
            
            logger.info(f"Loaded {len(captions_data)} captions from {self.caption_output}")
            return captions_data
            
        except Exception as e:
            logger.error(f"Error loading captions from {self.caption_output}: {str(e)}")
            return [] 