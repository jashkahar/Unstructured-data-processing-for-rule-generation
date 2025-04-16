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
import numpy as np

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
    
    def generate_caption(self, image_path: str) -> str:
        """Generate a single caption for an image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Generated caption as a string
        """
        try:
            logger.debug(f"Generating caption for image: {image_path}")
            
            # Load image
            image = Image.open(image_path).convert('RGB')
            
            # Use the first prompt as default
            default_prompt = self.prompts[0] if self.prompts else "Describe this pharmaceutical image:"
            
            # Generate caption
            caption = self._generate_caption(image, default_prompt)
            
            logger.debug(f"Generated caption: {caption[:50]}...")
            return caption
            
        except Exception as e:
            logger.error(f"Error generating caption for {image_path}: {str(e)}")
            return f"Caption generation failed: {str(e)}"
    
    def extract_features(self, image_path: str) -> Dict[str, Any]:
        """Extract visual features from an image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary containing visual features
        """
        try:
            logger.debug(f"Extracting features from image: {image_path}")
            
            # Load image
            image = Image.open(image_path).convert('RGB')
            
            # Extract basic image properties
            features = {
                "width": image.width,
                "height": image.height,
                "aspect_ratio": round(image.width / image.height, 2) if image.height > 0 else 0,
                "format": image.format or Path(image_path).suffix[1:].upper(),
                "mode": image.mode
            }
            
            # Extract color information
            try:
                # Calculate dominant colors
                dominant_colors = self._extract_dominant_colors(image)
                features["dominant_colors"] = dominant_colors
                
                # Calculate color balance
                brightness, saturation = self._calculate_color_stats(image)
                features["brightness"] = brightness
                features["saturation"] = saturation
            except Exception as e:
                logger.error(f"Error extracting color features: {str(e)}")
            
            # Generate captions for multiple aspects as features
            try:
                captions = {}
                for prompt in self.prompts:
                    caption = self._generate_caption(image, prompt)
                    prompt_key = prompt.split(':')[0].strip().lower().replace(' ', '_')
                    captions[prompt_key] = caption
                features["captions"] = captions
            except Exception as e:
                logger.error(f"Error generating caption features: {str(e)}")
            
            logger.debug(f"Extracted {len(features)} feature types from {image_path}")
            return features
            
        except Exception as e:
            logger.error(f"Error extracting features from {image_path}: {str(e)}")
            return {
                "error": str(e),
                "file_path": image_path
            }
    
    def _extract_dominant_colors(self, image: Image.Image, num_colors: int = 5) -> List[str]:
        """Extract dominant colors from an image.
        
        Args:
            image: PIL Image object
            num_colors: Number of dominant colors to extract
            
        Returns:
            List of dominant colors as hex strings
        """
        # Resize image to speed up processing
        img_small = image.resize((100, 100))
        
        # Convert to RGB if not already
        if img_small.mode != 'RGB':
            img_small = img_small.convert('RGB')
        
        # Get pixels
        pixels = np.array(img_small)
        pixels = pixels.reshape(-1, 3)
        
        # Use simple clustering of pixels
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=num_colors)
        kmeans.fit(pixels)
        
        # Get the colors
        colors = kmeans.cluster_centers_.astype(int)
        
        # Convert to hex
        hex_colors = []
        for color in colors:
            hex_color = '#{:02x}{:02x}{:02x}'.format(color[0], color[1], color[2])
            hex_colors.append(hex_color)
        
        return hex_colors
    
    def _calculate_color_stats(self, image: Image.Image) -> tuple:
        """Calculate color statistics of an image.
        
        Args:
            image: PIL Image object
            
        Returns:
            Tuple of (brightness, saturation)
        """
        # Convert to HSV for better color analysis
        try:
            hsv_image = image.convert('HSV')
            # Get pixels
            pixels = np.array(hsv_image)
            # Extract HSV channels (hue, saturation, value)
            h, s, v = pixels[:,:,0], pixels[:,:,1], pixels[:,:,2]
            
            # Calculate average brightness (value) and saturation
            brightness = float(np.mean(v) / 255)
            saturation = float(np.mean(s) / 255)
            
            return round(brightness, 2), round(saturation, 2)
        except Exception:
            # Simple fallback using RGB
            pixels = np.array(image)
            brightness = float(np.mean(pixels) / 255)
            return round(brightness, 2), 0.0 