"""
Image renderer module for the Pharmaceutical Compliance Analysis System.
Converts PDF pages to images for visual analysis.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional
from loguru import logger
from pdf2image import convert_from_path
from PIL import Image

class ImageRenderer:
    """Converts PDF pages to images for visual analysis."""
    
    def __init__(self, config: Dict = None):
        """Initialize the image renderer with configuration.
        
        Args:
            config: Configuration dictionary for image rendering parameters
        """
        self.config = config or {}
        
        # Set up image rendering parameters
        self.output_dir = Path(self.config.get("tmp_image_dir", "output/tmp_images"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_pages = self.config.get("max_images_per_document", 20)
        self.dpi = self.config.get("dpi", 200)
        self.image_format = self.config.get("image_format", "PNG")
        
        logger.info(f"Initialized ImageRenderer with output directory: {self.output_dir}")
    
    def render_pdf(self, pdf_path: str) -> List[Dict]:
        """Convert PDF pages to images and save them.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of dictionaries containing image information (document_id, page_number, image_path)
        """
        try:
            pdf_id = Path(pdf_path).stem
            logger.info(f"Converting PDF to images: {pdf_path}")
            
            # Convert PDF to images
            images = convert_from_path(
                pdf_path,
                dpi=self.dpi,
                first_page=1,
                last_page=self.max_pages
            )
            
            # Save images and collect metadata
            image_paths = []
            for i, image in enumerate(images):
                # Create filename and path
                img_filename = f"{pdf_id}_page_{i+1}.{self.image_format.lower()}"
                img_path = self.output_dir / img_filename
                
                # Save the image
                image.save(str(img_path), self.image_format)
                
                # Collect metadata
                image_metadata = {
                    "document_id": pdf_id,
                    "page_number": i+1,
                    "image_path": str(img_path),
                    "width": image.width,
                    "height": image.height,
                    "format": self.image_format
                }
                
                image_paths.append(image_metadata)
            
            logger.info(f"Converted {len(image_paths)} pages from {pdf_path}")
            return image_paths
            
        except Exception as e:
            logger.error(f"Error rendering PDF {pdf_path}: {str(e)}")
            raise
    
    def process_standalone_image(self, image_path: str) -> Dict:
        """Process a standalone image for visual analysis.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary containing image information
        """
        try:
            image_id = Path(image_path).stem
            logger.info(f"Processing standalone image: {image_path}")
            
            # Get image metadata
            image = Image.open(image_path)
            
            # Create metadata similar to PDF page images
            image_metadata = {
                "document_id": image_id,
                "page_number": 1,  # Single image is page 1
                "image_path": image_path,  # Use original path
                "width": image.width,
                "height": image.height,
                "format": image.format if hasattr(image, 'format') else Path(image_path).suffix[1:].upper()
            }
            
            logger.info(f"Processed standalone image: {image_path}")
            return image_metadata
            
        except Exception as e:
            logger.error(f"Error processing standalone image {image_path}: {str(e)}")
            raise
    
    def clean_up_images(self, image_paths: Optional[List[Dict]] = None) -> None:
        """Clean up temporary images.
        
        Args:
            image_paths: List of image metadata dictionaries. If None, clean all images in output directory.
        """
        try:
            if image_paths:
                # Delete specific images, but only if they are in the temporary directory
                for img_info in image_paths:
                    img_path = img_info.get("image_path")
                    # Only delete if it's in our temporary directory
                    if img_path and os.path.exists(img_path) and str(self.output_dir) in img_path:
                        os.remove(img_path)
                logger.info(f"Cleaned up temporary images")
            else:
                # Delete all images in the output directory
                count = 0
                for img_file in self.output_dir.glob(f"*.{self.image_format.lower()}"):
                    os.remove(img_file)
                    count += 1
                logger.info(f"Cleaned up {count} images from {self.output_dir}")
                
        except Exception as e:
            logger.error(f"Error cleaning up images: {str(e)}") 