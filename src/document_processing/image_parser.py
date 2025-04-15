"""
Image Parser module for the Pharmaceutical Compliance Analysis System.
Uses OCR to extract text from images.
"""

import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import pytesseract
from PIL import Image
from loguru import logger

from utils.types import Document, DocumentType


class ImageParser:
    """Parses images to extract text via OCR."""

    def __init__(self, config: Dict = None):
        """Initialize the image parser with configuration.

        Args:
            config: Configuration dictionary for the image parser
        """
        self.config = config or {}
        
        # Set pytesseract path if specified in config
        if "tesseract_path" in self.config:
            pytesseract.pytesseract.tesseract_cmd = self.config["tesseract_path"]
            
    def parse(self, image_path: str) -> Document:
        """Parse an image file and extract text via OCR.

        Args:
            image_path: Path to the image file

        Returns:
            Document object containing extracted content and metadata
        """
        logger.info(f"Parsing image file: {image_path}")
        
        path = Path(image_path)
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"Image file not found: {image_path}")
            
        try:
            # Extract basic file metadata
            file_size = path.stat().st_size
            file_id = self._generate_document_id(path)
            
            # Parse image content using OCR
            image = Image.open(path)
            
            # Get image metadata
            width, height = image.size
            format_name = image.format
            mode = image.mode
            
            # Perform OCR
            ocr_config = self.config.get("ocr_config", "")
            extracted_text = pytesseract.image_to_string(image, config=ocr_config)
            
            # Combine metadata
            metadata = {
                "format": "IMAGE",
                "image_format": format_name,
                "width": width,
                "height": height,
                "color_mode": mode,
                "timestamp": datetime.now().isoformat()
            }
            
            # Create document object
            document = Document(
                id=file_id,
                path=path,
                doc_type=DocumentType.IMAGE,
                content=extracted_text,
                metadata=metadata,
                processed_date=datetime.now(),
                file_size=file_size
            )
            
            return document
            
        except Exception as e:
            logger.error(f"Error parsing image {image_path}: {str(e)}")
            raise
    
    def _generate_document_id(self, path: Path) -> str:
        """Generate a unique document ID based on the filename.

        Args:
            path: Path to the image file

        Returns:
            String ID for the document
        """
        # Extract base name without extension
        base_name = path.stem
        
        # Add timestamp suffix for uniqueness
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        
        return f"IMG_{base_name}_{timestamp}"
