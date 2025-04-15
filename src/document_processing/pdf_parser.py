"""
PDF Parser module for the Pharmaceutical Compliance Analysis System.
Extracts text, headings, and metadata from PDF files using Unstructured.
"""

import os
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set, Any
import hashlib

from loguru import logger
from unstructured.partition.pdf import partition_pdf
from unstructured.documents.elements import Title, NarrativeText, ListItem, Table, Image
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer, LTImage, LTFigure, LAParams
from pydantic import BaseModel, Field

from utils.types import Document, DocumentType
from document_processing.chunking import ChunkingStrategy, Chunk

class PDFValidationError(Exception):
    """Custom exception for PDF validation errors."""
    pass

class PDFParser:
    """Parses PDF files to extract text, structure, and metadata using Unstructured."""

    def __init__(self, config: Dict = None):
        """Initialize the PDF parser with configuration.

        Args:
            config: Configuration dictionary for the PDF parser
        """
        self.config = config or {}
        self.chunking_strategy = ChunkingStrategy(self.config.get("chunking", {}))
        
        # Validation settings
        self.max_file_size_mb = self.config.get("max_file_size_mb", 50)
        self.min_page_count = self.config.get("min_page_count", 1)
        self.max_page_count = self.config.get("max_page_count", 1000)
        self.required_metadata = self.config.get("required_metadata", [])
        
        # Enhanced layout settings
        self.preserve_layout = self.config.get("preserve_layout", True)
        self.extract_images = self.config.get("extract_images", True)
        self.infer_table_structure = self.config.get("infer_table_structure", True)
        
        # Initialize Unstructured with enhanced settings
        self.unstructured_config = {
            "strategy": self.config.get("strategy", "hi_res"),
            "extract_images_in_pdf": self.extract_images,
            "infer_table_structure": self.infer_table_structure,
            "include_page_breaks": True,
            "include_metadata": True
        }

    def parse(self, pdf_path: str) -> Document:
        """Parse a PDF file and extract its content.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            Document object containing extracted content and metadata
        """
        logger.info(f"Parsing PDF file: {pdf_path}")
        
        path = Path(pdf_path)
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
            
        try:
            # Validate PDF
            self._validate_pdf(path)
            
            # Extract basic file metadata
            file_size = path.stat().st_size
            file_id = self._generate_document_id(path)
            
            # Extract detailed metadata
            metadata = self._extract_metadata(path)
            
            # Parse PDF content using Unstructured
            content, sections, page_count, pdf_metadata = self._extract_pdf_content(path)
            
            # Extract tables and figures
            tables, figures = self._extract_tables_and_figures(path)
            
            # Process sections into chunks
            processed_sections = []
            for section in sections:
                # Get page number for the section if available
                page_number = section.get("page_number")
                
                # Chunk the section content
                chunks = self.chunking_strategy.chunk_section(
                    section["title"],
                    section["content"],
                    page_number
                )
                
                # Add chunks to processed section
                processed_section = {
                    "title": section["title"],
                    "chunks": chunks  # Keep chunks as Chunk objects
                }
                processed_sections.append(processed_section)
            
            # Create document object using the imported Document class
            document = Document(
                id=file_id,
                path=path,
                doc_type=DocumentType.PDF,
                content=content,
                metadata={
                    **metadata,
                    **pdf_metadata,
                    "file_size": file_size,
                    "page_count": page_count,
                    "sections": processed_sections,
                    "tables": tables,
                    "figures": figures
                },
                processed_date=datetime.now(),
                file_size=file_size,
                page_count=page_count
            )
            
            logger.info(f"Successfully parsed PDF: {pdf_path}")
            return document
            
        except Exception as e:
            logger.error(f"Error parsing PDF {pdf_path}: {str(e)}")
            raise
    
    def _validate_pdf(self, path: Path) -> None:
        """Validate PDF file before processing.
        
        Args:
            path: Path to the PDF file
            
        Raises:
            PDFValidationError if validation fails
        """
        # Check file size
        file_size_mb = path.stat().st_size / (1024 * 1024)
        if file_size_mb > self.max_file_size_mb:
            raise PDFValidationError(
                f"PDF file size ({file_size_mb:.2f} MB) exceeds maximum allowed "
                f"size ({self.max_file_size_mb} MB)"
            )
        
        # Check if file is a valid PDF
        try:
            with open(path, 'rb') as f:
                header = f.read(4)
                if header != b'%PDF':
                    raise PDFValidationError("File is not a valid PDF")
        except Exception as e:
            raise PDFValidationError(f"Error validating PDF file: {str(e)}")
        
        # Check page count
        try:
            page_count = len(list(extract_pages(str(path))))
            if page_count < self.min_page_count:
                raise PDFValidationError(
                    f"PDF has fewer pages ({page_count}) than minimum required "
                    f"({self.min_page_count})"
                )
            if page_count > self.max_page_count:
                raise PDFValidationError(
                    f"PDF has more pages ({page_count}) than maximum allowed "
                    f"({self.max_page_count})"
                )
        except Exception as e:
            raise PDFValidationError(f"Error checking page count: {str(e)}")
    
    def _extract_metadata(self, path: Path) -> Dict:
        """Extract metadata from PDF file.
        
        Args:
            path: Path to the PDF file
            
        Returns:
            Dictionary containing extracted metadata
        """
        metadata = {}
        
        try:
            # Extract basic file metadata
            metadata["filename"] = path.name
            metadata["file_size"] = path.stat().st_size
            metadata["created_date"] = datetime.fromtimestamp(path.stat().st_ctime).isoformat()
            metadata["modified_date"] = datetime.fromtimestamp(path.stat().st_mtime).isoformat()
            
            # Calculate file hash
            with open(path, 'rb') as f:
                file_hash = hashlib.sha256(f.read()).hexdigest()
            metadata["file_hash"] = file_hash
            
            # Extract PDF-specific metadata using pdfminer
            laparams = LAParams()
            for page in extract_pages(str(path), laparams=laparams):
                # Extract text containers
                text_blocks = []
                for obj in page:
                    if isinstance(obj, LTTextContainer):
                        text_blocks.append({
                            "bbox": obj.bbox,
                            "text": obj.get_text().strip()
                        })
                
                # Use first text block for potential title
                if text_blocks:
                    first_text = text_blocks[0]["text"]
                    if len(first_text) < 100:  # Assume short text is a title
                        metadata["title"] = first_text
                
                # Detect tables using text block analysis
                potential_tables = self._detect_tables_from_text_blocks(text_blocks)
                if potential_tables:
                    metadata["has_tables"] = True
                
                # Extract figures
                figures = [obj for obj in page if isinstance(obj, LTFigure)]
                if figures:
                    metadata["has_figures"] = True
                
                # Extract images
                images = [obj for obj in page if isinstance(obj, LTImage)]
                if images:
                    metadata["has_images"] = True
            
            # Validate required metadata
            missing_metadata = [
                field for field in self.required_metadata 
                if field not in metadata
            ]
            if missing_metadata:
                logger.warning(
                    f"Missing required metadata fields: {missing_metadata}"
                )
            
        except Exception as e:
            logger.error(f"Error extracting metadata: {str(e)}")
        
        return metadata
    
    def _extract_tables_and_figures(self, path: Path) -> Tuple[List[Dict], List[Dict]]:
        """Extract tables and figures from PDF.
        
        Args:
            path: Path to the PDF file
            
        Returns:
            Tuple of (tables, figures) lists
        """
        tables = []
        figures = []
        
        try:
            laparams = LAParams()
            for page_num, page in enumerate(extract_pages(str(path), laparams=laparams), 1):
                # Extract tables using text layout analysis
                text_blocks = []
                for obj in page:
                    if isinstance(obj, LTTextContainer):
                        text_blocks.append({
                            "bbox": obj.bbox,
                            "text": obj.get_text().strip()
                        })
                
                # Simple table detection based on text block alignment
                potential_tables = self._detect_tables_from_text_blocks(text_blocks)
                tables.extend([{
                    "page": page_num,
                    "bbox": table["bbox"],
                    "rows": table["rows"],
                    "columns": table["columns"]
                } for table in potential_tables])
                
                # Extract figures
                for figure in page:
                    if isinstance(figure, LTFigure):
                        figure_data = {
                            "page": page_num,
                            "bbox": figure.bbox,
                            "has_image": any(
                                isinstance(obj, LTImage) for obj in figure
                            )
                        }
                        figures.append(figure_data)
        
        except Exception as e:
            logger.error(f"Error extracting tables and figures: {str(e)}")
        
        return tables, figures
    
    def _detect_tables_from_text_blocks(self, text_blocks: List[Dict]) -> List[Dict]:
        """Detect potential tables from text blocks based on alignment and spacing.
        
        Args:
            text_blocks: List of text blocks with their positions and content
            
        Returns:
            List of detected tables with their properties
        """
        tables = []
        if not text_blocks:
            return tables
            
        # Sort blocks by vertical position
        text_blocks.sort(key=lambda x: x["bbox"][1])
        
        # Group blocks that might form table rows
        current_row = []
        rows = []
        last_y = text_blocks[0]["bbox"][1]
        y_threshold = 5  # pixels
        
        for block in text_blocks:
            if abs(block["bbox"][1] - last_y) <= y_threshold:
                current_row.append(block)
            else:
                if current_row:
                    rows.append(current_row)
                current_row = [block]
            last_y = block["bbox"][1]
        
        if current_row:
            rows.append(current_row)
            
        # Detect tables from rows
        if len(rows) >= 2:  # Need at least 2 rows for a table
            table = {
                "bbox": (
                    min(block["bbox"][0] for row in rows for block in row),
                    min(block["bbox"][1] for row in rows for block in row),
                    max(block["bbox"][2] for row in rows for block in row),
                    max(block["bbox"][3] for row in rows for block in row)
                ),
                "rows": len(rows),
                "columns": max(len(row) for row in rows)
            }
            tables.append(table)
            
        return tables
    
    def _generate_document_id(self, path: Path) -> str:
        """Generate a unique document ID based on the filename.

        Args:
            path: Path to the PDF file

        Returns:
            String ID for the document
        """
        # Extract base name without extension
        base_name = path.stem
        
        # Remove spaces and non-alphanumeric characters
        clean_name = re.sub(r'[^a-zA-Z0-9]', '_', base_name)
        
        # Add timestamp suffix for uniqueness
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        
        return f"{clean_name}_{timestamp}"
    
    def _extract_pdf_content(self, path: Path) -> Tuple[str, List[Dict], int, Dict]:
        """Extract content, sections, and metadata from a PDF using Unstructured.

        Args:
            path: Path to the PDF file

        Returns:
            Tuple of (content, sections, page_count, metadata)
        """
        all_text = ""
        sections = []
        metadata = {}
        page_count = 0
        
        try:
            # Partition the PDF using Unstructured with enhanced settings
            elements = partition_pdf(
                str(path),
                **self.unstructured_config
            )
            
            # Process elements and build sections with enhanced metadata
            current_section = {
                "title": "Document Start",
                "content": "",
                "page_number": None,
                "layout": {
                    "bbox": None,
                    "font": None,
                    "font_size": None
                }
            }
            
            for element in elements:
                try:
                    # Extract element text and metadata
                    element_text = str(element)
                    
                    # Get page number and layout information
                    element_metadata = {}
                    if hasattr(element, 'metadata'):
                        # Get page number
                        page_number = getattr(element.metadata, 'page_number', None)
                        if page_number:
                            page_count = max(page_count, page_number)
                        
                        # Extract layout information
                        element_metadata = {
                            'page_number': page_number,
                            'bbox': getattr(element.metadata, 'bbox', None),
                            'font_type': getattr(element.metadata, 'font_type', None),
                            'font_size': getattr(element.metadata, 'font_size', None)
                        }
                        
                        # Extract document-level metadata if available
                        if hasattr(element.metadata, 'document_metadata'):
                            doc_metadata = element.metadata.document_metadata
                            metadata.update({
                                'title': getattr(doc_metadata, 'title', None),
                                'author': getattr(doc_metadata, 'author', None),
                                'creation_date': getattr(doc_metadata, 'creation_date', None),
                                'modification_date': getattr(doc_metadata, 'modification_date', None),
                                'page_count': getattr(doc_metadata, 'page_count', None)
                            })
                    
                    # Append to all text
                    all_text += element_text + "\n"
                    
                    # Track sections based on element types with enhanced metadata
                    if isinstance(element, Title):
                        # Save previous section if it has content
                        if current_section["content"].strip():
                            sections.append(current_section)
                        
                        # Start new section with enhanced metadata
                        current_section = {
                            "title": element_text,
                            "content": "",
                            "page_number": element_metadata.get('page_number'),
                            "layout": {
                                "bbox": element_metadata.get('bbox'),
                                "font_type": element_metadata.get('font_type'),
                                "font_size": element_metadata.get('font_size')
                            },
                            "element_type": "title"
                        }
                    else:
                        # Add content to current section with layout info
                        current_section["content"] += element_text + "\n"
                        if element_metadata.get('bbox'):
                            current_section["layout"] = {
                                "bbox": element_metadata.get('bbox'),
                                "font_type": element_metadata.get('font_type'),
                                "font_size": element_metadata.get('font_size')
                            }
                except Exception as e:
                    logger.warning(f"Error processing element: {str(e)}")
                    continue
            
            # Add the last section if it has content
            if current_section["content"].strip():
                sections.append(current_section)
            
            # Add file metadata
            metadata.update({
                'file_name': path.name,
                'file_size': path.stat().st_size,
                'element_count': len(elements),
                'page_count': page_count or len(sections) or 1
            })
            
        except Exception as e:
            logger.error(f"Error processing PDF {path}: {str(e)}")
            raise
            
        return all_text, sections, page_count, metadata

    def _process_table(self, table_element: Table) -> str:
        """Process a table element into a string representation.
        
        Args:
            table_element: Table element from Unstructured
            
        Returns:
            String representation of the table
        """
        try:
            # Extract table data safely
            table_data = []
            if hasattr(table_element, 'metadata'):
                try:
                    # Try to get table data as a property
                    if hasattr(table_element.metadata, 'table_data'):
                        raw_data = table_element.metadata.table_data
                        if isinstance(raw_data, list):
                            for row in raw_data:
                                if isinstance(row, list):
                                    table_data.append([str(cell) for cell in row])
                                elif isinstance(row, dict):
                                    table_data.append([str(row.get('text', ''))])
                except AttributeError:
                    pass
            
            # If no table data was extracted, use text content
            if not table_data and hasattr(table_element, 'text'):
                table_data = [[table_element.text]]
            
            # Convert to string
            table_str = []
            for row in table_data:
                table_str.append(" | ".join(row))
            
            return "\n".join(table_str) if table_str else str(table_element.text)
            
        except Exception as e:
            logger.warning(f"Error processing table: {str(e)}")
            return str(table_element.text)
