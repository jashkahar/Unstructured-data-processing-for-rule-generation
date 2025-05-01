"""
Document processor module that coordinates the parsing of different document types.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional
from loguru import logger

from utils.types import Document, DocumentType
from document_processing.pdf_parser import PDFParser
from document_processing.image_parser import ImageParser
from document_processing.video_parser import VideoParser
from document_processing.storage import DocumentStore

class DocumentProcessor:
    """Processes documents of various types using the appropriate parsers."""

    def __init__(self, config: Dict = None):
        """Initialize the document processor with configuration.
        
        Args:
            config: Configuration dictionary for document processing
        """
        self.config = config or {}
        
        # Initialize parsers with their specific configurations
        pdf_config = self.config.get("pdf_parser", {})
        image_config = self.config.get("image_parser", {})
        video_config = self.config.get("video_parser", {})
        
        # Initialize parsers
        self.pdf_parser = PDFParser(pdf_config)
        self.image_parser = ImageParser(image_config)
        self.video_parser = VideoParser(video_config)
        
        # Initialize document store
        self.store = DocumentStore()
        
        # Set up supported extensions
        self.supported_extensions = {
            DocumentType.PDF: [".pdf"],
            DocumentType.IMAGE: [".jpg", ".jpeg", ".png", ".tiff", ".bmp"],
            DocumentType.VIDEO: [".mp4", ".avi", ".mov", ".wmv"]
        }
        
        logger.info("Initialized document processor with support for PDF, Image, and Video files")
    
    def process_directory(self, directory_path: str) -> List[Document]:
        """Process all supported files in a directory.
        
        Args:
            directory_path: Path to the directory containing files
            
        Returns:
            List of processed documents
        """
        directory = Path(directory_path)
        if not directory.exists() or not directory.is_dir():
            raise NotADirectoryError(f"Directory not found: {directory_path}")
        
        processed_documents = []
        
        # Process all supported files in the directory
        for extension in [ext for exts in self.supported_extensions.values() for ext in exts]:
            for file_path in directory.glob(f"**/*{extension}"):
                try:
                    # Process the file
                    document = self.process_file(str(file_path))
                    if document:
                        # Add to store
                        self.store.add_document(document)
                        processed_documents.append(document)
                except Exception as e:
                    logger.error(f"Error processing {file_path}: {str(e)}")
        
        logger.info(f"Processed {len(processed_documents)} documents from {directory_path}")
        return processed_documents
    
    def process_file(self, file_path: str) -> Optional[Document]:
        """Process a single file using the appropriate parser.
        
        Args:
            file_path: Path to the file to process
            
        Returns:
            Processed Document object or None if file type not supported
        """
        path = Path(file_path)
        extension = path.suffix.lower()
        
        # Determine document type based on file extension
        doc_type = self._get_document_type(extension)
        if not doc_type:
            logger.warning(f"Unsupported file type: {extension}")
            return None
        
        # Use the appropriate parser based on document type
        try:
            if doc_type == DocumentType.PDF:
                logger.info(f"Processing PDF file: {file_path}")
                return self.pdf_parser.parse(file_path)
            elif doc_type == DocumentType.IMAGE:
                logger.info(f"Processing Image file: {file_path}")
                document = self.image_parser.parse(file_path)
                return document
            elif doc_type == DocumentType.VIDEO:
                logger.info(f"Processing Video file: {file_path}")
                return self.video_parser.parse(file_path)
        except Exception as e:
            logger.error(f"Error processing {doc_type.value} file {file_path}: {str(e)}")
            raise
            
        return None
    
    def _get_document_type(self, extension: str) -> Optional[DocumentType]:
        """Determine the document type based on file extension.
        
        Args:
            extension: File extension (with dot)
            
        Returns:
            DocumentType enum value or None if not supported
        """
        for doc_type, extensions in self.supported_extensions.items():
            if extension in extensions:
                return doc_type
        return None
    
    def get_document(self, doc_id: str) -> Optional[Document]:
        """Retrieve a document by ID.
        
        Args:
            doc_id: Document ID to retrieve
            
        Returns:
            Document if found, None otherwise
        """
        return self.store.get_document(doc_id)
    
    def get_documents_by_type(self, doc_type: DocumentType) -> List[Document]:
        """Get all documents of a specific type.
        
        Args:
            doc_type: Type of documents to retrieve
            
        Returns:
            List of documents of the specified type
        """
        return self.store.get_documents_by_type(doc_type)
    
    def search_documents(self, query: str) -> List[Document]:
        """Search documents by content.
        
        Args:
            query: Search query string
            
        Returns:
            List of documents matching the query
        """
        return self.store.search_documents(query)
    
    def get_statistics(self) -> Dict:
        """Get document processing statistics.
        
        Returns:
            Dictionary containing processing statistics
        """
        return self.store.get_statistics() 