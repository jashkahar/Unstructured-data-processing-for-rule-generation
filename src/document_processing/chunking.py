"""
Chunking strategy module for document processing.
Implements hierarchical splitting and adaptive chunk sizing for document content.
"""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, Tuple
from transformers import AutoTokenizer
from loguru import logger
from datetime import datetime

@dataclass
class Chunk:
    """Represents a chunk of a document."""
    chunk_id: str
    section_title: Optional[str]
    content: str
    page_number: Optional[int]
    metadata: Dict = field(default_factory=dict)
    token_count: int = 0
    section_type: Optional[str] = None
    visual_elements: List[Dict] = field(default_factory=list)
    document_id: Optional[str] = None  # Add document_id to track source document in cross-document analysis

class ChunkingStrategy:
    """Implements hierarchical document chunking with adaptive sizing."""

    def __init__(self, config: Dict = None):
        """Initialize the chunking strategy with configuration.
        
        Args:
            config: Configuration dictionary for chunking parameters
        """
        self.config = config or {}
        
        # Core chunking parameters
        self.max_chunk_size = self.config.get("max_chunk_size", 500)  # tokens
        self.overlap_size = self.config.get("overlap_size", 100)  # tokens
        self.min_chunk_size = self.config.get("min_chunk_size", 200)  # tokens
        
        # Semantic coherence settings
        self.preserve_sections = self.config.get("preserve_sections", True)
        self.respect_paragraphs = self.config.get("respect_paragraphs", True)
        self.min_paragraph_length = self.config.get("min_paragraph_length", 50)  # characters
        self.max_paragraph_length = self.config.get("max_paragraph_length", 1000)  # characters
        
        # Section type configuration
        self.section_types = self.config.get("section_types", {
            "header": ["introduction", "overview", "summary"],
            "body": ["main", "content", "details"],
            "footer": ["conclusion", "references", "appendix"]
        })
        
        # Initialize tokenizer using global model configuration
        model_name = self.config.get("models", {}).get("tokenizer_model", "sentence-transformers/all-MiniLM-L6-v2")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
    def chunk_section(self, section_title: str, content: str, page_number: Optional[int] = None, 
                     section_type: Optional[str] = None) -> List[Chunk]:
        """Split a section into chunks using adaptive sizing.
        
        Args:
            section_title: Title of the section
            content: Text content to chunk
            page_number: Optional page number for the section
            section_type: Optional type of section (header, body, footer)
            
        Returns:
            List of Chunk objects
        """
        # First, split content into paragraphs while preserving semantic structure
        paragraphs = self._split_into_paragraphs(content)
        
        chunks = []
        current_chunk = []
        current_tokens = 0
        chunk_id = 1
        
        for paragraph in paragraphs:
            # Skip empty or very short paragraphs
            if len(paragraph.strip()) < self.min_paragraph_length:
                continue
                
            # Get exact token count for the paragraph
            para_tokens = len(self.tokenizer.encode(paragraph))
            
            # If paragraph is too long, split it into smaller parts
            if para_tokens > self.max_chunk_size:
                sub_paragraphs = self._split_long_paragraph(paragraph)
                for sub_para in sub_paragraphs:
                    sub_tokens = len(self.tokenizer.encode(sub_para))
                    if current_tokens + sub_tokens > self.max_chunk_size and current_chunk:
                        # Create chunk with current content
                        chunk = self._create_chunk(
                            chunk_id=chunk_id,
                            section_title=section_title,
                            content="\n".join(current_chunk),
                            page_number=page_number,
                            token_count=current_tokens,
                            section_type=section_type
                        )
                        chunks.append(chunk)
                        
                        # Start new chunk with overlap
                        overlap_paragraphs = self._get_overlap_paragraphs(current_chunk)
                        current_chunk = overlap_paragraphs
                        current_tokens = sum(len(self.tokenizer.encode(p)) for p in overlap_paragraphs)
                        chunk_id += 1
                    
                    current_chunk.append(sub_para)
                    current_tokens += sub_tokens
            else:
                # If adding this paragraph would exceed max size, create a new chunk
                if current_tokens + para_tokens > self.max_chunk_size and current_chunk:
                    # Create chunk with current content
                    chunk = self._create_chunk(
                        chunk_id=chunk_id,
                        section_title=section_title,
                        content="\n".join(current_chunk),
                        page_number=page_number,
                        token_count=current_tokens,
                        section_type=section_type
                    )
                    chunks.append(chunk)
                    
                    # Start new chunk with overlap
                    overlap_paragraphs = self._get_overlap_paragraphs(current_chunk)
                    current_chunk = overlap_paragraphs
                    current_tokens = sum(len(self.tokenizer.encode(p)) for p in overlap_paragraphs)
                    chunk_id += 1
                
                current_chunk.append(paragraph)
                current_tokens += para_tokens
        
        # Add the last chunk if it exists
        if current_chunk:
            chunk = self._create_chunk(
                chunk_id=chunk_id,
                section_title=section_title,
                content="\n".join(current_chunk),
                page_number=page_number,
                token_count=current_tokens,
                section_type=section_type
            )
            chunks.append(chunk)
        
        return chunks
    
    def _create_chunk(self, chunk_id: int, section_title: str, content: str, 
                     page_number: Optional[int], token_count: int, 
                     section_type: Optional[str]) -> Chunk:
        """Create a Chunk object with enhanced metadata.
        
        Args:
            chunk_id: Unique identifier for the chunk
            section_title: Title of the section
            content: Text content
            page_number: Page number
            token_count: Number of tokens
            section_type: Type of section
            
        Returns:
            Chunk object
        """
        return Chunk(
            chunk_id=chunk_id,
            section_title=section_title,
            content=content,
            page_number=page_number,
            token_count=token_count,
            section_type=section_type,
            metadata={
                "section_title": section_title,
                "page_number": page_number,
                "token_count": token_count,
                "section_type": section_type,
                "created_at": datetime.now().isoformat()
            }
        )
    
    def _split_long_paragraph(self, paragraph: str) -> List[str]:
        """Split a long paragraph into smaller parts while preserving semantic coherence.
        
        Args:
            paragraph: Long paragraph to split
            
        Returns:
            List of paragraph parts
        """
        # Split on sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', paragraph)
        
        parts = []
        current_part = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            # If adding this sentence would exceed max length, start a new part
            if current_length + sentence_length > self.max_paragraph_length and current_part:
                parts.append(" ".join(current_part))
                current_part = [sentence]
                current_length = sentence_length
            else:
                current_part.append(sentence)
                current_length += sentence_length
        
        # Add the last part if it exists
        if current_part:
            parts.append(" ".join(current_part))
        
        return parts
    
    def _split_into_paragraphs(self, content: str) -> List[str]:
        """Split content into paragraphs while preserving structure.
        
        Args:
            content: Text content to split
            
        Returns:
            List of paragraphs
        """
        # Clean the content
        content = self._clean_text(content)
        
        # Split on double newlines, preserving single newlines within paragraphs
        paragraphs = [p.strip() for p in re.split(r'\n\s*\n', content) if p.strip()]
        return paragraphs
    
    def _clean_text(self, text: str) -> str:
        """Clean text by removing noise and normalizing whitespace.
        
        Args:
            text: Text to clean
            
        Returns:
            Cleaned text
        """
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove common OCR artifacts
        text = re.sub(r'[^\S\n]+', ' ', text)  # Normalize horizontal whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)  # Normalize vertical whitespace
        
        # Remove any remaining non-printable characters
        text = ''.join(char for char in text if char.isprintable() or char == '\n')
        
        return text.strip()
    
    def _get_overlap_paragraphs(self, paragraphs: List[str]) -> List[str]:
        """Get the last few paragraphs for overlap.
        
        Args:
            paragraphs: List of paragraphs
            
        Returns:
            List of paragraphs for overlap
        """
        overlap_paragraphs = []
        overlap_tokens = 0
        
        # Start from the end and add paragraphs until we reach overlap size
        for para in reversed(paragraphs):
            para_tokens = len(self.tokenizer.encode(para))
            if overlap_tokens + para_tokens > self.overlap_size:
                break
            overlap_paragraphs.insert(0, para)
            overlap_tokens += para_tokens
        
        return overlap_paragraphs 

    def chunk_document(self, document) -> List[Chunk]:
        """Split a document into chunks using hierarchical strategy.
        
        Args:
            document: Document object containing content and metadata
            
        Returns:
            List of Chunk objects
        """
        try:
            logger.info(f"Chunking document: {document.id}")
            chunks = []
            chunk_id = 0
            
            # Split content into sections based on headers
            sections = self._split_into_sections(document.content)
            
            for section_title, section_content in sections:
                # Get page number if available in metadata
                page_number = None
                if hasattr(document, 'metadata') and document.metadata:
                    page_number = document.metadata.get('page_number')
                
                # Determine section type
                section_type = self._determine_section_type(section_title, section_content)
                
                # Chunk the section
                section_chunks = self.chunk_section(
                    section_title=section_title,
                    content=section_content,
                    page_number=page_number,
                    section_type=section_type
                )
                
                # Add document metadata to chunks
                for chunk in section_chunks:
                    chunk.chunk_id = chunk_id
                    chunk.metadata = {
                        'document_id': document.id,
                        'document_type': getattr(document, 'type', None),
                        'section_title': section_title,
                        'section_type': section_type
                    }
                    chunks.append(chunk)
                    chunk_id += 1
            
            logger.info(f"Generated {len(chunks)} chunks for document {document.id}")
            return chunks
            
        except Exception as e:
            logger.error(f"Error chunking document: {str(e)}")
            raise
            
    def _split_into_sections(self, content: str) -> List[Tuple[str, str]]:
        """Split document content into sections based on headers.
        
        Args:
            content: Document content to split
            
        Returns:
            List of (section_title, section_content) tuples
        """
        sections = []
        current_title = "Main Content"
        current_content = []
        
        # Split content into lines
        lines = content.split('\n')
        
        for line in lines:
            # Check if line is a header (simple heuristic)
            if self._is_header(line):
                # Save previous section if exists
                if current_content:
                    sections.append((current_title, '\n'.join(current_content)))
                current_title = line.strip()
                current_content = []
            else:
                current_content.append(line)
        
        # Add last section
        if current_content:
            sections.append((current_title, '\n'.join(current_content)))
            
        return sections
        
    def _is_header(self, line: str) -> bool:
        """Determine if a line is likely a header.
        
        Args:
            line: Line of text to check
            
        Returns:
            True if line appears to be a header
        """
        # Simple heuristic: headers are usually short, all caps, or end with colon
        line = line.strip()
        return (
            len(line) < 100 and
            (line.isupper() or line.endswith(':') or
             bool(re.match(r'^[0-9]+\.\s+[A-Z]', line)))
        )
        
    def _determine_section_type(self, title: str, content: str) -> str:
        """Determine the type of a section based on its title and content.
        
        Args:
            title: Section title
            content: Section content
            
        Returns:
            Section type (header, body, footer, etc.)
        """
        title_lower = title.lower()
        
        if any(word in title_lower for word in ['introduction', 'overview', 'summary']):
            return 'header'
        elif any(word in title_lower for word in ['conclusion', 'references', 'appendix']):
            return 'footer'
        else:
            return 'body' 