"""
Document storage module for the Pharmaceutical Compliance Analysis System.
Provides in-memory storage for processed documents with indexing and search capabilities.
"""

from typing import Dict, List, Optional, Set
from datetime import datetime
from collections import defaultdict
from loguru import logger

from utils.types import Document, DocumentType

class DocumentStore:
    """In-memory document storage with indexing and search capabilities."""
    
    def __init__(self):
        """Initialize the document store."""
        # Main document storage
        self.documents: Dict[str, Document] = {}
        
        # Indexes for quick lookups
        self.type_index: Dict[DocumentType, Set[str]] = defaultdict(set)
        self.date_index: Dict[str, Set[str]] = defaultdict(set)  # YYYY-MM-DD -> doc_ids
        self.content_index: Dict[str, Set[str]] = defaultdict(set)  # word -> doc_ids
        
        # Statistics
        self.stats = {
            "total_documents": 0,
            "total_size_bytes": 0,
            "documents_by_type": defaultdict(int),
            "last_updated": None
        }
    
    def add_document(self, document: Document) -> None:
        """Add a document to the store.
        
        Args:
            document: Document to add
        """
        doc_id = document.id
        
        # Add to main storage
        self.documents[doc_id] = document
        
        # Update indexes
        self.type_index[document.doc_type].add(doc_id)
        date_str = document.processed_date.strftime("%Y-%m-%d")
        self.date_index[date_str].add(doc_id)
        
        # Index content words
        words = set(word.lower() for word in document.content.split())
        for word in words:
            self.content_index[word].add(doc_id)
        
        # Update statistics
        self.stats["total_documents"] += 1
        self.stats["total_size_bytes"] += document.file_size
        self.stats["documents_by_type"][document.doc_type] += 1
        self.stats["last_updated"] = datetime.now()
        
        logger.info(f"Added document {doc_id} to store")
    
    def get_document(self, doc_id: str) -> Optional[Document]:
        """Retrieve a document by ID.
        
        Args:
            doc_id: Document ID to retrieve
            
        Returns:
            Document if found, None otherwise
        """
        return self.documents.get(doc_id)
    
    def get_documents_by_type(self, doc_type: DocumentType) -> List[Document]:
        """Get all documents of a specific type.
        
        Args:
            doc_type: Type of documents to retrieve
            
        Returns:
            List of documents of the specified type
        """
        doc_ids = self.type_index.get(doc_type, set())
        return [self.documents[doc_id] for doc_id in doc_ids]
    
    def get_documents_by_date(self, date: datetime) -> List[Document]:
        """Get all documents processed on a specific date.
        
        Args:
            date: Date to search for
            
        Returns:
            List of documents processed on the specified date
        """
        date_str = date.strftime("%Y-%m-%d")
        doc_ids = self.date_index.get(date_str, set())
        return [self.documents[doc_id] for doc_id in doc_ids]
    
    def search_documents(self, query: str) -> List[Document]:
        """Search documents by content.
        
        Args:
            query: Search query string
            
        Returns:
            List of documents matching the query
        """
        query_words = set(word.lower() for word in query.split())
        
        # Find documents containing all query words
        matching_doc_ids = None
        for word in query_words:
            word_docs = self.content_index.get(word, set())
            if matching_doc_ids is None:
                matching_doc_ids = word_docs
            else:
                matching_doc_ids &= word_docs
        
        if matching_doc_ids is None:
            return []
        
        return [self.documents[doc_id] for doc_id in matching_doc_ids]
    
    def remove_document(self, doc_id: str) -> bool:
        """Remove a document from the store.
        
        Args:
            doc_id: ID of document to remove
            
        Returns:
            True if document was removed, False if not found
        """
        if doc_id not in self.documents:
            return False
        
        document = self.documents[doc_id]
        
        # Remove from indexes
        self.type_index[document.doc_type].remove(doc_id)
        date_str = document.processed_date.strftime("%Y-%m-%d")
        self.date_index[date_str].remove(doc_id)
        
        # Remove from content index
        words = set(word.lower() for word in document.content.split())
        for word in words:
            if doc_id in self.content_index[word]:
                self.content_index[word].remove(doc_id)
        
        # Update statistics
        self.stats["total_documents"] -= 1
        self.stats["total_size_bytes"] -= document.file_size
        self.stats["documents_by_type"][document.doc_type] -= 1
        self.stats["last_updated"] = datetime.now()
        
        # Remove from main storage
        del self.documents[doc_id]
        
        logger.info(f"Removed document {doc_id} from store")
        return True
    
    def get_statistics(self) -> Dict:
        """Get storage statistics.
        
        Returns:
            Dictionary containing storage statistics
        """
        return {
            **self.stats,
            "documents_by_type": dict(self.stats["documents_by_type"])
        }
    
    def clear(self) -> None:
        """Clear all documents from the store."""
        self.documents.clear()
        self.type_index.clear()
        self.date_index.clear()
        self.content_index.clear()
        
        # Reset statistics
        self.stats = {
            "total_documents": 0,
            "total_size_bytes": 0,
            "documents_by_type": defaultdict(int),
            "last_updated": datetime.now()
        }
        
        logger.info("Cleared document store") 