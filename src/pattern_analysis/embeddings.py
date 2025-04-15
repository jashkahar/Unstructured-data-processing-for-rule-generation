"""
Embeddings module for the Pharmaceutical Compliance Analysis System.
Handles text embedding generation and vector storage using FAISS.
"""

import numpy as np
from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer
import faiss
from loguru import logger

class EmbeddingManager:
    """Manages text embeddings and vector storage using FAISS."""
    
    def __init__(self, config: Dict = None):
        """Initialize the embedding manager with configuration.
        
        Args:
            config: Configuration dictionary for embedding parameters
        """
        self.config = config or {}
        # Use global model configuration
        self.model_name = self.config.get("models", {}).get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
        self.dimension = self.config.get("dimension", 384)  # Default for MiniLM
        
        # Initialize sentence transformer
        logger.info(f"Loading sentence transformer model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)
        
        # Initialize FAISS index with L2 normalization
        self.index = faiss.IndexFlatL2(self.dimension)
        
        # Store chunk metadata
        self.chunk_metadata: List[Dict] = []
        
    def generate_embeddings(self, chunks: List[str]) -> np.ndarray:
        """Generate embeddings for a list of text chunks.
        
        Args:
            chunks: List of text chunks to embed
            
        Returns:
            numpy array of normalized embeddings
        """
        logger.info(f"Generating embeddings for {len(chunks)} chunks")
        # Generate embeddings with normalization
        embeddings = self.model.encode(chunks, show_progress_bar=True, normalize_embeddings=True)
        
        # Verify normalization
        norms = np.linalg.norm(embeddings, axis=1)
        if not np.allclose(norms, 1.0, rtol=1e-5):
            logger.warning("Embeddings not properly normalized, applying L2 normalization")
            embeddings = embeddings / norms[:, np.newaxis]
            
        return embeddings
    
    def add_to_index(self, embeddings: np.ndarray, metadata: List[Dict]) -> None:
        """Add embeddings to the FAISS index with metadata.
        
        Args:
            embeddings: numpy array of embeddings
            metadata: List of metadata dictionaries for each embedding
        """
        if len(embeddings) != len(metadata):
            raise ValueError("Number of embeddings must match number of metadata entries")
            
        # Ensure embeddings are float32 for FAISS
        embeddings = embeddings.astype('float32')
        
        # Add to FAISS index
        self.index.add(embeddings)
        
        # Store metadata
        self.chunk_metadata.extend(metadata)
        
        logger.info(f"Added {len(embeddings)} embeddings to index")
        
    def search(self, query: str, k: int = 5) -> List[Dict]:
        """Search for similar chunks using a query string.
        
        Args:
            query: Query text to search for
            k: Number of results to return
            
        Returns:
            List of dictionaries containing similar chunks and their metadata
        """
        # Generate query embedding with normalization
        query_embedding = self.model.encode([query], normalize_embeddings=True)[0]
        query_embedding = query_embedding.astype('float32')
        
        # Search in FAISS index
        distances, indices = self.index.search(
            query_embedding.reshape(1, -1),
            k
        )
        
        # Get results with metadata
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.chunk_metadata):  # Ensure valid index
                result = {
                    "chunk": self.chunk_metadata[idx],
                    "distance": float(distances[0][i])
                }
                results.append(result)
                
        return results
    
    def get_cluster_embeddings(self, cluster_indices: List[int]) -> np.ndarray:
        """Get embeddings for chunks in a cluster.
        
        Args:
            cluster_indices: List of indices for chunks in the cluster
            
        Returns:
            numpy array of embeddings for the cluster
        """
        # This method is not needed anymore as we're passing embeddings directly
        # Return an empty array to avoid errors
        return np.array([])
    
    def get_centroid(self, cluster_indices: List[int], embeddings: np.ndarray) -> Optional[np.ndarray]:
        """Calculate the centroid embedding for a cluster.
        
        Args:
            cluster_indices: List of indices for chunks in the cluster
            embeddings: Full embeddings array for all chunks
            
        Returns:
            numpy array of the normalized centroid embedding, or None if cluster is empty
        """
        if len(cluster_indices) == 0:
            return None
            
        # Extract embeddings for the cluster
        cluster_embeddings = embeddings[cluster_indices]
        
        # Calculate centroid
        centroid = np.mean(cluster_embeddings, axis=0)
        
        # Normalize centroid
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid = centroid / norm
            
        return centroid 