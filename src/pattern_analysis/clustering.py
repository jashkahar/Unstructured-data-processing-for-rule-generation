"""
Clustering module for the Pharmaceutical Compliance Analysis System.
Discovers patterns in text chunks using unsupervised clustering.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from loguru import logger

from document_processing.chunking import Chunk
from pattern_analysis.embeddings import EmbeddingManager

class PatternClusterer:
    """Clusters text chunks to discover patterns in promotional content."""
    
    def __init__(self, config: Dict = None):
        """Initialize the pattern clusterer with configuration.
        
        Args:
            config: Configuration dictionary for clustering parameters
        """
        self.config = config or {}
        self.min_clusters = self.config.get("min_clusters", 3)
        self.max_clusters = self.config.get("max_clusters", 10)
        self.silhouette_threshold = self.config.get("silhouette_threshold", 0.3)
        self.embedding_manager = EmbeddingManager(self.config)
        
    def discover_patterns(self, chunks: List[Chunk], embeddings: np.ndarray, features: List[Dict]) -> Dict:
        """Discover patterns in text chunks using clustering.
        
        Args:
            chunks: List of Chunk objects
            embeddings: Pre-computed embeddings for the chunks
            features: List of extracted features for each chunk
            
        Returns:
            Dictionary containing cluster assignments and patterns
        """
        try:
            logger.info(f"Discovering patterns in {len(chunks)} chunks")
            
            # Find optimal number of clusters
            n_clusters = self._find_optimal_clusters(embeddings)
            logger.info(f"Optimal number of clusters: {n_clusters}")
            
            # Perform clustering
            clusterer = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = clusterer.fit_predict(embeddings)
            
            # Calculate cluster quality metrics
            silhouette_avg = silhouette_score(embeddings, cluster_labels)
            calinski_score = calinski_harabasz_score(embeddings, cluster_labels)
            
            logger.info(f"Cluster quality - Silhouette: {silhouette_avg:.3f}, Calinski-Harabasz: {calinski_score:.3f}")
            
            # Analyze clusters
            patterns = self._analyze_clusters(chunks, cluster_labels, embeddings, features)
            
            # Add feature information to patterns
            for pattern in patterns:
                cluster_chunks = [chunks[i] for i in range(len(chunks)) if cluster_labels[i] == pattern["cluster_id"]]
                pattern["features"] = self._analyze_cluster_features(cluster_chunks, features)
            
            results = {
                "num_clusters": n_clusters,
                "patterns": patterns,
                "cluster_labels": cluster_labels.tolist(),
                "quality_metrics": {
                    "silhouette_score": float(silhouette_avg),
                    "calinski_harabasz_score": float(calinski_score)
                }
            }
            
            logger.info(f"Discovered {len(patterns)} patterns")
            return results
            
        except Exception as e:
            logger.error(f"Error discovering patterns: {str(e)}")
            raise
    
    def _find_optimal_clusters(self, embeddings: np.ndarray) -> int:
        """Find optimal number of clusters using silhouette score and elbow method.
        
        Args:
            embeddings: numpy array of embeddings
            
        Returns:
            Optimal number of clusters
        """
        best_score = -1
        best_n = self.min_clusters
        scores = []
        inertias = []
        
        for n in range(self.min_clusters, min(self.max_clusters + 1, len(embeddings))):
            # Perform clustering
            clusterer = KMeans(n_clusters=n, random_state=42)
            labels = clusterer.fit_predict(embeddings)
            
            # Calculate silhouette score
            score = silhouette_score(embeddings, labels)
            scores.append(score)
            
            # Store inertia for elbow method
            inertias.append(clusterer.inertia_)
            
            logger.info(f"Clusters: {n}, Silhouette: {score:.3f}, Inertia: {clusterer.inertia_:.3f}")
            
            # Update best score if above threshold
            if score > best_score and score >= self.silhouette_threshold:
                best_score = score
                best_n = n
        
        # If no score above threshold, use elbow method
        if best_score < self.silhouette_threshold:
            # Calculate inertia differences
            inertia_diffs = np.diff(inertias)
            # Find the point of maximum curvature
            best_n = self.min_clusters + np.argmax(np.abs(inertia_diffs)) + 1
            logger.info(f"Using elbow method, selected {best_n} clusters")
        
        return best_n
    
    def _analyze_clusters(self, chunks: List[Chunk], labels: np.ndarray, 
                         embeddings: np.ndarray, features: List[Dict]) -> List[Dict]:
        """Analyze clusters to extract patterns.
        
        Args:
            chunks: List of Chunk objects
            labels: Cluster labels for each chunk
            embeddings: Embeddings for each chunk
            features: List of feature dictionaries for each chunk
            
        Returns:
            List of dictionaries containing cluster analysis
        """
        clusters = []
        unique_labels = np.unique(labels)
        
        for label in unique_labels:
            # Get chunks in this cluster
            cluster_indices = np.where(labels == label)[0]
            cluster_chunks = [chunks[i] for i in cluster_indices]
            
            # Get centroid
            centroid = self.embedding_manager.get_centroid(cluster_indices, embeddings)
            
            # Find representative chunks (closest to centroid)
            if centroid is not None:
                cluster_embeddings = embeddings[cluster_indices]
                distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)
                representative_idx = cluster_indices[np.argmin(distances)]
                representative = chunks[representative_idx]
            else:
                representative = cluster_chunks[0]
            
            # Convert representative chunk to dictionary
            representative_dict = {
                "chunk_id": representative.chunk_id,
                "content": representative.content,
                "section_title": representative.section_title,
                "page_number": representative.page_number,
                "section_type": representative.section_type,
                "metadata": representative.metadata
            }
            
            # Analyze cluster characteristics
            characteristics = self._analyze_cluster_characteristics(cluster_chunks)
            
            # Calculate cluster coherence
            coherence = self._calculate_cluster_coherence(cluster_embeddings, centroid)
            
            clusters.append({
                "cluster_id": int(label),
                "size": len(cluster_chunks),
                "representative": representative_dict,
                "characteristics": characteristics,
                "coherence": float(coherence),
                "confidence_score": min(0.8 + coherence, 1.0)  # Scale coherence to confidence
            })
            
        return clusters
    
    def _analyze_cluster_characteristics(self, chunks: List[Chunk]) -> Dict:
        """Analyze characteristics of chunks in a cluster.
        
        Args:
            chunks: List of Chunk objects in the cluster
            
        Returns:
            Dictionary of cluster characteristics
        """
        # Extract common words and phrases
        all_words = " ".join(chunk.content.lower() for chunk in chunks).split()
        word_freq = {}
        for word in all_words:
            if len(word) > 3:  # Skip short words
                word_freq[word] = word_freq.get(word, 0) + 1
                
        # Get top words
        top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
        
        # Analyze section types
        section_types = {}
        for chunk in chunks:
            section_type = chunk.section_type or "unknown"
            section_types[section_type] = section_types.get(section_type, 0) + 1
            
        # Get most common section type
        most_common_section = max(section_types.items(), key=lambda x: x[1])[0] if section_types else "unknown"
        
        # Calculate average length and standard deviation
        lengths = [len(chunk.content.split()) for chunk in chunks]
        avg_length = np.mean(lengths)
        std_length = np.std(lengths)
        
        return {
            "top_words": dict(top_words),
            "section_types": section_types,
            "most_common_section": most_common_section,
            "avg_length": float(avg_length),
            "length_std": float(std_length)
        }
    
    def _analyze_cluster_features(self, chunks: List[Chunk], features: List[Dict]) -> Dict:
        """Analyze features for a cluster of chunks.
        
        Args:
            chunks: List of Chunk objects in the cluster
            features: List of feature dictionaries for all chunks
            
        Returns:
            Dictionary containing feature analysis
        """
        # Get features for chunks in this cluster
        cluster_features = []
        for chunk in chunks:
            chunk_idx = chunk.chunk_id
            if chunk_idx < len(features):
                cluster_features.append(features[chunk_idx])
        
        if not cluster_features:
            return {}
            
        # Aggregate feature statistics
        feature_stats = {}
        for feature_type in cluster_features[0].keys():
            try:
                if feature_type == "sentiment":
                    # Special handling for sentiment
                    labels = [f[feature_type]["label"] for f in cluster_features]
                    scores = [f[feature_type]["score"] for f in cluster_features]
                    feature_stats[feature_type] = {
                        "most_common_label": max(set(labels), key=labels.count),
                        "mean_score": float(np.mean(scores)),
                        "std_score": float(np.std(scores))
                    }
                elif feature_type == "benefit_risk":
                    # Special handling for benefit-risk
                    ratios = [f[feature_type]["ratio"] for f in cluster_features]
                    feature_stats[feature_type] = {
                        "mean_ratio": float(np.mean(ratios)),
                        "std_ratio": float(np.std(ratios)),
                        "most_common_balance": max(
                            set(f[feature_type]["balance"] for f in cluster_features),
                            key=lambda x: sum(1 for f in cluster_features if f[feature_type]["balance"] == x)
                        )
                    }
                elif feature_type == "tone":
                    # Special handling for tone
                    scores = [f[feature_type]["promotional_score"] for f in cluster_features]
                    feature_stats[feature_type] = {
                        "mean_score": float(np.mean(scores)),
                        "std_score": float(np.std(scores)),
                        "most_common_tone": max(
                            set(f[feature_type]["tone"] for f in cluster_features),
                            key=lambda x: sum(1 for f in cluster_features if f[feature_type]["tone"] == x)
                        )
                    }
                else:
                    # Generic numeric feature handling
                    values = [float(f[feature_type]) for f in cluster_features if feature_type in f]
                    if values:
                        feature_stats[feature_type] = {
                            "mean": float(np.mean(values)),
                            "std": float(np.std(values)),
                            "min": float(np.min(values)),
                            "max": float(np.max(values))
                        }
            except (ValueError, TypeError):
                # Skip features that can't be converted to float
                continue
                
        return feature_stats
    
    def _calculate_cluster_coherence(self, cluster_embeddings: np.ndarray, centroid: np.ndarray) -> float:
        """Calculate the coherence of a cluster based on embedding distances.
        
        Args:
            cluster_embeddings: Embeddings for chunks in the cluster
            centroid: Centroid embedding of the cluster
            
        Returns:
            Coherence score between 0 and 1
        """
        if len(cluster_embeddings) == 0 or centroid is None:
            return 0.0
            
        # Calculate distances from centroid
        distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)
        
        # Convert distances to similarity scores (1 - normalized distance)
        max_distance = np.max(distances)
        if max_distance > 0:
            similarities = 1 - (distances / max_distance)
        else:
            similarities = np.ones_like(distances)
            
        # Calculate coherence as mean similarity
        coherence = float(np.mean(similarities))
        
        return coherence 