"""
Visual aggregator module for the Pharmaceutical Compliance Analysis System.
Processes and clusters visual captions to identify patterns.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Any
from loguru import logger
import numpy as np
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer

class VisualAggregator:
    """Aggregates and clusters visual captions to identify patterns."""
    
    def __init__(self, config: Dict = None):
        """Initialize the visual aggregator with configuration.
        
        Args:
            config: Configuration dictionary for visual aggregation parameters
        """
        self.config = config or {}
        
        # Clustering configuration
        self.perform_clustering = self.config.get("perform_clustering", True)
        self.num_clusters = self.config.get("num_clusters", 3)
        self.min_cluster_size = self.config.get("min_cluster_size", 2)
        
        # Model configuration
        self.embedding_model_name = self.config.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
        
        # Output configuration
        self.visual_patterns_output = Path(self.config.get("visual_patterns_output", "output/intermediate_results/visual_patterns.json"))
        self.visual_patterns_output.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize embedding model if clustering is enabled
        if self.perform_clustering:
            logger.info(f"Loading sentence embedding model: {self.embedding_model_name}")
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
            logger.info("Sentence embedding model loaded successfully")
    
    def aggregate_captions(self, captions_data: List[Dict]) -> Dict:
        """Aggregate and process visual captions.
        
        Args:
            captions_data: List of dictionaries containing caption data
            
        Returns:
            Dictionary containing visual patterns
        """
        try:
            if not captions_data:
                logger.warning("No caption data to aggregate")
                return {"patterns": []}
            
            logger.info(f"Aggregating {len(captions_data)} visual captions")
            
            # Extract all captions into a flat list
            all_captions = []
            caption_metadata = []
            
            for caption_entry in captions_data:
                doc_id = caption_entry.get("document_id")
                page_num = caption_entry.get("page_number")
                
                # Process each caption type
                for caption_type, caption_text in caption_entry.get("captions", {}).items():
                    all_captions.append(caption_text)
                    caption_metadata.append({
                        "document_id": doc_id,
                        "page_number": page_num,
                        "caption_type": caption_type,
                        "image_path": caption_entry.get("image_path")
                    })
            
            # Process captions based on configuration
            if self.perform_clustering and len(all_captions) >= self.min_cluster_size:
                patterns = self._cluster_captions(all_captions, caption_metadata)
            else:
                patterns = self._simple_aggregate(all_captions, caption_metadata)
            
            # Save patterns to file
            self._save_patterns(patterns)
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error aggregating captions: {str(e)}")
            return {"patterns": []}
    
    def _cluster_captions(self, captions: List[str], metadata: List[Dict]) -> Dict:
        """Cluster captions using sentence embeddings.
        
        Args:
            captions: List of caption texts
            metadata: List of caption metadata
            
        Returns:
            Dictionary containing clustered patterns
        """
        try:
            # Generate embeddings
            logger.info(f"Generating embeddings for {len(captions)} captions")
            embeddings = self.embedding_model.encode(captions)
            
            # Determine number of clusters
            k = min(self.num_clusters, len(captions) // 2)
            if k < 2:
                k = 2
            
            # Perform clustering
            logger.info(f"Clustering captions into {k} clusters")
            kmeans = KMeans(n_clusters=k, random_state=42)
            clusters = kmeans.fit_predict(embeddings)
            
            # Organize captions by cluster
            cluster_map = {}
            for i, cluster_id in enumerate(clusters):
                cluster_id = int(cluster_id)
                if cluster_id not in cluster_map:
                    cluster_map[cluster_id] = []
                
                cluster_map[cluster_id].append({
                    "caption": captions[i],
                    "metadata": metadata[i]
                })
            
            # Create pattern structure
            patterns = {
                "patterns": []
            }
            
            for cluster_id, cluster_items in cluster_map.items():
                # Get representative examples
                representative_examples = [item["caption"] for item in cluster_items[:5]]
                
                # Extract documents and pages in this cluster
                documents = set()
                pages = set()
                for item in cluster_items:
                    doc_id = item["metadata"]["document_id"]
                    page_num = item["metadata"]["page_number"]
                    documents.add(doc_id)
                    pages.add(f"{doc_id}_{page_num}")
                
                # Extract common visual patterns
                common_patterns = self._extract_common_patterns(representative_examples)
                
                # Create pattern object with matching structure to text patterns
                pattern = {
                    "cluster_id": f"visual_{cluster_id}",
                    "size": len(cluster_items),
                    "representative_examples": representative_examples,
                    "documents": list(documents),
                    "pages": list(pages),
                    "caption_types": self._count_caption_types(cluster_items),
                    "common_patterns": {
                        "layout": ", ".join(common_patterns.get("layout", [])) or "no consistent layout detected",
                        "color": ", ".join(common_patterns.get("colors", [])) or "no consistent color scheme detected",
                        "design": ", ".join(common_patterns.get("design_elements", [])) or "no consistent design elements detected"
                    },
                    "source": "visual",
                    "confidence_score": 0.8,  # Fixed confidence score for visual patterns
                    "visual_metadata": {
                        "caption_types": self._count_caption_types(cluster_items),
                        "raw_patterns": common_patterns
                    }
                }
                
                patterns["patterns"].append(pattern)
            
            logger.info(f"Created {len(patterns['patterns'])} visual patterns")
            return patterns
            
        except Exception as e:
            logger.error(f"Error clustering captions: {str(e)}")
            return {"patterns": []}
    
    def _simple_aggregate(self, captions: List[str], metadata: List[Dict]) -> Dict:
        """Create simple aggregation without clustering.
        
        Args:
            captions: List of caption texts
            metadata: List of caption metadata
            
        Returns:
            Dictionary containing aggregated patterns
        """
        try:
            # Group by caption type
            caption_types = {}
            for i, caption in enumerate(captions):
                caption_type = metadata[i]["caption_type"]
                if caption_type not in caption_types:
                    caption_types[caption_type] = []
                
                caption_types[caption_type].append({
                    "caption": caption,
                    "metadata": metadata[i]
                })
            
            # Create pattern structure
            patterns = {
                "patterns": []
            }
            
            for caption_type, items in caption_types.items():
                # Get representative examples
                representative_examples = [item["caption"] for item in items[:5]]
                
                # Extract documents and pages
                documents = set()
                pages = set()
                for item in items:
                    doc_id = item["metadata"]["document_id"]
                    page_num = item["metadata"]["page_number"]
                    documents.add(doc_id)
                    pages.add(f"{doc_id}_{page_num}")
                
                # Extract common visual patterns
                common_patterns = self._extract_common_patterns(representative_examples)
                
                # Create pattern object with matching structure to text patterns
                pattern = {
                    "cluster_id": f"visual_{caption_type}",
                    "size": len(items),
                    "representative_examples": representative_examples,
                    "documents": list(documents),
                    "pages": list(pages),
                    "common_patterns": {
                        "layout": ", ".join(common_patterns.get("layout", [])) or "no consistent layout detected",
                        "color": ", ".join(common_patterns.get("colors", [])) or "no consistent color scheme detected",
                        "design": ", ".join(common_patterns.get("design_elements", [])) or "no consistent design elements detected"
                    },
                    "source": "visual",
                    "confidence_score": 0.7,  # Slightly lower confidence score for non-clustered patterns
                    "visual_metadata": {
                        "caption_type": caption_type,
                        "raw_patterns": common_patterns
                    }
                }
                
                patterns["patterns"].append(pattern)
            
            logger.info(f"Created {len(patterns['patterns'])} visual patterns without clustering")
            return patterns
            
        except Exception as e:
            logger.error(f"Error creating simple aggregation: {str(e)}")
            return {"patterns": []}
    
    def _count_caption_types(self, items: List[Dict]) -> Dict[str, int]:
        """Count occurrences of each caption type in a cluster.
        
        Args:
            items: List of cluster items with metadata
            
        Returns:
            Dictionary with caption type counts
        """
        counts = {}
        for item in items:
            caption_type = item["metadata"]["caption_type"]
            counts[caption_type] = counts.get(caption_type, 0) + 1
        
        return counts
    
    def _extract_common_patterns(self, captions: List[str]) -> Dict[str, Any]:
        """Extract common patterns from a set of captions.
        
        Args:
            captions: List of caption texts
            
        Returns:
            Dictionary containing common patterns
        """
        # This is a simplified implementation
        # A more sophisticated version would use NLP to extract common themes
        
        # Check for common words related to layout
        layout_keywords = ["header", "footer", "sidebar", "column", "row", "centered", "aligned",
                          "top", "bottom", "left", "right", "horizontal", "vertical", "grid"]
        layout_patterns = self._find_keyword_patterns(captions, layout_keywords)
        
        # Check for color mentions
        color_keywords = ["blue", "red", "green", "yellow", "white", "black", "gray", "grey", "purple", 
                         "orange", "brown", "pink", "teal", "cyan", "magenta", "gold", "silver", 
                         "dark", "light", "bright", "vibrant", "muted", "pastel", "contrasting"]
        color_patterns = self._find_keyword_patterns(captions, color_keywords)
        
        # Check for design elements
        design_keywords = ["logo", "image", "photo", "icon", "button", "text", "banner", "border",
                          "box", "table", "chart", "graph", "diagram", "illustration", "arrow",
                          "box", "shadow", "gradient", "bold", "italic", "underline", "serif", "sans-serif",
                          "minimalist", "modern", "traditional", "clean", "busy", "simple", "complex"]
        design_patterns = self._find_keyword_patterns(captions, design_keywords)
        
        return {
            "layout": layout_patterns,
            "colors": color_patterns,
            "design_elements": design_patterns
        }
    
    def _find_keyword_patterns(self, texts: List[str], keywords: List[str]) -> List[str]:
        """Find occurrences of keywords in texts.
        
        Args:
            texts: List of texts to search
            keywords: List of keywords to find
            
        Returns:
            List of found keywords
        """
        found = set()
        for text in texts:
            text_lower = text.lower()
            for keyword in keywords:
                if keyword.lower() in text_lower:
                    found.add(keyword)
        
        return list(found)
    
    def _save_patterns(self, patterns: Dict) -> None:
        """Save visual patterns to a JSON file.
        
        Args:
            patterns: Dictionary containing visual patterns
        """
        try:
            with open(self.visual_patterns_output, 'w', encoding='utf-8') as f:
                json.dump(patterns, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved {len(patterns.get('patterns', []))} visual patterns to {self.visual_patterns_output}")
            
        except Exception as e:
            logger.error(f"Error saving visual patterns to {self.visual_patterns_output}: {str(e)}")
    
    def load_patterns(self) -> Dict:
        """Load visual patterns from a previously saved JSON file.
        
        Returns:
            Dictionary containing visual patterns
        """
        try:
            if not self.visual_patterns_output.exists():
                logger.warning(f"Visual patterns file not found: {self.visual_patterns_output}")
                return {"patterns": []}
            
            with open(self.visual_patterns_output, 'r', encoding='utf-8') as f:
                patterns = json.load(f)
            
            logger.info(f"Loaded {len(patterns.get('patterns', []))} visual patterns from {self.visual_patterns_output}")
            return patterns
            
        except Exception as e:
            logger.error(f"Error loading visual patterns from {self.visual_patterns_output}: {str(e)}")
            return {"patterns": []} 