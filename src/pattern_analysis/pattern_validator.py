"""
Pattern validation and persistence module for the Pharmaceutical Compliance Analysis System.
This module handles validation of discovered patterns and their persistence to storage.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

from loguru import logger

class PatternValidator:
    """Validates and persists discovered patterns."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the pattern validator.
        
        Args:
            config: Configuration dictionary containing validation settings
        """
        self.config = config
        self.storage_path = Path(config.get('pattern_storage_path', 'data/patterns'))
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Validation settings
        self.min_cluster_size = config.get('min_cluster_size', 3)
        self.min_pattern_confidence = config.get('min_pattern_confidence', 0.7)
        self.required_pattern_fields = {
            'cluster_id',
            'size',
            'representative',
            'characteristics',
            'confidence_score'
        }
        
        logger.info("Initialized PatternValidator")
        
    def validate_patterns(self, patterns: Dict[str, Any]) -> Dict[str, Any]:
        """Validate discovered patterns.
        
        Args:
            patterns: Dictionary containing patterns list and metadata
            
        Returns:
            Dictionary containing validated patterns and validation metadata
        """
        validated_patterns = []
        validation_metadata = {
            'timestamp': datetime.now().isoformat(),
            'total_patterns': len(patterns.get('patterns', [])),
            'valid_patterns': 0,
            'invalid_patterns': 0,
            'validation_errors': []
        }
        
        for pattern in patterns.get('patterns', []):
            try:
                if self._validate_pattern(pattern):
                    validated_patterns.append(pattern)
                    validation_metadata['valid_patterns'] += 1
                else:
                    validation_metadata['invalid_patterns'] += 1
                    validation_metadata['validation_errors'].append({
                        'pattern_id': pattern.get('cluster_id'),
                        'error': 'Pattern failed validation criteria'
                    })
            except Exception as e:
                logger.error(f"Error validating pattern {pattern.get('cluster_id')}: {str(e)}")
                validation_metadata['validation_errors'].append({
                    'pattern_id': pattern.get('cluster_id'),
                    'error': str(e)
                })
                
        return {
            'patterns': validated_patterns,
            'metadata': validation_metadata
        }
    
    def _validate_pattern(self, pattern: Dict[str, Any]) -> bool:
        """Validate a single pattern.
        
        Args:
            pattern: Pattern to validate
            
        Returns:
            True if pattern is valid, False otherwise
        """
        # Check required fields
        missing_fields = [field for field in self.required_pattern_fields if field not in pattern]
        if missing_fields:
            logger.warning(f"Pattern {pattern.get('cluster_id', 'unknown')} missing required fields: {missing_fields}")
            return False
            
        # Check cluster size
        size = pattern.get('size', 0)
        if size < self.min_cluster_size:
            logger.warning(f"Pattern {pattern.get('cluster_id', 'unknown')} size {size} below minimum {self.min_cluster_size}")
            return False
            
        # Check confidence score
        confidence = pattern.get('confidence_score', 0)
        if confidence < self.min_pattern_confidence:
            logger.warning(f"Pattern {pattern.get('cluster_id', 'unknown')} confidence {confidence} below minimum {self.min_pattern_confidence}")
            return False
            
        # Validate cluster characteristics
        characteristics = pattern.get('characteristics', {})
        if not self._validate_characteristics(characteristics):
            logger.warning(f"Pattern {pattern.get('cluster_id', 'unknown')} failed characteristics validation")
            return False
            
        logger.info(f"Pattern {pattern.get('cluster_id', 'unknown')} passed all validation checks")
        return True
    
    def _validate_characteristics(self, characteristics: Dict[str, Any]) -> bool:
        """Validate cluster characteristics.
        
        Args:
            characteristics: Cluster characteristics to validate
            
        Returns:
            True if characteristics are valid, False otherwise
        """
        required_characteristics = {
            'top_words',
            'section_types',
            'most_common_section',
            'avg_length'
        }
        
        missing_characteristics = [field for field in required_characteristics if field not in characteristics]
        if missing_characteristics:
            logger.warning(f"Missing required characteristics: {missing_characteristics}")
            return False
            
        # Validate top words
        if not isinstance(characteristics['top_words'], dict):
            logger.warning(f"top_words is not a dictionary: {type(characteristics['top_words'])}")
            return False
            
        # Validate section types
        if not isinstance(characteristics['section_types'], dict):
            logger.warning(f"section_types is not a dictionary: {type(characteristics['section_types'])}")
            return False
            
        # Validate average length
        if not isinstance(characteristics['avg_length'], (int, float)) or characteristics['avg_length'] <= 0:
            logger.warning(f"Invalid avg_length: {characteristics['avg_length']}")
            return False
            
        logger.info("All characteristics passed validation")
        return True
    
    def persist_patterns(self, patterns: Dict[str, Any], metadata: Dict[str, Any]) -> str:
        """Persist validated patterns to storage.
        
        Args:
            patterns: Dictionary of validated patterns
            metadata: Validation metadata
            
        Returns:
            Path to the persisted patterns file
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'patterns_{timestamp}.json'
        filepath = self.storage_path / filename
        
        data = {
            'patterns': patterns,
            'metadata': metadata,
            'persistence_timestamp': datetime.now().isoformat()
        }
        
        try:
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            logger.info(f"Persisted patterns to {filepath}")
            return str(filepath)
        except Exception as e:
            logger.error(f"Error persisting patterns: {str(e)}")
            raise
    
    def load_patterns(self, filepath: Optional[str] = None) -> Dict[str, Any]:
        """Load patterns from storage.
        
        Args:
            filepath: Optional path to specific patterns file. If None, loads most recent.
            
        Returns:
            Dictionary containing patterns and metadata
        """
        if filepath is None:
            # Get most recent patterns file
            pattern_files = list(self.storage_path.glob('patterns_*.json'))
            if not pattern_files:
                raise FileNotFoundError("No pattern files found")
            filepath = str(max(pattern_files, key=lambda x: x.stat().st_mtime))
            
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            logger.info(f"Loaded patterns from {filepath}")
            return data
        except Exception as e:
            logger.error(f"Error loading patterns: {str(e)}")
            raise 