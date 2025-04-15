"""
Pattern Analysis module for the Pharmaceutical Compliance Analysis System.
Analyzes content to identify brand-specific patterns in language and presentation.
"""

import re
from collections import Counter
import string
from typing import Dict, List, Set, Tuple

import nltk
from loguru import logger
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.util import ngrams

from utils.types import Document, BrandPattern

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')


class PatternAnalyzer:
    """Analyzes documents to identify brand-specific patterns."""

    def __init__(self, config: Dict = None):
        """Initialize the pattern analyzer with configuration.

        Args:
            config: Configuration dictionary for pattern analysis
        """
        self.config = config or {}
        
        # Set up stopwords for filtering
        self.stop_words = set(stopwords.words('english'))
        
        # Load common medical terms for filtering
        self.common_medical_terms = self._load_common_medical_terms()
        
        # Default confidence threshold for patterns
        self.confidence_threshold = self.config.get("confidence_threshold", 0.7)
        
    def analyze(self, documents: List[Document]) -> List[BrandPattern]:
        """Analyze a list of documents to identify brand-specific patterns.
        
        Args:
            documents: List of documents to analyze
            
        Returns:
            List of identified brand patterns
        """
        logger.info(f"Analyzing {len(documents)} documents for brand patterns")
        
        if not documents:
            return []
        
        # Combine text from all documents for analysis
        all_text = "\n".join([doc.content for doc in documents])
        
        # Perform pattern analysis
        patterns = []
        
        # Analyze keyword frequency
        keyword_patterns = self._analyze_keywords(all_text)
        patterns.extend(keyword_patterns)
        
        # Analyze semantic tone
        tone_patterns = self._analyze_tone(all_text)
        patterns.extend(tone_patterns)
        
        # Analyze aesthetic cues
        aesthetic_patterns = self._analyze_aesthetics(documents)
        patterns.extend(aesthetic_patterns)
        
        logger.info(f"Identified {len(patterns)} brand-specific patterns")
        return patterns
    
    def _analyze_keywords(self, text: str) -> List[BrandPattern]:
        """Analyze text for keyword patterns.
        
        Args:
            text: Text content to analyze
            
        Returns:
            List of keyword-related patterns
        """
        patterns = []
        
        # Tokenize and clean text
        tokens = word_tokenize(text.lower())
        filtered_tokens = [
            word for word in tokens 
            if word not in self.stop_words 
            and word not in string.punctuation
            and len(word) > 2
        ]
        
        # Count word frequencies
        word_counts = Counter(filtered_tokens)
        
        # Generate n-grams for phrase analysis
        bigrams = list(ngrams(filtered_tokens, 2))
        trigrams = list(ngrams(filtered_tokens, 3))
        
        bigram_counts = Counter(bigrams)
        trigram_counts = Counter(trigrams)
        
        # Find common medical terms that are avoided
        avoided_terms = self._identify_avoided_terms(filtered_tokens)
        if avoided_terms:
            patterns.append(BrandPattern(
                category="language_avoidance",
                pattern="avoided_terms",
                confidence=0.8,
                occurrences=len(avoided_terms),
                examples=list(avoided_terms)[:5],
                context={"avoided_terms": list(avoided_terms)}
            ))
        
        # Find frequently used words (potential brand keywords)
        frequent_words = [word for word, count in word_counts.most_common(10) 
                         if count >= 3 and word not in self.common_medical_terms]
        
        if frequent_words:
            patterns.append(BrandPattern(
                category="language_usage",
                pattern="frequent_keywords",
                confidence=0.85,
                occurrences=sum([word_counts[word] for word in frequent_words]),
                examples=frequent_words,
                context={"frequent_words": frequent_words}
            ))
        
        # Find repeated phrases (potential brand messaging)
        frequent_phrases = [" ".join(phrase) for phrase, count in bigram_counts.most_common(5) 
                           if count >= 2]
        
        if frequent_phrases:
            patterns.append(BrandPattern(
                category="language_usage",
                pattern="repeated_phrases",
                confidence=0.9,
                occurrences=sum([bigram_counts[bigram] for bigram in bigrams if " ".join(bigram) in frequent_phrases]),
                examples=frequent_phrases,
                context={"frequent_phrases": frequent_phrases}
            ))
        
        return patterns
    
    def _analyze_tone(self, text: str) -> List[BrandPattern]:
        """Analyze text for semantic tone patterns.
        
        Args:
            text: Text content to analyze
            
        Returns:
            List of tone-related patterns
        """
        patterns = []
        
        # Simplified sentiment analysis (in a real system, would use a proper sentiment analysis library)
        # Count words associated with different tones
        cautious_words = len(re.findall(r'\b(caution|careful|warning|risk|potential|may|might|possibly)\b', 
                                       text, re.IGNORECASE))
        
        optimistic_words = len(re.findall(r'\b(effective|benefit|improve|enhance|success|proven|better)\b', 
                                        text, re.IGNORECASE))
        
        scientific_words = len(re.findall(r'\b(study|research|evidence|clinical|data|results|analysis)\b', 
                                        text, re.IGNORECASE))
        
        # Calculate tone ratios
        total_words = len(text.split())
        if total_words > 0:
            cautious_ratio = cautious_words / total_words
            optimistic_ratio = optimistic_words / total_words
            scientific_ratio = scientific_words / total_words
            
            # Identify predominant tone
            if cautious_ratio > 0.01 and cautious_ratio > optimistic_ratio:
                patterns.append(BrandPattern(
                    category="semantic_tone",
                    pattern="risk_emphasis",
                    confidence=min(cautious_ratio * 10, 0.95),
                    occurrences=cautious_words,
                    examples=re.findall(r'\b(caution|careful|warning|risk|potential)\b', text, re.IGNORECASE)[:5],
                    context={"tone_ratio": cautious_ratio}
                ))
            
            if optimistic_ratio > 0.01 and optimistic_ratio > cautious_ratio:
                patterns.append(BrandPattern(
                    category="semantic_tone",
                    pattern="benefit_emphasis",
                    confidence=min(optimistic_ratio * 10, 0.95),
                    occurrences=optimistic_words,
                    examples=re.findall(r'\b(effective|benefit|improve|enhance|success)\b', text, re.IGNORECASE)[:5],
                    context={"tone_ratio": optimistic_ratio}
                ))
            
            if scientific_ratio > 0.01:
                patterns.append(BrandPattern(
                    category="semantic_tone",
                    pattern="scientific_emphasis",
                    confidence=min(scientific_ratio * 10, 0.95),
                    occurrences=scientific_words,
                    examples=re.findall(r'\b(study|research|evidence|clinical|data)\b', text, re.IGNORECASE)[:5],
                    context={"tone_ratio": scientific_ratio}
                ))
            
        return patterns
    
    def _analyze_aesthetics(self, documents: List[Document]) -> List[BrandPattern]:
        """Analyze documents for aesthetic and design pattern cues.
        
        Args:
            documents: List of documents to analyze
            
        Returns:
            List of aesthetic-related patterns
        """
        patterns = []
        
        # This is a simplified implementation; in reality, would need actual image analysis
        # Here we're just looking for text-based indicators of aesthetic choices
        
        # Look for logo mentions
        logo_mentions = []
        for doc in documents:
            matches = re.findall(r'logo|brand\s+mark|trademark', doc.content, re.IGNORECASE)
            if matches:
                logo_mentions.extend(matches)
        
        if logo_mentions:
            patterns.append(BrandPattern(
                category="aesthetics",
                pattern="logo_usage",
                confidence=0.75,
                occurrences=len(logo_mentions),
                examples=logo_mentions[:5],
                context={"logo_mentions": logo_mentions}
            ))
        
        # Look for color mentions
        color_mentions = []
        for doc in documents:
            matches = re.findall(r'\b(blue|red|green|yellow|orange|purple|white|black|gold|silver|color)\b', 
                               doc.content, re.IGNORECASE)
            if matches:
                color_mentions.extend(matches)
        
        if color_mentions:
            color_counts = Counter(color_mentions)
            most_common_colors = [color for color, count in color_counts.most_common(3)]
            
            patterns.append(BrandPattern(
                category="aesthetics",
                pattern="color_scheme",
                confidence=0.7,
                occurrences=len(color_mentions),
                examples=most_common_colors,
                context={"color_counts": {color: count for color, count in color_counts.items()}}
            ))
        
        return patterns
    
    def _identify_avoided_terms(self, tokens: List[str]) -> Set[str]:
        """Identify medical terms that are avoided in the content.
        
        Args:
            tokens: Tokenized text content
            
        Returns:
            Set of avoided common medical terms
        """
        token_set = set(tokens)
        avoided_terms = set()
        
        # Simplified approach: check which common terms are not present
        # This is naive and would need refinement in a real system
        for term in self.common_medical_terms:
            if term not in token_set and term.lower() not in token_set:
                avoided_terms.add(term)
        
        # Only consider terms as "avoided" if a significant portion are missing
        # Assuming we don't expect ALL common terms to be present
        if len(avoided_terms) > len(self.common_medical_terms) * 0.4:
            return avoided_terms
        return set()
    
    def _load_common_medical_terms(self) -> Set[str]:
        """Load a set of common medical/pharmaceutical terms.
        
        Returns:
            Set of common terms
        """
        # In a real system, would load from a comprehensive file
        # This is a simplified example set
        return {
            "dose", "dosage", "medication", "medicine", "drug", "treatment",
            "symptom", "condition", "disease", "disorder", "prescription",
            "tablet", "capsule", "injection", "oral", "topical", "therapy",
            "contraindication", "indication", "adverse", "effect", "reaction",
            "pharmacist", "doctor", "physician", "healthcare", "provider",
            "efficacy", "safety", "cure", "relief", "chronic", "acute"
        } 