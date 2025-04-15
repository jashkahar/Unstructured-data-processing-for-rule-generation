"""
Feature extraction module for the Pharmaceutical Compliance Analysis System.
Extracts semantic features like sentiment, benefit-risk ratio, and tone from text.
"""

import re
from typing import Dict, List, Tuple
from transformers import pipeline
from loguru import logger

class FeatureExtractor:
    """Extracts semantic features from text chunks."""
    
    def __init__(self, config: Dict = None):
        """Initialize the feature extractor with configuration.
        
        Args:
            config: Configuration dictionary for feature extraction parameters
        """
        self.config = config or {}
        
        # Initialize sentiment analysis pipeline using global model configuration
        sentiment_model = self.config.get("models", {}).get("sentiment_model", 
            "distilbert-base-uncased-finetuned-sst-2-english")
        self.sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model=sentiment_model
        )
        
        # Define benefit and risk keywords from config or use defaults
        benefit_keywords = self.config.get("feature_extraction", {}).get("benefit_risk", {}).get("keywords", {}).get("benefits", [
            "effective", "improves", "leading", "new", "innovative",
            "breakthrough", "advanced", "superior", "best", "optimal",
            "efficient", "powerful", "successful", "proven", "safe",
            "reliable", "trusted", "recommended", "preferred", "excellent",
            "benefit", "effectiveness", "improvement", "relief", "treatment",
            "efficacy"
        ])
        
        risk_keywords = self.config.get("feature_extraction", {}).get("benefit_risk", {}).get("keywords", {}).get("risks", [
            "side effects", "warning", "not recommended", "risk",
            "caution", "adverse", "contraindicated", "precaution",
            "interaction", "complication", "safety", "monitor",
            "consult", "prescription", "supervision", "emergency",
            "allergic", "reaction", "sensitivity", "avoid"
        ])
        
        self.benefit_keywords = set(benefit_keywords)
        self.risk_keywords = set(risk_keywords)
        
        # Define promotional tone indicators
        self.promotional_indicators = set([
            "!", "!!", "!!!",  # Excessive punctuation
            "best", "most", "greatest", "leading", "premier",
            "exclusive", "revolutionary", "groundbreaking",
            "unprecedented", "innovative", "cutting-edge",
            "state-of-the-art", "advanced", "superior"
        ])
        
        # Sentiment analysis thresholds
        self.sentiment_threshold = self.config.get("feature_extraction", {}).get("sentiment", {}).get("threshold", 0.6)
        self.include_neutral = self.config.get("feature_extraction", {}).get("sentiment", {}).get("include_neutral", True)
        
        # Benefit-risk analysis settings
        self.context_window = self.config.get("feature_extraction", {}).get("benefit_risk", {}).get("context_window", 5)
        
    def extract_features(self, text: str) -> Dict:
        """Extract all semantic features from a text chunk.
        
        Args:
            text: Text chunk to analyze
            
        Returns:
            Dictionary containing extracted features
        """
        # Get sentiment with context
        sentiment = self._analyze_sentiment(text)
        
        # Get benefit-risk ratio with context
        benefit_risk = self._analyze_benefit_risk(text)
        
        # Get tone
        tone = self._analyze_tone(text)
        
        return {
            "sentiment": sentiment,
            "benefit_risk": benefit_risk,
            "tone": tone
        }
    
    def _analyze_sentiment(self, text: str) -> Dict:
        """Analyze sentiment of text using transformer model.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary containing sentiment score and label
        """
        try:
            # Split into sentences for better analysis
            sentences = self._split_into_sentences(text)
            
            # Get sentiment for each sentence
            sentiments = []
            for sentence in sentences:
                if len(sentence.strip()) > 0:
                    result = self.sentiment_analyzer(sentence)[0]
                    # Apply threshold
                    if result["score"] >= self.sentiment_threshold:
                        sentiments.append(result)
                    elif self.include_neutral:
                        # Convert low confidence to neutral
                        sentiments.append({
                            "label": "NEUTRAL",
                            "score": 0.5
                        })
            
            if not sentiments:
                return {"label": "NEUTRAL", "score": 0.5}
            
            # Aggregate sentiments with weighted scoring
            positive_score = sum(s["score"] for s in sentiments if s["label"] == "POSITIVE")
            negative_score = sum(s["score"] for s in sentiments if s["label"] == "NEGATIVE")
            neutral_score = sum(s["score"] for s in sentiments if s["label"] == "NEUTRAL")
            
            # Calculate final sentiment
            total_score = positive_score + negative_score + neutral_score
            if total_score > 0:
                if positive_score > negative_score:
                    label = "POSITIVE"
                    score = positive_score / total_score
                elif negative_score > positive_score:
                    label = "NEGATIVE"
                    score = negative_score / total_score
                else:
                    label = "NEUTRAL"
                    score = neutral_score / total_score
            else:
                label = "NEUTRAL"
                score = 0.5
            
            return {
                "label": label,
                "score": score,
                "confidence": max(positive_score, negative_score, neutral_score) / total_score if total_score > 0 else 0.5
            }
            
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {str(e)}")
            return {"label": "NEUTRAL", "score": 0.5, "confidence": 0.5}
    
    def _analyze_benefit_risk(self, text: str) -> Dict:
        """Analyze benefit-risk balance in text with context.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary containing benefit-risk metrics
        """
        # Split text into words for context analysis
        words = text.lower().split()
        benefit_count = 0
        risk_count = 0
        benefit_contexts = []
        risk_contexts = []
        
        # Analyze each word with context
        for i, word in enumerate(words):
            # Check if word is a benefit keyword
            if word in self.benefit_keywords:
                benefit_count += 1
                # Get context
                start = max(0, i - self.context_window)
                end = min(len(words), i + self.context_window + 1)
                context = " ".join(words[start:end])
                benefit_contexts.append(context)
            
            # Check if word is a risk keyword
            if word in self.risk_keywords:
                risk_count += 1
                # Get context
                start = max(0, i - self.context_window)
                end = min(len(words), i + self.context_window + 1)
                context = " ".join(words[start:end])
                risk_contexts.append(context)
        
        # Calculate ratio
        total = benefit_count + risk_count
        if total == 0:
            ratio = 1.0  # Neutral if no keywords found
        else:
            ratio = benefit_count / total
        
        return {
            "benefit_count": benefit_count,
            "risk_count": risk_count,
            "ratio": ratio,
            "balance": "BALANCED" if 0.4 <= ratio <= 0.6 else "BENEFIT_BIASED" if ratio > 0.6 else "RISK_BIASED",
            "benefit_contexts": benefit_contexts[:5],  # Limit to top 5 contexts
            "risk_contexts": risk_contexts[:5]  # Limit to top 5 contexts
        }
    
    def _analyze_tone(self, text: str) -> Dict:
        """Analyze promotional tone of text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary containing tone metrics
        """
        # Count promotional indicators
        indicator_count = sum(1 for word in self.promotional_indicators if word.lower() in text.lower())
        
        # Count exclamation marks
        exclamation_count = text.count("!")
        
        # Calculate promotional score
        total_words = len(text.split())
        if total_words == 0:
            score = 0
        else:
            score = (indicator_count + exclamation_count) / total_words
        
        # Get examples of promotional language
        examples = []
        words = text.lower().split()
        for i, word in enumerate(words):
            if word in self.promotional_indicators:
                start = max(0, i - 2)
                end = min(len(words), i + 3)
                context = " ".join(words[start:end])
                examples.append(context)
        
        return {
            "promotional_score": score,
            "indicator_count": indicator_count,
            "exclamation_count": exclamation_count,
            "tone": "HIGHLY_PROMOTIONAL" if score > 0.1 else "MODERATELY_PROMOTIONAL" if score > 0.05 else "INFORMATIONAL",
            "examples": examples[:5]  # Limit to top 5 examples
        }
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences.
        
        Args:
            text: Text to split
            
        Returns:
            List of sentences
        """
        # Simple sentence splitting on common delimiters
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()] 