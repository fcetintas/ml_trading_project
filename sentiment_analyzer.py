import logging
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import Dict, List, Tuple, Optional
import numpy as np
from dataclasses import dataclass
import re

@dataclass
class SentimentResult:
    """Sentiment analysis result"""
    sentiment: str  # 'positive', 'negative', 'neutral'
    confidence: float  # 0-1
    raw_scores: Dict[str, float]  # Raw model scores
    text: str  # Original text

class SentimentAnalyzer:
    """FinBERT-based sentiment analyzer for cryptocurrency news"""
    
    def __init__(self, model_name: str = "ProsusAI/finbert"):
        """
        Initialize the sentiment analyzer
        
        Args:
            model_name: Name of the FinBERT model to use
        """
        self.logger = logging.getLogger(__name__)
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._load_model()
    
    def _load_model(self):
        """Load the FinBERT model and tokenizer"""
        import os
        
        # Create local model cache directory
        local_model_path = "./models/finbert"
        os.makedirs(local_model_path, exist_ok=True)
        
        try:
            # Try loading from local cache first
            if os.path.exists(f"{local_model_path}/config.json"):
                self.logger.info("Loading FinBERT model from local cache...")
                print("📂 Loading model from local cache...")
                model_path = local_model_path
            else:
                self.logger.info(f"Downloading FinBERT model: {self.model_name} (first time only)")
                print(f"📥 Downloading FinBERT model to local cache...")
                model_path = self.model_name
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path, 
                cache_dir=local_model_path if model_path == self.model_name else None
            )
            print("✅ Tokenizer loaded")
            
            print("📊 Loading model weights...")
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_path,
                cache_dir=local_model_path if model_path == self.model_name else None
            )
            
            # Save to local cache if downloaded
            if model_path == self.model_name:
                print(f"💾 Saving model to {local_model_path}...")
                self.tokenizer.save_pretrained(local_model_path)
                self.model.save_pretrained(local_model_path)
            
            print(f"🔄 Moving model to {self.device}...")
            self.model.to(self.device)
            self.model.eval()
            
            print("✅ Model ready!")
            self.logger.info(f"Model loaded successfully on device: {self.device}")
        except Exception as e:
            self.logger.error(f"Failed to load FinBERT model: {e}")
            raise
    
    def _preprocess_text(self, text: str) -> str:
        """
        Preprocess text for sentiment analysis
        
        Args:
            text: Raw text to preprocess
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?-]', '', text)
        
        # Trim and return
        return text.strip()
    
    def _extract_crypto_context(self, text: str, symbol: str) -> str:
        """
        Extract sentences that mention the specific cryptocurrency
        
        Args:
            text: Full text
            symbol: Cryptocurrency symbol (e.g., 'BTC', 'ETH')
            
        Returns:
            Relevant sentences mentioning the cryptocurrency
        """
        if not text or not symbol:
            return text
        
        # Common cryptocurrency name mappings
        crypto_names = {
            'BTC': ['bitcoin', 'btc'],
            'ETH': ['ethereum', 'eth', 'ether'],
            'ADA': ['cardano', 'ada'],
            'DOT': ['polkadot', 'dot'],
            'BNB': ['binance', 'bnb']
        }
        
        # Get possible names for the symbol
        search_terms = crypto_names.get(symbol.upper(), [symbol.lower()])
        
        # Split text into sentences
        sentences = re.split(r'[.!?]+', text)
        
        # Find sentences mentioning the cryptocurrency
        relevant_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if any(term in sentence.lower() for term in search_terms):
                relevant_sentences.append(sentence)
        
        # Return relevant context or full text if no specific mentions found
        if relevant_sentences:
            return '. '.join(relevant_sentences)
        return text
    
    def analyze_sentiment(self, text: str, symbol: Optional[str] = None) -> SentimentResult:
        """
        Analyze sentiment of a single text
        
        Args:
            text: Text to analyze
            symbol: Optional cryptocurrency symbol for context extraction
            
        Returns:
            SentimentResult object
        """
        if not text:
            return SentimentResult(
                sentiment='neutral',
                confidence=0.0,
                raw_scores={'positive': 0.33, 'negative': 0.33, 'neutral': 0.34},
                text=text
            )
        
        try:
            # Preprocess text
            processed_text = self._preprocess_text(text)
            
            # Extract crypto-specific context if symbol provided
            if symbol:
                processed_text = self._extract_crypto_context(processed_text, symbol)
            
            # Truncate text if too long for the model
            max_length = 512
            if len(processed_text) > max_length:
                processed_text = processed_text[:max_length]
            
            # Tokenize
            inputs = self.tokenizer(
                processed_text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=max_length
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get model predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=-1)
            
            # Convert to numpy and extract scores
            probs = probabilities.cpu().numpy()[0]
            
            # FinBERT typically outputs: [positive, negative, neutral]
            label_mapping = {0: 'positive', 1: 'negative', 2: 'neutral'}
            
            # Create raw scores dictionary
            raw_scores = {
                'positive': float(probs[0]),
                'negative': float(probs[1]),
                'neutral': float(probs[2])
            }
            
            # Determine sentiment and confidence
            max_idx = np.argmax(probs)
            sentiment = label_mapping[max_idx]
            confidence = float(probs[max_idx])
            
            return SentimentResult(
                sentiment=sentiment,
                confidence=confidence,
                raw_scores=raw_scores,
                text=processed_text
            )
            
        except Exception as e:
            self.logger.error(f"Error analyzing sentiment: {e}")
            return SentimentResult(
                sentiment='neutral',
                confidence=0.0,
                raw_scores={'positive': 0.33, 'negative': 0.33, 'neutral': 0.34},
                text=text
            )
    
    def analyze_batch(self, texts: List[str], symbol: Optional[str] = None) -> List[SentimentResult]:
        """
        Analyze sentiment for multiple texts
        
        Args:
            texts: List of texts to analyze
            symbol: Optional cryptocurrency symbol for context extraction
            
        Returns:
            List of SentimentResult objects
        """
        results = []
        for text in texts:
            result = self.analyze_sentiment(text, symbol)
            results.append(result)
        
        return results
    
    def get_aggregated_sentiment(self, texts: List[str], symbol: Optional[str] = None) -> SentimentResult:
        """
        Get aggregated sentiment from multiple texts
        
        Args:
            texts: List of texts to analyze
            symbol: Optional cryptocurrency symbol for context extraction
            
        Returns:
            Aggregated SentimentResult
        """
        if not texts:
            return SentimentResult(
                sentiment='neutral',
                confidence=0.0,
                raw_scores={'positive': 0.33, 'negative': 0.33, 'neutral': 0.34},
                text=""
            )
        
        # Analyze all texts
        results = self.analyze_batch(texts, symbol)
        
        # Log individual article sentiments for debugging
        self.logger.info(f"    📰 Analyzing {len(texts)} articles for {symbol or 'crypto'}:")
        for i, (text, result) in enumerate(zip(texts, results)):
            # Get first 60 characters of the article for preview
            preview = text[:60].replace('\n', ' ').strip()
            if len(text) > 60:
                preview += "..."
            
            sentiment_emoji = "😊" if result.sentiment == "positive" else "😐" if result.sentiment == "neutral" else "😞"
            self.logger.info(f"      {i+1}. {sentiment_emoji} \"{preview}\" → {result.sentiment.upper()} ({result.confidence:.0%})")
        
        # Calculate weighted average based on confidence
        total_weight = 0
        weighted_positive = 0
        weighted_negative = 0
        weighted_neutral = 0
        
        for result in results:
            weight = result.confidence
            total_weight += weight
            weighted_positive += result.raw_scores['positive'] * weight
            weighted_negative += result.raw_scores['negative'] * weight
            weighted_neutral += result.raw_scores['neutral'] * weight
        
        if total_weight == 0:
            # Fallback to simple average
            avg_positive = np.mean([r.raw_scores['positive'] for r in results])
            avg_negative = np.mean([r.raw_scores['negative'] for r in results])
            avg_neutral = np.mean([r.raw_scores['neutral'] for r in results])
        else:
            avg_positive = weighted_positive / total_weight
            avg_negative = weighted_negative / total_weight
            avg_neutral = weighted_neutral / total_weight
        
        # Determine overall sentiment
        scores = {'positive': avg_positive, 'negative': avg_negative, 'neutral': avg_neutral}
        sentiment = max(scores, key=scores.get)
        confidence = scores[sentiment]
        
        # Calculate overall confidence (average of individual confidences)
        overall_confidence = np.mean([r.confidence for r in results])
        
        return SentimentResult(
            sentiment=sentiment,
            confidence=overall_confidence,
            raw_scores=scores,
            text=f"Aggregated from {len(texts)} texts"
        )
    
    def is_model_loaded(self) -> bool:
        """Check if the model is properly loaded"""
        return self.model is not None and self.tokenizer is not None