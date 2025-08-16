import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
import warnings

# Suppress sklearn warnings
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    logging.warning("TA-Lib not available, using fallback indicators")

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    import joblib
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("scikit-learn not available, using simple rules")

@dataclass
class TechnicalIndicators:
    """Technical analysis indicators"""
    rsi_14: float
    rsi_21: float
    macd: float
    macd_signal: float
    macd_histogram: float
    bb_upper: float
    bb_middle: float
    bb_lower: float
    bb_position: float  # Where price is relative to bands (0-1)
    sma_20: float
    sma_50: float
    ema_12: float
    ema_26: float
    volume_sma: float
    volume_ratio: float  # Current volume vs average
    price_change_pct: float
    volatility: float

@dataclass
class TechnicalSignal:
    """AI-generated technical analysis signal"""
    action: str  # 'BUY', 'SELL', 'HOLD'
    confidence: float  # 0-1
    reasoning: str
    key_indicators: Dict[str, Any]
    ai_score: float  # Raw AI model score
    pattern_detected: Optional[str] = None

class TechnicalAnalyzer:
    """AI-powered technical analysis for cryptocurrency trading"""
    
    def __init__(self):
        """Initialize the technical analyzer"""
        self.logger = logging.getLogger(__name__)
        self.ai_model = None
        self.scaler = None
        self.model_trained = False
        
        # Feature names for consistency
        self.feature_names = [
            'rsi_14', 'rsi_21', 'macd', 'macd_signal', 'macd_histogram',
            'bb_position', 'sma_cross', 'ema_cross', 'volume_ratio',
            'price_change_pct', 'volatility', 'momentum_score'
        ]
        
        self._initialize_ai_model()
        
        self.logger.info("🤖 AI Technical Analyzer initialized")
    
    def _initialize_ai_model(self):
        """Initialize the AI model with default parameters"""
        if not SKLEARN_AVAILABLE:
            self.logger.warning("Using rule-based analysis (sklearn not available)")
            return
        
        try:
            # Try to load pre-trained model
            self.ai_model = joblib.load('./models/technical_ai_model.pkl')
            self.scaler = joblib.load('./models/technical_scaler.pkl')
            self.model_trained = True
            self.logger.info("📂 Loaded pre-trained technical analysis model")
        except FileNotFoundError:
            # Create new model
            self.ai_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                class_weight='balanced'
            )
            self.scaler = StandardScaler()
            self.model_trained = False
            self.logger.info("🎯 Created new AI model - will learn from live data")
    
    def calculate_indicators(self, price_data: List[Dict]) -> Optional[TechnicalIndicators]:
        """
        Calculate technical indicators from price data
        
        Args:
            price_data: List of OHLCV data dicts
            
        Returns:
            TechnicalIndicators object or None
        """
        try:
            if len(price_data) < 50:
                self.logger.warning("Insufficient price data for technical analysis")
                return None
            
            # Convert to pandas DataFrame
            df = pd.DataFrame(price_data)
            
            # Ensure we have the required columns
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_cols):
                self.logger.error("Missing required OHLCV columns")
                return None
            
            # Convert to float
            for col in required_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            closes = df['close'].values
            highs = df['high'].values
            lows = df['low'].values
            volumes = df['volume'].values
            
            if TALIB_AVAILABLE:
                # Use TA-Lib for accurate calculations
                indicators = self._calculate_talib_indicators(closes, highs, lows, volumes)
            else:
                # Use fallback calculations
                indicators = self._calculate_fallback_indicators(df)
            
            return indicators
            
        except Exception as e:
            self.logger.error(f"Error calculating technical indicators: {e}")
            return None
    
    def _calculate_talib_indicators(self, closes: np.array, highs: np.array, 
                                   lows: np.array, volumes: np.array) -> TechnicalIndicators:
        """Calculate indicators using TA-Lib"""
        
        # RSI
        rsi_14 = talib.RSI(closes, timeperiod=14)[-1]
        rsi_21 = talib.RSI(closes, timeperiod=21)[-1]
        
        # MACD
        macd, macd_signal, macd_hist = talib.MACD(closes)
        macd = macd[-1] if not np.isnan(macd[-1]) else 0
        macd_signal = macd_signal[-1] if not np.isnan(macd_signal[-1]) else 0
        macd_hist = macd_hist[-1] if not np.isnan(macd_hist[-1]) else 0
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = talib.BBANDS(closes)
        bb_upper = bb_upper[-1]
        bb_middle = bb_middle[-1]
        bb_lower = bb_lower[-1]
        
        # BB Position (where current price sits in the bands)
        bb_position = (closes[-1] - bb_lower) / (bb_upper - bb_lower) if bb_upper != bb_lower else 0.5
        bb_position = max(0, min(1, bb_position))  # Clamp to 0-1
        
        # Moving Averages
        sma_20 = talib.SMA(closes, timeperiod=20)[-1]
        sma_50 = talib.SMA(closes, timeperiod=50)[-1]
        ema_12 = talib.EMA(closes, timeperiod=12)[-1]
        ema_26 = talib.EMA(closes, timeperiod=26)[-1]
        
        # Volume indicators
        volume_sma = talib.SMA(volumes, timeperiod=20)[-1]
        volume_ratio = volumes[-1] / volume_sma if volume_sma > 0 else 1.0
        
        # Price change
        price_change_pct = (closes[-1] - closes[-2]) / closes[-2] * 100 if len(closes) > 1 else 0
        
        # Volatility (ATR-based)
        atr = talib.ATR(highs, lows, closes, timeperiod=14)[-1]
        volatility = atr / closes[-1] * 100 if closes[-1] > 0 else 0
        
        return TechnicalIndicators(
            rsi_14=rsi_14,
            rsi_21=rsi_21,
            macd=macd,
            macd_signal=macd_signal,
            macd_histogram=macd_hist,
            bb_upper=bb_upper,
            bb_middle=bb_middle,
            bb_lower=bb_lower,
            bb_position=bb_position,
            sma_20=sma_20,
            sma_50=sma_50,
            ema_12=ema_12,
            ema_26=ema_26,
            volume_sma=volume_sma,
            volume_ratio=volume_ratio,
            price_change_pct=price_change_pct,
            volatility=volatility
        )
    
    def _calculate_fallback_indicators(self, df: pd.DataFrame) -> TechnicalIndicators:
        """Calculate indicators using pandas (fallback when TA-Lib unavailable)"""
        
        # Simple RSI calculation
        def calculate_rsi(prices, period=14):
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            return 100 - (100 / (1 + rs))
        
        closes = df['close']
        
        # RSI
        rsi_14 = calculate_rsi(closes, 14).iloc[-1]
        rsi_21 = calculate_rsi(closes, 21).iloc[-1]
        
        # Simple MACD
        ema_12 = closes.ewm(span=12).mean()
        ema_26 = closes.ewm(span=26).mean()
        macd = ema_12 - ema_26
        macd_signal = macd.ewm(span=9).mean()
        macd_hist = macd - macd_signal
        
        # Bollinger Bands
        sma_20 = closes.rolling(20).mean()
        bb_std = closes.rolling(20).std()
        bb_upper = sma_20 + (bb_std * 2)
        bb_lower = sma_20 - (bb_std * 2)
        
        # Calculate BB position with safe division
        bb_width = bb_upper.iloc[-1] - bb_lower.iloc[-1]
        if bb_width != 0 and not np.isnan(bb_width):
            bb_position = (closes.iloc[-1] - bb_lower.iloc[-1]) / bb_width
            bb_position = max(0, min(1, bb_position))
        else:
            bb_position = 0.5  # Default to middle when bands have zero width
        
        # Moving averages
        sma_50 = closes.rolling(50).mean().iloc[-1]
        
        # Volume
        volume_sma = df['volume'].rolling(20).mean().iloc[-1]
        volume_ratio = df['volume'].iloc[-1] / volume_sma if volume_sma > 0 else 1.0
        
        # Price change
        price_change_pct = (closes.iloc[-1] - closes.iloc[-2]) / closes.iloc[-2] * 100
        
        # Volatility
        volatility = closes.rolling(14).std().iloc[-1] / closes.iloc[-1] * 100
        
        return TechnicalIndicators(
            rsi_14=rsi_14 if not np.isnan(rsi_14) else 50,
            rsi_21=rsi_21 if not np.isnan(rsi_21) else 50,
            macd=macd.iloc[-1] if not np.isnan(macd.iloc[-1]) else 0,
            macd_signal=macd_signal.iloc[-1] if not np.isnan(macd_signal.iloc[-1]) else 0,
            macd_histogram=macd_hist.iloc[-1] if not np.isnan(macd_hist.iloc[-1]) else 0,
            bb_upper=bb_upper.iloc[-1],
            bb_middle=sma_20.iloc[-1],
            bb_lower=bb_lower.iloc[-1],
            bb_position=bb_position,
            sma_20=sma_20.iloc[-1],
            sma_50=sma_50 if not np.isnan(sma_50) else closes.iloc[-1],
            ema_12=ema_12.iloc[-1],
            ema_26=ema_26.iloc[-1],
            volume_sma=volume_sma,
            volume_ratio=volume_ratio,
            price_change_pct=price_change_pct,
            volatility=volatility if not np.isnan(volatility) else 1.0
        )
    
    def _extract_ai_features(self, indicators: TechnicalIndicators) -> List[float]:
        """Extract features for AI model"""
        
        # Calculate derived features
        sma_cross = 1 if indicators.sma_20 > indicators.sma_50 else 0
        ema_cross = 1 if indicators.ema_12 > indicators.ema_26 else 0
        
        # Momentum score (combination of indicators)
        momentum_score = (
            (indicators.rsi_14 - 50) / 50 +  # RSI momentum
            np.tanh(indicators.macd) +        # MACD momentum  
            (indicators.bb_position - 0.5) * 2  # BB position momentum
        ) / 3
        
        features = [
            indicators.rsi_14,
            indicators.rsi_21,
            indicators.macd,
            indicators.macd_signal,
            indicators.macd_histogram,
            indicators.bb_position,
            sma_cross,
            ema_cross,
            indicators.volume_ratio,
            indicators.price_change_pct,
            indicators.volatility,
            momentum_score
        ]
        
        # Handle NaN values
        features = [0.0 if np.isnan(x) or np.isinf(x) else float(x) for x in features]
        
        return features
    
    def analyze_with_ai(self, indicators: TechnicalIndicators) -> TechnicalSignal:
        """
        Generate trading signal using AI model
        
        Args:
            indicators: Technical indicators
            
        Returns:
            TechnicalSignal with AI prediction
        """
        
        if not SKLEARN_AVAILABLE or not self.model_trained:
            return self._analyze_with_rules(indicators)
        
        try:
            # Extract features
            features = self._extract_ai_features(indicators)
            features_array = np.array(features).reshape(1, -1)
            
            # Scale features
            features_scaled = self.scaler.transform(features_array)
            
            # Get AI prediction
            prediction = self.ai_model.predict(features_scaled)[0]
            probabilities = self.ai_model.predict_proba(features_scaled)[0]
            
            # Map prediction to action
            action_map = {0: 'SELL', 1: 'HOLD', 2: 'BUY'}
            action = action_map[prediction]
            confidence = float(probabilities[prediction])
            
            # Get feature importance for reasoning
            feature_importance = dict(zip(self.feature_names, self.ai_model.feature_importances_))
            
            # Generate reasoning
            reasoning = self._generate_ai_reasoning(indicators, action, feature_importance)
            
            # Detect patterns
            pattern = self._detect_patterns(indicators)
            
            return TechnicalSignal(
                action=action,
                confidence=confidence,
                reasoning=reasoning,
                key_indicators=self._get_key_indicators(indicators),
                ai_score=float(probabilities[2] - probabilities[0]),  # Buy score - Sell score
                pattern_detected=pattern
            )
            
        except Exception as e:
            self.logger.error(f"AI analysis failed, falling back to rules: {e}")
            return self._analyze_with_rules(indicators)
    
    def _analyze_with_rules(self, indicators: TechnicalIndicators) -> TechnicalSignal:
        """Fallback rule-based analysis when AI is not available"""
        
        signals = []
        reasoning_parts = []
        
        # RSI signals
        if indicators.rsi_14 < 30:
            signals.append(2)  # Strong buy
            reasoning_parts.append(f"RSI oversold ({indicators.rsi_14:.1f})")
        elif indicators.rsi_14 > 70:
            signals.append(0)  # Strong sell
            reasoning_parts.append(f"RSI overbought ({indicators.rsi_14:.1f})")
        else:
            signals.append(1)  # Neutral
        
        # MACD signals
        if indicators.macd > indicators.macd_signal and indicators.macd_histogram > 0:
            signals.append(2)  # Buy
            reasoning_parts.append("MACD bullish")
        elif indicators.macd < indicators.macd_signal and indicators.macd_histogram < 0:
            signals.append(0)  # Sell
            reasoning_parts.append("MACD bearish")
        else:
            signals.append(1)  # Neutral
        
        # Bollinger Band signals
        if indicators.bb_position < 0.1:
            signals.append(2)  # Buy (price near lower band)
            reasoning_parts.append("Near BB lower band")
        elif indicators.bb_position > 0.9:
            signals.append(0)  # Sell (price near upper band)
            reasoning_parts.append("Near BB upper band")
        else:
            signals.append(1)  # Neutral
        
        # Volume confirmation
        if indicators.volume_ratio > 1.5:
            reasoning_parts.append("High volume")
        
        # Calculate final signal
        avg_signal = np.mean(signals)
        if avg_signal >= 1.5:
            action = 'BUY'
            confidence = min(0.8, (avg_signal - 1) / 1 + 0.5)
        elif avg_signal <= 0.5:
            action = 'SELL'
            confidence = min(0.8, (1 - avg_signal) / 1 + 0.5)
        else:
            action = 'HOLD'
            confidence = 0.4
        
        reasoning = "; ".join(reasoning_parts) if reasoning_parts else "Neutral conditions"
        
        return TechnicalSignal(
            action=action,
            confidence=confidence,
            reasoning=f"Rule-based: {reasoning}",
            key_indicators=self._get_key_indicators(indicators),
            ai_score=avg_signal - 1,  # -1 to 1 scale
            pattern_detected=self._detect_patterns(indicators)
        )
    
    def _generate_ai_reasoning(self, indicators: TechnicalIndicators, action: str, 
                              feature_importance: Dict[str, float]) -> str:
        """Generate human-readable reasoning from AI decision"""
        
        # Get top 3 most important features
        top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:3]
        
        reasoning_parts = [f"AI pattern recognition ({action.lower()})"]
        
        for feature_name, importance in top_features:
            if feature_name == 'rsi_14':
                if indicators.rsi_14 < 35:
                    reasoning_parts.append(f"RSI oversold signal ({indicators.rsi_14:.1f})")
                elif indicators.rsi_14 > 65:
                    reasoning_parts.append(f"RSI overbought signal ({indicators.rsi_14:.1f})")
            elif feature_name == 'macd':
                if indicators.macd > indicators.macd_signal:
                    reasoning_parts.append("MACD bullish momentum")
                else:
                    reasoning_parts.append("MACD bearish momentum")
            elif feature_name == 'bb_position':
                if indicators.bb_position < 0.2:
                    reasoning_parts.append("Price compression near support")
                elif indicators.bb_position > 0.8:
                    reasoning_parts.append("Price expansion near resistance")
            elif feature_name == 'volume_ratio' and indicators.volume_ratio > 1.3:
                reasoning_parts.append("Volume confirmation")
        
        return "; ".join(reasoning_parts)
    
    def _detect_patterns(self, indicators: TechnicalIndicators) -> Optional[str]:
        """Detect chart patterns from indicators"""
        
        # RSI Divergence patterns
        if indicators.rsi_14 < 25:
            return "Potential bullish divergence"
        elif indicators.rsi_14 > 75:
            return "Potential bearish divergence"
        
        # MACD patterns
        if (indicators.macd > indicators.macd_signal and 
            indicators.macd_histogram > 0 and 
            indicators.macd < 0):
            return "MACD bullish crossover"
        elif (indicators.macd < indicators.macd_signal and 
              indicators.macd_histogram < 0 and 
              indicators.macd > 0):
            return "MACD bearish crossover"
        
        # Bollinger Band squeeze
        bb_width = (indicators.bb_upper - indicators.bb_lower) / indicators.bb_middle
        if bb_width < 0.04:  # Narrow bands
            return "Bollinger squeeze (breakout pending)"
        
        # Trend patterns
        if (indicators.sma_20 > indicators.sma_50 and 
            indicators.ema_12 > indicators.ema_26 and
            indicators.rsi_14 > 50):
            return "Strong uptrend"
        elif (indicators.sma_20 < indicators.sma_50 and 
              indicators.ema_12 < indicators.ema_26 and
              indicators.rsi_14 < 50):
            return "Strong downtrend"
        
        return None
    
    def _get_key_indicators(self, indicators: TechnicalIndicators) -> Dict[str, Any]:
        """Get key indicator values for display"""
        return {
            'rsi_14': round(indicators.rsi_14, 1),
            'macd_signal': 'bullish' if indicators.macd > indicators.macd_signal else 'bearish',
            'bb_position': f"{indicators.bb_position:.0%}",
            'volume_ratio': round(indicators.volume_ratio, 2),
            'trend': 'up' if indicators.sma_20 > indicators.sma_50 else 'down',
            'volatility': f"{indicators.volatility:.1f}%"
        }
    
    def learn_from_outcome(self, features: List[float], actual_outcome: int):
        """
        Update AI model with trading outcome (for continuous learning)
        
        Args:
            features: The features used for prediction
            actual_outcome: 0=sell was right, 1=hold was right, 2=buy was right
        """
        if not SKLEARN_AVAILABLE or not hasattr(self, '_training_data'):
            return
        
        # Store training data for batch updates
        if not hasattr(self, '_training_data'):
            self._training_data = {'features': [], 'outcomes': []}
        
        self._training_data['features'].append(features)
        self._training_data['outcomes'].append(actual_outcome)
        
        # Retrain every 100 samples
        if len(self._training_data['features']) >= 100:
            self._retrain_model()
    
    def _retrain_model(self):
        """Retrain the AI model with accumulated data"""
        try:
            X = np.array(self._training_data['features'])
            y = np.array(self._training_data['outcomes'])
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Retrain model
            self.ai_model.fit(X_scaled, y)
            self.model_trained = True
            
            # Save updated model
            import os
            os.makedirs('./models', exist_ok=True)
            joblib.dump(self.ai_model, './models/technical_ai_model.pkl')
            joblib.dump(self.scaler, './models/technical_scaler.pkl')
            
            # Clear training data
            self._training_data = {'features': [], 'outcomes': []}
            
            self.logger.info("🎯 AI model retrained with new data")
            
        except Exception as e:
            self.logger.error(f"Model retraining failed: {e}")