#!/usr/bin/env python3
"""
Historical Data Training Script for Technical AI Model

This script downloads historical price data and trains the technical analysis
AI model before live trading begins.
"""

import logging
import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Add current directory to path to import local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from technical_analyzer import TechnicalAnalyzer, TechnicalIndicators
from exchange import Exchange
from config import config

class TechnicalAITrainer:
    """Train the technical analysis AI model with historical data"""
    
    def __init__(self):
        """Initialize the trainer"""
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.technical_analyzer = TechnicalAnalyzer()
        self.exchange = Exchange(
            api_key=config.api.binance_api_key,
            secret_key=config.api.binance_secret_key,
            testnet=config.api.use_testnet
        )
        
        self.logger.info("🤖 Technical AI Trainer initialized")
    
    def download_historical_data(self, symbol: str, days: int = 30) -> List[Dict]:
        """
        Download historical price data for a symbol
        
        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            days: Number of days of historical data
            
        Returns:
            List of OHLCV data dictionaries
        """
        try:
            self.logger.info(f"📊 Downloading {days} days of data for {symbol}...")
            
            # Calculate required intervals (1-minute data)
            total_minutes = days * 24 * 60
            intervals_needed = min(total_minutes, 1000)  # API limit is 1000
            
            # Download data in chunks if needed
            all_data = []
            
            while len(all_data) < total_minutes and len(all_data) < 10000:  # Max 10k points
                klines = self.exchange.get_historical_klines(
                    symbol=symbol,
                    interval='1m',
                    limit=min(1000, total_minutes - len(all_data))
                )
                
                if not klines:
                    break
                
                all_data.extend(klines)
                
                # If we got less than requested, we've reached the end
                if len(klines) < 1000:
                    break
            
            self.logger.info(f"✅ Downloaded {len(all_data)} data points for {symbol}")
            return all_data
            
        except Exception as e:
            self.logger.error(f"Error downloading data for {symbol}: {e}")
            return []
    
    def create_training_labels(self, price_data: List[Dict], 
                              future_minutes: int = 30) -> List[int]:
        """
        Create training labels based on future price movements
        
        Args:
            price_data: Historical OHLCV data
            future_minutes: Minutes ahead to look for labeling
            
        Returns:
            List of labels: 0=SELL, 1=HOLD, 2=BUY
        """
        labels = []
        
        for i in range(len(price_data) - future_minutes):
            current_price = price_data[i]['close']
            future_price = price_data[i + future_minutes]['close']
            
            # Calculate price change percentage
            price_change = (future_price - current_price) / current_price * 100
            
            # Label based on price movement
            if price_change > 2.0:  # Significant upward movement
                labels.append(2)  # BUY signal was correct
            elif price_change < -2.0:  # Significant downward movement
                labels.append(0)  # SELL signal was correct
            else:  # Small movement or sideways
                labels.append(1)  # HOLD signal was correct
        
        return labels
    
    def extract_features_from_data(self, price_data: List[Dict]) -> List[List[float]]:
        """
        Extract technical analysis features from historical data
        
        Args:
            price_data: Historical OHLCV data
            
        Returns:
            List of feature vectors
        """
        features_list = []
        
        # Process data in sliding windows
        window_size = 100  # Need enough data for indicators
        
        for i in range(window_size, len(price_data)):
            # Get data window
            window_data = price_data[i-window_size:i]
            
            # Calculate indicators
            indicators = self.technical_analyzer.calculate_indicators(window_data)
            
            if indicators:
                # Extract features for AI
                features = self.technical_analyzer._extract_ai_features(indicators)
                features_list.append(features)
        
        return features_list
    
    def train_model_on_symbol(self, symbol: str, days: int = 30) -> Tuple[int, int]:
        """
        Train the model on historical data for one symbol
        
        Args:
            symbol: Trading symbol
            days: Days of historical data
            
        Returns:
            Tuple of (total_samples, successful_samples)
        """
        try:
            # Download historical data
            price_data = self.download_historical_data(symbol, days)
            
            if len(price_data) < 200:  # Need minimum data
                self.logger.warning(f"Insufficient data for {symbol}: {len(price_data)} points")
                return 0, 0
            
            # Extract features
            self.logger.info(f"🔧 Extracting features from {symbol} data...")
            features_list = self.extract_features_from_data(price_data)
            
            # Create labels
            self.logger.info(f"🏷️  Creating training labels for {symbol}...")
            labels = self.create_training_labels(price_data)
            
            # Align features and labels (features start later due to indicator calculation)
            min_length = min(len(features_list), len(labels))
            if min_length == 0:
                self.logger.warning(f"No valid training data for {symbol}")
                return 0, 0
            
            # Take the last min_length samples to align
            features_aligned = features_list[-min_length:]
            labels_aligned = labels[-min_length:]
            
            # Train the model incrementally
            successful_samples = 0
            for features, label in zip(features_aligned, labels_aligned):
                try:
                    self.technical_analyzer.learn_from_outcome(features, label)
                    successful_samples += 1
                except Exception as e:
                    self.logger.debug(f"Training sample failed: {e}")
            
            self.logger.info(f"✅ Trained on {successful_samples}/{min_length} samples from {symbol}")
            return min_length, successful_samples
            
        except Exception as e:
            self.logger.error(f"Error training on {symbol}: {e}")
            return 0, 0
    
    def train_on_multiple_symbols(self, symbols: List[str], days_per_symbol: int = 7) -> None:
        """
        Train the model on multiple symbols
        
        Args:
            symbols: List of trading symbols
            days_per_symbol: Days of data per symbol
        """
        total_samples = 0
        successful_samples = 0
        
        self.logger.info(f"🚀 Starting training on {len(symbols)} symbols...")
        
        for i, symbol in enumerate(symbols, 1):
            self.logger.info(f"📈 [{i}/{len(symbols)}] Training on {symbol}...")
            
            try:
                samples, success = self.train_model_on_symbol(symbol, days_per_symbol)
                total_samples += samples
                successful_samples += success
                
                # Progress update
                if successful_samples > 0:
                    success_rate = (successful_samples / total_samples) * 100
                    self.logger.info(f"    Progress: {successful_samples}/{total_samples} samples ({success_rate:.1f}% success)")
                
                # Check if model has enough data to become trained
                if hasattr(self.technical_analyzer, '_training_data'):
                    training_count = len(self.technical_analyzer._training_data.get('features', []))
                    if training_count >= 100:
                        self.logger.info(f"🎯 Model will retrain soon ({training_count}/100 samples)")
                
            except Exception as e:
                self.logger.error(f"Failed to train on {symbol}: {e}")
                continue
        
        # Final summary
        self.logger.info(f"🎉 Training completed!")
        self.logger.info(f"📊 Total samples processed: {total_samples}")
        self.logger.info(f"✅ Successful samples: {successful_samples}")
        
        if total_samples > 0:
            success_rate = (successful_samples / total_samples) * 100
            self.logger.info(f"📈 Success rate: {success_rate:.1f}%")
        
        # Check if model is now trained
        if self.technical_analyzer.model_trained:
            self.logger.info("🤖 AI model is now trained and ready!")
        else:
            self.logger.info("⏳ AI model needs more data - will continue learning from live trading")

def main():
    """Main training function"""
    print("🤖 Technical Analysis AI Training Script")
    print("=" * 50)
    
    try:
        # Initialize trainer
        trainer = TechnicalAITrainer()
        
        # Define training symbols (start with major coins)
        training_symbols = [
            'BTCUSDT',   # Bitcoin
            'ETHUSDT',   # Ethereum
            'BNBUSDT',   # Binance Coin
            'ADAUSDT',   # Cardano
            'SOLUSDT',   # Solana
            'XRPUSDT',   # Ripple
            'DOTUSDT',   # Polkadot
            'AVAXUSDT',  # Avalanche
        ]
        
        # Train on historical data
        trainer.train_on_multiple_symbols(training_symbols, days_per_symbol=5)
        
        print("\n🎯 Training completed! The AI model is ready for live trading.")
        print("   Start the main trading bot to continue learning from real trades.")
        
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        logging.error(f"Training failed: {e}")

if __name__ == "__main__":
    main()