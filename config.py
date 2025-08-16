import os
import logging
from typing import Dict, Any
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass
class TradingConfig:
    """Trading configuration parameters"""
    max_trade_amount: float = 30.0  # Maximum trade amount in USD
    sentiment_threshold_buy: float = 0.85  # Minimum positive sentiment for buy
    sentiment_threshold_sell: float = -0.85  # Maximum negative sentiment for sell
    confidence_threshold: float = 0.6  # Minimum confidence for any trade
    trade_interval: int = 300   # Seconds between trading cycles (5 minutes)
    symbols: list = None  # Trading symbols
    
    def __post_init__(self):
        if self.symbols is None:
            # Top cryptocurrencies by market cap and trading volume
            self.symbols = [
                # Major coins
                'BTCUSDT',   # Bitcoin
                'ETHUSDT',   # Ethereum  
                'BNBUSDT',   # Binance Coin
                'ARBUSDT',   # Arbitrum
                'ENAUSDT',   # Ethereum Name Service
                'WLDUSDT',   # Worldcoin
                'XAIUSDT',   # Worldcoin
                'APTUSDT',   # Aptos
                'MINAUSDT',   # Mina Protocol
            ]

@dataclass
class APIConfig:
    """API configuration settings"""
    binance_api_key: str = ""
    binance_secret_key: str = ""
    cryptopanic_api_key: str = ""
    use_testnet: bool = True
    
    def __post_init__(self):
        self.binance_api_key = os.getenv('BINANCE_API_KEY', '')
        self.binance_secret_key = os.getenv('BINANCE_SECRET_KEY', '')
        self.cryptopanic_api_key = os.getenv('CRYPTOPANIC_API_KEY', '')
        self.use_testnet = os.getenv('USE_TESTNET', 'true').lower() == 'true'

@dataclass
class LoggingConfig:
    """Logging configuration"""
    level: str = "DEBUG"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_path: str = "trading_bot.log"
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5

class Config:
    """Main configuration class"""
    
    def __init__(self):
        self.trading = TradingConfig()
        self.api = APIConfig()
        self.logging = LoggingConfig()
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup logging configuration"""
        
        # Clear any existing handlers to avoid duplicates
        root_logger = logging.getLogger()
        root_logger.handlers.clear()
        
        # Create formatters
        # Console: Clean, human-readable format
        console_format = "%(message)s"
        # File: Detailed format with timestamps
        file_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        
        # Console handler - only show INFO and above, clean format
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(logging.Formatter(console_format))
        
        # File handler - show DEBUG and above, detailed format
        from logging.handlers import RotatingFileHandler
        file_handler = RotatingFileHandler(
            self.logging.file_path,
            maxBytes=self.logging.max_file_size,
            backupCount=self.logging.backup_count
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(file_format))
        
        # Configure root logger
        root_logger.setLevel(logging.DEBUG)
        root_logger.addHandler(console_handler)
        root_logger.addHandler(file_handler)
        
        # Silence noisy third-party loggers
        logging.getLogger('urllib3').setLevel(logging.WARNING)
        logging.getLogger('requests').setLevel(logging.WARNING)
        logging.getLogger('transformers').setLevel(logging.WARNING)
    
    def validate(self) -> bool:
        """Validate configuration settings"""
        errors = []
        
        if not self.api.binance_api_key:
            errors.append("BINANCE_API_KEY is required")
        
        if not self.api.binance_secret_key:
            errors.append("BINANCE_SECRET_KEY is required")
        
        if not self.api.cryptopanic_api_key:
            errors.append("CRYPTOPANIC_API_KEY is required")
        
        if self.trading.max_trade_amount <= 0:
            errors.append("max_trade_amount must be positive")
        
        if not (0 <= self.trading.confidence_threshold <= 1):
            errors.append("confidence_threshold must be between 0 and 1")
        
        if errors:
            for error in errors:
                logging.error(f"Configuration error: {error}")
            return False
        
        logging.info("Configuration validation passed")
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            'trading': {
                'max_trade_amount': self.trading.max_trade_amount,
                'sentiment_threshold_buy': self.trading.sentiment_threshold_buy,
                'sentiment_threshold_sell': self.trading.sentiment_threshold_sell,
                'confidence_threshold': self.trading.confidence_threshold,
                'trade_interval': self.trading.trade_interval,
                'symbols': self.trading.symbols
            },
            'api': {
                'use_testnet': self.api.use_testnet,
                'has_api_keys': bool(self.api.binance_api_key and self.api.binance_secret_key)
            },
            'logging': {
                'level': self.logging.level,
                'file_path': self.logging.file_path
            }
        }

# Global configuration instance
config = Config()