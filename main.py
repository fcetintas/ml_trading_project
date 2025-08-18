#!/usr/bin/env python3
"""
Cryptocurrency Sentiment Trading Bot

A trading bot that analyzes cryptocurrency news sentiment using FinBERT
and executes trades on Binance based on sentiment signals.
"""

import logging
import time
import signal
import sys
import os
import warnings
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import asyncio
from dataclasses import dataclass

# Suppress urllib3 SSL warnings
warnings.filterwarnings('ignore', message='urllib3 v2 only supports OpenSSL 1.1.1+')
warnings.filterwarnings('ignore', category=FutureWarning, module='torch')

from config import config
from sentiment_analyzer import SentimentAnalyzer, SentimentResult
from news_fetcher import NewsFetcher, NewsArticle
from multi_news_fetcher import MultiNewsAggregator
from exchange import Exchange, OrderResult, MarketData
from technical_analyzer import TechnicalAnalyzer, TechnicalSignal
from messaging_interface import create_messaging_interface
from trade_tracker import TradeTracker

@dataclass
class TradingSignal:
    """Trading signal based on sentiment analysis"""
    symbol: str
    action: str  # 'BUY', 'SELL', 'HOLD'
    sentiment: str
    confidence: float
    news_count: int
    reasoning: str
    timestamp: datetime

@dataclass
class TradingSession:
    """Represents a trading session with performance tracking"""
    start_time: datetime
    total_trades: int = 0
    successful_trades: int = 0
    total_pnl: float = 0.0
    positions: Dict[str, float] = None  # symbol -> quantity held
    
    def __post_init__(self):
        if self.positions is None:
            self.positions = {}

class CryptoSentimentBot:
    """Main cryptocurrency sentiment trading bot"""
    
    def __init__(self):
        """Initialize the trading bot"""
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.running = False
        
        # Initialize components
        self.sentiment_analyzer = None
        self.technical_analyzer = None
        self.news_fetcher = None
        self.multi_news_fetcher = None
        self.trader = None
        self.telegram_notifier = None
        self.telegram_command_handler = None
        self.trade_tracker = None
        
        # Trading session
        self.session = TradingSession(start_time=datetime.now())
        
        # Recent signals tracking for 30-minute cooldown
        self.recent_signals = {}  # {symbol: {'BUY': timestamp, 'SELL': timestamp}}
        
        # Signal handling for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        self.logger.info("🤖 Bot initialized and ready")
    
    def _evaluate_and_train_ai(self, symbol: str, technical_signal, indicators, market_data_before, delay_minutes=5):
        """
        Evaluate trading outcome after a delay and train the AI model
        
        Args:
            symbol: Trading symbol
            technical_signal: The technical signal that was generated
            indicators: Technical indicators used for the signal
            market_data_before: Market data at time of signal
            delay_minutes: Minutes to wait before evaluation
        """
        if not self.technical_analyzer or not hasattr(self.technical_analyzer, 'learn_from_outcome'):
            return
        
        try:
            import threading
            import time
            
            def delayed_evaluation():
                # Wait for the specified delay
                time.sleep(delay_minutes * 60)
                
                # Get current market data
                current_market_data = self.trader.get_market_data(symbol)
                if not current_market_data:
                    return
                
                # Calculate price change percentage
                price_change = (current_market_data.price - market_data_before.price) / market_data_before.price * 100
                
                # Determine if the technical signal was correct
                # 0=sell was right, 1=hold was right, 2=buy was right
                if price_change > 1.0:  # Price went up significantly
                    correct_action = 2  # BUY was right
                elif price_change < -1.0:  # Price went down significantly
                    correct_action = 0  # SELL was right
                else:  # Price stayed relatively stable
                    correct_action = 1  # HOLD was right
                
                # Extract features used for the prediction
                features = self.technical_analyzer._extract_ai_features(indicators)
                
                # Train the AI model with this outcome
                self.technical_analyzer.learn_from_outcome(features, correct_action)
                
                # Log the learning
                action_names = {0: 'SELL', 1: 'HOLD', 2: 'BUY'}
                self.logger.debug(f"🎯 AI Learning: {symbol} - Signal was {technical_signal.action}, "
                                f"correct was {action_names[correct_action]}, price change: {price_change:+.2f}%")
            
            # Start evaluation in background thread
            thread = threading.Thread(target=delayed_evaluation, daemon=True)
            thread.start()
            
        except Exception as e:
            self.logger.error(f"Error in AI training evaluation: {e}")
    
    def _should_dismiss_signal(self, symbol: str, action: str) -> bool:
        """
        Check if signal should be dismissed due to 30-minute cooldown
        
        Args:
            symbol: Trading symbol
            action: 'BUY' or 'SELL'
            
        Returns:
            True if signal should be dismissed, False otherwise
        """
        if symbol not in self.recent_signals:
            return False
        
        if action not in self.recent_signals[symbol]:
            return False
        
        last_signal_time = self.recent_signals[symbol][action]
        time_since_last = datetime.now() - last_signal_time
        cooldown_period = timedelta(minutes=30)
        
        if time_since_last < cooldown_period:
            remaining_time = cooldown_period - time_since_last
            remaining_minutes = remaining_time.total_seconds() / 60
            self.logger.info(f"  ⏰ {action} signal dismissed - cooldown active for {remaining_minutes:.1f} more minutes")
            return True
        
        return False
    
    def _record_signal(self, symbol: str, action: str):
        """Record a signal execution for cooldown tracking"""
        if symbol not in self.recent_signals:
            self.recent_signals[symbol] = {}
        
        self.recent_signals[symbol][action] = datetime.now()
        self.logger.debug(f"Recorded {action} signal for {symbol}")
    
    def _register_command_handlers(self):
        """Register command handlers with the messaging interface"""
        if not self.telegram_notifier:
            return
        
        # Register balance command - returns raw balance data
        def get_balance():
            try:
                if not self.trader:
                    return {"error": "Exchange not available"}
                
                balances = self.trader.get_balances()
                balance_data = []
                total_value_usdt = 0.0
                
                for balance in balances:
                    if balance.total > 0.00001:
                        if balance.asset == 'USDT':
                            usd_value = balance.total
                        else:
                            try:
                                market_data = self.trader.get_market_data(f"{balance.asset}USDT")
                                usd_value = balance.total * market_data.price if market_data else 0
                            except:
                                usd_value = 0
                        
                        total_value_usdt += usd_value
                        balance_data.append({
                            'asset': balance.asset,
                            'amount': balance.total,
                            'usd_value': usd_value
                        })
                
                return {
                    'balances': balance_data[:10],  # Limit to top 10
                    'total_value_usdt': total_value_usdt
                }
                
            except Exception as e:
                self.logger.error(f"Error in balance command: {e}")
                return {"error": "Failed to get balance information"}
        
        # Register positions command - returns raw position data
        def get_positions():
            try:
                open_positions = self.trade_tracker.get_open_positions()
                position_data = []
                total_value = 0.0
                
                for symbol, position in open_positions.items():
                    try:
                        market_data = self.trader.get_market_data(symbol) if self.trader else None
                        current_price = market_data.price if market_data else 0
                        position_value = position.quantity * current_price
                        total_value += position_value
                        
                        position_data.append({
                            'symbol': symbol,
                            'quantity': position.quantity,
                            'current_price': current_price,
                            'position_value': position_value
                        })
                    except Exception:
                        position_data.append({
                            'symbol': symbol,
                            'quantity': position.quantity,
                            'current_price': None,
                            'position_value': None
                        })
                
                return {
                    'positions': position_data,
                    'total_value': total_value
                }
                
            except Exception as e:
                self.logger.error(f"Error in positions command: {e}")
                return {"error": "Failed to get position information"}
        
        # Register status command - returns raw status data
        def get_status():
            try:
                runtime = datetime.now() - self.session.start_time
                
                return {
                    'runtime_seconds': runtime.total_seconds(),
                    'total_trades': self.session.total_trades,
                    'successful_trades': self.session.successful_trades,
                    'success_rate': (self.session.successful_trades / max(1, self.session.total_trades) * 100),
                    'active_positions': len(self.trade_tracker.get_open_positions())
                }
                
            except Exception as e:
                self.logger.error(f"Error in status command: {e}")
                return {"error": "Failed to get status information"}
        
        # Register trades command - returns raw trade data
        def get_trades():
            try:
                recent_trades = self.trade_tracker.get_trade_history(limit=10)
                trade_data = []
                
                for trade in recent_trades:
                    trade_data.append({
                        'side': trade.side,
                        'symbol': trade.symbol,
                        'quantity': trade.quantity,
                        'price': trade.price,
                        'amount_usd': trade.amount_usd,
                        'timestamp': trade.timestamp,
                        'status': trade.status.value
                    })
                
                stats = self.trade_tracker.get_trading_stats()
                
                return {
                    'trades': trade_data,
                    'stats': stats
                }
                
            except Exception as e:
                self.logger.error(f"Error in trades command: {e}")
                return {"error": "Failed to get trade information"}
        
        # Register history command - returns raw history data
        def get_history():
            try:
                open_positions = self.trade_tracker.get_open_positions()
                stats = self.trade_tracker.get_trading_stats()
                
                position_data = []
                for symbol, position in open_positions.items():
                    try:
                        if self.trader:
                            market_data = self.trader.get_market_data(symbol)
                            if market_data:
                                position.update_current_value(market_data.price)
                    except:
                        pass
                    
                    days_held = (datetime.now() - position.entry_time).days
                    position_data.append({
                        'symbol': symbol,
                        'entry_price': position.entry_price,
                        'current_value': position.current_value,
                        'pnl': position.pnl,
                        'pnl_pct': position.pnl_pct,
                        'days_held': days_held
                    })
                
                return {
                    'stats': stats,
                    'open_positions': position_data
                }
                
            except Exception as e:
                self.logger.error(f"Error in history command: {e}")
                return {"error": "Failed to get history information"}
        
        # Register orders command - returns raw orders data from exchange
        def get_orders():
            try:
                if not self.trader:
                    return {"error": "Exchange not available"}
                
                # Get all orders from exchange
                all_orders = self.trader.get_all_orders()
                
                if not all_orders:
                    return {
                        'orders': [],
                        'summary': {
                            'total_orders': 0,
                            'open_orders': 0,
                            'filled_orders': 0
                        }
                    }
                
                # Calculate summary statistics
                open_orders = sum(1 for order in all_orders if order.get('status', '').upper() in ['NEW', 'OPEN'])
                filled_orders = sum(1 for order in all_orders if order.get('status', '').upper() in ['FILLED', 'COMPLETED'])
                
                # Limit to recent orders (last 50) to avoid overwhelming the message
                recent_orders = all_orders[-50:] if len(all_orders) > 50 else all_orders
                
                return {
                    'orders': recent_orders,
                    'summary': {
                        'total_orders': len(all_orders),
                        'open_orders': open_orders,
                        'filled_orders': filled_orders
                    }
                }
                
            except Exception as e:
                self.logger.error(f"Error in orders command: {e}")
                return {"error": "Failed to get orders information"}
        
        # Register all commands (help is handled internally by messaging interface)
        self.telegram_notifier.register_command("balance", get_balance)
        self.telegram_notifier.register_command("positions", get_positions)
        self.telegram_notifier.register_command("status", get_status)
        self.telegram_notifier.register_command("trades", get_trades)
        self.telegram_notifier.register_command("history", get_history)
        self.telegram_notifier.register_command("orders", get_orders)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        self.logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.running = False
        self.telegram_notifier.stop()  # Stop messaging interface if running
    
    def initialize_components(self) -> bool:
        """
        Initialize all bot components
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Validate configuration
            if not self.config.validate():
                self.logger.error("Configuration validation failed")
                return False
            
            # Initialize sentiment analyzer
            self.logger.info("🧠 Loading AI sentiment model...")
            self.sentiment_analyzer = SentimentAnalyzer()
            
            if not self.sentiment_analyzer.is_model_loaded():
                self.logger.error("Failed to load sentiment analysis model")
                return False
            
            # Initialize technical analyzer
            self.logger.info("📈 Loading AI technical analysis...")
            self.technical_analyzer = TechnicalAnalyzer()
            
            # Initialize multi-source news aggregator
            self.logger.info("🌐 Connecting to multiple news sources...")
            api_keys = {
                'newsapi': os.getenv('NEWSAPI_KEY', ''),
                'cryptocompare': os.getenv('CRYPTOCOMPARE_KEY', ''),
                'messari': os.getenv('MESSARI_KEY', ''),
            }
            self.multi_news_fetcher = MultiNewsAggregator(api_keys)
            
            # Keep legacy CryptoPanic as fallback
            self.news_fetcher = NewsFetcher(self.config.api.cryptopanic_api_key)
            
            # Initialize exchange
            self.logger.info("💱 Connecting to exchange...")
            self.trader = Exchange(
                api_key=self.config.api.binance_api_key,
                secret_key=self.config.api.binance_secret_key,
                testnet=self.config.api.use_testnet
            )
            
            # Test connectivity
            if not self.trader.test_connectivity():
                self.logger.error("Failed to connect to exchange API")
                return False
            
            # Initialize trade tracker
            self.logger.info("📊 Initializing trade tracker...")
            self.trade_tracker = TradeTracker()
            
            # Initialize messaging interface (optional)
            self.logger.info("📱 Setting up messaging interface...")
            self.telegram_notifier = create_messaging_interface()
            
            # Register command handlers
            self._register_command_handlers()
            
            # Start messaging interface immediately if enabled
            if self.telegram_notifier.enabled:
                self.logger.info("✅ Telegram messaging interface started")
                
                # Test connection and send startup message
                if self.telegram_notifier.test_connection():
                    mode = "TESTNET" if self.config.api.use_testnet else "LIVE"
                    self.telegram_notifier.send_startup_message(
                        symbols=self.config.trading.symbols,
                        mode=mode
                    )
                    self.logger.info("✅ Telegram notifications and commands enabled")
                else:
                    self.logger.warning("⚠️  Telegram connection test failed, but interface is running")
            else:
                self.logger.info("📱 Telegram bot disabled")
            
            self.logger.info("✅ All systems ready - Bot is live!")
            return True
            
        except Exception as e:
            self.logger.error(f"Error initializing components: {e}")
            return False
    
    def fetch_and_analyze_news(self, symbol: str) -> Optional[SentimentResult]:
        """
        Fetch news and analyze sentiment for a symbol
        
        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            
        Returns:
            Aggregated sentiment result or None
        """
        try:
            # Extract base currency from symbol (e.g., 'BTC' from 'BTCUSDT')
            base_currency = symbol.replace('USDT', '')
            
            # Try multi-source aggregator first, fallback to CryptoPanic
            articles = self.multi_news_fetcher.fetch_aggregated_news(
                currencies=[base_currency],
                limit=20,
                hours_back=24  # 24 hours for better performance
            )
            
            # Fallback to CryptoPanic if multi-source fails
            if not articles:
                self.logger.info(f"  🔄 Falling back to CryptoPanic for {base_currency}")
                articles = self.news_fetcher.fetch_recent_news(
                    currencies=[base_currency],
                    limit=10,
                    hours_back=72
                )
            
            if not articles:
                self.logger.warning(f"  ⚠️  No recent news found for {base_currency}")
                return None
            
            # Log news articles being analyzed
            self.logger.info(f"  📄 Found {len(articles)} articles from sources:")
            sources = {}
            for article in articles:
                source = article.source
                if source not in sources:
                    sources[source] = []
                sources[source].append(article.title[:50] + "..." if len(article.title) > 50 else article.title)
            
            for source, titles in sources.items():
                self.logger.info(f"    📰 {source}: {len(titles)} articles")
                for title in titles[:2]:  # Show first 2 titles from each source
                    self.logger.info(f"      • \"{title}\"")
                if len(titles) > 2:
                    self.logger.info(f"      • ... and {len(titles)-2} more")
            
            # Extract text content for sentiment analysis
            texts = []
            for article in articles:
                # Combine title and content
                full_text = f"{article.title}. {article.content}".strip()
                texts.append(full_text)
            
            # Analyze sentiment
            sentiment_result = self.sentiment_analyzer.get_aggregated_sentiment(
                texts=texts,
                symbol=base_currency
            )
            
            # Create emoji based on sentiment
            sentiment_emoji = "😊" if sentiment_result.sentiment == "positive" else "😐" if sentiment_result.sentiment == "neutral" else "😞"
            self.logger.info(
                f"  {sentiment_emoji} {len(articles)} articles analyzed: {sentiment_result.sentiment.upper()} ({sentiment_result.confidence:.0%})"
            )
            
            return sentiment_result
            
        except Exception as e:
            self.logger.error(f"Error fetching/analyzing news for {symbol}: {e}")
            return None
    
    def generate_trading_signal(self, symbol: str, sentiment_result: SentimentResult, 
                              market_data: MarketData) -> TradingSignal:
        """
        Generate trading signal based on sentiment and market data
        
        Args:
            symbol: Trading symbol
            sentiment_result: Sentiment analysis result
            market_data: Current market data
            
        Returns:
            TradingSignal object
        """
        action = 'HOLD'
        reasoning = "No clear signal"
        
        # Get current position
        base_asset = symbol.replace('USDT', '')
        current_position = self.session.positions.get(symbol, 0.0)
        
        # Decision logic based on sentiment and confidence
        if sentiment_result.confidence >= self.config.trading.confidence_threshold:
            if sentiment_result.sentiment == 'positive':
                if sentiment_result.confidence >= self.config.trading.sentiment_threshold_buy:
                    if current_position == 0:  # Only buy if no position
                        action = 'BUY'
                        reasoning = f"Strong positive sentiment ({sentiment_result.confidence:.2f})"
                    else:
                        reasoning = f"Positive sentiment but already holding position"
            
            elif sentiment_result.sentiment == 'negative':
                if sentiment_result.confidence >= abs(self.config.trading.sentiment_threshold_sell):
                    if current_position > 0:  # Only sell if holding position
                        action = 'SELL'
                        reasoning = f"Strong negative sentiment ({sentiment_result.confidence:.2f})"
                    else:
                        reasoning = f"Negative sentiment but no position to sell"
        
        return TradingSignal(
            symbol=symbol,
            action=action,
            sentiment=sentiment_result.sentiment,
            confidence=sentiment_result.confidence,
            news_count=len(sentiment_result.text.split()) if sentiment_result.text else 0,
            reasoning=reasoning,
            timestamp=datetime.now()
        )
    
    def execute_trade(self, signal: TradingSignal) -> Optional[OrderResult]:
        """
        Execute a trade based on the signal
        
        Args:
            signal: Trading signal
            
        Returns:
            OrderResult or None if no trade executed
        """
        if signal.action == 'HOLD':
            return None
        
        # Check 30-minute cooldown before executing
        if self._should_dismiss_signal(signal.symbol, signal.action):
            return None
        
        try:
            if signal.action == 'BUY':
                # Check account balance instead of relying on trade tracker for positions
                base_asset = signal.symbol.replace('USDT', '')
                current_balance = self.trader.get_asset_balance(base_asset)
                
                if current_balance > 0:
                    self.logger.info(f"Already have {current_balance:.6f} {base_asset}, will still proceed with buy order")
                    # Continue with buy - we're not preventing multiple buys anymore
                
                # Get USDT balance and determine buy amount
                usdt_balance = self.trader.get_asset_balance('USDT')
                
                if usdt_balance <= 0:
                    self.logger.warning(f"No USDT balance available for buy: {usdt_balance}")
                    return None
                
                # Use full balance if less than $30, otherwise use $30
                if usdt_balance <= self.config.trading.max_trade_amount:
                    buy_amount = usdt_balance
                    buy_reason = f"full USDT balance (${usdt_balance:.2f} ≤ ${self.config.trading.max_trade_amount})"
                else:
                    buy_amount = self.config.trading.max_trade_amount
                    buy_reason = f"${self.config.trading.max_trade_amount} (standard amount)"
                
                self.logger.info(f"  💰 Buy decision: USDT balance=${usdt_balance:.2f}, using ${buy_amount:.2f} ({buy_reason})")
                
                # Check if we can afford the trade
                can_trade, reason = self.trader.can_trade(signal.symbol, buy_amount)
                
                if not can_trade:
                    self.logger.warning(f"Cannot execute buy for {signal.symbol}: {reason}")
                    return None
                
                # Execute buy order
                result = self.trader.place_market_buy_order(
                    symbol=signal.symbol,
                    amount_usd=buy_amount
                )
                
                if result.success:
                    # Record signal for cooldown tracking
                    self._record_signal(signal.symbol, 'BUY')
                    
                    # Record trade in trade tracker with actual amount used
                    trade = self.trade_tracker.add_buy_trade(
                        symbol=signal.symbol,
                        quantity=result.quantity,
                        price=result.price,
                        amount_usd=buy_amount,
                        order_id=result.order_id
                    )
                    
                    # Update legacy session for backwards compatibility
                    self.session.positions[signal.symbol] = result.quantity
                    self.logger.info(f"  ✅ BUY: {result.quantity:.6f} {signal.symbol} at ${result.price:.4f} (${buy_amount:.2f})")
                
                return result
            
            elif signal.action == 'SELL':
                # Get account balance for the base asset (ignore trade tracker for quantity decisions)
                base_asset = signal.symbol.replace('USDT', '')
                account_balance = self.trader.get_asset_balance(base_asset)
                
                if account_balance <= 0:
                    self.logger.warning(f"No balance available to sell for {signal.symbol}: account balance={account_balance}")
                    return None
                
                # Get current market price
                market_data = self.trader.get_market_data(signal.symbol)
                if not market_data:
                    self.logger.warning(f"Failed to get market data for {signal.symbol}")
                    return None
                
                # Calculate dollar value of available balance
                available_dollar_value = account_balance * market_data.bid_price
                
                # Determine sell quantity based on dollar threshold
                if available_dollar_value <= self.config.trading.max_trade_amount:
                    # If dollar value is less than or equal to $30, sell all available
                    sell_quantity = account_balance
                    sell_reason = f"full balance (${available_dollar_value:.2f} ≤ ${self.config.trading.max_trade_amount})"
                else:
                    # If dollar value is more than $30, sell only $30 worth
                    sell_quantity = self.config.trading.max_trade_amount / market_data.bid_price
                    sell_reason = f"${self.config.trading.max_trade_amount} worth"
                
                self.logger.info(f"  📊 Sell decision: account balance={account_balance:.6f}, value=${available_dollar_value:.2f}")
                self.logger.info(f"      Using: {sell_quantity:.6f} ({sell_reason})")
                
                # Execute sell order
                result = self.trader.place_market_sell_order(
                    symbol=signal.symbol,
                    quantity=sell_quantity
                )
                
                if result.success:
                    # Record signal for cooldown tracking
                    self._record_signal(signal.symbol, 'SELL')
                    
                    # Calculate actual USD amount received
                    amount_usd = result.quantity * result.price
                    
                    # Record trade in trade tracker
                    trade = self.trade_tracker.add_sell_trade(
                        symbol=signal.symbol,
                        quantity=result.quantity,
                        price=result.price,
                        amount_usd=amount_usd,
                        order_id=result.order_id
                    )
                    
                    # Update legacy session for backwards compatibility
                    self.session.positions[signal.symbol] = 0.0
                    
                    # Log successful sell (P&L will be calculated by trade tracker)
                    self.logger.info(f"  ✅ SELL: {result.quantity:.6f} {signal.symbol} at ${result.price:.4f} (${amount_usd:.2f})")
                
                return result
            
        except Exception as e:
            self.logger.error(f"Error executing trade: {e}")
            return None
    
    def update_session_stats(self, order_result: OrderResult):
        """Update trading session statistics"""
        if order_result and order_result.success:
            self.session.total_trades += 1
            self.session.successful_trades += 1
            
            # Note: PnL calculation would require tracking entry/exit prices
            # This is simplified for the demo
    
    def analyze_technical_indicators(self, symbol: str) -> Optional[TechnicalSignal]:
        """
        Analyze technical indicators for a symbol
        
        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            
        Returns:
            TechnicalSignal or None
        """
        try:
            # Get historical price data (using 5-minute intervals for responsive analysis)
            price_data = self.trader.get_historical_klines(symbol, '1m', 100)
            if not price_data or len(price_data) < 50:
                self.logger.warning(f"  ⚠️  Insufficient price data for {symbol}")
                return None
            
            # Calculate technical indicators
            indicators = self.technical_analyzer.calculate_indicators(price_data)
            if not indicators:
                return None
            
            # Get AI analysis
            technical_signal = self.technical_analyzer.analyze_with_ai(indicators)
            
            # Start AI training evaluation in background (learn from this signal)
            current_market_data = self.trader.get_market_data(symbol)
            if current_market_data:
                self._evaluate_and_train_ai(symbol, technical_signal, indicators, current_market_data)
            
            # Log technical analysis results
            tech_emoji = "📈" if technical_signal.action == "BUY" else "📉" if technical_signal.action == "SELL" else "📊"
            self.logger.info(
                f"  {tech_emoji} Technical AI: {technical_signal.action} ({technical_signal.confidence:.0%}) - {technical_signal.reasoning}"
            )
            
            # Show key indicators
            key_indicators = technical_signal.key_indicators
            self.logger.info(
                f"    📊 RSI: {key_indicators['rsi_14']} | MACD: {key_indicators['macd_signal']} | "
                f"Trend: {key_indicators['trend']} | Vol: {key_indicators['volume_ratio']}x"
            )
            
            if technical_signal.pattern_detected:
                self.logger.info(f"    🔍 Pattern: {technical_signal.pattern_detected}")
            
            return technical_signal
            
        except Exception as e:
            self.logger.error(f"Error analyzing technical indicators for {symbol}: {e}")
            return None
    
    def combine_signals(self, sentiment_result: SentimentResult, 
                       technical_signal: TechnicalSignal, symbol: str) -> TradingSignal:
        """
        Combine sentiment and technical analysis for final trading decision
        
        Args:
            sentiment_result: News sentiment analysis
            technical_signal: Technical analysis signal
            symbol: Trading symbol
            
        Returns:
            Combined TradingSignal
        """
        # Weights for combining signals
        sentiment_weight = 0.6  # News sentiment gets 60%
        technical_weight = 0.4  # Technical analysis gets 40%
        
        # Convert actions to numeric scores for combination
        action_scores = {'SELL': -1, 'HOLD': 0, 'BUY': 1}
        
        # Get sentiment score
        sentiment_action = 'BUY' if sentiment_result.sentiment == 'positive' else 'SELL' if sentiment_result.sentiment == 'negative' else 'HOLD'
        sentiment_score = action_scores[sentiment_action] * sentiment_result.confidence
        
        # Get technical score  
        technical_score = action_scores[technical_signal.action] * technical_signal.confidence
        
        # Combine scores
        combined_score = (sentiment_score * sentiment_weight) + (technical_score * technical_weight)
        combined_confidence = (sentiment_result.confidence * sentiment_weight) + (technical_signal.confidence * technical_weight)
        
        # Determine final action based on combined score (lowered thresholds for more trades)
        if combined_score > 0.25:  # Lower threshold for BUY signals
            final_action = 'BUY'
            reasoning = f"Combined AI signals: Sentiment {sentiment_result.sentiment.upper()} + Technical {technical_signal.action}"
        elif combined_score < -0.25:  # Lower threshold for SELL signals
            final_action = 'SELL'
            reasoning = f"Combined AI signals: Sentiment {sentiment_result.sentiment.upper()} + Technical {technical_signal.action}"
        else:
            final_action = 'HOLD'
            if abs(sentiment_score) > abs(technical_score):
                reasoning = f"Conflicting signals, sentiment dominates: {sentiment_result.sentiment.upper()}"
            else:
                reasoning = f"Conflicting signals, technical dominates: {technical_signal.action}"
        
        # Check position constraints using actual account balance
        base_asset = symbol.replace('USDT', '')
        account_balance = self.trader.get_asset_balance(base_asset)
        
        # Allow BUY regardless of current position (can accumulate)
        # Only check for SELL if we actually have balance to sell
        if final_action == 'SELL' and account_balance <= 0:
            final_action = 'HOLD'
            reasoning += " (no balance to sell)"
        
        return TradingSignal(
            symbol=symbol,
            action=final_action,
            sentiment=sentiment_result.sentiment,
            confidence=combined_confidence,
            news_count=len(sentiment_result.text.split()) if sentiment_result.text else 0,
            reasoning=reasoning,
            timestamp=datetime.now()
        )
    
    def process_symbol(self, symbol: str):
        """
        Process a single trading symbol with combined AI analysis
        
        Args:
            symbol: Trading symbol to process
        """
        try:
            self.logger.info(f"📊 Analyzing {symbol}...")
            
            # Get current market data
            market_data = self.trader.get_market_data(symbol)
            if not market_data:
                self.logger.warning(f"Failed to get market data for {symbol}")
                return
            
            # Fetch and analyze news sentiment
            sentiment_result = self.fetch_and_analyze_news(symbol)
            
            # Analyze technical indicators
            technical_signal = self.analyze_technical_indicators(symbol)
            
            # Skip if we don't have both analyses
            if not sentiment_result and not technical_signal:
                self.logger.warning(f"  ⚠️  Skipping {symbol} - no analysis data")
                return
            
            # Use fallback if one analysis is missing
            if sentiment_result and technical_signal:
                # Combine both signals
                combined_signal = self.combine_signals(sentiment_result, technical_signal, symbol)
                signal_type = "🤖 Combined AI"
            elif sentiment_result:
                # Use sentiment only
                combined_signal = self.generate_trading_signal(symbol, sentiment_result, market_data)
                signal_type = "😊 Sentiment Only"
            else:
                # Use technical only (create dummy sentiment)
                dummy_sentiment = SentimentResult(
                    sentiment='neutral',
                    confidence=0.5,
                    raw_scores={'positive': 0.33, 'negative': 0.33, 'neutral': 0.34},
                    text="Technical analysis only"
                )
                combined_signal = TradingSignal(
                    symbol=symbol,
                    action=technical_signal.action,
                    sentiment='neutral',
                    confidence=technical_signal.confidence,
                    news_count=0,
                    reasoning=f"Technical only: {technical_signal.reasoning}",
                    timestamp=datetime.now()
                )
                signal_type = "📈 Technical Only"
            
            # Log final decision
            action_emoji = "🟢" if combined_signal.action == "BUY" else "🔴" if combined_signal.action == "SELL" else "⚪"
            self.logger.info(
                f"  {action_emoji} {signal_type}: {combined_signal.action} ({combined_signal.confidence:.0%})"
            )
            self.logger.info(f"    💭 Reasoning: {combined_signal.reasoning}")
            
            # Send Telegram notification for BUY/SELL signals
            if combined_signal.action != 'HOLD' and self.telegram_notifier and self.telegram_notifier.enabled:
                self.telegram_notifier.send_trading_signal(
                    symbol=combined_signal.symbol,
                    action=combined_signal.action,
                    confidence=combined_signal.confidence,
                    reasoning=combined_signal.reasoning
                )
            
            # Execute trade if signal indicates action
            if combined_signal.action != 'HOLD':
                order_result = self.execute_trade(combined_signal)
                if order_result:
                    self.update_session_stats(order_result)
            
        except Exception as e:
            self.logger.error(f"Error processing {symbol}: {e}")
    
    def print_session_summary(self):
        """Print trading session summary"""
        runtime = datetime.now() - self.session.start_time
        
        self.logger.info("=== Trading Session Summary ===")
        self.logger.info(f"Runtime: {runtime}")
        self.logger.info(f"Total trades: {self.session.total_trades}")
        self.logger.info(f"Successful trades: {self.session.successful_trades}")
        
        # Show current positions
        if any(qty > 0 for qty in self.session.positions.values()):
            self.logger.info("Current positions:")
            for symbol, quantity in self.session.positions.items():
                if quantity > 0:
                    self.logger.info(f"  {symbol}: {quantity}")
        else:
            self.logger.info("No open positions")
    
    def run(self):
        """Main trading loop"""
        if not self.initialize_components():
            self.logger.error("Failed to initialize components")
            return
        
        self.running = True
        self.logger.info("🚀 Starting trading cycles...")
        
        try:
            while self.running:
                cycle_start = time.time()
                
                # Process each configured symbol
                for symbol in self.config.trading.symbols:
                    if not self.running:
                        break
                    
                    self.process_symbol(symbol)
                    
                    # Small delay between symbols to avoid rate limits
                    time.sleep(2)
                
                # Print periodic summary
                if self.session.total_trades > 0:
                    self.print_session_summary()
                
                # Calculate sleep time to maintain interval
                cycle_time = time.time() - cycle_start
                sleep_time = max(0, self.config.trading.trade_interval - cycle_time)
                
                if sleep_time > 0 and self.running:
                    self.logger.info(f"⏱️  Cycle done ({cycle_time:.1f}s) - Next check in {sleep_time//60:.0f}m {sleep_time%60:.0f}s")
                    # Sleep in small intervals to allow for graceful shutdown
                    sleep_interval = 1.0  # Check for shutdown every second
                    while sleep_time > 0 and self.running:
                        actual_sleep = min(sleep_interval, sleep_time)
                        time.sleep(actual_sleep)
                        sleep_time -= actual_sleep
                
        except Exception as e:
            self.logger.error(f"Error in main loop: {e}")
        finally:
            self.logger.info("🛑 Trading bot stopped")
            self.print_session_summary()
            
            # Send shutdown notification via Telegram
            if self.telegram_notifier and self.telegram_notifier.enabled:
                runtime = datetime.now() - self.session.start_time
                runtime_str = str(runtime).split('.')[0]  # Remove microseconds
                self.telegram_notifier.send_shutdown_message(
                    runtime=runtime_str,
                    total_trades=self.session.total_trades
                )

def main():
    """Main entry point"""
    print("🤖 Cryptocurrency Sentiment Trading Bot")
    print("=" * 50)
    
    # Create and run bot
    bot = CryptoSentimentBot()
    
    try:
        bot.run()
    except KeyboardInterrupt:
        print("\nShutdown requested by user")
    except Exception as e:
        print(f"Unexpected error: {e}")
        logging.error(f"Unexpected error: {e}")
    
    print("Bot stopped. Goodbye! 👋")

if __name__ == "__main__":
    main()