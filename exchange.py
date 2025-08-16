import logging
import hmac
import hashlib
import time
import requests
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from decimal import Decimal, ROUND_DOWN
import json

@dataclass
class OrderResult:
    """Result of a trading order"""
    success: bool
    order_id: Optional[str] = None
    symbol: str = ""
    side: str = ""  # 'BUY' or 'SELL'
    quantity: float = 0.0
    price: float = 0.0
    status: str = ""
    error_message: Optional[str] = None
    timestamp: Optional[int] = None

@dataclass
class Balance:
    """Account balance information"""
    asset: str
    free: float
    locked: float
    total: float

@dataclass
class MarketData:
    """Market data for a symbol"""
    symbol: str
    price: float
    bid_price: float
    ask_price: float
    volume: float
    change_24h: float
    timestamp: int

class Exchange:
    """Exchange trading interface with testnet support (currently Binance)"""
    
    def __init__(self, api_key: str, secret_key: str, testnet: bool = True):
        """
        Initialize exchange
        
        Args:
            api_key: Binance API key
            secret_key: Binance secret key  
            testnet: Whether to use testnet (default: True for safety)
        """
        self.logger = logging.getLogger(__name__)
        self.api_key = api_key
        self.secret_key = secret_key
        self.testnet = testnet
        
        # Set base URLs
        if testnet:
            self.base_url = "https://testnet.binance.vision"
            self.logger.info("🧪 Connected to Binance TESTNET (paper trading)")
        else:
            self.base_url = "https://api.binance.com"
            self.logger.warning("Using Binance LIVE trading - real money at risk!")
        
        # Session for HTTP requests
        self.session = requests.Session()
        self.session.headers.update({
            'X-MBX-APIKEY': self.api_key,
            'Content-Type': 'application/json'
        })
        
        # Trading parameters
        self.min_notional = {}  # Minimum order values per symbol
        self.symbol_info = {}   # Symbol trading rules
        
        # Initialize trading info
        self._load_exchange_info()
    
    def _get_timestamp(self) -> int:
        """Get current timestamp in milliseconds"""
        return int(time.time() * 1000)
    
    def _create_signature(self, query_string: str) -> str:
        """Create HMAC SHA256 signature for API request"""
        return hmac.new(
            self.secret_key.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
    
    def _make_request(self, method: str, endpoint: str, params: Dict = None, signed: bool = False) -> Optional[Dict]:
        """
        Make request to Binance API
        
        Args:
            method: HTTP method ('GET', 'POST', etc.)
            endpoint: API endpoint
            params: Request parameters
            signed: Whether request needs signature
            
        Returns:
            JSON response or None if failed
        """
        if params is None:
            params = {}
        
        url = f"{self.base_url}{endpoint}"
        
        try:
            if signed:
                # Add timestamp for signed requests
                params['timestamp'] = self._get_timestamp()
                
                # Create query string and signature
                query_string = '&'.join([f"{k}={v}" for k, v in params.items()])
                signature = self._create_signature(query_string)
                params['signature'] = signature
            
            # Make request
            if method == 'GET':
                response = self.session.get(url, params=params, timeout=30)
            elif method == 'POST':
                response = self.session.post(url, params=params, timeout=30)
            elif method == 'DELETE':
                response = self.session.delete(url, params=params, timeout=30)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Request failed: {e}")
            if hasattr(e, 'response') and e.response is not None:
                try:
                    error_data = e.response.json()
                    self.logger.error(f"API error: {error_data}")
                except:
                    self.logger.error(f"Response text: {e.response.text}")
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error: {e}")
            return None
    
    def _load_exchange_info(self):
        """Load exchange trading rules and symbol information"""
        try:
            response = self._make_request('GET', '/api/v3/exchangeInfo')
            if not response:
                self.logger.error("Failed to load exchange info")
                return
            
            for symbol_data in response.get('symbols', []):
                symbol = symbol_data['symbol']
                
                # Store symbol info
                self.symbol_info[symbol] = {
                    'status': symbol_data['status'],
                    'baseAsset': symbol_data['baseAsset'],
                    'quoteAsset': symbol_data['quoteAsset'],
                    'filters': symbol_data['filters']
                }
                
                # Extract minimum notional value
                for filter_data in symbol_data['filters']:
                    if filter_data['filterType'] == 'MIN_NOTIONAL':
                        self.min_notional[symbol] = float(filter_data['minNotional'])
                        break
                else:
                    self.min_notional[symbol] = 10.0  # Default minimum
            
            self.logger.info(f"📊 Ready to trade {len(self.symbol_info)} symbols")
            
        except Exception as e:
            self.logger.error(f"Error loading exchange info: {e}")
    
    def get_account_info(self) -> Optional[Dict]:
        """Get account information and balances"""
        return self._make_request('GET', '/api/v3/account', signed=True)
    
    def get_balances(self) -> List[Balance]:
        """
        Get account balances
        
        Returns:
            List of Balance objects
        """
        account_info = self.get_account_info()
        if not account_info:
            return []
        
        balances = []
        for balance_data in account_info.get('balances', []):
            asset = balance_data['asset']
            free = float(balance_data['free'])
            locked = float(balance_data['locked'])
            total = free + locked
            
            # Only include balances with non-zero amounts
            if total > 0:
                balances.append(Balance(
                    asset=asset,
                    free=free,
                    locked=locked,
                    total=total
                ))
        
        return balances
    
    def get_market_data(self, symbol: str) -> Optional[MarketData]:
        """
        Get current market data for a symbol
        
        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            
        Returns:
            MarketData object or None
        """
        # Get current price
        ticker_response = self._make_request('GET', '/api/v3/ticker/24hr', {'symbol': symbol})
        if not ticker_response:
            return None
        
        # Get order book for bid/ask prices
        book_response = self._make_request('GET', '/api/v3/ticker/bookTicker', {'symbol': symbol})
        if not book_response:
            return None
        
        try:
            return MarketData(
                symbol=symbol,
                price=float(ticker_response['lastPrice']),
                bid_price=float(book_response['bidPrice']),
                ask_price=float(book_response['askPrice']),
                volume=float(ticker_response['volume']),
                change_24h=float(ticker_response['priceChangePercent']),
                timestamp=self._get_timestamp()
            )
        except (KeyError, ValueError) as e:
            self.logger.error(f"Error parsing market data: {e}")
            return None
    
    def _calculate_quantity(self, symbol: str, amount_usd: float, price: float) -> Tuple[float, bool]:
        """
        Calculate the quantity to trade based on USD amount
        
        Args:
            symbol: Trading symbol
            amount_usd: Amount in USD to trade
            price: Current price
            
        Returns:
            Tuple of (quantity, is_valid)
        """
        if symbol not in self.symbol_info:
            self.logger.error(f"No symbol info for {symbol}")
            return 0.0, False
        
        # Calculate base quantity
        quantity = amount_usd / price
        
        # Find lot size filter
        filters = self.symbol_info[symbol]['filters']
        step_size = None
        min_qty = None
        
        for filter_data in filters:
            if filter_data['filterType'] == 'LOT_SIZE':
                step_size = float(filter_data['stepSize'])
                min_qty = float(filter_data['minQty'])
                break
        
        if step_size is None:
            self.logger.error(f"No lot size filter found for {symbol}")
            return 0.0, False
        
        # Round down to valid step size
        if step_size > 0:
            decimal_places = len(str(step_size).split('.')[-1].rstrip('0'))
            quantity = float(Decimal(str(quantity)).quantize(
                Decimal(str(step_size)), 
                rounding=ROUND_DOWN
            ))
        
        # Check minimum quantity
        if min_qty and quantity < min_qty:
            self.logger.warning(f"Quantity {quantity} below minimum {min_qty} for {symbol}")
            return 0.0, False
        
        # Check minimum notional value
        notional_value = quantity * price
        min_notional = self.min_notional.get(symbol, 10.0)
        
        if notional_value < min_notional:
            self.logger.warning(f"Notional value {notional_value} below minimum {min_notional} for {symbol}")
            return 0.0, False
        
        return quantity, True
    
    def place_market_buy_order(self, symbol: str, amount_usd: float) -> OrderResult:
        """
        Place a market buy order
        
        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            amount_usd: Amount in USD to buy
            
        Returns:
            OrderResult object
        """
        try:
            # Get current market price
            market_data = self.get_market_data(symbol)
            if not market_data:
                return OrderResult(
                    success=False,
                    error_message="Failed to get market data"
                )
            
            # Calculate quantity
            quantity, is_valid = self._calculate_quantity(symbol, amount_usd, market_data.ask_price)
            if not is_valid:
                return OrderResult(
                    success=False,
                    error_message=f"Invalid quantity calculation for {amount_usd} USD"
                )
            
            # Place order
            params = {
                'symbol': symbol,
                'side': 'BUY',
                'type': 'MARKET',
                'quantity': quantity
            }
            
            response = self._make_request('POST', '/api/v3/order', params, signed=True)
            
            if response:
                return OrderResult(
                    success=True,
                    order_id=str(response['orderId']),
                    symbol=symbol,
                    side='BUY',
                    quantity=float(response['executedQty']),
                    price=float(response.get('price', 0)) or market_data.ask_price,
                    status=response['status'],
                    timestamp=response['transactTime']
                )
            else:
                return OrderResult(
                    success=False,
                    error_message="Order placement failed"
                )
                
        except Exception as e:
            self.logger.error(f"Error placing buy order: {e}")
            return OrderResult(
                success=False,
                error_message=str(e)
            )
    
    def place_market_sell_order(self, symbol: str, quantity: float) -> OrderResult:
        """
        Place a market sell order
        
        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            quantity: Quantity to sell
            
        Returns:
            OrderResult object
        """
        try:
            # Get current market price for reference
            market_data = self.get_market_data(symbol)
            if not market_data:
                return OrderResult(
                    success=False,
                    error_message="Failed to get market data"
                )
            
            # Place order
            params = {
                'symbol': symbol,
                'side': 'SELL',
                'type': 'MARKET',
                'quantity': quantity
            }
            
            response = self._make_request('POST', '/api/v3/order', params, signed=True)
            
            if response:
                return OrderResult(
                    success=True,
                    order_id=str(response['orderId']),
                    symbol=symbol,
                    side='SELL',
                    quantity=float(response['executedQty']),
                    price=float(response.get('price', 0)) or market_data.bid_price,
                    status=response['status'],
                    timestamp=response['transactTime']
                )
            else:
                return OrderResult(
                    success=False,
                    error_message="Order placement failed"
                )
                
        except Exception as e:
            self.logger.error(f"Error placing sell order: {e}")
            return OrderResult(
                success=False,
                error_message=str(e)
            )
    
    def get_asset_balance(self, asset: str) -> float:
        """
        Get balance for a specific asset
        
        Args:
            asset: Asset symbol (e.g., 'BTC', 'USDT')
            
        Returns:
            Available balance
        """
        balances = self.get_balances()
        for balance in balances:
            if balance.asset == asset:
                return balance.free
        return 0.0
    
    def can_trade(self, symbol: str, amount_usd: float) -> Tuple[bool, str]:
        """
        Check if we can execute a trade
        
        Args:
            symbol: Trading symbol
            amount_usd: Amount in USD
            
        Returns:
            Tuple of (can_trade, reason)
        """
        # Check if symbol is supported
        if symbol not in self.symbol_info:
            return False, f"Symbol {symbol} not supported"
        
        # Check if symbol is trading
        if self.symbol_info[symbol]['status'] != 'TRADING':
            return False, f"Symbol {symbol} not currently trading"
        
        # Check minimum notional
        min_notional = self.min_notional.get(symbol, 10.0)
        if amount_usd < min_notional:
            return False, f"Amount {amount_usd} below minimum {min_notional}"
        
        # Check USDT balance for buy orders
        usdt_balance = self.get_asset_balance('USDT')
        if usdt_balance < amount_usd:
            return False, f"Insufficient USDT balance: {usdt_balance} < {amount_usd}"
        
        return True, "OK"
    
    def get_trading_fees(self) -> Dict[str, float]:
        """Get current trading fees"""
        try:
            response = self._make_request('GET', '/api/v3/account', signed=True)
            if response:
                return {
                    'maker_fee': float(response.get('makerCommission', 10)) / 10000,  # Convert from basis points
                    'taker_fee': float(response.get('takerCommission', 10)) / 10000
                }
        except Exception as e:
            self.logger.error(f"Error getting trading fees: {e}")
        
        # Default fees
        return {'maker_fee': 0.001, 'taker_fee': 0.001}
    
    def test_connectivity(self) -> bool:
        """Test API connectivity"""
        response = self._make_request('GET', '/api/v3/ping')
        return response is not None
    
    def get_server_time(self) -> Optional[int]:
        """Get server time"""
        response = self._make_request('GET', '/api/v3/time')
        return response.get('serverTime') if response else None
    
    def get_historical_klines(self, symbol: str, interval: str = '1h', limit: int = 100) -> List[Dict]:
        """
        Get historical kline/candlestick data for technical analysis
        
        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            interval: Kline interval (1m, 5m, 15m, 30m, 1h, 4h, 1d, etc.)
            limit: Number of klines to fetch (max 1000)
            
        Returns:
            List of kline data dictionaries
        """
        try:
            params = {
                'symbol': symbol,
                'interval': interval,
                'limit': min(limit, 1000)  # API limit
            }
            
            response = self._make_request('GET', '/api/v3/klines', params)
            
            if not response:
                self.logger.error(f"Failed to fetch klines for {symbol}")
                return []
            
            # Convert response to standardized format
            klines = []
            for kline in response:
                klines.append({
                    'timestamp': int(kline[0]),
                    'open': float(kline[1]),
                    'high': float(kline[2]),
                    'low': float(kline[3]),
                    'close': float(kline[4]),
                    'volume': float(kline[5]),
                    'close_time': int(kline[6]),
                    'quote_volume': float(kline[7]),
                    'trade_count': int(kline[8])
                })
            
            self.logger.debug(f"Fetched {len(klines)} klines for {symbol}")
            return klines
            
        except Exception as e:
            self.logger.error(f"Error fetching historical data for {symbol}: {e}")
            return []
    
    def get_all_orders(self, symbol: str = None, limit: int = 500) -> List[Dict]:
        """
        Get all orders for account (open and historical)
        
        Args:
            symbol: Trading symbol (optional, if None gets orders for all symbols)
            limit: Maximum number of orders to return (max 1000)
            
        Returns:
            List of order data dictionaries
        """
        try:
            # If no symbol specified, get orders for all symbols we trade
            if symbol is None:
                # Get orders for common trading symbols
                all_orders = []
                common_symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'DOGEUSDT']
                
                for sym in common_symbols:
                    try:
                        symbol_orders = self._get_orders_for_symbol(sym, limit=100)
                        all_orders.extend(symbol_orders)
                    except Exception as e:
                        self.logger.debug(f"No orders found for {sym}: {e}")
                        continue
                
                # Sort by timestamp (newest first)
                all_orders.sort(key=lambda x: x.get('time', 0), reverse=True)
                return all_orders[:limit]
            else:
                return self._get_orders_for_symbol(symbol, limit)
                
        except Exception as e:
            self.logger.error(f"Error fetching orders: {e}")
            return []
    
    def _get_orders_for_symbol(self, symbol: str, limit: int = 500) -> List[Dict]:
        """
        Get all orders for a specific symbol
        
        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            limit: Maximum number of orders to return
            
        Returns:
            List of order data dictionaries
        """
        try:
            params = {
                'symbol': symbol,
                'limit': min(limit, 1000)  # API limit
            }
            
            response = self._make_request('GET', '/api/v3/allOrders', params, signed=True)
            
            if not response:
                self.logger.debug(f"No orders found for {symbol}")
                return []
            
            # Filter and format the orders
            formatted_orders = []
            for order in response:
                formatted_orders.append({
                    'orderId': order.get('orderId'),
                    'symbol': order.get('symbol'),
                    'side': order.get('side'),
                    'type': order.get('type'),
                    'status': order.get('status'),
                    'origQty': float(order.get('origQty', 0)),
                    'executedQty': float(order.get('executedQty', 0)),
                    'price': float(order.get('price', 0)),
                    'time': order.get('time'),
                    'updateTime': order.get('updateTime')
                })
            
            self.logger.debug(f"Fetched {len(formatted_orders)} orders for {symbol}")
            return formatted_orders
            
        except Exception as e:
            self.logger.error(f"Error fetching orders for {symbol}: {e}")
            return []