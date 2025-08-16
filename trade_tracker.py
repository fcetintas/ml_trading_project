#!/usr/bin/env python3
"""
Trade Tracking System for Cryptocurrency Trading Bot

Manages trade history, position tracking, and provides trading analytics.
"""

import logging
import json
import os
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum

class TradeStatus(Enum):
    """Trade status enumeration"""
    OPEN = "open"
    CLOSED = "closed"
    FAILED = "failed"

@dataclass
class Trade:
    """Represents a single trade"""
    id: str
    symbol: str
    side: str  # 'BUY' or 'SELL'
    quantity: float
    price: float
    amount_usd: float
    timestamp: datetime
    order_id: Optional[str] = None
    status: TradeStatus = TradeStatus.OPEN
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert trade to dictionary for JSON serialization"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        data['status'] = self.status.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Trade':
        """Create trade from dictionary"""
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        data['status'] = TradeStatus(data['status'])
        return cls(**data)

@dataclass
class Position:
    """Represents an open trading position"""
    symbol: str
    quantity: float
    entry_price: float
    entry_time: datetime
    entry_trade_id: str
    current_value: float = 0.0
    pnl: float = 0.0
    pnl_pct: float = 0.0
    
    def update_current_value(self, current_price: float):
        """Update position with current market price"""
        self.current_value = self.quantity * current_price
        entry_value = self.quantity * self.entry_price
        self.pnl = self.current_value - entry_value
        self.pnl_pct = (self.pnl / entry_value) * 100 if entry_value > 0 else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert position to dictionary"""
        data = asdict(self)
        data['entry_time'] = self.entry_time.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Position':
        """Create position from dictionary"""
        data['entry_time'] = datetime.fromisoformat(data['entry_time'])
        return cls(**data)

class TradeTracker:
    """Comprehensive trade tracking and management system"""
    
    def __init__(self, data_file: str = "trades.json"):
        """
        Initialize trade tracker
        
        Args:
            data_file: File to store trade data
        """
        self.logger = logging.getLogger(__name__)
        self.data_file = data_file
        self.trades: List[Trade] = []
        self.positions: Dict[str, Position] = {}  # symbol -> Position
        self.load_data()
    
    def load_data(self):
        """Load trade history and positions from file"""
        try:
            if os.path.exists(self.data_file):
                with open(self.data_file, 'r') as f:
                    data = json.load(f)
                
                # Load trades
                self.trades = [Trade.from_dict(trade_data) for trade_data in data.get('trades', [])]
                
                # Load positions
                positions_data = data.get('positions', {})
                self.positions = {
                    symbol: Position.from_dict(pos_data) 
                    for symbol, pos_data in positions_data.items()
                }
                
                self.logger.info(f"📊 Loaded {len(self.trades)} trades and {len(self.positions)} positions")
            
        except Exception as e:
            self.logger.error(f"Error loading trade data: {e}")
            self.trades = []
            self.positions = {}
    
    def save_data(self):
        """Save trade history and positions to file"""
        try:
            data = {
                'trades': [trade.to_dict() for trade in self.trades],
                'positions': {symbol: pos.to_dict() for symbol, pos in self.positions.items()},
                'last_updated': datetime.now().isoformat()
            }
            
            with open(self.data_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Error saving trade data: {e}")
    
    def has_open_position(self, symbol: str) -> bool:
        """Check if there's an open position for a symbol"""
        return symbol in self.positions
    
    def add_buy_trade(self, symbol: str, quantity: float, price: float, 
                     amount_usd: float, order_id: str = None) -> Trade:
        """
        Add a buy trade and open a position
        
        Args:
            symbol: Trading symbol
            quantity: Quantity bought
            price: Purchase price
            amount_usd: USD amount spent
            order_id: Binance order ID
            
        Returns:
            Trade object
        """
        # Create trade ID
        trade_id = f"{symbol}_{int(datetime.now().timestamp())}"
        
        # Create trade record
        trade = Trade(
            id=trade_id,
            symbol=symbol,
            side='BUY',
            quantity=quantity,
            price=price,
            amount_usd=amount_usd,
            timestamp=datetime.now(),
            order_id=order_id,
            status=TradeStatus.OPEN
        )
        
        self.trades.append(trade)
        
        # Create position
        position = Position(
            symbol=symbol,
            quantity=quantity,
            entry_price=price,
            entry_time=datetime.now(),
            entry_trade_id=trade_id
        )
        
        self.positions[symbol] = position
        
        # Save data
        self.save_data()
        
        self.logger.info(f"📈 Opened position: {symbol} - {quantity:.6f} at ${price:.4f}")
        return trade
    
    def add_sell_trade(self, symbol: str, quantity: float, price: float, 
                      amount_usd: float, order_id: str = None) -> Optional[Trade]:
        """
        Add a sell trade and close a position
        
        Args:
            symbol: Trading symbol
            quantity: Quantity sold
            price: Sale price
            amount_usd: USD amount received
            order_id: Binance order ID
            
        Returns:
            Trade object or None if no position to close
        """
        if symbol not in self.positions:
            self.logger.warning(f"No open position to sell for {symbol}")
            return None
        
        position = self.positions[symbol]
        
        # Create trade ID
        trade_id = f"{symbol}_{int(datetime.now().timestamp())}"
        
        # Create sell trade record
        trade = Trade(
            id=trade_id,
            symbol=symbol,
            side='SELL',
            quantity=quantity,
            price=price,
            amount_usd=amount_usd,
            timestamp=datetime.now(),
            order_id=order_id,
            status=TradeStatus.CLOSED
        )
        
        self.trades.append(trade)
        
        # Calculate P&L
        entry_value = position.quantity * position.entry_price
        exit_value = quantity * price
        pnl = exit_value - entry_value
        pnl_pct = (pnl / entry_value) * 100 if entry_value > 0 else 0.0
        
        # Mark buy trade as closed
        for t in self.trades:
            if t.id == position.entry_trade_id:
                t.status = TradeStatus.CLOSED
                break
        
        # Remove position
        del self.positions[symbol]
        
        # Save data
        self.save_data()
        
        self.logger.info(f"📉 Closed position: {symbol} - {quantity:.6f} at ${price:.4f} | P&L: ${pnl:.2f} ({pnl_pct:+.2f}%)")
        return trade
    
    def update_positions_value(self, price_fetcher_func):
        """
        Update all positions with current market prices
        
        Args:
            price_fetcher_func: Function that takes symbol and returns current price
        """
        for symbol, position in self.positions.items():
            try:
                current_price = price_fetcher_func(symbol)
                if current_price:
                    position.update_current_value(current_price)
            except Exception as e:
                self.logger.warning(f"Failed to update price for {symbol}: {e}")
    
    def get_trade_history(self, limit: int = 20) -> List[Trade]:
        """Get recent trade history"""
        return sorted(self.trades, key=lambda t: t.timestamp, reverse=True)[:limit]
    
    def get_open_positions(self) -> Dict[str, Position]:
        """Get all open positions"""
        return self.positions.copy()
    
    def get_trading_stats(self) -> Dict[str, Any]:
        """Get comprehensive trading statistics"""
        closed_trades = [t for t in self.trades if t.status == TradeStatus.CLOSED]
        buy_trades = [t for t in self.trades if t.side == 'BUY']
        sell_trades = [t for t in self.trades if t.side == 'SELL']
        
        # Calculate P&L from closed trades
        total_pnl = 0.0
        profitable_trades = 0
        
        # Group buy/sell pairs to calculate P&L
        buy_dict = {t.symbol: t for t in buy_trades if t.status == TradeStatus.CLOSED}
        
        for sell_trade in sell_trades:
            if sell_trade.symbol in buy_dict:
                buy_trade = buy_dict[sell_trade.symbol]
                pnl = sell_trade.amount_usd - buy_trade.amount_usd
                total_pnl += pnl
                if pnl > 0:
                    profitable_trades += 1
        
        return {
            'total_trades': len(self.trades),
            'buy_trades': len(buy_trades),
            'sell_trades': len(sell_trades),
            'open_positions': len(self.positions),
            'closed_trades': len(closed_trades) // 2,  # Pairs of buy/sell
            'total_pnl': total_pnl,
            'profitable_trades': profitable_trades,
            'success_rate': (profitable_trades / max(1, len(closed_trades) // 2)) * 100,
            'symbols_traded': len(set(t.symbol for t in self.trades))
        }
    
    def get_symbol_history(self, symbol: str) -> List[Trade]:
        """Get trade history for a specific symbol"""
        return [t for t in self.trades if t.symbol == symbol]