#!/usr/bin/env python3
"""
Messaging Interface for Cryptocurrency Trading Bot

Simple, clean interface for sending/receiving messages and formatting data.
No trading logic - only communication and formatting.
"""

import logging
import os
import asyncio
import threading
from typing import Dict, Callable, Optional
from datetime import datetime
from venv import logger

try:
    from telegram import Update
    from telegram.ext import Application, ApplicationBuilder, CommandHandler, ContextTypes
    from telegram.request import HTTPXRequest
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False
    logging.warning("python-telegram-bot not available, install with: pip install python-telegram-bot")


class MessagingInterface:
    """
    Thread-safe messaging interface for Telegram bot communication
    
    Features:
    - Automatic startup in dedicated thread
    - Thread-safe message sending using asyncio.run_coroutine_threadsafe()
    - Built-in command handlers with custom callback registration
    - Proper async/await handling for Telegram API
    - Comprehensive message formatting for trading data
    
    Usage:
        # Basic setup - starts automatically in background thread
        messaging = MessagingInterface(bot_token, chat_id)
        
        # Register command handlers (can be done anytime after creation)
        def get_balance():
            return {
                'balances': [{'asset': 'BTC', 'amount': 1.5, 'usd_value': 45000}],
                'total_value_usdt': 45000
            }
        
        messaging.register_command("balance", get_balance)
        
        # Send messages from any thread
        messaging.send_trading_signal("BTCUSDT", "BUY", 0.85, "Strong bullish momentum")
        messaging.send_startup_message(["BTCUSDT", "ETHUSDT"], "LIVE")
        
        # Stop when done (optional - automatically handled on exit)
        messaging.stop()
    
    Built-in Commands:
    - /help - Shows available commands
    - /balance - Shows wallet balances (if registered)
    - /positions - Shows open positions (if registered)
    - /status - Shows bot status (if registered)
    - /trades - Shows recent trades (if registered)
    - /history - Shows trading history (if registered)
    - /orders - Shows all exchange orders (if registered)
    """
    
    def __init__(self, bot_token: str, chat_id: str):
        """
        Initialize messaging interface
        
        Args:
            bot_token: Telegram bot token from @BotFather
            chat_id: Telegram chat ID for authorized user
        """

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.authorized_chat_id = str(chat_id)
        
        # Simple state
        self.enabled = False
        self.running = False
        self._application = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        
        # Command callbacks
        self.command_callbacks: Dict[str, Callable] = {}
        
        # Check availability
        if not TELEGRAM_AVAILABLE or not bot_token or not chat_id:
            self.logger.warning("Messaging disabled - missing requirements")
            return
        
        self.enabled = True
        self._thread =threading.Thread(
            target=self.start,
            name="MessagingInterface"
        )

        self._thread.start()

        self.logger.info("📱 Messaging interface ready")

    def start(self):
        """
        Start messaging interface (handles both sync and async contexts)

        Returns:
            True if started successfully
        """
        try:
            # Store the event loop reference
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            # No running loop - create a new one
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)

        self._application = ApplicationBuilder().token(self.bot_token).build()

        # Register command handlers
        for cmd in ["help", "balance", "positions", "status", "trades", "history", "orders"]:
            self._application.add_handler(CommandHandler(cmd, self._handle_command))

        self._loop.run_until_complete(self._start_async())

    async def _start_async(self):
        """Internal async startup method"""

        # Start application
        await self._application.initialize()
        await self._application.start()

        # Start polling task (using the correct updater API)
        if self._application.updater:
            await self._application.updater.start_polling(
                poll_interval=1.0,
                timeout=10,
                drop_pending_updates=True
            )

        self.running = True
        self.logger.info("🤖 Messaging interface started (async)")
        while True:
            # Keep the async loop running
            await asyncio.sleep(10)
            if not self._application.updater.running:
                break

    async def _stop_async(self):
        """Stop messaging interface (async)"""
        if self.running:
            if self._application.updater:
                # Stop polling
                await self._application.updater.stop()

            # Stop application
            if self._application:
                await self._application.stop()
                await self._application.shutdown()
            
            self.running = False
            self.logger.info("🛑 Messaging interface stopped (async)")
    
    def stop(self):
        """Stop messaging interface"""
        self.logger.info("🛑 Stopping messaging interface...")
        self.logger.info(f"------------------------------------{self.running}")
        if not self.running:
            return
        asyncio.run_coroutine_threadsafe(self._stop_async(), self._loop)
        self._thread.join()  # Wait for the thread to finish


    
    def register_command(self, command: str, callback: Callable):
        """
        Register callback for command
        
        Args:
            command: Command name (without /)
            callback: Function to call when command received (should return dict with data)
            
        Example:
            def get_balance():
                return {
                    'balances': [{'asset': 'BTC', 'amount': 1.5, 'usd_value': 45000}],
                    'total_value_usdt': 45000
                }
            messaging.register_command("balance", get_balance)
        """
        self.command_callbacks[command] = callback

    async def _send_message(self, text: str) -> bool:
        """
        Send message to chat (async version)

        Args:
            text: Message text (supports HTML formatting)

        Returns:
            True if sent successfully
        """
        if not self.enabled or not self._application or not self._loop:
            return False

        # Create the coroutine
        await self._application.bot.send_message(
            chat_id=self.chat_id,
            text=text,
            parse_mode='HTML'
        )
        return True

    def send_message(self, text: str) -> bool:
        """
        Send message to chat

        Args:
            text: Message text (supports HTML formatting)

        Returns:
            True if sent successfully
        """
        # Run the async send in the event loop
        try:
            asyncio.run_coroutine_threadsafe(self._send_message(text), self._loop)
            return True
        except Exception as e:
            self.logger.error(f"Error sending message: {e}")
            return False
    
    def send_trading_signal(self, symbol: str, action: str, confidence: float, reasoning: str) -> bool:
        """Send trading signal"""
        if action == "HOLD":
            return False
        
        emoji = "🟢 BUY" if action == "BUY" else "🔴 SELL"
        conf_emoji = "🔥" if confidence > 0.8 else "⚡" if confidence > 0.6 else "⚠️"
        
        text = f"""🤖 <b>Trading Signal</b>

{emoji} <b>{symbol}</b>
{conf_emoji} <b>Confidence:</b> {confidence:.0%}

💭 {reasoning}

🕐 {datetime.now().strftime('%H:%M:%S')}"""
        
        return self.send_message(text)
    
    def send_startup_message(self, symbols: list, mode: str = "TESTNET") -> bool:
        """Send startup message"""
        emoji = "🧪" if mode == "TESTNET" else "💰"
        symbols_text = ", ".join(symbols)
        
        text = f"""🚀 <b>Trading Bot Started</b>

{emoji} <b>Mode:</b> {mode}
📊 <b>Symbols:</b> {symbols_text}

✅ Ready! 🕐 {datetime.now().strftime('%H:%M:%S')}"""
        
        return self.send_message(text)
    
    def send_shutdown_message(self, runtime: str, total_trades: int = 0) -> bool:
        """Send shutdown message"""
        text = f"""🛑 <b>Bot Stopped</b>

⏱️ Runtime: {runtime}
💼 Trades: {total_trades}

👋 Goodbye!"""
        
        return self.send_message(text)
    
    def test_connection(self) -> bool:
        """Test connection"""
        return self.send_message(f"🧪 Test OK - {datetime.now().strftime('%H:%M:%S')}")
    
    async def _handle_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle incoming commands"""
        # Check authorization
        if str(update.effective_chat.id) != self.authorized_chat_id:
            await update.message.reply_text("❌ Unauthorized")
            return
        
        # Get command
        command = update.message.text.split()[0][1:].split('@')[0].lower()
        
        try:
            # Handle help command internally
            if command == 'help':
                formatted_message = self._generate_help_message()
                await update.message.reply_text(formatted_message, parse_mode='HTML')
            # Call registered callback for other commands
            elif command in self.command_callbacks:
                data = self.command_callbacks[command]()
                if data:
                    formatted_message = self._format_data(data, command)
                    await update.message.reply_text(formatted_message, parse_mode='HTML')
            else:
                await update.message.reply_text("❓ Unknown command")
                
        except Exception as e:
            self.logger.error(f"Command {command} error: {e}")
            await update.message.reply_text("❌ Error")
    
    def _format_data(self, data: dict, command: str) -> str:
        """Format structured data into beautiful HTML message for Telegram"""
        if "error" in data:
            return f"❌ <b>Error:</b> {data['error']}"
        
        if command == 'balance':
            return self._format_balance_data(data)
        elif command == 'positions':
            return self._format_positions_data(data)
        elif command == 'status':
            return self._format_status_data(data)
        elif command == 'trades':
            return self._format_trades_data(data)
        elif command == 'history':
            return self._format_history_data(data)
        elif command == 'orders':
            return self._format_orders_data(data)
        elif command == 'help':
            return self._generate_help_message()
        else:
            return f"❓ Unknown command: {command}"
    
    def _format_balance_data(self, data: dict) -> str:
        """Format balance data"""
        balances = data.get('balances', [])
        total_value = data.get('total_value_usdt', 0)
        
        if not balances:
            return "💰 <b>Wallet Balance</b>\n\n📊 No significant balances found"
        
        message = f"💰 <b>Wallet Balance</b>\n\n"
        message += f"💵 <b>Total Value:</b> ${total_value:.2f} USDT\n\n"
        
        for balance in balances:
            asset = balance['asset']
            amount = balance['amount']
            usd_value = balance['usd_value']
            
            if asset == 'USDT':
                message += f"💵 <b>{asset}:</b> {amount:.2f} (${usd_value:.2f})\n"
            else:
                message += f"🪙 <b>{asset}:</b> {amount:.6f} (${usd_value:.2f})\n"
        
        message += f"\n🕐 {datetime.now().strftime('%H:%M:%S')}"
        return message
    
    def _format_positions_data(self, data: dict) -> str:
        """Format positions data"""
        positions = data.get('positions', [])
        total_value = data.get('total_value', 0)
        
        if not positions:
            return "📊 <b>Open Positions</b>\n\n📈 No open positions"
        
        message = f"📊 <b>Open Positions</b>\n\n"
        message += f"💰 <b>Total Value:</b> ${total_value:.2f}\n\n"
        
        for position in positions:
            symbol = position['symbol']
            quantity = position['quantity']
            current_price = position.get('current_price')
            position_value = position.get('position_value')
            
            message += f"🔸 <b>{symbol}</b>\n"
            message += f"   📦 Quantity: {quantity:.6f}\n"
            
            if current_price and position_value:
                message += f"   💰 Price: ${current_price:.4f}\n"
                message += f"   💵 Value: ${position_value:.2f}\n"
            else:
                message += f"   ⚠️ Price data unavailable\n"
            message += "\n"
        
        message += f"🕐 {datetime.now().strftime('%H:%M:%S')}"
        return message
    
    def _format_status_data(self, data: dict) -> str:
        """Format status data"""
        runtime_seconds = data.get('runtime_seconds', 0)
        total_trades = data.get('total_trades', 0)
        successful_trades = data.get('successful_trades', 0)
        success_rate = data.get('success_rate', 0)
        active_positions = data.get('active_positions', 0)
        
        # Convert runtime to readable format
        hours = int(runtime_seconds // 3600)
        minutes = int((runtime_seconds % 3600) // 60)
        seconds = int(runtime_seconds % 60)
        runtime_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        
        message = f"🤖 <b>Bot Status</b>\n\n"
        message += f"⏱️ <b>Runtime:</b> {runtime_str}\n"
        message += f"💼 <b>Total Trades:</b> {total_trades}\n"
        message += f"✅ <b>Successful:</b> {successful_trades}\n"
        message += f"📊 <b>Success Rate:</b> {success_rate:.1f}%\n"
        message += f"📈 <b>Active Positions:</b> {active_positions}\n\n"
        message += f"🟢 <b>Status:</b> Running\n"
        message += f"🕐 {datetime.now().strftime('%H:%M:%S')}"
        
        return message
    
    def _format_trades_data(self, data: dict) -> str:
        """Format trades data"""
        trades = data.get('trades', [])
        stats = data.get('stats', {})
        
        message = f"📈 <b>Recent Trades</b>\n\n"
        
        # Add statistics summary
        total_trades = stats.get('total_trades', 0)
        total_pnl = stats.get('total_pnl', 0)
        success_rate = stats.get('success_rate', 0)
        
        message += f"📊 <b>Statistics:</b>\n"
        message += f"   💼 Total Trades: {total_trades}\n"
        message += f"   💰 Total P&L: ${total_pnl:.2f}\n"
        message += f"   📈 Success Rate: {success_rate:.1f}%\n\n"
        
        if not trades:
            message += "📋 No recent trades"
        else:
            message += f"📋 <b>Last {len(trades)} Trades:</b>\n\n"
            
            for trade in trades:
                side = trade['side']
                symbol = trade['symbol']
                quantity = trade['quantity']
                price = trade['price']
                amount_usd = trade['amount_usd']
                timestamp = trade['timestamp']
                
                # Parse timestamp
                if isinstance(timestamp, str):
                    dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                else:
                    dt = timestamp
                
                time_str = dt.strftime('%m/%d %H:%M')
                side_emoji = "🟢" if side == "BUY" else "🔴"
                
                message += f"{side_emoji} <b>{side} {symbol}</b>\n"
                message += f"   📦 {quantity:.6f} @ ${price:.4f}\n"
                message += f"   💵 ${amount_usd:.2f} • {time_str}\n\n"
        
        message += f"🕐 {datetime.now().strftime('%H:%M:%S')}"
        return message
    
    def _format_history_data(self, data: dict) -> str:
        """Format history data"""
        stats = data.get('stats', {})
        open_positions = data.get('open_positions', [])
        
        message = f"📊 <b>Trading History</b>\n\n"
        
        # Overall statistics
        total_trades = stats.get('total_trades', 0)
        closed_trades = stats.get('closed_trades', 0)
        total_pnl = stats.get('total_pnl', 0)
        success_rate = stats.get('success_rate', 0)
        profitable_trades = stats.get('profitable_trades', 0)
        
        message += f"📈 <b>Overall Performance:</b>\n"
        message += f"   💼 Total Trades: {total_trades}\n"
        message += f"   ✅ Closed Pairs: {closed_trades}\n"
        message += f"   💰 Total P&L: ${total_pnl:.2f}\n"
        message += f"   🎯 Profitable: {profitable_trades}/{closed_trades}\n"
        message += f"   📊 Success Rate: {success_rate:.1f}%\n\n"
        
        # Open positions with P&L
        if open_positions:
            message += f"📈 <b>Open Positions P&L:</b>\n\n"
            
            total_unrealized_pnl = 0
            for position in open_positions:
                symbol = position['symbol']
                entry_price = position['entry_price']
                current_value = position.get('current_value', 0)
                pnl = position.get('pnl', 0)
                pnl_pct = position.get('pnl_pct', 0)
                days_held = position.get('days_held', 0)
                
                total_unrealized_pnl += pnl
                
                pnl_emoji = "🟢" if pnl >= 0 else "🔴"
                message += f"{pnl_emoji} <b>{symbol}</b>\n"
                message += f"   💰 Entry: ${entry_price:.4f}\n"
                message += f"   📊 Current: ${current_value:.2f}\n"
                message += f"   💵 P&L: ${pnl:.2f} ({pnl_pct:+.2f}%)\n"
                message += f"   📅 Held: {days_held} days\n\n"
            
            message += f"💰 <b>Unrealized P&L:</b> ${total_unrealized_pnl:.2f}\n\n"
        else:
            message += "📈 No open positions\n\n"
        
        message += f"🕐 {datetime.now().strftime('%H:%M:%S')}"
        return message
    
    def _format_orders_data(self, data: dict) -> str:
        """Format orders data"""
        orders = data.get('orders', [])
        summary = data.get('summary', {})
        
        message = f"📋 <b>Exchange Orders</b>\n\n"
        
        # Add summary statistics if available
        total_orders = summary.get('total_orders', len(orders))
        open_orders = summary.get('open_orders', 0)
        filled_orders = summary.get('filled_orders', 0)
        
        if summary:
            message += f"📊 <b>Summary:</b>\n"
            message += f"   📋 Total Orders: {total_orders}\n"
            message += f"   🔓 Open Orders: {open_orders}\n"
            message += f"   ✅ Filled Orders: {filled_orders}\n\n"
        
        if not orders:
            message += "📝 No orders found"
        else:
            message += f"📋 <b>Recent Orders ({len(orders)}):</b>\n\n"
            
            for order in orders:
                order_id = order.get('orderId', order.get('id', 'N/A'))
                symbol = order.get('symbol', 'N/A')
                side = order.get('side', 'N/A')
                order_type = order.get('type', 'N/A')
                status = order.get('status', 'N/A')
                quantity = order.get('origQty', order.get('quantity', 0))
                price = order.get('price', 0)
                filled_qty = order.get('executedQty', order.get('filled', 0))
                timestamp = order.get('time', order.get('timestamp'))
                
                # Status emoji
                if status.upper() in ['FILLED', 'COMPLETED']:
                    status_emoji = "✅"
                elif status.upper() in ['NEW', 'OPEN']:
                    status_emoji = "🔓"
                elif status.upper() in ['CANCELED', 'CANCELLED']:
                    status_emoji = "❌"
                elif status.upper() in ['PARTIALLY_FILLED']:
                    status_emoji = "🔄"
                else:
                    status_emoji = "❓"
                
                # Side emoji
                side_emoji = "🟢" if side.upper() == "BUY" else "🔴" if side.upper() == "SELL" else "⚪"
                
                # Parse timestamp
                if timestamp:
                    try:
                        if isinstance(timestamp, (int, float)):
                            # Unix timestamp (possibly in milliseconds)
                            if timestamp > 1e12:  # Likely milliseconds
                                dt = datetime.fromtimestamp(timestamp / 1000)
                            else:  # Likely seconds
                                dt = datetime.fromtimestamp(timestamp)
                        else:
                            # String timestamp
                            dt = datetime.fromisoformat(str(timestamp).replace('Z', '+00:00'))
                        time_str = dt.strftime('%m/%d %H:%M')
                    except:
                        time_str = "Unknown"
                else:
                    time_str = "Unknown"
                
                message += f"{status_emoji} <b>#{order_id}</b>\n"
                message += f"   {side_emoji} <b>{side} {symbol}</b> ({order_type})\n"
                message += f"   📦 Qty: {float(quantity):.6f}"
                
                if float(filled_qty) > 0:
                    fill_pct = (float(filled_qty) / float(quantity)) * 100 if float(quantity) > 0 else 0
                    message += f" (Filled: {float(filled_qty):.6f} - {fill_pct:.1f}%)\n"
                else:
                    message += "\n"
                
                if float(price) > 0:
                    message += f"   💰 Price: ${float(price):.4f}\n"
                else:
                    message += f"   💰 Price: Market\n"
                
                message += f"   📅 {time_str} • {status_emoji} {status}\n\n"
        
        message += f"🕐 {datetime.now().strftime('%H:%M:%S')}"
        return message
    
    def _format_help_data(self, data: dict) -> str:
        """Format help data"""
        commands = data.get('commands', [])
        
        message = f"🤖 <b>Bot Commands</b>\n\n"
        message += f"Available commands:\n\n"
        
        for cmd in commands:
            command = cmd['command']
            description = cmd['description']
            message += f"/{command} - {description}\n"
        
        message += f"\n💡 <b>Tip:</b> All commands work in this chat\n"
        message += f"🕐 {datetime.now().strftime('%H:%M:%S')}"
        
        return message
    
    def _generate_help_message(self) -> str:
        """Generate help message based on registered commands"""
        # Define descriptions for known commands
        command_descriptions = {
            'balance': 'Show wallet balances',
            'positions': 'Show open positions', 
            'status': 'Show bot status and stats',
            'trades': 'Show recent trade history',
            'history': 'Show detailed trading statistics',
            'orders': 'Show all exchange orders',
            'help': 'Show this help message'
        }
        
        message = f"🤖 <b>Bot Commands</b>\n\n"
        message += f"Available commands:\n\n"
        
        # Add help command first
        message += f"/help - Show this help message\n"
        
        # Add registered commands
        for command in sorted(self.command_callbacks.keys()):
            description = command_descriptions.get(command, 'Bot command')
            message += f"/{command} - {description}\n"
        
        message += f"\n💡 <b>Tip:</b> All commands work in this chat\n"
        message += f"🕐 {datetime.now().strftime('%H:%M:%S')}"
        
        return message


def create_messaging_interface(bot_token: str = None, chat_id: str = None) -> MessagingInterface:
    """
    Create messaging interface with environment variable fallback
    
    Args:
        bot_token: Telegram bot token (uses TELEGRAM_BOT_TOKEN env var if None)
        chat_id: Chat ID (uses TELEGRAM_CHAT_ID env var if None)
        
    Returns:
        MessagingInterface instance
        
    Example:
        # Using environment variables
        messaging = create_messaging_interface()
        
        # Or direct parameters
        messaging = create_messaging_interface("your_token", "your_chat_id")
    """
    return MessagingInterface(
        bot_token or os.getenv('TELEGRAM_BOT_TOKEN', ''),
        chat_id or os.getenv('TELEGRAM_CHAT_ID', '')
    )


# =============================================================================
# COMPLETE USAGE EXAMPLE
# =============================================================================
if __name__ == "__main__":
    """
    Complete usage example for MessagingInterface
    Demonstrates automatic thread startup and command registration
    """
    import time
    
    # Configure logging to see the messaging interface logs
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),  # Console output
        ]
    )
    
    print("🚀 Creating messaging interface...")
    
    # Create interface - automatically starts in background thread
    messaging = create_messaging_interface(
        bot_token='8336193307:AAHO_bP4r9D5gOyDEtFoOAzTQB9AkiC3X8o',
        chat_id='-4930325188'  # Replace with your chat ID
    )
    
    # Give it a moment to start up
    time.sleep(2)
    
    print("✅ Messaging interface created and running")
    
    # Register command handlers with proper data structures
    def get_balance():
        return {
            'balances': [
                {'asset': 'USDT', 'amount': 1000.0, 'usd_value': 1000.0},
                {'asset': 'BTC', 'amount': 0.025, 'usd_value': 1125.0},
                {'asset': 'ETH', 'amount': 0.5, 'usd_value': 1000.0}
            ],
            'total_value_usdt': 3125.0
        }
    
    def get_status():
        return {
            'runtime_seconds': 7200,  # 2 hours
            'total_trades': 15,
            'successful_trades': 12,
            'success_rate': 80.0,
            'active_positions': 3
        }
    
    def get_positions():
        return {
            'positions': [
                {
                    'symbol': 'BTCUSDT',
                    'quantity': 0.025,
                    'current_price': 45000.0,
                    'position_value': 1125.0
                },
                {
                    'symbol': 'ETHUSDT', 
                    'quantity': 0.5,
                    'current_price': 2000.0,
                    'position_value': 1000.0
                }
            ],
            'total_value': 2125.0
        }
    
    # Register commands
    messaging.register_command("balance", get_balance)
    messaging.register_command("status", get_status)
    messaging.register_command("positions", get_positions)
    
    print("📝 Commands registered: balance, status, positions")
    
    # Test message sending
    print("📤 Sending test messages...")
    messaging.send_startup_message(["BTCUSDT", "ETHUSDT", "ADAUSDT"], "LIVE")
    messaging.send_trading_signal("BTCUSDT", "BUY", 0.85, "Strong bullish momentum detected")
    messaging.test_connection()
    
    # Keep running for demo
    print("🤖 Bot running... Send /help, /balance, /status, or /positions in Telegram")
    print("Press Ctrl+C to stop")
    
    try:
        time.sleep(60)  # Run for 1 minute
    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
    
    # Send shutdown message
    messaging.send_shutdown_message("1m 30s", 5)
    
    # Stop messaging
    messaging.stop()
    print("✅ Messaging stopped gracefully")


# Backward compatibility
TelegramNotifier = MessagingInterface
create_telegram_notifier = create_messaging_interface
