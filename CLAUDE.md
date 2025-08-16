# Cryptocurrency Sentiment Trading Bot

A sophisticated trading bot that analyzes cryptocurrency news sentiment using FinBERT and executes trades on Binance based on sentiment signals.

## 🚀 Features

- **FinBERT Sentiment Analysis**: Uses state-of-the-art financial BERT model for accurate sentiment analysis of crypto news
- **Binance Integration**: Seamless integration with Binance API supporting both testnet and live trading
- **CryptoPanic News Feed**: Real-time cryptocurrency news from CryptoPanic API
- **Safety First**: Defaults to testnet mode with configurable trade limits and confidence thresholds
- **Comprehensive Logging**: Detailed logging and error handling for monitoring and debugging
- **Modular Architecture**: Clean, maintainable code structure with separate concerns

## 📁 Project Structure

```
├── main.py                 # Core trading bot logic and main entry point
├── config.py              # Configuration management and settings
├── sentiment_analyzer.py   # FinBERT-based sentiment analysis module
├── news_fetcher.py        # CryptoPanic API news collection
├── trader.py              # Binance trading interface with testnet support
├── requirements.txt       # Python dependencies
├── .env.example          # Environment variables template
└── CLAUDE.md             # Project documentation (this file)
```

## 🛠️ Installation

1. **Clone and setup the project**:
   ```bash
   # Navigate to project directory
   cd /path/to/your/project
   
   # Install dependencies
   pip install -r requirements.txt
   ```

2. **Configure environment variables**:
   ```bash
   # Copy the example environment file
   cp .env.example .env
   
   # Edit .env with your API keys
   nano .env
   ```

3. **Get required API keys**:
   - **Binance API**: Get testnet keys from [https://testnet.binance.vision/](https://testnet.binance.vision/)
   - **CryptoPanic API**: Get free API key from [https://cryptopanic.com/developers/api/](https://cryptopanic.com/developers/api/)

4. **Setup Telegram Notifications (Optional)**:
   - Create a Telegram bot via [@BotFather](https://t.me/botfather)
   - Get your chat ID by messaging [@userinfobot](https://t.me/userinfobot)
   - Add the credentials to your `.env` file

## ⚙️ Configuration

### Environment Variables (.env)

```bash
# Binance API (use testnet for safety)
BINANCE_API_KEY=your_testnet_api_key
BINANCE_SECRET_KEY=your_testnet_secret_key

# CryptoPanic API v2 (Developer tier)
CRYPTOPANIC_API_KEY=73485d83c29aca53e9573da2bcb96d1c1447fc58

# Telegram Bot Notifications (Optional)
TELEGRAM_BOT_TOKEN=your_telegram_bot_token_here
TELEGRAM_CHAT_ID=your_telegram_chat_id_here

# Safety: Use testnet by default
USE_TESTNET=true
```

### Trading Parameters

Default settings in `config.py`:
- **Max trade amount**: $10 USD per trade
- **Sentiment thresholds**: 60% confidence for buy/sell signals
- **Trading interval**: 5 minutes between cycles
- **Supported symbols**: BTC/USDT, ETH/USDT, ADA/USDT

## 🚀 Usage

### Basic Usage

```bash
# Run the bot (uses testnet by default)
python main.py
```

### Running with Custom Configuration

Modify parameters in `config.py` or through environment variables:

```python
# Example custom configuration
trading_config = TradingConfig(
    max_trade_amount=15.0,          # $15 max per trade
    sentiment_threshold_buy=0.7,     # 70% confidence for buy
    confidence_threshold=0.65,       # 65% minimum confidence
    symbols=['BTCUSDT', 'ETHUSDT']  # Trade only BTC and ETH
)
```

## 🔒 Safety Features

1. **Testnet Default**: Always starts in testnet mode to prevent accidental real trading
2. **Small Trade Sizes**: Maximum $10 per trade by default
3. **High Confidence Thresholds**: Requires 60%+ sentiment confidence
4. **Comprehensive Error Handling**: Graceful failure handling and logging
5. **Position Tracking**: Prevents over-trading and tracks open positions

## 📊 How It Works

1. **News Collection**: Fetches recent cryptocurrency news from CryptoPanic API v2
2. **Sentiment Analysis**: Analyzes news sentiment using FinBERT model
3. **Signal Generation**: Creates trading signals based on:
   - Sentiment polarity (positive/negative/neutral)
   - Confidence level (must exceed threshold)
   - Current position status
4. **Trade Execution**: Executes market orders on Binance based on signals
5. **Monitoring**: Logs all activities and maintains trading session statistics

## 🔌 CryptoPanic API v2 Integration

### API Endpoint Configuration

**Base URL**: `https://cryptopanic.com/api/developer/v2`  
**Authentication**: `auth_token=73485d83c29aca53e9573da2bcb96d1c1447fc58`  
**Plan Level**: Developer (with specific feature limitations)

### Available Parameters

The bot supports filtering news using these parameters:

- **currencies**: Filter by specific cryptocurrencies (e.g., `BTC,ETH,ADA`)
- **regions**: Language filtering (default: `en` for English)
- **filter**: Content filters like `rising`, `hot`, `bullish`, `bearish`, `important`
- **kind**: News type filtering (`news`, `media`, `all`)
- **public**: Set to `true` for public usage mode (recommended for apps)

### Example API Calls

```bash
# Get latest Bitcoin and Ethereum news
https://cryptopanic.com/api/developer/v2/posts/?auth_token=73485d83c29aca53e9573da2bcb96d1c1447fc58&currencies=BTC,ETH&public=true

# Get rising crypto news in English
https://cryptopanic.com/api/developer/v2/posts/?auth_token=73485d83c29aca53e9573da2bcb96d1c1447fc58&filter=rising&regions=en&public=true

# Get only news articles (exclude social media)
https://cryptopanic.com/api/developer/v2/posts/?auth_token=73485d83c29aca53e9573da2bcb96d1c1447fc58&kind=news&public=true
```

### Response Structure

Each news item contains:
- **id**: Unique identifier
- **title**: Article headline
- **description**: Brief summary
- **published_at**: Publication timestamp (ISO 8601)
- **instruments**: Associated cryptocurrencies with market data
- **votes**: Community engagement metrics (positive, negative, important, etc.)
- **source**: Publisher information
- **original_url**: Link to full article

### Rate Limits

**Developer Plan Limits**:
- Requests per second: Limited (check current usage)
- Monthly requests: Up to developer tier limit
- Error codes: 401 (unauthorized), 403 (forbidden), 429 (rate limited)

### RSS Feed Support

Alternative RSS access available:
```bash
# RSS format with currency filter
https://cryptopanic.com/api/developer/v2/posts/?auth_token=73485d83c29aca53e9573da2bcb96d1c1447fc58&currencies=BTC&format=rss
```

**Note**: RSS responses limited to 20 items regardless of plan level.

### Trading Logic

- **BUY Signal**: Triggered by strong positive sentiment (≥60% confidence) when no position held
- **SELL Signal**: Triggered by strong negative sentiment (≥60% confidence) when holding position
- **HOLD**: Default action when confidence is low or no clear sentiment direction

## 📱 Telegram Bot Notifications

The bot includes optional Telegram notifications to keep you informed of trading signals and bot status.

### 🚀 Features

- **Trading Signals**: Get instant notifications for BUY/SELL recommendations
- **Detailed Analysis**: Includes sentiment analysis, confidence levels, and reasoning
- **Bot Status**: Startup and shutdown notifications with session statistics
- **Rich Formatting**: HTML-formatted messages with emojis for easy reading

### 📋 Setup Guide

1. **Create a Telegram Bot**:
   - Message [@BotFather](https://t.me/botfather) on Telegram
   - Send `/newbot` and follow the instructions
   - Copy the bot token (format: `123456789:ABCdefGHIjklMNOpqrSTUvwxyz`)

2. **Get Your Chat ID**:
   - Start a conversation with your new bot
   - Message [@userinfobot](https://t.me/userinfobot) with `/start`
   - Copy your chat ID (format: `123456789`)

3. **Configure Environment Variables**:
   ```bash
   TELEGRAM_BOT_TOKEN=123456789:ABCdefGHIjklMNOpqrSTUvwxyz
   TELEGRAM_CHAT_ID=123456789
   ```

### 📲 Message Types

**🚀 Startup Notification**:
```
🚀 Crypto Trading Bot Started

🧪 Mode: TESTNET (Paper Trading)
📊 Monitoring: BTCUSDT, ETHUSDT, BNBUSDT
🤖 AI Features: FinBERT Sentiment + Technical Analysis

✅ Ready to analyze and trade!
```

**🟢 Buy Signal**:
```
🤖 Crypto Trading Signal

🟢 BUY BTCUSDT
🔥 Confidence: 85%
📊 Signal Type: Combined AI

💭 Analysis:
Combined AI signals: Sentiment POSITIVE + Technical BUY

🕐 Time: 2024-08-02 14:30:15
```

**🔴 Sell Signal**:
```
🤖 Crypto Trading Signal

🔴 SELL ETHUSDT
⚡ Confidence: 72%
📊 Signal Type: Sentiment Only

💭 Analysis:
Strong negative sentiment (0.78)

🕐 Time: 2024-08-02 14:35:22
```

**🛑 Shutdown Notification**:
```
🛑 Crypto Trading Bot Stopped

⏱️ Runtime: 2:15:30
💼 Total Trades: 5

🕐 Stopped: 2024-08-02 16:45:45

Thanks for using the crypto trading bot! 👋
```

### ⚙️ Configuration

Telegram notifications are **optional** and the bot will work without them. If configured:

- ✅ **Enabled**: Sends notifications for all BUY/SELL signals
- ⚠️ **Connection Failed**: Bot continues working, logs warning
- 📱 **Disabled**: No Telegram credentials provided (normal operation)

The system automatically tests the connection on startup and only enables notifications if successful.

## 📈 Monitoring

The bot provides comprehensive logging:

```
2024-01-01 10:00:00 - INFO - Processing BTCUSDT...
2024-01-01 10:00:01 - INFO - Analyzed 15 articles for BTC: positive (0.72)
2024-01-01 10:00:02 - INFO - BTCUSDT Signal: BUY | Sentiment: positive (0.72) | Reasoning: Strong positive sentiment
2024-01-01 10:00:03 - INFO - BUY executed: 0.0003 BTCUSDT at $45,230.50
```

## 🛡️ Risk Warnings

- **Use testnet first**: Always test with testnet before considering live trading
- **Start small**: Even in live trading, use minimal amounts
- **Monitor closely**: Sentiment-based trading can be volatile
- **No guarantees**: This is experimental software - trade at your own risk

## 🔧 Development

### Running Tests

```bash
# Install test dependencies
pip install pytest

# Run tests (when available)
pytest tests/
```

### Code Formatting

```bash
# Format code
black *.py

# Check code style
flake8 *.py
```

## 📝 Logs

- **File**: `trading_bot.log` (rotated at 10MB)
- **Console**: Real-time output with timestamps
- **Levels**: INFO, WARNING, ERROR for different event types

## 🤝 Support

For issues, questions, or contributions:
- Check the logs for error details
- Ensure all API keys are correctly configured
- Verify network connectivity to APIs
- Start with testnet mode for troubleshooting

## 📄 License

This project is for educational and research purposes. Use at your own risk in live trading environments.

---

**⚠️ Important**: This bot is designed for educational purposes and small-scale experimentation. Always use testnet mode first and never risk more than you can afford to lose.