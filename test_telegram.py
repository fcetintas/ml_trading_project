#!/usr/bin/env python3
"""
Test Telegram bot commands functionality
"""

import asyncio
import os
from telegram import Bot, Update
from telegram.ext import Application, CommandHandler, ContextTypes

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

async def test_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Test command handler"""
    await update.message.reply_text("✅ Test command working! Bot is responding.")

async def balance_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Test balance command"""
    await update.message.reply_text("💰 <b>Test Balance</b>\n\n• USDT: 100.00\n• BTC: 0.001", parse_mode='HTML')

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Test help command"""
    text = """
🤖 <b>Test Bot Commands</b>

/test - Test command
/balance - Test balance
/help - This help message
    """.strip()
    await update.message.reply_text(text, parse_mode='HTML')

async def main():
    """Main function to run the test bot"""
    print("🤖 Starting Telegram test bot...")
    
    # Create application
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    
    # Add command handlers
    application.add_handler(CommandHandler("test", test_command))
    application.add_handler(CommandHandler("balance", balance_command))
    application.add_handler(CommandHandler("help", help_command))
    
    # Set commands in Telegram
    from telegram import BotCommand
    commands = [
        BotCommand("test", "Test command"),
        BotCommand("balance", "Test balance"),
        BotCommand("help", "Show help"),
    ]
    
    bot = Bot(token=TELEGRAM_BOT_TOKEN)
    await bot.set_my_commands(commands)
    print("✅ Commands set in Telegram")
    
    # Start polling
    print("🚀 Starting polling...")
    await application.run_polling(
        poll_interval=1.0,
        timeout=10,
        drop_pending_updates=True
    )

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Test bot stopped")