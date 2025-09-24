# Crypto Price Alert System Backend

A Python backend system for monitoring cryptocurrency prices, detecting trends, and sending alerts using Supabase and Binance API.

## Features

- **Price Monitoring**: Monitor crypto prices every 15 minutes using Binance API
- **Alert System**: Trigger alerts when prices reach target levels
- **Trend Analysis**: Calculate trend direction and strength using SMA and ADX
- **Notifications**: Send alerts for triggered price levels
- **Database Integration**: Store data in Supabase PostgreSQL

## Setup

1. **Install dependencies**:
   ```bash
   pip install -e .
   ```

2. **Set up environment variables**:
   ```bash
   cp .env.example .env
   ```
   
   Edit `.env` with your credentials:
   ```
   SUPABASE_URL=your_supabase_url_here
   SUPABASE_KEY=your_supabase_key_here
   BINANCE_API_URL=https://api.binance.com/api/v3
   TELEGRAM_BOT_TOKEN=your_telegram_bot_token_here
   TELEGRAM_CHAT_ID=your_telegram_chat_id_here
   ```

3. **Set up Telegram Bot**:
   - Create a new bot by messaging @BotFather on Telegram
   - Get your bot token and add it to `.env`
   - Get your chat ID by messaging @userinfobot or using the bot API
   - Add your chat ID to `.env`

4. **Set up Supabase database**:
   Run the SQL schema from the specification to create the required tables:
   - `price_levels`
   - `alerts`
   - `trends`

## Usage

### Run the monitoring system:
```bash
python main.py
```

### Manual testing:
```python
from price_monitor import PriceMonitor
from trend_analysis import TrendAnalyzer
from notifications import NotificationService

# Monitor prices
monitor = PriceMonitor()
triggered = monitor.run_monitoring_cycle()

# Analyze trends
analyzer = TrendAnalyzer()
analyzed = analyzer.run_trend_analysis()

# Send notifications
notifier = NotificationService()
sent = notifier.run_notification_cycle()
```

## Database Schema

The system uses three main tables:

- **price_levels**: Store price targets to monitor
- **alerts**: Record triggered alerts
- **trends**: Store trend analysis results

## API Integration

- **Binance API**: Fetches current and historical price data
- **Supabase**: Database operations and data storage
- **Telegram Bot API**: Sends price alert notifications

## Deployment

This system is designed to run as a scheduled job (every 15 minutes) using:
- GitHub Actions
- Cron jobs
- Cloud functions
- Container orchestration

## Configuration

The system supports two types of alerts:
- **one_time**: Alert triggers once, then deactivates
- **continuous**: Alert triggers every time price reaches target
