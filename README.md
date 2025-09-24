# Crypto Price Alert System Backend

A Python backend system for monitoring cryptocurrency prices, detecting trends, and sending alerts using Supabase and Binance API.

## Features

- **Price Monitoring**: Monitor crypto prices every 15 minutes using Binance API
- **Alert System**: Trigger alerts when prices reach target levels with crossover detection
- **Trend Analysis**: Calculate trend direction and strength using SMA and ADX
- **Notifications**: Send alerts for triggered price levels via Telegram
- **Database Integration**: Store data in Supabase PostgreSQL

## Setup

1. **Install dependencies with uv**:
   ```bash
   # Install uv if you don't have it
   pip install uv
   
   # Install project dependencies
   uv sync
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
uv run python main.py
```

### Test setup:
```bash
uv run python test_setup.py
```

### Add example data:
```bash
uv run python example_usage.py
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

## GitHub Actions Deployment

### 1. Set up GitHub Secrets

Go to your repository → Settings → Secrets and variables → Actions, and add:

```
SUPABASE_URL=your_supabase_url_here
SUPABASE_KEY=your_supabase_key_here
BINANCE_API_URL=https://api.binance.com/api/v3
TELEGRAM_BOT_TOKEN=your_telegram_bot_token_here
TELEGRAM_CHAT_ID=your_telegram_chat_id_here
```

### 2. Enable GitHub Actions

The workflow is already configured in `.github/workflows/price-monitor.yml` and will:
- Run every 15 minutes automatically using `uv` for fast dependency management
- Test setup before running
- Cache uv dependencies for faster runs
- Upload error logs if something fails
- Allow manual triggering via GitHub UI

### 3. Monitor Execution

- Go to Actions tab in your GitHub repository
- View the "Crypto Price Monitor" workflow runs
- Check logs for any issues

## Configuration

The system supports two types of alerts:
- **one_time**: Alert triggers once, then deactivates
- **continuous**: Alert triggers only on price crossovers (not when staying in zone)

### Alert Directions:
- **above**: Triggers when price goes above target
- **below**: Triggers when price goes below target

## Local Development

For local development and testing:

```bash
# Install dependencies with uv
uv sync

# Test all connections
uv run python test_setup.py

# Run once manually
uv run python main.py

# Add test data
uv run python example_usage.py
```

## Benefits of using uv

- **Faster**: uv is significantly faster than pip for dependency resolution and installation
- **Reliable**: Better dependency resolution and conflict detection
- **Consistent**: Lock file ensures reproducible builds across environments
- **Modern**: Built with Rust for performance and reliability