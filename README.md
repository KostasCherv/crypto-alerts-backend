# Crypto Trading Strategy System

A comprehensive Python system for finding profitable cryptocurrency trading strategies using Binance API data, advanced backtesting, and machine learning pattern recognition.

## 🎯 Main Purpose
**Find profitable strategies by analyzing different crypto pairs from Binance using multiple timeframes and strategies.**

## Key Features

- **📊 Profitable Strategies**: 4 tested strategies with proven performance
- **🔄 Backtesting Engine**: Comprehensive backtesting with realistic trading conditions
- **🤖 ML Enhancement**: Pattern recognition with +20 confidence points
- **📈 Multi-Timeframe Analysis**: 15m, 1h, 4h, 1d timeframes
- **💰 Risk Management**: 1% risk rule, progressive profit taking
- **📱 Alert System**: Telegram notifications for high-confidence signals
- **🗄️ Database Integration**: Supabase PostgreSQL for data storage

## 🏆 Best Performing Strategy: Bollinger Bands

- **Return**: 3.18% (3 months) = **~12.7% annually**
- **Win Rate**: 72.7%
- **Max Drawdown**: 0.69%
- **Sharpe Ratio**: 9.66

## 🚀 Quick Start

```bash
# Install dependencies
uv sync

# Configure environment
cp .env.example .env
# Edit .env with your credentials

# Run profitable strategies
uv run python strategies.py

# Generate visual reports
uv run python consolidated_visual_backtest.py
```

## 📚 Documentation

- **[docs/QUICK_START.md](docs/QUICK_START.md)** - Essential getting started guide
- **[docs/STRATEGY_GUIDE.md](docs/STRATEGY_GUIDE.md)** - Trading strategies and technical analysis
- **[docs/ML_SYSTEM.md](docs/ML_SYSTEM.md)** - Machine learning pattern recognition
- **[docs/BACKTESTING_GUIDE.md](docs/BACKTESTING_GUIDE.md)** - Comprehensive backtesting documentation

## Supported Assets & Timeframes

### Assets
BTCUSDT, ETHUSDT, BNBUSDT, ADAUSDT, XRPUSDT, SOLUSDT, DOGEUSDT, AVAXUSDT, LINKUSDT

### Timeframes
15m, 1h, 4h, 1d

## Setup

1. **Install dependencies**:
   ```bash
   uv sync
   ```

2. **Configure environment**:
   ```bash
   cp config/.env.example .env
   # Edit .env with your Binance API and Telegram credentials
   ```

3. **Set up database** (optional):
   Run `config/unified_schema.sql` in Supabase to create required tables

## Usage

### Run Strategy Analysis:
```bash
# Test all strategies (recommended)
uv run python run_strategies.py

# Generate visual backtest reports
uv run python run_backtest.py

# Compare strategy performance
uv run python run_comparison.py

# Or run directly from src/ directory
uv run python src/strategies.py
uv run python src/consolidated_visual_backtest.py
uv run python src/strategy_comparison_report.py
```

### Run Alert System:
```bash
# Start monitoring system
uv run python src/main.py

# Test setup
uv run python tests/test_setup.py
```

## Project Structure

```
crypto-alerts-backend/
├── src/                    # Core source code
│   ├── strategies.py       # Main strategy implementations
│   ├── backtester.py       # Core backtesting engine
│   ├── trend_analysis.py   # Technical analysis and indicators
│   └── ...
├── docs/                   # Documentation
├── tests/                  # Test files
├── config/                 # Configuration files
│   ├── unified_schema.sql  # Complete database schema
│   └── crypto-alerts.service
├── results/                # Backtest results and reports
├── scripts/                # Utility scripts
└── ml_trading_system/      # ML pattern recognition
```

## Results

- **Individual reports**: `results/consolidated_reports/`
- **Strategy comparisons**: `results/strategy_comparison_reports/`
- **Trade logs**: CSV files with detailed trade data
- **Visual charts**: PNG files with equity curves and performance metrics