# 🚀 Quick Start Guide

## Installation
```bash
# Install dependencies
uv sync

# Download crypto data (first time)
uv run python download_data.py
```

## Run Backtests
```bash
# Run individual backtest
uv run python run_backtest.py

# Run strategy comparison
uv run python run_comparison.py

# Run multiple strategies
uv run python run_strategies.py
```

## Available Data
- **Assets**: BTC, ETH (with full year data)
- **Timeframes**: 15m, 1h, 4h, 1d
- **Coverage**: October 2024 - October 2025 (365 days)

## Key Files
- `src/strategies.py` - Trading strategy implementations
- `src/backtester.py` - Core backtesting engine
- `src/consolidated_visual_backtest.py` - Visual backtesting system
- `src/data_manager.py` - Data download and management

## Configuration
- **Position Sizing**: 1% risk per trade
- **Trading Fees**: 0.1% (Binance realistic)
- **Slippage**: 0.1% per trade
- **Min Confidence**: 70% (configurable)
