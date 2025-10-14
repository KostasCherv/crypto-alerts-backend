# 📊 Crypto Backtesting Engine

A backtesting engine for cryptocurrency trading strategies with comprehensive performance metrics and realistic execution simulation.

## 🌟 Features

### Core Backtesting
- ✅ **Historical Data**: Uses downloaded OHLCV data (15m, 1h, 4h, 1d)
- ✅ **Chronological Simulation**: Iterate through candles chronologically to avoid look-ahead bias
- ✅ **Signal Generation**: Integrates with strategy implementations
- ✅ **Confidence-Based Trading**: Only execute trades when confidence ≥ threshold (default 70%)

### Risk Management
- ✅ **Position Sizing**: 1% risk rule - `position_size = account_risk / (entry - stop_loss)`
- ✅ **Stop Loss & Take Profit**: Automatic exit based on support/resistance levels
- ✅ **Maximum Drawdown Protection**: Stop trading after configurable drawdown (default 20%)
- ✅ **Single Position Limit**: Prevents over-leveraging

### Realistic Execution
- ✅ **Slippage Simulation**: 0.1% slippage per trade to simulate real execution
- ✅ **Trading Fees**: 0.1% Binance trading fees included
- ✅ **Proper Entry/Exit**: Checks high/low for stop loss/take profit hits within candle

### Performance Metrics
- ✅ **Total Return**: Overall percentage return
- ✅ **Win Rate**: Percentage of winning trades
- ✅ **Profit Factor**: Gross profit / gross loss
- ✅ **Sharpe Ratio**: Risk-adjusted returns (annualized)
- ✅ **Maximum Drawdown**: Largest peak-to-trough decline
- ✅ **Expectancy**: Expected value per trade
- ✅ **Win/Loss Streaks**: Longest consecutive wins/losses
- ✅ **Average Trade Duration**: Time in market per trade

### Visualization & Reporting
- ✅ **Equity Curve**: Visual representation of account growth
- ✅ **Drawdown Chart**: Visualize drawdown periods
- ✅ **Buy & Hold Comparison**: Compare against passive strategy
- ✅ **CSV Export**: Detailed trade logs for further analysis
- ✅ **Comprehensive Reports**: Full performance breakdown

## 🚀 Quick Start

### Run Backtests

```bash
# Run individual backtest
uv run python run_backtest.py

# Run strategy comparison
uv run python run_comparison.py

# Run multiple strategies
uv run python run_strategies.py
```

### Basic Usage

```python
from src.backtester import BacktestEngine
from src.data_manager import DataManager

# Get data
data_manager = DataManager()
data = data_manager.get_data('BTC', '4h', 365)

# Run backtest
engine = BacktestEngine()
metrics = engine.run_backtest(data, 'bollinger_bands')

# View results
print(f"Total Return: {metrics['total_return']:.2f}%")
print(f"Win Rate: {metrics['win_rate']:.2f}%")
print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
```

## 📋 Configuration Options

### Available Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `initial_capital` | float | 10000.0 | Starting capital in USD |
| `risk_per_trade` | float | 0.01 | Risk per trade (1% = 0.01) |
| `min_confidence` | float | 70.0 | Minimum confidence to trade (0-100) |
| `slippage` | float | 0.001 | Slippage per trade (0.1% = 0.001) |
| `trading_fee` | float | 0.001 | Trading fee per trade (0.1% = 0.001) |
| `max_drawdown_stop` | float | 0.20 | Stop trading after this drawdown (20% = 0.20) |

## 📊 Performance Metrics Explained

### Return Metrics
- **Total Return**: `(final_equity - initial_capital) / initial_capital * 100`
- **Alpha**: Excess return compared to buy-and-hold strategy
- **Expectancy**: `(win_rate × avg_win) - (loss_rate × avg_loss)`

### Risk Metrics
- **Sharpe Ratio**: `(mean_return / std_return) × √periods_per_year`
  - < 1: Poor risk-adjusted returns
  - 1-2: Good risk-adjusted returns
  - \> 2: Excellent risk-adjusted returns
- **Maximum Drawdown**: Largest peak-to-trough decline in equity
- **Profit Factor**: `gross_profit / gross_loss`
  - < 1: Losing strategy
  - 1-2: Profitable but risky
  - \> 2: Strong strategy

### Trade Metrics
- **Win Rate**: `winning_trades / total_trades × 100`
- **Average Win/Loss**: Mean P&L of winning/losing trades
- **Longest Streaks**: Maximum consecutive wins/losses

## 🎯 Best Practices

### Avoiding Look-Ahead Bias
The engine prevents look-ahead bias by:
1. Only using data available up to the current candle
2. Calculating indicators on historical slices
3. Never using future data in signal generation

### Preventing Overfitting
1. **Use Walk-Forward Analysis**: Always validate on out-of-sample data
2. **Test Multiple Symbols**: Strategy should work across different assets
3. **Test Multiple Timeframes**: Robust strategies work on various intervals
4. **Avoid Over-Optimization**: Don't optimize too many parameters
5. **Keep It Simple**: Complex strategies often overfit

### Realistic Expectations
- Include slippage and fees in all backtests
- Use conservative position sizing (1-2% risk)
- Account for maximum drawdown in capital planning
- Expect live performance to be 20-30% worse than backtest

## 📁 Output Files

All results are saved to `results/` directory:

- `consolidated/`: Individual backtest reports
- `strategy_comparison_reports/`: Strategy comparison reports
- `comparisons/`: Session-based comparisons

## 🔧 Advanced Usage

### Multiple Timeframe Analysis

```python
from src.data_manager import DataManager
from src.backtester import BacktestEngine

data_manager = DataManager()
timeframes = ['15m', '1h', '4h', '1d']

for tf in timeframes:
    data = data_manager.get_data('BTC', tf, 365)
    engine = BacktestEngine()
    metrics = engine.run_backtest(data, 'bollinger_bands')
    print(f"{tf}: {metrics['total_return']:.2f}%")
```

### Multi-Asset Testing

```python
assets = ['BTC', 'ETH']
results = []

for asset in assets:
    data = data_manager.get_data(asset, '4h', 365)
    engine = BacktestEngine()
    metrics = engine.run_backtest(data, 'bollinger_bands')
    results.append(metrics)
```

## ⚠️ Important Considerations

### Data Quality
- Uses downloaded historical data (365 days for BTC/ETH)
- Data covers October 2024 - October 2025
- Always verify data completeness before backtesting

### Execution Assumptions
- Assumes instant order execution at candle close
- Slippage is applied uniformly (may vary in reality)
- Does not account for liquidity constraints
- Stop loss/take profit assumed to execute at exact levels

### Strategy Limitations
- Past performance does not guarantee future results
- Market conditions change over time
- Backtests cannot account for black swan events
- Emotional factors not included in simulation

## 🐛 Troubleshooting

### No Trades Executed
- Lower `min_confidence` threshold
- Check if signals are being generated
- Verify sufficient historical data
- Ensure position sizing allows trades

### Poor Performance
- Try different timeframes
- Adjust confidence threshold
- Test on different market conditions

---

**⚠️ Disclaimer**: This backtesting engine is for educational and research purposes only. Cryptocurrency trading involves substantial risk of loss. Past performance is not indicative of future results. Always do your own research and never invest more than you can afford to lose.
