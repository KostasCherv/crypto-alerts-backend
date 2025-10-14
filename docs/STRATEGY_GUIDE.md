# 📊 Strategy Guide

## Available Strategies

### 1. Bollinger Bands Strategy
- **Logic**: Buy at lower band + RSI < 40, Sell at upper band + RSI > 60
- **Best for**: Volatile markets with mean reversion
- **Risk-Reward**: 1:2.5
- **Position Size**: 1% risk per trade

### 2. RSI Mean Reversion
- **Logic**: Buy RSI < 30, Sell RSI > 70
- **Best for**: Ranging markets (ADX < 25)

### 3. EMA Crossover
- **Logic**: Fast EMA (8) crosses above/below Slow EMA (21)
- **Best for**: Trending markets (ADX > 25)

### 4. MACD Strategy
- **Logic**: MACD line crosses signal line
- **Best for**: Momentum trading

## Technical Indicators

### Bollinger Bands
- **Purpose**: Volatility and mean reversion
- **Calculation**: 20-period SMA ± (2 × Standard Deviation)
- **Trading Signals**:
  - **Mean Reversion**: Price at bands in ranging markets
  - **Breakout**: Price breaks above/below bands

### RSI (Relative Strength Index)
- **Purpose**: Overbought/oversold conditions
- **Scale**: 0-100
- **Key Levels**:
  - **> 70**: Overbought (potential sell)
  - **< 30**: Oversold (potential buy)
  - **50**: Neutral

### ADX (Average Directional Index)
- **Purpose**: Trend strength measurement
- **Scale**: 0-100
- **Interpretation**:
  - **0-25**: Weak/Ranging (avoid trend trading)
  - **25-50**: Strong trend (good for trading)
  - **50+**: Very strong trend

## Risk Management

### Position Sizing (1% Risk Rule)
```python
risk_amount = account_balance * 0.01  # 1% risk
position_size = risk_amount / (entry_price - stop_loss)
```

### Stop Loss Placement
- **Mean Reversion**: Behind recent swing point
- **Breakout**: Below/above opposite Bollinger Band
- **Trend Following**: At middle Bollinger Band

## Usage Examples

### Run Backtest
```bash
# Test individual strategy
uv run python run_backtest.py

# Compare all strategies
uv run python run_comparison.py
```

## Best Practices

### Strategy Selection
1. **Trending Markets (ADX > 25)**: Use EMA Crossover, MACD
2. **Ranging Markets (ADX < 25)**: Use Bollinger Bands, RSI Mean Reversion

### Entry Criteria
- ✅ Multiple timeframe alignment
- ✅ Risk-reward ratio ≥ 1:2
- ✅ Clear stop loss level
