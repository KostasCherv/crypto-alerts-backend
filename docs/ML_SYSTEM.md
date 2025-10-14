# 🤖 ML Trading System

## Overview
Machine learning system for cryptocurrency trading that uses ensemble models to predict profitable trading opportunities.

## Key Features
- **Feature Engineering**: Technical indicators, price patterns, market microstructure
- **Ensemble Models**: XGBoost, Random Forest, Gradient Boosting
- **Training Pipeline**: Automated model training and validation
- **Real-time Predictions**: Integration with trading strategies

## Architecture
```
ml_trading_system/
├── data_collection.py     # Generate labeled training data
├── feature_engineering.py # Extract features from market data
├── ml_models.py          # Train ensemble models
├── train_pipeline.py     # Complete training workflow
├── ml_predictor.py       # Real-time predictions
└── test_ml_system.py     # System testing
```

## Setup
```bash
# Install ML dependencies
cd ml_trading_system
pip install -r requirements.txt

# Train models (first time)
python train_pipeline.py

# Test the system
python test_ml_system.py
```

## Usage
```python
import sys
sys.path.append('ml_trading_system')
from ml_predictor import MLPredictor

# Initialize predictor
predictor = MLPredictor()
predictor.load_models()

# Get ML prediction
prediction = predictor.predict_trade_profitability("BTC", "15m")
print(f"ML Signal: {prediction['ml_signal']}")
print(f"Confidence: {prediction['confidence_score']}%")
```

## Feature Engineering
- **Technical Indicators**: RSI, MACD, ADX, Bollinger Bands, EMAs
- **Price Patterns**: Candlestick patterns, price action
- **Market Data**: Volume trends, volatility measures
- **Time-based Features**: Hour of day, day of week patterns

## Model Performance
- **Precision**: >60% (minimize false positives)
- **F1-Score**: >0.55 (balanced precision/recall)
- **High-confidence accuracy**: >70% for signals with >70% probability

## Configuration
```python
# In train_pipeline.py
trainer = ModelTrainer()
results = trainer.run_complete_pipeline(
    target_samples=10000,  # Training examples
    symbols=['BTC', 'ETH'],  # Training assets
    timeframes=['15m', '1h', '4h']  # Training timeframes
)
```

## Troubleshooting
- **Insufficient data**: Increase `target_samples` or data collection period
- **Poor performance**: Check feature engineering, adjust hyperparameters
- **Model loading issues**: Run training pipeline first, check file permissions
