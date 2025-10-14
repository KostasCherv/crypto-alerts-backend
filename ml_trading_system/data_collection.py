import os
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
from backtester import BacktestEngine, BacktestConfig
from trend_analysis import TrendAnalyzer

class DataCollector:
    """
    Collect labeled historical data for training ML models
    """
    
    def __init__(self):
        self.analyzer = TrendAnalyzer()
        self.symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT", "XRPUSDT"]
        self.intervals = ["15m", "1h", "4h"]
        
    def collect_labeled_data(self, target_samples: int = 10000) -> pd.DataFrame:
        """
        Collect labeled historical data using backtesting engine
        Each trade is labeled as profitable (1) or unprofitable (0)
        """
        print(f"Collecting {target_samples} labeled candle patterns...")
        
        all_data = []
        collected_samples = 0
        
        # Try different symbol and interval combinations
        for symbol in self.symbols:
            for interval in self.intervals:
                if collected_samples >= target_samples:
                    break
                    
                try:
                    print(f"Processing {symbol} {interval}...")
                    
                    # Configure backtest
                    config = BacktestConfig(
                        symbol=symbol,
                        interval=interval,
                        initial_capital=10000.0,
                        risk_per_trade=0.01,
                        min_confidence=60.0,  # Lower threshold to get more samples
                        slippage=0.001,
                        trading_fee=0.001,
                        max_drawdown_stop=0.30
                    )
                    
                    # Run backtest
                    engine = BacktestEngine(config)
                    engine.fetch_historical_data(limit=2000)  # Get more data for better patterns
                    
                    # Run with reduced lookback for more samples
                    metrics = engine.run_backtest(lookback_period=50)
                    
                    # Extract labeled data from trades
                    trade_data = self._extract_trade_features(engine)
                    all_data.extend(trade_data)
                    collected_samples += len(trade_data)
                    
                    print(f"Collected {len(trade_data)} samples from {symbol} {interval}")
                    
                except Exception as e:
                    print(f"Error processing {symbol} {interval}: {e}")
                    continue
                    
        # Convert to DataFrame
        if all_data:
            df = pd.DataFrame(all_data)
            print(f"Total collected samples: {len(df)}")
            return df
        else:
            raise ValueError("No data collected")
            
    def _extract_trade_features(self, engine: BacktestEngine) -> List[Dict]:
        """
        Extract features from trades and label them as profitable/unprofitable
        """
        trade_features = []
        
        # For each trade, extract features from the 50-candle window before entry
        df = engine.historical_data
        
        for trade in engine.trades:
            if not trade.entry_time or not trade.pnl:
                continue
                
            # Find the index of the entry time
            entry_idx = df[df['timestamp'] == trade.entry_time].index
            if len(entry_idx) == 0:
                continue
                
            entry_idx = entry_idx[0]
            
            # Get 50-candle window before entry (lookback period)
            start_idx = max(0, entry_idx - 50)
            end_idx = entry_idx
            window_df = df.iloc[start_idx:end_idx]
            
            if len(window_df) < 20:  # Need at least 20 candles for meaningful features
                continue
                
            # Extract features
            features = self._calculate_features(window_df, trade)
            
            # Label: 1 if profitable, 0 if not
            features['label'] = 1 if trade.pnl > 0 else 0
            features['pnl'] = trade.pnl
            features['pnl_percent'] = trade.pnl_percent or 0
            
            trade_features.append(features)
            
        return trade_features
        
    def _calculate_features(self, window_df: pd.DataFrame, trade) -> Dict:
        """
        Calculate comprehensive features from 50-candle window
        """
        features = {}
        
        # Convert to lists for trend analyzer
        closes = window_df['close'].tolist()
        highs = window_df['high'].tolist()
        lows = window_df['low'].tolist()
        volumes = window_df['volume'].tolist()
        opens = window_df['open'].tolist()
        
        # Basic price features
        features['current_price'] = closes[-1]
        features['price_change_5'] = (closes[-1] - closes[-6]) / closes[-6] if len(closes) >= 6 else 0
        features['price_change_10'] = (closes[-1] - closes[-11]) / closes[-11] if len(closes) >= 11 else 0
        features['price_change_20'] = (closes[-1] - closes[-21]) / closes[-21] if len(closes) >= 21 else 0
        
        # Volatility features
        features['volatility_10'] = np.std(closes[-10:]) if len(closes) >= 10 else 0
        features['volatility_20'] = np.std(closes[-20:]) if len(closes) >= 20 else 0
        features['price_range'] = (max(highs) - min(lows)) / closes[-1]  # Normalized range
        
        # Volume features
        features['volume_trend'] = (volumes[-1] - np.mean(volumes[-5:])) / np.mean(volumes[-5:]) if len(volumes) >= 5 and np.mean(volumes[-5:]) > 0 else 0
        features['volume_ratio'] = volumes[-1] / np.mean(volumes[-20:]) if len(volumes) >= 20 and np.mean(volumes[-20:]) > 0 else 1
        
        # Technical indicators
        try:
            # RSI
            rsi = self.analyzer.calculate_rsi(closes, period=14) if len(closes) >= 14 else 50
            features['rsi'] = rsi
            
            # MACD
            if len(closes) >= 35:  # Need enough data for MACD
                macd_line, signal_line, histogram = self.analyzer.calculate_macd(closes)
                features['macd_line'] = macd_line
                features['macd_signal'] = signal_line
                features['macd_histogram'] = histogram
            else:
                features['macd_line'] = 0
                features['macd_signal'] = 0
                features['macd_histogram'] = 0
                
            # ADX
            if len(closes) >= 20:
                adx, plus_di, minus_di = self.analyzer.calculate_adx(highs, lows, closes)
                features['adx'] = adx
                features['plus_di'] = plus_di
                features['minus_di'] = minus_di
            else:
                features['adx'] = 0
                features['plus_di'] = 0
                features['minus_di'] = 0
                
            # Bollinger Bands
            if len(closes) >= 20:
                bb = self.analyzer.calculate_bollinger_bands(closes)
                features['bb_width'] = bb.get('bandwidth', 0)
                features['bb_percent_b'] = bb.get('percent_b', 0.5)
                features['bb_upper'] = bb.get('upper_band', closes[-1])
                features['bb_lower'] = bb.get('lower_band', closes[-1])
                features['bb_middle'] = bb.get('middle_band', closes[-1])
                features['bb_squeeze'] = 1 if bb.get('squeeze', False) else 0
            else:
                features['bb_width'] = 0
                features['bb_percent_b'] = 0.5
                features['bb_upper'] = closes[-1]
                features['bb_lower'] = closes[-1]
                features['bb_middle'] = closes[-1]
                features['bb_squeeze'] = 0
                
            # EMA differences
            if len(closes) >= 20:
                ema9 = self.analyzer.calculate_ema(closes, 9)
                ema21 = self.analyzer.calculate_ema(closes, 21)
                ema50 = self.analyzer.calculate_ema(closes, 50)
                
                if ema9 and ema21 and ema50:
                    features['ema9_21_diff'] = (ema9[-1] - ema21[-1]) / closes[-1]
                    features['ema21_50_diff'] = (ema21[-1] - ema50[-1]) / closes[-1]
                    features['price_ema21_diff'] = (closes[-1] - ema21[-1]) / closes[-1]
                    
            # Support/Resistance
            sr = self.analyzer.find_support_resistance(highs, lows, closes)
            nearest_support = sr.get('nearest_support', closes[-1] * 0.95)
            nearest_resistance = sr.get('nearest_resistance', closes[-1] * 1.05)
            
            features['support_distance'] = (closes[-1] - nearest_support) / closes[-1]
            features['resistance_distance'] = (nearest_resistance - closes[-1]) / closes[-1]
            features['support_resistance_width'] = (nearest_resistance - nearest_support) / closes[-1]
            
            # Candlestick patterns (simplified)
            features['candle_body'] = abs(closes[-1] - opens[-1]) / closes[-1]
            features['upper_wick'] = (highs[-1] - max(closes[-1], opens[-1])) / closes[-1]
            features['lower_wick'] = (min(closes[-1], opens[-1]) - lows[-1]) / closes[-1]
            
            # Detect simple patterns
            features['is_hammer'] = 1 if (features['lower_wick'] > features['candle_body'] * 2 and 
                                        features['upper_wick'] < features['candle_body'] * 0.5) else 0
            features['is_shooting_star'] = 1 if (features['upper_wick'] > features['candle_body'] * 2 and 
                                               features['lower_wick'] < features['candle_body'] * 0.5) else 0
            features['is_engulfing'] = 1 if (len(closes) >= 2 and 
                                           abs(closes[-1] - opens[-1]) > abs(closes[-2] - opens[-2]) and
                                           (closes[-1] > opens[-1]) != (closes[-2] > opens[-2])) else 0
            
        except Exception as e:
            # If any indicator fails, use default values
            print(f"Warning: Could not calculate all indicators: {e}")
            # Fill missing features with defaults
            for key in ['rsi', 'macd_line', 'macd_signal', 'macd_histogram', 'adx', 
                       'plus_di', 'minus_di', 'bb_width', 'bb_percent_b']:
                if key not in features:
                    features[key] = 0
                    
        # Time-based features
        entry_time = window_df['timestamp'].iloc[-1]
        features['hour_of_day'] = entry_time.hour
        features['day_of_week'] = entry_time.dayofweek
        features['month'] = entry_time.month
        
        # Volatility regime (simple classification)
        if features['volatility_20'] > np.percentile([features['volatility_20']], 75) if features['volatility_20'] > 0 else False:
            features['volatility_regime'] = 'high'
        elif features['volatility_20'] < np.percentile([features['volatility_20']], 25) if features['volatility_20'] > 0 else False:
            features['volatility_regime'] = 'low'
        else:
            features['volatility_regime'] = 'medium'
            
        return features

if __name__ == "__main__":
    collector = DataCollector()
    try:
        data = collector.collect_labeled_data(target_samples=5000)
        print(f"Collected {len(data)} samples")
        print(data.head())
        data.to_csv("ml_training_data.csv", index=False)
        print("Data saved to ml_training_data.csv")
    except Exception as e:
        print(f"Error: {e}")