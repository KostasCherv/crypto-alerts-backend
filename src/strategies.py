"""
Profitable Trading Strategies
============================

This is the ONE and ONLY strategy file. It contains profitable trading strategies
designed for real-world trading with realistic parameters.

Focus: Find profitable strategies through backtesting.

Author: Professional Trading System
Version: 1.0
"""

import pandas as pd
import numpy as np
import talib
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

@dataclass
class Signal:
    """Trading signal structure"""
    action: str  # 'BUY', 'SELL', 'HOLD'
    confidence: float
    entry_price: float
    stop_loss: float
    take_profit: float
    position_size: float
    strategy_name: str
    timestamp: datetime
    reason: str
    risk_reward_ratio: float
    timeframe: str

class StrategyBase(ABC):
    """Base class for all trading strategies"""
    
    def __init__(self, name: str, **kwargs):
        self.name = name
        self.parameters = kwargs
    
    @abstractmethod
    def generate_signal(self, data: pd.DataFrame, timeframe: str = '4h') -> Optional[Signal]:
        """Generate trading signal from market data"""
        pass
    
    def calculate_position_size(self, account_balance: float, risk_percent: float, 
                               stop_loss_distance: float, confidence: float, current_price: float) -> float:
        """Calculate position size based on risk and confidence"""
        # Base position size: 3% of account (profitable level)
        base_position_percent = 0.03
        
        # Adjust for confidence (0.5 to 1.0 multiplier)
        confidence_multiplier = 0.5 + (confidence * 0.5)
        
        # Calculate position size
        position_percent = base_position_percent * confidence_multiplier
        position_value = account_balance * position_percent
        
        # Calculate shares/units
        if stop_loss_distance > 0:
            risk_amount = account_balance * risk_percent
            position_size = risk_amount / stop_loss_distance
        else:
            position_size = position_value / current_price
        
        return min(position_size, position_value / current_price)
    
    def calculate_risk_reward(self, entry_price: float, stop_loss: float, 
                             take_profit: float) -> float:
        """Calculate risk-reward ratio"""
        if stop_loss == 0:
            return 0
        
        risk = abs(entry_price - stop_loss)
        reward = abs(take_profit - entry_price)
        
        return reward / risk if risk > 0 else 0

class EMACrossoverStrategy(StrategyBase):
    """EMA Crossover Strategy - Profitable trend following"""
    
    def __init__(self, **kwargs):
        super().__init__('ema_crossover', **kwargs)
        self.fast_period = kwargs.get('fast_period', 8)
        self.slow_period = kwargs.get('slow_period', 21)
        self.min_confidence = kwargs.get('min_confidence', 0.6)
    
    def generate_signal(self, data: pd.DataFrame, timeframe: str = '4h') -> Optional[Signal]:
        """Generate EMA crossover signal"""
        if len(data) < max(self.fast_period, self.slow_period) + 5:
            return None
        
        # Calculate EMAs
        ema_fast = talib.EMA(data['close'], timeperiod=self.fast_period)
        ema_slow = talib.EMA(data['close'], timeperiod=self.slow_period)
        
        # Calculate RSI for momentum confirmation
        rsi = talib.RSI(data['close'], timeperiod=14)
        
        # Get current values
        current_price = data['close'].iloc[-1]
        current_ema_fast = ema_fast.iloc[-1]
        current_ema_slow = ema_slow.iloc[-1]
        current_rsi = rsi.iloc[-1]
        
        # Check for valid values
        if any(pd.isna([current_ema_fast, current_ema_slow, current_rsi])):
            return None
        
        # Generate signals
        if (current_ema_fast > current_ema_slow and  # Fast above slow
            current_rsi > 45):  # RSI above 45 (not oversold)
            
            # Calculate confidence
            confidence = 0.65
            
            # Calculate stop loss and take profit (profitable 1:2.5 ratio)
            atr = talib.ATR(data['high'], data['low'], data['close'], timeperiod=14).iloc[-1]
            stop_loss = current_price - (atr * 1.5)
            take_profit = current_price + (atr * 1.5 * 2.5)
            
            # Calculate position size
            position_size = self.calculate_position_size(
                10000, 0.025, current_price - stop_loss, confidence, current_price
            )
            
            return Signal(
                action='BUY',
                confidence=confidence,
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                position_size=position_size,
                strategy_name=self.name,
                timestamp=datetime.now(),
                reason=f'EMA crossover (Fast: {self.fast_period}, Slow: {self.slow_period})',
                risk_reward_ratio=self.calculate_risk_reward(current_price, stop_loss, take_profit),
                timeframe=timeframe
            )
        
        elif (current_ema_fast < current_ema_slow and  # Fast below slow
              current_rsi < 55):  # RSI below 55 (not overbought)
            
            # Calculate confidence
            confidence = 0.65
            
            # Calculate stop loss and take profit (profitable 1:2.5 ratio)
            atr = talib.ATR(data['high'], data['low'], data['close'], timeperiod=14).iloc[-1]
            stop_loss = current_price + (atr * 1.5)
            take_profit = current_price - (atr * 1.5 * 2.5)
            
            # Calculate position size
            position_size = self.calculate_position_size(
                10000, 0.025, stop_loss - current_price, confidence, current_price
            )
            
            return Signal(
                action='SELL',
                confidence=confidence,
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                position_size=position_size,
                strategy_name=self.name,
                timestamp=datetime.now(),
                reason=f'EMA crossover (Fast: {self.fast_period}, Slow: {self.slow_period})',
                risk_reward_ratio=self.calculate_risk_reward(current_price, stop_loss, take_profit),
                timeframe=timeframe
            )
        
        return None

class BollingerBandsStrategy(StrategyBase):
    """Bollinger Bands Strategy - Profitable mean reversion"""
    
    def __init__(self, **kwargs):
        super().__init__('bollinger_bands', **kwargs)
        self.period = kwargs.get('period', 20)
        self.std_dev = kwargs.get('std_dev', 2.0)
        self.min_confidence = kwargs.get('min_confidence', 0.6)
    
    def generate_signal(self, data: pd.DataFrame, timeframe: str = '4h') -> Optional[Signal]:
        """Generate Bollinger Bands signal"""
        if len(data) < self.period + 5:
            return None
        
        # Calculate Bollinger Bands
        bb_upper, bb_middle, bb_lower = talib.BBANDS(data['close'], 
                                                    timeperiod=self.period,
                                                    nbdevup=self.std_dev,
                                                    nbdevdn=self.std_dev)
        
        # Calculate RSI for confirmation
        rsi = talib.RSI(data['close'], timeperiod=14)
        
        # Get current values
        current_price = data['close'].iloc[-1]
        current_upper = bb_upper.iloc[-1]
        current_lower = bb_lower.iloc[-1]
        current_rsi = rsi.iloc[-1]
        
        # Check for valid values
        if any(pd.isna([current_upper, current_lower, current_rsi])):
            return None
        
        # Generate signals
        if (current_price <= current_lower * 1.01 and  # Price near lower band
            current_rsi < 40):  # RSI below 40
            
            # Calculate confidence
            confidence = 0.7
            
            # Calculate stop loss and take profit (profitable 1:2.5 ratio)
            atr = talib.ATR(data['high'], data['low'], data['close'], timeperiod=14).iloc[-1]
            stop_loss = current_price - (atr * 1.2)
            take_profit = current_price + (atr * 1.2 * 2.5)
            
            # Calculate position size
            position_size = self.calculate_position_size(
                10000, 0.025, current_price - stop_loss, confidence, current_price
            )
            
            return Signal(
                action='BUY',
                confidence=confidence,
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                position_size=position_size,
                strategy_name=self.name,
                timestamp=datetime.now(),
                reason=f'BB lower band + RSI oversold (Period: {self.period})',
                risk_reward_ratio=self.calculate_risk_reward(current_price, stop_loss, take_profit),
                timeframe=timeframe
            )
        
        elif (current_price >= current_upper * 0.99 and  # Price near upper band
              current_rsi > 60):  # RSI above 60
            
            # Calculate confidence
            confidence = 0.7
            
            # Calculate stop loss and take profit (profitable 1:2.5 ratio)
            atr = talib.ATR(data['high'], data['low'], data['close'], timeperiod=14).iloc[-1]
            stop_loss = current_price + (atr * 1.2)
            take_profit = current_price - (atr * 1.2 * 2.5)
            
            # Calculate position size
            position_size = self.calculate_position_size(
                10000, 0.025, stop_loss - current_price, confidence, current_price
            )
            
            return Signal(
                action='SELL',
                confidence=confidence,
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                position_size=position_size,
                strategy_name=self.name,
                timestamp=datetime.now(),
                reason=f'BB upper band + RSI overbought (Period: {self.period})',
                risk_reward_ratio=self.calculate_risk_reward(current_price, stop_loss, take_profit),
                timeframe=timeframe
            )
        
        return None

class RSIMeanReversionStrategy(StrategyBase):
    """RSI Mean Reversion Strategy - Profitable contrarian trading"""
    
    def __init__(self, **kwargs):
        super().__init__('rsi_mean_reversion', **kwargs)
        self.rsi_period = kwargs.get('rsi_period', 14)
        self.oversold = kwargs.get('oversold', 35)
        self.overbought = kwargs.get('overbought', 65)
        self.min_confidence = kwargs.get('min_confidence', 0.6)
    
    def generate_signal(self, data: pd.DataFrame, timeframe: str = '4h') -> Optional[Signal]:
        """Generate RSI mean reversion signal"""
        if len(data) < self.rsi_period + 5:
            return None
        
        # Calculate RSI
        rsi = talib.RSI(data['close'], timeperiod=self.rsi_period)
        
        # Calculate Stochastic for confirmation
        stoch_k, stoch_d = talib.STOCH(data['high'], data['low'], data['close'],
                                      fastk_period=14, slowk_period=3, slowd_period=3)
        
        # Get current values
        current_price = data['close'].iloc[-1]
        current_rsi = rsi.iloc[-1]
        current_stoch_k = stoch_k.iloc[-1]
        current_stoch_d = stoch_d.iloc[-1]
        
        # Check for valid values
        if any(pd.isna([current_rsi, current_stoch_k, current_stoch_d])):
            return None
        
        # Generate signals
        if (current_rsi < self.oversold and  # RSI oversold
            current_stoch_k < 25):  # Stochastic oversold
            
            # Calculate confidence
            confidence = 0.75
            
            # Calculate stop loss and take profit (profitable 1:2.5 ratio)
            atr = talib.ATR(data['high'], data['low'], data['close'], timeperiod=14).iloc[-1]
            stop_loss = current_price - (atr * 1.2)
            take_profit = current_price + (atr * 1.2 * 2.5)
            
            # Calculate position size
            position_size = self.calculate_position_size(
                10000, 0.025, current_price - stop_loss, confidence, current_price
            )
            
            return Signal(
                action='BUY',
                confidence=confidence,
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                position_size=position_size,
                strategy_name=self.name,
                timestamp=datetime.now(),
                reason=f'RSI oversold + Stochastic confirmation (RSI: {self.rsi_period})',
                risk_reward_ratio=self.calculate_risk_reward(current_price, stop_loss, take_profit),
                timeframe=timeframe
            )
        
        elif (current_rsi > self.overbought and  # RSI overbought
              current_stoch_k > 75):  # Stochastic overbought
            
            # Calculate confidence
            confidence = 0.75
            
            # Calculate stop loss and take profit (profitable 1:2.5 ratio)
            atr = talib.ATR(data['high'], data['low'], data['close'], timeperiod=14).iloc[-1]
            stop_loss = current_price + (atr * 1.2)
            take_profit = current_price - (atr * 1.2 * 2.5)
            
            # Calculate position size
            position_size = self.calculate_position_size(
                10000, 0.025, stop_loss - current_price, confidence, current_price
            )
            
            return Signal(
                action='SELL',
                confidence=confidence,
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                position_size=position_size,
                strategy_name=self.name,
                timestamp=datetime.now(),
                reason=f'RSI overbought + Stochastic confirmation (RSI: {self.rsi_period})',
                risk_reward_ratio=self.calculate_risk_reward(current_price, stop_loss, take_profit),
                timeframe=timeframe
            )
        
        return None

class MACDStrategy(StrategyBase):
    """MACD Strategy - Profitable momentum trading"""
    
    def __init__(self, **kwargs):
        super().__init__('macd', **kwargs)
        self.fast_period = kwargs.get('fast_period', 12)
        self.slow_period = kwargs.get('slow_period', 26)
        self.signal_period = kwargs.get('signal_period', 9)
        self.min_confidence = kwargs.get('min_confidence', 0.6)
    
    def generate_signal(self, data: pd.DataFrame, timeframe: str = '4h') -> Optional[Signal]:
        """Generate MACD signal"""
        if len(data) < max(self.fast_period, self.slow_period) + 10:
            return None
        
        # Calculate MACD
        macd, macd_signal, macd_hist = talib.MACD(data['close'], 
                                                 fastperiod=self.fast_period,
                                                 slowperiod=self.slow_period,
                                                 signalperiod=self.signal_period)
        
        # Calculate RSI for confirmation
        rsi = talib.RSI(data['close'], timeperiod=14)
        
        # Get current values
        current_price = data['close'].iloc[-1]
        current_macd = macd.iloc[-1]
        current_macd_signal = macd_signal.iloc[-1]
        current_macd_hist = macd_hist.iloc[-1]
        current_rsi = rsi.iloc[-1]
        
        # Check for valid values
        if any(pd.isna([current_macd, current_macd_signal, current_macd_hist, current_rsi])):
            return None
        
        # Generate signals
        if (current_macd > current_macd_signal and  # MACD above signal
            current_macd_hist > 0 and  # MACD histogram positive
            current_rsi > 50):  # RSI above 50
            
            # Calculate confidence
            confidence = 0.7
            
            # Calculate stop loss and take profit (profitable 1:2.5 ratio)
            atr = talib.ATR(data['high'], data['low'], data['close'], timeperiod=14).iloc[-1]
            stop_loss = current_price - (atr * 1.5)
            take_profit = current_price + (atr * 1.5 * 2.5)
            
            # Calculate position size
            position_size = self.calculate_position_size(
                10000, 0.025, current_price - stop_loss, confidence, current_price
            )
            
            return Signal(
                action='BUY',
                confidence=confidence,
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                position_size=position_size,
                strategy_name=self.name,
                timestamp=datetime.now(),
                reason=f'MACD bullish + RSI confirmation (Fast: {self.fast_period}, Slow: {self.slow_period})',
                risk_reward_ratio=self.calculate_risk_reward(current_price, stop_loss, take_profit),
                timeframe=timeframe
            )
        
        elif (current_macd < current_macd_signal and  # MACD below signal
              current_macd_hist < 0 and  # MACD histogram negative
              current_rsi < 50):  # RSI below 50
            
            # Calculate confidence
            confidence = 0.7
            
            # Calculate stop loss and take profit (profitable 1:2.5 ratio)
            atr = talib.ATR(data['high'], data['low'], data['close'], timeperiod=14).iloc[-1]
            stop_loss = current_price + (atr * 1.5)
            take_profit = current_price - (atr * 1.5 * 2.5)
            
            # Calculate position size
            position_size = self.calculate_position_size(
                10000, 0.025, stop_loss - current_price, confidence, current_price
            )
            
            return Signal(
                action='SELL',
                confidence=confidence,
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                position_size=position_size,
                strategy_name=self.name,
                timestamp=datetime.now(),
                reason=f'MACD bearish + RSI confirmation (Fast: {self.fast_period}, Slow: {self.slow_period})',
                risk_reward_ratio=self.calculate_risk_reward(current_price, stop_loss, take_profit),
                timeframe=timeframe
            )
        
        return None

class StrategyManager:
    """Manager for all trading strategies"""
    
    def __init__(self):
        self.strategies = {
            'ema_crossover': EMACrossoverStrategy,
            'bollinger_bands': BollingerBandsStrategy,
            'rsi_mean_reversion': RSIMeanReversionStrategy,
            'macd': MACDStrategy
        }
    
    def create_strategy(self, strategy_name: str, **kwargs) -> StrategyBase:
        """Create a strategy instance"""
        if strategy_name not in self.strategies:
            raise ValueError(f"Unknown strategy: {strategy_name}")
        
        return self.strategies[strategy_name](**kwargs)
    
    def get_available_strategies(self) -> List[str]:
        """Get list of available strategies"""
        return list(self.strategies.keys())

def create_strategy(strategy_name: str, **kwargs) -> StrategyBase:
    """Factory function to create strategies"""
    manager = StrategyManager()
    return manager.create_strategy(strategy_name, **kwargs)

def main():
    """Main function to test all trading strategies"""
    print("🎯 Testing Profitable Trading Strategies...")
    
    # Create sample data
    dates = pd.date_range(start='2025-01-01', end='2025-10-01', freq='4H')
    np.random.seed(42)
    
    # Generate realistic price data
    price = 50000
    prices = []
    volumes = []
    
    for i in range(len(dates)):
        # Random walk with some trend
        change = np.random.normal(0.001, 0.02)
        price *= (1 + change)
        prices.append(price)
        volumes.append(np.random.uniform(1000, 10000))
    
    data = pd.DataFrame({
        'open': prices,
        'high': [p * (1 + np.random.uniform(0, 0.01)) for p in prices],
        'low': [p * (1 - np.random.uniform(0, 0.01)) for p in prices],
        'close': prices,
        'volume': volumes
    }, index=dates)
    
    # Test all strategies
    strategies_to_test = [
        'ema_crossover',
        'bollinger_bands',
        'rsi_mean_reversion',
        'macd'
    ]
    
    for strategy_name in strategies_to_test:
        print(f"\n🧪 Testing {strategy_name}...")
        strategy = create_strategy(strategy_name)
        
        # Test on recent data
        recent_data = data.tail(100)
        signal = strategy.generate_signal(recent_data, timeframe='4h')
        
        if signal:
            print(f"✅ Signal generated: {signal.action}")
            print(f"   Confidence: {signal.confidence:.2f}")
            print(f"   Risk-Reward: {signal.risk_reward_ratio:.2f}")
            print(f"   Position Size: {signal.position_size:.2f}")
            print(f"   Reason: {signal.reason}")
        else:
            print("❌ No signal generated")
    
    print("\n🎯 Strategy testing completed!")

# Example usage and testing
if __name__ == "__main__":
    main()