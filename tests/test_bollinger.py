#!/usr/bin/env python3
"""
Test script for Bollinger Bands Analysis
Tests all Bollinger Bands functionality including:
- Band calculation
- Squeeze detection
- Band walk identification
- Mean reversion signals
- Breakout detection
"""

import os
import sys

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from trend_analysis import TrendAnalyzer
from datetime import datetime

def print_section(title: str):
    """Print a formatted section header"""
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}\n")

def test_basic_bollinger_bands(analyzer: TrendAnalyzer, symbol: str):
    """Test basic Bollinger Bands calculation"""
    print_section(f"TEST 1: Basic Bollinger Bands Analysis - {symbol}")
    
    # Fetch historical prices for testing
    opens, highs, lows, closes, volumes = analyzer.fetch_historical_prices(symbol, limit=150, interval="15m")
    
    if not closes or len(closes) < 20:
        print(f"❌ Insufficient price data for {symbol}")
        return False
    
    # Convert to float for calculation
    closes_float = [float(c) for c in closes]
    
    # Calculate Bollinger Bands
    bb_analysis = analyzer.calculate_bollinger_bands(closes_float, period=20, multiplier=2.0)
    
    if not bb_analysis:
        print(f"❌ Error calculating Bollinger Bands for {symbol}")
        return False
    
    print(f"✅ Bollinger Bands calculation successful!")
    print(f"   Symbol: {symbol}")
    print(f"   Current Price: ${closes_float[-1]:,.2f}")
    print(f"   Middle Band (SMA): ${bb_analysis['middle_band']:,.2f}")
    print(f"   Upper Band: ${bb_analysis['upper_band']:,.2f}")
    print(f"   Lower Band: ${bb_analysis['lower_band']:,.2f}")
    print(f"   Bandwidth: {bb_analysis['bandwidth']:.2f}%")
    print(f"   %B: {bb_analysis['percent_b']:.4f}")
    
    # Interpret %B value
    if bb_analysis['percent_b'] > 1:
        print(f"   Position: Price is ABOVE upper band (overbought)")
    elif bb_analysis['percent_b'] < 0:
        print(f"   Position: Price is BELOW lower band (oversold)")
    elif bb_analysis['percent_b'] > 0.8:
        print(f"   Position: Price is near upper band")
    elif bb_analysis['percent_b'] < 0.2:
        print(f"   Position: Price is near lower band")
    else:
        print(f"   Position: Price is within bands")
    
    return True

def test_bollinger_squeeze(analyzer: TrendAnalyzer, symbol: str):
    """Test Bollinger Bands squeeze detection"""
    print_section(f"TEST 2: Bollinger Bands Squeeze Detection - {symbol}")
    
    # Fetch more historical prices for squeeze analysis
    opens, highs, lows, closes, volumes = analyzer.fetch_historical_prices(symbol, limit=200, interval="15m")
    
    if not closes or len(closes) < 100:
        print(f"❌ Insufficient price data for squeeze analysis for {symbol}")
        return False
    
    # Convert to float for calculation
    closes_float = [float(c) for c in closes]
    
    # Calculate Bollinger Bands
    bb_analysis = analyzer.calculate_bollinger_bands(closes_float, period=20, multiplier=2.0)
    
    if not bb_analysis:
        print(f"❌ Error calculating Bollinger Bands for {symbol}")
        return False
    
    print(f"   Bandwidth: {bb_analysis['bandwidth']:.2f}%")
    print(f"   Squeeze Detected: {'✅ YES' if bb_analysis['squeeze'] else '❌ NO'}")
    
    if bb_analysis['squeeze']:
        print(f"   📣 SQUEEZE ALERT: Low volatility detected - potential breakout imminent!")
        print(f"   Recommendation: Watch for strong directional move after consolidation")
    else:
        print(f"   Normal volatility conditions")
    
    return True

def test_band_walk_detection(analyzer: TrendAnalyzer, symbol: str):
    """Test band walk detection"""
    print_section(f"TEST 3: Band Walk Detection - {symbol}")
    
    # Fetch historical prices
    opens, highs, lows, closes, volumes = analyzer.fetch_historical_prices(symbol, limit=100, interval="15m")
    
    if not closes or len(closes) < 20:
        print(f"❌ Insufficient price data for {symbol}")
        return False
    
    # Convert to float for calculation
    closes_float = [float(c) for c in closes]
    
    # Calculate Bollinger Bands
    bb_analysis = analyzer.calculate_bollinger_bands(closes_float, period=20, multiplier=2.0)
    
    if not bb_analysis:
        print(f"❌ Error calculating Bollinger Bands for {symbol}")
        return False
    
    print(f"   Current Price: ${closes_float[-1]:,.2f}")
    print(f"   Upper Band: ${bb_analysis['upper_band']:,.2f}")
    print(f"   Lower Band: ${bb_analysis['lower_band']:,.2f}")
    
    if bb_analysis['band_walk'] == "upper":
        print(f"   🚀 BAND WALK: Price is riding above upper band (strong uptrend)")
        print(f"   Recommendation: Don't fade the trend - look for long entries on dips")
    elif bb_analysis['band_walk'] == "lower":
        print(f"   📉 BAND WALK: Price is riding below lower band (strong downtrend)")
        print(f"   Recommendation: Don't fade the trend - look for short entries on rallies")
    else:
        print(f"   Price is within bands (no strong trend)")
    
    return True

def test_trading_signals(analyzer: TrendAnalyzer, symbol: str):
    """Test Bollinger Bands trading signals"""
    print_section(f"TEST 4: Trading Signals - {symbol}")
    
    # Fetch historical prices
    opens, highs, lows, closes, volumes = analyzer.fetch_historical_prices(symbol, limit=100, interval="15m")
    
    if not closes or len(closes) < 20:
        print(f"❌ Insufficient price data for {symbol}")
        return False
    
    # Convert to float for calculation
    closes_float = [float(c) for c in closes]
    
    # Calculate Bollinger Bands
    bb_analysis = analyzer.calculate_bollinger_bands(closes_float, period=20, multiplier=2.0)
    
    if not bb_analysis:
        print(f"❌ Error calculating Bollinger Bands for {symbol}")
        return False
    
    print(f"   %B Value: {bb_analysis['percent_b']:.4f}")
    
    # Mean reversion signals
    if bb_analysis['mean_reversion_signal'] == "buy":
        print(f"   🟢 MEAN REVERSION: Buy signal near lower band (price: ${closes_float[-1]:,.2f})")
        print(f"   Recommendation: Consider long entry with tight stop below lower band")
    elif bb_analysis['mean_reversion_signal'] == "sell":
        print(f"   🔴 MEAN REVERSION: Sell signal near upper band (price: ${closes_float[-1]:,.2f})")
        print(f"   Recommendation: Consider short entry with tight stop above upper band")
    else:
        print(f"   No mean reversion signal")
    
    # Breakout signals
    if bb_analysis['breakout_signal'] == "strong_up":
        print(f"   🚀 BREAKOUT: Strong upward breakout above upper band (price: ${closes_float[-1]:,.2f})")
        print(f"   Recommendation: Consider aggressive long entry with volume confirmation")
    elif bb_analysis['breakout_signal'] == "strong_down":
        print(f"   📉 BREAKOUT: Strong downward breakout below lower band (price: ${closes_float[-1]:,.2f})")
        print(f"   Recommendation: Consider aggressive short entry with volume confirmation")
    else:
        print(f"   No breakout signal")
    
    return True

def run_all_tests():
    """Run all Bollinger Bands tests"""
    print_section("BOLLINGER BANDS TEST SUITE")
    print("Testing Bollinger Bands implementation...")
    
    # Initialize analyzer
    analyzer = TrendAnalyzer()
    
    # Test with a few symbols
    test_symbols = ["BTCUSDT", "ETHUSDT"]
    
    for symbol in test_symbols:
        try:
            print(f"\n{'#'*80}")
            print(f"# TESTING {symbol}")
            print(f"{'#'*80}\n")
            
            # Run all tests for this symbol
            test_basic_bollinger_bands(analyzer, symbol)
            test_bollinger_squeeze(analyzer, symbol)
            test_band_walk_detection(analyzer, symbol)
            test_trading_signals(analyzer, symbol)
            
        except Exception as e:
            print(f"❌ Error testing {symbol}: {e}")
            import traceback
            traceback.print_exc()
    
    print_section("TEST SUITE COMPLETED")
    print("Bollinger Bands implementation test completed!")

if __name__ == "__main__":
    run_all_tests()