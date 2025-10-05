#!/usr/bin/env python3
"""
Test script for Fibonacci Retracement Analysis
Tests all Fibonacci functionality including:
- Swing detection
- Fibonacci level calculation
- Confluence detection
- Multi-timeframe analysis
- Trade setup generation
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

def test_basic_fibonacci(analyzer: TrendAnalyzer, symbol: str):
    """Test basic Fibonacci analysis"""
    print_section(f"TEST 1: Basic Fibonacci Analysis - {symbol}")
    
    fib_analysis = analyzer.analyze_fibonacci(symbol, lookback=100, interval="15m", min_move_pct=2.0)
    
    if not fib_analysis or "error" in fib_analysis:
        print(f"❌ Error: {fib_analysis.get('error', 'Unknown error')}")
        return False
    
    print(f"✅ Fibonacci analysis successful!")
    print(f"   Symbol: {fib_analysis['symbol']}")
    print(f"   Current Price: ${fib_analysis['current_price']:,.2f}")
    print(f"   Swing Type: {fib_analysis['swing']['swing_type']}")
    print(f"   Swing Range: {fib_analysis['swing']['move_pct']:.2f}%")
    print(f"   Swing High: ${fib_analysis['swing']['swing_high']:,.2f}")
    print(f"   Swing Low: ${fib_analysis['swing']['swing_low']:,.2f}")
    
    # Check retracement levels
    retracements = fib_analysis['fibonacci_levels']['retracements']
    print(f"\n📉 Retracement Levels:")
    for level, price in sorted(retracements.items(), key=lambda x: float(x[0])):
        print(f"   {level}%: ${price:,.2f}")
    
    # Check extension levels
    extensions = fib_analysis['fibonacci_levels']['extensions']
    print(f"\n🎯 Extension Levels:")
    for level, price in sorted(extensions.items(), key=lambda x: float(x[0])):
        print(f"   {level}%: ${price:,.2f}")
    
    return True

def test_nearest_fib_level(analyzer: TrendAnalyzer, symbol: str):
    """Test nearest Fibonacci level detection"""
    print_section(f"TEST 2: Nearest Fibonacci Level - {symbol}")
    
    fib_analysis = analyzer.analyze_fibonacci(symbol, interval="15m", min_move_pct=2.0)
    
    if not fib_analysis or "error" in fib_analysis:
        print(f"❌ Error: No Fibonacci analysis available")
        return False
    
    nearest_fib = fib_analysis.get("nearest_fib_level")
    
    if nearest_fib:
        print(f"✅ Found nearest Fibonacci level!")
        print(f"   Level: {nearest_fib['level_name']}")
        print(f"   Price: ${nearest_fib['level_price']:,.2f}")
        print(f"   Distance: {nearest_fib['distance_pct']:.2f}%")
        print(f"   Is Key Level: {'Yes ⭐' if nearest_fib['is_key_level'] else 'No'}")
    else:
        print(f"ℹ️ No Fibonacci level within tolerance (0.5%)")
    
    return True

def test_confluence_detection(analyzer: TrendAnalyzer, symbol: str):
    """Test Fibonacci confluence detection"""
    print_section(f"TEST 3: Confluence Detection - {symbol}")
    
    fib_analysis = analyzer.analyze_fibonacci(symbol, interval="15m", min_move_pct=2.0)
    
    if not fib_analysis or "error" in fib_analysis:
        print(f"❌ Error: No Fibonacci analysis available")
        return False
    
    confluences = fib_analysis.get("confluences", [])
    
    if confluences:
        print(f"✅ Found {len(confluences)} confluence zone(s)!")
        for i, conf in enumerate(confluences[:5], 1):  # Show top 5
            print(f"\n   Confluence Zone {i}:")
            print(f"   Price: ${conf['price']:,.2f}")
            print(f"   Distance: {conf['distance_pct']:.2f}%")
            print(f"   Confluence Count: {conf['confluence_count']}")
            print(f"   Factors: {', '.join(conf['factors'])}")
            print(f"   Key Level: {'Yes ⭐' if conf['is_key_level'] else 'No'}")
    else:
        print(f"ℹ️ No confluence zones detected")
    
    return True

def test_position_analysis(analyzer: TrendAnalyzer, symbol: str):
    """Test position analysis relative to Fibonacci levels"""
    print_section(f"TEST 4: Position Analysis - {symbol}")
    
    fib_analysis = analyzer.analyze_fibonacci(symbol, interval="15m", min_move_pct=2.0)
    
    if not fib_analysis or "error" in fib_analysis:
        print(f"❌ Error: No Fibonacci analysis available")
        return False
    
    position_analysis = fib_analysis.get("position_analysis", {})
    
    print(f"✅ Position Analysis:")
    print(f"   In Entry Zone: {'Yes 🎯' if position_analysis.get('in_entry_zone') else 'No'}")
    print(f"   At Key Level: {'Yes ⭐' if position_analysis.get('at_key_level') else 'No'}")
    print(f"   Near Invalidation: {'Yes 🚫' if position_analysis.get('near_invalidation') else 'No'}")
    print(f"   At Profit Target: {'Yes 💰' if position_analysis.get('at_profit_target') else 'No'}")
    print(f"\n   Recommendation: {position_analysis.get('recommendation', 'N/A')}")
    
    return True

def test_multi_timeframe_fibonacci(analyzer: TrendAnalyzer, symbol: str):
    """Test multi-timeframe Fibonacci analysis"""
    print_section(f"TEST 5: Multi-Timeframe Fibonacci - {symbol}")
    
    mtf_fib = analyzer.multi_timeframe_fibonacci(symbol)
    
    if not mtf_fib:
        print(f"❌ Error: Multi-timeframe analysis failed")
        return False
    
    print(f"✅ Multi-Timeframe Analysis:")
    
    for timeframe in ["15m", "1h", "4h"]:
        if timeframe in mtf_fib:
            data = mtf_fib[timeframe]
            if "error" not in data:
                print(f"\n   {timeframe.upper()} Timeframe:")
                print(f"   Swing Type: {data.get('swing_type', 'N/A')}")
                print(f"   Move %: {data.get('move_pct', 0):.2f}%")
                print(f"   In Entry Zone: {'Yes 🎯' if data.get('in_entry_zone') else 'No'}")
                print(f"   Confluences: {data.get('confluences', 0)}")
                
                key_levels = data.get('key_levels', {})
                if key_levels:
                    print(f"   Key Levels:")
                    print(f"      38.2%: ${key_levels.get('fib_382', 0):,.2f}")
                    print(f"      50.0%: ${key_levels.get('fib_50', 0):,.2f}")
                    print(f"      61.8%: ${key_levels.get('fib_618', 0):,.2f}")
            else:
                print(f"\n   {timeframe.upper()}: {data.get('error', 'Unknown error')}")
    
    # Check alignment
    alignment = mtf_fib.get("alignment", {})
    print(f"\n   Timeframe Alignment: {'Yes ✅' if alignment.get('is_aligned') else 'No ❌'}")
    print(f"   Consensus: {alignment.get('consensus_swing', 'N/A')}")
    
    return True

def test_trade_setup(analyzer: TrendAnalyzer, symbol: str):
    """Test complete trade setup generation"""
    print_section(f"TEST 6: Trade Setup Generation - {symbol}")
    
    trade_setup = analyzer.get_fibonacci_trade_setup(symbol, interval="15m")
    
    if not trade_setup:
        print(f"❌ Error: Could not generate trade setup")
        return False
    
    print(f"✅ Trade Setup Generated!")
    print(f"\n   Symbol: {trade_setup['symbol']}")
    print(f"   Direction: {trade_setup['direction']}")
    print(f"   Should Enter: {'Yes 🎯' if trade_setup['should_enter'] else 'No ⏸️'}")
    print(f"   Entry Price: ${trade_setup['entry_price']:,.2f}")
    print(f"   Stop Loss: ${trade_setup['stop_loss']:,.2f}")
    print(f"   Risk Amount: ${trade_setup['risk_amount']:,.2f} ({trade_setup['risk_percent']:.2f}%)")
    
    print(f"\n   Take Profit Levels:")
    for i in range(1, 4):
        tp = trade_setup[f'take_profit_{i}']
        print(f"      TP{i} ({tp['size']}): ${tp['price']:,.2f} (R:R {tp['rr_ratio']}:1)")
    
    print(f"\n   Confluences: {trade_setup['confluences']}")
    print(f"   Analysis: {trade_setup['position_analysis']}")
    
    return True

def test_formatted_alert(analyzer: TrendAnalyzer, symbol: str):
    """Test formatted alert generation"""
    print_section(f"TEST 7: Formatted Alert - {symbol}")
    
    fib_analysis = analyzer.analyze_fibonacci(symbol, interval="15m", min_move_pct=2.0)
    
    if not fib_analysis or "error" in fib_analysis:
        print(f"❌ Error: No Fibonacci analysis available")
        return False
    
    alert_text = analyzer.format_fibonacci_alert(fib_analysis)
    print(alert_text)
    
    return True

def test_integration_with_advanced_analysis(analyzer: TrendAnalyzer, symbol: str):
    """Test Fibonacci integration with advanced trend analysis"""
    print_section(f"TEST 8: Integration with Advanced Analysis - {symbol}")
    
    analysis = analyzer.analyze_trend_advanced(symbol)
    
    if not analysis:
        print(f"❌ Error: Advanced analysis failed")
        return False
    
    print(f"✅ Advanced Analysis with Fibonacci Integration:")
    print(f"\n   Symbol: {analysis['symbol']}")
    print(f"   Current Price: ${analysis['current_price']:,.2f}")
    print(f"   Trend Direction: {analysis['trend_direction']}")
    print(f"   Trend Strength (ADX): {analysis['adx']:.1f}")
    print(f"   Signal Strength: {analysis['signal_strength']}")
    print(f"   Confidence Score: {analysis['confidence_score']}/100")
    
    # Check if Fibonacci is available
    if analysis.get('fibonacci_available'):
        print(f"\n   ✅ Fibonacci Analysis Integrated!")
        fib = analysis.get('fibonacci')
        if fib:
            print(f"   Swing Type: {fib['swing']['swing_type']}")
            print(f"   In Entry Zone: {fib['position_analysis'].get('in_entry_zone', False)}")
            print(f"   Confluences: {len(fib.get('confluences', []))}")
    else:
        print(f"\n   ℹ️ Fibonacci analysis not available for this setup")
    
    print(f"\n   Reasons for Signal:")
    for reason in analysis['reasons']:
        print(f"      ✓ {reason}")
    
    if analysis['warnings']:
        print(f"\n   Warnings:")
        for warning in analysis['warnings']:
            print(f"      ⚠️ {warning}")
    
    print(f"\n   Action: {analysis['action']}")
    
    return True

def run_all_tests():
    """Run all Fibonacci tests"""
    print("\n" + "="*80)
    print("  FIBONACCI RETRACEMENT SYSTEM - COMPREHENSIVE TEST SUITE")
    print("="*80)
    
    # Initialize analyzer
    analyzer = TrendAnalyzer()
    
    # Test symbols (major crypto pairs)
    test_symbols = ["BTCUSDT", "ETHUSDT"]
    
    results = {}
    
    for symbol in test_symbols:
        print(f"\n\n{'#'*80}")
        print(f"  TESTING SYMBOL: {symbol}")
        print(f"{'#'*80}")
        
        symbol_results = {}
        
        # Run all tests
        symbol_results['basic'] = test_basic_fibonacci(analyzer, symbol)
        symbol_results['nearest_level'] = test_nearest_fib_level(analyzer, symbol)
        symbol_results['confluence'] = test_confluence_detection(analyzer, symbol)
        symbol_results['position'] = test_position_analysis(analyzer, symbol)
        symbol_results['multi_timeframe'] = test_multi_timeframe_fibonacci(analyzer, symbol)
        symbol_results['trade_setup'] = test_trade_setup(analyzer, symbol)
        symbol_results['formatted_alert'] = test_formatted_alert(analyzer, symbol)
        symbol_results['integration'] = test_integration_with_advanced_analysis(analyzer, symbol)
        
        results[symbol] = symbol_results
    
    # Print summary
    print_section("TEST SUMMARY")
    
    for symbol, symbol_results in results.items():
        print(f"\n{symbol}:")
        passed = sum(1 for result in symbol_results.values() if result)
        total = len(symbol_results)
        print(f"   Passed: {passed}/{total}")
        
        for test_name, result in symbol_results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"      {test_name}: {status}")
    
    # Overall summary
    all_passed = all(result for symbol_results in results.values() for result in symbol_results.values())
    
    print(f"\n{'='*80}")
    if all_passed:
        print("  🎉 ALL TESTS PASSED! Fibonacci system is working correctly.")
    else:
        print("  ⚠️ SOME TESTS FAILED. Please review the output above.")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    try:
        run_all_tests()
    except KeyboardInterrupt:
        print("\n\n⚠️ Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error running tests: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
