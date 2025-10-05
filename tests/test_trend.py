import os
import sys

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from trend_analysis import TrendAnalyzer
import numpy as np

analyzer = TrendAnalyzer()

# Test with multiple symbols to see different trend scenarios
test_symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT']

print("\n" + "="*80)
print("PROFESSIONAL TREND ANALYSIS - Using Industry-Standard Indicators")
print("="*80)
print("\nIndicators Used:")
print("  • ADX (Average Directional Index) - Trend Strength (0-100)")
print("  • EMA Crossover (9/21) - Fast trend detection")
print("  • MACD - Momentum confirmation")
print("  • +DI/-DI - Directional movement")
print("\nADX Interpretation:")
print("  • 0-25:  Weak/Ranging (avoid trend trading)")
print("  • 25-50: Strong Trend (good for trading)")
print("  • 50-75: Very Strong Trend (excellent)")
print("  • 75+:   Extremely Strong (may be overextended)")
print("="*80)

for symbol in test_symbols:
    print(f'\n{"="*80}')
    print(f'📊 {symbol} Analysis')
    print(f'{"="*80}')
    
    opens, highs, lows, closes = analyzer.fetch_historical_prices(symbol, 100)
    
    if not closes or len(closes) < 50:
        print(f'❌ Insufficient data for {symbol}\n')
        continue
    
    # Convert to float
    closes_float = [float(c) for c in closes]
    highs_float = [float(h) for h in highs]
    lows_float = [float(l) for l in lows]
    
    # Basic price info
    current_price = closes_float[-1]
    price_change = ((closes_float[-1] - closes_float[-14]) / closes_float[-14] * 100) if len(closes_float) >= 14 else 0
    
    print(f'\n💰 Price Information:')
    print(f'   Current Price: ${current_price:,.2f}')
    print(f'   14-Period Change: {price_change:+.2f}%')
    print(f'   Range (50 bars): ${min(closes_float):,.2f} - ${max(closes_float):,.2f}')
    
    # Calculate trend direction
    trend_direction, direction_details = analyzer.calculate_trend_direction(closes_float, highs_float, lows_float)
    
    # Calculate trend strength
    trend_strength, strength_details = analyzer.calculate_trend_strength(highs_float, lows_float, closes_float)
    
    # Display results
    direction_emoji = {
        'uptrend': '📈 UPTREND',
        'downtrend': '📉 DOWNTREND',
        'sideways': '➡️  SIDEWAYS'
    }
    
    print(f'\n🎯 TREND ANALYSIS:')
    print(f'   Direction: {direction_emoji.get(trend_direction, "❓")}')
    print(f'   Strength:  {trend_strength}/100 ({strength_details.get("strength_category", "N/A")})')
    
    # ADX Details
    print(f'\n📊 ADX (Trend Strength Indicator):')
    print(f'   ADX:  {strength_details.get("adx", 0):.2f}')
    print(f'   +DI:  {strength_details.get("plus_di", 0):.2f} (Bullish pressure)')
    print(f'   -DI:  {strength_details.get("minus_di", 0):.2f} (Bearish pressure)')
    
    # EMA Analysis
    print(f'\n📈 EMA Crossover (9/21):')
    print(f'   EMA 9:  ${direction_details.get("ema_9", 0):,.2f}')
    print(f'   EMA 21: ${direction_details.get("ema_21", 0):,.2f}')
    print(f'   Signal: {direction_details.get("ema_signal", "N/A").upper()}')
    
    # MACD Analysis
    print(f'\n📉 MACD (Momentum):')
    print(f'   MACD Line:   {direction_details.get("macd", 0):.4f}')
    print(f'   Signal Line: {direction_details.get("macd_signal", 0):.4f}')
    print(f'   Histogram:   {direction_details.get("macd_histogram", 0):.4f}')
    print(f'   Trend:       {direction_details.get("macd_trend", "N/A").upper()}')
    
    # Voting Summary
    print(f'\n🗳️  Signal Consensus:')
    bullish = direction_details.get("bullish_votes", 0)
    bearish = direction_details.get("bearish_votes", 0)
    total = bullish + bearish
    if total > 0:
        print(f'   Bullish: {bullish:.1f} votes ({bullish/total*100:.0f}%)')
        print(f'   Bearish: {bearish:.1f} votes ({bearish/total*100:.0f}%)')
    
    # Trading Recommendation
    print(f'\n💡 TRADING SIGNAL:')
    adx_val = strength_details.get("adx", 0)
    
    if trend_direction == "uptrend" and adx_val >= 25:
        print(f'   ✅ STRONG BUY - Confirmed uptrend with good strength')
    elif trend_direction == "uptrend" and adx_val < 25:
        print(f'   ⚠️  WEAK BUY - Uptrend but low strength (risky)')
    elif trend_direction == "downtrend" and adx_val >= 25:
        print(f'   ❌ STRONG SELL - Confirmed downtrend with good strength')
    elif trend_direction == "downtrend" and adx_val < 25:
        print(f'   ⚠️  WEAK SELL - Downtrend but low strength (risky)')
    else:
        if adx_val < 25:
            print(f'   ⏸️  NO TRADE - Ranging market, wait for trend')
        else:
            print(f'   ⚠️  CAUTION - Mixed signals, wait for clarity')

print(f'\n{"="*80}')
print(f'\n{"="*80}')
print('✅ Analysis Complete!')
print("="*80)
print('\n📚 Note: These are technical indicators only. Always:')
print('   • Use proper risk management')
print('   • Consider multiple timeframes')
print('   • Check fundamental factors')
print('   • Never risk more than you can afford to lose')
print("="*80 + "\n")
