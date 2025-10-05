import os
import sys

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from trend_analysis import TrendAnalyzer
import json

def print_section(title):
    print(f"\n{'='*80}")
    print(f"{title}")
    print(f"{'='*80}")

def print_subsection(title):
    print(f"\n{'-'*80}")
    print(f"{title}")
    print(f"{'-'*80}")

analyzer = TrendAnalyzer()

# Test symbols
test_symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']

print_section("🚀 ADVANCED PROFITABILITY ANALYSIS")
print("\n📊 New Features:")
print("  ✅ Multi-timeframe confirmation (15m, 1h, 4h)")
print("  ✅ RSI overbought/oversold detection")
print("  ✅ Volume trend analysis")
print("  ✅ Support/Resistance levels")
print("  ✅ Risk/Reward calculator")
print("  ✅ Confidence scoring (0-100)")
print("  ✅ Smart trade recommendations")

for symbol in test_symbols:
    print_section(f"📈 {symbol} - COMPLETE ANALYSIS")
    
    analysis = analyzer.analyze_trend_advanced(symbol)
    
    if not analysis:
        print("❌ Analysis failed")
        continue
    
    # Price Info
    print_subsection("💰 PRICE INFORMATION")
    print(f"  Current Price: ${analysis['current_price']:,.2f}")
    
    # Trend Analysis
    print_subsection("📊 TREND ANALYSIS")
    direction_emoji = {
        'uptrend': '📈',
        'downtrend': '📉',
        'sideways': '➡️'
    }
    print(f"  Direction: {direction_emoji.get(analysis['trend_direction'], '❓')} {analysis['trend_direction'].upper()}")
    print(f"  Strength: {analysis['trend_strength']:.2f}/100")
    print(f"  ADX: {analysis['adx']:.2f}")
    
    # RSI
    print_subsection("📉 RSI (Relative Strength Index)")
    rsi = analysis['rsi']
    rsi_emoji = "🔴" if rsi > 70 else "🟢" if rsi < 30 else "🟡"
    print(f"  RSI: {rsi_emoji} {rsi:.2f}")
    print(f"  Status: {analysis['rsi_status'].upper()}")
    if rsi > 70:
        print(f"  ⚠️ WARNING: Overbought - potential reversal")
    elif rsi < 30:
        print(f"  ⚠️ WARNING: Oversold - potential reversal")
    
    # Volume
    print_subsection("📊 VOLUME ANALYSIS")
    vol_emoji = "📈" if analysis['volume_trend'] == "increasing" else "📉" if analysis['volume_trend'] == "decreasing" else "➡️"
    print(f"  Trend: {vol_emoji} {analysis['volume_trend'].upper()}")
    print(f"  Ratio: {analysis['volume_ratio']:.2f}x average")
    if analysis['volume_trend'] == "increasing":
        print(f"  ✅ Strong volume confirms trend")
    elif analysis['volume_trend'] == "decreasing":
        print(f"  ⚠️ Weak volume - trend may be losing steam")
    
    # Support/Resistance
    print_subsection("🎯 SUPPORT & RESISTANCE LEVELS")
    if analysis['nearest_resistance']:
        print(f"  Nearest Resistance: ${analysis['nearest_resistance']:,.2f} (+{((analysis['nearest_resistance']/analysis['current_price']-1)*100):.2f}%)")
    if analysis['nearest_support']:
        print(f"  Nearest Support: ${analysis['nearest_support']:,.2f} ({((analysis['nearest_support']/analysis['current_price']-1)*100):.2f}%)")
    
    if analysis['resistance_levels']:
        print(f"\n  Resistance Levels:")
        for i, level in enumerate(analysis['resistance_levels'][:3], 1):
            print(f"    R{i}: ${level:,.2f}")
    
    if analysis['support_levels']:
        print(f"\n  Support Levels:")
        for i, level in enumerate(analysis['support_levels'][:3], 1):
            print(f"    S{i}: ${level:,.2f}")
    
    # Multi-timeframe
    print_subsection("⏰ MULTI-TIMEFRAME ANALYSIS")
    mtf = analysis['timeframe_analysis']
    
    for tf in ['15m', '1h', '4h']:
        if tf in mtf and 'direction' in mtf[tf]:
            data = mtf[tf]
            emoji = "📈" if data['direction'] == 'uptrend' else "📉" if data['direction'] == 'downtrend' else "➡️"
            strength_emoji = "💪" if data['is_strong'] else "😐"
            print(f"  {tf:4s}: {emoji} {data['direction']:10s} | ADX: {data['adx']:5.1f} {strength_emoji}")
    
    if analysis['timeframes_aligned']:
        print(f"\n  ✅ ALL TIMEFRAMES ALIGNED - High confidence!")
    else:
        print(f"\n  ⚠️ Timeframes NOT aligned - Mixed signals")
    
    # Trade Setup
    print_subsection("💼 TRADE SETUP")
    print(f"  Entry Price: ${analysis['entry_price']:,.2f}")
    print(f"  Stop Loss: ${analysis['stop_loss']:,.2f} ({analysis['risk_reward']['risk_percent']:.2f}%)")
    print(f"  Take Profit: ${analysis['take_profit']:,.2f} ({analysis['risk_reward']['reward_percent']:.2f}%)")
    print(f"  Risk/Reward: {analysis['risk_reward']['risk_reward_ratio']}:1")
    
    if analysis['risk_reward']['is_favorable']:
        print(f"  ✅ Favorable R:R (≥2:1)")
    else:
        print(f"  ❌ Unfavorable R:R (<2:1)")
    
    # Signal & Confidence
    print_subsection("🎯 TRADING SIGNAL")
    print(f"  Signal Strength: {analysis['signal_strength']}")
    print(f"  Confidence Score: {analysis['confidence_score']}/100")
    
    # Progress bar for confidence
    bar_length = 40
    filled = int(bar_length * analysis['confidence_score'] / 100)
    bar = '█' * filled + '░' * (bar_length - filled)
    print(f"  [{bar}] {analysis['confidence_score']}%")
    
    print(f"\n  {analysis['action']}")
    
    # Reasons
    if analysis['reasons']:
        print(f"\n  ✅ Positive Factors:")
        for reason in analysis['reasons']:
            print(f"     • {reason}")
    
    if analysis['warnings']:
        print(f"\n  ⚠️ Risk Factors:")
        for warning in analysis['warnings']:
            print(f"     • {warning}")
    
    # Final Recommendation
    print_subsection("💡 RECOMMENDATION")
    
    if analysis['confidence_score'] >= 70:
        print(f"  🟢 HIGH CONFIDENCE TRADE")
        print(f"     This setup has strong confirmation across multiple indicators.")
        print(f"     Consider taking this trade with proper position sizing.")
    elif analysis['confidence_score'] >= 50:
        print(f"  🟡 MODERATE CONFIDENCE")
        print(f"     Some positive signals but also risk factors present.")
        print(f"     Consider smaller position size or wait for better setup.")
    elif analysis['confidence_score'] >= 30:
        print(f"  🟠 LOW CONFIDENCE")
        print(f"     Weak signals with significant risk factors.")
        print(f"     Only for experienced traders with tight risk management.")
    else:
        print(f"  🔴 NO TRADE")
        print(f"     Too many risk factors. Wait for better opportunity.")
    
    print(f"\n  📋 Position Sizing Example (1% risk):")
    print(f"     Account: $10,000")
    print(f"     Risk per trade: $100 (1%)")
    risk_per_unit = abs(analysis['entry_price'] - analysis['stop_loss'])
    if risk_per_unit > 0:
        position_size = 100 / risk_per_unit
        position_value = position_size * analysis['entry_price']
        print(f"     Position size: {position_size:.4f} {symbol[:-4]}")
        print(f"     Position value: ${position_value:.2f}")
        print(f"     Potential profit: ${100 * analysis['risk_reward']['risk_reward_ratio']:.2f}")

print_section("✅ ANALYSIS COMPLETE")
print("\n📚 Remember:")
print("  • Always use stop losses")
print("  • Never risk more than 1-2% per trade")
print("  • Wait for high confidence setups (70+)")
print("  • Check news and fundamentals")
print("  • Practice with paper trading first")
print("\n" + "="*80 + "\n")
