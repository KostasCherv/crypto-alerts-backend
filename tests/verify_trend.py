import os
import sys

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from trend_analysis import TrendAnalyzer

print("\n" + "="*60)
print("Testing Professional Trend Analyzer")
print("="*60)

analyzer = TrendAnalyzer()
trend = analyzer.analyze_trend('ETHUSDT')

if trend:
    print(f"\n✅ SUCCESS!")
    print(f"Pair: {trend.pair}")
    print(f"Direction: {trend.trend_direction.upper()}")
    print(f"Strength: {trend.trend_strength}/100")
    print(f"Timestamp: {trend.calculated_at}")
    print("\n" + "="*60)
else:
    print("\n❌ FAILED to analyze trend")
    print("="*60)
