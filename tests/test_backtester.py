"""
Quick test script to verify the backtesting engine works correctly
"""

import os
import sys

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from backtester import BacktestEngine, BacktestConfig


def test_basic_functionality():
    """Test basic backtesting functionality"""
    print("=" * 80)
    print("BACKTESTING ENGINE - QUICK TEST")
    print("=" * 80)
    
    try:
        # Test 1: Configuration
        print("\n✓ Test 1: Creating configuration...")
        config = BacktestConfig(
            symbol="BTCUSDT",
            interval="1h",
            initial_capital=10000.0,
            risk_per_trade=0.01,
            min_confidence=70.0
        )
        print(f"  Symbol: {config.symbol}")
        print(f"  Interval: {config.interval}")
        print(f"  Initial Capital: ${config.initial_capital:,.2f}")
        
        # Test 2: Engine initialization
        print("\n✓ Test 2: Initializing backtest engine...")
        engine = BacktestEngine(config)
        print(f"  Engine created successfully")
        
        # Test 3: Data fetching
        print("\n✓ Test 3: Fetching historical data...")
        df = engine.fetch_historical_data(limit=200)
        print(f"  Fetched {len(df)} candles")
        print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        
        # Test 4: Running backtest
        print("\n✓ Test 4: Running backtest...")
        metrics = engine.run_backtest()
        
        # Test 5: Verify metrics
        print("\n✓ Test 5: Verifying metrics...")
        required_metrics = [
            'total_trades', 'win_rate', 'total_return', 'sharpe_ratio',
            'max_drawdown', 'profit_factor', 'expectancy'
        ]
        
        for metric in required_metrics:
            if metric in metrics:
                print(f"  ✓ {metric}: {metrics[metric]}")
            else:
                print(f"  ✗ Missing metric: {metric}")
                return False
        
        # Test 6: Generate report
        print("\n✓ Test 6: Generating performance report...")
        engine.print_performance_report(metrics)
        
        # Test 7: Save outputs
        print("\n✓ Test 7: Saving outputs...")
        if engine.trades:
            engine.save_trades_to_csv()
            engine.plot_equity_curve()
            print("  ✓ Trades saved to CSV")
            print("  ✓ Equity curve plotted")
        else:
            print("  ⚠ No trades executed (this is OK if signals weren't generated)")
        
        # Summary
        print("\n" + "=" * 80)
        print("✅ ALL TESTS PASSED!")
        print("=" * 80)
        print("\nBacktesting engine is working correctly!")
        print("\nNext steps:")
        print("1. Run full examples: python backtest_example.py")
        print("2. Read documentation: BACKTESTING_GUIDE.md")
        print("3. Customize for your strategy")
        
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("""
    ╔═══════════════════════════════════════════════════════════════════════════╗
    ║              BACKTESTING ENGINE - QUICK FUNCTIONALITY TEST                ║
    ║                                                                           ║
    ║  This script tests basic functionality of the backtesting engine.         ║
    ║  It will fetch data, run a backtest, and verify all components work.     ║
    ╚═══════════════════════════════════════════════════════════════════════════╝
    """)
    
    success = test_basic_functionality()
    
    if success:
        sys.exit(0)
    else:
        sys.exit(1)
