"""
Download Crypto Data Script
==========================

This script downloads real cryptocurrency data from Binance for Bitcoin and Ethereum
across all required timeframes (15m, 1h, 4h, 1d) for 1 year.

Usage:
    python download_data.py

Author: Professional Trading System
Version: 1.0
"""

import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_manager import DataManager, download_all_data, print_data_status

def main():
    """Main function to download crypto data"""
    print("🎯 Crypto Data Downloader")
    print("=" * 50)
    
    # Create data manager
    manager = DataManager()
    
    # Print current status
    print("📊 Current data status:")
    manager.print_data_status()
    
    # Download Bitcoin and Ethereum data
    print(f"\n🚀 Downloading BTC and ETH data for 1 year...")
    print("📋 Timeframes: 15m, 1h, 4h, 1d")
    print("📋 Assets: BTC, ETH")
    print("📋 Duration: 365 days")
    
    try:
        # Download all data
        manager.download_all_data(
            assets=['BTC', 'ETH'],
            timeframes=['15m', '1h', '4h', '1d'],
            days=365
        )
        
        print("\n✅ Download completed successfully!")
        
        # Print final status
        print(f"\n📊 Final data status:")
        manager.print_data_status()
        
        # Show data info
        print(f"\n📈 Data Summary:")
        btc_4h = manager.get_data('BTC', '4h')
        eth_4h = manager.get_data('ETH', '4h')
        
        print(f"BTC 4h: {len(btc_4h)} records from {btc_4h.index[0]} to {btc_4h.index[-1]}")
        print(f"ETH 4h: {len(eth_4h)} records from {eth_4h.index[0]} to {eth_4h.index[-1]}")
        
        print(f"\n🎉 Ready for backtesting with real data!")
        
    except Exception as e:
        print(f"❌ Error during download: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
