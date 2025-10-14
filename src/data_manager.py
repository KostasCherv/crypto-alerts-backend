"""
Data Manager for Crypto Trading Strategies
=========================================

This module handles downloading, storing, and retrieving real cryptocurrency data
from Binance API. It supports multiple timeframes and handles cold start scenarios.

Features:
- Download real data from Binance API
- Store data locally to avoid repeated downloads
- Support multiple timeframes (15m, 1h, 4h, 1d)
- Handle cold start scenarios
- Automatic data validation and cleaning

Author: Professional Trading System
Version: 1.0
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import ccxt
import warnings
warnings.filterwarnings('ignore')

class DataManager:
    """Manages cryptocurrency data download and storage"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.exchange = ccxt.binance({
            'enableRateLimit': True,
            'timeout': 30000,  # 30 seconds timeout
        })
        
        # Create data directory
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Supported assets and timeframes
        self.assets = ['BTC', 'ETH', 'BNB', 'ADA', 'XRP', 'SOL', 'DOGE', 'AVAX', 'LINK']
        self.timeframes = ['15m', '1h', '4h', '1d']
        
        # Timeframe to days mapping for 1 year of data
        self.timeframe_days = {
            '15m': 365,  # 1 year
            '1h': 365,   # 1 year
            '4h': 365,   # 1 year
            '1d': 365    # 1 year
        }
    
    def download_data(self, asset: str, timeframe: str, days: int = 365) -> pd.DataFrame:
        """Download cryptocurrency data from Binance"""
        print(f"⬇️ Downloading {asset} {timeframe} data for {days} days...")
        
        try:
            # Convert asset to Binance format
            symbol = f"{asset}/USDT"
            print(f"   Symbol: {symbol}")
            
            # Calculate total records needed
            if timeframe == '15m':
                total_records = days * 96  # 96 15-minute periods per day
            elif timeframe == '1h':
                total_records = days * 24  # 24 hours per day
            elif timeframe == '4h':
                total_records = days * 6   # 6 4-hour periods per day
            elif timeframe == '1d':
                total_records = days       # 1 day per day
            else:
                raise ValueError(f"Unsupported timeframe: {timeframe}")
            
            print(f"   Total records needed: {total_records}")
            
            # Binance API limit is 1000 records per request
            max_limit = 1000
            
            if total_records <= max_limit:
                # Single request for small datasets
                print(f"   Single request (≤{max_limit} records)")
                ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=total_records)
                print(f"   Fetched {len(ohlcv)} records")
            else:
                # Multiple requests for large datasets
                print(f"   Multiple requests needed (>{max_limit} records)")
                ohlcv = self._download_large_dataset(symbol, timeframe, total_records, max_limit)
                print(f"   Fetched {len(ohlcv)} records total")
            
            # Convert to DataFrame
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Sort by timestamp (oldest first)
            df = df.sort_index()
            
            # Remove duplicates (in case of overlap)
            df = df[~df.index.duplicated(keep='first')]
            
            # Validate data
            df = self._validate_data(df, asset, timeframe)
            
            print(f"✅ Downloaded {len(df)} records for {asset} {timeframe}")
            return df
            
        except Exception as e:
            print(f"❌ Error downloading {asset} {timeframe}: {str(e)}")
            raise
    
    def _download_large_dataset(self, symbol: str, timeframe: str, total_records: int, max_limit: int) -> list:
        """Download large datasets by making multiple API calls"""
        import time
        from datetime import datetime, timedelta
        
        all_data = []
        
        # Calculate timeframe interval in milliseconds
        timeframe_ms = {
            '15m': 15 * 60 * 1000,
            '1h': 60 * 60 * 1000,
            '4h': 4 * 60 * 60 * 1000,
            '1d': 24 * 60 * 60 * 1000
        }
        
        interval_ms = timeframe_ms.get(timeframe, 15 * 60 * 1000)
        
        # Calculate the start time (365 days ago)
        end_time = datetime.now()
        start_time = end_time - timedelta(days=365)
        start_timestamp = int(start_time.timestamp() * 1000)
        
        print(f"   Date range: {start_time} to {end_time}")
        print(f"   Making multiple API calls to get {total_records} records...")
        
        current_since = start_timestamp
        batch_count = 0
        
        while len(all_data) < total_records:
            batch_count += 1
            batch_size = min(max_limit, total_records - len(all_data))
            
            print(f"   Batch {batch_count}: Fetching {batch_size} records from {datetime.fromtimestamp(current_since/1000)}")
            
            try:
                # Fetch batch of data
                batch_data = self.exchange.fetch_ohlcv(symbol, timeframe, since=current_since, limit=batch_size)
                
                if not batch_data:
                    print(f"   No more data available")
                    break
                
                # Add to our collection
                all_data.extend(batch_data)
                
                # Update since timestamp to get next batch
                # Use the newest timestamp from this batch plus one interval
                newest_timestamp = batch_data[-1][0]
                current_since = newest_timestamp + interval_ms
                
                print(f"   Fetched {len(batch_data)} records, total: {len(all_data)}")
                
                # Rate limiting - small delay between requests
                time.sleep(0.1)
                
            except Exception as e:
                print(f"   Error in batch download: {str(e)}")
                break
        
        print(f"   Total records downloaded: {len(all_data)}")
        return all_data
    
    def _validate_data(self, df: pd.DataFrame, asset: str, timeframe: str) -> pd.DataFrame:
        """Validate and clean downloaded data"""
        print(f"🔍 Validating {asset} {timeframe} data...")
        
        # Check for missing values
        if df.isnull().any().any():
            print(f"⚠️ Found missing values in {asset} {timeframe}, filling...")
            df = df.fillna(method='ffill')
        
        # Check for zero or negative prices
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            if (df[col] <= 0).any():
                print(f"⚠️ Found invalid prices in {col}, cleaning...")
                df = df[df[col] > 0]
        
        # Check for high-low consistency
        if (df['high'] < df['low']).any():
            print(f"⚠️ Found high < low inconsistencies, fixing...")
            df.loc[df['high'] < df['low'], 'high'] = df.loc[df['high'] < df['low'], 'low']
        
        # Check for open-close consistency
        if (df['open'] > df['high']).any() or (df['open'] < df['low']).any():
            print(f"⚠️ Found open price inconsistencies, fixing...")
            df.loc[df['open'] > df['high'], 'open'] = df.loc[df['open'] > df['high'], 'high']
            df.loc[df['open'] < df['low'], 'open'] = df.loc[df['open'] < df['low'], 'low']
        
        # Check for reasonable price ranges (basic sanity check)
        if asset == 'BTC':
            if (df['close'] < 1000).any() or (df['close'] > 200000).any():
                print(f"⚠️ Found unrealistic BTC prices, filtering...")
                df = df[(df['close'] >= 1000) & (df['close'] <= 200000)]
        elif asset == 'ETH':
            if (df['close'] < 50).any() or (df['close'] > 20000).any():
                print(f"⚠️ Found unrealistic ETH prices, filtering...")
                df = df[(df['close'] >= 50) & (df['close'] <= 20000)]
        
        print(f"✅ Data validation completed for {asset} {timeframe}")
        return df
    
    def save_data(self, df: pd.DataFrame, asset: str, timeframe: str, days: int = 365):
        """Save data to local file"""
        filename = f"{asset.lower()}_{timeframe}_{days}d.csv"
        filepath = os.path.join(self.data_dir, filename)
        
        df.to_csv(filepath)
        print(f"💾 Saved {len(df)} records to {filepath}")
    
    def load_data(self, asset: str, timeframe: str, days: int = 365) -> Optional[pd.DataFrame]:
        """Load data from local file"""
        filename = f"{asset.lower()}_{timeframe}_{days}d.csv"
        filepath = os.path.join(self.data_dir, filename)
        
        if os.path.exists(filepath):
            try:
                df = pd.read_csv(filepath, index_col=0, parse_dates=True)
                print(f"📁 Loaded {len(df)} records from {filepath}")
                return df
            except Exception as e:
                print(f"❌ Error loading {filepath}: {str(e)}")
                return None
        else:
            return None
    
    def get_data(self, asset: str, timeframe: str, days: int = 365, force_download: bool = False) -> pd.DataFrame:
        """
        Get data with cold start handling
        
        Args:
            asset: Asset symbol (BTC, ETH, etc.)
            timeframe: Timeframe (15m, 1h, 4h, 1d)
            days: Number of days of data
            force_download: Force re-download even if file exists
        
        Returns:
            DataFrame with OHLCV data
        """
        # Validate inputs
        if asset.upper() not in self.assets:
            raise ValueError(f"Unsupported asset: {asset}. Supported: {self.assets}")
        
        if timeframe not in self.timeframes:
            raise ValueError(f"Unsupported timeframe: {timeframe}. Supported: {self.timeframes}")
        
        # Try to load from local file first (unless force download)
        if not force_download:
            df = self.load_data(asset, timeframe, days)
            if df is not None:
                return df
        
        # Download if not found locally or force download
        print(f"🔄 Cold start: Downloading {asset} {timeframe} data...")
        df = self.download_data(asset, timeframe, days)
        
        # Save for future use
        self.save_data(df, asset, timeframe, days)
        
        return df
    
    def download_all_data(self, assets: List[str] = None, timeframes: List[str] = None, days: int = 365):
        """Download data for all assets and timeframes"""
        if assets is None:
            assets = self.assets
        if timeframes is None:
            timeframes = self.timeframes
        
        print(f"🚀 Starting bulk download for {len(assets)} assets × {len(timeframes)} timeframes")
        print(f"📊 Total downloads: {len(assets) * len(timeframes)}")
        
        for asset in assets:
            for timeframe in timeframes:
                try:
                    # Check if data already exists
                    if self.load_data(asset, timeframe, days) is not None:
                        print(f"⏭️ Skipping {asset} {timeframe} (already exists)")
                        continue
                    
                    # Download data
                    df = self.download_data(asset, timeframe, days)
                    self.save_data(df, asset, timeframe, days)
                    
                    # Small delay to respect rate limits
                    import time
                    time.sleep(0.1)
                    
                except Exception as e:
                    print(f"❌ Failed to download {asset} {timeframe}: {str(e)}")
                    continue
        
        print("✅ Bulk download completed!")
    
    def get_data_info(self) -> Dict[str, Dict[str, bool]]:
        """Get information about available data files"""
        info = {}
        
        for asset in self.assets:
            info[asset] = {}
            for timeframe in self.timeframes:
                filename = f"{asset.lower()}_{timeframe}_365d.csv"
                filepath = os.path.join(self.data_dir, filename)
                info[asset][timeframe] = os.path.exists(filepath)
        
        return info
    
    def print_data_status(self):
        """Print status of available data files"""
        print("📊 Data Status:")
        print("=" * 50)
        
        info = self.get_data_info()
        
        for asset in self.assets:
            print(f"\n{asset}:")
            for timeframe in self.timeframes:
                status = "✅" if info[asset][timeframe] else "❌"
                print(f"  {timeframe}: {status}")

# Convenience functions for easy access
def get_data(asset: str, timeframe: str, days: int = 365, force_download: bool = False) -> pd.DataFrame:
    """
    Convenience function to get data with cold start handling
    
    Args:
        asset: Asset symbol (BTC, ETH, etc.)
        timeframe: Timeframe (15m, 1h, 4h, 1d)
        days: Number of days of data
        force_download: Force re-download even if file exists
    
    Returns:
        DataFrame with OHLCV data
    """
    manager = DataManager()
    return manager.get_data(asset, timeframe, days, force_download)

def download_all_data(assets: List[str] = None, timeframes: List[str] = None, days: int = 365):
    """Convenience function to download all data"""
    manager = DataManager()
    manager.download_all_data(assets, timeframes, days)

def print_data_status():
    """Convenience function to print data status"""
    manager = DataManager()
    manager.print_data_status()

# Example usage and testing
if __name__ == "__main__":
    print("🎯 Crypto Data Manager")
    print("=" * 50)
    
    # Create data manager
    manager = DataManager()
    
    # Print current status
    manager.print_data_status()
    
    # Download Bitcoin and Ethereum data for all timeframes
    print(f"\n🚀 Downloading BTC and ETH data...")
    manager.download_all_data(assets=['BTC', 'ETH'], timeframes=['15m', '1h', '4h', '1d'])
    
    # Test data retrieval
    print(f"\n🧪 Testing data retrieval...")
    btc_4h = manager.get_data('BTC', '4h')
    eth_1h = manager.get_data('ETH', '1h')
    
    print(f"BTC 4h data shape: {btc_4h.shape}")
    print(f"ETH 1h data shape: {eth_1h.shape}")
    
    # Print final status
    print(f"\n📊 Final data status:")
    manager.print_data_status()
    
    print("\n✅ Data manager testing completed!")
