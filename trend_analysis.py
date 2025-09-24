import os
import requests
import numpy as np
from decimal import Decimal
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Optional
from supabase import ClientOptions, create_client, Client
from schemas import Trend, PriceData

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
BINANCE_API_URL = os.getenv("BINANCE_API_URL", "https://api.binance.com/api/v3")

# Initialize Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY, options=ClientOptions(schema="public"))

class TrendAnalyzer:
    def __init__(self):
        self.binance_api_url = BINANCE_API_URL
    
    def fetch_historical_prices(self, symbol: str, limit: int = 50) -> List[Decimal]:
        """Fetch historical price data from Binance API"""
        try:
            url = f"{self.binance_api_url}/klines"
            params = {
                "symbol": symbol,
                "interval": "15m",  # 15-minute intervals
                "limit": limit
            }
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            # Extract closing prices (index 4 in klines data)
            prices = [Decimal(str(kline[4])) for kline in data]
            return prices
        except Exception as e:
            print(f"Error fetching historical prices for {symbol}: {e}")
            return []
    
    def calculate_sma(self, prices: List[Decimal], period: int) -> Decimal:
        """Calculate Simple Moving Average"""
        if len(prices) < period:
            return Decimal("0")
        
        recent_prices = prices[-period:]
        return sum(recent_prices) / len(recent_prices)
    
    def calculate_trend_direction(self, prices: List[Decimal]) -> str:
        """Calculate trend direction using SMA comparison"""
        if len(prices) < 30:
            return "sideways"
        
        short_sma = self.calculate_sma(prices, 10)
        long_sma = self.calculate_sma(prices, 30)
        
        if long_sma == 0:
            return "sideways"
        
        sma_diff = ((short_sma - long_sma) / long_sma) * 100
        
        if sma_diff > 2:
            return "uptrend"
        elif sma_diff < -2:
            return "downtrend"
        else:
            return "sideways"
    
    def calculate_trend_strength(self, prices: List[Decimal]) -> Decimal:
        """Calculate trend strength using simplified ADX"""
        if len(prices) < 2:
            return Decimal("0")
        
        # Convert to numpy array for easier calculations
        price_array = np.array([float(p) for p in prices])
        
        # Calculate high and low arrays (simplified - using price as both high and low)
        highs = price_array
        lows = price_array
        
        directional_movement = 0.0
        
        for i in range(1, len(price_array)):
            high_diff = highs[i] - highs[i-1]
            low_diff = lows[i-1] - lows[i]
            
            if high_diff > low_diff and high_diff > 0:
                directional_movement += high_diff
        
        # Normalize to 0-100 scale
        avg_price = np.mean(price_array)
        if avg_price == 0:
            return Decimal("0")
        
        strength = min(100, (directional_movement / avg_price) * 100)
        return Decimal(str(strength))
    
    def analyze_trend(self, symbol: str) -> Optional[Trend]:
        """Analyze trend for a given symbol"""
        prices = self.fetch_historical_prices(symbol, 50)
        if not prices:
            print(f"No price data available for {symbol}")
            return None
        
        trend_direction = self.calculate_trend_direction(prices)
        trend_strength = self.calculate_trend_strength(prices)
        
        return Trend(
            pair=symbol,
            trend_direction=trend_direction,
            trend_strength=trend_strength,
            calculated_at=datetime.now(timezone.utc)
        )
    
    def save_trend(self, trend: Trend) -> bool:
        """Save trend analysis to database"""
        try:
            trend_data = {
                "pair": trend.pair,
                "trend_direction": trend.trend_direction,
                "trend_strength": float(trend.trend_strength),
                "calculated_at": trend.calculated_at.isoformat()
            }
            
            result = supabase.table("trends").insert(trend_data).execute()
            return bool(result.data)
        except Exception as e:
            print(f"Error saving trend: {e}")
            return False
    
    def get_unique_pairs(self) -> List[str]:
        """Get unique trading pairs from price levels"""
        try:
            result = supabase.table("price_levels").select("pair").execute()
            pairs = list(set([row["pair"] for row in result.data]))
            return pairs
        except Exception as e:
            print(f"Error fetching pairs: {e}")
            return []
    
    def run_trend_analysis(self) -> int:
        """Run trend analysis for all unique pairs"""
        pairs = self.get_unique_pairs()
        analyzed_count = 0
        
        print(f"Analyzing trends for {len(pairs)} pairs...")
        
        for pair in pairs:
            print(f"Analyzing trend for {pair}...")
            trend = self.analyze_trend(pair)
            
            if trend:
                if self.save_trend(trend):
                    print(f"✅ Trend analysis saved for {pair}: {trend.trend_direction} (strength: {trend.trend_strength})")
                    analyzed_count += 1
                else:
                    print(f"❌ Failed to save trend for {pair}")
            else:
                print(f"❌ Failed to analyze trend for {pair}")
        
        return analyzed_count
