import os
import requests
from decimal import Decimal
from datetime import datetime, timezone
from typing import List, Dict, Optional, Set
from collections import defaultdict
from supabase import ClientOptions, create_client, Client
from schemas import PriceLevel, Alert, PriceData

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
BINANCE_API_URL = os.getenv("BINANCE_API_URL", "https://api.binance.com/api/v3")

# Initialize Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY, options=ClientOptions(schema="public"))

class PriceMonitor:
    def __init__(self):
        self.binance_api_url = BINANCE_API_URL
        self.price_cache: Dict[str, Decimal] = {}
        self.cache_timestamp: Dict[str, datetime] = {}
        self.cache_ttl_seconds = 30  # Cache prices for 30 seconds
        self.price_level_states: Dict[str, str] = {}  # Track previous state for each price level
    
    def fetch_price(self, symbol: str) -> Optional[Decimal]:
        """Fetch current price from Binance API with caching"""
        # Check cache first
        now = datetime.now(timezone.utc)
        if (symbol in self.price_cache and 
            symbol in self.cache_timestamp and 
            (now - self.cache_timestamp[symbol]).total_seconds() < self.cache_ttl_seconds):
            return self.price_cache[symbol]
        
        try:
            url = f"{self.binance_api_url}/ticker/price"
            params = {"symbol": symbol}
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            price = Decimal(data["price"])
            
            # Update cache
            self.price_cache[symbol] = price
            self.cache_timestamp[symbol] = now
            
            return price
        except Exception as e:
            print(f"Error fetching price for {symbol}: {e}")
            return None
    
    def fetch_prices_batch(self, symbols: Set[str]) -> Dict[str, Decimal]:
        """Fetch multiple prices in a single API call"""
        if not symbols:
            return {}
        
        try:
            # Use the 24hr ticker endpoint which can return multiple symbols
            url = f"{self.binance_api_url}/ticker/24hr"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            prices = {}
            now = datetime.now(timezone.utc)
            
            for item in data:
                symbol = item["symbol"]
                if symbol in symbols:
                    price = Decimal(item["lastPrice"])
                    prices[symbol] = price
                    
                    # Update cache
                    self.price_cache[symbol] = price
                    self.cache_timestamp[symbol] = now
            
            return prices
        except Exception as e:
            print(f"Error fetching batch prices: {e}")
            # Fallback to individual requests
            return {symbol: self.fetch_price(symbol) for symbol in symbols if self.fetch_price(symbol) is not None}
    
    def get_active_price_levels(self) -> List[PriceLevel]:
        """Get all active price levels from database"""
        try:
            result = supabase.table("price_levels").select("*").eq("is_active", True).execute()
            return [PriceLevel(**row) for row in result.data]
        except Exception as e:
            print(f"Error fetching active price levels: {e}")
            return []
    
    def check_price_triggers(self, price_level: PriceLevel, current_price: Decimal) -> bool:
        """Check if current price triggers the alert based on crossover direction"""
        if price_level.trigger_direction == "above":
            return current_price >= price_level.target_price
        elif price_level.trigger_direction == "below":
            return current_price <= price_level.target_price
        else:
            # Default to above for backward compatibility
            return current_price >= price_level.target_price
    
    def get_last_alert_for_price_level(self, price_level: PriceLevel) -> Optional[Alert]:
        """Get the last alert for this specific price level"""
        try:
            result = supabase.table("alerts").select("*").eq("price_level_id", price_level.id).order("triggered_at", desc=True).limit(1).execute()
            
            if result.data:
                return Alert(**result.data[0])
            return None
        except Exception as e:
            print(f"Error getting last alert for price level {price_level.id}: {e}")
            return None
    
    def should_trigger_alert(self, price_level: PriceLevel, current_price: Decimal) -> bool:
        """Check if alert should trigger (with crossover detection for continuous alerts)"""
        # One-time alerts always trigger when condition is met
        if price_level.trigger_type == "one_time":
            return self.check_price_triggers(price_level, current_price)
        
        # Continuous alerts only trigger on crossovers
        if price_level.trigger_type == "continuous":
            # Check if price condition is met
            if not self.check_price_triggers(price_level, current_price):
                # Update state even when not triggering
                current_above_threshold = current_price >= price_level.target_price
                state_key = f"{price_level.id}_{price_level.pair}"
                self.price_level_states[state_key] = "above" if current_above_threshold else "below"
                return False
            
            # Get the last alert for this price level
            last_alert = self.get_last_alert_for_price_level(price_level)
            
            if last_alert is None:
                # No previous alerts, so this is the first trigger
                return True
            
            # For continuous alerts, we need to check if this is a crossover
            current_above_threshold = current_price >= price_level.target_price
            
            # Use in-memory state tracking if available, otherwise fall back to database
            state_key = f"{price_level.id}_{price_level.pair}"
            if state_key in self.price_level_states:
                last_was_above_threshold = self.price_level_states[state_key] == "above"
            elif hasattr(last_alert, 'previous_state') and last_alert.previous_state:
                # Use the stored previous state from database
                last_was_above_threshold = last_alert.previous_state == "above"
            else:
                # Fallback to comparing triggered_price (for backward compatibility)
                last_was_above_threshold = last_alert.triggered_price >= price_level.target_price
            
            if price_level.trigger_direction == "above":
                # For "above" alerts: trigger if current price is above target 
                # AND the last alert was triggered when price was below target
                return current_above_threshold and not last_was_above_threshold
            else:  # below
                # For "below" alerts: trigger if current price is below target 
                # AND the last alert was triggered when price was above target
                return not current_above_threshold and last_was_above_threshold
        
        return False
    
    
    def create_alert(self, price_level: PriceLevel, current_price: Decimal) -> bool:
        """Create alert record in database"""
        try:
            # Determine the previous state for crossover tracking
            current_above_threshold = current_price >= price_level.target_price
            previous_state = "below" if current_above_threshold else "above"
            
            alert_data = {
                "price_level_id": price_level.id,
                "pair": price_level.pair,
                "triggered_price": float(current_price),
                "target_price": float(price_level.target_price),
                "trigger_direction": price_level.trigger_direction,
                "trigger_type": price_level.trigger_type,
                "previous_state": previous_state,
                "triggered_at": datetime.now(timezone.utc).isoformat(),
                "notified": False
            }
            
            result = supabase.table("alerts").insert(alert_data).execute()
            
            if result.data:
                # Update in-memory state tracking
                state_key = f"{price_level.id}_{price_level.pair}"
                self.price_level_states[state_key] = "above" if current_above_threshold else "below"
                return True
            return False
        except Exception as e:
            print(f"Error creating alert: {e}")
            return False
    
    def update_price_level(self, price_level: PriceLevel) -> bool:
        """Update price level after trigger"""
        try:
            if price_level.trigger_type == "one_time":
                # Deactivate one-time alerts
                update_data = {
                    "is_active": False,
                    "last_triggered_at": datetime.now(timezone.utc).isoformat()
                }
            else:
                # Update last triggered time for continuous alerts
                update_data = {
                    "last_triggered_at": datetime.now(timezone.utc).isoformat()
                }
            
            result = supabase.table("price_levels").update(update_data).eq("id", price_level.id).execute()
            return bool(result.data)
        except Exception as e:
            print(f"Error updating price level: {e}")
            return False
    
    def process_price_levels(self) -> int:
        """Process all active price levels and check for triggers with batch price fetching"""
        price_levels = self.get_active_price_levels()
        triggered_count = 0
        
        if not price_levels:
            print("No active price levels to process")
            return 0
        
        print(f"Processing {len(price_levels)} active price levels...")
        
        # Group price levels by trading pair for batch price fetching
        levels_by_pair = defaultdict(list)
        for price_level in price_levels:
            levels_by_pair[price_level.pair].append(price_level)
        
        # Fetch all prices in batch
        unique_pairs = set(levels_by_pair.keys())
        prices = self.fetch_prices_batch(unique_pairs)
        
        # Process each price level
        for price_level in price_levels:
            current_price = prices.get(price_level.pair)
            if current_price is None:
                print(f"Failed to fetch price for {price_level.pair}")
                continue
            
            print(f"Checking {price_level.pair}: current={current_price}, target={price_level.target_price} ({price_level.trigger_direction})")
            
            # Check if alert should trigger (with crossover detection)
            if self.should_trigger_alert(price_level, current_price):
                direction_emoji = "📈" if price_level.trigger_direction == "above" else "📉"
                print(f"🚨 ALERT TRIGGERED: {price_level.pair} {direction_emoji} {price_level.trigger_direction.upper()} {current_price}")
                
                # Create alert
                if self.create_alert(price_level, current_price):
                    print(f"✅ Alert created for {price_level.pair}")
                    triggered_count += 1
                    
                    # Update price level
                    if self.update_price_level(price_level):
                        print(f"✅ Price level updated for {price_level.pair}")
                    else:
                        print(f"❌ Failed to update price level for {price_level.pair}")
                else:
                    print(f"❌ Failed to create alert for {price_level.pair}")
            else:
                print(f"ℹ️  No trigger for {price_level.pair}")
        
        return triggered_count
    
    def run_monitoring_cycle(self) -> int:
        """Run one complete monitoring cycle"""
        print("🔍 Starting price monitoring cycle...")
        triggered_count = self.process_price_levels()
        
        if triggered_count > 0:
            print(f"✅ {triggered_count} alerts triggered!")
        else:
            print("ℹ️  No alerts triggered in this cycle")
        
        return triggered_count