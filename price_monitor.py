import os
import requests
from decimal import Decimal
from datetime import datetime, timezone
from typing import List, Dict, Optional
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
    
    def fetch_price(self, symbol: str) -> Optional[Decimal]:
        """Fetch current price from Binance API"""
        try:
            url = f"{self.binance_api_url}/ticker/price"
            params = {"symbol": symbol}
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            return Decimal(data["price"])
        except Exception as e:
            print(f"Error fetching price for {symbol}: {e}")
            return None
    
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
                return False
            
            # Get the last alert for this price level
            last_alert = self.get_last_alert_for_price_level(price_level)
            
            if last_alert is None:
                # No previous alerts, so this is the first trigger
                return True
            
            # Check if this is a crossover (price moved from one side of threshold to the other)
            if price_level.trigger_direction == "above":
                # Trigger if current price is above target AND last alert was below target
                return current_price >= price_level.target_price and last_alert.triggered_price < price_level.target_price
            else:  # below
                # Trigger if current price is below target AND last alert was above target
                return current_price <= price_level.target_price and last_alert.triggered_price > price_level.target_price
        
        return False
    
    def create_alert(self, price_level: PriceLevel, current_price: Decimal) -> bool:
        """Create alert record in database"""
        try:
            alert_data = {
                "price_level_id": price_level.id,
                "pair": price_level.pair,
                "triggered_price": float(current_price),
                "target_price": float(price_level.target_price),
                "trigger_direction": price_level.trigger_direction,
                "trigger_type": price_level.trigger_type,
                "triggered_at": datetime.now(timezone.utc).isoformat(),
                "notified": False
            }
            
            result = supabase.table("alerts").insert(alert_data).execute()
            return bool(result.data)
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
        """Process all active price levels and check for triggers"""
        price_levels = self.get_active_price_levels()
        triggered_count = 0
        
        print(f"Processing {len(price_levels)} active price levels...")
        
        for price_level in price_levels:
            current_price = self.fetch_price(price_level.pair)
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