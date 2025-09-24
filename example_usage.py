#!/usr/bin/env python3
"""
Example usage script for the crypto alerts backend
"""

import os
from decimal import Decimal
from datetime import datetime, timezone
from dotenv import load_dotenv
from supabase import ClientOptions, create_client, Client

# Load environment variables
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Initialize Supabase client
supabase = create_client(SUPABASE_URL, SUPABASE_KEY, options=ClientOptions(schema="public"))

def add_price_level(pair: str, target_price: float, trigger_direction: str = "above", trigger_type: str = "one_time"):
    """Add a new price level to monitor"""
    try:
        price_level_data = {
            "pair": pair,
            "target_price": target_price,
            "trigger_direction": trigger_direction,
            "is_active": True,
            "trigger_type": trigger_type,
            "created_at": datetime.now(timezone.utc).isoformat()
        }
        
        result = supabase.table("price_levels").insert(price_level_data).execute()
        
        if result.data:
            direction_emoji = "📈" if trigger_direction == "above" else "📉"
            print(f"✅ Added price level: {pair} {direction_emoji} ${target_price} ({trigger_type})")
            return result.data[0]
        else:
            print(f"❌ Failed to add price level: {pair}")
            return None
            
    except Exception as e:
        print(f"❌ Error adding price level: {e}")
        return None

def list_price_levels():
    """List all price levels"""
    try:
        result = supabase.table("price_levels").select("*").order("created_at", desc=True).execute()
        
        if result.data:
            print("\n📊 Current Price Levels:")
            print("-" * 70)
            for level in result.data:
                status = "🟢 Active" if level["is_active"] else "🔴 Inactive"
                direction_emoji = "📈" if level.get("trigger_direction") == "above" else "📉"
                direction = level.get("trigger_direction", "above")
                print(f"{level['pair']}: {direction_emoji} ${level['target_price']} ({direction}, {level['trigger_type']}) - {status}")
        else:
            print("No price levels found")
            
    except Exception as e:
        print(f"❌ Error listing price levels: {e}")

def list_recent_alerts():
    """List recent alerts"""
    try:
        result = supabase.table("alerts").select("*").order("triggered_at", desc=True).limit(10).execute()
        
        if result.data:
            print("\n🚨 Recent Alerts:")
            print("-" * 70)
            for alert in result.data:
                notified = "✅" if alert["notified"] else "⏳"
                direction_emoji = "📈" if alert.get("trigger_direction") == "above" else "📉"
                direction = alert.get("trigger_direction", "above")
                print(f"{alert['pair']}: {direction_emoji} ${alert['triggered_price']} (target: ${alert['target_price']}, {direction}) - {notified}")
        else:
            print("No alerts found")
            
    except Exception as e:
        print(f"❌ Error listing alerts: {e}")

def main():
    """Example usage"""
    print("🚀 Crypto Alerts Backend - Example Usage")
    print("=" * 50)
    
    # Add some example price levels
    print("\n1. Adding example price levels...")
    add_price_level("BTCUSDT", 112550.0, "above", "one_time")
    
    # List current price levels
    print("\n2. Current price levels:")
    list_price_levels()
    
    # List recent alerts
    print("\n3. Recent alerts:")
    list_recent_alerts()
    
    print("\n" + "=" * 50)
    print("💡 To run the monitoring system, use: python main.py")
    print("💡 To test setup, use: python test_setup.py")

if __name__ == "__main__":
    main()
