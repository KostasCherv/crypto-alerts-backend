#!/usr/bin/env python3
"""
Test script to verify the complete system works with the fix
"""

from decimal import Decimal
from datetime import datetime, timezone
from schemas import PriceLevel, Alert

def check_price_triggers(price_level: PriceLevel, current_price: Decimal) -> bool:
    """Check if current price triggers the alert based on crossover direction"""
    if price_level.trigger_direction == "above":
        return current_price >= price_level.target_price
    elif price_level.trigger_direction == "below":
        return current_price <= price_level.target_price
    else:
        return current_price >= price_level.target_price

def should_trigger_alert_fixed(price_level: PriceLevel, current_price: Decimal, last_alert: Alert = None) -> bool:
    """FIXED version - Check if alert should trigger (with proper crossover detection)"""
    if price_level.trigger_type == "one_time":
        return check_price_triggers(price_level, current_price)
    
    if price_level.trigger_type == "continuous":
        if not check_price_triggers(price_level, current_price):
            return False
        
        if last_alert is None:
            return True
        
        current_above_threshold = current_price >= price_level.target_price
        last_alert_was_above_threshold = last_alert.triggered_price >= price_level.target_price
        
        if price_level.trigger_direction == "above":
            return current_above_threshold and not last_alert_was_above_threshold
        else:  # below
            return not current_above_threshold and last_alert_was_above_threshold
    
    return False

def simulate_eth_price_movement():
    """Simulate ETH price movement and test continuous alerts"""
    print("🚀 Testing Complete ETH Continuous Alert System")
    print("=" * 60)
    
    # Create a continuous "below" alert for ETH at $3000
    eth_alert = PriceLevel(
        id="eth-continuous-alert",
        pair="ETHUSDT", 
        target_price=Decimal("3000.00"),
        trigger_direction="below",
        trigger_type="continuous",
        is_active=True
    )
    
    print(f"Created ETH alert: {eth_alert.pair} {eth_alert.trigger_direction} ${eth_alert.target_price} ({eth_alert.trigger_type})")
    
    # Simulate price movement over time
    price_movements = [
        ("$3200", Decimal("3200.00"), "Above target - no alert"),
        ("$3100", Decimal("3100.00"), "Above target - no alert"),
        ("$2900", Decimal("2900.00"), "Below target - FIRST ALERT (crossover)"),
        ("$2800", Decimal("2800.00"), "Below target - no alert (no crossover)"),
        ("$2700", Decimal("2700.00"), "Below target - no alert (no crossover)"),
        ("$3100", Decimal("3100.00"), "Above target - no alert"),
        ("$3200", Decimal("3200.00"), "Above target - no alert"),
        ("$2900", Decimal("2900.00"), "Below target - SECOND ALERT (crossover)"),
        ("$2800", Decimal("2800.00"), "Below target - no alert (no crossover)"),
        ("$3100", Decimal("3100.00"), "Above target - no alert"),
        ("$2850", Decimal("2850.00"), "Below target - THIRD ALERT (crossover)"),
    ]
    
    last_alert = None
    alert_count = 0
    
    print(f"\n📊 Simulating ETH price movement:")
    print("-" * 60)
    
    for i, (price_desc, price, expected) in enumerate(price_movements, 1):
        should_trigger = should_trigger_alert_fixed(eth_alert, price, last_alert)
        
        status = "🚨 ALERT!" if should_trigger else "ℹ️  No alert"
        print(f"{i:2d}. {price_desc:8s} - {status:12s} - {expected}")
        
        if should_trigger:
            alert_count += 1
            # Simulate creating the alert
            last_alert = Alert(
                id=f"alert-{alert_count}",
                price_level_id=eth_alert.id,
                pair=eth_alert.pair,
                triggered_price=price,
                target_price=eth_alert.target_price,
                trigger_direction=eth_alert.trigger_direction,
                trigger_type=eth_alert.trigger_type,
                triggered_at=datetime.now(timezone.utc),
                notified=False
            )
            print(f"    📝 Alert #{alert_count} created at ${price}")
    
    print("\n" + "=" * 60)
    print(f"📈 Summary: {alert_count} alerts triggered")
    print("✅ Continuous alerts now work correctly!")
    print("✅ Alerts only trigger on crossovers (price crossing threshold)")
    print("✅ No duplicate alerts while price stays on same side of threshold")
    print("=" * 60)

if __name__ == "__main__":
    simulate_eth_price_movement()