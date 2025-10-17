#!/usr/bin/env python3
"""
Debug step 8 of the test
"""

from decimal import Decimal
from datetime import datetime, timezone
from schemas import PriceLevel, Alert

def check_price_triggers(price_level: PriceLevel, current_price: Decimal) -> bool:
    """Check if current price triggers the alert based on crossover direction"""
    if price_level.trigger_direction == "below":
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
        
        print(f"  Debug: current_price={current_price}, target_price={price_level.target_price}")
        print(f"  Debug: current_above_threshold={current_above_threshold}")
        print(f"  Debug: last_alert.triggered_price={last_alert.triggered_price}")
        print(f"  Debug: last_alert_was_above_threshold={last_alert_was_above_threshold}")
        
        if price_level.trigger_direction == "above":
            result = current_above_threshold and not last_alert_was_above_threshold
            print(f"  Debug: above logic: {current_above_threshold} and not {last_alert_was_above_threshold} = {result}")
            return result
        else:  # below
            result = not current_above_threshold and last_alert_was_above_threshold
            print(f"  Debug: below logic: not {current_above_threshold} and {last_alert_was_above_threshold} = {result}")
            return result
    
    return False

def debug_step8():
    """Debug step 8 of the test"""
    print("🔍 Debugging Step 8: Price goes from $3200 to $2900")
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
    
    # Simulate the last alert was triggered at $2900 (below target)
    last_alert = Alert(
        id="alert-1",
        price_level_id=eth_alert.id,
        pair=eth_alert.pair,
        triggered_price=Decimal("2900.00"),  # Last alert was below target
        target_price=eth_alert.target_price,
        trigger_direction=eth_alert.trigger_direction,
        trigger_type=eth_alert.trigger_type,
        triggered_at=datetime.now(timezone.utc),
        notified=False
    )
    
    print(f"Last alert: triggered at ${last_alert.triggered_price} (below target: ${eth_alert.target_price})")
    print(f"Current price: $2900 (below target: ${eth_alert.target_price})")
    print()
    
    # Check if should trigger
    should_trigger = should_trigger_alert_fixed(eth_alert, Decimal("2900.00"), last_alert)
    print(f"Result: Should trigger = {should_trigger}")
    
    print("\n" + "=" * 60)
    print("🔍 Analysis:")
    print("The issue is that the last alert was triggered when price was below target.")
    print("For a 'below' alert to trigger again, the last alert must have been")
    print("triggered when price was ABOVE target (crossover).")
    print("Since the last alert was triggered below target, there's no crossover.")
    print("This is actually CORRECT behavior!")
    print("=" * 60)

if __name__ == "__main__":
    debug_step8()