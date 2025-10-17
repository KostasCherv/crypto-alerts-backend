#!/usr/bin/env python3
"""
Debug script to understand the continuous alert logic better
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
        # Default to above for backward compatibility
        return current_price >= price_level.target_price

def should_trigger_alert_fixed(price_level: PriceLevel, current_price: Decimal, last_alert: Alert = None) -> bool:
    """FIXED version - Check if alert should trigger (with proper crossover detection)"""
    # One-time alerts always trigger when condition is met
    if price_level.trigger_type == "one_time":
        return check_price_triggers(price_level, current_price)
    
    # Continuous alerts only trigger on crossovers
    if price_level.trigger_type == "continuous":
        # Check if price condition is met
        if not check_price_triggers(price_level, current_price):
            return False
        
        if last_alert is None:
            # No previous alerts, so this is the first trigger
            return True
        
        # For continuous alerts, we need to check if this is a crossover
        # The key insight: we need to determine if the price was previously on the 
        # opposite side of the threshold from where it is now
        
        current_above_threshold = current_price >= price_level.target_price
        last_alert_was_above_threshold = last_alert.triggered_price >= price_level.target_price
        
        print(f"  Debug: current_price={current_price}, target_price={price_level.target_price}")
        print(f"  Debug: current_above_threshold={current_above_threshold}")
        print(f"  Debug: last_alert.triggered_price={last_alert.triggered_price}")
        print(f"  Debug: last_alert_was_above_threshold={last_alert_was_above_threshold}")
        
        if price_level.trigger_direction == "above":
            # For "above" alerts: trigger if current price is above target 
            # AND the last alert was triggered when price was below target
            result = current_above_threshold and not last_alert_was_above_threshold
            print(f"  Debug: above logic: {current_above_threshold} and not {last_alert_was_above_threshold} = {result}")
            return result
        else:  # below
            # For "below" alerts: trigger if current price is below target 
            # AND the last alert was triggered when price was above target
            result = not current_above_threshold and last_alert_was_above_threshold
            print(f"  Debug: below logic: not {current_above_threshold} and {last_alert_was_above_threshold} = {result}")
            return result
    
    return False

def test_debug_continuous_alert():
    """Debug the continuous alert crossover logic"""
    print("🔍 Debugging Continuous Alert Logic")
    print("=" * 60)
    
    # Create a continuous "below" alert for ETH at $3000
    eth_alert = PriceLevel(
        id="test-eth-alert",
        pair="ETHUSDT", 
        target_price=Decimal("3000.00"),
        trigger_direction="below",
        trigger_type="continuous",
        is_active=True
    )
    
    print(f"Created ETH alert: {eth_alert.pair} {eth_alert.trigger_direction} ${eth_alert.target_price} ({eth_alert.trigger_type})")
    
    # Test scenario: Price goes from $3100 to $2900 (should trigger)
    print("\n📊 Test Scenario: Price goes up to $3100, then down to $2900")
    print("-" * 50)
    
    # Create an alert that was triggered when price was above target
    alert_above = Alert(
        id="alert-above",
        price_level_id=eth_alert.id,
        pair=eth_alert.pair,
        triggered_price=Decimal("3100.00"),  # Price when alert was triggered above target
        target_price=eth_alert.target_price,
        trigger_direction=eth_alert.trigger_direction,
        trigger_type=eth_alert.trigger_type,
        triggered_at=datetime.now(timezone.utc),
        notified=False
    )
    
    print(f"Last alert: triggered at ${alert_above.triggered_price} (above target: ${eth_alert.target_price})")
    
    # Price goes down below target - this should trigger (crossover!)
    price_2900 = Decimal("2900.00")
    print(f"\nChecking price ${price_2900}:")
    should_trigger = should_trigger_alert_fixed(eth_alert, price_2900, alert_above)
    print(f"Result: Should trigger = {should_trigger} (Expected: True - crossover!)")
    
    print("\n" + "=" * 60)
    print("🔍 Analysis:")
    print("For 'below' alerts, we want to trigger when:")
    print("1. Current price is below target (condition met)")
    print("2. Last alert was triggered when price was above target (crossover)")
    print("This ensures we only alert on crossovers, not continuously while below target.")
    print("=" * 60)

if __name__ == "__main__":
    test_debug_continuous_alert()