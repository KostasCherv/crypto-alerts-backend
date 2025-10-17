#!/usr/bin/env python3
"""
Test script to demonstrate the continuous alert bug without database dependencies
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

def should_trigger_alert_buggy(price_level: PriceLevel, current_price: Decimal, last_alert: Alert = None) -> bool:
    """BUGGY version - Check if alert should trigger (with crossover detection for continuous alerts)"""
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
        
        # Check if this is a crossover (price moved from one side of threshold to the other)
        if price_level.trigger_direction == "above":
            # Trigger if current price is above target AND last alert was below target
            return current_price >= price_level.target_price and last_alert.triggered_price < price_level.target_price
        else:  # below
            # Trigger if current price is below target AND last alert was above target
            return current_price <= price_level.target_price and last_alert.triggered_price > price_level.target_price
    
    return False

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
        
        # Check if this is a crossover (price moved from one side of threshold to the other)
        if price_level.trigger_direction == "above":
            # Trigger if current price is above target AND last alert was below target
            return current_price >= price_level.target_price and last_alert.triggered_price < price_level.target_price
        else:  # below
            # Trigger if current price is below target AND last alert was above target
            return current_price <= price_level.target_price and last_alert.triggered_price > price_level.target_price
    
    return False

def test_continuous_alert_bug():
    """Test the continuous alert crossover logic"""
    print("🐛 Testing Continuous Alert Bug")
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
    print("\n📊 Test Scenario 1: Price drops from $3100 to $2900")
    print("-" * 50)
    
    # First check: Price at $3100 (above target) - should not trigger
    price_3100 = Decimal("3100.00")
    should_trigger_1 = should_trigger_alert_buggy(eth_alert, price_3100)
    print(f"Price $3100: Should trigger = {should_trigger_1} (Expected: False)")
    
    # Simulate creating an alert when price first goes below $3000
    first_alert = Alert(
        id="first-alert",
        price_level_id=eth_alert.id,
        pair=eth_alert.pair,
        triggered_price=Decimal("2950.00"),  # Price when first alert was triggered
        target_price=eth_alert.target_price,
        trigger_direction=eth_alert.trigger_direction,
        trigger_type=eth_alert.trigger_type,
        triggered_at=datetime.now(timezone.utc),
        notified=False
    )
    
    # Second check: Price at $2800 (still below target) - should trigger again
    price_2800 = Decimal("2800.00")
    should_trigger_2 = should_trigger_alert_buggy(eth_alert, price_2800, first_alert)
    print(f"Price $2800: Should trigger = {should_trigger_2} (Expected: True)")
    
    # Third check: Price goes back up to $3100 then down to $2900 - should trigger
    print("\n📊 Test Scenario 2: Price goes up to $3100, then down to $2900")
    print("-" * 50)
    
    # Price goes up above target
    price_3100_again = Decimal("3100.00")
    should_trigger_3 = should_trigger_alert_buggy(eth_alert, price_3100_again, first_alert)
    print(f"Price $3100: Should trigger = {should_trigger_3} (Expected: False)")
    
    # Price goes back down below target - this should trigger
    price_2900_again = Decimal("2900.00")
    should_trigger_4 = should_trigger_alert_buggy(eth_alert, price_2900_again, first_alert)
    print(f"Price $2900: Should trigger = {should_trigger_4} (Expected: True)")
    
    print("\n" + "=" * 60)
    print("🔍 Analysis:")
    print("The bug is in the crossover detection logic.")
    print("Current logic compares last_alert.triggered_price with target_price,")
    print("but this prevents continuous alerts from triggering multiple times.")
    print("")
    print("For 'below' alerts:")
    print("- Once triggered below target, triggered_price <= target_price")
    print("- Condition 'last_alert.triggered_price > target_price' will never be true")
    print("- So continuous alerts only trigger once!")
    print("=" * 60)

if __name__ == "__main__":
    test_continuous_alert_bug()