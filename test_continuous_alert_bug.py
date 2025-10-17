#!/usr/bin/env python3
"""
Test script to demonstrate the continuous alert bug
"""

from decimal import Decimal
from datetime import datetime, timezone
from price_monitor import PriceMonitor
from schemas import PriceLevel, Alert

def test_continuous_alert_bug():
    """Test the continuous alert crossover logic"""
    print("🐛 Testing Continuous Alert Bug")
    print("=" * 50)
    
    # Create a mock price monitor
    monitor = PriceMonitor()
    
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
    print("-" * 40)
    
    # First check: Price at $3100 (above target) - should not trigger
    price_3100 = Decimal("3100.00")
    should_trigger_1 = monitor.should_trigger_alert(eth_alert, price_3100)
    print(f"Price $3100: Should trigger = {should_trigger_1} (Expected: False)")
    
    # Simulate creating an alert when price first goes below $3000
    # This would happen in real scenario
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
    
    # Mock the get_last_alert_for_price_level method to return our test alert
    def mock_get_last_alert(price_level):
        return first_alert
    
    monitor.get_last_alert_for_price_level = mock_get_last_alert
    
    # Second check: Price at $2800 (still below target) - should trigger again
    price_2800 = Decimal("2800.00")
    should_trigger_2 = monitor.should_trigger_alert(eth_alert, price_2800)
    print(f"Price $2800: Should trigger = {should_trigger_2} (Expected: True)")
    
    # Third check: Price goes back up to $3100 then down to $2900 - should trigger
    print("\n📊 Test Scenario 2: Price goes up to $3100, then down to $2900")
    print("-" * 40)
    
    # Price goes up above target
    price_3100_again = Decimal("3100.00")
    should_trigger_3 = monitor.should_trigger_alert(eth_alert, price_3100_again)
    print(f"Price $3100: Should trigger = {should_trigger_3} (Expected: False)")
    
    # Price goes back down below target - this should trigger
    price_2900_again = Decimal("2900.00")
    should_trigger_4 = monitor.should_trigger_alert(eth_alert, price_2900_again)
    print(f"Price $2900: Should trigger = {should_trigger_4} (Expected: True)")
    
    print("\n" + "=" * 50)
    print("🔍 Analysis:")
    print("The bug is in the crossover detection logic.")
    print("Current logic compares last_alert.triggered_price with target_price,")
    print("but this prevents continuous alerts from triggering multiple times.")
    print("=" * 50)

if __name__ == "__main__":
    test_continuous_alert_bug()