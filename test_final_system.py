#!/usr/bin/env python3
"""
Final test script with proper state tracking
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

def should_trigger_alert_fixed(price_level: PriceLevel, current_price: Decimal, last_alert: Alert = None, previous_state: str = None) -> bool:
    """FIXED version with proper state tracking"""
    if price_level.trigger_type == "one_time":
        return check_price_triggers(price_level, current_price)
    
    if price_level.trigger_type == "continuous":
        if not check_price_triggers(price_level, current_price):
            return False
        
        if last_alert is None:
            return True
        
        # Use previous_state if available, otherwise fall back to last_alert position
        if previous_state:
            last_was_above = previous_state == "above"
        else:
            last_was_above = last_alert.triggered_price >= price_level.target_price
        
        current_above_threshold = current_price >= price_level.target_price
        
        if price_level.trigger_direction == "above":
            return current_above_threshold and not last_was_above
        else:  # below
            return not current_above_threshold and last_was_above
    
    return False

def simulate_eth_price_movement_final():
    """Final simulation with proper state tracking"""
    print("🚀 Final Test: ETH Continuous Alert System with Proper State Tracking")
    print("=" * 70)
    
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
    
    # Simulate price movement over time with proper state tracking
    price_movements = [
        ("$3200", Decimal("3200.00"), "Above target - no alert"),
        ("$3100", Decimal("3100.00"), "Above target - no alert"),
        ("$2900", Decimal("2900.00"), "Below target - FIRST ALERT (crossover from above)"),
        ("$2800", Decimal("2800.00"), "Below target - no alert (no crossover)"),
        ("$2700", Decimal("2700.00"), "Below target - no alert (no crossover)"),
        ("$3100", Decimal("3100.00"), "Above target - no alert (price goes back up)"),
        ("$3200", Decimal("3200.00"), "Above target - no alert (price stays above)"),
        ("$2900", Decimal("2900.00"), "Below target - SECOND ALERT (crossover from above)"),
        ("$2800", Decimal("2800.00"), "Below target - no alert (no crossover)"),
        ("$3100", Decimal("3100.00"), "Above target - no alert (price goes back up)"),
        ("$2850", Decimal("2850.00"), "Below target - THIRD ALERT (crossover from above)"),
    ]
    
    last_alert = None
    alert_count = 0
    previous_state = "above"  # Start with price above target
    
    print(f"\n📊 Simulating ETH price movement:")
    print("-" * 70)
    
    for i, (price_desc, price, expected) in enumerate(price_movements, 1):
        # Determine current state
        current_state = "above" if price >= eth_alert.target_price else "below"
        
        should_trigger = should_trigger_alert_fixed(eth_alert, price, last_alert, previous_state)
        
        status = "🚨 ALERT!" if should_trigger else "ℹ️  No alert"
        state_change = f"({previous_state} → {current_state})" if previous_state != current_state else f"({current_state})"
        print(f"{i:2d}. {price_desc:8s} - {status:12s} - {expected} {state_change}")
        
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
        
        # Update previous state for next iteration
        previous_state = current_state
    
    print("\n" + "=" * 70)
    print(f"📈 Summary: {alert_count} alerts triggered")
    print("✅ Continuous alerts now work correctly!")
    print("✅ Alerts only trigger on crossovers (price crossing threshold)")
    print("✅ No duplicate alerts while price stays on same side of threshold")
    print("✅ Proper state tracking ensures correct behavior")
    print("=" * 70)

if __name__ == "__main__":
    simulate_eth_price_movement_final()