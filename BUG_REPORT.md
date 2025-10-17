# Bug Report: ETH Continuous Price Alert Failure

## Summary
The continuous price alert system for ETH (and other cryptocurrencies) is not working correctly. Continuous alerts only trigger once instead of triggering on every crossover of the price threshold.

## Root Cause
The bug is in the `should_trigger_alert` method in `price_monitor.py` (lines 117-144). The crossover detection logic is flawed for continuous alerts.

### Current Flawed Logic
```python
# For "below" alerts: trigger if current price is below target 
# AND the last alert was triggered when price was above target
return (current_price <= price_level.target_price and 
        last_alert.triggered_price > price_level.target_price)
```

### The Problem
1. **For "below" alerts**: Once an alert is triggered below the target, `last_alert.triggered_price ≤ target_price`. The condition `last_alert.triggered_price > target_price` will never be true again.

2. **For "above" alerts**: Once an alert is triggered above the target, `last_alert.triggered_price ≥ target_price`. The condition `last_alert.triggered_price < target_price` will never be true again.

This means continuous alerts only trigger once, not continuously as intended.

## Impact
- ETH continuous price alerts only trigger once when price first crosses the threshold
- No subsequent alerts when price crosses back and forth across the threshold
- Users miss important price movements and trading opportunities
- The "continuous" alert type is effectively broken

## Solution
The system needs to track the **previous state** (whether price was above or below the threshold) rather than just comparing the last triggered price.

### Option 1: Database Schema Update (Recommended)
Add a field to track the previous state:

```sql
ALTER TABLE price_levels ADD COLUMN previous_state TEXT CHECK (previous_state IN ('above', 'below'));
```

Then update the logic to use this field for crossover detection.

### Option 2: Enhanced Alert Tracking
Modify the alert creation logic to store the previous state in the alert record:

```sql
ALTER TABLE alerts ADD COLUMN previous_state TEXT CHECK (previous_state IN ('above', 'below'));
```

### Option 3: State Tracking in Application
Implement in-memory state tracking for the current monitoring session.

## Test Cases
The bug can be reproduced with this scenario:

1. Create a continuous "below" alert for ETH at $3000
2. Price goes from $3100 to $2900 → Alert triggers (correct)
3. Price goes from $2900 to $2800 → No alert (correct, no crossover)
4. Price goes from $2800 to $3100 → No alert (correct, no crossover)
5. Price goes from $3100 to $2900 → **Should trigger but doesn't** (BUG!)

## Files Affected
- `price_monitor.py` - Main bug location
- `schema.sql` - May need updates for state tracking
- `schemas.py` - May need updates for new fields

## Priority
**HIGH** - This is a critical bug that breaks the core functionality of continuous alerts.

## Status
- ✅ Bug identified and analyzed
- ✅ Root cause determined
- ✅ Solution designed
- ⏳ Implementation pending
- ⏳ Testing pending