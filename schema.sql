-- Crypto Price Alert System Database Schema
-- Run this SQL in your Supabase SQL editor

-- Price levels to monitor
CREATE TABLE price_levels (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  pair TEXT NOT NULL, -- e.g., 'BTCUSDT'
  target_price DECIMAL NOT NULL,
  trigger_direction TEXT CHECK (trigger_direction IN ('above', 'below')) DEFAULT 'above',
  is_active BOOLEAN DEFAULT TRUE, -- Enable/disable this alert
  trigger_type TEXT CHECK (trigger_type IN ('one_time', 'continuous')) DEFAULT 'one_time',
  created_at TIMESTAMP DEFAULT NOW(),
  last_triggered_at TIMESTAMP -- Track when last triggered for continuous alerts
);

-- Triggered alerts
CREATE TABLE alerts (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  price_level_id UUID REFERENCES price_levels(id),
  pair TEXT NOT NULL,
  triggered_price DECIMAL NOT NULL,
  target_price DECIMAL NOT NULL,
  trigger_direction TEXT NOT NULL, -- 'above' or 'below'
  trigger_type TEXT NOT NULL, -- 'one_time' or 'continuous'
  previous_state TEXT CHECK (previous_state IN ('above', 'below')), -- State before crossover
  triggered_at TIMESTAMP DEFAULT NOW(),
  notified BOOLEAN DEFAULT FALSE
);

-- Trend analysis results
CREATE TABLE trends (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  pair TEXT NOT NULL,
  trend_direction TEXT CHECK (trend_direction IN ('uptrend', 'downtrend', 'sideways')),
  trend_strength DECIMAL,
  calculated_at TIMESTAMP DEFAULT NOW()
);

-- Create indexes for better performance
CREATE INDEX idx_price_levels_active ON price_levels(is_active);
CREATE INDEX idx_price_levels_pair ON price_levels(pair);
CREATE INDEX idx_alerts_triggered_at ON alerts(triggered_at);
CREATE INDEX idx_alerts_notified ON alerts(notified);
CREATE INDEX idx_alerts_price_level_id ON alerts(price_level_id);
CREATE INDEX idx_trends_pair ON trends(pair);
CREATE INDEX idx_trends_calculated_at ON trends(calculated_at);


CREATE POLICY "Allow all operations on price_levels" ON price_levels
    FOR ALL USING (true);

CREATE POLICY "Allow all operations on alerts" ON alerts
    FOR ALL USING (true);

CREATE POLICY "Allow all operations on trends" ON trends
    FOR ALL USING (true);