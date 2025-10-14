-- Crypto Trading System - Unified Database Schema
-- Run this SQL in your Supabase SQL Editor to create all required tables
-- This schema combines price alerts, trading strategies, and system monitoring

-- ============================================
-- PRICE LEVELS TABLE (from schema.sql)
-- Basic price alerts for monitoring specific levels
-- ============================================
CREATE TABLE IF NOT EXISTS price_levels (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    pair TEXT NOT NULL, -- e.g., 'BTCUSDT'
    target_price DECIMAL(20,8) NOT NULL,
    trigger_direction TEXT CHECK (trigger_direction IN ('above', 'below')) DEFAULT 'above',
    is_active BOOLEAN DEFAULT TRUE, -- Enable/disable this alert
    trigger_type TEXT CHECK (trigger_type IN ('one_time', 'continuous')) DEFAULT 'one_time',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    last_triggered_at TIMESTAMPTZ -- Track when last triggered for continuous alerts
);

-- ============================================
-- TRADING ALERTS TABLE (enhanced from database_schema.sql)
-- Stores all generated trading alerts with strategy details
-- ============================================
CREATE TABLE IF NOT EXISTS trading_alerts (
    id BIGSERIAL PRIMARY KEY,
    alert_id VARCHAR(100) UNIQUE NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    symbol VARCHAR(20) NOT NULL,
    strategy_name VARCHAR(50) NOT NULL, -- 'bollinger_bands', 'ema_crossover', etc.
    signal_type VARCHAR(20) NOT NULL, -- 'BUY', 'SELL', 'REVERSAL', 'BREAKOUT'
    confidence DECIMAL(5,2) NOT NULL,
    price DECIMAL(20,8) NOT NULL,
    entry_price DECIMAL(20,8) NOT NULL,
    stop_loss DECIMAL(20,8) NOT NULL,
    take_profit DECIMAL(20,8) NOT NULL,
    risk_reward_ratio DECIMAL(10,2) NOT NULL,
    position_size DECIMAL(10,4) NOT NULL,
    priority VARCHAR(20) NOT NULL, -- 'CRITICAL', 'HIGH', 'MEDIUM', 'LOW'
    reasoning TEXT,
    timeframe VARCHAR(10) DEFAULT '15m',
    was_profitable BOOLEAN,
    profit_loss DECIMAL(20,8),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================
-- TREND ANALYSIS TABLE (enhanced from schema.sql)
-- Stores trend analysis results from technical indicators
-- ============================================
CREATE TABLE IF NOT EXISTS trend_analysis (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    pair TEXT NOT NULL,
    timeframe VARCHAR(10) NOT NULL DEFAULT '4h',
    trend_direction TEXT CHECK (trend_direction IN ('uptrend', 'downtrend', 'sideways')),
    trend_strength DECIMAL(5,2), -- ADX value
    adx_value DECIMAL(5,2),
    di_plus DECIMAL(5,2),
    di_minus DECIMAL(5,2),
    ema_signal VARCHAR(20), -- 'bullish', 'bearish', 'neutral'
    macd_signal VARCHAR(20), -- 'bullish', 'bearish', 'neutral'
    rsi_value DECIMAL(5,2),
    bollinger_position VARCHAR(20), -- 'upper', 'middle', 'lower'
    calculated_at TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================
-- USER PREFERENCES TABLE (from database_schema.sql)
-- Stores user settings and watchlists
-- ============================================
CREATE TABLE IF NOT EXISTS user_preferences (
    id BIGSERIAL PRIMARY KEY,
    user_id BIGINT UNIQUE NOT NULL,
    chat_id BIGINT NOT NULL,
    watchlist TEXT[] DEFAULT ARRAY['BTCUSDT', 'ETHUSDT'],
    min_confidence INTEGER DEFAULT 70,
    alert_priorities TEXT[] DEFAULT ARRAY['CRITICAL', 'HIGH'],
    timezone VARCHAR(50) DEFAULT 'UTC',
    risk_tolerance VARCHAR(20) DEFAULT 'medium',
    notifications_enabled BOOLEAN DEFAULT TRUE,
    quiet_hours_start INTEGER, -- Hour (0-23)
    quiet_hours_end INTEGER,   -- Hour (0-23)
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================
-- ALERT RULES TABLE (from database_schema.sql)
-- Stores custom alert rules defined by users
-- ============================================
CREATE TABLE IF NOT EXISTS alert_rules (
    id BIGSERIAL PRIMARY KEY,
    rule_id VARCHAR(100) UNIQUE NOT NULL,
    user_id BIGINT NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    strategy_name VARCHAR(50), -- Optional: specific strategy
    conditions JSONB NOT NULL, -- Flexible JSON for various conditions
    enabled BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    FOREIGN KEY (user_id) REFERENCES user_preferences(user_id) ON DELETE CASCADE
);

-- ============================================
-- USER ALERT TRACKING TABLE (from database_schema.sql)
-- Tracks which alerts were sent to which users
-- ============================================
CREATE TABLE IF NOT EXISTS user_alert_tracking (
    id BIGSERIAL PRIMARY KEY,
    user_id BIGINT NOT NULL,
    alert_id VARCHAR(100) NOT NULL,
    sent_at TIMESTAMPTZ DEFAULT NOW(),
    action VARCHAR(20), -- 'TRACKED', 'IGNORED', 'VIEWED'
    action_at TIMESTAMPTZ,
    FOREIGN KEY (user_id) REFERENCES user_preferences(user_id) ON DELETE CASCADE,
    FOREIGN KEY (alert_id) REFERENCES trading_alerts(alert_id) ON DELETE CASCADE
);

-- ============================================
-- SYSTEM HEALTH TABLE (from database_schema.sql)
-- Tracks system health and monitoring metrics
-- ============================================
CREATE TABLE IF NOT EXISTS system_health (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    status VARCHAR(20) NOT NULL, -- 'HEALTHY', 'WARNING', 'ERROR'
    active_users INTEGER DEFAULT 0,
    monitored_pairs INTEGER DEFAULT 0,
    alerts_sent INTEGER DEFAULT 0,
    consecutive_failures INTEGER DEFAULT 0,
    last_successful_run TIMESTAMPTZ,
    error_message TEXT,
    metrics JSONB -- Additional metrics in JSON format
);

-- ============================================
-- ALERT PERFORMANCE TABLE (from database_schema.sql)
-- Tracks actual performance of alerts for analysis
-- ============================================
CREATE TABLE IF NOT EXISTS alert_performance (
    id BIGSERIAL PRIMARY KEY,
    alert_id VARCHAR(100) NOT NULL,
    user_id BIGINT,
    entry_time TIMESTAMPTZ,
    exit_time TIMESTAMPTZ,
    entry_price DECIMAL(20,8),
    exit_price DECIMAL(20,8),
    profit_loss DECIMAL(20,8),
    profit_loss_percent DECIMAL(10,2),
    was_profitable BOOLEAN,
    exit_reason VARCHAR(50), -- 'TAKE_PROFIT', 'STOP_LOSS', 'MANUAL', 'TIMEOUT'
    created_at TIMESTAMPTZ DEFAULT NOW(),
    FOREIGN KEY (alert_id) REFERENCES trading_alerts(alert_id) ON DELETE CASCADE
);

-- ============================================
-- BACKTEST RESULTS TABLE (new)
-- Stores backtesting results for strategy analysis
-- ============================================
CREATE TABLE IF NOT EXISTS backtest_results (
    id BIGSERIAL PRIMARY KEY,
    strategy_name VARCHAR(50) NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    start_date TIMESTAMPTZ NOT NULL,
    end_date TIMESTAMPTZ NOT NULL,
    initial_capital DECIMAL(20,8) NOT NULL,
    final_capital DECIMAL(20,8) NOT NULL,
    total_return DECIMAL(10,4) NOT NULL,
    total_trades INTEGER NOT NULL,
    winning_trades INTEGER NOT NULL,
    losing_trades INTEGER NOT NULL,
    win_rate DECIMAL(5,2) NOT NULL,
    max_drawdown DECIMAL(10,4) NOT NULL,
    sharpe_ratio DECIMAL(10,4),
    profit_factor DECIMAL(10,4),
    avg_risk_reward DECIMAL(10,4),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================
-- INDEXES FOR PERFORMANCE
-- ============================================

-- Price levels indexes
CREATE INDEX IF NOT EXISTS idx_price_levels_active ON price_levels(is_active);
CREATE INDEX IF NOT EXISTS idx_price_levels_pair ON price_levels(pair);
CREATE INDEX IF NOT EXISTS idx_price_levels_target_price ON price_levels(target_price);

-- Trading alerts indexes
CREATE INDEX IF NOT EXISTS idx_trading_alerts_symbol ON trading_alerts(symbol);
CREATE INDEX IF NOT EXISTS idx_trading_alerts_timestamp ON trading_alerts(timestamp);
CREATE INDEX IF NOT EXISTS idx_trading_alerts_priority ON trading_alerts(priority);
CREATE INDEX IF NOT EXISTS idx_trading_alerts_signal_type ON trading_alerts(signal_type);
CREATE INDEX IF NOT EXISTS idx_trading_alerts_strategy ON trading_alerts(strategy_name);
CREATE INDEX IF NOT EXISTS idx_trading_alerts_confidence ON trading_alerts(confidence);

-- Trend analysis indexes
CREATE INDEX IF NOT EXISTS idx_trend_analysis_pair ON trend_analysis(pair);
CREATE INDEX IF NOT EXISTS idx_trend_analysis_timeframe ON trend_analysis(timeframe);
CREATE INDEX IF NOT EXISTS idx_trend_analysis_calculated_at ON trend_analysis(calculated_at);
CREATE INDEX IF NOT EXISTS idx_trend_analysis_direction ON trend_analysis(trend_direction);

-- User preferences indexes
CREATE INDEX IF NOT EXISTS idx_user_prefs_user_id ON user_preferences(user_id);
CREATE INDEX IF NOT EXISTS idx_user_prefs_notifications ON user_preferences(notifications_enabled);

-- Alert rules indexes
CREATE INDEX IF NOT EXISTS idx_alert_rules_user_id ON alert_rules(user_id);
CREATE INDEX IF NOT EXISTS idx_alert_rules_symbol ON alert_rules(symbol);
CREATE INDEX IF NOT EXISTS idx_alert_rules_enabled ON alert_rules(enabled);

-- User alert tracking indexes
CREATE INDEX IF NOT EXISTS idx_user_alert_tracking_user_id ON user_alert_tracking(user_id);
CREATE INDEX IF NOT EXISTS idx_user_alert_tracking_alert_id ON user_alert_tracking(alert_id);

-- System health indexes
CREATE INDEX IF NOT EXISTS idx_system_health_timestamp ON system_health(timestamp);
CREATE INDEX IF NOT EXISTS idx_system_health_status ON system_health(status);

-- Alert performance indexes
CREATE INDEX IF NOT EXISTS idx_alert_perf_alert_id ON alert_performance(alert_id);
CREATE INDEX IF NOT EXISTS idx_alert_perf_user_id ON alert_performance(user_id);
CREATE INDEX IF NOT EXISTS idx_alert_perf_profitable ON alert_performance(was_profitable);

-- Backtest results indexes
CREATE INDEX IF NOT EXISTS idx_backtest_strategy ON backtest_results(strategy_name);
CREATE INDEX IF NOT EXISTS idx_backtest_symbol ON backtest_results(symbol);
CREATE INDEX IF NOT EXISTS idx_backtest_timeframe ON backtest_results(timeframe);
CREATE INDEX IF NOT EXISTS idx_backtest_created_at ON backtest_results(created_at);

-- ============================================
-- FUNCTIONS AND TRIGGERS
-- ============================================

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Triggers for updated_at
CREATE TRIGGER update_trading_alerts_updated_at
    BEFORE UPDATE ON trading_alerts
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_user_preferences_updated_at
    BEFORE UPDATE ON user_preferences
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_alert_rules_updated_at
    BEFORE UPDATE ON alert_rules
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- ============================================
-- ROW LEVEL SECURITY (RLS)
-- ============================================

-- Enable RLS on tables
ALTER TABLE price_levels ENABLE ROW LEVEL SECURITY;
ALTER TABLE trading_alerts ENABLE ROW LEVEL SECURITY;
ALTER TABLE trend_analysis ENABLE ROW LEVEL SECURITY;
ALTER TABLE user_preferences ENABLE ROW LEVEL SECURITY;
ALTER TABLE alert_rules ENABLE ROW LEVEL SECURITY;
ALTER TABLE user_alert_tracking ENABLE ROW LEVEL SECURITY;
ALTER TABLE system_health ENABLE ROW LEVEL SECURITY;
ALTER TABLE alert_performance ENABLE ROW LEVEL SECURITY;
ALTER TABLE backtest_results ENABLE ROW LEVEL SECURITY;

-- Policies for service role (full access)
CREATE POLICY "Service role has full access to price_levels"
    ON price_levels FOR ALL
    USING (true)
    WITH CHECK (true);

CREATE POLICY "Service role has full access to trading_alerts"
    ON trading_alerts FOR ALL
    USING (true)
    WITH CHECK (true);

CREATE POLICY "Service role has full access to trend_analysis"
    ON trend_analysis FOR ALL
    USING (true)
    WITH CHECK (true);

CREATE POLICY "Service role has full access to user_preferences"
    ON user_preferences FOR ALL
    USING (true)
    WITH CHECK (true);

CREATE POLICY "Service role has full access to alert_rules"
    ON alert_rules FOR ALL
    USING (true)
    WITH CHECK (true);

CREATE POLICY "Service role has full access to user_alert_tracking"
    ON user_alert_tracking FOR ALL
    USING (true)
    WITH CHECK (true);

CREATE POLICY "Service role has full access to system_health"
    ON system_health FOR ALL
    USING (true)
    WITH CHECK (true);

CREATE POLICY "Service role has full access to alert_performance"
    ON alert_performance FOR ALL
    USING (true)
    WITH CHECK (true);

CREATE POLICY "Service role has full access to backtest_results"
    ON backtest_results FOR ALL
    USING (true)
    WITH CHECK (true);

-- ============================================
-- VIEWS FOR ANALYTICS
-- ============================================

-- View for trading alert statistics
CREATE OR REPLACE VIEW trading_alert_statistics AS
SELECT 
    symbol,
    strategy_name,
    signal_type,
    priority,
    COUNT(*) as total_alerts,
    COUNT(CASE WHEN was_profitable = TRUE THEN 1 END) as profitable_alerts,
    ROUND(AVG(confidence), 2) as avg_confidence,
    ROUND(AVG(risk_reward_ratio), 2) as avg_risk_reward,
    ROUND(AVG(profit_loss), 4) as avg_profit_loss,
    ROUND(
        (COUNT(CASE WHEN was_profitable = TRUE THEN 1 END)::DECIMAL / 
         NULLIF(COUNT(CASE WHEN was_profitable IS NOT NULL THEN 1 END), 0) * 100), 
        2
    ) as win_rate
FROM trading_alerts
WHERE timestamp >= NOW() - INTERVAL '30 days'
GROUP BY symbol, strategy_name, signal_type, priority
ORDER BY total_alerts DESC;

-- View for strategy performance comparison
CREATE OR REPLACE VIEW strategy_performance AS
SELECT 
    strategy_name,
    COUNT(*) as total_backtests,
    ROUND(AVG(total_return), 4) as avg_return,
    ROUND(AVG(win_rate), 2) as avg_win_rate,
    ROUND(AVG(max_drawdown), 4) as avg_max_drawdown,
    ROUND(AVG(sharpe_ratio), 4) as avg_sharpe_ratio,
    ROUND(AVG(profit_factor), 4) as avg_profit_factor
FROM backtest_results
WHERE created_at >= NOW() - INTERVAL '90 days'
GROUP BY strategy_name
ORDER BY avg_return DESC;

-- View for user activity
CREATE OR REPLACE VIEW user_activity AS
SELECT 
    up.user_id,
    up.chat_id,
    ARRAY_LENGTH(up.watchlist, 1) as watchlist_count,
    up.min_confidence,
    up.notifications_enabled,
    COUNT(DISTINCT uat.alert_id) as alerts_received,
    COUNT(CASE WHEN uat.action = 'TRACKED' THEN 1 END) as alerts_tracked,
    up.created_at as user_since
FROM user_preferences up
LEFT JOIN user_alert_tracking uat ON up.user_id = uat.user_id
GROUP BY up.user_id, up.chat_id, up.watchlist, up.min_confidence, 
         up.notifications_enabled, up.created_at;

-- ============================================
-- TABLE COMMENTS
-- ============================================

COMMENT ON TABLE price_levels IS 'Basic price level alerts for monitoring specific price points';
COMMENT ON TABLE trading_alerts IS 'Advanced trading alerts generated by strategies with full trade details';
COMMENT ON TABLE trend_analysis IS 'Technical analysis results including ADX, EMA, MACD, RSI, Bollinger Bands';
COMMENT ON TABLE user_preferences IS 'User settings, watchlists, and notification preferences';
COMMENT ON TABLE alert_rules IS 'Custom alert rules defined by users';
COMMENT ON TABLE user_alert_tracking IS 'Tracks which alerts were sent to which users';
COMMENT ON TABLE system_health IS 'System health monitoring and metrics';
COMMENT ON TABLE alert_performance IS 'Actual performance tracking of alerts';
COMMENT ON TABLE backtest_results IS 'Backtesting results for strategy performance analysis';

-- ============================================
-- SAMPLE DATA (Optional - for testing)
-- ============================================

-- Insert sample user preferences
-- INSERT INTO user_preferences (user_id, chat_id, watchlist, min_confidence)
-- VALUES 
--     (123456789, 123456789, ARRAY['BTCUSDT', 'ETHUSDT', 'BNBUSDT'], 70),
--     (987654321, 987654321, ARRAY['BTCUSDT', 'SOLUSDT'], 75);

-- ============================================
-- MAINTENANCE QUERIES (commented out)
-- ============================================

-- Delete old alerts (older than 90 days)
-- DELETE FROM trading_alerts WHERE timestamp < NOW() - INTERVAL '90 days';

-- Delete old system health records (older than 30 days)
-- DELETE FROM system_health WHERE timestamp < NOW() - INTERVAL '30 days';

-- Vacuum tables to reclaim space
-- VACUUM ANALYZE trading_alerts;
-- VACUUM ANALYZE user_preferences;
-- VACUUM ANALYZE alert_rules;
