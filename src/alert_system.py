"""
Automated Cryptocurrency Alert System

This module provides a comprehensive alert system that monitors cryptocurrency pairs,
detects high-confidence trading setups, and sends real-time notifications via Telegram.

Features:
- Continuous monitoring with 15-minute intervals (aligned with candle closes)
- High-confidence setup detection (confidence_score ≥ 70)
- Telegram notifications with trade details
- Alert throttling to prevent spam
- Custom alert rules engine
- Alert history tracking and performance analysis
- Multi-user support with preferences
- Health checks and fault tolerance
"""

import os
import sys
import asyncio
import logging
import time
import traceback
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import json

# Third-party imports
import pandas as pd
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    CallbackQueryHandler,
    ContextTypes,
    MessageHandler,
    filters
)
from telegram.constants import ParseMode
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from dotenv import load_dotenv

# Local imports
from trend_analysis import TrendAnalyzer
from supabase import create_client, Client

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('alert_system.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Supabase client
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY) if SUPABASE_URL and SUPABASE_KEY else None

# Telegram Bot Token
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

# Alert Configuration
DEFAULT_CONFIDENCE_THRESHOLD = 70
DEFAULT_ALERT_COOLDOWN = 3600  # 1 hour in seconds
MAX_ALERTS_PER_SYMBOL_PER_HOUR = 1
MONITORING_INTERVAL_MINUTES = 15

# Priority levels
class AlertPriority:
    CRITICAL = "CRITICAL"  # 80+ confidence
    HIGH = "HIGH"          # 70-79 confidence
    MEDIUM = "MEDIUM"      # 60-69 confidence
    LOW = "LOW"            # < 60 confidence


@dataclass
class AlertRule:
    """Custom alert rule definition"""
    rule_id: str
    user_id: int
    symbol: str
    conditions: Dict  # e.g., {"min_confidence": 80, "timeframes_aligned": True}
    enabled: bool = True
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class UserPreferences:
    """User preferences for alerts"""
    user_id: int
    chat_id: int
    watchlist: List[str] = field(default_factory=list)
    min_confidence: int = DEFAULT_CONFIDENCE_THRESHOLD
    alert_priorities: List[str] = field(default_factory=lambda: [AlertPriority.CRITICAL, AlertPriority.HIGH])
    timezone: str = "UTC"
    risk_tolerance: str = "medium"  # low, medium, high
    notifications_enabled: bool = True
    quiet_hours_start: Optional[int] = None  # Hour (0-23)
    quiet_hours_end: Optional[int] = None
    

@dataclass
class Alert:
    """Alert data structure"""
    alert_id: str
    timestamp: datetime
    symbol: str
    signal_type: str  # 'BUY', 'SELL', 'REVERSAL', 'BREAKOUT'
    confidence: float
    price: float
    entry_price: float
    stop_loss: float
    take_profit: float
    risk_reward_ratio: float
    priority: str
    reasoning: str
    timeframe: str = "15m"
    was_profitable: Optional[bool] = None
    profit_loss: Optional[float] = None
    

class AlertThrottler:
    """Manages alert throttling to prevent spam"""
    
    def __init__(self, cooldown_seconds: int = DEFAULT_ALERT_COOLDOWN):
        self.cooldown_seconds = cooldown_seconds
        self.last_alert_times: Dict[str, datetime] = {}
        
    def can_send_alert(self, symbol: str) -> bool:
        """Check if enough time has passed since last alert for this symbol"""
        if symbol not in self.last_alert_times:
            return True
            
        time_since_last = datetime.now() - self.last_alert_times[symbol]
        return time_since_last.total_seconds() >= self.cooldown_seconds
        
    def record_alert(self, symbol: str):
        """Record that an alert was sent for this symbol"""
        self.last_alert_times[symbol] = datetime.now
        
    def cleanup_old_entries(self):
        """Remove entries older than cooldown period"""
        cutoff_time = datetime.now() - timedelta(seconds=self.cooldown_seconds * 2)
        self.last_alert_times = {
            symbol: timestamp 
            for symbol, timestamp in self.last_alert_times.items()
            if timestamp > cutoff_time
        }


class AlertRulesEngine:
    """Evaluates custom alert rules"""
    
    def __init__(self):
        self.rules: Dict[int, List[AlertRule]] = defaultdict(list)
        
    def add_rule(self, rule: AlertRule):
        """Add a custom alert rule for a user"""
        self.rules[rule.user_id].append(rule)
        logger.info(f"Added rule {rule.rule_id} for user {rule.user_id}")
        
    def remove_rule(self, user_id: int, rule_id: str):
        """Remove a rule"""
        self.rules[user_id] = [r for r in self.rules[user_id] if r.rule_id != rule_id]
        
    def evaluate_rules(self, user_id: int, signal: Dict) -> bool:
        """Evaluate if signal matches any user rules"""
        if user_id not in self.rules:
            return False
            
        for rule in self.rules[user_id]:
            if not rule.enabled:
                continue
                
            if rule.symbol != signal.get('symbol'):
                continue
                
            # Evaluate conditions
            if self._check_conditions(rule.conditions, signal):
                return True
                
        return False
        
    def _check_conditions(self, conditions: Dict, signal: Dict) -> bool:
        """Check if signal meets all conditions"""
        for key, value in conditions.items():
            if key == "min_confidence":
                if signal.get('confidence', 0) < value:
                    return False
            elif key == "timeframes_aligned":
                if value and not signal.get('timeframes_aligned', False):
                    return False
            elif key == "volume_increasing":
                if value and not signal.get('volume_increasing', False):
                    return False
            elif key == "min_risk_reward":
                if signal.get('risk_reward_ratio', 0) < value:
                    return False
            elif key == "rsi_extreme":
                rsi = signal.get('rsi', 50)
                if value and not (rsi < 30 or rsi > 70):
                    return False
                    
        return True


class AlertDatabase:
    """Manages alert history and user preferences in Supabase"""
    
    def __init__(self, supabase_client: Client):
        self.supabase = supabase_client
        
    async def save_alert(self, alert: Alert) -> bool:
        """Save alert to database"""
        try:
            data = {
                'alert_id': alert.alert_id,
                'timestamp': alert.timestamp.isoformat(),
                'symbol': alert.symbol,
                'signal_type': alert.signal_type,
                'confidence': alert.confidence,
                'price': alert.price,
                'entry_price': alert.entry_price,
                'stop_loss': alert.stop_loss,
                'take_profit': alert.take_profit,
                'risk_reward_ratio': alert.risk_reward_ratio,
                'priority': alert.priority,
                'reasoning': alert.reasoning,
                'timeframe': alert.timeframe,
                'was_profitable': alert.was_profitable,
                'profit_loss': alert.profit_loss
            }
            
            result = self.supabase.table('alerts').insert(data).execute()
            logger.info(f"Saved alert {alert.alert_id} to database")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save alert to database: {e}")
            return False
            
    async def get_user_preferences(self, user_id: int) -> Optional[UserPreferences]:
        """Get user preferences from database"""
        try:
            result = self.supabase.table('user_preferences').select('*').eq('user_id', user_id).execute()
            
            if result.data and len(result.data) > 0:
                data = result.data[0]
                return UserPreferences(
                    user_id=data['user_id'],
                    chat_id=data['chat_id'],
                    watchlist=data.get('watchlist', []),
                    min_confidence=data.get('min_confidence', DEFAULT_CONFIDENCE_THRESHOLD),
                    alert_priorities=data.get('alert_priorities', [AlertPriority.CRITICAL, AlertPriority.HIGH]),
                    timezone=data.get('timezone', 'UTC'),
                    risk_tolerance=data.get('risk_tolerance', 'medium'),
                    notifications_enabled=data.get('notifications_enabled', True),
                    quiet_hours_start=data.get('quiet_hours_start'),
                    quiet_hours_end=data.get('quiet_hours_end')
                )
            return None
            
        except Exception as e:
            logger.error(f"Failed to get user preferences: {e}")
            return None
            
    async def save_user_preferences(self, prefs: UserPreferences) -> bool:
        """Save user preferences to database"""
        try:
            data = {
                'user_id': prefs.user_id,
                'chat_id': prefs.chat_id,
                'watchlist': prefs.watchlist,
                'min_confidence': prefs.min_confidence,
                'alert_priorities': prefs.alert_priorities,
                'timezone': prefs.timezone,
                'risk_tolerance': prefs.risk_tolerance,
                'notifications_enabled': prefs.notifications_enabled,
                'quiet_hours_start': prefs.quiet_hours_start,
                'quiet_hours_end': prefs.quiet_hours_end
            }
            
            # Upsert (insert or update)
            result = self.supabase.table('user_preferences').upsert(data).execute()
            logger.info(f"Saved preferences for user {prefs.user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save user preferences: {e}")
            return False
            
    async def get_alert_stats(self, user_id: int, days: int = 30) -> Dict:
        """Get alert performance statistics"""
        try:
            cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
            
            result = self.supabase.table('alerts').select('*').gte('timestamp', cutoff_date).execute()
            
            if not result.data:
                return {
                    'total_alerts': 0,
                    'profitable_alerts': 0,
                    'win_rate': 0,
                    'avg_profit': 0,
                    'best_setup': None
                }
                
            alerts = result.data
            total = len(alerts)
            profitable = sum(1 for a in alerts if a.get('was_profitable') is True)
            
            profits = [a.get('profit_loss', 0) for a in alerts if a.get('profit_loss') is not None]
            avg_profit = sum(profits) / len(profits) if profits else 0
            
            # Find best performing setup
            setup_performance = defaultdict(list)
            for alert in alerts:
                if alert.get('profit_loss') is not None:
                    key = f"{alert['signal_type']}_{alert['symbol']}"
                    setup_performance[key].append(alert['profit_loss'])
                    
            best_setup = None
            best_avg = float('-inf')
            for setup, profits in setup_performance.items():
                avg = sum(profits) / len(profits)
                if avg > best_avg:
                    best_avg = avg
                    best_setup = setup
                    
            return {
                'total_alerts': total,
                'profitable_alerts': profitable,
                'win_rate': (profitable / total * 100) if total > 0 else 0,
                'avg_profit': avg_profit,
                'best_setup': best_setup
            }
            
        except Exception as e:
            logger.error(f"Failed to get alert stats: {e}")
class TelegramNotifier:
    """Handles Telegram notifications with formatting and rate limiting"""

    def __init__(self, bot_token: str):
        self.bot_token = bot_token
        self.application = None

    async def initialize(self):
        """Initialize the Telegram bot application"""
        self.application = Application.builder().token(self.bot_token).build()

    def format_alert_message(self, alert: Alert, signal: Dict) -> str:
        """Format alert message with emojis and structure"""

        # Priority emoji
        priority_emoji = {
            AlertPriority.CRITICAL: "🔴",
            AlertPriority.HIGH: "🟠",
            AlertPriority.MEDIUM: "🟡",
            AlertPriority.LOW: "⚪"
        }

        # Signal type emoji
        signal_emoji = {
            'BUY': "📈",
            'SELL': "📉",
            'REVERSAL': "🔄",
            'BREAKOUT': "💥"
        }

        emoji = priority_emoji.get(alert.priority, "⚪")
        signal_icon = signal_emoji.get(alert.signal_type, "📊")

        # Build message
        message = f"{emoji} *{alert.priority} ALERT* {signal_icon}\n\n"
        message += f"*Symbol:* `{alert.symbol}`\n"
        message += f"*Signal:* {alert.signal_type}\n"
        message += f"*Confidence:* {alert.confidence:.1f}%\n"
        message += f"*Current Price:* ${alert.price:.4f}\n\n"

        message += f"📍 *Trade Setup:*\n"
        message += f"Entry: `${alert.entry_price:.4f}`\n"
        message += f"Stop Loss: `${alert.stop_loss:.4f}`\n"
        message += f"Take Profit: `${alert.take_profit:.4f}`\n"
        message += f"R:R Ratio: `{alert.risk_reward_ratio:.2f}`\n\n"

        # Additional indicators
        if signal.get('rsi'):
            message += f"RSI: {signal['rsi']:.1f}\n"
        if signal.get('adx'):
            message += f"ADX: {signal['adx']:.1f}\n"
        if signal.get('timeframes_aligned'):
            message += f"✅ Timeframes Aligned\n"
        if signal.get('volume_increasing'):
            message += f"📊 Volume Increasing\n"

        message += f"\n💡 *Reasoning:*\n{alert.reasoning}\n"
        message += f"\n⏰ {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}"

        return message

    async def send_alert(self, chat_id: int, alert: Alert, signal: Dict) -> bool:
        """Send alert to user via Telegram"""
        try:
            message = self.format_alert_message(alert, signal)

            # Add inline keyboard for quick actions
            keyboard = [
                [
                    InlineKeyboardButton("✅ Track", callback_data=f"track_{alert.alert_id}"),
                    InlineKeyboardButton("❌ Ignore", callback_data=f"ignore_{alert.alert_id}")
                ],
                [
                    InlineKeyboardButton("📊 Chart", callback_data=f"chart_{alert.symbol}")
                ]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)

            await self.application.bot.send_message(
                chat_id=chat_id,
                text=message,
                parse_mode=ParseMode.MARKDOWN,
                reply_markup=reply_markup
            )

            logger.info(f"Sent alert {alert.alert_id} to chat {chat_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to send Telegram alert: {e}")
            return False

    async def send_system_notification(self, chat_id: int, message: str):
        """Send system notification (errors, health checks, etc.)"""
        try:
            await self.application.bot.send_message(
                chat_id=chat_id,
                text=f"🔧 *System Notification*\n\n{message}",
                parse_mode=ParseMode.MARKDOWN
            )
        except Exception as e:
            logger.error(f"Failed to send system notification: {e}")


class AlertSystem:
    """Main alert system orchestrator"""

    def __init__(self):
        self.trend_analyzer = TrendAnalyzer()
        self.throttler = AlertThrottler()
        self.rules_engine = AlertRulesEngine()
        self.database = AlertDatabase(supabase) if supabase else None
        self.notifier = TelegramNotifier(TELEGRAM_BOT_TOKEN) if TELEGRAM_BOT_TOKEN else None
        self.scheduler = AsyncIOScheduler()

        # User management
        self.active_users: Dict[int, UserPreferences] = {}

        # Health monitoring
        self.last_successful_run: Optional[datetime] = None
        self.consecutive_failures = 0
        self.is_running = False

    async def initialize(self):
        """Initialize the alert system"""
        logger.info("Initializing Alert System...")

        # Initialize Telegram bot
        if self.notifier:
            await self.notifier.initialize()
            self._setup_bot_handlers()

        # Load active users from database
        await self._load_active_users()

        # Setup scheduler
        self._setup_scheduler()

        logger.info("Alert System initialized successfully")

    def _setup_scheduler(self):
        """Setup APScheduler for periodic monitoring"""
        # Run every 15 minutes at :00, :15, :30, :45
        self.scheduler.add_job(
            self.monitor_all_pairs,
            CronTrigger(minute='0,15,30,45'),
            id='monitor_pairs',
            name='Monitor cryptocurrency pairs',
            replace_existing=True
        )

        # Cleanup throttler every hour
        self.scheduler.add_job(
            self.throttler.cleanup_old_entries,
            CronTrigger(minute=0),
            id='cleanup_throttler',
            name='Cleanup alert throttler',
            replace_existing=True
        )

        # Health check every 5 minutes
        self.scheduler.add_job(
            self.health_check,
            CronTrigger(minute='*/5'),
            id='health_check',
            name='System health check',
            replace_existing=True
        )

    async def _load_active_users(self):
        """Load active users from database"""
        if not self.database:
            return

        try:
            result = supabase.table('user_preferences').select('*').eq('notifications_enabled', True).execute()

            for user_data in result.data:
                prefs = UserPreferences(
                    user_id=user_data['user_id'],
                    chat_id=user_data['chat_id'],
                    watchlist=user_data.get('watchlist', []),
                    min_confidence=user_data.get('min_confidence', DEFAULT_CONFIDENCE_THRESHOLD),
                    alert_priorities=user_data.get('alert_priorities', [AlertPriority.CRITICAL, AlertPriority.HIGH]),
                    timezone=user_data.get('timezone', 'UTC'),
                    risk_tolerance=user_data.get('risk_tolerance', 'medium'),
                    notifications_enabled=True,
                    quiet_hours_start=user_data.get('quiet_hours_start'),
                    quiet_hours_end=user_data.get('quiet_hours_end')
                )
                self.active_users[prefs.user_id] = prefs

            logger.info(f"Loaded {len(self.active_users)} active users")

        except Exception as e:
            logger.error(f"Failed to load active users: {e}")

    async def monitor_all_pairs(self):
        """Monitor all pairs in user watchlists"""
        logger.info("Starting monitoring cycle...")

        try:
            # Collect all unique symbols from all watchlists
            all_symbols = set()
            for user_prefs in self.active_users.values():
                all_symbols.update(user_prefs.watchlist)

            if not all_symbols:
                logger.info("No symbols to monitor")
                return

            logger.info(f"Monitoring {len(all_symbols)} symbols: {all_symbols}")

            # Monitor pairs concurrently
            tasks = [self.monitor_pair(symbol) for symbol in all_symbols]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results
            successful = sum(1 for r in results if not isinstance(r, Exception))
            failed = len(results) - successful

            logger.info(f"Monitoring cycle complete: {successful} successful, {failed} failed")

            self.last_successful_run = datetime.now()
            self.consecutive_failures = 0

        except Exception as e:
            logger.error(f"Error in monitoring cycle: {e}")
            logger.error(traceback.format_exc())
            self.consecutive_failures += 1

            # Alert admin if too many failures
            if self.consecutive_failures >= 3:
                await self._send_admin_alert(f"⚠️ System experiencing issues: {self.consecutive_failures} consecutive failures")

    async def monitor_pair(self, symbol: str) -> Optional[Dict]:
        """Monitor a single trading pair"""
        try:
            logger.info(f"Analyzing {symbol}...")

            # Run trend analysis
            signal = await asyncio.to_thread(
                self.trend_analyzer.analyze_trend_advanced,
                symbol,
                interval='15m'
            )

            if not signal:
                logger.debug(f"No signal generated for {symbol}")
                return None

            # Check if signal meets criteria
            confidence = signal.get('confidence', 0)

            if confidence < 60:  # Minimum threshold
                logger.debug(f"{symbol}: Confidence too low ({confidence:.1f})")
                return None

            # Check throttling
            if not self.throttler.can_send_alert(symbol):
                logger.debug(f"{symbol}: Alert throttled")
                return None

            # Determine priority
            priority = self._determine_priority(confidence)

            # Create alert
            alert = self._create_alert(symbol, signal, priority)

            # Save to database
            if self.database:
                await self.database.save_alert(alert)

            # Send to relevant users
            await self._send_alert_to_users(alert, signal)

            # Record alert sent
            self.throttler.record_alert(symbol)

            return signal

        except Exception as e:
            logger.error(f"Error monitoring {symbol}: {e}")
            logger.error(traceback.format_exc())
            return None

    def _determine_priority(self, confidence: float) -> str:
        """Determine alert priority based on confidence"""
        if confidence >= 80:
            return AlertPriority.CRITICAL
        elif confidence >= 70:
            return AlertPriority.HIGH
        elif confidence >= 60:
            return AlertPriority.MEDIUM
        else:
            return AlertPriority.LOW

    def _create_alert(self, symbol: str, signal: Dict, priority: str) -> Alert:
        """Create an Alert object from signal data"""
        alert_id = f"{symbol}_{int(datetime.now().timestamp())}"

        return Alert(
            alert_id=alert_id,
            timestamp=datetime.now(),
            symbol=symbol,
            signal_type=signal.get('direction', 'NEUTRAL').upper(),
            confidence=signal.get('confidence', 0),
            price=signal.get('current_price', 0),
            entry_price=signal.get('entry_price', signal.get('current_price', 0)),
            stop_loss=signal.get('stop_loss', 0),
            take_profit=signal.get('take_profit', 0),
            risk_reward_ratio=signal.get('risk_reward_ratio', 0),
            priority=priority,
            reasoning=self._generate_reasoning(signal),
            timeframe='15m'
        )

    def _generate_reasoning(self, signal: Dict) -> str:
        """Generate human-readable reasoning for the alert"""
        reasons = []

        if signal.get('timeframes_aligned'):
            reasons.append("Multiple timeframes aligned")

        adx = signal.get('adx', 0)
        if adx > 25:
            reasons.append(f"Strong trend (ADX: {adx:.1f})")

        rsi = signal.get('rsi', 50)
        if rsi < 30:
            reasons.append("Oversold conditions (RSI < 30)")
        elif rsi > 70:
            reasons.append("Overbought conditions (RSI > 70)")

        if signal.get('volume_increasing'):
            reasons.append("Increasing volume")

        rr = signal.get('risk_reward_ratio', 0)
        if rr >= 2:
            reasons.append(f"Favorable R:R ({rr:.2f})")

        return " | ".join(reasons) if reasons else "Technical setup detected"

    async def _send_alert_to_users(self, alert: Alert, signal: Dict):
        """Send alert to all relevant users"""
        for user_id, prefs in self.active_users.items():
            try:
                # Check if user is watching this symbol
                if alert.symbol not in prefs.watchlist:
                    continue

                # Check if notifications are enabled
                if not prefs.notifications_enabled:
                    continue

                # Check confidence threshold
                if alert.confidence < prefs.min_confidence:
                    continue

                # Check priority filter
                if alert.priority not in prefs.alert_priorities:
                    continue

                # Check quiet hours
                if self._is_quiet_hours(prefs):
                    continue

                # Check custom rules
                if self.rules_engine.evaluate_rules(user_id, signal):
                    # Send notification
                    if self.notifier:
                        await self.notifier.send_alert(prefs.chat_id, alert, signal)
                        logger.info(f"Sent alert to user {user_id}")

            except Exception as e:
                logger.error(f"Error sending alert to user {user_id}: {e}")

    def _is_quiet_hours(self, prefs: UserPreferences) -> bool:
        """Check if current time is within user's quiet hours"""
        if prefs.quiet_hours_start is None or prefs.quiet_hours_end is None:
            return False

        current_hour = datetime.now().hour

        if prefs.quiet_hours_start < prefs.quiet_hours_end:
            return prefs.quiet_hours_start <= current_hour < prefs.quiet_hours_end
        else:
            # Quiet hours span midnight
            return current_hour >= prefs.quiet_hours_start or current_hour < prefs.quiet_hours_end

    async def health_check(self):
        """Perform system health check"""
        try:
            # Check last successful run
            if self.last_successful_run:
                time_since_last = datetime.now() - self.last_successful_run
                if time_since_last.total_seconds() > 1800:  # 30 minutes
                    await self._send_admin_alert(f"⚠️ No successful monitoring cycle in {time_since_last.total_seconds() / 60:.0f} minutes")

            # Check consecutive failures
            if self.consecutive_failures >= 5:
                await self._send_admin_alert(f"🔴 CRITICAL: {self.consecutive_failures} consecutive failures")

            # Check database connection
            if self.database and supabase:
                try:
                    supabase.table('alerts').select('alert_id').limit(1).execute()
                except Exception as e:
                    await self._send_admin_alert(f"⚠️ Database connection issue: {e}")

            logger.info("Health check completed")

        except Exception as e:
            logger.error(f"Error in health check: {e}")

    async def _send_admin_alert(self, message: str):
        """Send alert to system administrator"""
        admin_chat_id = os.getenv("ADMIN_TELEGRAM_CHAT_ID")
        if admin_chat_id and self.notifier:
            try:
                await self.notifier.send_system_notification(int(admin_chat_id), message)
            except Exception as e:
                logger.error(f"Failed to send admin alert: {e}")

    def _setup_bot_handlers(self):
        """Setup Telegram bot command handlers"""
        if not self.notifier or not self.notifier.application:
            return

        app = self.notifier.application

        # Command handlers
        app.add_handler(CommandHandler("start", self.cmd_start))
        app.add_handler(CommandHandler("stop", self.cmd_stop))
        app.add_handler(CommandHandler("help", self.cmd_help))
        app.add_handler(CommandHandler("watchlist", self.cmd_watchlist))
        app.add_handler(CommandHandler("add", self.cmd_add_symbol))
        app.add_handler(CommandHandler("remove", self.cmd_remove_symbol))
        app.add_handler(CommandHandler("settings", self.cmd_settings))
        app.add_handler(CommandHandler("stats", self.cmd_stats))
        app.add_handler(CommandHandler("status", self.cmd_status))

        # Callback query handler for inline buttons
        app.add_handler(CallbackQueryHandler(self.handle_callback))

        logger.info("Bot handlers registered")

    async def cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command"""
        user_id = update.effective_user.id
        chat_id = update.effective_chat.id

        # Create default preferences
        prefs = UserPreferences(
            user_id=user_id,
            chat_id=chat_id,
            watchlist=['BTCUSDT', 'ETHUSDT'],  # Default watchlist
            notifications_enabled=True
        )

        # Save to database
        if self.database:
            await self.database.save_user_preferences(prefs)

        # Add to active users
        self.active_users[user_id] = prefs

        welcome_message = """
🚀 *Welcome to Crypto Alert System!*

I'll monitor cryptocurrency pairs and send you high-confidence trading alerts.

*Quick Start:*
• Your default watchlist: BTC, ETH
• Minimum confidence: 70%
• Alert priorities: CRITICAL, HIGH

*Available Commands:*
/watchlist - View your watchlist
/add SYMBOL - Add symbol to watchlist
/remove SYMBOL - Remove symbol
/settings - Configure preferences
/stats - View alert performance
/status - System status
/stop - Disable alerts
/help - Show this message

Let's start monitoring! 📊
        """

        await update.message.reply_text(welcome_message, parse_mode=ParseMode.MARKDOWN)
        logger.info(f"User {user_id} started the bot")

    async def cmd_stop(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /stop command"""
        user_id = update.effective_user.id

        if user_id in self.active_users:
            self.active_users[user_id].notifications_enabled = False

            # Update database
            if self.database:
                await self.database.save_user_preferences(self.active_users[user_id])

            await update.message.reply_text(
                "✅ Alerts disabled. Use /start to re-enable.",
                parse_mode=ParseMode.MARKDOWN
            )
            logger.info(f"User {user_id} stopped alerts")
        else:
            await update.message.reply_text("You don't have an active subscription.")

    async def cmd_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /help command"""
        help_text = """
📚 *Crypto Alert System - Help*

*Commands:*
/start - Start receiving alerts
/stop - Stop receiving alerts
/watchlist - View your watchlist
/add SYMBOL - Add symbol (e.g., /add BTCUSDT)
/remove SYMBOL - Remove symbol
/settings - Configure alert preferences
/stats - View alert performance
/status - Check system status

*Alert Priorities:*
🔴 CRITICAL - 80%+ confidence
🟠 HIGH - 70-79% confidence
🟡 MEDIUM - 60-69% confidence

*Understanding Alerts:*
Each alert includes:
• Entry price, stop loss, take profit
• Risk/reward ratio
• Confidence score
• Technical reasoning
• RSI, ADX indicators

*Tips:*
• Start with 2-3 pairs to avoid spam
• Set confidence threshold ≥ 70%
• Enable quiet hours for sleep
• Track alert performance with /stats

Need help? Contact @YourSupport
        """

        await update.message.reply_text(help_text, parse_mode=ParseMode.MARKDOWN)

    async def cmd_watchlist(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /watchlist command"""
        user_id = update.effective_user.id

        if user_id not in self.active_users:
            await update.message.reply_text("Please use /start first.")
            return

        prefs = self.active_users[user_id]

        if not prefs.watchlist:
            message = "📋 Your watchlist is empty.\n\nAdd symbols with: /add BTCUSDT"
        else:
            watchlist_str = "\n".join([f"• {symbol}" for symbol in prefs.watchlist])
            message = f"📋 *Your Watchlist* ({len(prefs.watchlist)} pairs)\n\n{watchlist_str}\n\n"
            message += f"Min Confidence: {prefs.min_confidence}%\n"
            message += f"Priorities: {', '.join(prefs.alert_priorities)}"

        await update.message.reply_text(message, parse_mode=ParseMode.MARKDOWN)

    async def cmd_add_symbol(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /add command"""
        user_id = update.effective_user.id

        if user_id not in self.active_users:
            await update.message.reply_text("Please use /start first.")
            return

        if not context.args:
            await update.message.reply_text("Usage: /add BTCUSDT")
            return

        symbol = context.args[0].upper()

        # Validate symbol format
        if not symbol.endswith('USDT'):
            await update.message.reply_text("⚠️ Symbol must end with USDT (e.g., BTCUSDT)")
            return

        prefs = self.active_users[user_id]

        if symbol in prefs.watchlist:
            await update.message.reply_text(f"✅ {symbol} is already in your watchlist.")
            return

        # Add symbol
        prefs.watchlist.append(symbol)

        # Save to database
        if self.database:
            await self.database.save_user_preferences(prefs)

        await update.message.reply_text(
            f"✅ Added {symbol} to your watchlist.\n\nTotal pairs: {len(prefs.watchlist)}",
            parse_mode=ParseMode.MARKDOWN
        )
        logger.info(f"User {user_id} added {symbol} to watchlist")

    async def cmd_remove_symbol(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /remove command"""
        user_id = update.effective_user.id

        if user_id not in self.active_users:
            await update.message.reply_text("Please use /start first.")
            return

        if not context.args:
            await update.message.reply_text("Usage: /remove BTCUSDT")
            return

        symbol = context.args[0].upper()
        prefs = self.active_users[user_id]

        if symbol not in prefs.watchlist:
            await update.message.reply_text(f"⚠️ {symbol} is not in your watchlist.")
            return

        # Remove symbol
        prefs.watchlist.remove(symbol)

        # Save to database
        if self.database:
            await self.database.save_user_preferences(prefs)

        await update.message.reply_text(
            f"✅ Removed {symbol} from your watchlist.\n\nRemaining pairs: {len(prefs.watchlist)}",
            parse_mode=ParseMode.MARKDOWN
        )
        logger.info(f"User {user_id} removed {symbol} from watchlist")

    async def cmd_settings(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /settings command"""
        user_id = update.effective_user.id

        if user_id not in self.active_users:
            await update.message.reply_text("Please use /start first.")
            return

        prefs = self.active_users[user_id]

        # Create inline keyboard for settings
        keyboard = [
            [
                InlineKeyboardButton("🎯 Confidence", callback_data="setting_confidence"),
                InlineKeyboardButton("🔔 Priorities", callback_data="setting_priorities")
            ],
            [
                InlineKeyboardButton("🌙 Quiet Hours", callback_data="setting_quiet"),
                InlineKeyboardButton("⚡ Risk Level", callback_data="setting_risk")
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)

        settings_text = f"""
⚙️ *Your Settings*

*Confidence Threshold:* {prefs.min_confidence}%
*Alert Priorities:* {', '.join(prefs.alert_priorities)}
*Risk Tolerance:* {prefs.risk_tolerance.upper()}
*Quiet Hours:* {'Not set' if prefs.quiet_hours_start is None else f'{prefs.quiet_hours_start}:00 - {prefs.quiet_hours_end}:00'}
*Notifications:* {'✅ Enabled' if prefs.notifications_enabled else '❌ Disabled'}

Select a setting to modify:
        """

        await update.message.reply_text(
            settings_text,
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=reply_markup
        )

    async def cmd_stats(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /stats command"""
        user_id = update.effective_user.id

        if user_id not in self.active_users:
            await update.message.reply_text("Please use /start first.")
            return

        if not self.database:
            await update.message.reply_text("⚠️ Database not available.")
            return

        # Get stats for last 30 days
        stats = await self.database.get_alert_stats(user_id, days=30)

        if stats.get('total_alerts', 0) == 0:
            await update.message.reply_text("📊 No alerts yet. Keep monitoring!")
            return

        stats_text = f"""
📊 *Alert Performance (Last 30 Days)*

*Total Alerts:* {stats['total_alerts']}
*Profitable:* {stats['profitable_alerts']}
*Win Rate:* {stats['win_rate']:.1f}%
*Avg Profit:* ${stats['avg_profit']:.2f}

*Best Setup:* {stats.get('best_setup', 'N/A')}

Keep tracking your alerts to improve performance! 📈
        """

        await update.message.reply_text(stats_text, parse_mode=ParseMode.MARKDOWN)

    async def cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /status command"""
        status_text = f"""
🔧 *System Status*

*Status:* {'🟢 Running' if self.is_running else '🔴 Stopped'}
*Last Check:* {self.last_successful_run.strftime('%H:%M:%S') if self.last_successful_run else 'Never'}
*Active Users:* {len(self.active_users)}
*Monitored Pairs:* {len(set(s for u in self.active_users.values() for s in u.watchlist))}
*Consecutive Failures:* {self.consecutive_failures}

*Next Check:* Every 15 minutes (:00, :15, :30, :45)

System is operational! ✅
        """

        await update.message.reply_text(status_text, parse_mode=ParseMode.MARKDOWN)

    async def handle_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle inline button callbacks"""
        query = update.callback_query
        await query.answer()

        data = query.data
        user_id = update.effective_user.id

        if data.startswith("track_"):
            alert_id = data.replace("track_", "")
            await query.edit_message_text(
                f"✅ Tracking alert {alert_id}\n\nI'll notify you when this trade closes.",
                parse_mode=ParseMode.MARKDOWN
            )

        elif data.startswith("ignore_"):
            alert_id = data.replace("ignore_", "")
            await query.edit_message_text(
                f"❌ Alert {alert_id} ignored.",
                parse_mode=ParseMode.MARKDOWN
            )

        elif data.startswith("chart_"):
            symbol = data.replace("chart_", "")
            await query.edit_message_text(
                f"📊 View {symbol} chart:\nhttps://www.tradingview.com/chart/?symbol=BINANCE:{symbol}",
                parse_mode=ParseMode.MARKDOWN
            )

        elif data.startswith("setting_"):
            setting = data.replace("setting_", "")
            await self._handle_setting_change(query, user_id, setting)

    async def _handle_setting_change(self, query, user_id: int, setting: str):
        """Handle setting change callbacks"""
        if user_id not in self.active_users:
            await query.edit_message_text("Please use /start first.")
            return

        prefs = self.active_users[user_id]

        if setting == "confidence":
            keyboard = [
                [InlineKeyboardButton("60%", callback_data="conf_60"),
                 InlineKeyboardButton("70%", callback_data="conf_70"),
                 InlineKeyboardButton("80%", callback_data="conf_80")]
            ]
            await query.edit_message_text(
                "Select minimum confidence threshold:",
                reply_markup=InlineKeyboardMarkup(keyboard)
            )

        elif setting == "priorities":
            await query.edit_message_text(
                "Priority settings coming soon! Currently: CRITICAL, HIGH"
            )

        elif setting == "quiet":
            await query.edit_message_text(
                "Quiet hours settings coming soon!"
            )

        elif setting == "risk":
            keyboard = [
                [InlineKeyboardButton("Low", callback_data="risk_low"),
                 InlineKeyboardButton("Medium", callback_data="risk_medium"),
                 InlineKeyboardButton("High", callback_data="risk_high")]
            ]
            await query.edit_message_text(
                "Select risk tolerance:",
                reply_markup=InlineKeyboardMarkup(keyboard)
            )

    async def start(self):
        """Start the alert system"""
        logger.info("Starting Alert System...")

        try:
            # Initialize
            await self.initialize()

            # Start scheduler
            self.scheduler.start()
            self.is_running = True

            # Start Telegram bot
            if self.notifier and self.notifier.application:
                await self.notifier.application.initialize()
                await self.notifier.application.start()
                await self.notifier.application.updater.start_polling()

            logger.info("✅ Alert System started successfully")
            logger.info(f"Monitoring {len(self.active_users)} users")

            # Keep running
            while self.is_running:
                await asyncio.sleep(1)

        except KeyboardInterrupt:
            logger.info("Received shutdown signal")
            await self.stop()

        except Exception as e:
            logger.error(f"Fatal error in alert system: {e}")
            logger.error(traceback.format_exc())
            await self.stop()
            raise

    async def stop(self):
        """Stop the alert system gracefully"""
        logger.info("Stopping Alert System...")

        self.is_running = False

        # Stop scheduler
        if self.scheduler.running:
            self.scheduler.shutdown(wait=True)

        # Stop Telegram bot
        if self.notifier and self.notifier.application:
            await self.notifier.application.updater.stop()
            await self.notifier.application.stop()
            await self.notifier.application.shutdown()

        logger.info("✅ Alert System stopped")


async def main():
    """Main entry point"""
    # Check required environment variables
    if not TELEGRAM_BOT_TOKEN:
        logger.error("TELEGRAM_BOT_TOKEN not set in environment")
        sys.exit(1)

    if not SUPABASE_URL or not SUPABASE_KEY:
        logger.warning("Supabase credentials not set - database features disabled")

    # Create and start alert system
    alert_system = AlertSystem()

    try:
        await alert_system.start()
    except KeyboardInterrupt:
        logger.info("Shutdown requested")
    finally:
        await alert_system.stop()


if __name__ == "__main__":
    print("🚀 Crypto Alert System")
    print("=" * 50)
    print("Starting automated monitoring...")
    print("Press Ctrl+C to stop")
    print("=" * 50)

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
