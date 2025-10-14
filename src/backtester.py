import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Optional, Tuple, Any
from decimal import Decimal
import json
from dataclasses import dataclass, asdict
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from trend_analysis import TrendAnalyzer
from dotenv import load_dotenv

load_dotenv()

BINANCE_API_URL = os.getenv("BINANCE_API_URL", "https://api.binance.com/api/v3")


@dataclass
class Trade:
    """Represents a single trade"""
    entry_time: datetime
    exit_time: Optional[datetime]
    symbol: str
    direction: str  # 'LONG' or 'SHORT'
    entry_price: float
    exit_price: Optional[float]
    stop_loss: float
    take_profit: float
    position_size: float
    confidence: float
    pnl: Optional[float]
    pnl_percent: Optional[float]
    fees: float
    slippage: float
    status: str  # 'OPEN', 'WIN', 'LOSS', 'STOPPED'
    exit_reason: Optional[str]
    
    def to_dict(self):
        """Convert to dictionary for CSV export"""
        return asdict(self)


@dataclass
class BacktestConfig:
    """Configuration for backtesting"""
    symbol: str
    interval: str = "15m"
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    initial_capital: float = 10000.0
    risk_per_trade: float = 0.01  # 1% risk per trade
    min_confidence: float = 70.0
    slippage: float = 0.001  # 0.1%
    trading_fee: float = 0.001  # 0.1%
    max_drawdown_stop: float = 0.20  # Stop trading after 20% drawdown
    max_open_trades: int = 1
    use_stop_loss: bool = True
    use_take_profit: bool = True


class BacktestEngine:
    """
    Comprehensive backtesting engine for crypto trading strategies
    """
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.analyzer = TrendAnalyzer()
        self.trades: List[Trade] = []
        self.equity_curve: List[Dict] = []
        self.current_capital = config.initial_capital
        self.peak_capital = config.initial_capital
        self.open_trades: List[Trade] = []
        self.historical_data: Optional[pd.DataFrame] = None
        
    def fetch_historical_data(self, limit: int = 1000) -> pd.DataFrame:
        """
        Fetch historical OHLCV data from Binance API
        """
        print(f"Fetching historical data for {self.config.symbol}...")
        
        all_data = []
        end_time = None
        
        # Calculate how many requests we need
        total_candles_needed = limit
        candles_per_request = 1000
        
        while len(all_data) < total_candles_needed:
            try:
                url = f"{BINANCE_API_URL}/klines"
                params = {
                    "symbol": self.config.symbol,
                    "interval": self.config.interval,
                    "limit": min(candles_per_request, total_candles_needed - len(all_data))
                }
                
                if end_time:
                    params["endTime"] = end_time
                
                response = requests.get(url, params=params, timeout=10)
                response.raise_for_status()
                data = response.json()
                
                if not data:
                    break
                
                all_data = data + all_data
                end_time = data[0][0] - 1  # Get earlier data
                
                print(f"Fetched {len(all_data)} candles...")
                
                if len(data) < candles_per_request:
                    break
                    
            except Exception as e:
                print(f"Error fetching data: {e}")
                break
        
        # Convert to DataFrame
        df = pd.DataFrame(all_data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])
        
        # Convert types
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['open'] = df['open'].astype(float)
        df['high'] = df['high'].astype(float)
        df['low'] = df['low'].astype(float)
        df['close'] = df['close'].astype(float)
        df['volume'] = df['volume'].astype(float)
        
        # Filter by date range if specified
        if self.config.start_date:
            start = pd.to_datetime(self.config.start_date)
            df = df[df['timestamp'] >= start]
        
        if self.config.end_date:
            end = pd.to_datetime(self.config.end_date)
            df = df[df['timestamp'] <= end]
        
        df = df.reset_index(drop=True)
        
        print(f"Loaded {len(df)} candles from {df['timestamp'].min()} to {df['timestamp'].max()}")
        
        self.historical_data = df
        return df

    def calculate_position_size(self, entry_price: float, stop_loss: float) -> float:
        """
        Calculate position size based on 1% risk rule
        position_size = (account_risk) / (entry_price - stop_loss)
        """
        risk_amount = self.current_capital * self.config.risk_per_trade
        price_risk = abs(entry_price - stop_loss)

        if price_risk == 0:
            return 0

        position_size = risk_amount / price_risk

        # Ensure we don't use more than available capital
        max_position = self.current_capital * 0.95  # Leave 5% buffer
        position_value = position_size * entry_price

        if position_value > max_position:
            position_size = max_position / entry_price

        return position_size

    def apply_slippage(self, price: float, direction: str) -> float:
        """
        Apply slippage to simulate real execution
        """
        if direction == 'LONG':
            # Buy at slightly higher price
            return price * (1 + self.config.slippage)
        else:
            # Sell at slightly lower price
            return price * (1 - self.config.slippage)

    def calculate_fees(self, position_value: float) -> float:
        """
        Calculate trading fees
        """
        return position_value * self.config.trading_fee

    def open_trade(self, signal: Dict, current_time: datetime, current_price: float) -> Optional[Trade]:
        """
        Open a new trade based on signal
        """
        # Check if we can open more trades
        if len(self.open_trades) >= self.config.max_open_trades:
            return None

        # Check confidence threshold
        confidence = signal.get('confidence_score', 0)
        if confidence < self.config.min_confidence:
            return None

        # Determine direction
        action = signal.get('action', 'HOLD')
        if action not in ['STRONG_BUY', 'BUY', 'STRONG_SELL', 'SELL']:
            return None

        direction = 'LONG' if action in ['STRONG_BUY', 'BUY'] else 'SHORT'

        # Get stop loss and take profit from signal
        support_resistance = signal.get('support_resistance', {})

        if direction == 'LONG':
            stop_loss = support_resistance.get('nearest_support', current_price * 0.97)
            take_profit = support_resistance.get('nearest_resistance', current_price * 1.05)
        else:
            stop_loss = support_resistance.get('nearest_resistance', current_price * 1.03)
            take_profit = support_resistance.get('nearest_support', current_price * 0.95)

        # Apply slippage to entry
        entry_price = self.apply_slippage(current_price, direction)

        # Calculate position size
        position_size = self.calculate_position_size(entry_price, stop_loss)

        if position_size <= 0:
            return None

        # Calculate entry fees
        position_value = position_size * entry_price
        entry_fees = self.calculate_fees(position_value)

        # Check if we have enough capital
        total_cost = position_value + entry_fees
        if total_cost > self.current_capital:
            return None

        # Create trade
        trade = Trade(
            entry_time=current_time,
            exit_time=None,
            symbol=self.config.symbol,
            direction=direction,
            entry_price=entry_price,
            exit_price=None,
            stop_loss=stop_loss,
            take_profit=take_profit,
            position_size=position_size,
            confidence=confidence,
            pnl=None,
            pnl_percent=None,
            fees=entry_fees,
            slippage=abs(entry_price - current_price),
            status='OPEN',
            exit_reason=None
        )

        # Deduct capital
        self.current_capital -= total_cost
        self.open_trades.append(trade)

        return trade

    def close_trade(self, trade: Trade, exit_price: float, exit_time: datetime, reason: str):
        """
        Close an open trade
        """
        # Apply slippage to exit
        exit_price_with_slippage = self.apply_slippage(
            exit_price,
            'SHORT' if trade.direction == 'LONG' else 'LONG'
        )

        # Calculate P&L
        if trade.direction == 'LONG':
            pnl = (exit_price_with_slippage - trade.entry_price) * trade.position_size
        else:
            pnl = (trade.entry_price - exit_price_with_slippage) * trade.position_size

        # Calculate exit fees
        exit_value = trade.position_size * exit_price_with_slippage
        exit_fees = self.calculate_fees(exit_value)

        # Net P&L after fees
        net_pnl = pnl - trade.fees - exit_fees
        pnl_percent = (net_pnl / (trade.position_size * trade.entry_price)) * 100

        # Update trade
        trade.exit_time = exit_time
        trade.exit_price = exit_price_with_slippage
        trade.pnl = net_pnl
        trade.pnl_percent = pnl_percent
        trade.fees += exit_fees
        trade.slippage += abs(exit_price_with_slippage - exit_price)
        trade.exit_reason = reason

        if net_pnl > 0:
            trade.status = 'WIN'
        else:
            trade.status = 'LOSS'

        # Return capital
        self.current_capital += (trade.position_size * trade.entry_price) + net_pnl + trade.fees

        # Remove from open trades
        self.open_trades.remove(trade)
        self.trades.append(trade)

        # Update peak capital
        if self.current_capital > self.peak_capital:
            self.peak_capital = self.current_capital

    def check_stop_loss_take_profit(self, trade: Trade, high: float, low: float,
                                     close: float, current_time: datetime):
        """
        Check if stop loss or take profit was hit
        """
        if trade.direction == 'LONG':
            # Check stop loss
            if self.config.use_stop_loss and low <= trade.stop_loss:
                self.close_trade(trade, trade.stop_loss, current_time, 'STOP_LOSS')
                return True

            # Check take profit
            if self.config.use_take_profit and high >= trade.take_profit:
                self.close_trade(trade, trade.take_profit, current_time, 'TAKE_PROFIT')
                return True

        else:  # SHORT
            # Check stop loss
            if self.config.use_stop_loss and high >= trade.stop_loss:
                self.close_trade(trade, trade.stop_loss, current_time, 'STOP_LOSS')
                return True

            # Check take profit
            if self.config.use_take_profit and low <= trade.take_profit:
                self.close_trade(trade, trade.take_profit, current_time, 'TAKE_PROFIT')
                return True

        return False

    def run_backtest(self, lookback_period: int = 100) -> Dict:
        """
        Run the backtest simulation
        Avoids look-ahead bias by only using data available at each point
        """
        if self.historical_data is None:
            raise ValueError("No historical data loaded. Call fetch_historical_data() first.")

        df = self.historical_data
        print(f"\nStarting backtest for {self.config.symbol}...")
        print(f"Period: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"Initial Capital: ${self.config.initial_capital:,.2f}")
        print(f"Risk per trade: {self.config.risk_per_trade * 100}%")
        print(f"Min Confidence: {self.config.min_confidence}%")
        print("-" * 80)

        # Need enough data for indicators
        start_idx = max(lookback_period, 100)

        for i in range(start_idx, len(df)):
            current_row = df.iloc[i]
            current_time = current_row['timestamp']
            current_price = current_row['close']
            high = current_row['high']
            low = current_row['low']

            # Check drawdown stop
            current_drawdown = (self.peak_capital - self.current_capital) / self.peak_capital
            if current_drawdown >= self.config.max_drawdown_stop:
                print(f"\n⚠️  Maximum drawdown of {self.config.max_drawdown_stop*100}% reached. Stopping backtest.")
                break

            # Check existing trades for stop loss / take profit
            for trade in self.open_trades[:]:  # Use slice to avoid modification during iteration
                self.check_stop_loss_take_profit(trade, high, low, current_price, current_time)

            # Get historical data up to current point (avoid look-ahead bias)
            historical_slice = df.iloc[max(0, i - lookback_period):i + 1]

            # Prepare data for analyzer
            closes = historical_slice['close'].tolist()
            highs = historical_slice['high'].tolist()
            lows = historical_slice['low'].tolist()
            volumes = historical_slice['volume'].tolist()

            # Run trend analysis (this simulates real-time analysis)
            try:
                # We need to manually run the analysis with available data
                signal = self._analyze_with_available_data(closes, highs, lows, volumes)

                if signal:
                    # Try to open trade
                    trade = self.open_trade(signal, current_time, current_price)
                    if trade:
                        print(f"[{current_time}] Opened {trade.direction} trade at ${trade.entry_price:.2f} "
                              f"(Confidence: {trade.confidence:.1f}%, Size: {trade.position_size:.4f})")

            except Exception as e:
                # Skip this candle if analysis fails
                pass

            # Record equity
            total_equity = self.current_capital
            for trade in self.open_trades:
                if trade.direction == 'LONG':
                    unrealized_pnl = (current_price - trade.entry_price) * trade.position_size
                else:
                    unrealized_pnl = (trade.entry_price - current_price) * trade.position_size
                total_equity += (trade.position_size * trade.entry_price) + unrealized_pnl

            self.equity_curve.append({
                'timestamp': current_time,
                'equity': total_equity,
                'cash': self.current_capital,
                'open_trades': len(self.open_trades)
            })

            # Progress update
            if i % 100 == 0:
                progress = (i - start_idx) / (len(df) - start_idx) * 100
                print(f"Progress: {progress:.1f}% | Equity: ${total_equity:,.2f} | "
                      f"Trades: {len(self.trades)} | Open: {len(self.open_trades)}")

        # Close any remaining open trades at final price
        final_price = df.iloc[-1]['close']
        final_time = df.iloc[-1]['timestamp']
        for trade in self.open_trades[:]:
            self.close_trade(trade, final_price, final_time, 'END_OF_BACKTEST')

        print("\n" + "=" * 80)
        print("\n" + "=" * 80)
        print("Backtest completed!")
        print("=" * 80)

        return self.calculate_performance_metrics()

    def _analyze_with_available_data(self, closes: List[float], highs: List[float],
                                     lows: List[float], volumes: List[float]) -> Optional[Dict]:
        """
        Run trend analysis with available data (avoiding look-ahead bias)
        """
        if len(closes) < 50:
            return None

        try:
            # Calculate indicators manually to avoid look-ahead bias
            # Use the TrendAnalyzer methods
            trend_direction, details = self.analyzer.calculate_trend_direction(closes, highs, lows)
            trend_strength, strength_details = self.analyzer.calculate_trend_strength(highs, lows, closes)

            # Get additional metrics
            rsi = self.analyzer.calculate_rsi(closes, period=14) if len(closes) >= 14 else 50
            volume_trend, volume_strength = self.analyzer.calculate_volume_trend(volumes, period=20) if len(volumes) >= 20 else ('neutral', 0)

            # Calculate support/resistance
            support_resistance = self.analyzer.find_support_resistance(highs, lows, closes, num_levels=3)

            # Determine action based on trend
            adx = details.get('adx', 0)
            macd_signal = details.get('macd_signal', 'neutral')

            # Simple signal logic
            action = 'HOLD'
            confidence = 50.0

            if trend_direction == 'uptrend' and adx > 25:
                if macd_signal == 'bullish' and rsi < 70:
                    action = 'STRONG_BUY'
                    confidence = min(95, 60 + (adx - 25) * 0.5 + (10 if volume_trend == 'increasing' else 0))
                elif rsi < 65:
                    action = 'BUY'
                    confidence = min(85, 55 + (adx - 25) * 0.4)

            elif trend_direction == 'downtrend' and adx > 25:
                if macd_signal == 'bearish' and rsi > 30:
                    action = 'STRONG_SELL'
                    confidence = min(95, 60 + (adx - 25) * 0.5 + (10 if volume_trend == 'increasing' else 0))
                elif rsi > 35:
                    action = 'SELL'
                    confidence = min(85, 55 + (adx - 25) * 0.4)

            return {
                'action': action,
                'confidence_score': confidence,
                'trend_direction': trend_direction,
                'trend_strength': float(trend_strength),
                'adx': adx,
                'rsi': rsi,
                'macd_signal': macd_signal,
                'volume_trend': volume_trend,
                'support_resistance': support_resistance
            }

        except Exception as e:
            # Log the exception and return a structured error so callers can handle it gracefully.
            # This helps avoid silent failures (returning None) and provides useful debugging info.
            try:
                import logging
                import traceback
                logging.exception("Error computing indicators/features: %s", e)
                tb = traceback.format_exc()
            except Exception:
                # If logging or traceback import fails for any reason, fall back to minimal info.
                tb = None

            # Store last error info on the engine instance for external inspection / tests
            try:
                self.last_exception = e
                self.last_traceback = tb
            except Exception:
                # If self is not available or setting attributes fails, ignore silently.
                pass

            # Return a structured error object rather than plain None so calling code can decide
            # whether to abort, skip, or attempt recovery. This maintains visibility into failures.
            return {
                'error': 'indicator_calculation_failed',
                'exception': str(e),
                'traceback': tb
            }

    def calculate_performance_metrics(self) -> Dict:
        """
        Calculate comprehensive performance metrics
        """
        if not self.trades:
            return {
                'error': 'No trades executed',
                'total_trades': 0
            }

        # Convert equity curve to DataFrame
        equity_df = pd.DataFrame(self.equity_curve)

        # Basic metrics
        total_trades = len(self.trades)
        winning_trades = [t for t in self.trades if t.status == 'WIN']
        losing_trades = [t for t in self.trades if t.status == 'LOSS']

        wins = len(winning_trades)
        losses = len(losing_trades)
        win_rate = (wins / total_trades * 100) if total_trades > 0 else 0

        # P&L metrics
        total_pnl = sum(t.pnl for t in self.trades if t.pnl is not None)
        gross_profit = sum(t.pnl for t in winning_trades if t.pnl is not None)
        gross_loss = abs(sum(t.pnl for t in losing_trades if t.pnl is not None))

        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float('inf')

        avg_win = (gross_profit / wins) if wins > 0 else 0
        avg_loss = (gross_loss / losses) if losses > 0 else 0

        # Expectancy
        loss_rate = (losses / total_trades) if total_trades > 0 else 0
        expectancy = (win_rate/100 * avg_win) - (loss_rate * avg_loss)

        # Returns
        initial_capital = self.config.initial_capital
        final_equity = equity_df['equity'].iloc[-1]
        total_return = ((final_equity - initial_capital) / initial_capital) * 100

        # Calculate returns for Sharpe ratio
        equity_df['returns'] = equity_df['equity'].pct_change()
        returns = equity_df['returns'].dropna()

        # Sharpe Ratio (annualized)
        if len(returns) > 0 and returns.std() > 0:
            mean_return = returns.mean()
            std_return = returns.std()
            # Annualize based on interval
            periods_per_year = self._get_periods_per_year()
            sharpe_ratio = (mean_return / std_return) * np.sqrt(periods_per_year)
        else:
            sharpe_ratio = 0

        # Maximum Drawdown
        equity_df['cummax'] = equity_df['equity'].cummax()
        equity_df['drawdown'] = (equity_df['equity'] - equity_df['cummax']) / equity_df['cummax']
        max_drawdown = equity_df['drawdown'].min() * 100

        # Drawdown duration
        drawdown_periods = (equity_df['drawdown'] < 0).astype(int)
        drawdown_groups = (drawdown_periods != drawdown_periods.shift()).cumsum()
        drawdown_durations = drawdown_periods.groupby(drawdown_groups).sum()
        max_drawdown_duration = drawdown_durations.max() if len(drawdown_durations) > 0 else 0

        # Win/Loss streaks
        win_streak = 0
        loss_streak = 0
        current_win_streak = 0
        current_loss_streak = 0

        for trade in self.trades:
            if trade.status == 'WIN':
                current_win_streak += 1
                current_loss_streak = 0
                win_streak = max(win_streak, current_win_streak)
            elif trade.status == 'LOSS':
                current_loss_streak += 1
                current_win_streak = 0
                loss_streak = max(loss_streak, current_loss_streak)

        # Trade duration
        trade_durations = []
        for trade in self.trades:
            if trade.exit_time and trade.entry_time:
                duration = (trade.exit_time - trade.entry_time).total_seconds() / 3600  # hours
                trade_durations.append(duration)

        avg_trade_duration = np.mean(trade_durations) if trade_durations else 0

        # Fees and slippage
        total_fees = sum((t.fees or 0) for t in self.trades)
        total_slippage = sum((t.slippage or 0) * (t.position_size or 0) for t in self.trades)

        # Buy and Hold comparison
        first_price = self.historical_data['close'].iloc[0]
        last_price = self.historical_data['close'].iloc[-1]
        buy_hold_return = ((last_price - first_price) / first_price) * 100

        metrics = {
            'total_trades': total_trades,
            'winning_trades': wins,
            'losing_trades': losses,
            'win_rate': round(win_rate, 2),
            'total_return': round(total_return, 2),
            'total_pnl': round(total_pnl, 2),
            'gross_profit': round(gross_profit, 2),
            'gross_loss': round(gross_loss, 2),
            'profit_factor': round(profit_factor, 2),
            'avg_win': round(avg_win, 2),
            'avg_loss': round(avg_loss, 2),
            'expectancy': round(expectancy, 2),
            'sharpe_ratio': round(sharpe_ratio, 2),
            'max_drawdown': round(max_drawdown, 2),
            'max_drawdown_duration': int(max_drawdown_duration),
            'longest_win_streak': win_streak,
            'longest_loss_streak': loss_streak,
            'avg_trade_duration_hours': round(avg_trade_duration, 2),
            'total_fees': round(total_fees, 2),
            'total_slippage': round(total_slippage, 2),
            'initial_capital': initial_capital,
            'final_equity': round(final_equity, 2),
            'buy_hold_return': round(buy_hold_return, 2),
            'alpha': round(total_return - buy_hold_return, 2)
        }

        return metrics

    def _get_periods_per_year(self) -> int:
        """
        Get number of periods per year based on interval
        """
        interval_map = {
            '1m': 525600,
            '5m': 105120,
            '15m': 35040,
            '30m': 17520,
            '1h': 8760,
            '4h': 2190,
            '1d': 365,
            '1w': 52
        }
        return interval_map.get(self.config.interval, 35040)

    def save_trades_to_csv(self, filename: str = None):
        """
        Save all trades to CSV file
        """
        if not getattr(self, "trades", None):
            print("No trades to save")
            return

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"backtest_trades_{self.config.symbol}_{timestamp}.csv"

        trades_data = [t.to_dict() for t in self.trades]
        df = pd.DataFrame(trades_data)

        # Create results directory if it doesn't exist
        Path("backtest_results").mkdir(exist_ok=True)
        filepath = Path("backtest_results") / filename

        try:
            df.to_csv(filepath, index=False)
            print(f"✅ Trades saved to {filepath}")
        except Exception as e:
            print(f"❌ Failed to save trades to CSV: {e}")
            return None

        return filepath

    def plot_equity_curve(self, save_path: str = None):
        """
        Plot equity curve and drawdown
        """
        if not getattr(self, "equity_curve", None):
            print("No equity data to plot")
            return

        equity_df = pd.DataFrame(self.equity_curve)
        if equity_df.empty:
            print("Equity data is empty")
            return

        # Ensure timestamp column is datetime
        if not pd.api.types.is_datetime64_any_dtype(equity_df['timestamp']):
            equity_df['timestamp'] = pd.to_datetime(equity_df['timestamp'])

        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

        # Plot equity curve
        ax1.plot(equity_df['timestamp'], equity_df['equity'], label='Strategy Equity', linewidth=2, color='#2E86AB')
        ax1.axhline(y=self.config.initial_capital, color='gray', linestyle='--', label='Initial Capital', alpha=0.7)

        # Add buy & hold comparison
        if getattr(self, "historical_data", None) is not None and not self.historical_data.empty:
            hist = self.historical_data.copy()
            if not pd.api.types.is_datetime64_any_dtype(hist['timestamp']):
                hist['timestamp'] = pd.to_datetime(hist['timestamp'])
                self.historical_data = hist  # update stored data

            first_price = hist['close'].iloc[0]
            buy_hold_equity = []
            for _, row in equity_df.iterrows():
                # find the closest historical index up to the equity timestamp
                valid_idx = hist[hist['timestamp'] <= row['timestamp']].index
                if len(valid_idx) == 0:
                    current_price = first_price
                else:
                    idx = valid_idx[-1]
                    current_price = hist.loc[idx, 'close']
                bh_equity = self.config.initial_capital * (current_price / first_price)
                buy_hold_equity.append(bh_equity)

            ax1.plot(equity_df['timestamp'], buy_hold_equity, label='Buy & Hold',
                     linewidth=2, color='#A23B72', alpha=0.7, linestyle='--')

        ax1.set_ylabel('Equity ($)', fontsize=12, fontweight='bold')
        ax1.set_title(f'Backtest Results - {self.config.symbol} ({self.config.interval})',
                      fontsize=14, fontweight='bold')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

        # Plot drawdown
        equity_df['cummax'] = equity_df['equity'].cummax()
        equity_df['drawdown'] = (equity_df['equity'] - equity_df['cummax']) / equity_df['cummax'] * 100

        ax2.fill_between(equity_df['timestamp'], equity_df['drawdown'], 0,
                         color='#F18F01', alpha=0.6, label='Drawdown')
        ax2.set_ylabel('Drawdown (%)', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Date', fontsize=12, fontweight='bold')
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)

        # Format x-axis
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=45)

        plt.tight_layout()

        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = f"backtest_results/equity_curve_{self.config.symbol}_{timestamp}.png"

        Path("backtest_results").mkdir(exist_ok=True)
        try:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Equity curve saved to {save_path}")
        except Exception as e:
            print(f"❌ Failed to save equity curve: {e}")

        plt.close()
        return save_path

    def print_performance_report(self, metrics: Dict):
        """
        Print comprehensive performance report
        """
        print("\n" + "=" * 80)
        print("BACKTEST PERFORMANCE REPORT".center(80))
        print("=" * 80)

        print(f"\n📊 TRADING STATISTICS")
        print("-" * 80)
        print(f"Total Trades:              {metrics.get('total_trades', 0)}")
        print(f"Winning Trades:            {metrics.get('winning_trades', 0)} ({metrics.get('win_rate', 0):.2f}%)")
        print(f"Losing Trades:             {metrics.get('losing_trades', 0)}")
        print(f"Longest Win Streak:        {metrics.get('longest_win_streak', 0)}")
        print(f"Longest Loss Streak:       {metrics.get('longest_loss_streak', 0)}")
        print(f"Avg Trade Duration:        {metrics.get('avg_trade_duration_hours', 0):.2f} hours")

        print(f"\n💰 PROFITABILITY")
        print("-" * 80)
        print(f"Initial Capital:           ${metrics.get('initial_capital', 0):,.2f}")
        print(f"Final Equity:              ${metrics.get('final_equity', 0):,.2f}")
        print(f"Total Return:              {metrics.get('total_return', 0):.2f}%")
        print(f"Total P&L:                 ${metrics.get('total_pnl', 0):,.2f}")
        print(f"Gross Profit:              ${metrics.get('gross_profit', 0):,.2f}")
        print(f"Gross Loss:                ${metrics.get('gross_loss', 0):,.2f}")
        print(f"Profit Factor:             {metrics.get('profit_factor', 0):.2f}")
        print(f"Average Win:               ${metrics.get('avg_win', 0):,.2f}")
        print(f"Average Loss:              ${metrics.get('avg_loss', 0):,.2f}")
        print(f"Expectancy:                ${metrics.get('expectancy', 0):,.2f}")

        print(f"\n📈 RISK METRICS")
        print("-" * 80)
        print(f"Sharpe Ratio:              {metrics.get('sharpe_ratio', 0):.2f}")
        print(f"Maximum Drawdown:          {metrics.get('max_drawdown', 0):.2f}%")
        print(f"Max Drawdown Duration:     {metrics.get('max_drawdown_duration', 0)} periods")

        print(f"\n💸 COSTS")
        print("-" * 80)
        print(f"Total Fees:                ${metrics.get('total_fees', 0):,.2f}")
        print(f"Total Slippage:            ${metrics.get('total_slippage', 0):,.2f}")

        print(f"\n🎯 BENCHMARK COMPARISON")
        print("-" * 80)
        print(f"Buy & Hold Return:         {metrics.get('buy_hold_return', 0):.2f}%")
        print(f"Alpha (vs Buy & Hold):     {metrics.get('alpha', 0):.2f}%")

        print("\n" + "=" * 80)

    def calculate_monthly_returns(self) -> pd.DataFrame:
        """Calculate monthly returns breakdown from the engine's equity_curve.

        Returns a DataFrame with columns:
        - month (YYYY-MM)
        - monthly_return_pct
        - start_equity
        - end_equity
        - cumulative_return_pct (since start of series)
        """
        if not getattr(self, "equity_curve", None):
            return pd.DataFrame()

        equity_df = pd.DataFrame(self.equity_curve).copy()
        if equity_df.empty:
            return pd.DataFrame()

        # Ensure timestamp column is datetime
        if not pd.api.types.is_datetime64_any_dtype(equity_df.get("timestamp")):
            equity_df["timestamp"] = pd.to_datetime(equity_df["timestamp"])

        equity_df = equity_df.sort_values("timestamp").reset_index(drop=True)
        equity_df["month"] = equity_df["timestamp"].dt.to_period("M").astype(str)

        monthly = equity_df.groupby("month").agg(
            start_equity=("equity", "first"),
            end_equity=("equity", "last")
        ).reset_index()

        # Avoid divide-by-zero and compute returns
        monthly["monthly_return_pct"] = monthly.apply(
            lambda row: ((row["end_equity"] - row["start_equity"]) / row["start_equity"] * 100)
            if row["start_equity"] not in (0, None) else float("nan"),
            axis=1
        )

        # Cumulative return relative to the first month start
        first_start = monthly["start_equity"].iloc[0] if not monthly.empty else None
        monthly["cumulative_return_pct"] = monthly["end_equity"].apply(
            lambda v: ((v - first_start) / first_start * 100) if first_start not in (0, None) else float("nan")
        )

        # Round numeric columns for readability
        monthly[["start_equity", "end_equity", "monthly_return_pct", "cumulative_return_pct"]] = (
            monthly[["start_equity", "end_equity", "monthly_return_pct", "cumulative_return_pct"]]
            .round(2)
        )

        return monthly[["month", "monthly_return_pct", "start_equity", "end_equity", "cumulative_return_pct"]]


class WalkForwardAnalyzer:
    """Implements walk-forward analysis to help assess overfitting.

    Two modes supported:
    - Single split (train_ratio): behaves like a simple in-sample / out-of-sample split.
    - Rolling walk-forward (train_window & test_window provided): performs multiple
      train/test iterations across the time series.

    Example usage:
        wfa = WalkForwardAnalyzer(config)
        results = wfa.run_walk_forward(total_candles=2000, rolling=False)
        # or rolling
        results = wfa.run_walk_forward(df=df, train_window=1000, test_window=250, step=250)
    """

    def __init__(self, config: BacktestConfig, train_ratio: float = 0.7):
        self.config = config
        self.train_ratio = train_ratio

    def run_walk_forward(
        self,
        total_candles: int = 2000,
        df: pd.DataFrame = None,
        rolling: bool = False,
        train_window: int = None,
        test_window: int = None,
        step: int = None
    ) -> Dict:
        """
        Run walk-forward analysis.

        If rolling is False (default) the method will perform a single train/test split
        based on train_ratio using up to total_candles of historical data.

        If rolling is True, df, train_window and test_window must be provided. The method
        will perform multiple rolling train/test evaluations advancing by `step`.
        """
        print("\n" + "=" * 80)
        print("WALK-FORWARD ANALYSIS".center(80))
        print("=" * 80)

        # Fetch data if not provided
        engine = BacktestEngine(self.config)
        if df is None:
            df = engine.fetch_historical_data(limit=total_candles)

        if df is None or df.empty:
            raise ValueError("No historical data available for walk-forward analysis.")

        results = []

        if not rolling:
            # Single train/test split
            split_idx = int(len(df) * self.train_ratio)
            train_df = df.iloc[:split_idx].copy()
            test_df = df.iloc[split_idx:].copy()

            print(f"\nTrain Period: {train_df['timestamp'].min()} to {train_df['timestamp'].max()}")
            print(f"Test Period:  {test_df['timestamp'].min()} to {test_df['timestamp'].max()}")

            # Run backtest on train data
            print("\nTRAINING PHASE...")
            train_engine = BacktestEngine(self.config)
            train_engine.historical_data = train_df
            train_metrics = train_engine.run_backtest()

            # Run backtest on test data (out-of-sample)
            print("\nTESTING PHASE (Out-of-Sample)...")
            test_engine = BacktestEngine(self.config)
            test_engine.historical_data = test_df
            test_metrics = test_engine.run_backtest()

            print(f"\n{'Metric':<30} {'Train':<20} {'Test':<20}")
            print("-" * 70)
            for metric in ['total_return', 'win_rate', 'sharpe_ratio', 'max_drawdown']:
                print(f"{metric:<30} {train_metrics.get(metric, 0):<20} {test_metrics.get(metric, 0):<20}")

            return {'type': 'single_split', 'train_metrics': train_metrics, 'test_metrics': test_metrics}

        # Rolling walk-forward
        if not (isinstance(train_window, int) and isinstance(test_window, int) and isinstance(step, int)):
            raise ValueError("train_window, test_window and step (all ints) are required for rolling walk-forward.")

        idx = 0
        iteration = 0
        while idx + train_window + test_window <= len(df):
            iteration += 1
            train_df = df.iloc[idx: idx + train_window].copy()
            test_df = df.iloc[idx + train_window: idx + train_window + test_window].copy()

            print(f"\nIteration {iteration}: Train {train_df['timestamp'].min()} to {train_df['timestamp'].max()} | "
                  f"Test {test_df['timestamp'].min()} to {test_df['timestamp'].max()}")

            # Train
            train_engine = BacktestEngine(self.config)
            train_engine.historical_data = train_df
            train_metrics = train_engine.run_backtest()

            # Test
            test_engine = BacktestEngine(self.config)
            test_engine.historical_data = test_df
            test_metrics = test_engine.run_backtest()

            results.append({
                'iteration': iteration,
                'train_start': train_df['timestamp'].min(),
                'train_end': train_df['timestamp'].max(),
                'test_start': test_df['timestamp'].min(),
                'test_end': test_df['timestamp'].max(),
                'train_metrics': train_metrics,
                'test_metrics': test_metrics
            })

            idx += step

        # Summarize rolling results
        print("\n" + "=" * 80)
        print("ROLLING WALK-FORWARD SUMMARY".center(80))
        print("=" * 80)
        summary_rows = []
        for r in results:
            tm = r['train_metrics']
            om = r['test_metrics']
            summary_rows.append({
                'iteration': r['iteration'],
                'test_start': r['test_start'],
                'test_end': r['test_end'],
                'test_return': om.get('total_return'),
                'test_sharpe': om.get('sharpe_ratio'),
                'test_max_drawdown': om.get('max_drawdown')
            })

        summary_df = pd.DataFrame(summary_rows)
        if not summary_df.empty:
            print(summary_df.to_string(index=False))

        return {'type': 'rolling', 'results': results, 'summary': summary_df}


class ParameterOptimizer:
    """Grid search parameter optimization for BacktestConfig.

    Usage:
        optimizer = ParameterOptimizer(base_config)
        results_df = optimizer.optimize(param_grid, total_candles=1500, scoring_metric='total_return')
    """

    def __init__(self, base_config: BacktestConfig):
        self.base_config = base_config
        self.results: List[Dict] = []

    def optimize(
        self,
        param_grid: Dict[str, List],
        total_candles: int = 1500,
        scoring_metric: str = 'total_return',
        maximize: bool = True
    ) -> pd.DataFrame:
        """Run grid search over param_grid.

        param_grid is a dict where keys are BacktestConfig attribute names and values are lists of values.
        scoring_metric chooses which metric to sort results by.
        """
        from itertools import product

        print("\n" + "=" * 80)
        print("PARAMETER OPTIMIZATION".center(80))
        print("=" * 80)

        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        combinations = list(product(*param_values))

        print(f"Testing {len(combinations)} parameter combinations...")

        for idx, params in enumerate(combinations, start=1):
            # Create a shallow copy of base_config with overridden params
            config = BacktestConfig(
                symbol=self.base_config.symbol,
                interval=self.base_config.interval,
                initial_capital=self.base_config.initial_capital
            )

            for param_name, param_value in zip(param_names, params):
                setattr(config, param_name, param_value)

            print(f"\n[{idx}/{len(combinations)}] Testing: {dict(zip(param_names, params))}")

            try:
                engine = BacktestEngine(config)
                # Preload historical data for this config
                engine.fetch_historical_data(limit=total_candles)
                metrics = engine.run_backtest()

                result = {**dict(zip(param_names, params)), **metrics}
                self.results.append(result)

                print(f"   Return: {metrics.get('total_return', 0):.2f}% | Sharpe: {metrics.get('sharpe_ratio', 0):.2f}")
            except Exception as e:
                print(f"   ❌ Error: {e}")

        results_df = pd.DataFrame(self.results)
        if results_df.empty:
            print("No results were produced by the optimizer.")
            return results_df

        # Sort by chosen scoring metric
        results_df = results_df.sort_values(by=scoring_metric, ascending=not maximize).reset_index(drop=True)

        print("\n" + "=" * 80)
        print("TOP 5 RESULTS".center(80))
        print("=" * 80)
        print(results_df.head(5).to_string(index=False))

        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"backtest_results/optimization_{self.base_config.symbol}_{timestamp}.csv"
        Path("backtest_results").mkdir(exist_ok=True)
        results_df.to_csv(filename, index=False)
        print(f"\n✅ Results saved to {filename}")

        return results_df
