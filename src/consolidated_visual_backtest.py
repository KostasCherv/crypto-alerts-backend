"""
Consolidated Visual Backtesting System
====================================

This system creates a single comprehensive report image that includes all
backtesting information in one organized, professional layout.

Features:
- Single consolidated report image per backtest
- Multiple strategies comparison
- Clean, organized layout with grouped metrics
- Optional CSV/JSON/HTML export (disabled by default)
- Dedicated reports folder

Author: Professional Trading System
Version: 1.0
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from datetime import datetime, timedelta
import warnings
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict

warnings.filterwarnings('ignore')

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from strategies import create_strategy
from data_manager import get_data

@dataclass
class ConsolidatedBacktestResult:
    """Results from consolidated backtesting"""
    strategy_name: str
    asset: str
    timeframe: str
    start_date: datetime
    end_date: datetime
    initial_capital: float
    final_capital: float
    total_return: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    profit_factor: float
    sharpe_ratio: float
    max_drawdown: float
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float
    total_fees: float
    equity_curve: pd.DataFrame
    trades: List[Dict]
    signals: List[Dict]

class ConsolidatedVisualBacktest:
    """
    Consolidated visual backtesting system
    """
    
    def __init__(self):
        self.results = []
        # Create session-based folder structure
        session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.reports_dir = f"results/consolidated/session_{session_timestamp}"
        
        # Create directory
        os.makedirs(self.reports_dir, exist_ok=True)
        
        # Set up plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        print(f"📁 Reports will be saved to: {self.reports_dir}")
    
    def run_backtest(self, asset: str, strategy: str, timeframe: str, 
                    initial_capital: float = 10000.0, export_additional: bool = False,
                    **strategy_params) -> ConsolidatedBacktestResult:
        """Run consolidated backtest with single report image"""
        print(f"🎯 Running Consolidated Backtest: {asset} {strategy} {timeframe}")
        print(f"💰 Initial Capital: ${initial_capital:,.2f}")
        print("-" * 60)
        
        # Get real data (with fallback to sample data)
        data = self._get_real_data(asset, timeframe)
        
        # Create strategy
        strategy_obj = create_strategy(strategy, **strategy_params)
        
        # Generate signals
        signals = self._generate_signals(data, strategy_obj)
        
        # Simulate trades
        trades = self._simulate_trades(data, signals, initial_capital)
        
        # Calculate equity curve
        equity_curve = self._calculate_equity_curve(data, trades, initial_capital)
        
        # Calculate performance metrics
        performance = self._calculate_performance_metrics(equity_curve, trades, initial_capital)
        
        # Create result
        result = ConsolidatedBacktestResult(
            strategy_name=strategy,
            asset=asset,
            timeframe=timeframe,
            start_date=data.index[0],
            end_date=data.index[-1],
            initial_capital=initial_capital,
            final_capital=equity_curve['equity'].iloc[-1],
            total_return=performance['total_return'],
            total_trades=len(trades),
            winning_trades=performance['winning_trades'],
            losing_trades=performance['losing_trades'],
            win_rate=performance['win_rate'],
            profit_factor=performance['profit_factor'],
            sharpe_ratio=performance['sharpe_ratio'],
            max_drawdown=performance['max_drawdown'],
            avg_win=performance['avg_win'],
            avg_loss=performance['avg_loss'],
            largest_win=performance['largest_win'],
            largest_loss=performance['largest_loss'],
            total_fees=performance['total_fees'],
            equity_curve=equity_curve,
            trades=trades,
            signals=signals
        )
        
        # Generate consolidated report
        self._generate_consolidated_report(result)
        
        # Optional additional exports
        if export_additional:
            self._generate_additional_exports(result)
        
        print(f"✅ Consolidated backtest completed!")
        print(f"📊 Total Return: {result.total_return:.2f}%")
        print(f"📈 Total Trades: {result.total_trades}")
        print(f"🎯 Win Rate: {result.win_rate:.1f}%")
        print(f"📉 Max Drawdown: {result.max_drawdown:.2f}%")
        print(f"📊 Sharpe Ratio: {result.sharpe_ratio:.2f}")
        
        return result
    
    def _get_real_data(self, asset: str, timeframe: str) -> pd.DataFrame:
        """Get real cryptocurrency data for backtesting"""
        # Convert asset format (BTCUSDT -> BTC)
        asset_clean = asset.replace('USDT', '')
        
        try:
            # Try to get real data first
            data = get_data(asset_clean, timeframe, days=365)
            print(f"📊 Using real {asset} {timeframe} data: {len(data)} records")
            return data
        except Exception as e:
            print(f"⚠️ Could not load real data for {asset} {timeframe}: {str(e)}")
            print(f"🔄 Falling back to sample data generation...")
            return self._create_sample_data(asset, timeframe)
    
    def _create_sample_data(self, asset: str, timeframe: str) -> pd.DataFrame:
        """Create sample data as fallback (only if real data unavailable)"""
        print(f"⚠️ Generating sample data for {asset} {timeframe} (not recommended for production)")
        
        # Create proper date range
        start_date = datetime(2024, 1, 1)  # Use 2024 for more realistic dates
        if timeframe == '1h':
            periods = 2000
            dates = pd.date_range(start=start_date, periods=periods, freq='1H')
        elif timeframe == '4h':
            periods = 1000
            dates = pd.date_range(start=start_date, periods=periods, freq='4H')
        else:
            periods = 500
            dates = pd.date_range(start=start_date, periods=periods, freq='1D')
        
        # Generate realistic price data
        np.random.seed(42)  # For reproducible results
        
        # Base price based on asset
        base_prices = {
            'BTCUSDT': 50000,
            'ETHUSDT': 3000,
            'BNBUSDT': 300,
            'XRPUSDT': 0.5,
            'ADAUSDT': 0.4,
            'SOLUSDT': 100,
            'DOGEUSDT': 0.08,
            'AVAXUSDT': 25,
            'LINKUSDT': 15
        }
        
        base_price = base_prices.get(asset, 100)
        
        # Generate price series with trend and volatility
        returns = np.random.normal(0.0001, 0.02, periods)  # Slight upward bias
        prices = [base_price]
        
        for ret in returns[1:]:
            new_price = prices[-1] * (1 + ret)
            prices.append(new_price)
        
        # Generate OHLC data
        data = pd.DataFrame(index=dates)
        data['close'] = prices
        data['open'] = data['close'].shift(1).fillna(data['close'])
        data['high'] = data[['open', 'close']].max(axis=1) * (1 + np.random.uniform(0, 0.01, len(data)))
        data['low'] = data[['open', 'close']].min(axis=1) * (1 - np.random.uniform(0, 0.01, len(data)))
        data['volume'] = np.random.uniform(1000, 10000, len(data))
        
        return data
    
    def _generate_signals(self, data: pd.DataFrame, strategy_obj) -> List[Dict]:
        """Generate trading signals"""
        signals = []
        
        # Test on rolling windows
        window_size = 50
        for i in range(window_size, len(data)):
            window_data = data.iloc[i-window_size:i]
            signal = strategy_obj.generate_signal(window_data, timeframe='4h')
            if signal:
                signals.append({
                    'timestamp': data.index[i],
                    'action': signal.action,
                    'price': signal.entry_price,
                    'confidence': signal.confidence,
                    'reason': signal.reason
                })
        
        return signals
    
    def _simulate_trades(self, data: pd.DataFrame, signals: List[Dict], 
                        initial_capital: float) -> List[Dict]:
        """Simulate trades from signals"""
        trades = []
        current_capital = initial_capital
        position = None
        
        for signal in signals:
            timestamp = signal['timestamp']
            action = signal['action']
            price = signal['price']
            confidence = signal['confidence']
            
            # Calculate position size (3% of capital for profitability)
            position_size = current_capital * 0.03 / price
            
            if action == 'BUY' and position is None:
                # Open long position
                position = {
                    'entry_time': timestamp,
                    'entry_price': price,
                    'position_size': position_size,
                    'action': 'BUY',
                    'confidence': confidence
                }
            
            elif action == 'SELL' and position is not None:
                # Close position
                exit_price = price
                pnl = (exit_price - position['entry_price']) * position['position_size']
                
                # Calculate fees (0.1% each way)
                entry_fee = position['entry_price'] * position['position_size'] * 0.001
                exit_fee = exit_price * position['position_size'] * 0.001
                total_fees = entry_fee + exit_fee
                
                # Net P&L after fees
                net_pnl = pnl - total_fees
                current_capital += net_pnl
                
                trade = {
                    'entry_time': position['entry_time'],
                    'exit_time': timestamp,
                    'entry_price': position['entry_price'],
                    'exit_price': exit_price,
                    'position_size': position['position_size'],
                    'pnl': net_pnl,
                    'fees': total_fees,
                    'confidence': position['confidence']
                }
                
                trades.append(trade)
                position = None
        
        return trades
    
    def _calculate_equity_curve(self, data: pd.DataFrame, trades: List[Dict], 
                              initial_capital: float) -> pd.DataFrame:
        """Calculate equity curve"""
        equity_curve = pd.DataFrame(index=data.index)
        equity_curve['price'] = data['close']
        equity_curve['equity'] = initial_capital
        equity_curve['drawdown'] = 0.0
        
        current_equity = initial_capital
        peak_equity = initial_capital
        
        # Create trade lookup
        trade_lookup = {}
        for trade in trades:
            exit_time = trade['exit_time']
            trade_lookup[exit_time] = trade
        
        # Update equity at each time point
        for timestamp in data.index:
            if timestamp in trade_lookup:
                trade = trade_lookup[timestamp]
                current_equity += trade['pnl']
                
                # Update peak and drawdown
                if current_equity > peak_equity:
                    peak_equity = current_equity
                
                drawdown = (peak_equity - current_equity) / peak_equity * 100
                equity_curve.loc[timestamp, 'drawdown'] = drawdown
            
            equity_curve.loc[timestamp, 'equity'] = current_equity
        
        # Forward fill drawdown
        equity_curve['drawdown'] = equity_curve['drawdown'].fillna(method='ffill')
        
        return equity_curve
    
    def _calculate_performance_metrics(self, equity_curve: pd.DataFrame, 
                                     trades: List[Dict], initial_capital: float) -> Dict:
        """Calculate performance metrics"""
        final_capital = equity_curve['equity'].iloc[-1]
        total_return = (final_capital - initial_capital) / initial_capital * 100
        
        # Trade statistics
        winning_trades = 0
        losing_trades = 0
        total_wins = 0
        total_losses = 0
        largest_win = 0
        largest_loss = 0
        total_fees = 0
        
        for trade in trades:
            pnl = trade['pnl']
            fees = trade['fees']
            
            total_fees += fees
            
            if pnl > 0:
                winning_trades += 1
                total_wins += pnl
                largest_win = max(largest_win, pnl)
            elif pnl < 0:
                losing_trades += 1
                total_losses += abs(pnl)
                largest_loss = min(largest_loss, pnl)
        
        win_rate = (winning_trades / len(trades) * 100) if trades else 0
        profit_factor = (total_wins / total_losses) if total_losses > 0 else float('inf')
        avg_win = (total_wins / winning_trades) if winning_trades > 0 else 0
        avg_loss = (total_losses / losing_trades) if losing_trades > 0 else 0
        
        # Sharpe ratio (simplified)
        if len(trades) > 1:
            trade_returns = [trade['pnl'] / initial_capital for trade in trades]
            sharpe_ratio = np.mean(trade_returns) / np.std(trade_returns) * np.sqrt(252) if np.std(trade_returns) > 0 else 0
        else:
            sharpe_ratio = 0
        
        # Maximum drawdown
        max_drawdown = equity_curve['drawdown'].max()
        
        return {
            'total_return': total_return,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'largest_win': largest_win,
            'largest_loss': largest_loss,
            'total_fees': total_fees
        }
    
    def _generate_consolidated_report(self, result: ConsolidatedBacktestResult):
        """Generate single consolidated report image"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Create time-ordered filename with clear timestamp
        filename = f"{timestamp}_{result.asset}_{result.strategy_name}_{result.timeframe}_report.png"
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 16))
        fig.suptitle(f'CONSOLIDATED BACKTEST REPORT - {result.asset} {result.strategy_name.upper()} ({result.timeframe})', 
                    fontsize=20, fontweight='bold', y=0.98)
        
        # Create grid layout
        gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3, 
                             height_ratios=[1, 1, 1, 0.8], width_ratios=[1, 1, 1, 1])
        
        # 1. Price Chart with Signals (top row, spans 2 columns)
        ax1 = fig.add_subplot(gs[0, :2])
        self._plot_price_with_signals(ax1, result)
        
        # 2. Equity Curve (top row, right side)
        ax2 = fig.add_subplot(gs[0, 2:])
        self._plot_equity_curve(ax2, result)
        
        # 3. Performance Metrics (second row, left)
        ax3 = fig.add_subplot(gs[1, :2])
        self._plot_performance_metrics(ax3, result)
        
        # 4. Risk Analysis (second row, right)
        ax4 = fig.add_subplot(gs[1, 2:])
        self._plot_risk_analysis(ax4, result)
        
        # 5. Trade Analysis (third row, left)
        ax5 = fig.add_subplot(gs[2, :2])
        self._plot_trade_analysis(ax5, result)
        
        # 6. Drawdown Chart (third row, right)
        ax6 = fig.add_subplot(gs[2, 2:])
        self._plot_drawdown(ax6, result)
        
        # 7. Summary Table (bottom row, full width)
        ax7 = fig.add_subplot(gs[3, :])
        self._plot_summary_table(ax7, result)
        
        # Save the consolidated report
        plt.savefig(f'{self.reports_dir}/{filename}', dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close()
        
        print(f"📊 Consolidated report saved: {filename}")
    
    def _plot_price_with_signals(self, ax, result: ConsolidatedBacktestResult):
        """Plot price chart with signals"""
        # Plot price
        ax.plot(result.equity_curve.index, result.equity_curve['price'], 
                label='Price', color='black', linewidth=1.5)
        
        # Plot buy signals
        buy_signals = [s for s in result.signals if s['action'] == 'BUY']
        if buy_signals:
            buy_times = [s['timestamp'] for s in buy_signals]
            buy_prices = [s['price'] for s in buy_signals]
            ax.scatter(buy_times, buy_prices, color='green', marker='^', 
                      s=80, label=f'Buy ({len(buy_signals)})', zorder=5, alpha=0.8)
        
        # Plot sell signals
        sell_signals = [s for s in result.signals if s['action'] == 'SELL']
        if sell_signals:
            sell_times = [s['timestamp'] for s in sell_signals]
            sell_prices = [s['price'] for s in sell_signals]
            ax.scatter(sell_times, sell_prices, color='red', marker='v', 
                      s=80, label=f'Sell ({len(sell_signals)})', zorder=5, alpha=0.8)
        
        # Plot trades
        for trade in result.trades:
            entry_time = trade['entry_time']
            exit_time = trade['exit_time']
            entry_price = trade['entry_price']
            exit_price = trade['exit_price']
            pnl = trade['pnl']
            
            color = 'green' if pnl > 0 else 'red'
            ax.plot([entry_time, exit_time], [entry_price, exit_price], 
                   color=color, alpha=0.6, linewidth=2)
            ax.scatter([entry_time, exit_time], [entry_price, exit_price], 
                      color=color, s=40, zorder=6, alpha=0.8)
        
        ax.set_title('Price Chart with Trading Signals', fontsize=14, fontweight='bold')
        ax.set_ylabel('Price ($)')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    def _plot_equity_curve(self, ax, result: ConsolidatedBacktestResult):
        """Plot equity curve"""
        ax.plot(result.equity_curve.index, result.equity_curve['equity'], 
                label='Equity', color='blue', linewidth=2)
        ax.axhline(y=result.initial_capital, color='gray', linestyle='--', 
                  alpha=0.7, label='Initial Capital')
        
        ax.set_title('Equity Curve', fontsize=14, fontweight='bold')
        ax.set_ylabel('Equity ($)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Format y-axis as currency
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    def _plot_performance_metrics(self, ax, result: ConsolidatedBacktestResult):
        """Plot performance metrics bar chart"""
        metrics = [
            ('Total Return (%)', result.total_return, 'green' if result.total_return > 0 else 'red'),
            ('Win Rate (%)', result.win_rate, 'blue'),
            ('Sharpe Ratio', result.sharpe_ratio, 'purple'),
            ('Profit Factor', result.profit_factor, 'orange'),
            ('Total Trades', result.total_trades, 'brown')
        ]
        
        metric_names = [m[0] for m in metrics]
        metric_values = [m[1] for m in metrics]
        colors = [m[2] for m in metrics]
        
        bars = ax.bar(range(len(metrics)), metric_values, color=colors, alpha=0.7, edgecolor='black')
        ax.set_title('Performance Metrics', fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(metrics)))
        ax.set_xticklabels(metric_names, rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, value in zip(bars, metric_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
    
    def _plot_risk_analysis(self, ax, result: ConsolidatedBacktestResult):
        """Plot risk analysis"""
        risk_metrics = [
            ('Max Drawdown (%)', result.max_drawdown, 'red'),
            ('Avg Win ($)', result.avg_win, 'green'),
            ('Avg Loss ($)', abs(result.avg_loss), 'red'),
            ('Largest Win ($)', result.largest_win, 'green'),
            ('Largest Loss ($)', abs(result.largest_loss), 'red')
        ]
        
        metric_names = [r[0] for r in risk_metrics]
        metric_values = [r[1] for r in risk_metrics]
        colors = [r[2] for r in risk_metrics]
        
        bars = ax.bar(range(len(risk_metrics)), metric_values, color=colors, alpha=0.7, edgecolor='black')
        ax.set_title('Risk Analysis', fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(risk_metrics)))
        ax.set_xticklabels(metric_names, rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, value in zip(bars, metric_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
    
    def _plot_trade_analysis(self, ax, result: ConsolidatedBacktestResult):
        """Plot trade analysis pie chart"""
        if not result.trades:
            ax.text(0.5, 0.5, 'No trades executed', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=14)
            ax.set_title('Trade Analysis', fontsize=14, fontweight='bold')
            return
        
        # Win/Loss pie chart
        win_loss_data = [result.winning_trades, result.losing_trades]
        win_loss_labels = [f'Wins\n({result.winning_trades})', f'Losses\n({result.losing_trades})']
        colors = ['green', 'red']
        
        wedges, texts, autotexts = ax.pie(win_loss_data, labels=win_loss_labels, colors=colors, 
                                         autopct='%1.1f%%', startangle=90)
        ax.set_title(f'Trade Analysis\nWin Rate: {result.win_rate:.1f}%', 
                    fontsize=14, fontweight='bold')
        
        # Make percentage text bold
        for autotext in autotexts:
            autotext.set_fontweight('bold')
    
    def _plot_drawdown(self, ax, result: ConsolidatedBacktestResult):
        """Plot drawdown chart"""
        ax.fill_between(result.equity_curve.index, 0, -result.equity_curve['drawdown'], 
                       color='red', alpha=0.3, label='Drawdown')
        ax.plot(result.equity_curve.index, -result.equity_curve['drawdown'], 
                color='red', linewidth=1)
        
        ax.set_title(f'Drawdown Analysis\nMax: {result.max_drawdown:.2f}%', 
                    fontsize=14, fontweight='bold')
        ax.set_ylabel('Drawdown (%)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    def _plot_summary_table(self, ax, result: ConsolidatedBacktestResult):
        """Plot summary table"""
        ax.axis('off')
        
        # Create summary data
        summary_data = [
            ['Metric', 'Value', 'Metric', 'Value'],
            ['Strategy', result.strategy_name.replace('_', ' ').title(), 'Asset', result.asset],
            ['Timeframe', result.timeframe, 'Period', f"{result.start_date.strftime('%Y-%m-%d')} to {result.end_date.strftime('%Y-%m-%d')}"],
            ['Initial Capital', f"${result.initial_capital:,.2f}", 'Final Capital', f"${result.final_capital:,.2f}"],
            ['Total Return', f"{result.total_return:.2f}%", 'Total Trades', str(result.total_trades)],
            ['Win Rate', f"{result.win_rate:.1f}%", 'Profit Factor', f"{result.profit_factor:.2f}"],
            ['Sharpe Ratio', f"{result.sharpe_ratio:.2f}", 'Max Drawdown', f"{result.max_drawdown:.2f}%"],
            ['Winning Trades', str(result.winning_trades), 'Losing Trades', str(result.losing_trades)],
            ['Avg Win', f"${result.avg_win:.2f}", 'Avg Loss', f"${result.avg_loss:.2f}"],
            ['Largest Win', f"${result.largest_win:.2f}", 'Largest Loss', f"${result.largest_loss:.2f}"],
            ['Total Fees', f"${result.total_fees:.2f}", 'Total Signals', str(len(result.signals))]
        ]
        
        # Create table
        table = ax.table(cellText=summary_data[1:], colLabels=summary_data[0],
                        cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
        
        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Color the header
        for i in range(4):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Color alternate rows
        for i in range(1, len(summary_data)):
            for j in range(4):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#f0f0f0')
        
        ax.set_title('Summary Table', fontsize=16, fontweight='bold', pad=20)
    
    def _generate_additional_exports(self, result: ConsolidatedBacktestResult):
        """Generate additional exports (CSV, JSON, HTML) if requested"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"{result.asset}_{result.strategy_name}_{result.timeframe}_{timestamp}"
        
        # JSON Report
        report_data = {
            'backtest_info': {
                'strategy_name': result.strategy_name,
                'asset': result.asset,
                'timeframe': result.timeframe,
                'start_date': result.start_date.isoformat(),
                'end_date': result.end_date.isoformat(),
                'initial_capital': result.initial_capital,
                'final_capital': result.final_capital
            },
            'performance_metrics': {
                'total_return': result.total_return,
                'winning_trades': result.winning_trades,
                'losing_trades': result.losing_trades,
                'win_rate': result.win_rate,
                'profit_factor': result.profit_factor,
                'sharpe_ratio': result.sharpe_ratio,
                'max_drawdown': result.max_drawdown,
                'avg_win': result.avg_win,
                'avg_loss': result.avg_loss,
                'largest_win': result.largest_win,
                'largest_loss': result.largest_loss,
                'total_fees': result.total_fees
            },
            'trades': result.trades,
            'signals': result.signals
        }
        
        with open(f'{self.reports_dir}/{base_filename}_additional.json', 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        # CSV Reports
        result.equity_curve.to_csv(f'{self.reports_dir}/{base_filename}_equity_curve.csv')
        
        if result.trades:
            trades_df = pd.DataFrame(result.trades)
            trades_df.to_csv(f'{self.reports_dir}/{base_filename}_trades.csv', index=False)
        
        if result.signals:
            signals_df = pd.DataFrame(result.signals)
            signals_df.to_csv(f'{self.reports_dir}/{base_filename}_signals.csv', index=False)
        
        print(f"📋 Additional exports saved: JSON, CSV files")

def main():
    """Main function to run consolidated visual backtesting"""
    print("🎯 Consolidated Visual Backtesting System")
    print("=" * 50)
    
    # Create consolidated visual backtesting system
    visual_system = ConsolidatedVisualBacktest()
    
    # Test configurations with different capital amounts
    test_configs = [
        {
            'asset': 'BTCUSDT',
            'strategy': 'ema_crossover',
            'timeframe': '4h',
            'initial_capital': 1000,
            'fast_period': 8,
            'slow_period': 21,
            'export_additional': False  # Only consolidated report by default
        },
        {
            'asset': 'ETHUSDT',
            'strategy': 'rsi_mean_reversion',
            'timeframe': '4h',
            'initial_capital': 1000,
            'rsi_period': 14,
            'oversold': 35,
            'overbought': 65,
            'export_additional': False
        },
        {
            'asset': 'BTCUSDT',
            'strategy': 'bollinger_bands',
            'timeframe': '1h',
            'initial_capital': 10000,
            'period': 20,
            'std_dev': 2.0,
            'export_additional': False
        },
        {
            'asset': 'ETHUSDT',
            'strategy': 'macd',
            'timeframe': '4h',
            'initial_capital': 10000,
            'fast_period': 12,
            'slow_period': 26,
            'export_additional': False
        }
    ]
    
    results = []
    
    for config in test_configs:
        try:
            print(f"\n🚀 Testing: {config['asset']} {config['strategy']} {config['timeframe']} (${config['initial_capital']:,})")
            result = visual_system.run_backtest(**config)
            results.append(result)
            
            print(f"✅ Completed: {result.total_return:.2f}% return, {result.total_trades} trades")
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")
    
    print(f"\n🎉 Consolidated visual backtesting completed!")
    print(f"📊 Generated {len(results)} consolidated reports")
    print(f"📁 Reports saved to: {visual_system.reports_dir}/")
    
    # Summary
    if results:
        print(f"\n📈 SUMMARY:")
        for result in results:
            print(f"  {result.asset} {result.strategy_name} (${result.initial_capital:,}): "
                  f"{result.total_return:.2f}% return, {result.win_rate:.1f}% win rate, "
                  f"{result.max_drawdown:.2f}% max drawdown")

if __name__ == "__main__":
    main()
