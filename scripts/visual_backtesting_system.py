"""
Visual Backtesting System
========================

This system provides comprehensive backtesting with detailed visualizations,
performance metrics, and analysis reports. It shows exactly where trades
were entered and exited, calculates all key performance metrics, and
generates professional charts and reports.

Features:
- Interactive price charts with buy/sell signals
- Performance metrics (Sharpe ratio, drawdown, profitability)
- Capital-based analysis (starting from $1000, $10000, etc.)
- Detailed trade analysis and statistics
- Export to multiple formats (PNG, PDF, CSV, HTML)

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
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo

warnings.filterwarnings('ignore')

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from simple_profitable_strategies import create_simple_strategy
from strategy_integration import EnhancedBacktestEngine
from backtester import BacktestConfig
from strategies import create_strategy

@dataclass
class VisualBacktestResult:
    """Results from visual backtesting"""
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
    total_slippage: float
    equity_curve: pd.DataFrame
    trades: List[Dict]
    signals: List[Dict]
    performance_metrics: Dict

class VisualBacktestingSystem:
    """
    Comprehensive visual backtesting system
    """
    
    def __init__(self):
        self.results = []
        self.plots_dir = "backtest_plots"
        self.reports_dir = "backtest_reports"
        
        # Create directories
        os.makedirs(self.plots_dir, exist_ok=True)
        os.makedirs(self.reports_dir, exist_ok=True)
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def run_visual_backtest(self, asset: str, strategy: str, timeframe: str, 
                          initial_capital: float = 10000.0, **strategy_params) -> VisualBacktestResult:
        """Run comprehensive visual backtest"""
        print(f"🎯 Running Visual Backtest: {asset} {strategy} {timeframe}")
        print(f"💰 Initial Capital: ${initial_capital:,.2f}")
        print("-" * 60)
        
        # Configure backtest
        config = BacktestConfig(
            symbol=asset,
            interval=timeframe,
            initial_capital=initial_capital,
            risk_per_trade=0.01,  # 1% risk per trade
            min_confidence=20.0,  # Low threshold to catch signals
            slippage=0.0005,  # 0.05% slippage
            trading_fee=0.001,  # 0.1% trading fee
            max_drawdown_stop=0.25,  # 25% max drawdown
            max_open_trades=1
        )
        
        # Create enhanced backtest engine
        engine = EnhancedBacktestEngine(config, strategy)
        
        # Fetch historical data
        data_limit = 2000 if timeframe == '1h' else 1000 if timeframe == '4h' else 500
        engine.fetch_historical_data(limit=data_limit)
        
        if engine.historical_data is None or len(engine.historical_data) < 200:
            raise ValueError(f"Insufficient data for {asset} {timeframe}")
        
        # Run backtest
        backtest_results = engine.run_strategy_backtest()
        
        if not backtest_results:
            raise ValueError(f"No backtest results for {asset} {strategy} {timeframe}")
        
        # Extract data
        data = engine.historical_data
        trades = engine.trades
        
        # Calculate equity curve
        equity_curve = self._calculate_equity_curve(data, trades, initial_capital)
        
        # Extract signals
        signals = self._extract_signals(data, strategy, **strategy_params)
        
        # Calculate performance metrics
        performance_metrics = self._calculate_performance_metrics(
            equity_curve, trades, initial_capital
        )
        
        # Create result object
        result = VisualBacktestResult(
            strategy_name=strategy,
            asset=asset,
            timeframe=timeframe,
            start_date=data.index[0],
            end_date=data.index[-1],
            initial_capital=initial_capital,
            final_capital=equity_curve['equity'].iloc[-1],
            total_return=performance_metrics['total_return'],
            total_trades=len(trades),
            winning_trades=performance_metrics['winning_trades'],
            losing_trades=performance_metrics['losing_trades'],
            win_rate=performance_metrics['win_rate'],
            profit_factor=performance_metrics['profit_factor'],
            sharpe_ratio=performance_metrics['sharpe_ratio'],
            max_drawdown=performance_metrics['max_drawdown'],
            avg_win=performance_metrics['avg_win'],
            avg_loss=performance_metrics['avg_loss'],
            largest_win=performance_metrics['largest_win'],
            largest_loss=performance_metrics['largest_loss'],
            total_fees=performance_metrics['total_fees'],
            total_slippage=performance_metrics['total_slippage'],
            equity_curve=equity_curve,
            trades=trades,
            signals=signals,
            performance_metrics=performance_metrics
        )
        
        # Generate visualizations
        self._generate_visualizations(result)
        
        # Generate reports
        self._generate_reports(result)
        
        print(f"✅ Backtest completed!")
        print(f"📊 Total Return: {result.total_return:.2f}%")
        print(f"📈 Total Trades: {result.total_trades}")
        print(f"🎯 Win Rate: {result.win_rate:.1f}%")
        print(f"📉 Max Drawdown: {result.max_drawdown:.2f}%")
        print(f"📊 Sharpe Ratio: {result.sharpe_ratio:.2f}")
        
        return result
    
    def _calculate_equity_curve(self, data: pd.DataFrame, trades: List, 
                              initial_capital: float) -> pd.DataFrame:
        """Calculate equity curve from trades"""
        equity_curve = pd.DataFrame(index=data.index)
        equity_curve['price'] = data['close']
        equity_curve['equity'] = initial_capital
        equity_curve['drawdown'] = 0.0
        equity_curve['returns'] = 0.0
        
        current_equity = initial_capital
        peak_equity = initial_capital
        
        for trade in trades:
            if hasattr(trade, 'entry_time') and hasattr(trade, 'exit_time'):
                entry_time = trade.entry_time
                exit_time = trade.exit_time
                pnl = getattr(trade, 'pnl', 0)
                
                # Update equity at exit time
                if exit_time in equity_curve.index:
                    current_equity += pnl
                    equity_curve.loc[exit_time, 'equity'] = current_equity
                    
                    # Update peak and drawdown
                    if current_equity > peak_equity:
                        peak_equity = current_equity
                    
                    drawdown = (peak_equity - current_equity) / peak_equity * 100
                    equity_curve.loc[exit_time, 'drawdown'] = drawdown
        
        # Forward fill equity values
        equity_curve['equity'] = equity_curve['equity'].fillna(method='ffill')
        equity_curve['drawdown'] = equity_curve['drawdown'].fillna(method='ffill')
        
        # Calculate returns
        equity_curve['returns'] = equity_curve['equity'].pct_change() * 100
        
        return equity_curve
    
    def _extract_signals(self, data: pd.DataFrame, strategy: str, **params) -> List[Dict]:
        """Extract trading signals from data"""
        try:
            # Try simple strategies first
            if strategy.startswith('simple_'):
                strategy_obj = create_simple_strategy(strategy, **params)
            else:
                # Try professional strategies
                strategy_obj = create_strategy(strategy, **params)
            
            signals = []
            
            # Test on rolling windows
            window_size = 50
            for i in range(window_size, len(data)):
                window_data = data.iloc[i-window_size:i]
                signal = strategy_obj.generate_signal(window_data)
                if signal:
                    signals.append({
                        'timestamp': data.index[i],
                        'action': signal.action,
                        'price': signal.entry_price,
                        'confidence': signal.confidence,
                        'reason': getattr(signal, 'reason', f'{strategy} signal')
                    })
            
            return signals
        except Exception as e:
            print(f"Warning: Could not extract signals: {e}")
            return []
    
    def _calculate_performance_metrics(self, equity_curve: pd.DataFrame, 
                                     trades: List, initial_capital: float) -> Dict:
        """Calculate comprehensive performance metrics"""
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
        total_slippage = 0
        
        for trade in trades:
            pnl = getattr(trade, 'pnl', 0)
            fees = getattr(trade, 'fees', 0)
            slippage = getattr(trade, 'slippage', 0)
            
            total_fees += fees
            total_slippage += slippage
            
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
        
        # Sharpe ratio
        returns = equity_curve['returns'].dropna()
        if len(returns) > 1:
            sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
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
            'total_fees': total_fees,
            'total_slippage': total_slippage
        }
    
    def _generate_visualizations(self, result: VisualBacktestResult):
        """Generate comprehensive visualizations"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"{result.asset}_{result.strategy_name}_{result.timeframe}_{timestamp}"
        
        # 1. Main Price Chart with Signals
        self._plot_price_with_signals(result, base_filename)
        
        # 2. Equity Curve
        self._plot_equity_curve(result, base_filename)
        
        # 3. Drawdown Chart
        self._plot_drawdown(result, base_filename)
        
        # 4. Trade Analysis
        self._plot_trade_analysis(result, base_filename)
        
        # 5. Performance Dashboard
        self._plot_performance_dashboard(result, base_filename)
        
        # 6. Interactive Plotly Chart
        self._plot_interactive_chart(result, base_filename)
        
        print(f"📊 Visualizations saved to: {self.plots_dir}/")
    
    def _plot_price_with_signals(self, result: VisualBacktestResult, base_filename: str):
        """Plot price chart with buy/sell signals"""
        fig, ax = plt.subplots(figsize=(15, 8))
        
        # Plot price
        ax.plot(result.equity_curve.index, result.equity_curve['price'], 
                label='Price', color='black', linewidth=1)
        
        # Plot buy signals
        buy_signals = [s for s in result.signals if s['action'] == 'BUY']
        if buy_signals:
            buy_times = [s['timestamp'] for s in buy_signals]
            buy_prices = [s['price'] for s in buy_signals]
            ax.scatter(buy_times, buy_prices, color='green', marker='^', 
                      s=100, label=f'Buy Signals ({len(buy_signals)})', zorder=5)
        
        # Plot sell signals
        sell_signals = [s for s in result.signals if s['action'] == 'SELL']
        if sell_signals:
            sell_times = [s['timestamp'] for s in sell_signals]
            sell_prices = [s['price'] for s in sell_signals]
            ax.scatter(sell_times, sell_prices, color='red', marker='v', 
                      s=100, label=f'Sell Signals ({len(sell_signals)})', zorder=5)
        
        # Plot trades
        for trade in result.trades:
            if hasattr(trade, 'entry_time') and hasattr(trade, 'exit_time'):
                entry_time = trade.entry_time
                exit_time = trade.exit_time
                entry_price = getattr(trade, 'entry_price', 0)
                exit_price = getattr(trade, 'exit_price', 0)
                pnl = getattr(trade, 'pnl', 0)
                
                color = 'green' if pnl > 0 else 'red'
                ax.plot([entry_time, exit_time], [entry_price, exit_price], 
                       color=color, alpha=0.7, linewidth=2)
                ax.scatter([entry_time, exit_time], [entry_price, exit_price], 
                          color=color, s=50, zorder=6)
        
        ax.set_title(f'{result.asset} - {result.strategy_name} - {result.timeframe}\n'
                    f'Total Return: {result.total_return:.2f}% | '
                    f'Trades: {result.total_trades} | '
                    f'Win Rate: {result.win_rate:.1f}%', fontsize=14)
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(f'{self.plots_dir}/{base_filename}_price_signals.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_equity_curve(self, result: VisualBacktestResult, base_filename: str):
        """Plot equity curve"""
        fig, ax = plt.subplots(figsize=(15, 6))
        
        ax.plot(result.equity_curve.index, result.equity_curve['equity'], 
                label='Equity Curve', color='blue', linewidth=2)
        ax.axhline(y=result.initial_capital, color='gray', linestyle='--', 
                  alpha=0.7, label='Initial Capital')
        
        ax.set_title(f'Equity Curve - {result.asset} {result.strategy_name}\n'
                    f'Initial: ${result.initial_capital:,.2f} | '
                    f'Final: ${result.final_capital:,.2f} | '
                    f'Return: {result.total_return:.2f}%', fontsize=14)
        ax.set_xlabel('Date')
        ax.set_ylabel('Equity ($)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Format y-axis as currency
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(f'{self.plots_dir}/{base_filename}_equity_curve.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_drawdown(self, result: VisualBacktestResult, base_filename: str):
        """Plot drawdown chart"""
        fig, ax = plt.subplots(figsize=(15, 6))
        
        ax.fill_between(result.equity_curve.index, 0, -result.equity_curve['drawdown'], 
                       color='red', alpha=0.3, label='Drawdown')
        ax.plot(result.equity_curve.index, -result.equity_curve['drawdown'], 
                color='red', linewidth=1)
        
        ax.set_title(f'Drawdown Analysis - {result.asset} {result.strategy_name}\n'
                    f'Maximum Drawdown: {result.max_drawdown:.2f}%', fontsize=14)
        ax.set_xlabel('Date')
        ax.set_ylabel('Drawdown (%)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(f'{self.plots_dir}/{base_filename}_drawdown.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_trade_analysis(self, result: VisualBacktestResult, base_filename: str):
        """Plot trade analysis"""
        if not result.trades:
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Trade P&L distribution
        trade_pnls = [getattr(trade, 'pnl', 0) for trade in result.trades]
        ax1.hist(trade_pnls, bins=20, alpha=0.7, color='blue', edgecolor='black')
        ax1.set_title('Trade P&L Distribution')
        ax1.set_xlabel('P&L ($)')
        ax1.set_ylabel('Frequency')
        ax1.axvline(x=0, color='red', linestyle='--', alpha=0.7)
        ax1.grid(True, alpha=0.3)
        
        # Win/Loss pie chart
        win_loss_data = [result.winning_trades, result.losing_trades]
        win_loss_labels = ['Wins', 'Losses']
        colors = ['green', 'red']
        ax2.pie(win_loss_data, labels=win_loss_labels, colors=colors, autopct='%1.1f%%')
        ax2.set_title(f'Win/Loss Ratio\nWin Rate: {result.win_rate:.1f}%')
        
        # Cumulative P&L
        cumulative_pnl = np.cumsum(trade_pnls)
        ax3.plot(range(len(cumulative_pnl)), cumulative_pnl, color='blue', linewidth=2)
        ax3.set_title('Cumulative P&L')
        ax3.set_xlabel('Trade Number')
        ax3.set_ylabel('Cumulative P&L ($)')
        ax3.axhline(y=0, color='red', linestyle='--', alpha=0.7)
        ax3.grid(True, alpha=0.3)
        
        # Trade duration (if available)
        durations = []
        for trade in result.trades:
            if hasattr(trade, 'entry_time') and hasattr(trade, 'exit_time'):
                duration = (trade.exit_time - trade.entry_time).total_seconds() / 3600  # hours
                durations.append(duration)
        
        if durations:
            ax4.hist(durations, bins=20, alpha=0.7, color='orange', edgecolor='black')
            ax4.set_title('Trade Duration Distribution')
            ax4.set_xlabel('Duration (hours)')
            ax4.set_ylabel('Frequency')
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'No duration data available', 
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Trade Duration Distribution')
        
        plt.suptitle(f'Trade Analysis - {result.asset} {result.strategy_name}', fontsize=16)
        plt.tight_layout()
        plt.savefig(f'{self.plots_dir}/{base_filename}_trade_analysis.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_performance_dashboard(self, result: VisualBacktestResult, base_filename: str):
        """Plot performance dashboard"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Performance metrics
        metrics = [
            ('Total Return (%)', result.total_return),
            ('Win Rate (%)', result.win_rate),
            ('Sharpe Ratio', result.sharpe_ratio),
            ('Max Drawdown (%)', result.max_drawdown),
            ('Profit Factor', result.profit_factor),
            ('Total Trades', result.total_trades)
        ]
        
        metric_names = [m[0] for m in metrics]
        metric_values = [m[1] for m in metrics]
        
        bars = ax1.bar(range(len(metrics)), metric_values, color='skyblue', edgecolor='black')
        ax1.set_title('Performance Metrics')
        ax1.set_xticks(range(len(metrics)))
        ax1.set_xticklabels(metric_names, rotation=45, ha='right')
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, metric_values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.2f}', ha='center', va='bottom')
        
        # Capital growth
        capital_data = [result.initial_capital, result.final_capital]
        capital_labels = ['Initial', 'Final']
        bars2 = ax2.bar(capital_labels, capital_data, color=['lightcoral', 'lightgreen'], 
                       edgecolor='black')
        ax2.set_title('Capital Growth')
        ax2.set_ylabel('Capital ($)')
        ax2.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars2, capital_data):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'${value:,.0f}', ha='center', va='bottom')
        
        # Risk metrics
        risk_metrics = [
            ('Max Drawdown (%)', result.max_drawdown),
            ('Avg Win ($)', result.avg_win),
            ('Avg Loss ($)', abs(result.avg_loss)),
            ('Largest Win ($)', result.largest_win),
            ('Largest Loss ($)', abs(result.largest_loss))
        ]
        
        risk_names = [r[0] for r in risk_metrics]
        risk_values = [r[1] for r in risk_metrics]
        
        bars3 = ax3.bar(range(len(risk_metrics)), risk_values, color='orange', edgecolor='black')
        ax3.set_title('Risk Metrics')
        ax3.set_xticks(range(len(risk_metrics)))
        ax3.set_xticklabels(risk_names, rotation=45, ha='right')
        ax3.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars3, risk_values):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.2f}', ha='center', va='bottom')
        
        # Summary text
        summary_text = f"""
Strategy: {result.strategy_name}
Asset: {result.asset}
Timeframe: {result.timeframe}
Period: {result.start_date.strftime('%Y-%m-%d')} to {result.end_date.strftime('%Y-%m-%d')}

Total Return: {result.total_return:.2f}%
Win Rate: {result.win_rate:.1f}%
Sharpe Ratio: {result.sharpe_ratio:.2f}
Max Drawdown: {result.max_drawdown:.2f}%
Profit Factor: {result.profit_factor:.2f}

Total Trades: {result.total_trades}
Winning Trades: {result.winning_trades}
Losing Trades: {result.losing_trades}

Initial Capital: ${result.initial_capital:,.2f}
Final Capital: ${result.final_capital:,.2f}
        """
        
        ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
        ax4.axis('off')
        
        plt.suptitle(f'Performance Dashboard - {result.asset} {result.strategy_name}', fontsize=16)
        plt.tight_layout()
        plt.savefig(f'{self.plots_dir}/{base_filename}_performance_dashboard.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_interactive_chart(self, result: VisualBacktestResult, base_filename: str):
        """Create interactive Plotly chart"""
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=('Price with Signals', 'Equity Curve', 'Drawdown'),
            vertical_spacing=0.08,
            row_heights=[0.5, 0.3, 0.2]
        )
        
        # Price chart with signals
        fig.add_trace(
            go.Scatter(x=result.equity_curve.index, y=result.equity_curve['price'],
                      mode='lines', name='Price', line=dict(color='black')),
            row=1, col=1
        )
        
        # Buy signals
        buy_signals = [s for s in result.signals if s['action'] == 'BUY']
        if buy_signals:
            buy_times = [s['timestamp'] for s in buy_signals]
            buy_prices = [s['price'] for s in buy_signals]
            fig.add_trace(
                go.Scatter(x=buy_times, y=buy_prices, mode='markers',
                          marker=dict(symbol='triangle-up', size=10, color='green'),
                          name='Buy Signals'),
                row=1, col=1
            )
        
        # Sell signals
        sell_signals = [s for s in result.signals if s['action'] == 'SELL']
        if sell_signals:
            sell_times = [s['timestamp'] for s in sell_signals]
            sell_prices = [s['price'] for s in sell_signals]
            fig.add_trace(
                go.Scatter(x=sell_times, y=sell_prices, mode='markers',
                          marker=dict(symbol='triangle-down', size=10, color='red'),
                          name='Sell Signals'),
                row=1, col=1
            )
        
        # Equity curve
        fig.add_trace(
            go.Scatter(x=result.equity_curve.index, y=result.equity_curve['equity'],
                      mode='lines', name='Equity', line=dict(color='blue')),
            row=2, col=1
        )
        
        # Drawdown
        fig.add_trace(
            go.Scatter(x=result.equity_curve.index, y=-result.equity_curve['drawdown'],
                      mode='lines', fill='tonexty', name='Drawdown', 
                      line=dict(color='red'), fillcolor='rgba(255,0,0,0.3)'),
            row=3, col=1
        )
        
        fig.update_layout(
            title=f'{result.asset} - {result.strategy_name} - {result.timeframe}<br>'
                  f'Total Return: {result.total_return:.2f}% | '
                  f'Trades: {result.total_trades} | '
                  f'Win Rate: {result.win_rate:.1f}%',
            height=800,
            showlegend=True
        )
        
        # Save interactive chart
        html_filename = f'{self.plots_dir}/{base_filename}_interactive.html'
        pyo.plot(fig, filename=html_filename, auto_open=False)
    
    def _generate_reports(self, result: VisualBacktestResult):
        """Generate comprehensive reports"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"{result.asset}_{result.strategy_name}_{result.timeframe}_{timestamp}"
        
        # 1. JSON Report
        self._generate_json_report(result, base_filename)
        
        # 2. CSV Report
        self._generate_csv_report(result, base_filename)
        
        # 3. HTML Report
        self._generate_html_report(result, base_filename)
        
        print(f"📋 Reports saved to: {self.reports_dir}/")
    
    def _generate_json_report(self, result: VisualBacktestResult, base_filename: str):
        """Generate JSON report"""
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
            'performance_metrics': result.performance_metrics,
            'trades': [
                {
                    'entry_time': trade.entry_time.isoformat() if hasattr(trade, 'entry_time') else None,
                    'exit_time': trade.exit_time.isoformat() if hasattr(trade, 'exit_time') else None,
                    'entry_price': getattr(trade, 'entry_price', 0),
                    'exit_price': getattr(trade, 'exit_price', 0),
                    'pnl': getattr(trade, 'pnl', 0),
                    'fees': getattr(trade, 'fees', 0),
                    'slippage': getattr(trade, 'slippage', 0)
                }
                for trade in result.trades
            ],
            'signals': result.signals
        }
        
        with open(f'{self.reports_dir}/{base_filename}_report.json', 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
    
    def _generate_csv_report(self, result: VisualBacktestResult, base_filename: str):
        """Generate CSV reports"""
        # Equity curve CSV
        result.equity_curve.to_csv(f'{self.reports_dir}/{base_filename}_equity_curve.csv')
        
        # Trades CSV
        if result.trades:
            trades_data = []
            for trade in result.trades:
                trades_data.append({
                    'entry_time': trade.entry_time if hasattr(trade, 'entry_time') else None,
                    'exit_time': trade.exit_time if hasattr(trade, 'exit_time') else None,
                    'entry_price': getattr(trade, 'entry_price', 0),
                    'exit_price': getattr(trade, 'exit_price', 0),
                    'pnl': getattr(trade, 'pnl', 0),
                    'fees': getattr(trade, 'fees', 0),
                    'slippage': getattr(trade, 'slippage', 0)
                })
            
            trades_df = pd.DataFrame(trades_data)
            trades_df.to_csv(f'{self.reports_dir}/{base_filename}_trades.csv', index=False)
        
        # Signals CSV
        if result.signals:
            signals_df = pd.DataFrame(result.signals)
            signals_df.to_csv(f'{self.reports_dir}/{base_filename}_signals.csv', index=False)
    
    def _generate_html_report(self, result: VisualBacktestResult, base_filename: str):
        """Generate HTML report"""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Backtest Report - {result.asset} {result.strategy_name}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 20px 0; }}
                .metric {{ background-color: #e8f4f8; padding: 15px; border-radius: 5px; text-align: center; }}
                .metric h3 {{ margin: 0; color: #333; }}
                .metric p {{ margin: 5px 0; font-size: 24px; font-weight: bold; color: #0066cc; }}
                .chart {{ margin: 20px 0; text-align: center; }}
                .chart img {{ max-width: 100%; height: auto; }}
                table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Backtest Report</h1>
                <h2>{result.asset} - {result.strategy_name} - {result.timeframe}</h2>
                <p>Period: {result.start_date.strftime('%Y-%m-%d')} to {result.end_date.strftime('%Y-%m-%d')}</p>
            </div>
            
            <div class="metrics">
                <div class="metric">
                    <h3>Total Return</h3>
                    <p>{result.total_return:.2f}%</p>
                </div>
                <div class="metric">
                    <h3>Win Rate</h3>
                    <p>{result.win_rate:.1f}%</p>
                </div>
                <div class="metric">
                    <h3>Sharpe Ratio</h3>
                    <p>{result.sharpe_ratio:.2f}</p>
                </div>
                <div class="metric">
                    <h3>Max Drawdown</h3>
                    <p>{result.max_drawdown:.2f}%</p>
                </div>
                <div class="metric">
                    <h3>Profit Factor</h3>
                    <p>{result.profit_factor:.2f}</p>
                </div>
                <div class="metric">
                    <h3>Total Trades</h3>
                    <p>{result.total_trades}</p>
                </div>
            </div>
            
            <div class="chart">
                <h3>Price Chart with Signals</h3>
                <img src="../{self.plots_dir}/{base_filename}_price_signals.png" alt="Price Chart">
            </div>
            
            <div class="chart">
                <h3>Equity Curve</h3>
                <img src="../{self.plots_dir}/{base_filename}_equity_curve.png" alt="Equity Curve">
            </div>
            
            <div class="chart">
                <h3>Performance Dashboard</h3>
                <img src="../{self.plots_dir}/{base_filename}_performance_dashboard.png" alt="Performance Dashboard">
            </div>
            
            <h3>Trade Summary</h3>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                </tr>
                <tr><td>Initial Capital</td><td>${result.initial_capital:,.2f}</td></tr>
                <tr><td>Final Capital</td><td>${result.final_capital:,.2f}</td></tr>
                <tr><td>Total Return</td><td>{result.total_return:.2f}%</td></tr>
                <tr><td>Winning Trades</td><td>{result.winning_trades}</td></tr>
                <tr><td>Losing Trades</td><td>{result.losing_trades}</td></tr>
                <tr><td>Average Win</td><td>${result.avg_win:.2f}</td></tr>
                <tr><td>Average Loss</td><td>${result.avg_loss:.2f}</td></tr>
                <tr><td>Largest Win</td><td>${result.largest_win:.2f}</td></tr>
                <tr><td>Largest Loss</td><td>${result.largest_loss:.2f}</td></tr>
                <tr><td>Total Fees</td><td>${result.total_fees:.2f}</td></tr>
                <tr><td>Total Slippage</td><td>${result.total_slippage:.2f}</td></tr>
            </table>
        </body>
        </html>
        """
        
        with open(f'{self.reports_dir}/{base_filename}_report.html', 'w') as f:
            f.write(html_content)

def main():
    """Main function to run visual backtesting"""
    print("🎯 Visual Backtesting System")
    print("=" * 50)
    
    # Create visual backtesting system
    visual_system = VisualBacktestingSystem()
    
    # Test configurations
    test_configs = [
        {
            'asset': 'BTCUSDT',
            'strategy': 'multi_timeframe_trend',
            'timeframe': '4h',
            'initial_capital': 1000
        },
        {
            'asset': 'ETHUSDT',
            'strategy': 'bollinger_mean_reversion',
            'timeframe': '4h',
            'initial_capital': 1000
        },
        {
            'asset': 'BTCUSDT',
            'strategy': 'breakout_momentum',
            'timeframe': '1h',
            'initial_capital': 10000
        }
    ]
    
    results = []
    
    for config in test_configs:
        try:
            print(f"\n🚀 Testing: {config['asset']} {config['strategy']} {config['timeframe']}")
            result = visual_system.run_visual_backtest(**config)
            results.append(result)
            
            print(f"✅ Completed: {result.total_return:.2f}% return, {result.total_trades} trades")
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")
    
    print(f"\n🎉 Visual backtesting completed!")
    print(f"📊 Generated {len(results)} backtest results")
    print(f"📁 Plots saved to: {visual_system.plots_dir}/")
    print(f"📋 Reports saved to: {visual_system.reports_dir}/")
    
    # Summary
    if results:
        print(f"\n📈 SUMMARY:")
        for result in results:
            print(f"  {result.asset} {result.strategy_name}: {result.total_return:.2f}% return, "
                  f"{result.win_rate:.1f}% win rate, {result.max_drawdown:.2f}% max drawdown")

if __name__ == "__main__":
    main()
