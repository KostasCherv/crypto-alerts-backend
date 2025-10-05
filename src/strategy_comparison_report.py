"""
Strategy Comparison Report Generator
==================================

This system creates a single comprehensive report that compares multiple
strategies side by side in one organized image.

Features:
- Single consolidated comparison report
- Multiple strategies comparison
- Side-by-side performance metrics
- Strategy ranking and analysis
- Clean, professional layout

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
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

warnings.filterwarnings('ignore')

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from consolidated_visual_backtest import ConsolidatedVisualBacktest, ConsolidatedBacktestResult

class StrategyComparisonReport:
    """
    Strategy comparison report generator
    """
    
    def __init__(self):
        # Create session-based folder structure (same as consolidated system)
        session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.reports_dir = f"results/comparisons/session_{session_timestamp}"
        self.backtest_system = ConsolidatedVisualBacktest()
        
        # Create directory
        os.makedirs(self.reports_dir, exist_ok=True)
        
        # Set up plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        print(f"📁 Comparison reports will be saved to: {self.reports_dir}")
    
    def generate_comparison_report(self, asset: str, timeframe: str, 
                                 initial_capital: float = 10000.0,
                                 strategies: List[Dict] = None) -> str:
        """Generate comprehensive strategy comparison report"""
        
        if strategies is None:
            strategies = self._get_default_strategies()
        
        print(f"🎯 Generating Strategy Comparison Report")
        print(f"📊 Asset: {asset} | Timeframe: {timeframe} | Capital: ${initial_capital:,}")
        print(f"🔍 Comparing {len(strategies)} strategies")
        print("-" * 60)
        
        # Run backtests for all strategies
        results = []
        for i, strategy_config in enumerate(strategies):
            print(f"🚀 [{i+1}/{len(strategies)}] Testing: {strategy_config['strategy']}")
            
            try:
                # Add common parameters
                strategy_config.update({
                    'asset': asset,
                    'timeframe': timeframe,
                    'initial_capital': initial_capital,
                    'export_additional': False  # Only consolidated report
                })
                
                result = self.backtest_system.run_backtest(**strategy_config)
                results.append(result)
                
                print(f"✅ {strategy_config['strategy']}: {result.total_return:.2f}% return, "
                      f"{result.win_rate:.1f}% win rate, {result.total_trades} trades")
                
            except Exception as e:
                print(f"❌ Error with {strategy_config['strategy']}: {str(e)}")
        
        if not results:
            print("❌ No successful backtests to compare")
            return None
        
        # Generate comparison report
        filename = self._create_comparison_report(results, asset, timeframe, initial_capital)
        
        # Print summary
        self._print_comparison_summary(results)
        
        return filename
    
    def _get_default_strategies(self) -> List[Dict]:
        """Get default strategies for comparison"""
        return [
            {
                'strategy': 'ema_crossover',
                'fast_period': 8,
                'slow_period': 21
            },
            {
                'strategy': 'ema_crossover',
                'fast_period': 5,
                'slow_period': 15
            },
            {
                'strategy': 'rsi_mean_reversion',
                'rsi_period': 14,
                'oversold': 35,
                'overbought': 65
            },
            {
                'strategy': 'rsi_mean_reversion',
                'rsi_period': 21,
                'oversold': 30,
                'overbought': 70
            },
            {
                'strategy': 'bollinger_bands',
                'period': 20,
                'std_dev': 2.0
            },
            {
                'strategy': 'macd',
                'fast_period': 12,
                'slow_period': 26
            }
        ]
    
    def _create_comparison_report(self, results: List[ConsolidatedBacktestResult], 
                                asset: str, timeframe: str, initial_capital: float) -> str:
        """Create comprehensive comparison report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Create time-ordered filename with clear timestamp
        filename = f"{timestamp}_{asset}_strategy_comparison_{timeframe}.png"
        
        # Sort results by total return (descending)
        results_sorted = sorted(results, key=lambda x: x.total_return, reverse=True)
        
        # Create figure
        fig = plt.figure(figsize=(24, 18))
        fig.suptitle(f'STRATEGY COMPARISON REPORT - {asset} ({timeframe}) - ${initial_capital:,} Capital', 
                    fontsize=24, fontweight='bold', y=0.98)
        
        # Create grid layout
        gs = fig.add_gridspec(4, 6, hspace=0.4, wspace=0.3, 
                             height_ratios=[1, 1, 1, 0.8], width_ratios=[1, 1, 1, 1, 1, 1])
        
        # 1. Strategy Performance Ranking (top row, spans 3 columns)
        ax1 = fig.add_subplot(gs[0, :3])
        self._plot_strategy_ranking(ax1, results_sorted)
        
        # 2. Key Metrics Comparison (top row, right side)
        ax2 = fig.add_subplot(gs[0, 3:])
        self._plot_metrics_comparison(ax2, results_sorted)
        
        # 3. Risk vs Return Scatter (second row, left)
        ax3 = fig.add_subplot(gs[1, :2])
        self._plot_risk_return_scatter(ax3, results_sorted)
        
        # 4. Win Rate vs Total Trades (second row, middle)
        ax4 = fig.add_subplot(gs[1, 2:4])
        self._plot_winrate_trades(ax4, results_sorted)
        
        # 5. Sharpe Ratio Comparison (second row, right)
        ax5 = fig.add_subplot(gs[1, 4:])
        self._plot_sharpe_comparison(ax5, results_sorted)
        
        # 6. Drawdown Analysis (third row, left)
        ax6 = fig.add_subplot(gs[2, :3])
        self._plot_drawdown_comparison(ax6, results_sorted)
        
        # 7. Trade Analysis (third row, right)
        ax7 = fig.add_subplot(gs[2, 3:])
        self._plot_trade_analysis_comparison(ax7, results_sorted)
        
        # 8. Detailed Comparison Table (bottom row, full width)
        ax8 = fig.add_subplot(gs[3, :])
        self._plot_detailed_comparison_table(ax8, results_sorted)
        
        # Save the comparison report
        plt.savefig(f'{self.reports_dir}/{filename}', dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close()
        
        print(f"📊 Strategy comparison report saved: {filename}")
        return filename
    
    def _plot_strategy_ranking(self, ax, results: List[ConsolidatedBacktestResult]):
        """Plot strategy performance ranking"""
        strategy_names = [r.strategy_name.replace('_', ' ').title() for r in results]
        returns = [r.total_return for r in results]
        colors = ['gold' if i == 0 else 'silver' if i == 1 else '#CD7F32' if i == 2 else 'lightblue' 
                 for i in range(len(results))]
        
        bars = ax.barh(range(len(strategy_names)), returns, color=colors, alpha=0.8, edgecolor='black')
        ax.set_yticks(range(len(strategy_names)))
        ax.set_yticklabels(strategy_names)
        ax.set_xlabel('Total Return (%)')
        ax.set_title('Strategy Performance Ranking', fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, (bar, value) in enumerate(zip(bars, returns)):
            width = bar.get_width()
            ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                   f'{value:.2f}%', ha='left', va='center', fontweight='bold')
            
            # Add ranking number
            ax.text(-0.05, bar.get_y() + bar.get_height()/2, 
                   f'#{i+1}', ha='right', va='center', fontweight='bold', fontsize=12)
        
        # Invert y-axis to show best at top
        ax.invert_yaxis()
    
    def _plot_metrics_comparison(self, ax, results: List[ConsolidatedBacktestResult]):
        """Plot key metrics comparison radar chart style"""
        metrics = ['Total Return', 'Win Rate', 'Sharpe Ratio', 'Profit Factor']
        
        # Normalize metrics for comparison (0-100 scale)
        normalized_data = []
        for result in results:
            normalized = [
                max(0, min(100, result.total_return * 10)),  # Scale return
                result.win_rate,  # Already 0-100
                max(0, min(100, result.sharpe_ratio * 10)),  # Scale Sharpe
                max(0, min(100, result.profit_factor * 20))  # Scale profit factor
            ]
            normalized_data.append(normalized)
        
        x = np.arange(len(metrics))
        width = 0.8 / len(results)
        
        for i, (result, data) in enumerate(zip(results, normalized_data)):
            strategy_name = result.strategy_name.replace('_', ' ').title()[:15]
            ax.bar(x + i * width, data, width, label=strategy_name, alpha=0.8)
        
        ax.set_xlabel('Metrics')
        ax.set_ylabel('Normalized Score (0-100)')
        ax.set_title('Key Metrics Comparison', fontsize=16, fontweight='bold')
        ax.set_xticks(x + width * (len(results) - 1) / 2)
        ax.set_xticklabels(metrics, rotation=45, ha='right')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3, axis='y')
    
    def _plot_risk_return_scatter(self, ax, results: List[ConsolidatedBacktestResult]):
        """Plot risk vs return scatter plot"""
        returns = [r.total_return for r in results]
        risks = [r.max_drawdown for r in results]
        strategy_names = [r.strategy_name.replace('_', ' ').title() for r in results]
        
        scatter = ax.scatter(risks, returns, s=200, alpha=0.7, c=range(len(results)), 
                           cmap='viridis', edgecolors='black')
        
        # Add strategy labels
        for i, (risk, ret, name) in enumerate(zip(risks, returns, strategy_names)):
            ax.annotate(f'{i+1}', (risk, ret), xytext=(5, 5), 
                       textcoords='offset points', fontweight='bold', fontsize=10)
        
        ax.set_xlabel('Max Drawdown (%)')
        ax.set_ylabel('Total Return (%)')
        ax.set_title('Risk vs Return Analysis', fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add quadrant lines
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        ax.axvline(x=np.mean(risks), color='blue', linestyle='--', alpha=0.5)
    
    def _plot_winrate_trades(self, ax, results: List[ConsolidatedBacktestResult]):
        """Plot win rate vs total trades"""
        win_rates = [r.win_rate for r in results]
        total_trades = [r.total_trades for r in results]
        strategy_names = [r.strategy_name.replace('_', ' ').title() for r in results]
        
        scatter = ax.scatter(total_trades, win_rates, s=200, alpha=0.7, c=range(len(results)), 
                           cmap='plasma', edgecolors='black')
        
        # Add strategy labels
        for i, (trades, win_rate, name) in enumerate(zip(total_trades, win_rates, strategy_names)):
            ax.annotate(f'{i+1}', (trades, win_rate), xytext=(5, 5), 
                       textcoords='offset points', fontweight='bold', fontsize=10)
        
        ax.set_xlabel('Total Trades')
        ax.set_ylabel('Win Rate (%)')
        ax.set_title('Win Rate vs Trade Frequency', fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    def _plot_sharpe_comparison(self, ax, results: List[ConsolidatedBacktestResult]):
        """Plot Sharpe ratio comparison"""
        strategy_names = [r.strategy_name.replace('_', ' ').title() for r in results]
        sharpe_ratios = [r.sharpe_ratio for r in results]
        
        bars = ax.bar(range(len(strategy_names)), sharpe_ratios, 
                     color='green', alpha=0.7, edgecolor='black')
        ax.set_xticks(range(len(strategy_names)))
        ax.set_xticklabels([f'{i+1}' for i in range(len(strategy_names))])
        ax.set_ylabel('Sharpe Ratio')
        ax.set_title('Risk-Adjusted Returns (Sharpe Ratio)', fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, value in zip(bars, sharpe_ratios):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                   f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
    
    def _plot_drawdown_comparison(self, ax, results: List[ConsolidatedBacktestResult]):
        """Plot drawdown comparison"""
        strategy_names = [r.strategy_name.replace('_', ' ').title() for r in results]
        max_drawdowns = [r.max_drawdown for r in results]
        
        bars = ax.bar(range(len(strategy_names)), max_drawdowns, 
                     color='red', alpha=0.7, edgecolor='black')
        ax.set_xticks(range(len(strategy_names)))
        ax.set_xticklabels([f'{i+1}' for i in range(len(strategy_names))])
        ax.set_ylabel('Max Drawdown (%)')
        ax.set_title('Maximum Drawdown Comparison', fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, value in zip(bars, max_drawdowns):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{value:.2f}%', ha='center', va='bottom', fontweight='bold')
    
    def _plot_trade_analysis_comparison(self, ax, results: List[ConsolidatedBacktestResult]):
        """Plot trade analysis comparison"""
        strategy_names = [r.strategy_name.replace('_', ' ').title() for r in results]
        winning_trades = [r.winning_trades for r in results]
        losing_trades = [r.losing_trades for r in results]
        
        x = np.arange(len(strategy_names))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, winning_trades, width, label='Winning Trades', 
                      color='green', alpha=0.7)
        bars2 = ax.bar(x + width/2, losing_trades, width, label='Losing Trades', 
                      color='red', alpha=0.7)
        
        ax.set_xlabel('Strategy')
        ax.set_ylabel('Number of Trades')
        ax.set_title('Winning vs Losing Trades', fontsize=16, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([f'{i+1}' for i in range(len(strategy_names))])
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                           f'{int(height)}', ha='center', va='bottom', fontweight='bold')
    
    def _plot_detailed_comparison_table(self, ax, results: List[ConsolidatedBacktestResult]):
        """Plot detailed comparison table"""
        ax.axis('off')
        
        # Create comparison data
        table_data = []
        headers = ['Rank', 'Strategy', 'Return (%)', 'Win Rate (%)', 'Trades', 
                  'Sharpe', 'Max DD (%)', 'Profit Factor', 'Avg Win ($)', 'Avg Loss ($)']
        
        table_data.append(headers)
        
        for i, result in enumerate(results):
            row = [
                f'#{i+1}',
                result.strategy_name.replace('_', ' ').title()[:20],
                f'{result.total_return:.2f}',
                f'{result.win_rate:.1f}',
                str(result.total_trades),
                f'{result.sharpe_ratio:.2f}',
                f'{result.max_drawdown:.2f}',
                f'{result.profit_factor:.2f}',
                f'{result.avg_win:.2f}',
                f'{result.avg_loss:.2f}'
            ]
            table_data.append(row)
        
        # Create table
        table = ax.table(cellText=table_data[1:], colLabels=table_data[0],
                        cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
        
        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2.5)
        
        # Color the header
        for i in range(len(headers)):
            table[(0, i)].set_facecolor('#2E8B57')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Color rows based on ranking
        for i in range(1, len(table_data)):
            for j in range(len(headers)):
                if i == 1:  # Winner
                    table[(i, j)].set_facecolor('#FFD700')
                elif i == 2:  # Second
                    table[(i, j)].set_facecolor('#C0C0C0')
                elif i == 3:  # Third
                    table[(i, j)].set_facecolor('#CD7F32')
                elif i % 2 == 0:
                    table[(i, j)].set_facecolor('#f0f0f0')
        
        ax.set_title('Detailed Strategy Comparison', fontsize=18, fontweight='bold', pad=20)
    
    def _print_comparison_summary(self, results: List[ConsolidatedBacktestResult]):
        """Print comparison summary"""
        print(f"\n🏆 STRATEGY COMPARISON SUMMARY")
        print("=" * 60)
        
        # Sort by total return
        results_sorted = sorted(results, key=lambda x: x.total_return, reverse=True)
        
        for i, result in enumerate(results_sorted):
            rank_emoji = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else "📊"
            print(f"{rank_emoji} #{i+1} {result.strategy_name.replace('_', ' ').title()}")
            print(f"   Return: {result.total_return:.2f}% | Win Rate: {result.win_rate:.1f}% | "
                  f"Trades: {result.total_trades} | Sharpe: {result.sharpe_ratio:.2f} | "
                  f"Max DD: {result.max_drawdown:.2f}%")
        
        # Best strategy analysis
        best = results_sorted[0]
        print(f"\n🎯 BEST STRATEGY: {best.strategy_name.replace('_', ' ').title()}")
        print(f"   💰 Total Return: {best.total_return:.2f}%")
        print(f"   🎯 Win Rate: {best.win_rate:.1f}%")
        print(f"   📊 Sharpe Ratio: {best.sharpe_ratio:.2f}")
        print(f"   📉 Max Drawdown: {best.max_drawdown:.2f}%")
        print(f"   🔄 Total Trades: {best.total_trades}")

def main():
    """Main function to run strategy comparison"""
    print("🎯 Strategy Comparison Report Generator")
    print("=" * 50)
    
    # Create comparison system
    comparison_system = StrategyComparisonReport()
    
    # Test configurations
    test_configs = [
        {
            'asset': 'BTCUSDT',
            'timeframe': '4h',
            'initial_capital': 10000
        },
        {
            'asset': 'ETHUSDT',
            'timeframe': '4h',
            'initial_capital': 10000
        }
    ]
    
    for config in test_configs:
        print(f"\n🚀 Generating comparison for {config['asset']} {config['timeframe']}")
        filename = comparison_system.generate_comparison_report(**config)
        
        if filename:
            print(f"✅ Comparison report generated: {filename}")
        else:
            print(f"❌ Failed to generate comparison report")
    
    print(f"\n🎉 Strategy comparison completed!")
    print(f"📁 Reports saved to: {comparison_system.reports_dir}/")

if __name__ == "__main__":
    main()
