#!/usr/bin/env python3
"""
Quick runner script for the visual backtesting system.
This script handles the path setup and runs the consolidated visual backtest.
"""

import sys
import os

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import and run visual backtest
if __name__ == "__main__":
    try:
        from consolidated_visual_backtest import main
        main()
    except ImportError as e:
        print(f"Error importing backtest system: {e}")
        print("Make sure you're in the project root directory and have installed dependencies with 'uv sync'")
        sys.exit(1)
    except Exception as e:
        print(f"Error running backtest: {e}")
        sys.exit(1)
