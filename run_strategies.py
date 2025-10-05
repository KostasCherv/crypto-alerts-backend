#!/usr/bin/env python3
"""
Quick runner script for the crypto trading strategy system.
This script handles the path setup and runs the main strategy analysis.
"""

import sys
import os
from src.strategies import main
# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import and run strategies
if __name__ == "__main__":
    try:        
        # Execute the main function
        print("🚀 Running Trading Strategies Analysis...")
        print("=" * 60)
        
        # Call the main function from strategies module
        main()
        
    except ImportError as e:
        print(f"Error importing strategies: {e}")
        print("Make sure you're in the project root directory and have installed dependencies with 'uv sync'")
        sys.exit(1)
    except Exception as e:
        print(f"Error running strategies: {e}")
        sys.exit(1)
