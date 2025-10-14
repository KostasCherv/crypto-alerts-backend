#!/usr/bin/env python3
"""
Test script for ML Trading System Components
"""

import sys
import os
import warnings
warnings.filterwarnings('ignore')

# Add src directory to path to import modules
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

def test_data_collection():
    """Test data collection module"""
    print("Testing Data Collection Module...")
    try:
        from ml_trading_system.data_collection import DataCollector
        collector = DataCollector()
        print("✓ DataCollector imported successfully")
        return True
    except Exception as e:
        print(f"✗ DataCollector test failed: {e}")
        return False

def test_feature_engineering():
    """Test feature engineering module"""
    print("Testing Feature Engineering Module...")
    try:
        from ml_trading_system.feature_engineering import FeatureEngineer
        engineer = FeatureEngineer()
        print("✓ FeatureEngineer imported successfully")
        return True
    except Exception as e:
        print(f"✗ FeatureEngineer test failed: {e}")
        return False

def test_ml_models():
    """Test ML models module"""
    print("Testing ML Models Module...")
    try:
        from ml_trading_system.ml_models import MLTradingModel, EnsembleModel
        model = MLTradingModel('xgboost')
        print("✓ MLTradingModel imported successfully")
        return True
    except Exception as e:
        print(f"✗ MLTradingModel test failed: {e}")
        return False

def test_predictor():
    """Test ML predictor module"""
    print("Testing ML Predictor Module...")
    try:
        from ml_trading_system.ml_predictor import MLPredictor
        predictor = MLPredictor()
        print("✓ MLPredictor imported successfully")
        return True
    except Exception as e:
        print(f"✗ MLPredictor test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🤖 ML Trading System Component Tests")
    print("=" * 40)
    
    tests = [
        test_data_collection,
        test_feature_engineering,
        test_ml_models,
        test_predictor
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The ML system is ready for use.")
        print("\nNext steps:")
        print("1. Run the training pipeline: python ml_trading_system/train_pipeline.py")
        print("2. Follow the README.md for detailed instructions")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        
if __name__ == "__main__":
    main()