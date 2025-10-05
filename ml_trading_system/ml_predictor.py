import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

from ml_trading_system.ml_models import MLTradingModel, EnsembleModel
from ml_trading_system.feature_engineering import FeatureEngineer
from trend_analysis import TrendAnalyzer

class MLPredictor:
    """
    Real-time ML prediction system for trading signals
    """
    
    def __init__(self, model_path: str = 'ml_models'):
        self.model_path = model_path
        self.models = {}
        self.ensemble_model = None
        self.feature_engineer = FeatureEngineer()
        self.trend_analyzer = TrendAnalyzer()
        self.is_loaded = False
        
    def load_models(self):
        """
        Load trained models from disk
        """
        print("Loading ML models...")
        
        try:
            # Load individual models
            model_types = ['xgboost', 'random_forest', 'gradient_boosting']
            
            for model_type in model_types:
                try:
                    model = MLTradingModel(model_type)
                    model_file = f'{self.model_path}/{model_type}_model.pkl'
                    model.load_model(model_file)
                    self.models[model_type] = model
                    print(f"   ✓ Loaded {model_type} model")
                except Exception as e:
                    print(f"   ✗ Failed to load {model_type} model: {e}")
            
            # Create ensemble if we have models
            if self.models:
                self.ensemble_model = EnsembleModel(list(self.models.values()))
                print("   ✓ Created ensemble model")
                self.is_loaded = True
            else:
                print("   ⚠️  No models loaded")
                
        except Exception as e:
            print(f"Error loading models: {e}")
            
    def extract_features_from_current_data(self, symbol: str, interval: str = "15m", 
                                          lookback_periods: int = 50) -> Dict:
        """
        Extract features from current market data for prediction
        """
        try:
            # Fetch recent historical data
            opens, highs, lows, closes, volumes = self.trend_analyzer.fetch_historical_prices(
                symbol, limit=lookback_periods, interval=interval
            )
            
            if not closes:
                return None
                
            # Convert to lists of floats for processing
            closes_float = [float(c) for c in closes]
            highs_float = [float(h) for h in highs]
            lows_float = [float(l) for l in lows]
            volumes_float = volumes
            opens_float = [float(o) for o in opens]
            
            # Create a simple dataframe-like structure for feature extraction
            features = {}
            
            # Basic price features
            features['current_price'] = closes_float[-1]
            features['price_change_5'] = (closes_float[-1] - closes_float[-6]) / closes_float[-6] if len(closes_float) >= 6 else 0
            features['price_change_10'] = (closes_float[-1] - closes_float[-11]) / closes_float[-11] if len(closes_float) >= 11 else 0
            features['price_change_20'] = (closes_float[-1] - closes_float[-21]) / closes_float[-21] if len(closes_float) >= 21 else 0
            
            # Volatility features
            features['volatility_10'] = np.std(closes_float[-10:]) if len(closes_float) >= 10 else 0
            features['volatility_20'] = np.std(closes_float[-20:]) if len(closes_float) >= 20 else 0
            features['price_range'] = (max(highs_float) - min(lows_float)) / closes_float[-1]
            
            # Volume features
            features['volume_trend'] = (volumes_float[-1] - np.mean(volumes_float[-5:])) / np.mean(volumes_float[-5:]) if len(volumes_float) >= 5 and np.mean(volumes_float[-5:]) > 0 else 0
            features['volume_ratio'] = volumes_float[-1] / np.mean(volumes_float[-20:]) if len(volumes_float) >= 20 and np.mean(volumes_float[-20:]) > 0 else 1
            
            # Technical indicators
            try:
                # RSI
                rsi = self.trend_analyzer.calculate_rsi(closes_float, period=14) if len(closes_float) >= 14 else 50
                features['rsi'] = rsi
                
                # MACD
                if len(closes_float) >= 35:
                    macd_line, signal_line, histogram = self.trend_analyzer.calculate_macd(closes_float)
                    features['macd_line'] = macd_line
                    features['macd_signal'] = signal_line
                    features['macd_histogram'] = histogram
                else:
                    features['macd_line'] = 0
                    features['macd_signal'] = 0
                    features['macd_histogram'] = 0
                    
                # ADX
                if len(closes_float) >= 20:
                    adx, plus_di, minus_di = self.trend_analyzer.calculate_adx(highs_float, lows_float, closes_float)
                    features['adx'] = adx
                    features['plus_di'] = plus_di
                    features['minus_di'] = minus_di
                else:
                    features['adx'] = 0
                    features['plus_di'] = 0
                    features['minus_di'] = 0
                    
                # Bollinger Bands
                if len(closes_float) >= 20:
                    bb = self.trend_analyzer.calculate_bollinger_bands(closes_float)
                    features['bb_width'] = bb.get('bandwidth', 0)
                    features['bb_percent_b'] = bb.get('percent_b', 0.5)
                    features['bb_upper'] = bb.get('upper_band', closes_float[-1])
                    features['bb_lower'] = bb.get('lower_band', closes_float[-1])
                    features['bb_middle'] = bb.get('middle_band', closes_float[-1])
                    features['bb_squeeze'] = 1 if bb.get('squeeze', False) else 0
                else:
                    features['bb_width'] = 0
                    features['bb_percent_b'] = 0.5
                    features['bb_upper'] = closes_float[-1]
                    features['bb_lower'] = closes_float[-1]
                    features['bb_middle'] = closes_float[-1]
                    features['bb_squeeze'] = 0
                    
                # EMA differences
                if len(closes_float) >= 20:
                    ema9 = self.trend_analyzer.calculate_ema(closes_float, 9)
                    ema21 = self.trend_analyzer.calculate_ema(closes_float, 21)
                    ema50 = self.trend_analyzer.calculate_ema(closes_float, 50)
                    
                    if ema9 and ema21 and ema50:
                        features['ema9_21_diff'] = (ema9[-1] - ema21[-1]) / closes_float[-1]
                        features['ema21_50_diff'] = (ema21[-1] - ema50[-1]) / closes_float[-1]
                        features['price_ema21_diff'] = (closes_float[-1] - ema21[-1]) / closes_float[-1]
                        
                # Support/Resistance
                sr = self.trend_analyzer.find_support_resistance(highs_float, lows_float, closes_float)
                nearest_support = sr.get('nearest_support', closes_float[-1] * 0.95)
                nearest_resistance = sr.get('nearest_resistance', closes_float[-1] * 0.05)
                
                features['support_distance'] = (closes_float[-1] - nearest_support) / closes_float[-1]
                features['resistance_distance'] = (nearest_resistance - closes_float[-1]) / closes_float[-1]
                features['support_resistance_width'] = (nearest_resistance - nearest_support) / closes_float[-1]
                
                # Candlestick patterns
                features['candle_body'] = abs(closes_float[-1] - opens_float[-1]) / closes_float[-1]
                features['upper_wick'] = (highs_float[-1] - max(closes_float[-1], opens_float[-1])) / closes_float[-1]
                features['lower_wick'] = (min(closes_float[-1], opens_float[-1]) - lows_float[-1]) / closes_float[-1]
                
                # Simple patterns
                features['is_hammer'] = 1 if (features['lower_wick'] > features['candle_body'] * 2 and 
                                            features['upper_wick'] < features['candle_body'] * 0.5) else 0
                features['is_shooting_star'] = 1 if (features['upper_wick'] > features['candle_body'] * 2 and 
                                                   features['lower_wick'] < features['candle_body'] * 0.5) else 0
                features['is_engulfing'] = 1 if (len(closes_float) >= 2 and 
                                               abs(closes_float[-1] - opens_float[-1]) > abs(closes_float[-2] - opens_float[-2]) and
                                               (closes_float[-1] > opens_float[-1]) != (closes_float[-2] > opens_float[-2])) else 0
                
            except Exception as e:
                print(f"Warning: Could not calculate all indicators: {e}")
                # Fill missing features with defaults
                for key in ['rsi', 'macd_line', 'macd_signal', 'macd_histogram', 'adx', 
                           'plus_di', 'minus_di', 'bb_width', 'bb_percent_b']:
                    if key not in features:
                        features[key] = 0
                        
            # Time-based features (using current time as proxy)
            from datetime import datetime
            current_time = datetime.now()
            features['hour_of_day'] = current_time.hour
            features['day_of_week'] = current_time.weekday()
            features['month'] = current_time.month
            
            # Volatility regime
            if features['volatility_20'] > np.percentile([features['volatility_20']], 75) if features['volatility_20'] > 0 else False:
                features['volatility_regime'] = 'high'
            elif features['volatility_20'] < np.percentile([features['volatility_20']], 25) if features['volatility_20'] > 0 else False:
                features['volatility_regime'] = 'low'
            else:
                features['volatility_regime'] = 'medium'
                
            return features
            
        except Exception as e:
            print(f"Error extracting features: {e}")
            return None
            
    def predict_trade_profitability(self, symbol: str, interval: str = "15m") -> Dict:
        """
        Predict the profitability of a potential trade
        """
        if not self.is_loaded:
            self.load_models()
            
        if not self.is_loaded:
            return {"error": "Models not loaded"}
            
        # Extract features from current data
        features = self.extract_features_from_current_data(symbol, interval)
        if not features:
            return {"error": "Could not extract features"}
            
        # Convert features to array format
        # This is a simplified approach - in practice, you'd want to ensure
        # the same features used in training are provided in the same order
        feature_df = pd.DataFrame([features])
        
        # Apply the same feature engineering as during training
        try:
            engineered_features = self.feature_engineer.engineer_features(feature_df)
            
            # Select only numeric features that were used in training
            if hasattr(self.feature_engineer, 'feature_columns') and self.feature_engineer.feature_columns:
                # Use the feature columns from training
                feature_names = self.feature_engineer.feature_columns
            else:
                # Fallback to numeric columns
                numeric_features = engineered_features.select_dtypes(include=[np.number])
                feature_names = [col for col in numeric_features.columns if col not in ['label', 'pnl', 'pnl_percent']]
                
            # Create feature array in correct order
            X = np.zeros((1, len(feature_names)))
            for i, feature_name in enumerate(feature_names):
                if feature_name in engineered_features.columns:
                    X[0, i] = engineered_features.iloc[0][feature_name]
                # Missing features default to 0
                    
            # Make predictions with all models
            predictions = {}
            probabilities = {}
            
            # Individual model predictions
            for model_name, model in self.models.items():
                try:
                    pred_proba = model.predict_proba(X)
                    probabilities[model_name] = pred_proba[0, 1]  # Probability of profitable trade
                    predictions[model_name] = model.predict(X)[0]
                except Exception as e:
                    print(f"Error with {model_name} prediction: {e}")
                    probabilities[model_name] = 0.0
                    predictions[model_name] = 0
                    
            # Ensemble prediction
            try:
                ensemble_proba = self.ensemble_model.predict_proba(X)
                ensemble_prob = ensemble_proba[0, 1]
                ensemble_pred = self.ensemble_model.predict(X)[0]
                probabilities['ensemble'] = ensemble_prob
                predictions['ensemble'] = ensemble_pred
            except Exception as e:
                print(f"Error with ensemble prediction: {e}")
                probabilities['ensemble'] = 0.0
                predictions['ensemble'] = 0
                
            # Calculate confidence score (0-100)
            ensemble_confidence = probabilities.get('ensemble', 0.0)
            confidence_score = int(ensemble_confidence * 100)
            
            # Determine trading signal based on ML prediction
            if ensemble_confidence > 0.7:
                ml_signal = "STRONG_BUY" if ensemble_pred == 1 else "STRONG_SELL"
            elif ensemble_confidence > 0.6:
                ml_signal = "BUY" if ensemble_pred == 1 else "SELL"
            else:
                ml_signal = "HOLD"
                
            return {
                'symbol': symbol,
                'interval': interval,
                'ml_signal': ml_signal,
                'confidence_score': confidence_score,
                'ensemble_probability': ensemble_prob,
                'individual_probabilities': probabilities,
                'individual_predictions': predictions,
                'features_used': len(feature_names),
                'timestamp': pd.Timestamp.now().isoformat()
            }
            
        except Exception as e:
            print(f"Error during prediction: {e}")
            return {"error": f"Prediction error: {e}"}
            
    def integrate_with_confidence_system(self, base_confidence: int, ml_prediction: Dict) -> int:
        """
        Integrate ML prediction with existing confidence scoring system
        """
        if 'confidence_score' not in ml_prediction:
            return base_confidence
            
        ml_confidence = ml_prediction['confidence_score']
        
        # If ML confidence is high (>70%), add bonus points
        if ml_confidence > 70:
            bonus = 20  # +20 points as specified in requirements
            return min(100, base_confidence + bonus)
        elif ml_confidence > 60:
            bonus = 10  # +10 points for moderate confidence
            return min(100, base_confidence + bonus)
        else:
            return base_confidence

def main():
    """
    Example usage of the ML predictor
    """
    predictor = MLPredictor()
    
    # In a real scenario, you would load pre-trained models
    # For now, we'll just show the structure
    print("ML Predictor initialized")
    print("To use this system:")
    print("1. Train models using train_pipeline.py")
    print("2. Load models with load_models()")
    print("3. Get predictions with predict_trade_profitability()")
    
    # Example of how it would work:
    # predictor.load_models()
    # prediction = predictor.predict_trade_profitability("BTCUSDT")
    # print(prediction)

if __name__ == "__main__":
    main()