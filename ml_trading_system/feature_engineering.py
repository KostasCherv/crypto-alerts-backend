import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

class FeatureEngineer:
    """
    Advanced feature engineering for trading pattern recognition
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.pca = None
        self.feature_columns = []
        
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer comprehensive features from raw price data
        """
        print("Engineering features...")
        
        # Create copy to avoid modifying original
        df_features = df.copy()
        
        # Handle categorical features
        df_features = self._encode_categorical_features(df_features)
        
        # Add advanced technical indicators
        df_features = self._add_advanced_indicators(df_features)
        
        # Add pattern recognition features
        df_features = self._add_pattern_features(df_features)
        
        # Add market microstructure features
        df_features = self._add_microstructure_features(df_features)
        
        # Add statistical features
        df_features = self._add_statistical_features(df_features)
        
        print(f"Engineered {len(df_features.columns)} features")
        return df_features
        
    def _encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode categorical features
        """
        df_encoded = df.copy()
        
        # Encode volatility regime
        if 'volatility_regime' in df_encoded.columns:
            le = LabelEncoder()
            df_encoded['volatility_regime_encoded'] = le.fit_transform(
                df_encoded['volatility_regime'].fillna('medium')
            )
            self.label_encoders['volatility_regime'] = le
            
        return df_encoded
        
    def _add_advanced_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add advanced technical indicators
        """
        df_indicators = df.copy()
        
        # Price momentum indicators
        for period in [5, 10, 15, 20]:
            if 'price_change_' + str(period) not in df_indicators.columns:
                if len(df_indicators) > period:
                    df_indicators['price_change_' + str(period)] = df_indicators['current_price'].pct_change(period)
        
        # Moving average convergence divergence (extended)
        if 'macd_histogram' in df_indicators.columns:
            df_indicators['macd_momentum'] = df_indicators['macd_histogram'].diff()
            df_indicators['macd_trend'] = np.where(df_indicators['macd_histogram'] > 0, 1, -1)
            
        # RSI momentum
        if 'rsi' in df_indicators.columns:
            df_indicators['rsi_momentum'] = df_indicators['rsi'].diff()
            df_indicators['rsi_overbought'] = np.where(df_indicators['rsi'] > 70, 1, 0)
            df_indicators['rsi_oversold'] = np.where(df_indicators['rsi'] < 30, 1, 0)
            
        # ADX trend strength
        if 'adx' in df_indicators.columns:
            df_indicators['adx_trend_strength'] = np.where(df_indicators['adx'] > 25, 1, 0)
            
        # Bollinger Band signals
        if 'bb_percent_b' in df_indicators.columns:
            df_indicators['bb_signal'] = np.where(df_indicators['bb_percent_b'] < 0.1, 1, 0)  # Oversold
            df_indicators['bb_signal'] = np.where(df_indicators['bb_percent_b'] > 0.9, -1, df_indicators['bb_signal'])  # Overbought
            
        return df_indicators
        
    def _add_pattern_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add candlestick and chart pattern recognition features
        """
        df_patterns = df.copy()
        
        # Enhanced candlestick patterns
        if all(col in df_patterns.columns for col in ['candle_body', 'upper_wick', 'lower_wick']):
            # Doji pattern
            df_patterns['is_doji'] = np.where(
                df_patterns['candle_body'] < (df_patterns['upper_wick'] + df_patterns['lower_wick']) * 0.1, 1, 0
            )
            
            # Long-legged doji
            df_patterns['is_long_legged_doji'] = np.where(
                (df_patterns['upper_wick'] > df_patterns['candle_body'] * 2) & 
                (df_patterns['lower_wick'] > df_patterns['candle_body'] * 2) & 
                (df_patterns['candle_body'] < df_patterns['current_price'] * 0.005), 1, 0
            )
            
            # Dragonfly doji
            df_patterns['is_dragonfly_doji'] = np.where(
                (df_patterns['lower_wick'] > df_patterns['candle_body'] * 3) & 
                (df_patterns['upper_wick'] < df_patterns['candle_body'] * 0.5) & 
                (df_patterns['candle_body'] < df_patterns['current_price'] * 0.005), 1, 0
            )
            
            # Gravestone doji
            df_patterns['is_gravestone_doji'] = np.where(
                (df_patterns['upper_wick'] > df_patterns['candle_body'] * 3) & 
                (df_patterns['lower_wick'] < df_patterns['candle_body'] * 0.5) & 
                (df_patterns['candle_body'] < df_patterns['current_price'] * 0.005), 1, 0
            )
            
        # Price pattern features
        if 'current_price' in df_patterns.columns:
            # Price relative to support/resistance
            df_patterns['price_to_support'] = df_patterns.get('support_distance', 0)
            df_patterns['price_to_resistance'] = df_patterns.get('resistance_distance', 0)
            
        return df_patterns
        
    def _add_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add market microstructure features
        """
        df_micro = df.copy()
        
        # Volume-weighted features
        if 'volume_ratio' in df_micro.columns:
            df_micro['volume_trend_strength'] = np.where(df_micro['volume_ratio'] > 1.5, 1, 0)
            
        # Price impact estimation
        if all(col in df_micro.columns for col in ['price_change_5', 'volume_ratio']):
            df_micro['price_volume_impact'] = df_micro['price_change_5'] * df_micro['volume_ratio']
            
        return df_micro
        
    def _add_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add statistical features for pattern recognition
        """
        df_stats = df.copy()
        
        # Z-score normalization for price levels
        if 'current_price' in df_stats.columns:
            df_stats['price_zscore'] = (df_stats['current_price'] - df_stats['current_price'].mean()) / df_stats['current_price'].std()
            
        # Volatility clustering
        if 'volatility_20' in df_stats.columns:
            df_stats['volatility_zscore'] = (df_stats['volatility_20'] - df_stats['volatility_20'].mean()) / df_stats['volatility_20'].std()
            df_stats['volatility_quartile'] = pd.qcut(df_stats['volatility_20'], 4, labels=[1, 2, 3, 4])
            
        # Time-based cyclical features
        if 'hour_of_day' in df_stats.columns:
            df_stats['hour_sin'] = np.sin(2 * np.pi * df_stats['hour_of_day'] / 24)
            df_stats['hour_cos'] = np.cos(2 * np.pi * df_stats['hour_of_day'] / 24)
            
        if 'day_of_week' in df_stats.columns:
            df_stats['day_sin'] = np.sin(2 * np.pi * df_stats['day_of_week'] / 7)
            df_stats['day_cos'] = np.cos(2 * np.pi * df_stats['day_of_week'] / 7)
            
        return df_stats
        
    def prepare_features_for_training(self, df: pd.DataFrame, target_column: str = 'label') -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Prepare features for ML training with scaling and feature selection
        """
        print("Preparing features for training...")
        
        # Select numeric features only
        numeric_df = df.select_dtypes(include=[np.number])
        
        # Remove target and identifier columns
        feature_columns = [col for col in numeric_df.columns if col not in [target_column, 'pnl', 'pnl_percent']]
        X = numeric_df[feature_columns]
        
        # Handle missing values
        X = X.fillna(0)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Store feature names
        self.feature_columns = feature_columns
        
        # Target variable
        y = df[target_column].values if target_column in df.columns else np.zeros(len(df))
        
        print(f"Prepared {X_scaled.shape[1]} features for {X_scaled.shape[0]} samples")
        return X_scaled, y, feature_columns
        
    def reduce_dimensions(self, X: np.ndarray, n_components: int = 50) -> np.ndarray:
        """
        Apply PCA for dimensionality reduction
        """
        print(f"Reducing dimensions to {n_components} components...")
        
        self.pca = PCA(n_components=n_components)
        X_reduced = self.pca.fit_transform(X)
        
        print(f"Explained variance ratio: {self.pca.explained_variance_ratio_.sum():.4f}")
        return X_reduced
        
    def get_feature_importance(self, feature_importances: np.ndarray) -> pd.DataFrame:
        """
        Get feature importance rankings
        """
        if not self.feature_columns:
            raise ValueError("Features not prepared yet")
            
        importance_df = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': feature_importances
        }).sort_values('importance', ascending=False)
        
        return importance_df

if __name__ == "__main__":
    # Test the feature engineering
    print("Feature Engineering Module Ready")