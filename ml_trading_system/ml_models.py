import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import joblib
import os

class MLTradingModel:
    """
    Machine Learning models for trading pattern recognition
    """
    
    def __init__(self, model_type: str = 'xgboost'):
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self.is_trained = False
        
        # Model parameters
        self.model_params = {
            'xgboost': {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42
            },
            'random_forest': {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'random_state': 42
            },
            'gradient_boosting': {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'random_state': 42
            }
        }
        
    def _create_model(self):
        """
        Create the specified ML model
        """
        if self.model_type == 'xgboost':
            self.model = xgb.XGBClassifier(**self.model_params['xgboost'])
        elif self.model_type == 'random_forest':
            self.model = RandomForestClassifier(**self.model_params['random_forest'])
        elif self.model_type == 'gradient_boosting':
            self.model = GradientBoostingClassifier(**self.model_params['gradient_boosting'])
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
            
    def train(self, X: np.ndarray, y: np.ndarray, feature_names: List[str], 
              test_size: float = 0.15, validation_size: float = 0.15) -> Dict:
        """
        Train the ML model with proper validation
        """
        print(f"Training {self.model_type} model...")
        print(f"Data shape: {X.shape}")
        
        # Store feature names
        self.feature_names = feature_names
        
        # Create model
        self._create_model()
        
        # Split data maintaining temporal order (70/15/15)
        # First split: 85% train+val, 15% test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, shuffle=False, random_state=42
        )
        
        # Second split: 70% train, 15% validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=validation_size/(1-test_size), 
            shuffle=False, random_state=42
        )
        
        print(f"Train set: {X_train.shape[0]} samples")
        print(f"Validation set: {X_val.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        print("Training model...")
        self.model.fit(X_train_scaled, y_train)
        
        # Validate on validation set
        print("Validating model...")
        val_predictions = self.model.predict(X_val_scaled)
        val_probabilities = self.model.predict_proba(X_val_scaled)[:, 1]
        
        # Test on test set
        print("Testing model...")
        test_predictions = self.model.predict(X_test_scaled)
        test_probabilities = self.model.predict_proba(X_test_scaled)[:, 1]
        
        # Calculate metrics
        val_metrics = self._calculate_metrics(y_val, val_predictions, val_probabilities)
        test_metrics = self._calculate_metrics(y_test, test_predictions, test_probabilities)
        
        # Cross-validation with time series split
        print("Performing time series cross-validation...")
        tscv = TimeSeriesSplit(n_splits=5)
        cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=tscv, scoring='f1')
        
        # Store results
        results = {
            'model_type': self.model_type,
            'validation_metrics': val_metrics,
            'test_metrics': test_metrics,
            'cv_scores': cv_scores,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'train_size': len(X_train),
            'val_size': len(X_val),
            'test_size': len(X_test)
        }
        
        self.is_trained = True
        
        print("Training completed!")
        return results
        
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                          y_proba: np.ndarray) -> Dict:
        """
        Calculate comprehensive metrics for model evaluation
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0),
        }
        
        # Add confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm
        
        # Calculate profit factor if we have probability scores
        if len(y_proba) > 0:
            # Simulate trading based on probability threshold
            threshold = 0.7  # High confidence threshold
            high_confidence_mask = y_proba >= threshold
            if np.sum(high_confidence_mask) > 0:
                high_confidence_accuracy = np.mean(y_true[high_confidence_mask] == 1)
                metrics['high_confidence_accuracy'] = high_confidence_accuracy
                metrics['high_confidence_count'] = np.sum(high_confidence_mask)
        
        return metrics
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data
        """
        if not self.is_trained:
            raise ValueError("Model not trained yet")
            
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
        
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get prediction probabilities
        """
        if not self.is_trained:
            raise ValueError("Model not trained yet")
            
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)
        
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance from the trained model
        """
        if not self.is_trained:
            raise ValueError("Model not trained yet")
            
        if hasattr(self.model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            return importance_df
        else:
            return pd.DataFrame()
            
    def save_model(self, filepath: str):
        """
        Save the trained model and scaler
        """
        if not self.is_trained:
            raise ValueError("Model not trained yet")
            
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save model components
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'model_type': self.model_type,
            'is_trained': self.is_trained
        }
        
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
        
    def load_model(self, filepath: str):
        """
        Load a trained model
        """
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.model_type = model_data['model_type']
        self.is_trained = model_data['is_trained']
        
        print(f"Model loaded from {filepath}")
        
    def create_ensemble(self, models: List['MLTradingModel']) -> 'EnsembleModel':
        """
        Create an ensemble of multiple models
        """
        return EnsembleModel(models)

class EnsembleModel:
    """
    Ensemble of multiple ML models for robust predictions
    """
    
    def __init__(self, models: List[MLTradingModel]):
        self.models = models
        self.is_trained = all(model.is_trained for model in models)
        
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get ensemble prediction probabilities (average of all models)
        """
        if not self.is_trained:
            raise ValueError("Not all models are trained")
            
        # Get predictions from all models
        predictions = []
        for model in self.models:
            pred_proba = model.predict_proba(X)
            predictions.append(pred_proba[:, 1])  # Probability of positive class
            
        # Average predictions
        ensemble_proba = np.mean(predictions, axis=0)
        
        # Return in same format as sklearn predict_proba
        result = np.column_stack([1 - ensemble_proba, ensemble_proba])
        return result
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make ensemble predictions (majority vote)
        """
        pred_proba = self.predict_proba(X)
        return (pred_proba[:, 1] > 0.5).astype(int)

if __name__ == "__main__":
    print("ML Trading Model Module Ready")