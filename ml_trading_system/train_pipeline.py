import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

from ml_trading_system.data_collection import DataCollector
from ml_trading_system.feature_engineering import FeatureEngineer
from ml_trading_system.ml_models import MLTradingModel, EnsembleModel

class ModelTrainer:
    """
    Complete pipeline for training and evaluating ML trading models
    """
    
    def __init__(self):
        self.data_collector = DataCollector()
        self.feature_engineer = FeatureEngineer()
        self.models = {}
        self.ensemble_model = None
        
    def run_complete_pipeline(self, target_samples: int = 10000) -> Dict:
        """
        Run the complete ML training pipeline
        """
        print("🚀 Starting ML Trading Model Training Pipeline")
        print("=" * 50)
        
        # Step 1: Collect data
        print("\n📊 Step 1: Collecting Labeled Historical Data")
        raw_data = self.data_collector.collect_labeled_data(target_samples)
        
        # Step 2: Engineer features
        print("\n🔧 Step 2: Engineering Advanced Features")
        engineered_data = self.feature_engineer.engineer_features(raw_data)
        
        # Step 3: Prepare for training
        print("\n⚙️  Step 3: Preparing Data for Training")
        X, y, feature_names = self.feature_engineer.prepare_features_for_training(engineered_data)
        
        print(f"   Final dataset: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"   Class distribution: {np.mean(y):.2%} profitable trades")
        
        # Step 4: Train multiple models
        print("\n🤖 Step 4: Training Multiple ML Models")
        model_results = {}
        
        # Train XGBoost
        print("\n   Training XGBoost Model...")
        xgb_model = MLTradingModel('xgboost')
        xgb_results = xgb_model.train(X, y, feature_names)
        self.models['xgboost'] = xgb_model
        model_results['xgboost'] = xgb_results
        
        # Train Random Forest
        print("\n   Training Random Forest Model...")
        rf_model = MLTradingModel('random_forest')
        rf_results = rf_model.train(X, y, feature_names)
        self.models['random_forest'] = rf_model
        model_results['random_forest'] = rf_results
        
        # Train Gradient Boosting
        print("\n   Training Gradient Boosting Model...")
        gb_model = MLTradingModel('gradient_boosting')
        gb_results = gb_model.train(X, y, feature_names)
        self.models['gradient_boosting'] = gb_model
        model_results['gradient_boosting'] = gb_results
        
        # Step 5: Create ensemble
        print("\n🤝 Step 5: Creating Ensemble Model")
        self.ensemble_model = EnsembleModel(list(self.models.values()))
        
        # Evaluate ensemble
        print("\n   Evaluating Ensemble Model...")
        ensemble_results = self._evaluate_ensemble(X, y)
        model_results['ensemble'] = ensemble_results
        
        # Step 6: Analyze feature importance
        print("\n🔍 Step 6: Analyzing Feature Importance")
        self._analyze_feature_importance()
        
        # Step 7: Save models
        print("\n💾 Step 7: Saving Trained Models")
        self._save_models()
        
        print("\n✅ Pipeline Completed Successfully!")
        print("=" * 50)
        
        return model_results
        
    def _evaluate_ensemble(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """
        Evaluate the ensemble model
        """
        # Split data for evaluation
        split_idx = int(0.85 * len(X))
        X_test = X[split_idx:]
        y_test = y[split_idx:]
        
        # Get ensemble predictions
        ensemble_proba = self.ensemble_model.predict_proba(X_test)
        ensemble_pred = self.ensemble_model.predict(X_test)
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        metrics = {
            'accuracy': accuracy_score(y_test, ensemble_pred),
            'precision': precision_score(y_test, ensemble_pred, zero_division=0),
            'recall': recall_score(y_test, ensemble_pred, zero_division=0),
            'f1_score': f1_score(y_test, ensemble_pred, zero_division=0),
            'high_confidence_accuracy': np.mean(y_test[ensemble_proba[:, 1] > 0.7] == 1) if np.sum(ensemble_proba[:, 1] > 0.7) > 0 else 0
        }
        
        return {'validation_metrics': metrics}
        
    def _analyze_feature_importance(self):
        """
        Analyze and display feature importance from models
        """
        for model_name, model in self.models.items():
            if hasattr(model, 'get_feature_importance'):
                try:
                    importance_df = model.get_feature_importance()
                    if not importance_df.empty:
                        print(f"\n   Top 10 Features - {model_name.upper()}:")
                        print(importance_df.head(10))
                except Exception as e:
                    print(f"   Could not get feature importance for {model_name}: {e}")
                    
    def _save_models(self):
        """
        Save all trained models
        """
        import os
        os.makedirs('ml_models', exist_ok=True)
        
        for model_name, model in self.models.items():
            try:
                filepath = f'ml_models/{model_name}_model.pkl'
                model.save_model(filepath)
            except Exception as e:
                print(f"   Failed to save {model_name} model: {e}")
                
        # Save engineered data for reference
        try:
            # This would be the raw engineered data, but we don't have access to it here
            # In a real implementation, we'd save the engineered features
            pass
        except Exception as e:
            print(f"   Failed to save engineered data: {e}")
            
    def print_results_summary(self, results: Dict):
        """
        Print a summary of model performance
        """
        print("\n📈 MODEL PERFORMANCE SUMMARY")
        print("=" * 50)
        
        for model_name, result in results.items():
            metrics = result.get('validation_metrics', {})
            print(f"\n{model_name.upper()} MODEL:")
            print(f"  Accuracy:  {metrics.get('accuracy', 0):.4f}")
            print(f"  Precision: {metrics.get('precision', 0):.4f}")
            print(f"  Recall:    {metrics.get('recall', 0):.4f}")
            print(f"  F1-Score:  {metrics.get('f1_score', 0):.4f}")
            
            if 'cv_mean' in result:
                print(f"  CV Score:  {result['cv_mean']:.4f} (±{result['cv_std']:.4f})")
                
            if 'high_confidence_accuracy' in metrics:
                print(f"  High Conf. Accuracy: {metrics['high_confidence_accuracy']:.4f} "
                      f"({metrics.get('high_confidence_count', 0)} trades)")

def main():
    """
    Main execution function
    """
    trainer = ModelTrainer()
    
    try:
        # Run the complete pipeline
        results = trainer.run_complete_pipeline(target_samples=5000)
        
        # Print summary
        trainer.print_results_summary(results)
        
        print("\n🎉 Training pipeline completed successfully!")
        print("📁 Models saved in 'ml_models' directory")
        print("📊 Check the detailed results above for performance metrics")
        
    except Exception as e:
        print(f"❌ Error during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()