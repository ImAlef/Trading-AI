import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb
import joblib
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import logging
from config import config

logger = logging.getLogger(__name__)

class MLModel:
    """
    Machine Learning Model for Trading Signal Prediction
    """
    
    def __init__(self):
        self.model = None
        self.feature_columns = None
        self.is_trained = False
        self.model_info = {}
        
        # Create models directory if it doesn't exist
        os.makedirs("data/models", exist_ok=True)
        
    def prepare_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare data for training/prediction
        """
        try:
            # Remove rows with NaN values
            df_clean = df.dropna()
            
            # Get feature columns (exclude OHLCV and target columns)
            exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'target', 'target_return', 'future_return']
            feature_cols = [col for col in df_clean.columns if col not in exclude_cols]
            
            # Extract features and target
            X = df_clean[feature_cols]
            y = df_clean['target'] if 'target' in df_clean.columns else None
            
            # Store feature columns for future use
            self.feature_columns = feature_cols
            
            logger.info(f"Prepared data: {len(X)} samples, {len(feature_cols)} features")
            
            if y is not None:
                positive_samples = y.sum()
                logger.info(f"Target distribution: {positive_samples}/{len(y)} ({positive_samples/len(y)*100:.1f}% positive)")
            
            return X, y
            
        except Exception as e:
            logger.error(f"Error preparing data: {str(e)}")
            raise
    
    def train(self, df: pd.DataFrame) -> Dict:
        """
        Train the ML model
        """
        try:
            logger.info("Starting model training...")
            
            # Prepare data
            X, y = self.prepare_data(df)
            
            if y is None:
                raise ValueError("No target variable found in data")
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=config.TRAIN_TEST_SPLIT, random_state=42, stratify=y
            )
            
            # Calculate class weights for imbalanced data
            pos_weight = len(y_train) / (2 * y_train.sum())
            logger.info(f"Applied class weighting: {pos_weight:.2f}")
            
            # Initialize XGBoost model
            self.model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                scale_pos_weight=pos_weight,
                eval_metric='logloss'
            )
            
            # Train model
            self.model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = self.model.predict(X_test)
            y_pred_proba = self.model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            
            # Cross-validation
            cv_scores = cross_val_score(self.model, X_train, y_train, cv=5, scoring='accuracy')
            
            # Store model info
            version = datetime.now().strftime("v%Y%m%d_%H%M%S")
            self.model_info = {
                'version': version,
                'features': len(self.feature_columns),
                'training_samples': len(X_train),
                'test_samples': len(X_test),
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'positive_samples': y_train.sum(),
                'trained_at': datetime.now().isoformat(),
                'model_type': 'XGBoost'
            }
            
            self.is_trained = True
            
            logger.info("Model training completed successfully!")
            logger.info(f"Accuracy: {accuracy:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")
            
            return self.model_info
            
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            raise
    
    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions on new data
        """
        try:
            if not self.is_trained:
                raise ValueError("Model is not trained yet")
            
            # Prepare data
            X_prepared, _ = self.prepare_data(X)
            
            logger.info(f"Prepared data: {len(X_prepared)} samples, {len(self.feature_columns)} features")
            
            # Make predictions
            predictions = self.model.predict(X_prepared)
            probabilities = self.model.predict_proba(X_prepared)[:, 1]
            
            return predictions, probabilities
            
        except Exception as e:
            logger.error(f"Error making predictions: {str(e)}")
            raise
    
    def predict_single(self, X: pd.DataFrame) -> Tuple[int, float]:
        """
        Make prediction for a single sample
        """
        try:
            if not self.is_trained:
                raise ValueError("Model is not trained yet")
            
            # Ensure we have the right features
            if self.feature_columns is None:
                raise ValueError("Feature columns not defined")
            
            # Select only the features used in training
            available_features = [col for col in self.feature_columns if col in X.columns]
            
            if len(available_features) != len(self.feature_columns):
                missing_features = set(self.feature_columns) - set(available_features)
                logger.warning(f"Missing features: {missing_features}")
                
                # Add missing features with default values
                for feature in missing_features:
                    X[feature] = 0
            
            # Select features in the correct order
            X_features = X[self.feature_columns]
            
            # Make prediction
            prediction = self.model.predict(X_features)[0]
            probability = self.model.predict_proba(X_features)[0, 1]
            
            return int(prediction), float(probability)
            
        except Exception as e:
            logger.error(f"Error making single prediction: {str(e)}")
            raise
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance from trained model
        """
        try:
            if not self.is_trained:
                raise ValueError("Model is not trained yet")
            
            importance = self.model.feature_importances_
            feature_importance = dict(zip(self.feature_columns, importance))
            
            # Sort by importance
            sorted_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
            
            return sorted_importance
            
        except Exception as e:
            logger.error(f"Error getting feature importance: {str(e)}")
            return {}
    
    def save_model(self, filename: str = None) -> str:
        """
        Save the trained model
        """
        try:
            if not self.is_trained:
                raise ValueError("Model is not trained yet")
            
            if filename is None:
                version = self.model_info.get('version', datetime.now().strftime("v%Y%m%d_%H%M%S"))
                filename = f"crypto_model_{version}.pkl"
            
            filepath = os.path.join("data/models", filename)
            
            # Create model data to save
            model_data = {
                'model': self.model,
                'feature_columns': self.feature_columns,
                'model_info': self.model_info,
                'is_trained': self.is_trained
            }
            
            joblib.dump(model_data, filepath)
            
            logger.info(f"Model saved to: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise
    
    def load_model(self, filepath: str) -> bool:
        """
        Load a trained model
        """
        try:
            if not os.path.exists(filepath):
                logger.error(f"Model file not found: {filepath}")
                return False
            
            model_data = joblib.load(filepath)
            
            self.model = model_data['model']
            self.feature_columns = model_data['feature_columns']
            self.model_info = model_data['model_info']
            self.is_trained = model_data['is_trained']
            
            logger.info(f"Model loaded successfully: {self.model_info.get('version', 'unknown')}")
            logger.info(f"Features: {len(self.feature_columns)}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False
    
    def load_latest_model(self) -> bool:
        """
        Load the latest trained model
        """
        try:
            models_dir = "data/models"
            
            if not os.path.exists(models_dir):
                logger.error("Models directory not found")
                return False
            
            # Find all model files
            model_files = [f for f in os.listdir(models_dir) if f.endswith('.pkl')]
            
            if not model_files:
                logger.error("No model files found")
                return False
            
            # Sort by modification time (newest first)
            model_files.sort(key=lambda x: os.path.getmtime(os.path.join(models_dir, x)), reverse=True)
            
            latest_model = model_files[0]
            filepath = os.path.join(models_dir, latest_model)
            
            return self.load_model(filepath)
            
        except Exception as e:
            logger.error(f"Error loading latest model: {str(e)}")
            return False
    
    def evaluate_model(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """
        Evaluate model performance
        """
        try:
            if not self.is_trained:
                raise ValueError("Model is not trained yet")
            
            # Prepare data
            X_prepared, _ = self.prepare_data(X_test)
            
            logger.info(f"Prepared data: {len(X_prepared)} samples, {len(self.feature_columns)} features")
            
            # Make predictions
            predictions = self.model.predict(X_prepared)
            probabilities = self.model.predict_proba(X_prepared)[:, 1]
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, predictions)
            precision = precision_score(y_test, predictions, zero_division=0)
            recall = recall_score(y_test, predictions, zero_division=0)
            f1 = f1_score(y_test, predictions, zero_division=0)
            
            # High confidence predictions
            high_conf_mask = probabilities >= config.CONFIDENCE_THRESHOLD
            high_conf_predictions = predictions[high_conf_mask]
            high_conf_actual = y_test.iloc[high_conf_mask] if len(high_conf_mask) > 0 else []
            
            high_conf_accuracy = accuracy_score(high_conf_actual, high_conf_predictions) if len(high_conf_predictions) > 0 else 0
            
            evaluation = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'high_conf_samples': len(high_conf_predictions),
                'high_conf_accuracy': high_conf_accuracy,
                'total_samples': len(X_prepared)
            }
            
            return evaluation
            
        except Exception as e:
            logger.error(f"Error evaluating model: {str(e)}")
            return {}
    
    def get_model_summary(self) -> Dict:
        """
        Get summary of model status
        """
        return {
            'is_trained': self.is_trained,
            'model_info': self.model_info,
            'feature_count': len(self.feature_columns) if self.feature_columns else 0,
            'model_type': 'XGBoost' if self.model else None
        }