#!/usr/bin/env python3
"""
Test script for ML Model
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backend.services.fallback_data_collector import FallbackDataCollector
from backend.services.feature_engineer import FeatureEngineer
from backend.services.ml_model import MLModel
from config import config
import logging
import pandas as pd

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def collect_training_data(symbols: list = None, limit: int = 6000) -> pd.DataFrame:
    """Collect training data from multiple symbols"""
    if symbols is None:
        symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT", "SOLUSDT" , "XRPUSDT"]  # Start with 2 symbols for testing
    
    collector = FallbackDataCollector()
    all_data = []
    
    for symbol in symbols:
        print(f"   Collecting data for {symbol}...")
        df = collector.get_klines(symbol, "1h", limit=limit)
        if not df.empty:
            all_data.append(df)
    
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=False)
        combined_df.reset_index(drop=True, inplace=True)
        return combined_df
    else:
        return pd.DataFrame()

def test_ml_model():
    """Test the ML model functionality"""
    
    print("🤖 Testing ML Model...")
    print("="*50)
    
    try:
        # Step 1: Collect training data
        print("\n📊 Step 1: Collecting training data...")
        training_data = collect_training_data()
        
        if training_data.empty:
            print("❌ Failed to collect training data")
            return False
        
        print(f"✅ Collected {len(training_data)} total candles from 2 symbols")
        
        # Step 2: Feature engineering
        print("\n🔧 Step 2: Feature engineering...")
        fe = FeatureEngineer()
        features_df = fe.prepare_features(training_data)
        
        if features_df.empty:
            print("❌ Feature engineering failed")
            return False
        
        feature_cols = fe.get_feature_columns()
        available_features = [col for col in feature_cols if col in features_df.columns]
        
        print(f"✅ Features prepared: {len(features_df)} samples, {len(available_features)} features")
        
        if 'target' in features_df.columns:
            target_sum = features_df['target'].sum()
            print(f"   Target distribution: {target_sum}/{len(features_df)} ({target_sum/len(features_df)*100:.1f}% positive)")
        
        # Step 3: Initialize ML model
        print("\n🧠 Step 3: Initializing ML model...")
        model = MLModel()
        print("✅ ML model initialized")
        
        # Step 4: Train model
        print("\n🚀 Step 4: Training model...")
        training_info = model.train(features_df)
        
        if training_info:
            print("✅ Model training completed")
            print(f"   Accuracy: {training_info['accuracy']:.3f}")
            print(f"   Precision: {training_info['precision']:.3f}")
            print(f"   Recall: {training_info['recall']:.3f}")
            print(f"   F1 Score: {training_info['f1_score']:.3f}")
            print(f"   CV Score: {training_info['cv_mean']:.3f} ± {training_info['cv_std']:.3f}")
        else:
            print("❌ Model training failed")
            return False
        
        # Step 5: Test predictions
        print("\n🔮 Step 5: Testing predictions...")
        test_data = features_df.tail(100)  # Use last 100 samples
        
        predictions, probabilities = model.predict(test_data)
        
        print(f"✅ Predictions generated for {len(predictions)} samples")
        
        # Analyze predictions
        high_conf_mask = probabilities >= config.CONFIDENCE_THRESHOLD
        high_conf_count = high_conf_mask.sum()
        
        print(f"   High confidence predictions: {high_conf_count}/{len(predictions)}")
        print(f"   Average confidence: {probabilities.mean():.3f}")
        
        if high_conf_count > 0:
            high_conf_probs = probabilities[high_conf_mask]
            print(f"   Sample 1: {high_conf_probs[0]:.3f} confidence")
            if len(high_conf_probs) > 1:
                print(f"   Sample 2: {high_conf_probs[1]:.3f} confidence")
        
        # Step 6: Test single prediction
        print("\n🎯 Step 6: Testing single prediction...")
        single_sample = features_df.iloc[-1:].copy()
        
        prediction, confidence = model.predict_single(single_sample)
        
        print("✅ Single prediction completed")
        print(f"   Prediction: {prediction} (0=No Signal, 1=Signal)")
        print(f"   Confidence: {confidence:.3f}")
        
        if confidence >= config.CONFIDENCE_THRESHOLD:
            print("   🟢 Above confidence threshold")
        else:
            print("   ⚪ Below confidence threshold")
        
        # Step 7: Feature importance
        print("\n📊 Step 7: Feature importance analysis...")
        importance = model.get_feature_importance()
        
        if importance:
            print("✅ Top 10 most important features:")
            for i, (feature, score) in enumerate(list(importance.items())[:10], 1):
                print(f"   {i:2d}. {feature}: {score:.4f}")
        else:
            print("❌ Could not get feature importance")
            return False
        
        # Step 8: Save and load model
        print("\n💾 Step 8: Testing model save/load...")
        
        # Save model
        model_path = model.save_model()
        print(f"✅ Model saved to: {model_path}")
        
        # Test loading
        new_model = MLModel()
        load_success = new_model.load_model(model_path)
        
        if load_success:
            print("✅ Model loaded successfully")
            
            # Test if loaded model works
            test_pred, test_conf = new_model.predict_single(single_sample)
            
            if test_pred == prediction and abs(test_conf - confidence) < 0.001:
                print("✅ Loaded model produces same predictions")
            else:
                print("❌ Loaded model produces different predictions")
                return False
        else:
            print("❌ Failed to load model")
            return False
        
        # Step 9: Model evaluation
        print("\n📈 Step 9: Model evaluation...")
        test_data = features_df.tail(100)
        test_target = test_data['target']
        
        evaluation = model.evaluate_model(test_data, test_target)
        
        if evaluation:
            print("✅ Model evaluation completed:")
            print(f"   Test Accuracy: {evaluation['accuracy']:.3f}")
            print(f"   Test Precision: {evaluation['precision']:.3f}")
            print(f"   High Conf Samples: {evaluation['high_conf_samples']}")
            print(f"   High Conf Accuracy: {evaluation['high_conf_accuracy']:.3f}")
        else:
            print("❌ Model evaluation failed")
            return False
        
        # Step 10: Model summary
        print("\n📋 Step 10: Model information...")
        summary = model.get_model_summary()
        
        print("✅ Model information:")
        print(f"   Status: {'trained' if summary['is_trained'] else 'not trained'}")
        print(f"   Version: {summary['model_info'].get('version', 'Unknown')}")
        print(f"   Features: {summary['feature_count']}")
        print(f"   Model Type: {summary['model_type']}")
        
        print("\n" + "="*50)
        print("🎉 All ML model tests passed!")
        print("✅ ML model system is ready!")
        
        # Summary
        print("\n📊 Summary:")
        print(f"   🧠 Model trained: {summary['is_trained']}")
        print(f"   📊 Training samples: {training_info['training_samples']}")
        print(f"   🎯 Model accuracy: {training_info['accuracy']:.3f}")
        print(f"   📈 Features used: {summary['feature_count']}")
        print(f"   🚨 Confidence threshold: {config.CONFIDENCE_THRESHOLD}")
        print(f"   💾 Model saved: {model_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"ML model test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🤖 Crypto Signal Bot - ML Model Test")
    print("="*50)
    
    print(f"📋 Configuration:")
    print(f"   Confidence threshold: {config.CONFIDENCE_THRESHOLD}")
    print(f"   Target return: {config.MIN_PROFIT_TARGET*100:.1f}%")
    print(f"   Signal expiry: {config.SIGNAL_EXPIRY_HOURS} hours")
    
    try:
        success = test_ml_model()
        
        if success:
            print("\n🎉 ML model test completed successfully!")
            print("✅ Ready for signal generation!")
            print("\n🚀 Next steps:")
            print("   1. Test signal detection")
            print("   2. Setup email alerts")
            print("   3. Create web dashboard")
        else:
            print("\n❌ ML model test failed")
            
    except KeyboardInterrupt:
        print("\n⚠️  Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()