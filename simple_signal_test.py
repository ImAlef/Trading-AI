#!/usr/bin/env python3
"""
Simple test for signal generation
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backend.services.fallback_data_collector import FallbackDataCollector
from backend.services.feature_engineer import FeatureEngineer
from backend.services.ml_model import MLModel
import pandas as pd
import numpy as np

def simple_signal_test():
    """Simple test to check signal generation"""
    
    print("🔍 Simple Signal Generation Test")
    print("="*50)
    
    try:
        # Step 1: Collect data
        print("\n📊 Step 1: Collecting data...")
        collector = FallbackDataCollector()
        df = collector.get_klines("BTCUSDT", "1h", limit=200)
        
        if df.empty:
            print("❌ No data collected")
            return False
        
        print(f"✅ Collected {len(df)} candles")
        
        # Step 2: Feature engineering for prediction
        print("\n🔧 Step 2: Feature engineering...")
        fe = FeatureEngineer()
        df_features = fe.prepare_features_for_prediction(df)
        
        if df_features.empty:
            print("❌ Feature engineering failed")
            return False
        
        print(f"✅ Features created: {len(df_features)} rows")
        
        # Step 3: Load model
        print("\n🧠 Step 3: Loading model...")
        model = MLModel()
        if not model.load_latest_model():
            print("❌ Could not load model")
            return False
        
        print("✅ Model loaded")
        
        # Step 4: Make prediction
        print("\n🔮 Step 4: Making prediction...")
        latest_data = df_features.tail(1)
        
        prediction, confidence = model.predict_single(latest_data)
        
        print(f"✅ Prediction made:")
        print(f"   Signal: {prediction} (0=No, 1=Yes)")
        print(f"   Confidence: {confidence:.3f}")
        
        if prediction == 1 and confidence >= 0.6:
            print("🟢 SIGNAL GENERATED!")
            
            # Calculate signal parameters
            current_price = df_features['close'].iloc[-1]
            target_price = current_price * 1.02  # 2% profit
            stop_loss = current_price * 0.99     # 1% loss
            
            print(f"   Entry: ${current_price:.2f}")
            print(f"   Target: ${target_price:.2f} (+2.0%)")
            print(f"   Stop: ${stop_loss:.2f} (-1.0%)")
        else:
            print("⚪ No signal (this is normal)")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = simple_signal_test()
    
    if success:
        print("\n✅ Simple test completed successfully!")
    else:
        print("\n❌ Simple test failed")