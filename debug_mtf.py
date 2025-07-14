#!/usr/bin/env python3
"""
Debug و Fix کامل Multi-Timeframe Detector
"""

# 🔧 اول یه فایل debug تست بسازید: debug_mtf.py

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import logging
from backend.services.multi_timeframe_detector import MultiTimeframeDetector
from backend.services.ml_model import MLModel

# Setup logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def debug_multi_timeframe():
    """Debug کامل Multi-Timeframe"""
    print("🔍 Debugging Multi-Timeframe Detector...")
    
    try:
        # 1. بررسی مدل
        print("\n1️⃣ Testing ML Model...")
        ml_model = MLModel()
        
        # بررسی وجود مدل
        import os
        models_dir = "data/models"
        if os.path.exists(models_dir):
            model_files = [f for f in os.listdir(models_dir) if f.endswith('.pkl')]
            print(f"   📁 Found {len(model_files)} model files: {model_files}")
            
            if model_files:
                # تست بارگذاری
                model_loaded = ml_model.load_latest_model()
                print(f"   🧠 Model loaded: {model_loaded}")
                print(f"   🎯 Model trained: {ml_model.is_trained}")
                print(f"   📊 Model info: {ml_model.model_info}")
            else:
                print("   ❌ No model files found!")
                return False
        else:
            print("   ❌ Models directory not found!")
            return False
        
        # 2. بررسی Multi-Timeframe Detector
        print("\n2️⃣ Testing Multi-Timeframe Detector...")
        mtf_detector = MultiTimeframeDetector()
        
        print(f"   📊 Timeframes configured: {list(mtf_detector.timeframes.keys())}")
        print(f"   🔧 Models loaded initially: {mtf_detector.models_loaded}")
        
        # 3. تست بارگذاری مدل‌ها
        print("\n3️⃣ Testing model loading...")
        load_success = mtf_detector.load_models()
        print(f"   ✅ Load models result: {load_success}")
        print(f"   📊 Models dict: {list(mtf_detector.models.keys())}")
        
        if not load_success:
            print("   ❌ Model loading failed!")
            return False
        
        # 4. تست تحلیل single timeframe
        print("\n4️⃣ Testing single timeframe analysis...")
        test_symbol = "BTCUSDT"
        
        for tf, tf_config in mtf_detector.timeframes.items():
            try:
                print(f"   🔍 Testing {tf} for {test_symbol}...")
                result = mtf_detector.analyze_single_timeframe(test_symbol, tf, tf_config)
                
                if result:
                    print(f"      ✅ {tf}: prediction={result['prediction']}, confidence={result['confidence']:.3f}")
                else:
                    print(f"      ❌ {tf}: No result")
                    
            except Exception as e:
                print(f"      ⚠️ {tf}: Error - {str(e)}")
        
        # 5. تست تحلیل multi-timeframe کامل
        print("\n5️⃣ Testing full multi-timeframe analysis...")
        try:
            mtf_result = mtf_detector.analyze_symbol_multi_timeframe(test_symbol)
            
            if mtf_result:
                print(f"   ✅ Multi-timeframe result:")
                print(f"      Confidence: {mtf_result['confidence']:.3f}")
                print(f"      Timeframes agreed: {mtf_result['timeframes_agreed']}/{mtf_result['total_timeframes']}")
                print(f"      Trend alignment: {mtf_result['trend_alignment']:.2f}")
                return True
            else:
                print(f"   ❌ No multi-timeframe result")
                return False
                
        except Exception as e:
            print(f"   ❌ Multi-timeframe error: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    except Exception as e:
        print(f"❌ Debug failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    debug_multi_timeframe()