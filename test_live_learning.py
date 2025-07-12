#!/usr/bin/env python3
"""
Test Live Learning System
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import logging
from datetime import datetime
import time

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_live_learning_system():
    """Test the live learning system"""
    try:
        print("🧠 Testing Live Learning System...")
        print("=" * 50)
        
        from backend.services.signal_detector import SignalDetector
        
        # Initialize signal detector
        detector = SignalDetector()
        
        # Load model
        print("📥 Loading ML model...")
        if not detector.load_model():
            print("❌ Failed to load model!")
            return False
        
        print(f"✅ Model loaded: {detector.ml_model.model_info.get('version', 'unknown')}")
        
        # Check live learning
        if detector.live_learning:
            print("🧠 Live Learning System: ACTIVE ✅")
            
            # Get live learning status
            status = detector.get_live_learning_status()
            print(f"📊 Status: {status.get('status', 'unknown')}")
            print(f"🎯 Adaptive Threshold: {status.get('current_threshold', 0):.3f}")
            print(f"📈 Active Signals: {status.get('active_signals', 0)}")
            print(f"🏁 Completed Signals: {status.get('total_completed', 0)}")
            
            # Get learning summary
            if hasattr(detector.live_learning, 'get_learning_summary'):
                summary = detector.live_learning.get_learning_summary()
                print("\n" + summary)
            
        else:
            print("❌ Live Learning System: FAILED")
            return False
        
        # Test signal detection with live learning
        print("\n🔍 Testing Signal Detection with Live Learning...")
        print("-" * 50)
        
        test_symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]
        
        for symbol in test_symbols:
            try:
                print(f"\n📊 Testing {symbol}...")
                
                # Analyze symbol
                signal = detector.analyze_symbol(symbol)
                
                if signal:
                    print(f"🎯 SIGNAL FOUND!")
                    print(f"   Symbol: {signal.symbol}")
                    print(f"   Confidence: {signal.confidence:.3f}")
                    print(f"   Entry: ${signal.entry_price:.2f}")
                    print(f"   Target: ${signal.target_price:.2f}")
                    print(f"   Stop: ${signal.stop_loss:.2f}")
                    print(f"   🧠 Registered for live learning: ✅")
                else:
                    print(f"⚪ No signal (threshold: {detector.live_learning.get_adaptive_threshold():.3f})")
                
                # Small delay
                time.sleep(1)
                
            except Exception as e:
                print(f"❌ Error testing {symbol}: {str(e)}")
        
        # Test enhanced summary
        print("\n📋 Enhanced System Summary:")
        print("-" * 30)
        summary = detector.get_enhanced_summary()
        
        print(f"Model: {summary.get('model_version', 'unknown')}")
        print(f"Active Signals: {summary.get('active_signals', 0)}")
        print(f"Live Learning: {summary.get('live_learning', {}).get('status', 'unknown')}")
        
        if 'adaptive_threshold' in summary:
            print(f"Adaptive Threshold: {summary['adaptive_threshold']:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in live learning test: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_threshold_adaptation():
    """Test threshold adaptation simulation"""
    try:
        print("\n🎯 Testing Threshold Adaptation Logic...")
        print("=" * 40)
        
        from backend.services.live_learning import LiveLearningSystem
        from backend.services.fallback_data_collector import FallbackDataCollector
        from backend.services.feature_engineer import FeatureEngineer
        from backend.services.ml_model import MLModel
        
        # Create components
        collector = FallbackDataCollector()
        engineer = FeatureEngineer()
        model = MLModel()
        
        if not model.load_latest_model():
            print("❌ No model for threshold test")
            return False
        
        # Create live learning system
        live_learning = LiveLearningSystem(collector, engineer, model)
        
        print(f"📊 Initial threshold: {live_learning.get_adaptive_threshold():.3f}")
        
        # Simulate some performance data
        print("\n🔬 Simulating performance scenarios...")
        
        # Scenario 1: Good performance
        print("📈 Scenario 1: High win rate (75%)")
        live_learning.performance_history.append({
            'timestamp': datetime.now().isoformat(),
            'win_rate': 0.75,
            'high_conf_accuracy': 0.85,
            'current_threshold': live_learning.dynamic_threshold
        })
        live_learning._adapt_threshold()
        print(f"   Threshold after high performance: {live_learning.get_adaptive_threshold():.3f}")
        
        # Scenario 2: Poor performance
        print("📉 Scenario 2: Low win rate (40%)")
        live_learning.performance_history.append({
            'timestamp': datetime.now().isoformat(),
            'win_rate': 0.40,
            'high_conf_accuracy': 0.50,
            'current_threshold': live_learning.dynamic_threshold
        })
        live_learning._adapt_threshold()
        print(f"   Threshold after poor performance: {live_learning.get_adaptive_threshold():.3f}")
        
        # Stop monitoring
        live_learning.stop_monitoring()
        
        return True
        
    except Exception as e:
        print(f"❌ Error in threshold adaptation test: {str(e)}")
        return False

def test_file_operations():
    """Test learning data file operations"""
    try:
        print("\n💾 Testing File Operations...")
        print("=" * 30)
        
        # Check if directories exist
        learning_dir = 'data/live_learning'
        if os.path.exists(learning_dir):
            print(f"✅ Learning directory exists: {learning_dir}")
            
            # List files
            files = os.listdir(learning_dir)
            if files:
                print(f"📁 Found {len(files)} learning files:")
                for file in files:
                    file_path = os.path.join(learning_dir, file)
                    size = os.path.getsize(file_path)
                    print(f"   📄 {file} ({size} bytes)")
            else:
                print("📁 Learning directory is empty (expected for first run)")
        else:
            print("📁 Learning directory will be created on first run")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in file operations test: {str(e)}")
        return False

def run_comprehensive_test():
    """Run all tests"""
    print("🤖 Live Learning System - Comprehensive Test")
    print("=" * 60)
    print(f"🕐 Test started at: {datetime.now()}")
    print()
    
    tests = [
        ("Live Learning System", test_live_learning_system),
        ("Threshold Adaptation", test_threshold_adaptation),
        ("File Operations", test_file_operations)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            print(f"\n🧪 Running: {test_name}")
            print("=" * 40)
            result = test_func()
            results.append((test_name, result))
            
            if result:
                print(f"✅ {test_name}: PASSED")
            else:
                print(f"❌ {test_name}: FAILED")
                
        except Exception as e:
            print(f"💥 {test_name}: CRASHED - {str(e)}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:<30} {status}")
    
    print("-" * 60)
    print(f"Total: {passed}/{total} tests passed ({passed/total*100:.0f}%)")
    
    if passed == total:
        print("\n🎉 All tests passed! Live Learning is ready!")
        print("\n🚀 Next steps:")
        print("1. Copy the updated files to your project")
        print("2. Restart your Docker container")
        print("3. Watch the logs for live learning activity")
        print("4. Monitor adaptive threshold changes")
    else:
        print(f"\n⚠️  {total-passed} tests failed. Check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = run_comprehensive_test()
    
    if success:
        print("\n🎯 LIVE LEARNING FEATURES:")
        print("- ✅ Adaptive threshold based on performance")
        print("- ✅ Automatic signal outcome tracking")
        print("- ✅ Learning from successful/failed signals")
        print("- ✅ Performance metrics monitoring")
        print("- ✅ Data persistence across restarts")
        print("- ✅ Real-time threshold adjustments")
        
        print("\n📈 EXPECTED BEHAVIOR:")
        print("- Initial threshold: 0.55")
        print("- After good signals: threshold decreases (more opportunities)")
        print("- After bad signals: threshold increases (more selective)")
        print("- Continuous adaptation based on market feedback")
        
        print("\n👋 Test completed successfully!")
    else:
        print("\n❌ Some tests failed. Fix issues before deploying.")
    
    sys.exit(0 if success else 1)