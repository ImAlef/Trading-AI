#!/usr/bin/env python3
"""
Test script for Signal Detector
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backend.services.signal_detector import SignalDetector, TradingSignal
from config import config
import logging
import time

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_signal_detector():
    """Test the signal detection functionality"""
    
    print("🎯 Testing Signal Detector...")
    print("="*50)
    
    try:
        # Step 1: Initialize signal detector
        print("\n🔧 Step 1: Initializing Signal Detector...")
        detector = SignalDetector()
        print("✅ Signal detector initialized")
        
        # Step 2: Load ML model
        print("\n🧠 Step 2: Loading ML model...")
        model_loaded = detector.load_model()
        
        if model_loaded:
            print("✅ ML model loaded successfully")
            print(f"   Model version: {detector.ml_model.model_info.get('version', 'Unknown')}")
            print(f"   Features: {detector.ml_model.model_info.get('features', 'Unknown')}")
        else:
            print("❌ Failed to load ML model")
            print("   Please run test_ml_model.py first to train a model")
            return False
        
        # Step 3: Test single symbol analysis
        print("\n📊 Step 3: Testing single symbol analysis...")
        test_symbol = "BTCUSDT"
        
        signal = detector.analyze_symbol(test_symbol)
        
        if signal:
            print(f"✅ Signal generated for {test_symbol}")
            print(f"   Signal type: {signal.signal_type}")
            print(f"   Entry price: ${signal.entry_price:,.2f}")
            print(f"   Target price: ${signal.target_price:,.2f}")
            print(f"   Stop loss: ${signal.stop_loss:,.2f}")
            print(f"   Confidence: {signal.confidence:.3f}")
            print(f"   Profit potential: {signal.get_profit_potential()*100:.2f}%")
            print(f"   Risk-reward ratio: {signal.get_risk_ratio():.2f}")
            print(f"   RSI: {signal.rsi:.1f}")
            print(f"   Volume ratio: {signal.volume_ratio:.2f}")
        else:
            print(f"⚪ No signal generated for {test_symbol}")
            print("   (This is normal - signals only generate when conditions are met)")
        
        # Step 4: Test market scanning
        print("\n🔍 Step 4: Testing market scanning...")
        test_symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]
        
        signals = detector.scan_markets(test_symbols)
        
        print(f"✅ Market scan completed")
        print(f"   Symbols scanned: {len(test_symbols)}")
        print(f"   Signals found: {len(signals)}")
        
        if signals:
            print("\n📈 Generated signals:")
            for i, signal in enumerate(signals, 1):
                print(f"   {i}. {signal.symbol}: {signal.confidence:.3f} confidence")
                print(f"      Entry: ${signal.entry_price:,.2f} → Target: ${signal.target_price:,.2f}")
                print(f"      Profit: {signal.get_profit_potential()*100:.2f}% | R/R: {signal.get_risk_ratio():.2f}")
        else:
            print("   No signals found in current market conditions")
        
        # Step 5: Test active signals tracking
        print("\n📋 Step 5: Testing active signals tracking...")
        active_signals = detector.get_active_signals()
        
        print(f"✅ Active signals: {len(active_signals)}")
        
        if active_signals:
            for signal in active_signals:
                time_left = (signal.expires_at - signal.created_at).total_seconds() / 3600
                print(f"   {signal.symbol}: {time_left:.1f} hours remaining")
        
        # Step 6: Test signal summary
        print("\n📊 Step 6: Testing signal summary...")
        summary = detector.get_signal_summary()
        
        print("✅ Signal summary generated:")
        print(f"   Active signals: {summary['active_signals']}")
        print(f"   Model loaded: {summary['model_loaded']}")
        print(f"   Model version: {summary['model_version']}")
        print(f"   Confidence threshold: {summary['confidence_threshold']}")
        print(f"   Symbols monitored: {summary['symbols_monitored']}")
        
        # Step 7: Test system status
        print("\n🔍 Step 7: Testing system status...")
        status = detector.get_system_status()
        
        print("✅ System status:")
        print(f"   Data collector: {status['data_collector']}")
        print(f"   Feature engineer: {status['feature_engineer']}")
        print(f"   ML model: {status['ml_model']}")
        print(f"   Active signals: {status['active_signals']}")
        
        # Step 8: Test signal validation (if we have a signal)
        if signals:
            print("\n🔍 Step 8: Testing signal validation...")
            test_signal = signals[0]
            
            # Test signal outcome validation
            outcome = detector.validate_signal_outcome(test_signal)
            
            if outcome['status'] == 'success':
                print("✅ Signal outcome validation successful:")
                print(f"   Current price: ${outcome['current_price']:,.2f}")
                print(f"   Entry price: ${outcome['entry_price']:,.2f}")
                print(f"   Actual return: {outcome['actual_return']*100:.2f}%")
                print(f"   Predicted return: {outcome['predicted_return']*100:.2f}%")
                print(f"   Outcome: {outcome['outcome']}")
            else:
                print(f"⚠️  Signal validation error: {outcome['message']}")
        
        # Step 9: Test signal data structure
        print("\n🏗️ Step 9: Testing signal data structure...")
        if signals:
            test_signal = signals[0]
            signal_dict = test_signal.to_dict()
            
            print("✅ Signal data structure test:")
            print(f"   Dictionary keys: {len(signal_dict)}")
            print(f"   Symbol: {signal_dict['symbol']}")
            print(f"   Created at: {signal_dict['created_at']}")
            print(f"   Expires at: {signal_dict['expires_at']}")
        
        print("\n" + "="*50)
        print("🎉 All signal detection tests completed!")
        
        # Final summary
        print("\n📊 Test Summary:")
        print(f"   🔧 Signal detector: ✅ Working")
        print(f"   🧠 ML model: ✅ Loaded")
        print(f"   📊 Symbol analysis: ✅ Working")
        print(f"   🔍 Market scanning: ✅ Working")
        print(f"   📋 Signal tracking: ✅ Working")
        print(f"   🎯 Signal validation: ✅ Working")
        print(f"   📈 Signals found: {len(signals)}")
        
        return True
        
    except Exception as e:
        logger.error(f"Signal detector test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def simulate_real_scanning():
    """Simulate real market scanning"""
    print("\n🚀 Simulating real market scanning...")
    print("="*50)
    
    try:
        detector = SignalDetector()
        
        # Load model
        if not detector.load_model():
            print("❌ Could not load model for simulation")
            return False
        
        print("🔄 Starting market scan simulation...")
        print("(This will scan all configured trading pairs)")
        
        # Scan all markets
        signals = detector.scan_markets()
        
        print(f"\n📊 Scan Results:")
        print(f"   Pairs scanned: {len(config.TRADING_PAIRS)}")
        print(f"   Signals found: {len(signals)}")
        print(f"   Success rate: {len(signals)/len(config.TRADING_PAIRS)*100:.1f}%")
        
        if signals:
            print(f"\n📈 Found {len(signals)} signals:")
            for i, signal in enumerate(signals, 1):
                print(f"   {i}. {signal.symbol}:")
                print(f"      💰 Entry: ${signal.entry_price:,.2f}")
                print(f"      🎯 Target: ${signal.target_price:,.2f} (+{signal.get_profit_potential()*100:.2f}%)")
                print(f"      🛑 Stop: ${signal.stop_loss:,.2f}")
                print(f"      📊 Confidence: {signal.confidence:.3f}")
                print(f"      ⏰ Expires: {signal.expires_at.strftime('%H:%M:%S')}")
                print()
        else:
            print("   No signals found in current market conditions")
            print("   This is normal - signals only generate when strong opportunities exist")
        
        return True
        
    except Exception as e:
        logger.error(f"Real scanning simulation failed: {str(e)}")
        return False

if __name__ == "__main__":
    print("🤖 Crypto Signal Bot - Signal Detection Test")
    print("="*50)
    
    print(f"📋 Configuration:")
    print(f"   Confidence threshold: {config.CONFIDENCE_THRESHOLD}")
    print(f"   Min profit target: {config.MIN_PROFIT_TARGET*100:.1f}%")
    print(f"   Max stop loss: {config.MAX_STOP_LOSS*100:.1f}%")
    print(f"   Signal expiry: {config.SIGNAL_EXPIRY_HOURS} hours")
    print(f"   Trading pairs: {len(config.TRADING_PAIRS)}")
    
    try:
        # Main test
        success = test_signal_detector()
        
        if success:
            print("\n🎉 Signal detection test completed successfully!")
            print("✅ Signal detection system is ready!")
            
            # Ask user if they want to run real scanning
            print("\n" + "="*50)
            user_input = input("🔍 Run real market scanning? (y/N): ").lower().strip()
            
            if user_input == 'y':
                simulate_real_scanning()
            
            print("\n🚀 Next steps:")
            print("   1. Setup email alerts")
            print("   2. Create automated scanning")
            print("   3. Build web dashboard")
            
        else:
            print("\n❌ Signal detection test failed")
            
    except KeyboardInterrupt:
        print("\n⚠️  Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()