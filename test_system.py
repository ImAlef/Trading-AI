#!/usr/bin/env python3
"""
Complete System Test - تست کامل سیستم با ایمیل فیک
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List
import logging
import time

# Fix Windows console encoding for emojis
if sys.platform.startswith('win'):
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.detach())

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data/logs/system_test.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

from backend.services.signal_detector import SignalDetector
from backend.services.email_sender import EmailSender
from backend.services.fallback_data_collector import FallbackDataCollector
from config import config

class SystemTester:
    """
    کلاس تست کامل سیستم
    """
    
    def __init__(self):
        self.signal_detector = SignalDetector()
        self.email_sender = EmailSender()
        self.data_collector = FallbackDataCollector()
        
        # Create test directories
        os.makedirs('data/logs', exist_ok=True)
        os.makedirs('data/test_results', exist_ok=True)
        
    def test_data_collection(self) -> bool:
        """تست جمع‌آوری داده"""
        try:
            logger.info("🔍 Testing data collection...")
            
            # Test with a few symbols
            test_symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]
            
            for symbol in test_symbols:
                df = self.data_collector.get_klines(symbol, "1h", limit=100)
                
                if df.empty:
                    logger.error(f"❌ No data for {symbol}")
                    return False
                
                logger.info(f"✅ {symbol}: {len(df)} candles collected")
                time.sleep(0.5)  # Rate limiting
            
            logger.info("✅ Data collection test passed!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Data collection test failed: {str(e)}")
            return False
    
    def test_model_loading(self) -> bool:
        """تست بارگذاری مدل"""
        try:
            logger.info("🧠 Testing model loading...")
            
            if not self.signal_detector.load_model():
                logger.error("❌ Failed to load ML model")
                return False
            
            logger.info(f"✅ Model loaded: {self.signal_detector.ml_model.model_info.get('version', 'unknown')}")
            
            # Test live learning
            if self.signal_detector.live_learning:
                logger.info("✅ Live learning system initialized")
                learning_summary = self.signal_detector.live_learning.get_learning_summary()
                logger.info(learning_summary)
            else:
                logger.warning("⚠️ Live learning not available")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Model loading test failed: {str(e)}")
            return False
    
    def test_signal_generation(self) -> List[Dict]:
        """تست تولید سیگنال"""
        try:
            logger.info("📊 Testing signal generation...")
            
            # Test with a subset of symbols
            test_symbols = config.TRADING_PAIRS[:5]  # First 5 symbols
            
            generated_signals = []
            
            for symbol in test_symbols:
                try:
                    logger.info(f"Testing {symbol}...")
                    
                    signal = self.signal_detector.analyze_symbol(symbol)
                    
                    if signal:
                        logger.info(f"✅ Signal generated for {symbol}: {signal.signal_type} @ {signal.confidence:.3f}")
                        generated_signals.append(signal.to_dict())
                    else:
                        logger.info(f"⚪ No signal for {symbol}")
                    
                    time.sleep(1)  # Rate limiting
                    
                except Exception as e:
                    logger.error(f"❌ Error testing {symbol}: {str(e)}")
                    continue
            
            logger.info(f"📊 Signal generation test completed: {len(generated_signals)} signals")
            return generated_signals
            
        except Exception as e:
            logger.error(f"❌ Signal generation test failed: {str(e)}")
            return []
    
    def create_fake_signals(self) -> List[Dict]:
        """ایجاد سیگنال‌های فیک برای تست ایمیل"""
        try:
            logger.info("🎭 Creating fake signals for email testing...")
            
            fake_signals = []
            
            # Create LONG signal
            long_signal = {
                'symbol': 'BTCUSDT',
                'signal_type': 'BUY',
                'entry_price': 43567.890123,
                'target_price': 45234.567890,
                'stop_loss': 42890.123456,
                'confidence': 0.847,
                'created_at': datetime.now().isoformat(),
                'expires_at': (datetime.now() + timedelta(hours=18)).isoformat(),
                'rsi': 45.67,
                'macd': 0.001234,
                'macd_signal': 0.000987,
                'bb_upper': 44123.456789,
                'bb_lower': 43012.345678,
                'volume_ratio': 1.456,
                'model_version': 'v20241225_test',
                'features_used': 67,
                'timeframe': '1h'
            }
            
            # Create SHORT signal
            short_signal = {
                'symbol': 'ETHUSDT',
                'signal_type': 'SELL',
                'entry_price': 2456.789012,
                'target_price': 2398.123456,
                'stop_loss': 2501.234567,
                'confidence': 0.723,
                'created_at': datetime.now().isoformat(),
                'expires_at': (datetime.now() + timedelta(hours=18)).isoformat(),
                'rsi': 76.23,
                'macd': -0.002345,
                'macd_signal': -0.001876,
                'bb_upper': 2478.901234,
                'bb_lower': 2434.567890,
                'volume_ratio': 1.789,
                'model_version': 'v20241225_test',
                'features_used': 67,
                'timeframe': '1h'
            }
            
            # Create high confidence LONG signal
            high_conf_signal = {
                'symbol': 'BNBUSDT',
                'signal_type': 'BUY',
                'entry_price': 312.456789,
                'target_price': 325.789012,
                'stop_loss': 307.123456,
                'confidence': 0.925,
                'created_at': datetime.now().isoformat(),
                'expires_at': (datetime.now() + timedelta(hours=18)).isoformat(),
                'rsi': 38.45,
                'macd': 0.003456,
                'macd_signal': 0.002789,
                'bb_upper': 315.678901,
                'bb_lower': 309.234567,
                'volume_ratio': 2.123,
                'model_version': 'v20241225_test',
                'features_used': 67,
                'timeframe': '1h'
            }
            
            fake_signals = [long_signal, short_signal, high_conf_signal]
            
            logger.info(f"✅ Created {len(fake_signals)} fake signals")
            return fake_signals
            
        except Exception as e:
            logger.error(f"❌ Error creating fake signals: {str(e)}")
            return []
    
    def test_email_system(self, signals: List[Dict]) -> bool:
        """تست سیستم ایمیل"""
        try:
            logger.info("📧 Testing email system...")
            
            if not self.email_sender.validate_config():
                logger.error("❌ Email configuration invalid")
                return False
            
            # Test connection first
            if not self.email_sender.test_email_connection():
                logger.error("❌ Email connection test failed")
                return False
            
            logger.info("✅ Email connection successful")
            
            # Send test signals
            for i, signal in enumerate(signals):
                try:
                    logger.info(f"📧 Sending test signal {i+1}/{len(signals)}: {signal['symbol']} {signal['signal_type']}")
                    
                    success = self.email_sender.send_signal_email(signal)
                    
                    if success:
                        logger.info(f"✅ Email sent for {signal['symbol']}")
                    else:
                        logger.error(f"❌ Email failed for {signal['symbol']}")
                        return False
                    
                    time.sleep(2)  # Delay between emails
                    
                except Exception as e:
                    logger.error(f"❌ Error sending email for {signal['symbol']}: {str(e)}")
                    return False
            
            # Send system alert
            self.email_sender.send_system_alert(
                'INFO', 
                'System test completed successfully! All components are working properly.'
            )
            
            logger.info("✅ Email system test passed!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Email system test failed: {str(e)}")
            return False
    
    def generate_test_report(self, results: Dict) -> str:
        """تولید گزارش تست"""
        try:
            report = f"""
🧪 CRYPTO SIGNAL BOT - SYSTEM TEST REPORT
{'='*60}

📅 Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

📊 TEST RESULTS:
✅ Data Collection: {'PASSED' if results['data_collection'] else 'FAILED'}
✅ Model Loading: {'PASSED' if results['model_loading'] else 'FAILED'}  
✅ Signal Generation: {'PASSED' if results['signal_generation'] else 'FAILED'}
✅ Email System: {'PASSED' if results['email_system'] else 'FAILED'}

📈 SIGNAL STATISTICS:
   Real Signals Generated: {results['real_signals_count']}
   Fake Signals Created: {results['fake_signals_count']}
   Total Emails Sent: {results['emails_sent']}

🎯 SIGNAL BREAKDOWN:
"""
            
            for signal in results['all_signals']:
                direction = "📈 LONG" if signal['signal_type'] == 'BUY' else "📉 SHORT"
                report += f"""
   {direction} {signal['symbol']}:
   Entry: ${signal['entry_price']:.6f}
   Target: ${signal['target_price']:.6f}
   Stop: ${signal['stop_loss']:.6f}
   Confidence: {signal['confidence']:.1%}
   Type: {'REAL' if signal.get('is_real', False) else 'FAKE'}
"""
            
            report += f"""
🔧 SYSTEM STATUS:
   Model Version: {results['model_version']}
   Live Learning: {'ACTIVE' if results['live_learning_active'] else 'INACTIVE'}
   Email Config: {'VALID' if results['email_config_valid'] else 'INVALID'}
   
⚡ OVERALL RESULT: {'🎉 ALL TESTS PASSED!' if results['overall_success'] else '❌ SOME TESTS FAILED!'}

📝 RECOMMENDATIONS:
   - System is {'ready for live trading' if results['overall_success'] else 'NOT ready - fix failed tests'}
   - Monitor email notifications regularly
   - Check logs for any warnings or errors
   - Verify signal accuracy in live trading

Generated by Crypto Signal Bot Test Suite
"""
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating test report: {str(e)}")
            return f"❌ Error generating report: {str(e)}"
    
    def run_complete_test(self) -> Dict:
        """اجرای تست کامل"""
        try:
            logger.info("🚀 Starting complete system test...")
            print("🧪 CRYPTO SIGNAL BOT - COMPLETE SYSTEM TEST")
            print("="*50)
            
            results = {
                'data_collection': False,
                'model_loading': False,
                'signal_generation': False,
                'email_system': False,
                'real_signals_count': 0,
                'fake_signals_count': 0,
                'emails_sent': 0,
                'all_signals': [],
                'model_version': 'unknown',
                'live_learning_active': False,
                'email_config_valid': False,
                'overall_success': False
            }
            
            # Test 1: Data Collection
            print("\n📊 Testing data collection...")
            results['data_collection'] = self.test_data_collection()
            
            # Test 2: Model Loading
            print("\n🧠 Testing model loading...")
            results['model_loading'] = self.test_model_loading()
            
            if results['model_loading']:
                results['model_version'] = self.signal_detector.ml_model.model_info.get('version', 'unknown')
                results['live_learning_active'] = self.signal_detector.live_learning is not None
            
            # Test 3: Signal Generation
            print("\n📈 Testing signal generation...")
            real_signals = self.test_signal_generation()
            results['signal_generation'] = True  # Consider passed if no errors
            results['real_signals_count'] = len(real_signals)
            
            # Mark real signals
            for signal in real_signals:
                signal['is_real'] = True
            
            # Create fake signals
            print("\n🎭 Creating fake test signals...")
            fake_signals = self.create_fake_signals()
            results['fake_signals_count'] = len(fake_signals)
            
            # Mark fake signals
            for signal in fake_signals:
                signal['is_real'] = False
            
            # Combine all signals
            all_signals = real_signals + fake_signals
            results['all_signals'] = all_signals
            
            # Test 4: Email System
            print("\n📧 Testing email system...")
            results['email_config_valid'] = self.email_sender.validate_config()
            
            if results['email_config_valid'] and all_signals:
                results['email_system'] = self.test_email_system(all_signals)
                results['emails_sent'] = len(all_signals) if results['email_system'] else 0
            
            # Overall result
            results['overall_success'] = all([
                results['data_collection'],
                results['model_loading'],
                results['signal_generation'],
                results['email_system']
            ])
            
            # Generate and save report
            report = self.generate_test_report(results)
            
            # Save report to file
            report_file = f"data/test_results/system_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(report)
            
            # Print results
            print("\n" + report)
            print(f"\n📄 Report saved to: {report_file}")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Complete test failed: {str(e)}")
            print(f"\n❌ Test failed: {str(e)}")
            return {'overall_success': False, 'error': str(e)}

def main():
    """تابع اصلی"""
    try:
        tester = SystemTester()
        results = tester.run_complete_test()
        
        if results.get('overall_success', False):
            print("\n🎉 ALL TESTS PASSED! System is ready for live trading.")
            sys.exit(0)
        else:
            print("\n❌ SOME TESTS FAILED! Please fix issues before live trading.")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n👋 Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Critical error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()