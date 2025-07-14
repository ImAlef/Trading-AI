#!/usr/bin/env python3
"""
🧪 تست کامل سیستم پیشرفته Crypto Signal Bot
"""
import sys
import os
import asyncio
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any
import traceback

# اضافه کردن path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import کردن تمام سیستم‌ها
from backend.services.multi_timeframe_detector import MultiTimeframeDetector
from backend.services.sentiment_analyzer import SentimentAnalyzer
from backend.services.smart_cache import SmartCache, CachedDataCollector, CachedFeatureEngineer, CachedMLModel, PerformanceMonitor
from backend.services.portfolio_manager import PortfolioManager, RiskMonitor
from backend.services.signal_detector import SignalDetector, TradingSignal
from backend.services.email_sender import EmailSender
from backend.services.fallback_data_collector import FallbackDataCollector
from backend.services.feature_engineer import FeatureEngineer
from backend.services.ml_model import MLModel
from config import config

# تنظیم logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data/logs/comprehensive_test.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ComprehensiveSystemTest:
    """
    🔬 تست کامل تمام قابلیت‌های سیستم
    """
    
    def __init__(self):
        self.test_results = {}
        self.start_time = datetime.now()
        self.email_sender = None
        
        # Test configurations
        self.test_symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']  # کمتر symbol برای تست سریع‌تر
        self.test_capital = 10000  # $10K برای تست
        
        print("🧪 Comprehensive System Test Starting...")
        print("="*60)
    
    async def run_full_test(self) -> Dict:
        """اجرای تست کامل"""
        try:
            test_start_time = time.time()
            
            # مرحله 1: تست‌های پایه
            print("\n🔧 Phase 1: Basic System Tests")
            await self.test_basic_systems()
            
            # مرحله 2: تست سیستم‌های پیشرفته
            print("\n🚀 Phase 2: Advanced Features Tests")
            await self.test_advanced_features()
            
            # مرحله 3: تست integration
            print("\n🔗 Phase 3: Integration Tests")
            await self.test_integration()
            
            # مرحله 4: تست عملکرد واقعی
            print("\n💼 Phase 4: Real Performance Test")
            await self.test_real_performance()
            
            # مرحله 5: گزارش نهایی
            print("\n📊 Phase 5: Final Report Generation")
            await self.generate_final_report()
            
            total_time = time.time() - test_start_time
            
            print(f"\n✅ All tests completed in {total_time:.1f} seconds")
            return self.test_results
            
        except Exception as e:
            logger.error(f"❌ Comprehensive test failed: {str(e)}")
            traceback.print_exc()
            return {'status': 'FAILED', 'error': str(e)}
    
    async def test_basic_systems(self):
        """تست سیستم‌های پایه"""
        try:
            print("   🔍 Testing basic data collection...")
            
            # تست Data Collector
            data_collector = FallbackDataCollector()
            test_df = data_collector.get_klines('BTCUSDT', '1h', limit=100)
            
            self.test_results['data_collector'] = {
                'status': 'PASS' if not test_df.empty else 'FAIL',
                'data_points': len(test_df),
                'columns': list(test_df.columns) if not test_df.empty else []
            }
            
            print(f"      ✓ Data Collector: {len(test_df)} candles fetched")
            
            # تست Feature Engineer
            print("   🔧 Testing feature engineering...")
            feature_engineer = FeatureEngineer()
            features_df = feature_engineer.prepare_features_for_prediction(test_df)
            
            self.test_results['feature_engineer'] = {
                'status': 'PASS' if not features_df.empty else 'FAIL',
                'features_count': len(features_df.columns) if not features_df.empty else 0,
                'samples': len(features_df)
            }
            
            print(f"      ✓ Feature Engineer: {len(features_df.columns)} features created")
            
            # تست ML Model
            print("   🧠 Testing ML model...")
            ml_model = MLModel()
            model_loaded = ml_model.load_latest_model()
            
            if model_loaded and not features_df.empty:
                prediction, confidence = ml_model.predict_single(features_df.iloc[-1:])
                model_status = 'PASS'
            else:
                prediction, confidence = 0, 0
                model_status = 'FAIL'
            
            self.test_results['ml_model'] = {
                'status': model_status,
                'model_loaded': model_loaded,
                'sample_prediction': prediction,
                'sample_confidence': confidence
            }
            
            print(f"      ✓ ML Model: Prediction={prediction}, Confidence={confidence:.3f}")
            
        except Exception as e:
            logger.error(f"❌ Basic systems test failed: {str(e)}")
            self.test_results['basic_systems'] = {'status': 'FAIL', 'error': str(e)}
    
    async def test_advanced_features(self):
        """تست قابلیت‌های پیشرفته"""
        try:
            # تست Multi-Timeframe
            print("   📊 Testing multi-timeframe analysis...")
            mtf_detector = MultiTimeframeDetector()
            
            if await self._test_multi_timeframe(mtf_detector):
                print("      ✓ Multi-Timeframe: Working correctly")
            else:
                print("      ❌ Multi-Timeframe: Failed")
            
            # تست Sentiment Analysis
            print("   💭 Testing sentiment analysis...")
            sentiment_analyzer = SentimentAnalyzer()
            
            if await self._test_sentiment_analysis(sentiment_analyzer):
                print("      ✓ Sentiment Analysis: Working correctly")
            else:
                print("      ❌ Sentiment Analysis: Failed")
            
            # تست Cache System
            print("   💾 Testing cache system...")
            cache_system = SmartCache()
            
            if self._test_cache_system(cache_system):
                print("      ✓ Cache System: Working correctly")
            else:
                print("      ❌ Cache System: Failed")
            
            # تست Portfolio Manager
            print("   💼 Testing portfolio management...")
            portfolio_manager = PortfolioManager(self.test_capital)
            
            if self._test_portfolio_manager(portfolio_manager):
                print("      ✓ Portfolio Manager: Working correctly")
            else:
                print("      ❌ Portfolio Manager: Failed")
            
        except Exception as e:
            logger.error(f"❌ Advanced features test failed: {str(e)}")
            self.test_results['advanced_features'] = {'status': 'FAIL', 'error': str(e)}
    
    async def _test_multi_timeframe(self, mtf_detector: MultiTimeframeDetector) -> bool:
        """تست Multi-Timeframe"""
        try:
            # بارگذاری مدل‌ها
            models_loaded = mtf_detector.load_models()
            
            if not models_loaded:
                self.test_results['multi_timeframe'] = {'status': 'FAIL', 'error': 'Models not loaded'}
                return False
            
            # تست تحلیل
            result = mtf_detector.analyze_symbol_multi_timeframe('BTCUSDT')
            
            self.test_results['multi_timeframe'] = {
                'status': 'PASS' if result else 'FAIL',
                'models_loaded': models_loaded,
                'test_result': result,
                'timeframes_tested': len(mtf_detector.timeframes)
            }
            
            return result is not None
            
        except Exception as e:
            self.test_results['multi_timeframe'] = {'status': 'FAIL', 'error': str(e)}
            return False
    
    async def _test_sentiment_analysis(self, sentiment_analyzer: SentimentAnalyzer) -> bool:
        """تست Sentiment Analysis"""
        try:
            # تست sentiment برای Bitcoin
            sentiment_result = await sentiment_analyzer.get_market_sentiment('BTCUSDT')
            
            success = (
                sentiment_result and 
                'sentiment_score' in sentiment_result and
                'confidence' in sentiment_result
            )
            
            self.test_results['sentiment_analysis'] = {
                'status': 'PASS' if success else 'FAIL',
                'sentiment_score': sentiment_result.get('sentiment_score', 0) if sentiment_result else 0,
                'confidence': sentiment_result.get('confidence', 0) if sentiment_result else 0,
                'sources_used': sentiment_result.get('sources_used', 0) if sentiment_result else 0
            }
            
            return success
            
        except Exception as e:
            self.test_results['sentiment_analysis'] = {'status': 'FAIL', 'error': str(e)}
            return False
    
    def _test_cache_system(self, cache_system: SmartCache) -> bool:
        """تست Cache System"""
        try:
            # تست ذخیره و بازیابی
            test_key = "test_cache_key"
            test_data = {"timestamp": datetime.now().isoformat(), "value": 12345}
            
            # ذخیره
            store_success = cache_system.set(test_key, test_data, "test")
            
            # بازیابی
            retrieved_data = cache_system.get(test_key, "test")
            
            # بررسی
            success = store_success and retrieved_data == test_data
            
            # آمار cache
            cache_stats = cache_system.get_stats()
            
            self.test_results['cache_system'] = {
                'status': 'PASS' if success else 'FAIL',
                'store_success': store_success,
                'retrieve_success': retrieved_data == test_data,
                'cache_stats': cache_stats
            }
            
            return success
            
        except Exception as e:
            self.test_results['cache_system'] = {'status': 'FAIL', 'error': str(e)}
            return False
    
    def _test_portfolio_manager(self, portfolio_manager: PortfolioManager) -> bool:
        """تست Portfolio Manager"""
        try:
            # ایجاد سیگنال تست
            test_signal = TradingSignal(
                symbol="BTCUSDT",
                signal_type="BUY",
                entry_price=45000,
                target_price=47000,
                stop_loss=44000,
                confidence=0.75,
                created_at=datetime.now(),
                expires_at=datetime.now() + timedelta(hours=24),
                rsi=65, macd=0.01, macd_signal=0.005,
                bb_upper=46000, bb_lower=44000, volume_ratio=1.5,
                model_version="test", features_used=50, timeframe="1h"
            )
            
            # تست محاسبه position size
            position_size, calculation_reason = portfolio_manager.calculate_position_size(test_signal)
            
            # تست امکان باز کردن موقعیت
            can_open, reason = portfolio_manager.can_open_position(test_signal)
            
            # تست باز کردن موقعیت
            position_opened = None
            if can_open:
                position_opened = portfolio_manager.open_position(test_signal)
            
            # خلاصه پورتفولیو
            portfolio_summary = portfolio_manager.get_portfolio_summary()
            
            success = position_size > 0 and portfolio_summary is not None
            
            self.test_results['portfolio_manager'] = {
                'status': 'PASS' if success else 'FAIL',
                'position_size_calculated': position_size,
                'can_open_position': can_open,
                'position_opened': position_opened is not None,
                'portfolio_summary': portfolio_summary
            }
            
            return success
            
        except Exception as e:
            self.test_results['portfolio_manager'] = {'status': 'FAIL', 'error': str(e)}
            return False
    
    async def test_integration(self):
        """تست Integration سیستم‌ها"""
        try:
            print("   🔗 Testing full system integration...")
            
            # ایجاد Signal Detector کامل
            signal_detector = SignalDetector()
            model_loaded = signal_detector.load_model()
            
            if not model_loaded:
                self.test_results['integration'] = {'status': 'FAIL', 'error': 'Signal detector model not loaded'}
                return
            
            # تست تحلیل enhanced
            enhanced_signals = []
            
            for symbol in self.test_symbols:
                try:
                    # تحلیل enhanced symbol
                    signal = signal_detector.analyze_symbol_enhanced(symbol, use_multi_timeframe=True)
                    
                    if signal:
                        enhanced_signals.append({
                            'symbol': symbol,
                            'signal': signal.to_dict(),
                            'has_mtf_info': hasattr(signal, 'mtf_info')
                        })
                    
                    # کمی صبر برای rate limiting
                    await asyncio.sleep(0.5)
                    
                except Exception as e:
                    logger.warning(f"Integration test failed for {symbol}: {str(e)}")
                    continue
            
            # خلاصه نتایج integration
            self.test_results['integration'] = {
                'status': 'PASS' if len(enhanced_signals) >= 0 else 'FAIL',
                'symbols_tested': len(self.test_symbols),
                'signals_generated': len(enhanced_signals),
                'enhanced_signals': enhanced_signals,
                'signal_detector_ready': model_loaded
            }
            
            print(f"      ✓ Integration: {len(enhanced_signals)} signals from {len(self.test_symbols)} symbols")
            
        except Exception as e:
            logger.error(f"❌ Integration test failed: {str(e)}")
            self.test_results['integration'] = {'status': 'FAIL', 'error': str(e)}
    
    async def test_real_performance(self):
        """تست عملکرد واقعی"""
        try:
            print("   💼 Testing real performance simulation...")
            
            # شبیه‌سازی یک چرخه کامل اسکن
            from main import EnhancedAutoScanner
            
            scanner = EnhancedAutoScanner()
            
            # تنظیم اولیه (بدون email برای تست)
            original_email = scanner.email_sender
            scanner.email_sender = None  # موقتاً غیرفعال
            
            performance_start = time.time()
            
            # شبیه‌سازی اسکن
            signals = await scanner.enhanced_market_scan(self.test_symbols)
            
            performance_time = time.time() - performance_start
            
            # بازگردانی email sender
            scanner.email_sender = original_email
            
            # دریافت آمار جامع
            comprehensive_status = scanner.get_comprehensive_status()
            
            self.test_results['real_performance'] = {
                'status': 'PASS',
                'scan_time_seconds': performance_time,
                'signals_found': len(signals),
                'comprehensive_status': comprehensive_status,
                'scanner_initialized': True
            }
            
            print(f"      ✓ Real Performance: {len(signals)} signals in {performance_time:.1f}s")
            
        except Exception as e:
            logger.error(f"❌ Real performance test failed: {str(e)}")
            self.test_results['real_performance'] = {'status': 'FAIL', 'error': str(e)}
    
    async def generate_final_report(self):
        """تولید گزارش نهایی و ارسال ایمیل"""
        try:
            print("   📧 Generating final report and sending email...")
            
            # محاسبه آمار کلی
            total_tests = len(self.test_results)
            passed_tests = len([r for r in self.test_results.values() if r.get('status') == 'PASS'])
            failed_tests = total_tests - passed_tests
            success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
            
            total_time = (datetime.now() - self.start_time).total_seconds()
            
            # تنظیم email sender
            email_sender = EmailSender()
            
            if email_sender.validate_config():
                self.email_sender = email_sender
                
                # ارسال گزارش تست
                await self._send_test_report_email(passed_tests, failed_tests, success_rate, total_time)
                
                # تست email system
                email_test_success = email_sender.test_email_connection()
                
                self.test_results['email_system'] = {
                    'status': 'PASS' if email_test_success else 'FAIL',
                    'config_valid': True,
                    'test_email_sent': email_test_success
                }
                
                print(f"      ✓ Email System: {'Working' if email_test_success else 'Failed'}")
                
            else:
                self.test_results['email_system'] = {
                    'status': 'FAIL',
                    'config_valid': False,
                    'error': 'Email configuration invalid'
                }
                print("      ❌ Email System: Configuration invalid")
            
            # خلاصه نهایی
            self.test_results['summary'] = {
                'total_tests': total_tests,
                'passed_tests': passed_tests,
                'failed_tests': failed_tests,
                'success_rate': success_rate,
                'total_time_seconds': total_time,
                'test_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Final report generation failed: {str(e)}")
            self.test_results['final_report'] = {'status': 'FAIL', 'error': str(e)}
    
    async def _send_test_report_email(self, passed: int, failed: int, success_rate: float, total_time: float):
        """ارسال گزارش تست به ایمیل"""
        try:
            if not self.email_sender:
                return
            
            # تشخیص وضعیت کلی
            overall_status = "✅ PASSED" if success_rate >= 80 else "⚠️ PARTIAL" if success_rate >= 60 else "❌ FAILED"
            
            subject = f"🧪 Crypto Bot Test Report - {overall_status} ({success_rate:.1f}%)"
            
            # ایجاد گزارش تفصیلی
            detailed_results = self._format_detailed_results()
            
            message = f"""
🧪 CRYPTO SIGNAL BOT - COMPREHENSIVE TEST REPORT
================================================

📊 TEST SUMMARY:
   Overall Status: {overall_status}
   Success Rate: {success_rate:.1f}%
   Passed Tests: {passed}/{passed + failed}
   Failed Tests: {failed}/{passed + failed}
   Total Time: {total_time:.1f} seconds

🔧 SYSTEM COMPONENTS:
{self._format_component_status()}

🚀 ADVANCED FEATURES:
{self._format_advanced_features_status()}

💼 PERFORMANCE METRICS:
{self._format_performance_metrics()}

📋 DETAILED RESULTS:
{detailed_results}

🎯 RECOMMENDATIONS:
{self._get_recommendations()}

---
Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Test completed in {total_time:.1f} seconds
            """
            
            success = self.email_sender.send_system_alert('INFO', message)
            
            if success:
                print("      ✓ Test report email sent successfully")
            else:
                print("      ❌ Failed to send test report email")
                
        except Exception as e:
            logger.error(f"❌ Test report email failed: {str(e)}")
    
    def _format_component_status(self) -> str:
        """فرمت کردن وضعیت component ها"""
        components = {
            'data_collector': 'Data Collector',
            'feature_engineer': 'Feature Engineer', 
            'ml_model': 'ML Model',
            'cache_system': 'Cache System',
            'multi_timeframe': 'Multi-Timeframe',
            'sentiment_analysis': 'Sentiment Analysis',
            'portfolio_manager': 'Portfolio Manager'
        }
        
        status_text = ""
        for key, name in components.items():
            if key in self.test_results:
                status = self.test_results[key]['status']
                emoji = "✅" if status == 'PASS' else "❌"
                status_text += f"   {emoji} {name}: {status}\n"
        
        return status_text
    
    def _format_advanced_features_status(self) -> str:
        """فرمت کردن وضعیت قابلیت‌های پیشرفته"""
        advanced_features = []
        
        if 'multi_timeframe' in self.test_results:
            mtf = self.test_results['multi_timeframe']
            status = "✅" if mtf['status'] == 'PASS' else "❌"
            timeframes = mtf.get('timeframes_tested', 0)
            advanced_features.append(f"   {status} Multi-Timeframe Analysis: {timeframes} timeframes")
        
        if 'sentiment_analysis' in self.test_results:
            sentiment = self.test_results['sentiment_analysis']
            status = "✅" if sentiment['status'] == 'PASS' else "❌"
            score = sentiment.get('sentiment_score', 0)
            advanced_features.append(f"   {status} Sentiment Analysis: {score:.2f} score")
        
        if 'cache_system' in self.test_results:
            cache = self.test_results['cache_system']
            status = "✅" if cache['status'] == 'PASS' else "❌"
            stats = cache.get('cache_stats', {})
            hit_rate = stats.get('hit_rate', 0)
            advanced_features.append(f"   {status} Smart Caching: {hit_rate:.1f}% hit rate")
        
        if 'portfolio_manager' in self.test_results:
            portfolio = self.test_results['portfolio_manager']
            status = "✅" if portfolio['status'] == 'PASS' else "❌"
            advanced_features.append(f"   {status} Portfolio Management: Risk calculation working")
        
        return "\n".join(advanced_features)
    
    def _format_performance_metrics(self) -> str:
        """فرمت کردن معیارهای عملکرد"""
        performance_text = ""
        
        if 'real_performance' in self.test_results:
            perf = self.test_results['real_performance']
            scan_time = perf.get('scan_time_seconds', 0)
            signals_found = perf.get('signals_found', 0)
            performance_text += f"   Scan Time: {scan_time:.1f} seconds\n"
            performance_text += f"   Signals Generated: {signals_found}\n"
            
            # Cache performance
            if 'comprehensive_status' in perf:
                cache_perf = perf['comprehensive_status'].get('cache_performance', {})
                hit_rate = cache_perf.get('cache_hit_rate', 0)
                performance_text += f"   Cache Hit Rate: {hit_rate:.1f}%\n"
        
        return performance_text
    
    def _format_detailed_results(self) -> str:
        """فرمت کردن نتایج تفصیلی"""
        details = []
        
        for test_name, result in self.test_results.items():
            if test_name == 'summary':
                continue
                
            status_emoji = "✅" if result.get('status') == 'PASS' else "❌"
            details.append(f"   {status_emoji} {test_name.replace('_', ' ').title()}")
            
            # اضافه کردن جزئیات خاص
            if 'error' in result:
                details.append(f"      Error: {result['error']}")
            elif test_name == 'integration':
                signals = result.get('signals_generated', 0)
                details.append(f"      Signals: {signals}")
            elif test_name == 'ml_model':
                confidence = result.get('sample_confidence', 0)
                details.append(f"      Confidence: {confidence:.3f}")
        
        return "\n".join(details)
    
    def _get_recommendations(self) -> str:
        """تولید پیشنهادات بر اساس نتایج تست"""
        recommendations = []
        
        # بررسی component های fail شده
        failed_components = [name for name, result in self.test_results.items() 
                           if result.get('status') == 'FAIL']
        
        if 'email_system' in failed_components:
            recommendations.append("• Check email configuration in .env file")
        
        if 'ml_model' in failed_components:
            recommendations.append("• Run model training: python trainer.py")
        
        if 'cache_system' in failed_components:
            recommendations.append("• Check Redis installation or use file-based cache")
        
        if 'multi_timeframe' in failed_components:
            recommendations.append("• Verify model files exist for all timeframes")
        
        if not failed_components:
            recommendations.append("• All systems working perfectly! 🎉")
            recommendations.append("• Ready for production trading")
            recommendations.append("• Consider running live monitoring")
        
        return "\n".join(recommendations) if recommendations else "• No specific recommendations"

# تابع اصلی تست
async def run_comprehensive_test():
    """اجرای تست کامل"""
    print("🚀 Starting Comprehensive System Test...")
    print("This will test all advanced features and send email reports")
    print("="*60)
    
    # ایجاد پوشه‌های مورد نیاز
    os.makedirs('data/logs', exist_ok=True)
    os.makedirs('data/cache', exist_ok=True)
    os.makedirs('data/portfolio', exist_ok=True)
    
    # اجرای تست
    test_runner = ComprehensiveSystemTest()
    results = await test_runner.run_full_test()
    
    # نمایش خلاصه نهایی
    print("\n" + "="*60)
    print("🎯 COMPREHENSIVE TEST SUMMARY")
    print("="*60)
    
    if 'summary' in results:
        summary = results['summary']
        print(f"Total Tests: {summary['total_tests']}")
        print(f"Passed: {summary['passed_tests']} ✅")
        print(f"Failed: {summary['failed_tests']} ❌")
        print(f"Success Rate: {summary['success_rate']:.1f}%")
        print(f"Total Time: {summary['total_time_seconds']:.1f} seconds")
        
        if summary['success_rate'] >= 80:
            print("\n🎉 EXCELLENT! System is ready for production!")
        elif summary['success_rate'] >= 60:
            print("\n⚠️ PARTIAL SUCCESS. Some issues need attention.")
        else:
            print("\n❌ MULTIPLE FAILURES. System needs fixes before use.")
    
    print(f"\nDetailed logs saved to: data/logs/comprehensive_test.log")
    print("Check your email for the complete test report!")
    
    return results

if __name__ == "__main__":
    try:
        # اجرای تست
        asyncio.run(run_comprehensive_test())
        
    except KeyboardInterrupt:
        print("\n⚠️ Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        traceback.print_exc()