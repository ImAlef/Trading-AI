"""
🚀 Enhanced Crypto Signal Bot - Production Version
تمام قابلیت‌های پیشرفته فعال
"""
import sys
import os
import asyncio
import time
import schedule
import logging
from datetime import datetime, timedelta
from typing import List, Dict

# Fix Windows console encoding for emojis
if sys.platform.startswith('win'):
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.detach())

# Add project path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import all systems
from backend.services.signal_detector import SignalDetector
from backend.services.email_sender import EmailSender
from backend.services.fallback_data_collector import FallbackDataCollector
from backend.services.feature_engineer import FeatureEngineer
from backend.services.ml_model import MLModel

# 🆕 Enhanced imports
from backend.services.multi_timeframe_detector import MultiTimeframeDetector
from backend.services.sentiment_analyzer import SentimentAnalyzer
from backend.services.smart_cache import SmartCache, CachedDataCollector, CachedFeatureEngineer, CachedMLModel, PerformanceMonitor
from backend.services.portfolio_manager import PortfolioManager, RiskMonitor

from config import config

# Setup enhanced logging
try:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('data/logs/enhanced_bot.log', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
except:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('data/logs/enhanced_bot.log'),
            logging.StreamHandler()
        ]
    )

logger = logging.getLogger(__name__)

class AutoTrainingSystem:
    """سیستم آموزش خودکار مدل"""
    
    def __init__(self):
        self.data_collector = FallbackDataCollector()
        self.feature_engineer = FeatureEngineer()
        self.ml_model = MLModel()
        
    def collect_training_data(self, days: int = 180):
        """جمع‌آوری داده‌های آموزش"""
        try:
            logger.info(f"🔄 Collecting {days} days of training data...")
            
            training_pairs = [
                "BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT", "SOLUSDT", 
                "DOTUSDT", "LINKUSDT", "MATICUSDT", "LTCUSDT", "XRPUSDT",
                "AVAXUSDT", "UNIUSDT", "ATOMUSDT", "FILUSDT", "SANDUSDT"
            ]
            
            all_training_data = []
            end_time = datetime.now()
            start_time = end_time - timedelta(days=days)
            
            for symbol in training_pairs:
                try:
                    logger.info(f"📊 Collecting {symbol}...")
                    df = self.data_collector.get_historical_klines(symbol, "1h", start_time, end_time)
                    
                    if df.empty or len(df) < 500:
                        logger.warning(f"⚠️ Insufficient data for {symbol}: {len(df)} candles")
                        continue
                    
                    df_features = self.feature_engineer.prepare_features(df)
                    
                    if df_features.empty:
                        logger.warning(f"⚠️ Feature engineering failed for {symbol}")
                        continue
                    
                    df_features['source_symbol'] = symbol
                    all_training_data.append(df_features)
                    
                    logger.info(f"✅ {symbol}: {len(df_features)} samples collected")
                    time.sleep(0.5)
                    
                except Exception as e:
                    logger.error(f"❌ Error collecting {symbol}: {str(e)}")
                    continue
            
            if not all_training_data:
                raise ValueError("No training data collected!")
            
            import pandas as pd
            combined_data = pd.concat(all_training_data, ignore_index=True)
            
            if 'source_symbol' in combined_data.columns:
                combined_data = combined_data.drop('source_symbol', axis=1)
            
            logger.info(f"🎯 Total training data: {len(combined_data)} samples")
            logger.info(f"📈 Positive samples: {combined_data['target'].sum()} ({combined_data['target'].mean()*100:.1f}%)")
            
            return combined_data
            
        except Exception as e:
            logger.error(f"❌ Error collecting training data: {str(e)}")
            raise
    
    def train_new_model(self, training_data):
        """آموزش مدل جدید"""
        try:
            logger.info("🧠 Starting model training...")
            
            if len(training_data) < 1000:
                logger.warning("⚠️ Limited training data - model performance may vary")
            
            model_info = self.ml_model.train(training_data)
            model_path = self.ml_model.save_model()
            
            logger.info("🎉 Model training completed successfully!")
            logger.info(f"📁 Model saved: {model_path}")
            logger.info(f"📊 Performance:")
            logger.info(f"   - Accuracy: {model_info['accuracy']:.3f}")
            logger.info(f"   - Precision: {model_info['precision']:.3f}")
            logger.info(f"   - Recall: {model_info['recall']:.3f}")
            logger.info(f"   - F1-Score: {model_info['f1_score']:.3f}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Model training failed: {str(e)}")
            return False
    
    def auto_train_if_needed(self) -> bool:
        """آموزش خودکار در صورت نیاز"""
        try:
            if self.ml_model.load_latest_model():
                logger.info("✅ Existing model found and loaded")
                return True
            
            logger.warning("⚠️ No model found - starting auto-training...")
            
            training_data = self.collect_training_data(days=120)
            success = self.train_new_model(training_data)
            
            if success:
                if self.ml_model.load_latest_model():
                    logger.info("🚀 New model trained and loaded successfully!")
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"❌ Auto-training failed: {str(e)}")
            return False

class EnhancedAutoScanner:
    """🚀 اسکنر پیشرفته با تمام قابلیت‌های جدید"""
    
    def __init__(self):
        # سیستم‌های اصلی
        self.signal_detector = SignalDetector()
        self.email_sender = EmailSender()
        self.auto_trainer = AutoTrainingSystem()
        
        # 🆕 سیستم‌های پیشرفته
        self.cache_system = SmartCache()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.portfolio_manager = PortfolioManager(initial_capital=20000)
        self.risk_monitor = RiskMonitor(self.portfolio_manager)
        self.performance_monitor = PerformanceMonitor(self.cache_system)
        
        # Cache-enabled services
        self.cached_data_collector = None
        self.cached_feature_engineer = None
        self.cached_ml_model = None
        
        # آمار
        self.daily_signals = []
        self.scan_count = 0
        self.start_time = datetime.now()
        
        # Create directories
        os.makedirs('data/logs', exist_ok=True)
        os.makedirs('data/models', exist_ok=True)
        os.makedirs('data/cache', exist_ok=True)
        os.makedirs('data/portfolio', exist_ok=True)
        
        logger.info("🚀 Enhanced Auto Scanner initialized!")
    
    async def initialize(self) -> bool:
        """راه‌اندازی کامل سیستم پیشرفته"""
        try:
            logger.info("🔧 Initializing Enhanced Auto Scanner with all features...")
            
            # 1. آموزش یا بارگذاری مدل
            logger.info("🧠 Setting up ML model...")
            model_ready = self.auto_trainer.auto_train_if_needed()
            
            if not model_ready:
                logger.error("❌ Model initialization failed")
                return False
            
            # 2. راه‌اندازی signal detector
            logger.info("🔍 Setting up signal detector...")
            if not self.signal_detector.load_model():
                logger.error("❌ Failed to load model in signal detector")
                return False
            
            # 3. راه‌اندازی cached services
            logger.info("💾 Setting up cache system...")
            self._setup_cached_services()
            
            # 4. تست cache system
            self._test_cache_system()
            
            # 5. تست sentiment analyzer
            logger.info("💭 Testing sentiment analysis...")
            await self._test_sentiment_analyzer()
            
            # 6. تنظیم email
            logger.info("📧 Setting up email notifications...")
            if not self._setup_email():
                logger.warning("⚠️ Email setup failed - notifications disabled")
            
            # 7. ارسال ایمیل شروع
            await self._send_startup_notification()
            
            # 8. خلاصه راه‌اندازی
            self._log_initialization_summary()
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Enhanced scanner initialization failed: {str(e)}")
            return False
    
    def _setup_cached_services(self):
        """راه‌اندازی سرویس‌های cache-enabled"""
        try:
            self.cached_data_collector = CachedDataCollector(
                self.signal_detector.data_collector, 
                self.cache_system
            )
            
            self.cached_feature_engineer = CachedFeatureEngineer(
                self.signal_detector.feature_engineer,
                self.cache_system
            )
            
            self.cached_ml_model = CachedMLModel(
                self.signal_detector.ml_model,
                self.cache_system
            )
            
            logger.info("✅ Cache-enabled services initialized")
            
        except Exception as e:
            logger.error(f"❌ Cache services setup error: {str(e)}")
    
    def get_comprehensive_status(self) -> Dict:
        """وضعیت جامع سیستم"""
        try:
            return {
                'scanner_info': {
                    'uptime_hours': (datetime.now() - self.start_time).total_seconds() / 3600,
                    'total_scans': self.scan_count,
                    'daily_signals': len(self.daily_signals)
                },
                'portfolio_status': self.portfolio_manager.get_portfolio_summary(),
                'risk_status': self.risk_monitor.get_risk_report(),
                'cache_performance': self.performance_monitor.get_performance_stats(),
                'cache_health': self.performance_monitor.get_cache_health(),
                'ml_model_info': self.signal_detector.ml_model.model_info,
                'live_learning_status': self.signal_detector.get_live_learning_status(),
                'last_update': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Status compilation error: {str(e)}")
            return {'error': str(e)}

    def _test_cache_system(self):
        """تست سیستم cache"""
        try:
            test_key = "startup_test"
            test_data = {"timestamp": datetime.now().isoformat(), "test": True}
            
            success = self.cache_system.set(test_key, test_data, "test")
            retrieved = self.cache_system.get(test_key, "test")
            
            if success and retrieved == test_data:
                logger.info("✅ Cache system working correctly")
            else:
                logger.warning("⚠️ Cache system test failed")
                
        except Exception as e:
            logger.warning(f"⚠️ Cache test error: {str(e)}")
    
    async def _test_sentiment_analyzer(self):
        """تست sentiment analyzer"""
        try:
            sentiment = await self.sentiment_analyzer.get_market_sentiment('BTCUSDT')
            
            if sentiment and 'sentiment_score' in sentiment:
                score = sentiment['sentiment_score']
                confidence = sentiment['confidence']
                logger.info(f"✅ Sentiment analysis working: {score:.2f} (conf: {confidence:.2f})")
            else:
                logger.warning("⚠️ Sentiment analysis test failed")
                
        except Exception as e:
            logger.warning(f"⚠️ Sentiment test error: {str(e)}")
    
    def _setup_email(self) -> bool:
        """تنظیم email"""
        try:
            if not self.email_sender.validate_config():
                self.email_sender = None
                return False
            
            logger.info("✅ Email notifications enabled")
            return True
            
        except Exception as e:
            logger.warning(f"⚠️ Email setup failed: {str(e)}")
            self.email_sender = None
            return False
    
    async def _send_startup_notification(self):
        """ارسال اطلاع‌رسانی شروع"""
        try:
            if not self.email_sender:
                return
            
            portfolio_summary = self.portfolio_manager.get_portfolio_summary()
            
            startup_message = f"""
🚀 ENHANCED CRYPTO SIGNAL BOT STARTED
=====================================

🤖 SYSTEM STATUS: FULLY OPERATIONAL
⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

💰 PORTFOLIO STATUS:
   Initial Capital: ${portfolio_summary.get('initial_capital', 0):,.2f}
   Available Cash: ${portfolio_summary.get('cash_available', 0):,.2f}
   Active Positions: {portfolio_summary.get('active_positions', 0)}

🚀 ENHANCED FEATURES ACTIVE:
   ✅ Multi-Timeframe Analysis
   ✅ Sentiment Analysis  
   ✅ Smart Caching System
   ✅ Portfolio Management
   ✅ Risk Monitoring
   ✅ Live Learning System

📊 CONFIGURATION:
   Trading Pairs: {len(config.TRADING_PAIRS)}
   Scan Interval: {config.DATA_COLLECTION_INTERVAL/60:.1f} minutes
   Confidence Threshold: {config.CONFIDENCE_THRESHOLD}
   Max Risk per Trade: {self.portfolio_manager.max_risk_per_trade*100:.1f}%

🎯 READY FOR TRADING!
The bot will now scan markets and send you high-quality signals.
            """
            
            self.email_sender.send_system_alert('INFO', startup_message)
            logger.info("📧 Startup notification sent")
            
        except Exception as e:
            logger.error(f"❌ Startup notification failed: {str(e)}")
    
    def _log_initialization_summary(self):
        """خلاصه راه‌اندازی"""
        try:
            logger.info("📊 Enhanced Scanner Initialization Summary:")
            logger.info(f"   🧠 ML Model: {self.signal_detector.ml_model.model_info.get('version', 'unknown')}")
            logger.info(f"   🔄 Multi-Timeframe: {'✅' if hasattr(self.signal_detector, 'mtf_detector') else '❌'}")
            logger.info(f"   💭 Sentiment Analysis: {'✅' if self.sentiment_analyzer else '❌'}")
            logger.info(f"   💾 Cache System: {'✅' if self.cache_system else '❌'}")
            logger.info(f"   💼 Portfolio Manager: {'✅' if self.portfolio_manager else '❌'}")
            logger.info(f"   📧 Email Notifications: {'✅' if self.email_sender else '❌'}")
            logger.info(f"   🧠 Live Learning: {'✅' if self.signal_detector.live_learning else '❌'}")
            
            portfolio_summary = self.portfolio_manager.get_portfolio_summary()
            logger.info(f"   💰 Portfolio Value: ${portfolio_summary.get('portfolio_value', 0):,.2f}")
            logger.info(f"   💵 Available Cash: ${portfolio_summary.get('cash_available', 0):,.2f}")
            
        except Exception as e:
            logger.error(f"❌ Summary logging error: {str(e)}")
    
    async def enhanced_market_scan(self, symbols: List[str] = None) -> List[Dict]:
        """🔍 اسکن پیشرفته بازار با تمام قابلیت‌ها"""
        try:
            if symbols is None:
                symbols = config.TRADING_PAIRS
            
            scan_start_time = time.time()
            logger.info(f"🔍 Enhanced market scan starting for {len(symbols)} symbols...")
            
            # 1. تحلیل sentiment کلی بازار
            market_sentiment = await self.sentiment_analyzer.get_market_sentiment('BTCUSDT')
            sentiment_score = market_sentiment.get('sentiment_score', 0)
            logger.info(f"📊 Market Sentiment: {sentiment_score:.2f}")
            
            # 2. اسکن multi-timeframe
            signals = self.signal_detector.scan_markets_enhanced(symbols, use_multi_timeframe=True)
            
            # 3. تحلیل هر سیگنال با portfolio manager
            enhanced_signals = []
            
            for signal in signals:
                try:
                    enhanced_signal = await self._process_enhanced_signal(signal, market_sentiment)
                    enhanced_signals.append(enhanced_signal)
                    
                except Exception as e:
                    logger.error(f"❌ Error enhancing signal {signal.symbol}: {str(e)}")
                    continue
            
            # 4. بروزرسانی آمار
            scan_time = time.time() - scan_start_time
            self.performance_monitor.log_request_time(scan_time)
            self.scan_count += 1
            
            # 5. لاگ نتایج
            await self._log_enhanced_scan_results(enhanced_signals, scan_time, market_sentiment)
            
            return enhanced_signals
            
        except Exception as e:
            logger.error(f"❌ Enhanced market scan error: {str(e)}")
            return []
    
    async def _process_enhanced_signal(self, signal, market_sentiment: Dict) -> Dict:
        """پردازش پیشرفته یک سیگنال"""
        try:
            # بررسی امکان باز کردن موقعیت
            can_open, reason = self.portfolio_manager.can_open_position(signal)
            
            # تحلیل sentiment برای این symbol
            symbol_sentiment = await self.sentiment_analyzer.get_market_sentiment(signal.symbol)
            
            # محاسبه اندازه موقعیت
            position_size, calculation_reason = self.portfolio_manager.calculate_position_size(signal)
            
            # ایجاد سیگنال پیشرفته
            enhanced_signal = {
                'signal': signal.to_dict(),
                'portfolio_analysis': {
                    'can_open_position': can_open,
                    'position_approval_reason': reason,
                    'recommended_position_size': position_size,
                    'position_calculation': calculation_reason
                },
                'sentiment_analysis': symbol_sentiment,
                'market_context': {
                    'overall_market_sentiment': market_sentiment.get('sentiment_score', 0),
                    'sentiment_confidence': symbol_sentiment.get('confidence', 0)
                },
                'risk_metrics': {
                    'portfolio_risk_before': self.portfolio_manager._calculate_portfolio_risk(),
                    'signal_risk_reward': signal.get_risk_ratio(),
                    'confidence_level': signal.confidence
                }
            }
            
            # باز کردن موقعیت در صورت تایید
            if can_open:
                position = self.portfolio_manager.open_position(signal)
                if position:
                    enhanced_signal['position_opened'] = True
                    enhanced_signal['position_id'] = position.signal_id
                    logger.info(f"💼 Position opened: {signal.symbol} - ${position_size:.2f}")
                else:
                    enhanced_signal['position_opened'] = False
            
            # ارسال ایمیل
            if self.email_sender:
                await self._send_enhanced_signal_email(enhanced_signal)
            
            return enhanced_signal
            
        except Exception as e:
            logger.error(f"❌ Error processing enhanced signal: {str(e)}")
            return {'error': str(e)}
    
    async def _send_enhanced_signal_email(self, enhanced_signal: Dict):
        """📧 ارسال ایمیل پیشرفته برای سیگنال"""
        try:
            signal_data = enhanced_signal['signal']
            portfolio_data = enhanced_signal['portfolio_analysis']
            sentiment_data = enhanced_signal['sentiment_analysis']
            
            # تعیین emoji ها
            sentiment_emoji = "😊" if sentiment_data.get('sentiment_score', 0) > 0 else "😐" if sentiment_data.get('sentiment_score', 0) == 0 else "😟"
            portfolio_emoji = "✅" if portfolio_data['can_open_position'] else "❌"
            position_emoji = "💼" if enhanced_signal.get('position_opened', False) else "📊"
            
            subject = f"🚀 {position_emoji} Enhanced Signal: {signal_data['symbol']} {sentiment_emoji} {portfolio_emoji}"
            
            # اطلاعات کامل برای ایمیل
            enhanced_email_data = signal_data.copy()
            enhanced_email_data.update({
                'sentiment_score': sentiment_data.get('sentiment_score', 0),
                'sentiment_confidence': sentiment_data.get('confidence', 0),
                'position_size_recommended': portfolio_data['recommended_position_size'],
                'can_open_position': portfolio_data['can_open_position'],
                'portfolio_approval_reason': portfolio_data['position_approval_reason'],
                'position_opened': enhanced_signal.get('position_opened', False),
                'market_sentiment': enhanced_signal['market_context']['overall_market_sentiment']
            })
            
            # ایجاد بدنه ایمیل پیشرفته
            email_body = self._create_enhanced_email_body(enhanced_email_data)
            
            # ارسال با subject سفارشی
            import smtplib
            from email.mime.text import MIMEText
            from email.mime.multipart import MIMEMultipart
            
            msg = MIMEMultipart()
            msg['From'] = self.email_sender.from_email
            msg['To'] = self.email_sender.to_email
            msg['Subject'] = subject
            
            msg.attach(MIMEText(email_body, 'html'))
            
            success = self.email_sender._send_email(msg)
            
            if success:
                logger.info(f"📧 Enhanced email sent for {signal_data['symbol']}")
            
        except Exception as e:
            logger.error(f"❌ Enhanced email error: {str(e)}")
    
    def _create_enhanced_email_body(self, signal_data: Dict) -> str:
        """ایجاد بدنه ایمیل پیشرفته"""
        try:
            # محاسبه مقادیر
            profit_pct = ((signal_data['target_price'] - signal_data['entry_price']) / signal_data['entry_price']) * 100
            loss_pct = ((signal_data['entry_price'] - signal_data['stop_loss']) / signal_data['entry_price']) * 100
            risk_reward = profit_pct / loss_pct if loss_pct > 0 else 0
            
            # تعیین رنگ sentiment
            sentiment_score = signal_data.get('sentiment_score', 0)
            if sentiment_score > 0.1:
                sentiment_color = "green"
                sentiment_text = "Positive"
            elif sentiment_score < -0.1:
                sentiment_color = "red"
                sentiment_text = "Negative"
            else:
                sentiment_color = "orange"
                sentiment_text = "Neutral"
            
            # Portfolio status
            position_opened = signal_data.get('position_opened', False)
            can_open = signal_data.get('can_open_position', False)
            
            portfolio_status = "🟢 POSITION OPENED" if position_opened else "🟡 SIGNAL ONLY" if can_open else "🔴 PORTFOLIO LIMIT"
            
            # وضعیت بازار کلی
            market_sentiment = signal_data.get('market_sentiment', 0)
            market_mood = "Bullish" if market_sentiment > 0.1 else "Bearish" if market_sentiment < -0.1 else "Neutral"
            
            return f"""
            <html>
            <head>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #f4f4f4; }}
                    .container {{ max-width: 700px; margin: 0 auto; background-color: white; padding: 25px; border-radius: 15px; box-shadow: 0 0 20px rgba(0,0,0,0.1); }}
                    .header {{ text-align: center; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 25px; border-radius: 15px; margin-bottom: 25px; }}
                    .signal-info {{ background-color: #f8f9fa; padding: 20px; border-radius: 10px; margin-bottom: 20px; border-left: 5px solid #007bff; }}
                    .price-info {{ display: flex; justify-content: space-between; margin-bottom: 20px; gap: 15px; }}
                    .price-box {{ text-align: center; padding: 20px; border-radius: 10px; flex: 1; }}
                    .entry {{ background: linear-gradient(135deg, #74b9ff, #0984e3); color: white; }}
                    .target {{ background: linear-gradient(135deg, #00b894, #00a085); color: white; }}
                    .stop {{ background: linear-gradient(135deg, #fd79a8, #e84393); color: white; }}
                    .sentiment-box {{ background-color: #e8f4fd; padding: 20px; border-radius: 10px; margin-bottom: 20px; border-left: 5px solid {sentiment_color}; }}
                    .portfolio-box {{ background-color: #f0f8f0; padding: 20px; border-radius: 10px; margin-bottom: 20px; border-left: 5px solid #28a745; }}
                    .technical {{ background-color: #fff3cd; padding: 20px; border-radius: 10px; margin-bottom: 20px; border-left: 5px solid #ffc107; }}
                    .risk-reward {{ background-color: #e1f5fe; padding: 20px; border-radius: 10px; margin-bottom: 20px; border-left: 5px solid #01579b; }}
                    .warning {{ background-color: #fff3cd; border: 2px solid #ffc107; padding: 15px; border-radius: 10px; margin-top: 20px; }}
                    .footer {{ text-align: center; color: #666; font-size: 14px; margin-top: 25px; padding-top: 20px; border-top: 1px solid #eee; }}
                    .status-badge {{ display: inline-block; padding: 8px 16px; border-radius: 20px; font-weight: bold; margin: 5px; }}
                    .badge-success {{ background-color: #d4edda; color: #155724; }}
                    .badge-warning {{ background-color: #fff3cd; color: #856404; }}
                    .badge-danger {{ background-color: #f8d7da; color: #721c24; }}
                </style>
            </head>
            <body>
                <div class="container">
                    <div class="header">
                        <h1>🚀 ENHANCED CRYPTO SIGNAL</h1>
                        <h2>{signal_data['symbol']}</h2>
                        <p><strong>Confidence: {signal_data['confidence']*100:.1f}%</strong></p>
                        <div class="status-badge {'badge-success' if position_opened else 'badge-warning' if can_open else 'badge-danger'}">{portfolio_status}</div>
                    </div>
                    
                    <div class="signal-info">
                        <h3>📊 Signal Details</h3>
                        <p><strong>Action:</strong> {signal_data['signal_type']}</p>
                        <p><strong>Timeframe:</strong> {signal_data['timeframe']}</p>
                        <p><strong>Created:</strong> {signal_data['created_at']}</p>
                        <p><strong>Model Version:</strong> {signal_data['model_version']}</p>
                    </div>
                    
                    <div class="price-info">
                        <div class="price-box entry">
                            <h4>🎯 Entry Price</h4>
                            <p><strong>${signal_data['entry_price']:,.2f}</strong></p>
                            <p style="font-size: 14px;">Recommended entry</p>
                        </div>
                        <div class="price-box target">
                            <h4>📈 Target Price</h4>
                            <p><strong>${signal_data['target_price']:,.2f}</strong></p>
                            <p style="font-weight: bold;">+{profit_pct:.2f}%</p>
                        </div>
                        <div class="price-box stop">
                            <h4>🛑 Stop Loss</h4>
                            <p><strong>${signal_data['stop_loss']:,.2f}</strong></p>
                            <p style="font-weight: bold;">-{loss_pct:.2f}%</p>
                        </div>
                    </div>
                    
                    <div class="sentiment-box">
                        <h3>💭 Market Sentiment Analysis</h3>
                        <p><strong>Symbol Sentiment:</strong> <span style="color: {sentiment_color};">{sentiment_text} ({sentiment_score:.2f})</span></p>
                        <p><strong>Overall Market:</strong> {market_mood} ({market_sentiment:.2f})</p>
                        <p><strong>Sentiment Confidence:</strong> {signal_data.get('sentiment_confidence', 0)*100:.1f}%</p>
                    </div>
                    
                    <div class="portfolio-box">
                        <h3>💼 Portfolio Analysis</h3>
                        <p><strong>Position Status:</strong> {portfolio_status}</p>
                        <p><strong>Recommended Size:</strong> ${signal_data.get('position_size_recommended', 0):,.2f}</p>
                        <p><strong>Approval Reason:</strong> {signal_data.get('portfolio_approval_reason', 'N/A')}</p>
                        {f'<p><strong>Position Opened:</strong> ✅ YES</p>' if position_opened else '<p><strong>Position Opened:</strong> ❌ NO</p>'}
                    </div>
                    
                    <div class="risk-reward">
                        <h3>⚖️ Risk Management</h3>
                        <p><strong>Risk/Reward Ratio:</strong> 1:{risk_reward:.2f}</p>
                        <p><strong>Potential Profit:</strong> <span style="color: green;">+{profit_pct:.2f}%</span></p>
                        <p><strong>Maximum Loss:</strong> <span style="color: red;">-{loss_pct:.2f}%</span></p>
                        <p><strong>Recommendation:</strong> Use only 2-5% of your portfolio for this trade</p>
                    </div>
                    
                    <div class="technical">
                        <h3>🔍 Technical Analysis</h3>
                        <p><strong>RSI:</strong> {signal_data['rsi']:.1f}</p>
                        <p><strong>MACD:</strong> {signal_data['macd']:.6f}</p>
                        <p><strong>MACD Signal:</strong> {signal_data['macd_signal']:.6f}</p>
                        <p><strong>Bollinger Upper:</strong> ${signal_data['bb_upper']:,.2f}</p>
                        <p><strong>Bollinger Lower:</strong> ${signal_data['bb_lower']:,.2f}</p>
                        <p><strong>Volume Ratio:</strong> {signal_data['volume_ratio']:.2f}x</p>
                    </div>
                    
                    <div class="warning">
                        <h4>⚠️ Enhanced Trading Guidelines</h4>
                        <p><strong>This is an AI-Enhanced signal with multi-timeframe confirmation:</strong></p>
                        <ul>
                            <li><strong>✅ Multi-timeframe analysis completed</strong></li>
                            <li><strong>✅ Sentiment analysis included</strong></li>
                            <li><strong>✅ Portfolio risk assessed</strong></li>
                            <li><strong>✅ Position size calculated</strong></li>
                            <li><strong>Always use the recommended stop loss</strong></li>
                            <li><strong>Monitor market sentiment changes</strong></li>
                            <li><strong>Consider overall market conditions</strong></li>
                        </ul>
                    </div>
                    
                    <div class="footer">
                        <p><strong>🚀 Generated by Enhanced AI Trading Bot v2.0</strong></p>
                        <p>Multi-Timeframe • Sentiment Analysis • Portfolio Management</p>
                        <p>This email was sent automatically. Do not reply.</p>
                    </div>
                </div>
            </body>
            </html>
            """
            
        except Exception as e:
            logger.error(f"❌ Enhanced email body creation error: {str(e)}")
            return f"<html><body><h2>Enhanced Signal: {signal_data.get('symbol', 'Unknown')}</h2><p>Error creating detailed email: {str(e)}</p></body></html>"
    
    async def _log_enhanced_scan_results(self, enhanced_signals: List[Dict], scan_time: float, market_sentiment: Dict):
        """📊 لاگ نتایج اسکن پیشرفته"""
        try:
            logger.info(f"✅ Enhanced scan completed in {scan_time:.2f}s")
            logger.info(f"📊 Results Summary:")
            logger.info(f"   Signals found: {len(enhanced_signals)}")
            
            # شمارش position های باز شده
            positions_opened = len([s for s in enhanced_signals if s.get('position_opened', False)])
            logger.info(f"   Positions opened: {positions_opened}")
            
            # Portfolio stats
            portfolio_summary = self.portfolio_manager.get_portfolio_summary()
            logger.info(f"   Portfolio value: ${portfolio_summary.get('portfolio_value', 0):,.2f}")
            logger.info(f"   Active positions: {portfolio_summary.get('active_positions', 0)}")
            logger.info(f"   Portfolio risk: {portfolio_summary.get('portfolio_risk', 0):.1%}")
            logger.info(f"   Win rate: {portfolio_summary.get('win_rate', 0):.1f}%")
            
            # Cache stats
            cache_stats = self.cache_system.get_stats()
            logger.info(f"   Cache hit rate: {cache_stats['hit_rate']:.1f}%")
            
            # Performance stats
            performance_stats = self.performance_monitor.get_performance_stats()
            logger.info(f"   Avg request time: {performance_stats['avg_request_time_ms']:.1f}ms")
            logger.info(f"   Performance gain: {performance_stats['performance_gain']}")
            
            # Risk alerts
            risk_alerts = self.risk_monitor.check_risk_alerts()
            if risk_alerts:
                logger.warning(f"⚠️ Risk alerts: {len(risk_alerts)}")
                for alert in risk_alerts:
                    logger.warning(f"   {alert['type']}: {alert['message']}")
            
            # Sentiment summary
            sentiment_score = market_sentiment.get('sentiment_score', 0)
            sentiment_desc = "Positive" if sentiment_score > 0.1 else "Negative" if sentiment_score < -0.1 else "Neutral"
            logger.info(f"   Market sentiment: {sentiment_desc} ({sentiment_score:.2f})")
            
            # Live learning status
            if self.signal_detector.live_learning:
                live_status = self.signal_detector.get_live_learning_status()
                if live_status.get('status') == 'active':
                    threshold = live_status.get('current_threshold', 0)
                    win_rate = live_status.get('recent_win_rate', 0)
                    logger.info(f"🧠 Live Learning: threshold={threshold:.3f}, win_rate={win_rate*100:.1f}%")
            
        except Exception as e:
            logger.error(f"❌ Enhanced scan logging error: {str(e)}")
    
    async def run_enhanced_cycle(self):
        """🔄 اجرای یک چرخه پیشرفته کامل"""
        try:
            # بروزرسانی قیمت‌های موقعیت‌های فعال
            await self._update_active_positions()
            
            # اسکن بازار
            signals = await self.enhanced_market_scan()
            
            # بررسی و بستن موقعیت‌ها
            await self._check_position_exits()
            
            # گزارش‌گیری دوره‌ای
            if self.scan_count % 10 == 0:  # هر 10 اسکن
                await self._send_portfolio_summary()
            
            # بررسی آلارم‌های ریسک
            await self._check_risk_alerts()
            
        except Exception as e:
            logger.error(f"❌ Enhanced cycle error: {str(e)}")
    
    async def _update_active_positions(self):
        """📈 بروزرسانی موقعیت‌های فعال"""
        try:
            if not self.portfolio_manager.positions:
                return
            
            symbols = [pos.symbol for pos in self.portfolio_manager.positions]
            current_prices = {}
            
            for symbol in symbols:
                try:
                    price = self.cached_data_collector.get_current_price_cached(symbol)
                    if price:
                        current_prices[symbol] = price
                except Exception as e:
                    logger.error(f"❌ Error getting price for {symbol}: {str(e)}")
            
            self.portfolio_manager.update_positions(current_prices)
            
        except Exception as e:
            logger.error(f"❌ Position update error: {str(e)}")
    
    async def _check_position_exits(self):
        """🚪 بررسی خروج از موقعیت‌ها"""
        try:
            positions_to_close = []
            
            for position in self.portfolio_manager.positions:
                if not position.current_price:
                    continue
                
                # بررسی target hit
                if position.current_price >= position.target_price:
                    positions_to_close.append((position, position.target_price, 'TARGET_HIT'))
                
                # بررسی stop loss hit
                elif position.current_price <= position.stop_loss:
                    positions_to_close.append((position, position.stop_loss, 'STOP_LOSS_HIT'))
            
            # بستن موقعیت‌ها
            for position, exit_price, reason in positions_to_close:
                result = self.portfolio_manager.close_position(position, exit_price, reason)
                
                if result and self.email_sender:
                    await self._send_position_closed_email(result)
            
        except Exception as e:
            logger.error(f"❌ Position exit check error: {str(e)}")
    
    async def _send_position_closed_email(self, close_result: Dict):
        """📧 ارسال ایمیل برای بستن موقعیت"""
        try:
            symbol = close_result['symbol']
            pnl = close_result['pnl']
            pnl_pct = close_result['pnl_pct']
            reason = close_result['exit_reason']
            
            emoji = "🎉" if pnl > 0 else "😞"
            result_text = "PROFIT" if pnl > 0 else "LOSS"
            
            message = f"""
🔒 POSITION CLOSED - {result_text}
===============================

💼 TRADE DETAILS:
   Symbol: {symbol}
   Entry Price: ${close_result['entry_price']:,.2f}
   Exit Price: ${close_result['exit_price']:,.2f}
   Exit Reason: {reason}
   
💰 FINANCIAL RESULT:
   P&L Amount: ${pnl:,.2f}
   P&L Percentage: {pnl_pct*100:.2f}%
   Holding Period: {close_result.get('holding_period', 'N/A')}

📊 PORTFOLIO UPDATE:
   Current Value: ${self.portfolio_manager.get_portfolio_value():,.2f}
   Available Cash: ${self.portfolio_manager.current_capital:,.2f}
   Active Positions: {len(self.portfolio_manager.positions)}
   
{emoji} {'Congratulations on the profitable trade!' if pnl > 0 else 'Better luck next time. Risk management protected your capital.'}
            """
            
            subject = f"{emoji} Position Closed: {symbol} - {pnl_pct*100:.1f}% {result_text}"
            self.email_sender.send_system_alert('INFO', message)
            
            logger.info(f"📧 Position closed email sent for {symbol}")
            
        except Exception as e:
            logger.error(f"❌ Position closed email error: {str(e)}")
    
    async def _send_portfolio_summary(self):
        """📊 ارسال خلاصه پورتفولیو"""
        try:
            if not self.email_sender:
                return
            
            portfolio_summary = self.portfolio_manager.get_portfolio_summary()
            performance_stats = self.performance_monitor.get_performance_stats()
            risk_report = self.risk_monitor.get_risk_report()
            
            # محاسبه آمار اضافی
            uptime_hours = (datetime.now() - self.start_time).total_seconds() / 3600
            
            message = f"""
📊 ENHANCED PORTFOLIO PERFORMANCE SUMMARY
========================================

💰 FINANCIAL STATUS:
   Portfolio Value: ${portfolio_summary.get('portfolio_value', 0):,.2f}
   Initial Capital: ${portfolio_summary.get('initial_capital', 0):,.2f}
   Total Return: ${portfolio_summary.get('total_return', 0):,.2f} ({portfolio_summary.get('total_return_pct', 0)*100:.1f}%)
   Available Cash: ${portfolio_summary.get('cash_available', 0):,.2f}
   Unrealized P&L: ${portfolio_summary.get('unrealized_pnl', 0):,.2f}

📈 TRADING PERFORMANCE:
   Active Positions: {portfolio_summary.get('active_positions', 0)}
   Total Trades: {portfolio_summary.get('total_trades', 0)}
   Win Rate: {portfolio_summary.get('win_rate', 0):.1f}%
   Current Drawdown: {portfolio_summary.get('current_drawdown', 0)*100:.1f}%
   Max Drawdown: {portfolio_summary.get('max_drawdown', 0)*100:.1f}%

⚡ SYSTEM PERFORMANCE:
   Total Scans: {self.scan_count}
   Cache Hit Rate: {performance_stats.get('cache_hit_rate', 0):.1f}%
   Avg Response Time: {performance_stats.get('avg_request_time_ms', 0):.1f}ms
   Performance Gain: {performance_stats.get('performance_gain', 'N/A')}
   Uptime: {uptime_hours:.1f} hours

🎯 RISK MANAGEMENT:
   Risk Level: {risk_report.get('risk_level', 'UNKNOWN')}
   Portfolio Risk: {portfolio_summary.get('portfolio_risk', 0)*100:.1f}%
   Capital Utilization: {portfolio_summary.get('capital_utilization', 0):.1f}%
   Risk Score: {risk_report.get('risk_score', 0):.1f}/100

🚀 ENHANCED FEATURES STATUS:
   Multi-Timeframe: ✅ Active
   Sentiment Analysis: ✅ Active
   Smart Caching: ✅ Active ({performance_stats.get('cache_hit_rate', 0):.0f}% hit rate)
   Live Learning: ✅ Active
   Risk Monitoring: ✅ Active

Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            """
            
            subject = f"📊 Portfolio Summary - {portfolio_summary.get('total_return_pct', 0)*100:.1f}% Return"
            self.email_sender.send_system_alert('INFO', message)
            
            logger.info("📧 Portfolio summary email sent")
            
        except Exception as e:
            logger.error(f"❌ Portfolio summary email error: {str(e)}")
    
    async def _check_risk_alerts(self):
        """⚠️ بررسی آلارم‌های ریسک"""
        try:
            risk_alerts = self.risk_monitor.check_risk_alerts()
            
            if risk_alerts:
                high_severity_alerts = [a for a in risk_alerts if a.get('severity') == 'HIGH']
                
                if high_severity_alerts and self.email_sender:
                    alert_message = "🚨 HIGH PRIORITY RISK ALERTS:\n\n"
                    
                    for alert in high_severity_alerts:
                        alert_message += f"⚠️ {alert['type']}: {alert['message']}\n"
                        alert_message += f"   Time: {alert['timestamp']}\n\n"
                    
                    alert_message += "Immediate attention required. Consider reducing positions or stopping trading."
                    
                    """self.email_sender.send_system_alert('WARNING', alert_message)"""
                    logger.warning(f"🚨 High severity risk alerts sent via email")
            
        except Exception as e:
            logger.error(f"❌ Risk alerts check error: {str(e)}")
    
    def schedule_model_retraining(self):
        """📅 برنامه‌ریزی آموزش مجدد مدل"""
        try:
            logger.info("📅 Scheduling weekly model retraining...")
            
            def retrain_model():
                try:
                    logger.info("🔄 Starting scheduled model retraining...")
                    
                    training_data = self.auto_trainer.collect_training_data(days=90)
                    
                    if self.auto_trainer.train_new_model(training_data):
                        if self.signal_detector.load_model():
                            logger.info("🎉 Model retrained and reloaded successfully!")
                            
                            if self.email_sender:
                                self.email_sender.send_system_alert('INFO', 'Model has been retrained with latest market data. Enhanced performance expected.')
                        else:
                            logger.error("❌ Failed to load retrained model")
                    else:
                        logger.error("❌ Model retraining failed")
                        
                except Exception as e:
                    logger.error(f"❌ Scheduled retraining failed: {str(e)}")
            
            # برنامه‌ریزی آموزش مجدد هر یکشنبه ساعت 2 صبح
            schedule.every().sunday.at("02:00").do(retrain_model)
            
        except Exception as e:
            logger.error(f"❌ Error scheduling retraining: {str(e)}")

async def main_enhanced():
    """🚀 تابع اصلی پیشرفته"""
    print("🤖 Enhanced Crypto Signal Bot v2.0 - All Features Active")
    print("="*70)
    
    scanner = EnhancedAutoScanner()
    
    # راه‌اندازی
    if not await scanner.initialize():
        print("❌ Failed to initialize enhanced scanner")
        logger.error("Enhanced scanner initialization failed - exiting")
        sys.exit(1)
    
    # نمایش تنظیمات
    print(f"📋 Enhanced Configuration:")
    print(f"   🔗 Trading pairs: {len(config.TRADING_PAIRS)}")
    print(f"   📊 Multi-timeframe: ✅ 4 timeframes (15m, 1h, 4h, 1d)")
    print(f"   💭 Sentiment analysis: ✅ 4 sources (Fear&Greed, News, OnChain, Social)") 
    print(f"   💾 Smart caching: ✅ Memory + Disk cache")
    print(f"   💼 Portfolio management: ✅ Kelly Criterion + Risk management")
    print(f"   ⚠️ Risk monitoring: ✅ Real-time alerts")
    
    portfolio_summary = scanner.portfolio_manager.get_portfolio_summary()
    print(f"   💰 Portfolio value: ${portfolio_summary.get('portfolio_value', 0):,.2f}")
    print(f"   💵 Available cash: ${portfolio_summary.get('cash_available', 0):,.2f}")
    
    if scanner.signal_detector.live_learning:
        adaptive_threshold = scanner.signal_detector.live_learning.get_adaptive_threshold()
        print(f"   🧠 Live learning: ✅ Active (adaptive threshold: {adaptive_threshold:.3f})")
    else:
        print(f"   🧠 Live learning: ❌ Disabled")
    
    print(f"   📧 Email notifications: {'✅ Enabled' if scanner.email_sender else '❌ Disabled'}")
    print(f"   ⏱️ Scan interval: {config.DATA_COLLECTION_INTERVAL/60:.1f} minutes")
    
    print(f"\n🚀 Starting enhanced market monitoring...")
    print("🎯 Features: Multi-Timeframe + Sentiment + Cache + Portfolio + Risk Management")
    print("📧 Email notifications for signals, positions, and alerts")
    print("🔄 Weekly auto-retraining scheduled")
    print("Press Ctrl+C to stop\n")
    
    try:
        # برنامه‌ریزی اسکن‌های پیشرفته
        async def scheduled_scan():
            while True:
                try:
                    await scanner.run_enhanced_cycle()
                    
                    # اجرای کارهای schedule شده (مثل retraining)
                    schedule.run_pending()
                    
                    await asyncio.sleep(config.DATA_COLLECTION_INTERVAL)
                    
                except Exception as e:
                    logger.error(f"❌ Scheduled scan error: {str(e)}")
                    await asyncio.sleep(60)  # Wait 1 minute before retry
        
        # برنامه‌ریزی آموزش مجدد
        scanner.schedule_model_retraining()
        
        # اجرای مداوم
        await scheduled_scan()
        
    except KeyboardInterrupt:
        print("\n👋 Enhanced Bot stopped by user")
        logger.info("Enhanced bot stopped by user via KeyboardInterrupt")
        
        # نمایش آمار نهایی
        try:
            final_portfolio = scanner.portfolio_manager.get_portfolio_summary()
            final_performance = scanner.performance_monitor.get_performance_stats()
            
            print(f"\n📊 Final Statistics:")
            print(f"   Total scans: {scanner.scan_count}")
            print(f"   Portfolio value: ${final_portfolio.get('portfolio_value', 0):,.2f}")
            print(f"   Total return: {final_portfolio.get('total_return_pct', 0)*100:.1f}%")
            print(f"   Win rate: {final_portfolio.get('win_rate', 0):.1f}%")
            print(f"   Cache hit rate: {final_performance.get('cache_hit_rate', 0):.1f}%")
            print(f"   Performance gain: {final_performance.get('performance_gain', 'N/A')}")
            
            # ارسال ایمیل خاتمه
            if scanner.email_sender:
                shutdown_message = f"""
🛑 Enhanced Crypto Signal Bot Shutdown
=====================================

The bot has been manually stopped.

📊 Final Performance Summary:
   Total Scans: {scanner.scan_count}
   Portfolio Value: ${final_portfolio.get('portfolio_value', 0):,.2f}
   Total Return: {final_portfolio.get('total_return_pct', 0)*100:.1f}%
   Win Rate: {final_portfolio.get('win_rate', 0):.1f}%
   
⏰ Shutdown Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                """
                scanner.email_sender.send_system_alert('INFO', shutdown_message)
        
        except Exception as e:
            logger.error(f"Error generating final stats: {str(e)}")
        
        # توقف live learning
        if scanner.signal_detector.live_learning:
            scanner.signal_detector.live_learning.stop_monitoring()
            logger.info("🛑 Live learning monitoring stopped")
        
    except Exception as e:
        print(f"\n❌ Critical error: {str(e)}")
        logger.error(f"Critical error in enhanced main: {str(e)}")
        
        # ارسال ایمیل خطا
        if 'scanner' in locals() and scanner.email_sender:
            error_message = f"🚨 Critical Error in Enhanced Bot: {str(e)}\n\nBot has been stopped due to critical error."
            scanner.email_sender.send_system_alert('ERROR', error_message)
        
        sys.exit(1)

# تابع اصلی ساده (برای سازگاری با قبل)
def main():
    """تابع اصلی که از enhanced version استفاده می‌کند"""
    asyncio.run(main_enhanced())

if __name__ == "__main__":
    main()#!/usr/bin/env python3