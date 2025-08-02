#!/usr/bin/env python3
"""
Automated Market Scanner - با Auto-Training
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backend.services.signal_detector import SignalDetector
from backend.services.email_sender import EmailSender
from backend.services.fallback_data_collector import FallbackDataCollector
from backend.services.feature_engineer import FeatureEngineer
from backend.services.ml_model import MLModel
from config import config
import logging
import time
import schedule
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict

# Fix Windows console encoding for emojis
if sys.platform.startswith('win'):
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.detach())

# Setup logging
try:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('data/logs/scanner.log', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
except:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('data/logs/scanner.log'),
            logging.StreamHandler()
        ]
    )

logger = logging.getLogger(__name__)

class AutoTrainingSystem:
    """
    سیستم آموزش خودکار مدل
    """
    
    def __init__(self):
        self.data_collector = FallbackDataCollector()
        self.feature_engineer = FeatureEngineer()
        self.ml_model = MLModel()
        
    def collect_training_data(self, days: int = 180) -> pd.DataFrame:
        """جمع‌آوری داده‌های آموزش"""
        try:
            logger.info(f"🔄 Collecting {days} days of training data...")
            
            # لیست جفت ارزهای اصلی برای آموزش
            training_pairs = [
                "BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT", "SOLUSDT", 
                "DOTUSDT", "LINKUSDT", "MATICUSDT", "LTCUSDT", "XRPUSDT",
                "AVAXUSDT", "UNIUSDT", "ATOMUSDT", "FILUSDT", "SANDUSDT"  # اضافی برای تنوع
            ]
            
            all_training_data = []
            end_time = datetime.now()
            start_time = end_time - timedelta(days=days)
            
            logger.info(f"📅 Training period: {start_time.strftime('%Y-%m-%d')} to {end_time.strftime('%Y-%m-%d')}")
            
            for symbol in training_pairs:
                try:
                    logger.info(f"📊 Collecting {symbol}...")
                    
                    # جمع‌آوری داده‌های تاریخی
                    df = self.data_collector.get_historical_klines(symbol, "1h", start_time, end_time)
                    
                    if df.empty or len(df) < 500:
                        logger.warning(f"⚠️ Insufficient data for {symbol}: {len(df)} candles")
                        continue
                    
                    # مهندسی ویژگی + ساخت target
                    df_features = self.feature_engineer.prepare_features(df)
                    
                    if df_features.empty:
                        logger.warning(f"⚠️ Feature engineering failed for {symbol}")
                        continue
                    
                    # اضافه کردن اطلاعات symbol
                    df_features['source_symbol'] = symbol
                    all_training_data.append(df_features)
                    
                    logger.info(f"✅ {symbol}: {len(df_features)} samples collected")
                    
                    # Rate limiting
                    time.sleep(0.5)
                    
                except Exception as e:
                    logger.error(f"❌ Error collecting {symbol}: {str(e)}")
                    continue
            
            if not all_training_data:
                raise ValueError("No training data collected!")
            
            # ترکیب تمام داده‌ها
            combined_data = pd.concat(all_training_data, ignore_index=True)
            
            # حذف ستون symbol برای آموزش
            if 'source_symbol' in combined_data.columns:
                combined_data = combined_data.drop('source_symbol', axis=1)
            
            logger.info(f"🎯 Total training data: {len(combined_data)} samples")
            logger.info(f"📈 Positive samples: {combined_data['target'].sum()} ({combined_data['target'].mean()*100:.1f}%)")
            
            return combined_data
            
        except Exception as e:
            logger.error(f"❌ Error collecting training data: {str(e)}")
            raise
    
    def train_new_model(self, training_data: pd.DataFrame) -> bool:
        """آموزش مدل جدید"""
        try:
            logger.info("🧠 Starting model training...")
            
            if len(training_data) < 1000:
                logger.warning("⚠️ Limited training data - model performance may vary")
            
            # آموزش مدل
            model_info = self.ml_model.train(training_data)
            
            # ذخیره مدل
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
            # بررسی وجود مدل
            if self.ml_model.load_latest_model():
                logger.info("✅ Existing model found and loaded")
                return True
            
            logger.warning("⚠️ No model found - starting auto-training...")
            
            # جمع‌آوری داده‌ها
            training_data = self.collect_training_data(days=120)  # 4 ماه داده
            
            # آموزش مدل
            success = self.train_new_model(training_data)
            
            if success:
                # بارگذاری مدل جدید
                if self.ml_model.load_latest_model():
                    logger.info("🚀 New model trained and loaded successfully!")
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"❌ Auto-training failed: {str(e)}")
            return False

class AutomatedScanner:
    """
    اسکنر خودکار با آموزش خودکار
    """
    
    def __init__(self):
        self.signal_detector = SignalDetector()
        self.email_sender = EmailSender()
        self.auto_trainer = AutoTrainingSystem()
        self.daily_signals = []
        self.scan_count = 0
        self.start_time = datetime.now()
        
        # Create logs directory
        os.makedirs('data/logs', exist_ok=True)
        os.makedirs('data/models', exist_ok=True)
        
    def initialize(self) -> bool:
        """راه‌اندازی اسکنر با آموزش خودکار"""
        try:
            logger.info("🚀 Initializing Automated Scanner with Auto-Training...")
            
            # 🧠 تلاش برای بارگذاری مدل یا آموزش جدید
            try:
                # آموزش خودکار در صورت نیاز
                model_ready = self.auto_trainer.auto_train_if_needed()
                
                if model_ready:
                    # بارگذاری مدل در signal detector
                    if self.signal_detector.load_model():
                        logger.info(f"✅ ML model ready: {self.signal_detector.ml_model.model_info.get('version', 'unknown')}")
                        
                        # بررسی live learning
                        if self.signal_detector.live_learning:
                            logger.info("🧠 Live Learning System: ACTIVE")
                            learning_summary = self.signal_detector.live_learning.get_learning_summary()
                            logger.info(learning_summary)
                        else:
                            logger.warning("⚠️ Live Learning System: FAILED TO START")
                    else:
                        logger.error("❌ Failed to load trained model")
                        return False
                else:
                    logger.error("❌ Model training failed - cannot continue")
                    return False
                    
            except Exception as e:
                logger.error(f"❌ Model initialization failed: {str(e)}")
                return False
            
            # تنظیم email
            try:
                if not self.email_sender.validate_config():
                    logger.warning("⚠️ Email configuration invalid - notifications disabled")
                    self.email_sender = None
                else:
                    logger.info("✅ Email notifications enabled")
            except Exception as e:
                logger.warning(f"⚠️ Email setup failed: {str(e)}")
                self.email_sender = None
            
            logger.info("🎯 Scanner initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize scanner: {str(e)}")
            return False
    
    def schedule_model_retraining(self):
        """برنامه‌ریزی آموزش مجدد مدل"""
        try:
            logger.info("📅 Scheduling weekly model retraining...")
            
            def retrain_model():
                try:
                    logger.info("🔄 Starting scheduled model retraining...")
                    
                    # جمع‌آوری داده‌های جدید
                    training_data = self.auto_trainer.collect_training_data(days=90)
                    
                    # آموزش مدل جدید
                    if self.auto_trainer.train_new_model(training_data):
                        # بارگذاری مدل جدید
                        if self.signal_detector.load_model():
                            logger.info("🎉 Model retrained and reloaded successfully!")
                            
                            # ارسال اطلاع‌رسانی
                            if self.email_sender:
                                self.send_system_alert('INFO', 'Model has been retrained with latest market data')
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
    
    # بقیه متدها مثل قبل...
    def scan_markets(self) -> List[Dict]:
        """اسکن بازارها"""
        try:
            logger.info("🔍 Starting market scan...")
            start_time = time.time()
            
            signals = self.signal_detector.scan_markets()
            
            scan_time = time.time() - start_time
            self.scan_count += 1
            
            logger.info(f"✅ Market scan completed in {scan_time:.2f}s")
            logger.info(f"📊 Scanned {len(config.TRADING_PAIRS)} pairs, found {len(signals)} signals")
            
            if signals:
                for signal in signals:
                    self.process_signal(signal)
                    self.daily_signals.append(signal.to_dict())
            
            self.log_scan_stats(len(signals), scan_time)
            
            return [signal.to_dict() for signal in signals]
            
        except Exception as e:
            logger.error(f"❌ Error during market scan: {str(e)}")
            return []
    
    def process_signal(self, signal) -> None:
        """پردازش سیگنال جدید"""
        try:
            signal_emoji = "📈" if signal.signal_type == 'BUY' else "📉"
            direction = "LONG" if signal.signal_type == 'BUY' else "SHORT"
            
            logger.info(f"{signal_emoji} Processing {direction} signal: {signal.symbol} @ {signal.confidence:.3f}")
            
            if self.email_sender:
                success = self.email_sender.send_signal_email(signal.to_dict())
                if success:
                    logger.info(f"📧 Email sent for {signal.symbol}")
                else:
                    logger.error(f"❌ Failed to send email for {signal.symbol}")
            
            logger.info(f"🎯 {direction} Signal Details:")
            logger.info(f"   Symbol: {signal.symbol}")
            logger.info(f"   Entry: ${signal.entry_price:.6f}")
            logger.info(f"   Target: ${signal.target_price:.6f} ({signal.get_profit_potential()*100:+.3f}%)")
            logger.info(f"   Stop: ${signal.stop_loss:.6f}")
            logger.info(f"   Confidence: {signal.confidence:.3f}")
            logger.info(f"   Risk/Reward: {signal.get_risk_ratio():.2f}")
            
        except Exception as e:
            logger.error(f"❌ Error processing signal: {str(e)}")

    def log_scan_stats(self, signals_found: int, scan_time: float) -> None:
        """لاگ آمار اسکن"""
        try:
            active_signals = len(self.signal_detector.get_active_signals())
            uptime = datetime.now() - self.start_time
            
            logger.info(f"📊 Scan Stats:")
            logger.info(f"   Scan #{self.scan_count}")
            logger.info(f"   Signals found: {signals_found}")
            logger.info(f"   Active signals: {active_signals}")
            logger.info(f"   Scan time: {scan_time:.2f}s")
            logger.info(f"   Daily signals: {len(self.daily_signals)}")
            
            # Live Learning Stats
            if self.signal_detector.live_learning:
                live_status = self.signal_detector.get_live_learning_status()
                
                if live_status.get('status') == 'active':
                    logger.info(f"🧠 Live Learning:")
                    logger.info(f"   Adaptive threshold: {live_status.get('current_threshold', 0):.3f}")
                    logger.info(f"   Recent win rate: {live_status.get('recent_win_rate', 0)*100:.1f}%")
                    logger.info(f"   Signals learned: {live_status.get('total_signals_learned', 0)}")
                    logger.info(f"   Active tracking: {live_status.get('active_signals', 0)}")
                elif live_status.get('status') == 'initializing':
                    logger.info(f"🧠 Live Learning: Initializing... ({live_status.get('total_completed', 0)} completed signals)")
            
        except Exception as e:
            logger.error(f"❌ Error logging stats: {str(e)}")
    
    def send_system_alert(self, alert_type: str, message: str) -> None:
        """ارسال هشدار سیستم"""
        try:
            if self.email_sender:
                self.email_sender.send_system_alert(alert_type, message)
            logger.log(getattr(logging, alert_type, logging.INFO), f"ALERT: {message}")
            
        except Exception as e:
            logger.error(f"❌ Error sending system alert: {str(e)}")
    
    def run_once(self) -> None:
        """اجرای یک چرخه اسکن"""
        try:
            signals = self.scan_markets()
            
            if signals:
                logger.info(f"🚨 {len(signals)} signals generated!")
                for signal in signals:
                    signal_emoji = "📈" if signal['signal_type'] == 'BUY' else "📉"
                    direction = "LONG" if signal['signal_type'] == 'BUY' else "SHORT"
                    logger.info(f"   {signal_emoji} {signal['symbol']}: {direction} {signal['confidence']*100:.1f}% confidence")
            else:
                logger.info("⚪ No signals found in current scan")
            
        except Exception as e:
            logger.error(f"❌ Error in scanning cycle: {str(e)}")
            self.send_system_alert('ERROR', f"Scanner error: {str(e)}")
            
    def run_continuous(self) -> None:
        """اجرای مداوم اسکنر"""
        try:
            logger.info("🔄 Starting continuous scanning mode...")
            logger.info(f"📅 Scan interval: {config.DATA_COLLECTION_INTERVAL/60:.1f} minutes")
            
            # برنامه‌ریزی اسکن
            schedule.every(config.DATA_COLLECTION_INTERVAL//60).minutes.do(self.run_once)
            
            # برنامه‌ریزی آموزش مجدد
            self.schedule_model_retraining()
            
            # ارسال اطلاع‌رسانی شروع
            startup_message = 'Crypto Signal Bot with Auto-Training started successfully'
            if self.signal_detector.live_learning:
                threshold = self.signal_detector.live_learning.get_adaptive_threshold()
                startup_message += f' (Adaptive threshold: {threshold:.3f})'
            
            self.send_system_alert('INFO', startup_message)
            
            # حلقه اصلی
            while True:
                schedule.run_pending()
                time.sleep(60)
                
        except KeyboardInterrupt:
            logger.info("👋 Scanning stopped by user")
            
            if self.signal_detector.live_learning:
                self.signal_detector.live_learning.stop_monitoring()
                logger.info("🛑 Live learning monitoring stopped")
            
            self.send_system_alert('INFO', 'Crypto Signal Bot stopped by user.')
            
        except Exception as e:
            logger.error(f"❌ Error in continuous scanning: {str(e)}")
            self.send_system_alert('ERROR', f"Scanner crashed: {str(e)}")

def main():
    """تابع اصلی"""
    print("🤖 Crypto Signal Bot - Auto-Training Scanner")
    print("="*60)
    
    scanner = AutomatedScanner()
    
    if not scanner.initialize():
        print("❌ Failed to initialize scanner")
        logger.error("Scanner initialization failed - exiting")
        sys.exit(1)
    
    print(f"📋 Configuration:")
    print(f"   Trading pairs: {len(config.TRADING_PAIRS)}")
    print(f"   Confidence threshold: {config.CONFIDENCE_THRESHOLD}")
    
    if scanner.signal_detector.live_learning:
        adaptive_threshold = scanner.signal_detector.live_learning.get_adaptive_threshold()
        print(f"   Live Learning: ACTIVE")
        print(f"   Adaptive threshold: {adaptive_threshold:.3f}")
    else:
        print(f"   Live Learning: DISABLED")
    
    print(f"   Scan interval: {config.DATA_COLLECTION_INTERVAL/60:.1f} minutes")
    print(f"   Email notifications: {'YES' if scanner.email_sender else 'NO'}")
    print(f"   Auto-training: ENABLED (weekly)")
    
    print("\n🔄 Starting continuous market monitoring...")
    print("Bot will automatically train if no model exists")
    print("Weekly retraining scheduled for Sundays at 2 AM")
    print("Press Ctrl+C to stop")
    
    try:
        scanner.run_continuous()
        
    except KeyboardInterrupt:
        print("\n👋 Bot stopped by user")
        logger.info("Bot stopped by user via KeyboardInterrupt")
    except Exception as e:
        print(f"\n❌ Critical error: {str(e)}")
        logger.error(f"Critical error in main: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()