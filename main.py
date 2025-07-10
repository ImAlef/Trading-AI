#!/usr/bin/env python3
"""
Automated Market Scanner - Main Bot Application
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backend.services.signal_detector import SignalDetector
from backend.services.email_sender import EmailSender
from config import config
import logging
import time
import schedule
from datetime import datetime, timedelta
from typing import List, Dict

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data/logs/scanner.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AutomatedScanner:
    """
    Main automated scanning application
    """
    
    def __init__(self):
        self.signal_detector = SignalDetector()
        self.email_sender = EmailSender()
        self.daily_signals = []
        self.scan_count = 0
        self.start_time = datetime.now()
        
        # Create logs directory
        os.makedirs('data/logs', exist_ok=True)
        
    def initialize(self) -> bool:
        """Initialize the scanner"""
        try:
            logger.info("🚀 Initializing Automated Scanner...")
            
            # Load ML model
            if not self.signal_detector.load_model():
                logger.error("Failed to load ML model")
                return False
            
            logger.info(f"✅ ML model loaded: {self.signal_detector.ml_model.model_info.get('version', 'unknown')}")
            
            # Validate email configuration
            if not self.email_sender.validate_config():
                logger.warning("Email configuration invalid - notifications disabled")
                self.email_sender = None
            else:
                logger.info("✅ Email notifications enabled")
            
            logger.info("🎯 Scanner initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize scanner: {str(e)}")
            return False
    
    def scan_markets(self) -> List[Dict]:
        """Scan all markets for signals"""
        try:
            logger.info("🔍 Starting market scan...")
            start_time = time.time()
            
            # Scan markets
            signals = self.signal_detector.scan_markets()
            
            scan_time = time.time() - start_time
            self.scan_count += 1
            
            logger.info(f"✅ Market scan completed in {scan_time:.2f}s")
            logger.info(f"📊 Scanned {len(config.TRADING_PAIRS)} pairs, found {len(signals)} signals")
            
            # Process new signals
            if signals:
                for signal in signals:
                    self.process_signal(signal)
                    self.daily_signals.append(signal.to_dict())
            
            # Log scan statistics
            self.log_scan_stats(len(signals), scan_time)
            
            return [signal.to_dict() for signal in signals]
            
        except Exception as e:
            logger.error(f"Error during market scan: {str(e)}")
            return []
    
    def process_signal(self, signal) -> None:
        """Process a new signal"""
        try:
            logger.info(f"📈 Processing signal: {signal.symbol} @ {signal.confidence:.3f}")
            
            # Send email notification
            if self.email_sender:
                success = self.email_sender.send_signal_email(signal.to_dict())
                if success:
                    logger.info(f"📧 Email sent for {signal.symbol}")
                else:
                    logger.error(f"❌ Failed to send email for {signal.symbol}")
            
            # Log signal details
            logger.info(f"🎯 Signal Details:")
            logger.info(f"   Symbol: {signal.symbol}")
            logger.info(f"   Entry: ${signal.entry_price:.2f}")
            logger.info(f"   Target: ${signal.target_price:.2f} (+{signal.get_profit_potential()*100:.2f}%)")
            logger.info(f"   Stop: ${signal.stop_loss:.2f}")
            logger.info(f"   Confidence: {signal.confidence:.3f}")
            logger.info(f"   Risk/Reward: {signal.get_risk_ratio():.2f}")
            
        except Exception as e:
            logger.error(f"Error processing signal: {str(e)}")
    
    def log_scan_stats(self, signals_found: int, scan_time: float) -> None:
        """Log scanning statistics"""
        try:
            active_signals = len(self.signal_detector.get_active_signals())
            uptime = datetime.now() - self.start_time
            
            logger.info(f"📊 Scan Stats:")
            logger.info(f"   Scan #{self.scan_count}")
            logger.info(f"   Signals found: {signals_found}")
            logger.info(f"   Active signals: {active_signals}")
            logger.info(f"   Scan time: {scan_time:.2f}s")
            logger.info(f"   Uptime: {uptime}")
            logger.info(f"   Daily signals: {len(self.daily_signals)}")
            
        except Exception as e:
            logger.error(f"Error logging stats: {str(e)}")
    
    def send_daily_summary(self) -> None:
        """Send daily summary email"""
        try:
            if not self.email_sender:
                logger.info("Email not configured - skipping daily summary")
                return
            
            logger.info("📊 Sending daily summary...")
            
            # Calculate summary statistics
            summary_data = {
                'pairs_scanned': len(config.TRADING_PAIRS),
                'scans_performed': self.scan_count,
                'success_rate': self.calculate_success_rate(),
                'avg_confidence': self.calculate_avg_confidence(),
                'uptime_hours': (datetime.now() - self.start_time).total_seconds() / 3600
            }
            
            # Send summary email
            success = self.email_sender.send_summary_email(self.daily_signals, summary_data)
            
            if success:
                logger.info("✅ Daily summary sent successfully")
            else:
                logger.error("❌ Failed to send daily summary")
            
            # Reset daily signals
            self.daily_signals = []
            
        except Exception as e:
            logger.error(f"Error sending daily summary: {str(e)}")
    
    def calculate_success_rate(self) -> float:
        """Calculate signal success rate"""
        try:
            if not self.daily_signals:
                return 0.0
            
            # This is a simplified calculation
            # In a real implementation, you'd track actual outcomes
            return len(self.daily_signals) / self.scan_count * 100 if self.scan_count > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating success rate: {str(e)}")
            return 0.0
    
    def calculate_avg_confidence(self) -> float:
        """Calculate average confidence of signals"""
        try:
            if not self.daily_signals:
                return 0.0
            
            total_confidence = sum(signal['confidence'] for signal in self.daily_signals)
            return (total_confidence / len(self.daily_signals)) * 100
            
        except Exception as e:
            logger.error(f"Error calculating average confidence: {str(e)}")
            return 0.0
    
    def send_system_alert(self, alert_type: str, message: str) -> None:
        """Send system alert"""
        try:
            if self.email_sender:
                self.email_sender.send_system_alert(alert_type, message)
            logger.log(getattr(logging, alert_type, logging.INFO), f"ALERT: {message}")
            
        except Exception as e:
            logger.error(f"Error sending system alert: {str(e)}")
    
    def run_once(self) -> None:
        """Run one scanning cycle"""
        try:
            signals = self.scan_markets()
            
            if signals:
                logger.info(f"🚨 {len(signals)} signals generated!")
                for signal in signals:
                    logger.info(f"   📈 {signal['symbol']}: {signal['confidence']*100:.1f}% confidence")
            else:
                logger.info("⚪ No signals found in current scan")
            
        except Exception as e:
            logger.error(f"Error in scanning cycle: {str(e)}")
            self.send_system_alert('ERROR', f"Scanner error: {str(e)}")
    
    def run_continuous(self) -> None:
        """Run continuous scanning"""
        try:
            logger.info("🔄 Starting continuous scanning mode...")
            logger.info(f"📅 Scan interval: {config.DATA_COLLECTION_INTERVAL/60:.1f} minutes")
            
            # Schedule scanning
            schedule.every(config.DATA_COLLECTION_INTERVAL//60).minutes.do(self.run_once)
            
            # Schedule daily summary (at 9 AM)
            schedule.every().day.at("09:00").do(self.send_daily_summary)
            
            # Send startup notification
            self.send_system_alert('INFO', 'Crypto Signal Bot started successfully and is monitoring the markets.')
            
            # Main loop
            while True:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
                
        except KeyboardInterrupt:
            logger.info("👋 Scanning stopped by user")
            self.send_system_alert('INFO', 'Crypto Signal Bot stopped by user.')
        except Exception as e:
            logger.error(f"Error in continuous scanning: {str(e)}")
            self.send_system_alert('ERROR', f"Scanner crashed: {str(e)}")

def main():
    """Main function - Automatically starts continuous monitoring"""
    print("🤖 Crypto Signal Bot - Automated Scanner")
    print("="*50)
    
    # Initialize scanner
    scanner = AutomatedScanner()
    
    if not scanner.initialize():
        print("❌ Failed to initialize scanner")
        logger.error("Scanner initialization failed - exiting")
        sys.exit(1)
    
    print(f"📋 Configuration:")
    print(f"   Trading pairs: {len(config.TRADING_PAIRS)}")
    print(f"   Confidence threshold: {config.CONFIDENCE_THRESHOLD}")
    print(f"   Scan interval: {config.DATA_COLLECTION_INTERVAL/60:.1f} minutes")
    print(f"   Email notifications: {'✅' if scanner.email_sender else '❌'}")
    
    # Automatically start continuous scanning
    print("\n🔄 Starting continuous market monitoring...")
    print("Bot will run 24/7 and scan markets automatically")
    print("Press Ctrl+C to stop (but this is designed to run forever)")
    
    try:
        scanner.run_continuous()
        
    except KeyboardInterrupt:
        print("\n👋 Bot stopped by user")
        logger.info("Bot stopped by user via KeyboardInterrupt")
    except Exception as e:
        print(f"\n❌ Critical error: {str(e)}")
        logger.error(f"Critical error in main: {str(e)}")
        import traceback
        traceback.print_exc()
        # Try to send alert before exiting
        try:
            scanner.send_system_alert('CRITICAL', f"Bot crashed with error: {str(e)}")
        except:
            pass
        sys.exit(1)

if __name__ == "__main__":
    main()