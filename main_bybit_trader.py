#!/usr/bin/env python3
"""
Automated Crypto Trading Bot - Bybit Edition with RR-Based Leverage
"""
import sys
import os
import time
import signal
from datetime import datetime
import logging
from dotenv import load_dotenv
load_dotenv()
# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backend.services.bybit_auto_trader import BybitAutoTrader
from backend.services.bybit_api import BybitAPI
from config import config

# Fix Windows console encoding for emojis
if sys.platform.startswith('win'):
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.detach())

# Setup logging
try:
    os.makedirs('data/logs', exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('data/logs/bybit_trader.log', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
except:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('data/logs/bybit_trader.log'),
            logging.StreamHandler()
        ]
    )

logger = logging.getLogger(__name__)

class BybitTradingBotManager:
    """
    مدیریت ربات ترید خودکار Bybit
    """
    
    def __init__(self):
        self.auto_trader = None
        self.is_running = False
        
        # Signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"🛑 Received signal {signum}, shutting down...")
        self.shutdown()
        sys.exit(0)
        
    def initialize(self) -> bool:
        """راه‌اندازی سیستم"""
        try:
            print("🤖 Initializing Bybit Automated Trading Bot...")
            print("="*60)
            
            # Get API credentials
            api_key = os.getenv("BYBIT_API_KEY")
            if not api_key:
                print("❌ API Key is required!")
                return False
                
            secret_key = os.getenv("BYBIT_SECRET_KEY")
            if not secret_key:
                print("❌ Secret Key is required!")
                return False
            
            # Ask for testnet/mainnet
            use_testnet = input("🧪 Use Testnet? (y/n) [default: y]: ").strip().lower()
            testnet = use_testnet != 'n'
            
            if testnet:
                print("⚠️  TESTNET MODE - No real money will be used")
                print("💡 You can get testnet USDT from: https://testnet.bybit.com/faucet/usdt")
            else:
                confirm = input("🚨 MAINNET MODE - Real money will be used! Continue? (yes/no): ").strip().lower()
                if confirm != 'yes':
                    print("❌ Cancelled by user")
                    return False
            
            # Initialize AutoTrader
            self.auto_trader = BybitAutoTrader(api_key, secret_key, testnet)
            
            # Test API connection
            health = self.auto_trader.bybit_api.health_check()
            if health['status'] != 'OK':
                print(f"❌ API Connection Failed: {health['message']}")
                return False
            
            print(f"✅ API Connection: {health['status']}")
            print(f"💰 Account Balance: ${health['balance']:.2f} USDT")
            print(f"🧪 Mode: {'TESTNET' if health['testnet'] else 'LIVE'}")
            
            # Display settings
            self._display_settings()
            
            # Final confirmation
            start_trading = input("\n🚀 Start Automated Trading? (y/n): ").strip().lower()
            if start_trading != 'y':
                print("❌ Trading not started")
                return False
            
            return True
            
        except KeyboardInterrupt:
            print("\n❌ Setup cancelled by user")
            return False
        except Exception as e:
            logger.error(f"❌ Initialization failed: {str(e)}")
            return False
    
    def _display_settings(self):
        """نمایش تنظیمات فعلی"""
        print("\n📋 Current Settings (RR-Based Strategy):")
        print("="*50)
        print("💪 Leverage Strategy:")
        print("   • RR > 3:  35x leverage + 30% position size")
        print("   • RR 2-3:  20x leverage + 20% position size")
        print("   • RR 1-2:  10x leverage + 10% position size")
        print("   • RR < 1:   5x leverage +  5% position size")
        print(f"📊 Max Concurrent Trades: 5")
        print(f"🚨 Emergency Stop: 25% total loss")
        print(f"⏰ Scan Interval: 5 minutes")
        print(f"🪙 Trading Pairs: {len(config.TRADING_PAIRS)} pairs")
        print(f"📧 Email Notifications: {'YES' if config.SMTP_USERNAME else 'NO'}")
    
    def start(self):
        """شروع ربات"""
        try:
            if not self.auto_trader:
                logger.error("AutoTrader not initialized")
                return False
            
            # Start automated trading
            if self.auto_trader.start_trading():
                self.is_running = True
                logger.info("🚀 Bybit Automated Trading Bot Started Successfully!")
                
                # Display real-time status
                self._run_status_display()
                return True
            else:
                logger.error("❌ Failed to start automated trading")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error starting bot: {str(e)}")
            return False
    
    def _run_status_display(self):
        """نمایش وضعیت لایو"""
        logger.info("📊 Live Status Display Started")
        logger.info("Press Ctrl+C to stop the bot")
        
        try:
            while self.is_running:
                # Get trading summary
                summary = self.auto_trader.get_trading_summary()
                
                # Display current status
                print(f"\n{'='*60}")
                print(f"🤖 BYBIT TRADING BOT - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"{'='*60}")
                
                if 'error' in summary:
                    print(f"❌ Error: {summary['error']}")
                else:
                    mode_icon = "🧪" if summary['testnet'] else "🔴"
                    print(f"{mode_icon} Mode: {'TESTNET' if summary['testnet'] else 'LIVE'}")
                    print(f"💰 Balance: ${summary['balance']:.2f} USDT")
                    print(f"📊 Active Trades: {summary['active_trades']}/{summary['max_trades']}")
                    print(f"💸 Unrealized P/L: ${summary['total_unrealized_pnl']:.2f}")
                    print(f"⚡ Status: {'ACTIVE' if summary['is_active'] else 'STOPPED'}")
                    
                    # Show active trades
                    if summary['trades_summary']:
                        print(f"\n📈 Active Positions:")
                        for trade in summary['trades_summary']:
                            status_emoji = "⏳" if trade['status'] == 'PENDING' else "✅"
                            pnl_color = "+" if trade['pnl'] >= 0 else ""
                            print(f"   {status_emoji} {trade['symbol']}: {pnl_color}${trade['pnl']:.2f} ({trade['leverage']}x)")
                    else:
                        print(f"\n📭 No active positions")
                    
                    # Show recent performance
                    try:
                        perf = self.auto_trader.get_performance_stats(7)  # Last 7 days
                        if 'total_trades' in perf:
                            print(f"\n📊 7-Day Performance:")
                            print(f"   Trades: {perf['total_trades']} (WR: {perf['win_rate']:.1f}%)")
                            print(f"   Total P/L: ${perf['total_pnl']:.2f}")
                            print(f"   Best Trade: ${perf['best_trade']:.2f}")
                    except:
                        pass
                
                print(f"{'='*60}")
                
                # Wait before next update
                time.sleep(30)  # Update every 30 seconds
                
        except KeyboardInterrupt:
            logger.info("\n🛑 Status display stopped by user")
        except Exception as e:
            logger.error(f"Error in status display: {str(e)}")
    
    def shutdown(self):
        """خاموش کردن سیستم"""
        try:
            logger.info("🛑 Shutting down Bybit trading bot...")
            self.is_running = False
            
            if self.auto_trader:
                # Get final summary
                summary = self.auto_trader.get_trading_summary()
                
                if 'active_trades' in summary and summary['active_trades'] > 0:
                    logger.warning(f"⚠️  {summary['active_trades']} active trades will continue running")
                    
                    # Ask if user wants to close all positions
                    try:
                        close_all = input("\n❓ Close all active positions? (y/n): ").strip().lower()
                        if close_all == 'y':
                            logger.info("🔄 Closing all positions...")
                            self.auto_trader._emergency_close_all()
                            time.sleep(5)  # Wait for orders to process
                    except:
                        pass
                
                # Stop trading
                self.auto_trader.stop_trading()
                
                # Final summary
                final_balance = self.auto_trader.bybit_api.get_balance()
                if final_balance:
                    logger.info(f"💰 Final Balance: ${final_balance:.2f} USDT")
            
            logger.info("✅ Bot shutdown completed")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {str(e)}")
    
    def run_manual_commands(self):
        """حالت دستی برای کنترل ربات"""
        try:
            while self.is_running:
                print("\n🎛️  Manual Commands:")
                print("1. Show Status")
                print("2. Show Performance")
                print("3. Close Specific Trade")
                print("4. Emergency Close All")
                print("5. Stop Bot")
                print("0. Return to Auto Mode")
                
                choice = input("\nSelect option: ").strip()
                
                if choice == '1':
                    summary = self.auto_trader.get_trading_summary()
                    print(f"\n{summary}")
                    
                elif choice == '2':
                    days = input("Enter days (default 7): ").strip()
                    days = int(days) if days.isdigit() else 7
                    perf = self.auto_trader.get_performance_stats(days)
                    print(f"\n{perf}")
                    
                elif choice == '3':
                    symbol = input("Enter symbol to close (e.g., BTCUSDT): ").strip().upper()
                    success = self.auto_trader.force_close_trade(symbol)
                    print(f"✅ Trade closed" if success else "❌ Trade not found or failed to close")
                    
                elif choice == '4':
                    confirm = input("⚠️  Close ALL positions? (yes/no): ").strip().lower()
                    if confirm == 'yes':
                        self.auto_trader._emergency_close_all()
                        print("✅ All positions closed")
                        
                elif choice == '5':
                    self.shutdown()
                    break
                    
                elif choice == '0':
                    break
                    
                else:
                    print("❌ Invalid option")
                    
        except KeyboardInterrupt:
            pass
        except Exception as e:
            logger.error(f"Error in manual commands: {str(e)}")

def main():
    """تابع اصلی"""
    print("🤖 Bybit Automated Trading Bot - RR-Based Leverage Strategy")
    print("="*60)
    print("⚠️  WARNING: This bot trades with leverage on Bybit!")
    print("📚 Make sure you understand leverage trading risks")
    print("🔒 Never share your API keys with anyone")
    print("🧪 Start with TESTNET to learn the system")
    print(f"🌐 Need VPN for some regions")
    print("="*60)
    
    bot_manager = BybitTradingBotManager()
    
    try:
        # Initialize
        if not bot_manager.initialize():
            logger.error("❌ Bot initialization failed")
            sys.exit(1)
        
        # Start trading
        if bot_manager.start():
            print("\n✅ Bot is now running with RR-based leverage!")
            print("📊 Check the logs for detailed information")
            print("💡 You can use Ctrl+C to stop the bot safely")
            
            # Show strategy explanation
            print("\n📋 Strategy Explanation:")
            print("🎯 The bot calculates Risk-Reward ratio for each signal")
            print("📊 Higher RR = Higher leverage + Larger position size")
            print("🛡️  Lower RR = Lower leverage + Smaller position size")
            print("⚖️  This balances risk and reward automatically")
            
            # Keep main thread alive
            try:
                while bot_manager.is_running:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\n🛑 Bot stopped by user")
        else:
            print("❌ Failed to start the bot")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n👋 Bot stopped by user")
    except Exception as e:
        logger.error(f"❌ Critical error: {str(e)}")
        sys.exit(1)
    finally:
        bot_manager.shutdown()

if __name__ == "__main__":
    main()