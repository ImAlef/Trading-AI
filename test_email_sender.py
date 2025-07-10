#!/usr/bin/env python3
"""
Test script for Email Sender
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backend.services.email_sender import EmailSender
from config import config
import logging
from datetime import datetime, timedelta

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_email_sender():
    """Test the email sender functionality"""
    
    print("📧 Testing Email Sender...")
    print("="*50)
    
    try:
        # Step 1: Initialize email sender
        print("\n🔧 Step 1: Initializing Email Sender...")
        email_sender = EmailSender()
        print("✅ Email sender initialized")
        
        # Step 2: Validate configuration
        print("\n⚙️ Step 2: Validating email configuration...")
        config_valid = email_sender.validate_config()
        
        if config_valid:
            print("✅ Email configuration is valid")
            print(f"   SMTP Server: {email_sender.smtp_server}")
            print(f"   Port: {email_sender.smtp_port}")
            print(f"   From: {email_sender.from_email}")
            print(f"   To: {email_sender.to_email}")
        else:
            print("❌ Email configuration is invalid")
            print("   Please check your .env file and configure:")
            print("   - SMTP_SERVER")
            print("   - SMTP_PORT") 
            print("   - SMTP_USERNAME")
            print("   - SMTP_PASSWORD")
            print("   - FROM_EMAIL")
            print("   - TO_EMAIL")
            return False
        
        # Step 3: Test email connection
        print("\n🔗 Step 3: Testing email connection...")
        print("   (This will send a test email)")
        
        user_input = input("   Continue with test email? (y/N): ").lower().strip()
        
        if user_input != 'y':
            print("   Skipping email connection test")
            return True
        
        connection_success = email_sender.test_email_connection()
        
        if connection_success:
            print("✅ Email connection test successful")
            print("   Check your email for the test message")
        else:
            print("❌ Email connection test failed")
            print("   Common issues:")
            print("   - Wrong SMTP server or port")
            print("   - Invalid credentials")
            print("   - App password not enabled (for Gmail)")
            print("   - Firewall blocking SMTP")
            return False
        
        # Step 4: Test signal email
        print("\n📈 Step 4: Testing signal email...")
        
        # Create sample signal data
        sample_signal = {
            'symbol': 'BTCUSDT',
            'signal_type': 'BUY',
            'entry_price': 43250.00,
            'target_price': 44115.00,
            'stop_loss': 42817.50,
            'confidence': 0.675,
            'created_at': datetime.now().isoformat(),
            'expires_at': (datetime.now() + timedelta(hours=4)).isoformat(),
            'rsi': 45.2,
            'macd': 0.0123,
            'macd_signal': 0.0089,
            'bb_upper': 44500.00,
            'bb_lower': 42000.00,
            'volume_ratio': 1.35,
            'model_version': 'v20250709_143510',
            'features_used': 66,
            'timeframe': '1h'
        }
        
        user_input = input("   Send sample signal email? (y/N): ").lower().strip()
        
        if user_input == 'y':
            signal_success = email_sender.send_signal_email(sample_signal)
            
            if signal_success:
                print("✅ Signal email sent successfully")
                print("   Check your email for the signal notification")
            else:
                print("❌ Failed to send signal email")
                return False
        else:
            print("   Skipping signal email test")
        
        # Step 5: Test summary email
        print("\n📊 Step 5: Testing summary email...")
        
        # Create sample summary data
        sample_signals = [
            {
                'symbol': 'BTCUSDT',
                'confidence': 0.675,
                'entry_price': 43250.00,
                'target_price': 44115.00,
                'created_at': datetime.now().isoformat()
            },
            {
                'symbol': 'ETHUSDT',
                'confidence': 0.623,
                'entry_price': 2580.00,
                'target_price': 2631.60,
                'created_at': (datetime.now() - timedelta(hours=2)).isoformat()
            }
        ]
        
        sample_summary = {
            'pairs_scanned': 10,
            'success_rate': 67.5,
            'avg_confidence': 64.9
        }
        
        user_input = input("   Send sample summary email? (y/N): ").lower().strip()
        
        if user_input == 'y':
            summary_success = email_sender.send_summary_email(sample_signals, sample_summary)
            
            if summary_success:
                print("✅ Summary email sent successfully")
                print("   Check your email for the daily summary")
            else:
                print("❌ Failed to send summary email")
                return False
        else:
            print("   Skipping summary email test")
        
        # Step 6: Test system alert
        print("\n🚨 Step 6: Testing system alert...")
        
        user_input = input("   Send sample system alert? (y/N): ").lower().strip()
        
        if user_input == 'y':
            alert_success = email_sender.send_system_alert(
                'INFO',
                'This is a test system alert to verify that the alert system is working properly. The bot is running normally and monitoring the markets.'
            )
            
            if alert_success:
                print("✅ System alert sent successfully")
                print("   Check your email for the system alert")
            else:
                print("❌ Failed to send system alert")
                return False
        else:
            print("   Skipping system alert test")
        
        print("\n" + "="*50)
        print("🎉 All email tests completed successfully!")
        print("✅ Email notification system is ready!")
        
        return True
        
    except Exception as e:
        logger.error(f"Email sender test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def show_email_setup_guide():
    """Show email setup guide"""
    print("\n📧 Email Setup Guide")
    print("="*50)
    print("To use email notifications, you need to configure these settings in your .env file:")
    print()
    print("For Gmail:")
    print("1. Enable 2-factor authentication")
    print("2. Generate an app password: https://support.google.com/accounts/answer/185833")
    print("3. Use these settings:")
    print("   SMTP_SERVER=smtp.gmail.com")
    print("   SMTP_PORT=587")
    print("   SMTP_USERNAME=your_email@gmail.com")
    print("   SMTP_PASSWORD=your_app_password")
    print("   FROM_EMAIL=your_email@gmail.com")
    print("   TO_EMAIL=recipient@gmail.com")
    print()
    print("For Outlook/Hotmail:")
    print("   SMTP_SERVER=smtp-mail.outlook.com")
    print("   SMTP_PORT=587")
    print("   SMTP_USERNAME=your_email@outlook.com")
    print("   SMTP_PASSWORD=your_password")
    print()
    print("For Yahoo:")
    print("   SMTP_SERVER=smtp.mail.yahoo.com")
    print("   SMTP_PORT=587")
    print("   SMTP_USERNAME=your_email@yahoo.com")
    print("   SMTP_PASSWORD=your_app_password")

if __name__ == "__main__":
    print("🤖 Crypto Signal Bot - Email Sender Test")
    print("="*50)
    
    print(f"📋 Current Configuration:")
    print(f"   SMTP Server: {config.SMTP_SERVER}")
    print(f"   Port: {config.SMTP_PORT}")
    print(f"   From Email: {config.FROM_EMAIL}")
    print(f"   To Email: {config.TO_EMAIL}")
    
    # Check if email is configured
    if not config.validate_config():
        print("\n⚠️  Email configuration is incomplete!")
        show_email_setup_guide()
        
        user_input = input("\n📧 Have you configured your email settings? (y/N): ").lower().strip()
        if user_input != 'y':
            print("Please configure your email settings in .env file and try again.")
            sys.exit(1)
    
    try:
        success = test_email_sender()
        
        if success:
            print("\n🎉 Email sender test completed successfully!")
            print("✅ Email notification system is ready!")
            print("\n🚀 Next steps:")
            print("   1. Test automated signal scanning")
            print("   2. Build web dashboard")
            print("   3. Deploy the system")
        else:
            print("\n❌ Email sender test failed")
            print("Please check your email configuration and try again.")
            
    except KeyboardInterrupt:
        print("\n⚠️  Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()