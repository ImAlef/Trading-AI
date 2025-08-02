import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import ssl
from datetime import datetime
from typing import List, Dict, Optional
import logging
from config import config

logger = logging.getLogger(__name__)

class EmailSender:
    """
    Email notification service for trading signals
    """
    
    def __init__(self):
        self.smtp_server = config.SMTP_SERVER
        self.smtp_port = config.SMTP_PORT
        self.username = config.SMTP_USERNAME
        self.password = config.SMTP_PASSWORD
        self.from_email = config.FROM_EMAIL
        self.to_email = config.TO_EMAIL
        
    def validate_config(self) -> bool:
        """Validate email configuration"""
        required_fields = [
            self.smtp_server, self.smtp_port, self.username, 
            self.password, self.from_email, self.to_email
        ]
        
        if not all(required_fields):
            logger.error("Email configuration is incomplete")
            return False
        
        return True
    
    def send_signal_email(self, signal_data: Dict) -> bool:
        """
        Send email notification for a trading signal
        """
        try:
            if not self.validate_config():
                return False
            
            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.from_email
            msg['To'] = self.to_email
            msg['Subject'] = f"🚀 CRYPTO SIGNAL: {signal_data['symbol']} - {signal_data['confidence']*100:.1f}% Confidence"
            
            # Create email body
            body = self._create_signal_email_body(signal_data)
            msg.attach(MIMEText(body, 'html'))
            
            # Send email
            return self._send_email(msg)
            
        except Exception as e:
            logger.error(f"Error sending signal email: {str(e)}")
            return False
    
    def send_summary_email(self, signals: List[Dict], summary_data: Dict) -> bool:
        """
        Send daily summary email
        """
        try:
            if not self.validate_config():
                return False
            
            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.from_email
            msg['To'] = self.to_email
            msg['Subject'] = f"📊 Daily Crypto Signals Summary - {len(signals)} Signals"
            
            # Create email body
            body = self._create_summary_email_body(signals, summary_data)
            msg.attach(MIMEText(body, 'html'))
            
            # Send email
            return self._send_email(msg)
            
        except Exception as e:
            logger.error(f"Error sending summary email: {str(e)}")
            return False
    
    def send_system_alert(self, alert_type: str, message: str) -> bool:
        """
        Send system alert email
        """
        try:
            if not self.validate_config():
                return False
            
            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.from_email
            msg['To'] = self.to_email
            msg['Subject'] = f"🚨 System Alert: {alert_type}"
            
            # Create email body
            body = self._create_alert_email_body(alert_type, message)
            msg.attach(MIMEText(body, 'html'))
            
            # Send email
            return self._send_email(msg)
            
        except Exception as e:
            logger.error(f"Error sending system alert: {str(e)}")
            return False
    
    def _send_email(self, msg: MIMEMultipart) -> bool:
        """
        Send email using SMTP
        """
        try:
            # Create secure connection
            context = ssl.create_default_context()
            
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls(context=context)
                server.login(self.username, self.password)
                
                # Send email
                text = msg.as_string()
                server.sendmail(self.from_email, self.to_email, text)
                
            logger.info(f"Email sent successfully to {self.to_email}")
            return True
            
        except Exception as e:
            logger.error(f"Error sending email: {str(e)}")
            return False
    
    def _create_signal_email_body(self, signal_data: Dict) -> str:
        """
        Create HTML email body for trading signal with support for LONG and SHORT
        """
        # Calculate profit/loss percentages based on signal type
        if signal_data['signal_type'] == 'BUY':
            profit_pct = ((signal_data['target_price'] - signal_data['entry_price']) / signal_data['entry_price']) * 100
            loss_pct = ((signal_data['entry_price'] - signal_data['stop_loss']) / signal_data['entry_price']) * 100
            signal_emoji = "📈"
            signal_color = "#2E7D32"  # Green
            direction = "LONG"
        else:  # SELL (SHORT)
            profit_pct = ((signal_data['entry_price'] - signal_data['target_price']) / signal_data['entry_price']) * 100
            loss_pct = ((signal_data['stop_loss'] - signal_data['entry_price']) / signal_data['entry_price']) * 100
            signal_emoji = "📉"
            signal_color = "#D32F2F"  # Red
            direction = "SHORT"
        
        risk_reward_ratio = profit_pct / loss_pct if loss_pct > 0 else 0
        
        # Format timestamps
        created_time = datetime.fromisoformat(signal_data['created_at']).strftime('%Y-%m-%d %H:%M:%S')
        expires_time = datetime.fromisoformat(signal_data['expires_at']).strftime('%Y-%m-%d %H:%M:%S')
        
        # Create email body with high precision pricing and SHORT support
        body = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #f4f4f4; }}
                .container {{ max-width: 600px; margin: 0 auto; background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }}
                .header {{ text-align: center; background-color: {signal_color}; color: white; padding: 20px; border-radius: 10px; margin-bottom: 20px; }}
                .signal-info {{ background-color: #f8f9fa; padding: 15px; border-radius: 8px; margin-bottom: 20px; }}
                .price-info {{ display: flex; justify-content: space-between; margin-bottom: 15px; }}
                .price-box {{ text-align: center; padding: 10px; border-radius: 5px; flex: 1; margin: 0 5px; }}
                .entry {{ background-color: #e3f2fd; }}
                .target {{ background-color: #e8f5e8; }}
                .stop {{ background-color: #ffebee; }}
                .technical {{ background-color: #f3e5f5; padding: 15px; border-radius: 8px; margin-bottom: 20px; }}
                .footer {{ text-align: center; color: #666; font-size: 12px; margin-top: 20px; }}
                .warning {{ background-color: #fff3cd; border: 1px solid #ffeaa7; padding: 10px; border-radius: 5px; margin-top: 20px; }}
                .precision {{ font-family: 'Courier New', monospace; font-weight: bold; }}
                .direction-badge {{ display: inline-block; padding: 5px 15px; border-radius: 20px; color: white; background-color: {signal_color}; font-weight: bold; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>{signal_emoji} CRYPTO TRADING SIGNAL</h1>
                    <h2>{signal_data['symbol']}</h2>
                    <span class="direction-badge">{direction}</span>
                    <p>Confidence: {signal_data['confidence']*100:.1f}%</p>
                </div>
                
                <div class="signal-info">
                    <h3>📊 Signal Details</h3>
                    <p><strong>Action:</strong> {signal_data['signal_type']} ({direction})</p>
                    <p><strong>Timeframe:</strong> {signal_data['timeframe']}</p>
                    <p><strong>Created:</strong> {created_time}</p>
                    <p><strong>Expires:</strong> {expires_time}</p>
                    <p><strong>Risk/Reward:</strong> <span class="precision">1:{risk_reward_ratio:.2f}</span></p>
                </div>
                
                <div class="price-info">
                    <div class="price-box entry">
                        <h4>🎯 Entry Price</h4>
                        <p class="precision"><strong>${signal_data['entry_price']:.6f}</strong></p>
                    </div>
                    <div class="price-box target">
                        <h4>{'📈' if signal_data['signal_type'] == 'BUY' else '📉'} Target Price</h4>
                        <p class="precision"><strong>${signal_data['target_price']:.6f}</strong></p>
                        <p style="color: green;">+{profit_pct:.3f}%</p>
                    </div>
                    <div class="price-box stop">
                        <h4>🛑 Stop Loss</h4>
                        <p class="precision"><strong>${signal_data['stop_loss']:.6f}</strong></p>
                        <p style="color: red;">-{loss_pct:.3f}%</p>
                    </div>
                </div>
                
                <div class="technical">
                    <h3>🔍 Technical Analysis</h3>
                    <p><strong>RSI:</strong> {signal_data['rsi']:.2f} {'(Oversold)' if signal_data['rsi'] < 30 else '(Overbought)' if signal_data['rsi'] > 70 else '(Neutral)'}</p>
                    <p><strong>MACD:</strong> {signal_data['macd']:.6f}</p>
                    <p><strong>MACD Signal:</strong> {signal_data['macd_signal']:.6f}</p>
                    <p><strong>MACD Trend:</strong> {'Bullish' if signal_data['macd'] > signal_data['macd_signal'] else 'Bearish'}</p>
                    <p><strong>Bollinger Upper:</strong> <span class="precision">${signal_data['bb_upper']:.6f}</span></p>
                    <p><strong>Bollinger Lower:</strong> <span class="precision">${signal_data['bb_lower']:.6f}</span></p>
                    <p><strong>Volume Ratio:</strong> {signal_data['volume_ratio']:.3f} {'(High Volume)' if signal_data['volume_ratio'] > 1.5 else '(Normal Volume)'}</p>
                </div>
                
                <div class="warning">
                    <h4>⚠️ Risk Warning</h4>
                    <p>This is an automated {direction} signal based on technical analysis. Please:</p>
                    <ul>
                        <li>Do your own research before trading</li>
                        <li>Only invest what you can afford to lose</li>
                        <li>Use proper risk management</li>
                        <li>Consider market conditions</li>
                        <li>{'Be aware that SHORT positions have unlimited risk potential' if signal_data['signal_type'] == 'SELL' else 'Monitor your position closely'}</li>
                    </ul>
                </div>
                
                <div class="footer">
                    <p>Generated by Crypto Signal Bot v{signal_data['model_version']}</p>
                    <p>Model used {signal_data['features_used']} features for prediction</p>
                    <p>Signal Type: {direction} | Confidence: {signal_data['confidence']*100:.1f}%</p>
                    <p>This email was sent automatically. Do not reply.</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        return body
    
    def _create_summary_email_body(self, signals: List[Dict], summary_data: Dict) -> str:
        """
        Create HTML email body for daily summary with LONG/SHORT support
        """
        date_str = datetime.now().strftime('%Y-%m-%d')
        
        # Count LONG vs SHORT signals
        long_signals = [s for s in signals if s['signal_type'] == 'BUY']
        short_signals = [s for s in signals if s['signal_type'] == 'SELL']
        
        # Create signals table
        signals_html = ""
        if signals:
            for signal in signals:
                if signal['signal_type'] == 'BUY':
                    profit_pct = ((signal['target_price'] - signal['entry_price']) / signal['entry_price']) * 100
                    signal_emoji = "📈"
                    direction = "LONG"
                else:  # SELL (SHORT)
                    profit_pct = ((signal['entry_price'] - signal['target_price']) / signal['entry_price']) * 100
                    signal_emoji = "📉"
                    direction = "SHORT"
                
                created_time = datetime.fromisoformat(signal['created_at']).strftime('%H:%M')
                
                signals_html += f"""
                <tr>
                    <td>{signal_emoji} {signal['symbol']}</td>
                    <td>{direction}</td>
                    <td>{signal['confidence']*100:.1f}%</td>
                    <td>${signal['entry_price']:.6f}</td>
                    <td>${signal['target_price']:.6f}</td>
                    <td style="color: green;">+{profit_pct:.2f}%</td>
                    <td>{created_time}</td>
                </tr>
                """
        else:
            signals_html = '<tr><td colspan="7" style="text-align: center;">No signals generated today</td></tr>'
        
        body = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #f4f4f4; }}
                .container {{ max-width: 800px; margin: 0 auto; background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }}
                .header {{ text-align: center; background-color: #1976D2; color: white; padding: 20px; border-radius: 10px; margin-bottom: 20px; }}
                .stats {{ display: flex; justify-content: space-around; margin-bottom: 20px; }}
                .stat-box {{ text-align: center; padding: 15px; background-color: #f8f9fa; border-radius: 8px; }}
                .long-box {{ border-left: 4px solid #2E7D32; }}
                .short-box {{ border-left: 4px solid #D32F2F; }}
                table {{ width: 100%; border-collapse: collapse; margin-bottom: 20px; }}
                th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #f8f9fa; }}
                .footer {{ text-align: center; color: #666; font-size: 12px; margin-top: 20px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>📊 Daily Crypto Signals Summary</h1>
                    <h2>{date_str}</h2>
                </div>
                
                <div class="stats">
                    <div class="stat-box">
                        <h3>{len(signals)}</h3>
                        <p>Total Signals</p>
                    </div>
                    <div class="stat-box long-box">
                        <h3>📈 {len(long_signals)}</h3>
                        <p>LONG Signals</p>
                    </div>
                    <div class="stat-box short-box">
                        <h3>📉 {len(short_signals)}</h3>
                        <p>SHORT Signals</p>
                    </div>
                    <div class="stat-box">
                        <h3>{summary_data.get('pairs_scanned', 0)}</h3>
                        <p>Pairs Scanned</p>
                    </div>
                    <div class="stat-box">
                        <h3>{summary_data.get('success_rate', 0):.1f}%</h3>
                        <p>Success Rate</p>
                    </div>
                    <div class="stat-box">
                        <h3>{summary_data.get('avg_confidence', 0):.1f}%</h3>
                        <p>Avg Confidence</p>
                    </div>
                </div>
                
                <h3>📈📉 Today's Signals</h3>
                <table>
                    <thead>
                        <tr>
                            <th>Symbol</th>
                            <th>Direction</th>
                            <th>Confidence</th>
                            <th>Entry Price</th>
                            <th>Target Price</th>
                            <th>Profit %</th>
                            <th>Time</th>
                        </tr>
                    </thead>
                    <tbody>
                        {signals_html}
                    </tbody>
                </table>
                
                <div class="footer">
                    <p>Generated by Crypto Signal Bot</p>
                    <p>LONG Signals: {len(long_signals)} | SHORT Signals: {len(short_signals)} | Total: {len(signals)}</p>
                    <p>This email was sent automatically. Do not reply.</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        return body
    
    def _create_alert_email_body(self, alert_type: str, message: str) -> str:
        """
        Create HTML email body for system alerts
        """
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Choose color based on alert type
        color_map = {
            'ERROR': '#f44336',
            'WARNING': '#ff9800',
            'INFO': '#2196f3',
            'SUCCESS': '#4caf50'
        }
        
        color = color_map.get(alert_type, '#2196f3')
        
        body = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #f4f4f4; }}
                .container {{ max-width: 600px; margin: 0 auto; background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }}
                .header {{ text-align: center; background-color: {color}; color: white; padding: 20px; border-radius: 10px; margin-bottom: 20px; }}
                .alert-info {{ background-color: #f8f9fa; padding: 15px; border-radius: 8px; margin-bottom: 20px; }}
                .footer {{ text-align: center; color: #666; font-size: 12px; margin-top: 20px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🚨 System Alert</h1>
                    <h2>{alert_type}</h2>
                </div>
                
                <div class="alert-info">
                    <h3>📋 Alert Details</h3>
                    <p><strong>Type:</strong> {alert_type}</p>
                    <p><strong>Time:</strong> {timestamp}</p>
                    <p><strong>Message:</strong></p>
                    <p>{message}</p>
                </div>
                
                <div class="footer">
                    <p>Generated by Crypto Signal Bot</p>
                    <p>This email was sent automatically. Do not reply.</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        return body
    
    def test_email_connection(self) -> bool:
        """
        Test email connection and send test email
        """
        try:
            if not self.validate_config():
                return False
            
            # Create test message
            msg = MIMEMultipart()
            msg['From'] = self.from_email
            msg['To'] = self.to_email
            msg['Subject'] = "✅ Crypto Signal Bot - Email Test"
            
            body = """
            <html>
            <body>
                <h2>✅ Email Test Successful!</h2>
                <p>This is a test email from your Crypto Signal Bot.</p>
                <p>If you receive this email, the email notification system is working correctly.</p>
                <p><strong>Configuration:</strong></p>
                <ul>
                    <li>SMTP Server: {}</li>
                    <li>Port: {}</li>
                    <li>From: {}</li>
                    <li>To: {}</li>
                </ul>
                <p>The bot is now ready to send you trading signals!</p>
            </body>
            </html>
            """.format(self.smtp_server, self.smtp_port, self.from_email, self.to_email)
            
            msg.attach(MIMEText(body, 'html'))
            
            # Send test email
            success = self._send_email(msg)
            
            if success:
                logger.info("Test email sent successfully")
            else:
                logger.error("Failed to send test email")
            
            return success
            
        except Exception as e:
            logger.error(f"Error testing email connection: {str(e)}")
            return False