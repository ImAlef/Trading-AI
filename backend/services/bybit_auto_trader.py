import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging
from dataclasses import dataclass, asdict
import json
import os

from backend.services.bybit_api import BybitAPI, Position, OrderResult
from backend.services.signal_detector import SignalDetector, TradingSignal
from backend.services.email_sender import EmailSender

logger = logging.getLogger(__name__)

@dataclass
class ActiveTrade:
    """Active trade tracking"""
    signal_id: str
    symbol: str
    side: str
    entry_price: float
    quantity: float
    leverage: int
    target_price: float
    stop_loss: float
    confidence: float
    
    # Order IDs
    entry_order_id: Optional[str] = None
    tp_order_id: Optional[str] = None
    sl_order_id: Optional[str] = None
    
    # Status tracking
    status: str = 'PENDING'  # PENDING, FILLED, TP_HIT, SL_HIT, CLOSED
    pnl: float = 0.0
    created_at: datetime = None
    filled_at: Optional[datetime] = None
    closed_at: Optional[datetime] = None
    
    def to_dict(self):
        return asdict(self)

class BybitAutoTrader:
    """
    Automated Trading System for Bybit Futures
    """
    
    def __init__(self, api_key: str, secret_key: str, testnet: bool = True):
        self.bybit_api = BybitAPI(api_key, secret_key, testnet)
        self.signal_detector = SignalDetector()
        self.email_sender = EmailSender()
        
        # Trading state
        self.active_trades: Dict[str, ActiveTrade] = {}
        self.is_trading_active = False
        self.monitoring_thread = None
        
        # Settings with your RR-based leverage strategy
        self.position_size_percent = 0.10  # 10% of balance per trade
        self.max_concurrent_trades = 5
        self.emergency_stop_loss = 0.25  # 25% total loss = emergency stop
        
        # Create directories
        os.makedirs('data/trades', exist_ok=True)
        
        # Load previous trades
        self.load_trade_history()
        
        logger.info("🤖 Bybit AutoTrader initialized")
    
    def load_trade_history(self):
        """Load previous trade history"""
        try:
            history_file = 'data/trades/active_trades.json'
            if os.path.exists(history_file):
                with open(history_file, 'r') as f:
                    trades_data = json.load(f)
                    
                for trade_id, trade_data in trades_data.items():
                    # Convert datetime strings back to datetime objects
                    if trade_data.get('created_at'):
                        trade_data['created_at'] = datetime.fromisoformat(trade_data['created_at'])
                    if trade_data.get('filled_at'):
                        trade_data['filled_at'] = datetime.fromisoformat(trade_data['filled_at'])
                    if trade_data.get('closed_at'):
                        trade_data['closed_at'] = datetime.fromisoformat(trade_data['closed_at'])
                    
                    self.active_trades[trade_id] = ActiveTrade(**trade_data)
                
                logger.info(f"📚 Loaded {len(self.active_trades)} previous trades")
        except Exception as e:
            logger.error(f"Error loading trade history: {str(e)}")
    
    def save_trade_history(self):
        """Save current trades to file"""
        try:
            history_file = 'data/trades/active_trades.json'
            trades_data = {}
            
            for trade_id, trade in self.active_trades.items():
                trade_dict = trade.to_dict()
                # Convert datetime objects to strings for JSON serialization
                if trade_dict.get('created_at'):
                    trade_dict['created_at'] = trade_dict['created_at'].isoformat()
                if trade_dict.get('filled_at'):
                    trade_dict['filled_at'] = trade_dict['filled_at'].isoformat()
                if trade_dict.get('closed_at'):
                    trade_dict['closed_at'] = trade_dict['closed_at'].isoformat()
                
                trades_data[trade_id] = trade_dict
            
            with open(history_file, 'w') as f:
                json.dump(trades_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving trade history: {str(e)}")
    
    def start_trading(self) -> bool:
        """Start automated trading"""
        try:
            # Health check first
            health = self.bybit_api.health_check()
            if health['status'] != 'OK':
                logger.error(f"API Health Check Failed: {health['message']}")
                return False
            
            # Load signal detector model
            if not self.signal_detector.load_model():
                logger.error("Failed to load ML model")
                return False
            
            # Start monitoring thread
            self.is_trading_active = True
            self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self.monitoring_thread.start()
            
            logger.info("🚀 Bybit automated trading started!")
            logger.info(f"💰 Account Balance: ${health['balance']:.2f}")
            logger.info(f"📊 Max Concurrent Trades: {self.max_concurrent_trades}")
            logger.info(f"💪 Position Size: {self.position_size_percent*100}% per trade")
            logger.info(f"🧪 Mode: {'TESTNET' if self.bybit_api.testnet else 'LIVE'}")
            
            # Send notification
            if self.email_sender:
                mode = "TESTNET" if self.bybit_api.testnet else "LIVE"
                self.email_sender.send_system_alert(
                    'INFO', 
                    f'Bybit automated trading started ({mode}) with ${health["balance"]:.2f} balance'
                )
            
            return True
            
        except Exception as e:
            logger.error(f"Error starting trading: {str(e)}")
            return False
    
    def stop_trading(self):
        """Stop automated trading"""
        try:
            self.is_trading_active = False
            
            if self.monitoring_thread and self.monitoring_thread.is_alive():
                self.monitoring_thread.join(timeout=10)
            
            logger.info("🛑 Bybit automated trading stopped")
            
            # Send notification
            if self.email_sender:
                active_count = len([t for t in self.active_trades.values() if t.status in ['PENDING', 'FILLED']])
                self.email_sender.send_system_alert(
                    'INFO', 
                    f'Bybit automated trading stopped. {active_count} trades still active.'
                )
                
        except Exception as e:
            logger.error(f"Error stopping trading: {str(e)}")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        logger.info("🔄 Starting monitoring loop...")
        
        while self.is_trading_active:
            try:
                # Check for new signals
                self._check_for_new_signals()
                
                # Monitor active trades
                self._monitor_active_trades()
                
                # Check emergency conditions
                self._check_emergency_conditions()
                
                # Save state
                self.save_trade_history()
                
                # Wait before next iteration
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {str(e)}")
                time.sleep(60)  # Wait longer on error
    
    def _check_for_new_signals(self):
        """Check for new trading signals"""
        try:
            # Don't get new signals if we're at max capacity
            active_count = len([t for t in self.active_trades.values() if t.status in ['PENDING', 'FILLED']])
            if active_count >= self.max_concurrent_trades:
                return
            
            # Get signals from detector
            signals = self.signal_detector.scan_markets()
            
            for signal in signals:
                if self._should_execute_signal(signal):
                    self._execute_signal(signal)
                    
        except Exception as e:
            logger.error(f"Error checking for new signals: {str(e)}")
    
    def _should_execute_signal(self, signal: TradingSignal) -> bool:
        """Check if signal should be executed"""
        try:
            # Check if we already have a position for this symbol
            for trade in self.active_trades.values():
                if trade.symbol == signal.symbol.replace('USDT', 'USDT') and trade.status in ['PENDING', 'FILLED']:
                    logger.info(f"❌ Skipping {signal.symbol}: Already have active position")
                    return False
            
            # Check account balance
            balance = self.bybit_api.get_balance()
            if not balance or balance < 10:  # Minimum $10 balance
                logger.warning("❌ Insufficient balance for new trades")
                return False
            
            # All checks passed
            return True
            
        except Exception as e:
            logger.error(f"Error checking signal eligibility: {str(e)}")
            return False
    
    def _execute_signal(self, signal: TradingSignal):
        """Execute trading signal with RR-based leverage"""
        try:
            logger.info(f"🎯 Executing signal: {signal.symbol} @ {signal.confidence:.1%}")
            
            # Get account balance
            balance = self.bybit_api.get_balance()
            if not balance:
                logger.error("Failed to get account balance")
                return
            
            # Calculate RR-based leverage and position size
            leverage, position_size_pct = self._calculate_rr_leverage_and_size(signal)
            
            # Convert symbol format for Bybit (e.g., BTCUSDT)
            bybit_symbol = signal.symbol.replace('USDT', 'USDT')
            
            # Calculate quantity
            quantity = self.bybit_api.calculate_position_size(
                balance, signal.entry_price, leverage, position_size_pct
            )
            
            if quantity <= 0:
                logger.error("Invalid position size calculated")
                return
            
            # Create trade record
            trade_id = f"{signal.symbol}_{int(signal.created_at.timestamp())}"
            active_trade = ActiveTrade(
                signal_id=trade_id,
                symbol=bybit_symbol,
                side='Buy',  # Currently only long positions
                entry_price=signal.entry_price,
                quantity=quantity,
                leverage=leverage,
                target_price=signal.target_price,
                stop_loss=signal.stop_loss,
                confidence=signal.confidence,
                created_at=datetime.now()
            )
            
            # Place entry order
            current_price = self.bybit_api.get_current_price(bybit_symbol)
            if not current_price:
                logger.error(f"Failed to get current price for {bybit_symbol}")
                return
            
            # Choose order type based on entry price vs current price
            price_diff_pct = abs(current_price - signal.entry_price) / current_price
            
            if price_diff_pct < 0.002:  # Less than 0.2% difference
                # Market order for immediate entry
                order_result = self.bybit_api.place_market_order(
                    symbol=bybit_symbol,
                    side='Buy',
                    quantity=quantity,
                    leverage=leverage
                )
            else:
                # Limit order for better entry
                order_result = self.bybit_api.place_limit_order(
                    symbol=bybit_symbol,
                    side='Buy',
                    quantity=quantity,
                    price=signal.entry_price,
                    leverage=leverage
                )
            
            if order_result.success:
                active_trade.entry_order_id = order_result.order_id
                active_trade.status = 'PENDING'
                self.active_trades[trade_id] = active_trade
                
                logger.info(f"✅ Trade placed: {signal.symbol} - {leverage}x leverage - {position_size_pct*100:.0f}% position")
                
                # Send notification
                self._send_trade_notification(active_trade, "OPENED")
                
            else:
                logger.error(f"❌ Failed to place trade: {order_result.message}")
                
        except Exception as e:
            logger.error(f"Error executing signal: {str(e)}")
    
    def _calculate_rr_leverage_and_size(self, signal: TradingSignal) -> tuple[int, float]:
        """Calculate leverage and position size based on RR ratio (your strategy)"""
        try:
            # Calculate risk-reward ratio
            risk = abs(signal.entry_price - signal.stop_loss) / signal.entry_price
            reward = abs(signal.target_price - signal.entry_price) / signal.entry_price
            rr_ratio = reward / risk if risk > 0 else 1
            
            # Your RR-based strategy
            if rr_ratio > 3:
                leverage = 35
                position_size_pct = 0.30  # 30% of balance
            elif rr_ratio >= 2:
                leverage = 20
                position_size_pct = 0.20  # 20% of balance
            elif rr_ratio >= 1:
                leverage = 10
                position_size_pct = 0.10  # 10% of balance
            else:
                leverage = 5
                position_size_pct = 0.05  # 5% of balance
                
            # Cap leverage at Bybit's maximum (100x for most pairs)
            leverage = min(leverage, 100)
                
            logger.info(f"📊 RR={rr_ratio:.1f} → {leverage}x leverage, {position_size_pct*100:.0f}% position size")
            return leverage, position_size_pct
            
        except Exception as e:
            logger.error(f"Error calculating RR leverage: {str(e)}")
            return 10, 0.10  # Safe fallback
    
    def _monitor_active_trades(self):
        """Monitor active trades for TP/SL"""
        try:
            for trade_id, trade in list(self.active_trades.items()):
                if trade.status not in ['PENDING', 'FILLED']:
                    continue
                
                current_price = self.bybit_api.get_current_price(trade.symbol)
                if not current_price:
                    continue
                
                # Update PnL
                if trade.status == 'FILLED':
                    price_change = (current_price - trade.entry_price) / trade.entry_price
                    trade.pnl = price_change * trade.quantity * trade.entry_price * trade.leverage
                
                # Check Take Profit
                if current_price >= trade.target_price and trade.side == 'Buy':
                    self._close_trade(trade_id, 'TP_HIT', current_price)
                    continue
                
                # Check Stop Loss
                if current_price <= trade.stop_loss and trade.side == 'Buy':
                    self._close_trade(trade_id, 'SL_HIT', current_price)
                    continue
                
                # Check if pending order was filled
                if trade.status == 'PENDING':
                    self._check_order_status(trade_id, trade)
                    
        except Exception as e:
            logger.error(f"Error monitoring active trades: {str(e)}")
    
    def _check_order_status(self, trade_id: str, trade: ActiveTrade):
        """Check if pending order was filled"""
        try:
            # Get open orders for symbol
            open_orders = self.bybit_api.get_open_orders(trade.symbol)
            
            # Check if entry order still exists
            entry_order_exists = False
            for order in open_orders:
                if order.get('orderId') == trade.entry_order_id:
                    entry_order_exists = True
                    break
            
            # If entry order doesn't exist, it was likely filled
            if not entry_order_exists and trade.status == 'PENDING':
                trade.status = 'FILLED'
                trade.filled_at = datetime.now()
                
                # Place TP and SL orders
                self._place_tp_sl_orders(trade)
                
                logger.info(f"✅ Trade filled: {trade.symbol}")
                self._send_trade_notification(trade, "FILLED")
                
        except Exception as e:
            logger.error(f"Error checking order status: {str(e)}")
    
    def _place_tp_sl_orders(self, trade: ActiveTrade):
        """Place Take Profit and Stop Loss orders"""
        try:
            # Place Take Profit order
            tp_result = self.bybit_api.place_limit_order(
                symbol=trade.symbol,
                side='Sell',
                quantity=trade.quantity,
                price=trade.target_price,
                leverage=trade.leverage
            )
            
            if tp_result.success:
                trade.tp_order_id = tp_result.order_id
                logger.info(f"📈 TP order placed: {trade.symbol} @ ${trade.target_price}")
            
            # Place Stop Loss order
            sl_result = self.bybit_api.place_stop_order(
                symbol=trade.symbol,
                side='Sell',
                quantity=trade.quantity,
                stop_price=trade.stop_loss,
                leverage=trade.leverage
            )
            
            if sl_result.success:
                trade.sl_order_id = sl_result.order_id
                logger.info(f"🛑 SL order placed: {trade.symbol} @ ${trade.stop_loss}")
                
        except Exception as e:
            logger.error(f"Error placing TP/SL orders: {str(e)}")
    
    def _close_trade(self, trade_id: str, reason: str, exit_price: float):
        """Close an active trade"""
        try:
            trade = self.active_trades[trade_id]
            
            # Close position via API
            close_result = self.bybit_api.close_position(trade.symbol)
            
            if close_result.success:
                # Update trade status
                trade.status = reason
                trade.closed_at = datetime.now()
                
                # Calculate final PnL
                price_change = (exit_price - trade.entry_price) / trade.entry_price
                trade.pnl = price_change * trade.quantity * trade.entry_price * trade.leverage
                
                # Cancel remaining orders
                if trade.tp_order_id:
                    self.bybit_api.cancel_order(trade.symbol, trade.tp_order_id)
                if trade.sl_order_id:
                    self.bybit_api.cancel_order(trade.symbol, trade.sl_order_id)
                
                logger.info(f"🏁 Trade closed: {trade.symbol} - {reason} - P/L: ${trade.pnl:.2f}")
                
                # Send notification
                self._send_trade_notification(trade, reason)
                
                # Archive completed trade
                self._archive_trade(trade_id)
                
            else:
                logger.error(f"Failed to close position: {close_result.message}")
                
        except Exception as e:
            logger.error(f"Error closing trade: {str(e)}")
    
    def _archive_trade(self, trade_id: str):
        """Archive completed trade"""
        try:
            trade = self.active_trades[trade_id]
            
            # Save to completed trades file
            completed_file = 'data/trades/completed_trades.json'
            completed_trades = []
            
            if os.path.exists(completed_file):
                with open(completed_file, 'r') as f:
                    completed_trades = json.load(f)
            
            # Add current trade
            trade_dict = trade.to_dict()
            if trade_dict.get('created_at'):
                trade_dict['created_at'] = trade_dict['created_at'].isoformat()
            if trade_dict.get('filled_at'):
                trade_dict['filled_at'] = trade_dict['filled_at'].isoformat()
            if trade_dict.get('closed_at'):
                trade_dict['closed_at'] = trade_dict['closed_at'].isoformat()
            
            completed_trades.append(trade_dict)
            
            # Keep only last 100 trades
            if len(completed_trades) > 100:
                completed_trades = completed_trades[-100:]
            
            # Save to file
            with open(completed_file, 'w') as f:
                json.dump(completed_trades, f, indent=2)
            
            # Remove from active trades
            del self.active_trades[trade_id]
            
        except Exception as e:
            logger.error(f"Error archiving trade: {str(e)}")
    
    def _check_emergency_conditions(self):
        """Check for emergency stop conditions"""
        try:
            # Get current balance
            current_balance = self.bybit_api.get_balance()
            if not current_balance:
                return
            
            # Calculate total unrealized PnL
            total_pnl = sum(trade.pnl for trade in self.active_trades.values() if trade.status == 'FILLED')
            
            # Check if total loss exceeds emergency threshold
            total_loss_percent = abs(total_pnl) / current_balance if total_pnl < 0 else 0
            
            if total_loss_percent >= self.emergency_stop_loss:
                logger.critical(f"🚨 EMERGENCY STOP TRIGGERED! Total loss: {total_loss_percent*100:.1f}%")
                
                # Close all positions
                self._emergency_close_all()
                
                # Stop trading
                self.stop_trading()
                
                # Send critical alert
                if self.email_sender:
                    self.email_sender.send_system_alert(
                        'ERROR',
                        f'EMERGENCY STOP: Total loss {total_loss_percent*100:.1f}% exceeded {self.emergency_stop_loss*100}% limit. All positions closed.'
                    )
                    
        except Exception as e:
            logger.error(f"Error checking emergency conditions: {str(e)}")
    
    def _emergency_close_all(self):
        """Emergency close all positions"""
        try:
            logger.warning("🚨 Emergency closing all positions...")
            
            for trade_id, trade in list(self.active_trades.items()):
                if trade.status in ['PENDING', 'FILLED']:
                    current_price = self.bybit_api.get_current_price(trade.symbol)
                    if current_price:
                        self._close_trade(trade_id, 'EMERGENCY_STOP', current_price)
                        time.sleep(1)  # Small delay between closes
                        
        except Exception as e:
            logger.error(f"Error in emergency close: {str(e)}")
    
    def _send_trade_notification(self, trade: ActiveTrade, event: str):
        """Send email notification for trade events"""
        try:
            if not self.email_sender:
                return
            
            subject_map = {
                'OPENED': f"🎯 Bybit Trade Opened: {trade.symbol}",
                'FILLED': f"✅ Bybit Trade Filled: {trade.symbol}",
                'TP_HIT': f"📈 Take Profit Hit: {trade.symbol}",
                'SL_HIT': f"🛑 Stop Loss Hit: {trade.symbol}",
                'EMERGENCY_STOP': f"🚨 Emergency Stop: {trade.symbol}"
            }
            
            # Create message content
            message = f"""
            Bybit Trade Event: {event}
            Symbol: {trade.symbol}
            Side: {trade.side}
            Entry Price: ${trade.entry_price:.4f}
            Target Price: ${trade.target_price:.4f}
            Stop Loss: ${trade.stop_loss:.4f}
            Quantity: {trade.quantity:.6f}
            Leverage: {trade.leverage}x
            Confidence: {trade.confidence:.1%}
            Current P/L: ${trade.pnl:.2f}
            Status: {trade.status}
            """
            
            subject = subject_map.get(event, f"Bybit Trade Update: {trade.symbol}")
            self.email_sender.send_system_alert('INFO', message)
            
        except Exception as e:
            logger.error(f"Error sending trade notification: {str(e)}")
    
    def get_trading_summary(self) -> Dict:
        """Get current trading summary"""
        try:
            balance = self.bybit_api.get_balance()
            active_count = len([t for t in self.active_trades.values() if t.status in ['PENDING', 'FILLED']])
            total_pnl = sum(trade.pnl for trade in self.active_trades.values() if trade.status == 'FILLED')
            
            return {
                'is_active': self.is_trading_active,
                'balance': balance,
                'active_trades': active_count,
                'max_trades': self.max_concurrent_trades,
                'total_unrealized_pnl': total_pnl,
                'position_size_percent': self.position_size_percent * 100,
                'emergency_threshold': self.emergency_stop_loss * 100,
                'testnet': self.bybit_api.testnet,
                'trades_summary': [
                    {
                        'symbol': trade.symbol,
                        'status': trade.status,
                        'pnl': trade.pnl,
                        'leverage': trade.leverage,
                        'confidence': trade.confidence
                    }
                    for trade in self.active_trades.values()
                    if trade.status in ['PENDING', 'FILLED']
                ]
            }
            
        except Exception as e:
            logger.error(f"Error getting trading summary: {str(e)}")
            return {'error': str(e)}
    
    def force_close_trade(self, symbol: str) -> bool:
        """Manually close a specific trade"""
        try:
            for trade_id, trade in self.active_trades.items():
                if trade.symbol == symbol and trade.status in ['PENDING', 'FILLED']:
                    current_price = self.bybit_api.get_current_price(symbol)
                    if current_price:
                        self._close_trade(trade_id, 'MANUAL_CLOSE', current_price)
                        return True
            return False
            
        except Exception as e:
            logger.error(f"Error force closing trade: {str(e)}")
            return False
    
    def get_performance_stats(self, days: int = 30) -> Dict:
        """Get performance statistics"""
        try:
            # Load completed trades
            completed_file = 'data/trades/completed_trades.json'
            if not os.path.exists(completed_file):
                return {'message': 'No completed trades found'}
            
            with open(completed_file, 'r') as f:
                completed_trades = json.load(f)
            
            # Filter by date range
            cutoff_date = datetime.now() - timedelta(days=days)
            recent_trades = []
            
            for trade_data in completed_trades:
                if trade_data.get('closed_at'):
                    closed_at = datetime.fromisoformat(trade_data['closed_at'])
                    if closed_at >= cutoff_date:
                        recent_trades.append(trade_data)
            
            if not recent_trades:
                return {'message': f'No trades in last {days} days'}
            
            # Calculate statistics
            total_trades = len(recent_trades)
            winning_trades = len([t for t in recent_trades if t['pnl'] > 0])
            losing_trades = len([t for t in recent_trades if t['pnl'] < 0])
            
            total_pnl = sum(t['pnl'] for t in recent_trades)
            winning_pnl = sum(t['pnl'] for t in recent_trades if t['pnl'] > 0)
            losing_pnl = sum(t['pnl'] for t in recent_trades if t['pnl'] < 0)
            
            win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
            avg_win = winning_pnl / winning_trades if winning_trades > 0 else 0
            avg_loss = losing_pnl / losing_trades if losing_trades > 0 else 0
            profit_factor = abs(winning_pnl / losing_pnl) if losing_pnl != 0 else float('inf')
            
            return {
                'period_days': days,
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'win_rate': win_rate,
                'total_pnl': total_pnl,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'profit_factor': profit_factor,
                'best_trade': max(recent_trades, key=lambda x: x['pnl'])['pnl'],
                'worst_trade': min(recent_trades, key=lambda x: x['pnl'])['pnl']
            }
            
        except Exception as e:
            logger.error(f"Error calculating performance stats: {str(e)}")
            return {'error': str(e)}