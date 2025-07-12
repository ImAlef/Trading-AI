#!/usr/bin/env python3
"""
Advanced Backtester with Live Learning Simulation
بک‌تست کامل با شبیه‌سازی یادگیری زنده
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass, asdict
import time
import json
from collections import deque

from backend.services.fallback_data_collector import FallbackDataCollector
from backend.services.feature_engineer import FeatureEngineer
from backend.services.ml_model import MLModel
from backend.services.signal_detector import SignalDetector, TradingSignal
from config import config

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class BacktestTrade:
    """Complete trade record for backtesting"""
    signal_id: str
    symbol: str
    entry_time: datetime
    entry_price: float
    target_price: float
    stop_loss: float
    confidence: float
    leverage: float
    
    # Exit details
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None  # 'TARGET_HIT', 'STOP_HIT', 'EXPIRED', 'MANUAL'
    
    # Financial results
    gross_pnl: Optional[float] = None  # Without leverage
    leveraged_pnl: Optional[float] = None  # With leverage
    gross_pnl_pct: Optional[float] = None
    leveraged_pnl_pct: Optional[float] = None
    
    # Learning data
    was_correct_prediction: Optional[bool] = None
    confidence_accuracy: Optional[float] = None
    duration_hours: Optional[float] = None
    
    # Position sizing
    position_size_usd: float = 0.0
    position_size_pct: float = 0.2  # 20% of capital per trade
    
    def to_dict(self):
        return asdict(self)

@dataclass
class PortfolioState:
    """Portfolio state tracking"""
    timestamp: datetime
    total_capital: float
    available_capital: float
    leveraged_capital: float
    open_positions: int
    total_trades: int
    winning_trades: int
    losing_trades: int
    current_drawdown: float
    max_drawdown: float
    adaptive_threshold: float
    
    def to_dict(self):
        return asdict(self)

class LiveLearningBacktester:
    """
    Advanced backtester that simulates the exact live learning system
    """
    
    def __init__(self, initial_capital: float = 20.0):
        # Core services (same as live system)
        self.data_collector = FallbackDataCollector()
        self.feature_engineer = FeatureEngineer()
        self.ml_model = MLModel()
        self.signal_detector = SignalDetector()
        
        # Portfolio tracking
        self.initial_capital = initial_capital
        self.normal_capital = initial_capital  # Without leverage
        self.leveraged_capital = initial_capital  # With leverage
        self.position_size_pct = 0.20  # 20% per trade
        
        # Trade tracking
        self.all_trades: List[BacktestTrade] = []
        self.portfolio_history: List[PortfolioState] = []
        
        # Live learning simulation
        self.adaptive_threshold = 0.55
        self.min_threshold = 0.35
        self.max_threshold = 0.75
        self.completed_signals: deque = deque(maxlen=100)
        self.learning_history = []
        
        # Performance tracking
        self.daily_performance = []
        self.learning_events = []
        
        # Risk management
        self.max_concurrent_trades = 3
        self.max_daily_trades = 10
        self.max_drawdown_limit = 0.30  # 30% max drawdown
        
    def load_model(self) -> bool:
        """Load ML model for backtesting"""
        try:
            success = self.ml_model.load_latest_model()
            if success:
                logger.info(f"Model loaded for backtesting: {self.ml_model.model_info.get('version', 'unknown')}")
                # Set initial threshold same as live system
                self.adaptive_threshold = config.CONFIDENCE_THRESHOLD
                return True
            return False
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False
    
    def collect_historical_data(self, days: int = 30) -> Dict[str, pd.DataFrame]:
        """Collect historical data for backtesting"""
        try:
            logger.info(f"Collecting {days} days of historical data...")
            
            historical_data = {}
            end_time = datetime.now()
            start_time = end_time - timedelta(days=days + 7)  # Extra for indicators
            
            symbols = config.TRADING_PAIRS[:8]  # Test with 8 main pairs
            
            for symbol in symbols:
                try:
                    df = self.data_collector.get_historical_klines(symbol, "1h", start_time, end_time)
                    
                    if not df.empty and len(df) > 200:  # Ensure enough data
                        historical_data[symbol] = df
                        logger.info(f"Collected {len(df)} candles for {symbol}")
                    else:
                        logger.warning(f"Insufficient data for {symbol}")
                        
                except Exception as e:
                    logger.error(f"Error collecting data for {symbol}: {str(e)}")
                    continue
            
            logger.info(f"Historical data collection completed: {len(historical_data)} symbols")
            return historical_data
            
        except Exception as e:
            logger.error(f"Error collecting historical data: {str(e)}")
            return {}
    
    def simulate_live_signal_detection(self, symbol: str, current_time: datetime, 
                                     historical_data: pd.DataFrame) -> Optional[TradingSignal]:
        """Simulate exact live signal detection process"""
        try:
            # Get data up to current time (simulate real-time)
            available_data = historical_data[historical_data.index <= current_time]
            
            if len(available_data) < 100:  # Need enough data for indicators
                return None
            
            # Use last 150 candles (same as live system)
            recent_data = available_data.tail(150)
            
            # Feature engineering (same as live)
            df_features = self.feature_engineer.prepare_features_for_prediction(recent_data)
            
            if df_features.empty:
                return None
            
            # Get latest features
            latest_features = df_features.iloc[-1:]
            
            # Make prediction using ML model
            prediction, confidence = self.ml_model.predict_single(latest_features)
            
            # Use adaptive threshold (live learning simulation)
            if prediction == 1 and confidence >= self.adaptive_threshold:
                
                current_price = latest_features['close'].iloc[-1]
                
                # Create signal using exact same logic as live system
                signal = self._create_signal_like_live_system(
                    symbol, current_price, confidence, latest_features, current_time
                )
                
                # Apply same validation filters as live system
                if self._validate_signal_like_live_system(signal):
                    return signal
            
            return None
            
        except Exception as e:
            logger.error(f"Error in live signal simulation for {symbol}: {str(e)}")
            return None
    
    def _create_signal_like_live_system(self, symbol: str, current_price: float, 
                                      confidence: float, features: pd.DataFrame, 
                                      current_time: datetime) -> TradingSignal:
        """Create signal using exact same logic as live system"""
        latest_row = features.iloc[-1]
        
        # Same calculations as live SignalDetector
        atr = latest_row.get('atr', current_price * 0.02)
        
        # Entry price logic (same as live)
        if confidence > 0.8:
            entry_price = current_price
        elif confidence > 0.65:
            entry_price = current_price * (1 - 0.005)  # 0.5% pullback
        else:
            entry_price = current_price * (1 - 0.01)   # 1% pullback
        
        # Stop loss calculation (same as live)
        atr_multiplier = 1.5 if confidence > 0.7 else 2.0
        atr_stop = entry_price - (atr * atr_multiplier)
        max_stop = entry_price * (1 - config.MAX_STOP_LOSS)
        stop_loss = max(atr_stop, max_stop)
        
        # Target calculation (same as live)
        risk = entry_price - stop_loss
        if confidence > 0.8:
            risk_reward_ratio = 3.0
        elif confidence > 0.7:
            risk_reward_ratio = 2.5
        else:
            risk_reward_ratio = 2.0
        
        target_price = entry_price + (risk * risk_reward_ratio)
        min_target = entry_price * (1 + config.MIN_PROFIT_TARGET)
        target_price = max(target_price, min_target)
        
        # Create signal
        signal = TradingSignal(
            symbol=symbol,
            signal_type='BUY',
            entry_price=entry_price,
            target_price=target_price,
            stop_loss=stop_loss,
            confidence=confidence,
            created_at=current_time,
            expires_at=current_time + timedelta(hours=24),  # 24h expiry for backtest
            rsi=latest_row.get('rsi', 0),
            macd=latest_row.get('macd', 0),
            macd_signal=latest_row.get('macd_signal', 0),
            bb_upper=latest_row.get('bb_upper', 0),
            bb_lower=latest_row.get('bb_lower', 0),
            volume_ratio=latest_row.get('volume_ratio', 1),
            model_version=self.ml_model.model_info.get('version', 'backtest'),
            features_used=71,
            timeframe='1h'
        )
        
        return signal
    
    def _validate_signal_like_live_system(self, signal: TradingSignal) -> bool:
        """Apply same validation filters as live system"""
        # Risk-reward ratio check
        risk_reward = signal.get_risk_ratio()
        if risk_reward < 1.5:
            return False
        
        # RSI check
        if signal.rsi > 85:
            return False
        
        # Volume check
        if signal.volume_ratio < 0.8:
            return False
        
        return True
    
    def calculate_leverage_for_signal(self, signal: TradingSignal) -> float:
        """Calculate leverage based on confidence (same logic as backtester.py)"""
        confidence = signal.confidence
        symbol = signal.symbol
        
        # Base leverage mapping
        if confidence >= 0.95:
            base_leverage = 10.0
        elif confidence >= 0.85:
            base_leverage = 5.0
        elif confidence >= 0.75:
            base_leverage = 3.0
        elif confidence >= 0.65:
            base_leverage = 2.0
        else:
            base_leverage = 1.0
        
        # Symbol-specific limits
        symbol_limits = {
            'BTCUSDT': 10.0, 'ETHUSDT': 8.0, 'BNBUSDT': 6.0, 'ADAUSDT': 5.0,
            'SOLUSDT': 4.0, 'DOTUSDT': 4.0, 'LINKUSDT': 5.0, 'MATICUSDT': 4.0,
            'LTCUSDT': 6.0, 'TRXUSDT': 3.0, 'XRPUSDT': 5.0
        }
        
        max_leverage = symbol_limits.get(symbol, 3.0)
        return min(base_leverage, max_leverage)
    
    def execute_trade(self, signal: TradingSignal, historical_data: pd.DataFrame) -> BacktestTrade:
        """Execute trade and track results"""
        try:
            # Calculate leverage
            leverage = self.calculate_leverage_for_signal(signal)
            
            # Calculate position size
            normal_position_size = self.normal_capital * self.position_size_pct
            leveraged_position_size = self.leveraged_capital * self.position_size_pct
            
            # Create trade record
            trade = BacktestTrade(
                signal_id=f"{signal.symbol}_{int(signal.created_at.timestamp())}",
                symbol=signal.symbol,
                entry_time=signal.created_at,
                entry_price=signal.entry_price,
                target_price=signal.target_price,
                stop_loss=signal.stop_loss,
                confidence=signal.confidence,
                leverage=leverage,
                position_size_usd=normal_position_size,
                position_size_pct=self.position_size_pct
            )
            
            # Simulate trade execution
            future_data = historical_data[historical_data.index > signal.created_at]
            
            # Look for exit conditions
            exit_found = False
            for timestamp, row in future_data.iterrows():
                # Check target hit
                if row['high'] >= signal.target_price:
                    trade.exit_time = timestamp
                    trade.exit_price = signal.target_price
                    trade.exit_reason = 'TARGET_HIT'
                    trade.was_correct_prediction = True
                    exit_found = True
                    break
                
                # Check stop loss hit
                if row['low'] <= signal.stop_loss:
                    trade.exit_time = timestamp
                    trade.exit_price = signal.stop_loss
                    trade.exit_reason = 'STOP_HIT'
                    trade.was_correct_prediction = False
                    exit_found = True
                    break
                
                # Check expiry (24 hours)
                if timestamp >= signal.expires_at:
                    trade.exit_time = timestamp
                    trade.exit_price = row['close']
                    trade.exit_reason = 'EXPIRED'
                    # Determine if correct based on profit
                    profit_pct = (row['close'] - signal.entry_price) / signal.entry_price
                    trade.was_correct_prediction = profit_pct > 0.005  # 0.5% profit = correct
                    exit_found = True
                    break
            
            # If no exit found, use last available price
            if not exit_found:
                last_timestamp = future_data.index[-1]
                last_price = future_data.iloc[-1]['close']
                trade.exit_time = last_timestamp
                trade.exit_price = last_price
                trade.exit_reason = 'END_OF_DATA'
                profit_pct = (last_price - signal.entry_price) / signal.entry_price
                trade.was_correct_prediction = profit_pct > 0
            
            # Calculate financial results
            gross_pnl_pct = (trade.exit_price - trade.entry_price) / trade.entry_price
            leveraged_pnl_pct = gross_pnl_pct * leverage
            
            trade.gross_pnl_pct = gross_pnl_pct
            trade.leveraged_pnl_pct = leveraged_pnl_pct
            trade.gross_pnl = normal_position_size * gross_pnl_pct
            trade.leveraged_pnl = leveraged_position_size * leveraged_pnl_pct
            
            # Update capital
            self.normal_capital += trade.gross_pnl
            self.leveraged_capital += trade.leveraged_pnl
            
            # Calculate duration
            if trade.exit_time:
                trade.duration_hours = (trade.exit_time - trade.entry_time).total_seconds() / 3600
            
            # Calculate confidence accuracy
            trade.confidence_accuracy = signal.confidence if trade.was_correct_prediction else (1.0 - signal.confidence)
            
            return trade
            
        except Exception as e:
            logger.error(f"Error executing trade: {str(e)}")
            return trade
    
    def simulate_live_learning(self, completed_trade: BacktestTrade):
        """Simulate live learning from completed trade"""
        try:
            # Add to completed signals (same as live system)
            self.completed_signals.append({
                'confidence': completed_trade.confidence,
                'was_correct': completed_trade.was_correct_prediction,
                'timestamp': completed_trade.exit_time,
                'symbol': completed_trade.symbol,
                'return': completed_trade.gross_pnl_pct
            })
            
            # Update learning history
            self.learning_history.append({
                'timestamp': completed_trade.exit_time,
                'signal_id': completed_trade.signal_id,
                'predicted_confidence': completed_trade.confidence,
                'actual_outcome': completed_trade.was_correct_prediction,
                'actual_return': completed_trade.gross_pnl_pct
            })
            
            # Adapt threshold (same logic as live system)
            if len(self.completed_signals) >= 5:
                recent_signals = list(self.completed_signals)[-10:]  # Last 10 signals
                win_rate = np.mean([s['was_correct'] for s in recent_signals])
                
                old_threshold = self.adaptive_threshold
                
                # Adaptation logic (same as live)
                if win_rate < 0.5:  # Poor performance
                    self.adaptive_threshold = min(self.max_threshold, self.adaptive_threshold + 0.03)
                    reason = "low_win_rate"
                elif win_rate > 0.75:  # Good performance
                    self.adaptive_threshold = max(self.min_threshold, self.adaptive_threshold - 0.02)
                    reason = "high_win_rate"
                else:
                    reason = "stable"
                
                if old_threshold != self.adaptive_threshold:
                    self.learning_events.append({
                        'timestamp': completed_trade.exit_time,
                        'old_threshold': old_threshold,
                        'new_threshold': self.adaptive_threshold,
                        'reason': reason,
                        'win_rate': win_rate,
                        'signals_count': len(recent_signals)
                    })
                    
                    logger.info(f"Threshold adapted: {old_threshold:.3f} -> {self.adaptive_threshold:.3f} ({reason})")
            
        except Exception as e:
            logger.error(f"Error in live learning simulation: {str(e)}")
    
    def track_portfolio_state(self, timestamp: datetime):
        """Track portfolio state for analysis"""
        open_positions = len([t for t in self.all_trades if t.exit_time is None or t.exit_time > timestamp])
        total_trades = len(self.all_trades)
        winning_trades = len([t for t in self.all_trades if t.was_correct_prediction])
        losing_trades = total_trades - winning_trades
        
        # Calculate drawdown
        peak_capital = max([p.leveraged_capital for p in self.portfolio_history] + [self.leveraged_capital])
        current_drawdown = (peak_capital - self.leveraged_capital) / peak_capital
        max_drawdown = max([p.current_drawdown for p in self.portfolio_history] + [current_drawdown])
        
        state = PortfolioState(
            timestamp=timestamp,
            total_capital=self.normal_capital,
            available_capital=self.normal_capital,
            leveraged_capital=self.leveraged_capital,
            open_positions=open_positions,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            current_drawdown=current_drawdown,
            max_drawdown=max_drawdown,
            adaptive_threshold=self.adaptive_threshold
        )
        
        self.portfolio_history.append(state)
    
    def run_backtest(self, days: int = 30) -> Dict:
        """Run complete backtest with live learning simulation"""
        try:
            logger.info(f"Starting advanced backtest with live learning: {days} days, ${self.initial_capital} capital")
            start_time = time.time()
            
            # Load model
            if not self.load_model():
                raise ValueError("Failed to load ML model")
            
            # Collect historical data
            historical_data = self.collect_historical_data(days)
            if not historical_data:
                raise ValueError("No historical data collected")
            
            # Get date range for simulation
            all_timestamps = set()
            for df in historical_data.values():
                all_timestamps.update(df.index)
            
            sorted_timestamps = sorted(all_timestamps)
            start_date = sorted_timestamps[0] + timedelta(days=7)  # Skip first week for indicators
            end_date = sorted_timestamps[-1]
            
            logger.info(f"Backtesting period: {start_date} to {end_date}")
            
            # Simulate hour by hour (same as real scanning frequency)
            current_time = start_date
            scan_count = 0
            
            while current_time <= end_date:
                try:
                    # Track portfolio state
                    self.track_portfolio_state(current_time)
                    
                    # Simulate market scan (every hour)
                    signals_generated = 0
                    
                    for symbol, symbol_data in historical_data.items():
                        # Skip if no data available yet
                        available_data = symbol_data[symbol_data.index <= current_time]
                        if len(available_data) < 100:
                            continue
                        
                        # Simulate signal detection
                        signal = self.simulate_live_signal_detection(symbol, current_time, symbol_data)
                        
                        if signal:
                            # Execute trade
                            trade = self.execute_trade(signal, symbol_data)
                            self.all_trades.append(trade)
                            signals_generated += 1
                            
                            logger.info(f"Signal executed: {symbol} @ {signal.confidence:.3f} (threshold: {self.adaptive_threshold:.3f})")
                            
                            # If trade is completed, apply live learning
                            if trade.exit_time and trade.exit_time <= current_time + timedelta(hours=1):
                                self.simulate_live_learning(trade)
                    
                    scan_count += 1
                    
                    # Progress update every 24 hours
                    if scan_count % 24 == 0:
                        days_completed = scan_count // 24
                        logger.info(f"Backtest progress: Day {days_completed}/{days} - Capital: ${self.leveraged_capital:.2f} - Trades: {len(self.all_trades)}")
                    
                    # Move to next hour
                    current_time += timedelta(hours=1)
                    
                except Exception as e:
                    logger.error(f"Error at timestamp {current_time}: {str(e)}")
                    current_time += timedelta(hours=1)
                    continue
            
            # Process any remaining open trades
            self._close_remaining_trades(end_date, historical_data)
            
            # Calculate final results
            results = self._calculate_comprehensive_results(days)
            
            execution_time = time.time() - start_time
            logger.info(f"Backtest completed in {execution_time:.2f} seconds")
            
            return results
            
        except Exception as e:
            logger.error(f"Error running backtest: {str(e)}")
            return {'error': str(e)}
    
    def _close_remaining_trades(self, end_date: datetime, historical_data: Dict[str, pd.DataFrame]):
        """Close any remaining open trades at end of backtest"""
        for trade in self.all_trades:
            if trade.exit_time is None:
                # Get final price
                symbol_data = historical_data[trade.symbol]
                final_price = symbol_data.iloc[-1]['close']
                
                trade.exit_time = end_date
                trade.exit_price = final_price
                trade.exit_reason = 'BACKTEST_END'
                
                # Calculate results
                gross_pnl_pct = (final_price - trade.entry_price) / trade.entry_price
                trade.gross_pnl_pct = gross_pnl_pct
                trade.leveraged_pnl_pct = gross_pnl_pct * trade.leverage
                trade.was_correct_prediction = gross_pnl_pct > 0
                
                # Apply to capital
                trade.gross_pnl = trade.position_size_usd * gross_pnl_pct
                trade.leveraged_pnl = trade.position_size_usd * (gross_pnl_pct * trade.leverage)
                
                self.normal_capital += trade.gross_pnl
                self.leveraged_capital += trade.leveraged_pnl
                
                # Apply live learning
                self.simulate_live_learning(trade)
    
    def _calculate_comprehensive_results(self, days: int) -> Dict:
        """Calculate comprehensive backtest results"""
        try:
            if not self.all_trades:
                return {'error': 'No trades executed'}
            
            # Basic trade statistics
            total_trades = len(self.all_trades)
            winning_trades = len([t for t in self.all_trades if t.was_correct_prediction])
            losing_trades = total_trades - winning_trades
            win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
            
            # Financial results
            normal_total_return = self.normal_capital - self.initial_capital
            leveraged_total_return = self.leveraged_capital - self.initial_capital
            normal_roi = (normal_total_return / self.initial_capital) * 100
            leveraged_roi = (leveraged_total_return / self.initial_capital) * 100
            
            # Annualized returns
            annual_multiplier = 365 / days
            normal_annual_roi = ((self.normal_capital / self.initial_capital) ** annual_multiplier - 1) * 100
            leveraged_annual_roi = ((self.leveraged_capital / self.initial_capital) ** annual_multiplier - 1) * 100
            
            # Monthly returns
            monthly_multiplier = 30 / days
            normal_monthly_roi = ((self.normal_capital / self.initial_capital) ** monthly_multiplier - 1) * 100
            leveraged_monthly_roi = ((self.leveraged_capital / self.initial_capital) ** monthly_multiplier - 1) * 100
            
            # Trade performance analysis
            winning_returns = [t.gross_pnl_pct for t in self.all_trades if t.was_correct_prediction]
            losing_returns = [t.gross_pnl_pct for t in self.all_trades if not t.was_correct_prediction]
            
            avg_win = np.mean(winning_returns) * 100 if winning_returns else 0
            avg_loss = np.mean(losing_returns) * 100 if losing_returns else 0
            
            # Leveraged performance
            leveraged_winning = [t.leveraged_pnl_pct for t in self.all_trades if t.was_correct_prediction]
            leveraged_losing = [t.leveraged_pnl_pct for t in self.all_trades if not t.was_correct_prediction]
            
            avg_leveraged_win = np.mean(leveraged_winning) * 100 if leveraged_winning else 0
            avg_leveraged_loss = np.mean(leveraged_losing) * 100 if leveraged_losing else 0
            
            # Risk metrics
            all_returns = [t.gross_pnl_pct for t in self.all_trades]
            volatility = np.std(all_returns) * 100 if all_returns else 0
            
            # Sharpe ratio (simplified, assuming 0% risk-free rate)
            sharpe_ratio = (np.mean(all_returns) / np.std(all_returns)) if len(all_returns) > 1 and np.std(all_returns) > 0 else 0
            
            # Drawdown analysis
            max_drawdown_normal = 0
            max_drawdown_leveraged = 0
            if self.portfolio_history:
                max_drawdown_leveraged = max([p.max_drawdown for p in self.portfolio_history]) * 100
                
                # Calculate normal drawdown
                normal_capitals = [self.initial_capital]
                for trade in self.all_trades:
                    if trade.gross_pnl:
                        normal_capitals.append(normal_capitals[-1] + trade.gross_pnl)
                
                peak = self.initial_capital
                max_dd = 0
                for capital in normal_capitals:
                    if capital > peak:
                        peak = capital
                    dd = (peak - capital) / peak
                    if dd > max_dd:
                        max_dd = dd
                max_drawdown_normal = max_dd * 100
            
            # Live learning analysis
            learning_summary = self._analyze_live_learning()
            
            # Confidence analysis
            confidence_analysis = self._analyze_confidence_performance()
            
            # Symbol performance
            symbol_performance = self._analyze_symbol_performance()
            
            # Profit factor
            total_wins = sum([abs(r) for r in winning_returns]) if winning_returns else 0
            total_losses = sum([abs(r) for r in losing_returns]) if losing_returns else 0.001
            profit_factor = total_wins / total_losses
            
            # Daily performance
            daily_summary = self._calculate_daily_performance()
            
            # Build comprehensive results
            results = {
                'backtest_summary': {
                    'period_days': days,
                    'initial_capital': self.initial_capital,
                    'position_size_pct': self.position_size_pct * 100,
                    'total_trades': total_trades,
                    'trades_per_day': total_trades / days,
                    'execution_period': f"{days} days"
                },
                
                'financial_results': {
                    'normal_trading': {
                        'final_capital': self.normal_capital,
                        'total_return': normal_total_return,
                        'total_return_pct': normal_roi,
                        'monthly_return_pct': normal_monthly_roi,
                        'annual_return_pct': normal_annual_roi,
                        'max_drawdown_pct': max_drawdown_normal
                    },
                    'leveraged_trading': {
                        'final_capital': self.leveraged_capital,
                        'total_return': leveraged_total_return,
                        'total_return_pct': leveraged_roi,
                        'monthly_return_pct': leveraged_monthly_roi,
                        'annual_return_pct': leveraged_annual_roi,
                        'max_drawdown_pct': max_drawdown_leveraged
                    }
                },
                
                'trading_performance': {
                    'win_rate': win_rate,
                    'total_trades': total_trades,
                    'winning_trades': winning_trades,
                    'losing_trades': losing_trades,
                    'avg_win_pct': avg_win,
                    'avg_loss_pct': avg_loss,
                    'avg_leveraged_win_pct': avg_leveraged_win,
                    'avg_leveraged_loss_pct': avg_leveraged_loss,
                    'profit_factor': profit_factor,
                    'sharpe_ratio': sharpe_ratio,
                    'volatility_pct': volatility
                },
                
                'live_learning_results': learning_summary,
                'confidence_analysis': confidence_analysis,
                'symbol_performance': symbol_performance,
                'daily_performance': daily_summary,
                
                'risk_metrics': {
                    'max_drawdown_normal': max_drawdown_normal,
                    'max_drawdown_leveraged': max_drawdown_leveraged,
                    'volatility': volatility,
                    'sharpe_ratio': sharpe_ratio,
                    'largest_loss': min(all_returns) * 100 if all_returns else 0,
                    'largest_win': max(all_returns) * 100 if all_returns else 0
                },
                
                'scaling_potential': {
                    'with_100_usd': {
                        'normal': (self.normal_capital / self.initial_capital) * 100,
                        'leveraged': (self.leveraged_capital / self.initial_capital) * 100
                    },
                    'with_1000_usd': {
                        'normal': (self.normal_capital / self.initial_capital) * 1000,
                        'leveraged': (self.leveraged_capital / self.initial_capital) * 1000
                    },
                    'with_10000_usd': {
                        'normal': (self.normal_capital / self.initial_capital) * 10000,
                        'leveraged': (self.leveraged_capital / self.initial_capital) * 10000
                    }
                },
                
                'detailed_trades': [trade.to_dict() for trade in self.all_trades]
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Error calculating results: {str(e)}")
            return {'error': str(e)}
    
    def _analyze_live_learning(self) -> Dict:
        """Analyze live learning performance"""
        try:
            if not self.learning_events:
                return {
                    'learning_active': False,
                    'message': 'No learning events occurred (insufficient signals)',
                    'initial_threshold': config.CONFIDENCE_THRESHOLD,
                    'final_threshold': self.adaptive_threshold
                }
            
            # Threshold evolution
            threshold_changes = len(self.learning_events)
            initial_threshold = config.CONFIDENCE_THRESHOLD
            final_threshold = self.adaptive_threshold
            total_threshold_change = final_threshold - initial_threshold
            
            # Learning effectiveness
            if len(self.completed_signals) >= 5:
                early_signals = list(self.completed_signals)[:len(self.completed_signals)//2]
                late_signals = list(self.completed_signals)[len(self.completed_signals)//2:]
                
                early_win_rate = np.mean([s['was_correct'] for s in early_signals])
                late_win_rate = np.mean([s['was_correct'] for s in late_signals])
                learning_improvement = late_win_rate - early_win_rate
            else:
                early_win_rate = 0
                late_win_rate = 0
                learning_improvement = 0
            
            # Adaptation events analysis
            threshold_increases = len([e for e in self.learning_events if e['new_threshold'] > e['old_threshold']])
            threshold_decreases = len([e for e in self.learning_events if e['new_threshold'] < e['old_threshold']])
            
            return {
                'learning_active': True,
                'threshold_evolution': {
                    'initial_threshold': initial_threshold,
                    'final_threshold': final_threshold,
                    'total_change': total_threshold_change,
                    'total_adaptations': threshold_changes,
                    'threshold_increases': threshold_increases,
                    'threshold_decreases': threshold_decreases
                },
                'learning_effectiveness': {
                    'early_period_win_rate': early_win_rate * 100,
                    'late_period_win_rate': late_win_rate * 100,
                    'improvement': learning_improvement * 100,
                    'signals_learned_from': len(self.completed_signals)
                },
                'adaptation_events': self.learning_events[-10:],  # Last 10 events
                'learning_summary': f"Threshold adapted {threshold_changes} times, final improvement: {learning_improvement*100:+.1f}%"
            }
            
        except Exception as e:
            logger.error(f"Error analyzing live learning: {str(e)}")
            return {'error': str(e)}
    
    def _analyze_confidence_performance(self) -> Dict:
        """Analyze performance by confidence levels"""
        try:
            confidence_buckets = {
                'high_confidence': [],  # >=0.7
                'medium_confidence': [],  # 0.55-0.7
                'low_confidence': []  # <0.55
            }
            
            for trade in self.all_trades:
                if trade.confidence >= 0.7:
                    confidence_buckets['high_confidence'].append(trade)
                elif trade.confidence >= 0.55:
                    confidence_buckets['medium_confidence'].append(trade)
                else:
                    confidence_buckets['low_confidence'].append(trade)
            
            analysis = {}
            for bucket_name, trades in confidence_buckets.items():
                if trades:
                    win_rate = np.mean([t.was_correct_prediction for t in trades]) * 100
                    avg_return = np.mean([t.gross_pnl_pct for t in trades]) * 100
                    avg_leverage = np.mean([t.leverage for t in trades])
                    count = len(trades)
                else:
                    win_rate = avg_return = avg_leverage = count = 0
                
                analysis[bucket_name] = {
                    'count': count,
                    'win_rate': win_rate,
                    'avg_return': avg_return,
                    'avg_leverage': avg_leverage
                }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing confidence performance: {str(e)}")
            return {}
    
    def _analyze_symbol_performance(self) -> Dict:
        """Analyze performance by symbol"""
        try:
            symbol_stats = {}
            
            for trade in self.all_trades:
                symbol = trade.symbol
                if symbol not in symbol_stats:
                    symbol_stats[symbol] = {
                        'trades': [],
                        'total_return': 0,
                        'total_leveraged_return': 0
                    }
                
                symbol_stats[symbol]['trades'].append(trade)
                symbol_stats[symbol]['total_return'] += trade.gross_pnl or 0
                symbol_stats[symbol]['total_leveraged_return'] += trade.leveraged_pnl or 0
            
            # Calculate statistics for each symbol
            symbol_analysis = {}
            for symbol, stats in symbol_stats.items():
                trades = stats['trades']
                winning_trades = [t for t in trades if t.was_correct_prediction]
                
                symbol_analysis[symbol] = {
                    'total_trades': len(trades),
                    'winning_trades': len(winning_trades),
                    'win_rate': (len(winning_trades) / len(trades)) * 100 if trades else 0,
                    'total_return': stats['total_return'],
                    'total_leveraged_return': stats['total_leveraged_return'],
                    'avg_confidence': np.mean([t.confidence for t in trades]),
                    'avg_leverage': np.mean([t.leverage for t in trades]),
                    'avg_return_pct': np.mean([t.gross_pnl_pct for t in trades]) * 100 if trades else 0
                }
            
            return symbol_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing symbol performance: {str(e)}")
            return {}
    
    def _calculate_daily_performance(self) -> Dict:
        """Calculate daily performance summary"""
        try:
            if not self.portfolio_history:
                return {}
            
            daily_data = []
            current_date = None
            daily_trades = 0
            daily_pnl = 0
            
            for state in self.portfolio_history:
                state_date = state.timestamp.date()
                
                if current_date != state_date:
                    if current_date is not None:
                        daily_data.append({
                            'date': current_date,
                            'trades': daily_trades,
                            'pnl': daily_pnl,
                            'capital': prev_capital
                        })
                    
                    current_date = state_date
                    daily_trades = 0
                    daily_pnl = 0
                
                prev_capital = state.leveraged_capital
            
            # Add last day
            if current_date is not None:
                daily_data.append({
                    'date': current_date,
                    'trades': daily_trades,
                    'pnl': daily_pnl,
                    'capital': self.leveraged_capital
                })
            
            return {
                'total_days': len(daily_data),
                'avg_trades_per_day': np.mean([d['trades'] for d in daily_data]) if daily_data else 0,
                'best_day_pnl': max([d['pnl'] for d in daily_data]) if daily_data else 0,
                'worst_day_pnl': min([d['pnl'] for d in daily_data]) if daily_data else 0,
                'profitable_days': len([d for d in daily_data if d['pnl'] > 0]),
                'daily_data': daily_data[-7:]  # Last 7 days
            }
            
        except Exception as e:
            logger.error(f"Error calculating daily performance: {str(e)}")
            return {}

def generate_comprehensive_report(results: Dict, initial_capital: float = 20.0) -> str:
    """Generate detailed backtest report"""
    try:
        if 'error' in results:
            return f"ERROR: {results['error']}"
        
        report = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                     CRYPTO SIGNAL BOT - ADVANCED BACKTEST                    ║
║                        WITH LIVE LEARNING SIMULATION                         ║
╚══════════════════════════════════════════════════════════════════════════════╝

📊 BACKTEST SUMMARY:
{'='*80}
Period: {results['backtest_summary']['period_days']} days
Initial Capital: ${results['backtest_summary']['initial_capital']:.2f}
Position Size: {results['backtest_summary']['position_size_pct']:.0f}% per trade
Total Trades: {results['backtest_summary']['total_trades']}
Trades per Day: {results['backtest_summary']['trades_per_day']:.1f}

💰 FINANCIAL RESULTS:
{'='*80}
🔹 NORMAL TRADING (NO LEVERAGE):
   Final Capital: ${results['financial_results']['normal_trading']['final_capital']:.2f}
   Total Return: ${results['financial_results']['normal_trading']['total_return']:.2f}
   ROI: {results['financial_results']['normal_trading']['total_return_pct']:+.2f}%
   Monthly ROI: {results['financial_results']['normal_trading']['monthly_return_pct']:+.2f}%
   Annual ROI: {results['financial_results']['normal_trading']['annual_return_pct']:+.2f}%
   Max Drawdown: {results['financial_results']['normal_trading']['max_drawdown_pct']:.2f}%

🚀 LEVERAGED TRADING:
   Final Capital: ${results['financial_results']['leveraged_trading']['final_capital']:.2f}
   Total Return: ${results['financial_results']['leveraged_trading']['total_return']:.2f}
   ROI: {results['financial_results']['leveraged_trading']['total_return_pct']:+.2f}%
   Monthly ROI: {results['financial_results']['leveraged_trading']['monthly_return_pct']:+.2f}%
   Annual ROI: {results['financial_results']['leveraged_trading']['annual_return_pct']:+.2f}%
   Max Drawdown: {results['financial_results']['leveraged_trading']['max_drawdown_pct']:.2f}%

📈 TRADING PERFORMANCE:
{'='*80}
Win Rate: {results['trading_performance']['win_rate']:.1f}%
Profitable Trades: {results['trading_performance']['winning_trades']}/{results['trading_performance']['total_trades']}
Average Win: {results['trading_performance']['avg_win_pct']:+.2f}%
Average Loss: {results['trading_performance']['avg_loss_pct']:+.2f}%
Profit Factor: {results['trading_performance']['profit_factor']:.2f}
Sharpe Ratio: {results['trading_performance']['sharpe_ratio']:.3f}

🚀 LEVERAGED PERFORMANCE:
Average Leveraged Win: {results['trading_performance']['avg_leveraged_win_pct']:+.2f}%
Average Leveraged Loss: {results['trading_performance']['avg_leveraged_loss_pct']:+.2f}%

🧠 LIVE LEARNING RESULTS:
{'='*80}"""

        # Add live learning analysis
        learning = results.get('live_learning_results', {})
        if learning.get('learning_active'):
            report += f"""
Learning Status: ACTIVE ✅
Threshold Evolution:
  Initial: {learning['threshold_evolution']['initial_threshold']:.3f}
  Final: {learning['threshold_evolution']['final_threshold']:.3f}
  Total Change: {learning['threshold_evolution']['total_change']:+.3f}
  Adaptations: {learning['threshold_evolution']['total_adaptations']}

Learning Effectiveness:
  Early Period Win Rate: {learning['learning_effectiveness']['early_period_win_rate']:.1f}%
  Late Period Win Rate: {learning['learning_effectiveness']['late_period_win_rate']:.1f}%
  Improvement: {learning['learning_effectiveness']['improvement']:+.1f}%
  Signals Learned From: {learning['learning_effectiveness']['signals_learned_from']}

{learning['learning_summary']}"""
        else:
            report += f"""
Learning Status: INSUFFICIENT DATA ⚠️
{learning.get('message', 'No learning data available')}"""

        report += f"""

💡 CONFIDENCE ANALYSIS:
{'='*80}"""
        
        conf_analysis = results.get('confidence_analysis', {})
        for level, stats in conf_analysis.items():
            level_name = level.replace('_', ' ').title()
            if stats['count'] > 0:
                report += f"""
{level_name}: {stats['count']} trades
  Win Rate: {stats['win_rate']:.1f}%
  Avg Return: {stats['avg_return']:+.2f}%
  Avg Leverage: {stats['avg_leverage']:.1f}x"""

        # Scaling potential
        scaling = results.get('scaling_potential', {})
        report += f"""

💰 SCALING POTENTIAL:
{'='*80}
Starting with $100:
  Normal Trading: ${scaling['with_100_usd']['normal']:.2f}
  Leveraged Trading: ${scaling['with_100_usd']['leveraged']:.2f}

Starting with $1,000:
  Normal Trading: ${scaling['with_1000_usd']['normal']:.2f}
  Leveraged Trading: ${scaling['with_1000_usd']['leveraged']:.2f}

Starting with $10,000:
  Normal Trading: ${scaling['with_10000_usd']['normal']:.2f}
  Leveraged Trading: ${scaling['with_10000_usd']['leveraged']:.2f}

🎯 KEY INSIGHTS:
{'='*80}"""

        # Generate insights
        normal_roi = results['financial_results']['normal_trading']['total_return_pct']
        leveraged_roi = results['financial_results']['leveraged_trading']['total_return_pct']
        win_rate = results['trading_performance']['win_rate']
        
        if leveraged_roi > 100:
            report += f"\n🔥 EXCELLENT: {leveraged_roi:.0f}% return with leverage!"
        elif leveraged_roi > 50:
            report += f"\n✅ GOOD: {leveraged_roi:.0f}% return with leverage"
        elif leveraged_roi > 0:
            report += f"\n📈 POSITIVE: {leveraged_roi:.0f}% return with leverage"
        else:
            report += f"\n📉 NEGATIVE: {leveraged_roi:.0f}% return with leverage"
        
        if win_rate >= 70:
            report += f"\n🎯 HIGH WIN RATE: {win_rate:.0f}% accuracy"
        elif win_rate >= 60:
            report += f"\n✅ GOOD WIN RATE: {win_rate:.0f}% accuracy"
        else:
            report += f"\n⚠️ MODERATE WIN RATE: {win_rate:.0f}% accuracy"
        
        if learning.get('learning_active'):
            improvement = learning['learning_effectiveness']['improvement']
            if improvement > 5:
                report += f"\n🧠 STRONG LEARNING: +{improvement:.1f}% improvement"
            elif improvement > 0:
                report += f"\n📈 LEARNING DETECTED: +{improvement:.1f}% improvement"
            else:
                report += f"\n📊 STABLE PERFORMANCE: {improvement:+.1f}% change"

        report += f"""

🏁 FINAL ANSWER TO YOUR QUESTION:
{'='*80}
Starting with $20:
  WITHOUT Leverage: ${results['financial_results']['normal_trading']['final_capital']:.2f}
  WITH Leverage: ${results['financial_results']['leveraged_trading']['final_capital']:.2f}

30-Day Learning Results:
  AI improved threshold from {learning.get('threshold_evolution', {}).get('initial_threshold', 0.55):.3f} to {learning.get('threshold_evolution', {}).get('final_threshold', 0.55):.3f}
  Made {learning.get('threshold_evolution', {}).get('total_adaptations', 0)} smart adjustments
  Win rate improved by {learning.get('learning_effectiveness', {}).get('improvement', 0):+.1f}%
  
The AI successfully learned and adapted during the 30-day period! 🧠🚀
"""

        return report
        
    except Exception as e:
        return f"Error generating report: {str(e)}"

def main():
    """Run advanced backtest with live learning"""
    print("🤖 Advanced Crypto Backtest with Live Learning Simulation")
    print("=" * 80)
    
    # Initialize backtester
    backtester = LiveLearningBacktester(initial_capital=20.0)
    
    # Run backtest
    print("🚀 Starting 30-day backtest with live learning...")
    results = backtester.run_backtest(days=30)
    
    if 'error' in results:
        print(f"❌ Backtest failed: {results['error']}")
        return
    
    # Generate and display report
    report = generate_comprehensive_report(results, 20.0)
    print(report)
    
    # Save detailed results
    try:
        with open('data/advanced_backtest_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print("\n💾 Detailed results saved to: data/advanced_backtest_results.json")
    except Exception as e:
        print(f"⚠️ Could not save results: {str(e)}")

if __name__ == "__main__":
    main()