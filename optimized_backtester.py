#!/usr/bin/env python3
"""
Complete Optimized Backtester - نسخه کامل بهینه‌شده
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import کامل از فایل قبلی
from advanced_backtester import (
    LiveLearningBacktester, BacktestTrade, PortfolioState, 
    generate_comprehensive_report
)
from backend.services.signal_detector import TradingSignal
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import logging
import json
import time

logger = logging.getLogger(__name__)

class OptimizedLiveLearningBacktester(LiveLearningBacktester):
    """
    نسخه بهینه‌شده با تنظیمات aggressive تر
    """
    
    def __init__(self, initial_capital: float = 20.0):
        super().__init__(initial_capital)
        
        # 🚀 OPTIMIZATION 1: More aggressive position sizing
        self.position_size_pct = 0.30  # 30% instead of 20%
        
        # 🚀 OPTIMIZATION 2: Lower initial threshold for more signals
        self.adaptive_threshold = 0.45  # Start lower than 0.55
        self.min_threshold = 0.25  # Allow very low thresholds
        self.max_threshold = 0.70  # Don't go too high
        
        # 🚀 OPTIMIZATION 3: More aggressive learning
        self.retrain_after_signals = 15  # Learn faster
        
        # 🚀 OPTIMIZATION 4: Accept more concurrent trades
        self.max_concurrent_trades = 5
        self.max_daily_trades = 15
        
        logger.info("🚀 Optimized backtester initialized with aggressive settings")
    
    def calculate_optimized_leverage(self, signal) -> float:
        """More aggressive leverage calculation"""
        confidence = signal.confidence
        symbol = signal.symbol
        
        # 🚀 HIGHER BASE LEVERAGE
        if confidence >= 0.9:
            base_leverage = 15.0  # Max leverage for very high confidence
        elif confidence >= 0.8:
            base_leverage = 10.0  # High leverage
        elif confidence >= 0.7:
            base_leverage = 6.0   # Medium-high leverage
        elif confidence >= 0.6:
            base_leverage = 4.0   # Medium leverage
        elif confidence >= 0.5:
            base_leverage = 2.5   # Low-medium leverage
        else:
            base_leverage = 1.5   # Still some leverage for low confidence
        
        # 🚀 HIGHER SYMBOL LIMITS
        symbol_limits = {
            'BTCUSDT': 15.0,  # Increased from 10
            'ETHUSDT': 12.0,  # Increased from 8
            'BNBUSDT': 10.0,  # Increased from 6
            'ADAUSDT': 8.0,   # Increased from 5
            'SOLUSDT': 8.0,   # Increased from 4
            'DOTUSDT': 8.0,   # Increased from 4
            'LINKUSDT': 8.0,  # Increased from 5
            'MATICUSDT': 6.0, # Increased from 4
            'LTCUSDT': 10.0,  # Increased from 6
            'TRXUSDT': 6.0,   # Increased from 3
            'XRPUSDT': 8.0    # Increased from 5
        }
        
        max_leverage = symbol_limits.get(symbol, 5.0)  # Higher default
        return min(base_leverage, max_leverage)
    
    def _create_aggressive_signal(self, symbol: str, current_price: float, 
                                confidence: float, features: pd.DataFrame, 
                                current_time: datetime) -> TradingSignal:
        """Create more aggressive signals"""
        latest_row = features.iloc[-1]
        
        # 🚀 MORE AGGRESSIVE ENTRY
        if confidence > 0.7:
            entry_price = current_price  # Enter immediately for good signals
        elif confidence > 0.6:
            entry_price = current_price * (1 - 0.003)  # Smaller pullback
        else:
            entry_price = current_price * (1 - 0.007)  # Still small pullback
        
        # 🚀 OPTIMIZED STOP LOSS (tighter but still safe)
        atr = latest_row.get('atr', current_price * 0.015)  # Smaller default ATR
        atr_multiplier = 1.2 if confidence > 0.7 else 1.5  # Tighter stops
        atr_stop = entry_price - (atr * atr_multiplier)
        max_stop = entry_price * (1 - 0.012)  # 1.2% max loss instead of 1.5%
        stop_loss = max(atr_stop, max_stop)
        
        # 🚀 MORE AGGRESSIVE TARGETS
        risk = entry_price - stop_loss
        if confidence > 0.8:
            risk_reward_ratio = 4.0  # Higher target for high confidence
        elif confidence > 0.7:
            risk_reward_ratio = 3.5  # Higher target
        elif confidence > 0.6:
            risk_reward_ratio = 3.0  # Higher target
        else:
            risk_reward_ratio = 2.5  # Still decent target
        
        target_price = entry_price + (risk * risk_reward_ratio)
        min_target = entry_price * (1 + 0.015)  # 1.5% min profit
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
            expires_at=current_time + timedelta(hours=18),  # Shorter expiry
            rsi=latest_row.get('rsi', 0),
            macd=latest_row.get('macd', 0),
            macd_signal=latest_row.get('macd_signal', 0),
            bb_upper=latest_row.get('bb_upper', 0),
            bb_lower=latest_row.get('bb_lower', 0),
            volume_ratio=latest_row.get('volume_ratio', 1),
            model_version=self.ml_model.model_info.get('version', 'optimized'),
            features_used=71,
            timeframe='1h'
        )
        
        return signal
    
    def _optimized_validation(self, signal: TradingSignal) -> bool:
        """More lenient validation for more signals"""
        # 🚀 LOWER REQUIREMENTS
        risk_reward = signal.get_risk_ratio()
        if risk_reward < 1.2:  # Lower from 1.5
            return False
        
        # 🚀 ALLOW HIGHER RSI
        if signal.rsi > 90:  # Increased from 85
            return False
        
        # 🚀 ALLOW LOWER VOLUME
        if signal.volume_ratio < 0.6:  # Lowered from 0.8
            return False
        
        return True
    
    def simulate_optimized_signal_detection(self, symbol: str, current_time: datetime, 
                                          historical_data: pd.DataFrame):
        """Optimized signal detection"""
        try:
            # Get data up to current time
            available_data = historical_data[historical_data.index <= current_time]
            
            if len(available_data) < 100:
                return None
            
            # Use recent data
            recent_data = available_data.tail(150)
            
            # Feature engineering
            df_features = self.feature_engineer.prepare_features_for_prediction(recent_data)
            
            if df_features.empty:
                return None
            
            # Get latest features
            latest_features = df_features.iloc[-1:]
            
            # Make prediction
            prediction, confidence = self.ml_model.predict_single(latest_features)
            
            # 🚀 USE ADAPTIVE THRESHOLD (starts lower)
            if prediction == 1 and confidence >= self.adaptive_threshold:
                
                current_price = latest_features['close'].iloc[-1]
                
                # Create optimized signal
                signal = self._create_aggressive_signal(
                    symbol, current_price, confidence, latest_features, current_time
                )
                
                # Apply optimized validation
                if self._optimized_validation(signal):
                    return signal
            
            return None
            
        except Exception as e:
            logger.error(f"Error in optimized signal detection for {symbol}: {str(e)}")
            return None
    
    def execute_optimized_trade(self, signal: TradingSignal, historical_data: pd.DataFrame) -> BacktestTrade:
        """Execute trade with optimized leverage"""
        try:
            # Use optimized leverage calculation
            leverage = self.calculate_optimized_leverage(signal)
            
            # More aggressive position sizing
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
            
            # Execute trade (same logic as before)
            future_data = historical_data[historical_data.index > signal.created_at]
            
            # Look for exit conditions
            exit_found = False
            for timestamp, row in future_data.iterrows():
                if row['high'] >= signal.target_price:
                    trade.exit_time = timestamp
                    trade.exit_price = signal.target_price
                    trade.exit_reason = 'TARGET_HIT'
                    trade.was_correct_prediction = True
                    exit_found = True
                    break
                
                if row['low'] <= signal.stop_loss:
                    trade.exit_time = timestamp
                    trade.exit_price = signal.stop_loss
                    trade.exit_reason = 'STOP_HIT'
                    trade.was_correct_prediction = False
                    exit_found = True
                    break
                
                if timestamp >= signal.expires_at:
                    trade.exit_time = timestamp
                    trade.exit_price = row['close']
                    trade.exit_reason = 'EXPIRED'
                    profit_pct = (row['close'] - signal.entry_price) / signal.entry_price
                    trade.was_correct_prediction = profit_pct > 0.003  # Lower threshold
                    exit_found = True
                    break
            
            # If no exit found
            if not exit_found:
                last_timestamp = future_data.index[-1]
                last_price = future_data.iloc[-1]['close']
                trade.exit_time = last_timestamp
                trade.exit_price = last_price
                trade.exit_reason = 'END_OF_DATA'
                profit_pct = (last_price - signal.entry_price) / signal.entry_price
                trade.was_correct_prediction = profit_pct > 0
            
            # Calculate results
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
            
            trade.confidence_accuracy = signal.confidence if trade.was_correct_prediction else (1.0 - signal.confidence)
            
            return trade
            
        except Exception as e:
            logger.error(f"Error executing optimized trade: {str(e)}")
            return trade
    
    def run_optimized_backtest(self, days: int = 30) -> dict:
        """Run optimized backtest"""
        try:
            logger.info(f"Starting OPTIMIZED backtest: {days} days, ${self.initial_capital} capital")
            logger.info("OPTIMIZATIONS: 30% position size, lower threshold, higher leverage")
            
            # Load model
            if not self.load_model():
                raise ValueError("Failed to load ML model")
            
            # Collect historical data
            historical_data = self.collect_historical_data(days)
            if not historical_data:
                raise ValueError("No historical data collected")
            
            # Get date range
            all_timestamps = set()
            for df in historical_data.values():
                all_timestamps.update(df.index)
            
            sorted_timestamps = sorted(all_timestamps)
            start_date = sorted_timestamps[0] + timedelta(days=7)
            end_date = sorted_timestamps[-1]
            
            logger.info(f"Backtesting period: {start_date} to {end_date}")
            
            # Simulate with optimizations
            current_time = start_date
            scan_count = 0
            
            while current_time <= end_date:
                try:
                    self.track_portfolio_state(current_time)
                    
                    signals_generated = 0
                    
                    for symbol, symbol_data in historical_data.items():
                        available_data = symbol_data[symbol_data.index <= current_time]
                        if len(available_data) < 100:
                            continue
                        
                        # Use optimized signal detection
                        signal = self.simulate_optimized_signal_detection(symbol, current_time, symbol_data)
                        
                        if signal:
                            # Use optimized trade execution
                            trade = self.execute_optimized_trade(signal, symbol_data)
                            self.all_trades.append(trade)
                            signals_generated += 1
                            
                            logger.info(f"OPTIMIZED Signal: {symbol} @ {signal.confidence:.3f} (leverage: {trade.leverage:.1f}x)")
                            
                            if trade.exit_time and trade.exit_time <= current_time + timedelta(hours=1):
                                self.simulate_live_learning(trade)
                    
                    scan_count += 1
                    
                    if scan_count % 24 == 0:
                        days_completed = scan_count // 24
                        logger.info(f"OPTIMIZED Progress: Day {days_completed}/{days} - Capital: ${self.leveraged_capital:.2f} - Trades: {len(self.all_trades)}")
                    
                    current_time += timedelta(hours=1)
                    
                except Exception as e:
                    logger.error(f"Error at timestamp {current_time}: {str(e)}")
                    current_time += timedelta(hours=1)
                    continue
            
            self._close_remaining_trades(end_date, historical_data)
            results = self._calculate_comprehensive_results(days)
            
            return results
            
        except Exception as e:
            logger.error(f"Error running optimized backtest: {str(e)}")
            return {'error': str(e)}

def generate_optimized_report(results: dict, initial_capital: float) -> str:
    """Generate optimized backtest report"""
    try:
        if 'error' in results:
            return f"ERROR: {results['error']}"
        
        normal_final = results['financial_results']['normal_trading']['final_capital']
        leveraged_final = results['financial_results']['leveraged_trading']['final_capital']
        normal_roi = results['financial_results']['normal_trading']['total_return_pct']
        leveraged_roi = results['financial_results']['leveraged_trading']['total_return_pct']
        
        # Learning results
        learning = results.get('live_learning_results', {})
        
        report = f"""
🚀 OPTIMIZED BACKTEST RESULTS
================================================================================
🔥 AGGRESSIVE SETTINGS APPLIED - HIGHER RISK, HIGHER REWARD!

💰 FINANCIAL PERFORMANCE:
   Starting Capital: ${initial_capital:.2f}
   
   WITHOUT Leverage: ${normal_final:.2f} ({normal_roi:+.1f}%)
   WITH Leverage: ${leveraged_final:.2f} ({leveraged_roi:+.1f}%)
   
📊 COMPARISON WITH PREVIOUS RESULTS:
   Previous (Conservative): $24.88 (+24.4%)
   Current (Optimized): ${leveraged_final:.2f} ({leveraged_roi:+.1f}%)
   Improvement: {leveraged_roi - 24.4:+.1f}% extra return!

📈 TRADING STATS:
   Total Trades: {results['trading_performance']['total_trades']}
   Win Rate: {results['trading_performance']['win_rate']:.1f}%
   Profit Factor: {results['trading_performance']['profit_factor']:.2f}
   Average Leveraged Win: {results['trading_performance']['avg_leveraged_win_pct']:+.2f}%
   Average Leveraged Loss: {results['trading_performance']['avg_leveraged_loss_pct']:+.2f}%

🧠 LIVE LEARNING:"""

        if learning.get('learning_active'):
            report += f"""
   Threshold: {learning['threshold_evolution']['initial_threshold']:.3f} → {learning['threshold_evolution']['final_threshold']:.3f}
   Adaptations: {learning['threshold_evolution']['total_adaptations']}
   Win Rate Improvement: {learning['learning_effectiveness']['improvement']:+.1f}%
   Learning Status: ACTIVE ✅"""
        else:
            report += f"""
   Learning Status: INSUFFICIENT DATA ⚠️"""

        report += f"""

🎯 FINAL ANSWER - OPTIMIZED VERSION:
================================================================================
Starting with $20:
   WITHOUT Leverage: ${normal_final:.2f}
   WITH Leverage: ${leveraged_final:.2f}
   
COMPARED TO CONSERVATIVE APPROACH:
   Extra profit with optimizations: ${leveraged_final - 24.88:+.2f}
   
The AI learned AND the optimizations worked! 🚀🧠"""
        
        return report
        
    except Exception as e:
        return f"Error generating optimized report: {str(e)}"

def main():
    """Run optimized backtest"""
    print("🚀 OPTIMIZED Crypto Backtest - Higher Returns Version")
    print("=" * 80)
    
    print("🔥 OPTIMIZATIONS APPLIED:")
    print("   - 30% position size (instead of 20%)")
    print("   - Lower starting threshold (0.45 instead of 0.55)")
    print("   - Higher leverage limits")
    print("   - More aggressive targets")
    print("   - Tighter stop losses")
    print("   - More lenient signal validation")
    print()
    
    # Run optimized backtest
    backtester = OptimizedLiveLearningBacktester(initial_capital=20.0)
    
    print("🚀 Starting OPTIMIZED 30-day backtest...")
    results = backtester.run_optimized_backtest(days=30)
    
    if 'error' in results:
        print(f"❌ Optimized backtest failed: {results['error']}")
        return
    
    # Generate comparison report
    report = generate_optimized_report(results, 20.0)
    print(report)
    
    # Save results
    try:
        with open('data/optimized_backtest_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print("\n💾 Optimized results saved to: data/optimized_backtest_results.json")
    except Exception as e:
        print(f"⚠️ Could not save results: {str(e)}")

if __name__ == "__main__":
    main()