import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass
from backend.services.fallback_data_collector import FallbackDataCollector
from backend.services.feature_engineer import FeatureEngineer
from backend.services.ml_model import MLModel
from config import config

logger = logging.getLogger(__name__)

@dataclass
class BacktestSignal:
    """Backtest signal data"""
    symbol: str
    timestamp: datetime
    entry_price: float
    target_price: float
    stop_loss: float
    confidence: float
    leverage: float = 1.0  # Default no leverage
    signal_type: str = 'BUY'
    
    # Results (filled after execution)
    exit_price: Optional[float] = None
    exit_timestamp: Optional[datetime] = None
    exit_reason: Optional[str] = None  # 'TARGET_HIT', 'STOP_HIT', 'EXPIRED'
    profit_loss: Optional[float] = None
    profit_loss_pct: Optional[float] = None
    leveraged_profit_loss_pct: Optional[float] = None  # P/L with leverage
    duration_hours: Optional[float] = None
    
    def to_dict(self) -> Dict:
        return {
            'symbol': self.symbol,
            'timestamp': self.timestamp.isoformat(),
            'entry_price': self.entry_price,
            'target_price': self.target_price,
            'stop_loss': self.stop_loss,
            'confidence': self.confidence,
            'leverage': self.leverage,
            'signal_type': self.signal_type,
            'exit_price': self.exit_price,
            'exit_timestamp': self.exit_timestamp.isoformat() if self.exit_timestamp else None,
            'exit_reason': self.exit_reason,
            'profit_loss': self.profit_loss,
            'profit_loss_pct': self.profit_loss_pct,
            'leveraged_profit_loss_pct': self.leveraged_profit_loss_pct,
            'duration_hours': self.duration_hours
        }

class Backtester:
    """
    Backtesting system for trading signals
    """
    
    def __init__(self):
        self.data_collector = FallbackDataCollector()
        self.feature_engineer = FeatureEngineer()
        self.ml_model = MLModel()
        
    def load_model(self, model_path: str = None) -> bool:
        """Load ML model for backtesting"""
        try:
            if model_path:
                return self.ml_model.load_model(model_path)
            else:
                return self.ml_model.load_latest_model()
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False
    
    def collect_historical_data(self, symbols: List[str], days: int = 30) -> Dict[str, pd.DataFrame]:
        """Collect historical data for backtesting"""
        try:
            logger.info(f"Collecting historical data for {len(symbols)} symbols, {days} days...")
            
            historical_data = {}
            end_time = datetime.now()
            start_time = end_time - timedelta(days=days)
            
            for symbol in symbols:
                try:
                    # Get more data than needed to account for indicator warmup
                    df = self.data_collector.get_historical_klines(
                        symbol, "1h", start_time - timedelta(days=7), end_time
                    )
                    
                    if not df.empty:
                        historical_data[symbol] = df
                        logger.info(f"Collected {len(df)} candles for {symbol}")
                    else:
                        logger.warning(f"No data collected for {symbol}")
                        
                except Exception as e:
                    logger.error(f"Error collecting data for {symbol}: {str(e)}")
                    continue
            
            logger.info(f"Historical data collection completed: {len(historical_data)} symbols")
            return historical_data
            
        except Exception as e:
            logger.error(f"Error collecting historical data: {str(e)}")
            return {}
    
    def generate_signals(self, df: pd.DataFrame, symbol: str) -> List[BacktestSignal]:
        """Generate trading signals from historical data"""
        try:
            logger.info(f"Generating signals for {symbol}...")
            
            # Feature engineering
            df_features = self.feature_engineer.prepare_features_for_prediction(df)
            
            if df_features.empty:
                logger.warning(f"No features generated for {symbol}")
                return []
            
            signals = []
            
            # Simulate real-time signal generation
            for i in range(len(df_features)):
                try:
                    # Use data up to current point
                    current_features = df_features.iloc[i:i+1]
                    
                    # Make prediction
                    prediction, confidence = self.ml_model.predict_single(current_features)
                    
                    # Check if signal meets criteria
                    if prediction == 1 and confidence >= config.CONFIDENCE_THRESHOLD:
                        
                        current_price = df_features.iloc[i]['close']
                        timestamp = df_features.index[i]
                        
                        # Calculate signal parameters using the same logic as real system
                        entry_price, target_price, stop_loss = self._calculate_signal_params(
                            current_price, df_features.iloc[max(0, i-20):i+1], confidence
                        )
                        
                        # Calculate optimal leverage
                        leverage = self._calculate_leverage(confidence, symbol)
                        
                        signal = BacktestSignal(
                            symbol=symbol,
                            timestamp=timestamp,
                            entry_price=entry_price,
                            target_price=target_price,
                            stop_loss=stop_loss,
                            confidence=confidence,
                            leverage=leverage
                        )
                        
                        signals.append(signal)
                        logger.info(f"Signal generated: {symbol} @ {timestamp} - {confidence:.3f}")
                        
                except Exception as e:
                    logger.error(f"Error generating signal at index {i}: {str(e)}")
                    continue
            
            logger.info(f"Generated {len(signals)} signals for {symbol}")
            return signals
            
        except Exception as e:
            logger.error(f"Error generating signals for {symbol}: {str(e)}")
            return []
    
    def _calculate_signal_params(self, current_price: float, recent_data: pd.DataFrame, 
                                confidence: float) -> Tuple[float, float, float]:
        """Calculate entry, target, and stop loss prices"""
        try:
            # Simple calculation for backtesting
            # In real implementation, this would use the same logic as SignalDetector
            
            # Entry price (small pullback for non-high confidence signals)
            if confidence > 0.8:
                entry_price = current_price
            else:
                pullback_pct = 0.005  # 0.5% pullback
                entry_price = current_price * (1 - pullback_pct)
            
            # Stop loss based on ATR or percentage
            if 'atr' in recent_data.columns and not recent_data['atr'].empty:
                atr = recent_data['atr'].iloc[-1]
                atr_multiplier = 1.5 if confidence > 0.7 else 2.0
                stop_loss = entry_price - (atr * atr_multiplier)
            else:
                stop_loss_pct = config.MAX_STOP_LOSS
                stop_loss = entry_price * (1 - stop_loss_pct)
            
            # Target price based on risk-reward
            risk = entry_price - stop_loss
            if confidence > 0.8:
                risk_reward_ratio = 3.0
            elif confidence > 0.7:
                risk_reward_ratio = 2.5
            else:
                risk_reward_ratio = 2.0
            
            target_price = entry_price + (risk * risk_reward_ratio)
            
            # Apply minimum profit requirement
            min_target = entry_price * (1 + config.MIN_PROFIT_TARGET)
            target_price = max(target_price, min_target)
            
            return entry_price, target_price, stop_loss
            
        except Exception as e:
            logger.error(f"Error calculating signal parameters: {str(e)}")
            # Fallback to simple calculation
            entry_price = current_price
            target_price = current_price * (1 + config.MIN_PROFIT_TARGET)
            stop_loss = current_price * (1 - config.MAX_STOP_LOSS)
            return entry_price, target_price, stop_loss

    def _calculate_leverage(self, confidence: float, symbol: str) -> float:
        """
        Calculate optimal leverage based on confidence and risk
        """
        try:
            # Base leverage mapping based on confidence
            if confidence >= 0.95:
                base_leverage = 10.0  # Very high confidence
            elif confidence >= 0.85:
                base_leverage = 5.0   # High confidence
            elif confidence >= 0.75:
                base_leverage = 3.0   # Medium-high confidence
            elif confidence >= 0.65:
                base_leverage = 2.0   # Medium confidence
            else:
                base_leverage = 1.0   # Low confidence, no leverage
            
            # Symbol-specific leverage limits (based on volatility)
            symbol_limits = {
                'BTCUSDT': 10.0,    # Less volatile
                'ETHUSDT': 8.0,     # Medium volatile
                'BNBUSDT': 6.0,     # Medium volatile
                'ADAUSDT': 5.0,     # More volatile
                'SOLUSDT': 4.0,     # High volatile
                'DOTUSDT': 4.0,     # High volatile
                'LINKUSDT': 5.0,    # Medium volatile
                'MATICUSDT': 4.0,   # High volatile
                'LTCUSDT': 6.0,     # Medium volatile
                'TRXUSDT': 3.0      # High volatile
            }
            
            # Get symbol limit
            max_leverage = symbol_limits.get(symbol, 3.0)  # Default 3x for unknown symbols
            
            # Apply the lower of base leverage or symbol limit
            final_leverage = min(base_leverage, max_leverage)
            
            return final_leverage
            
        except Exception as e:
            logger.error(f"Error calculating leverage: {str(e)}")
            return 1.0  # Safe fallback
        """Calculate entry, target, and stop loss prices"""
        try:
            # Simple calculation for backtesting
            # In real implementation, this would use the same logic as SignalDetector
            
            # Entry price (small pullback for non-high confidence signals)
            if confidence > 0.8:
                entry_price = current_price
            else:
                pullback_pct = 0.005  # 0.5% pullback
                entry_price = current_price * (1 - pullback_pct)
            
            # Stop loss based on ATR or percentage
            if 'atr' in recent_data.columns and not recent_data['atr'].empty:
                atr = recent_data['atr'].iloc[-1]
                atr_multiplier = 1.5 if confidence > 0.7 else 2.0
                stop_loss = entry_price - (atr * atr_multiplier)
            else:
                stop_loss_pct = config.MAX_STOP_LOSS
                stop_loss = entry_price * (1 - stop_loss_pct)
            
            # Target price based on risk-reward
            risk = entry_price - stop_loss
            if confidence > 0.8:
                risk_reward_ratio = 3.0
            elif confidence > 0.7:
                risk_reward_ratio = 2.5
            else:
                risk_reward_ratio = 2.0
            
            target_price = entry_price + (risk * risk_reward_ratio)
            
            # Apply minimum profit requirement
            min_target = entry_price * (1 + config.MIN_PROFIT_TARGET)
            target_price = max(target_price, min_target)
            
            return entry_price, target_price, stop_loss
            
        except Exception as e:
            logger.error(f"Error calculating signal parameters: {str(e)}")
            # Fallback to simple calculation
            entry_price = current_price
            target_price = current_price * (1 + config.MIN_PROFIT_TARGET)
            stop_loss = current_price * (1 - config.MAX_STOP_LOSS)
            return entry_price, target_price, stop_loss
    
    def execute_signals(self, signals: List[BacktestSignal], price_data: pd.DataFrame) -> List[BacktestSignal]:
        """Execute signals and calculate results"""
        try:
            logger.info(f"Executing {len(signals)} signals...")
            
            executed_signals = []
            
            for signal in signals:
                try:
                    # Find the signal timestamp in price data
                    signal_idx = price_data.index.get_loc(signal.timestamp)
                    
                    # Look for exit conditions in future data (NO EXPIRY!)
                    # Signal stays active until target or stop is hit
                    
                    # Get ALL future price data (no expiry limit)
                    future_data = price_data.iloc[signal_idx+1:]
                    
                    # Check for exit conditions
                    exit_price = None
                    exit_timestamp = None
                    exit_reason = None
                    
                    for i, (timestamp, row) in enumerate(future_data.iterrows()):
                        # Check if target hit (exit with profit)
                        if row['high'] >= signal.target_price:
                            exit_price = signal.target_price
                            exit_timestamp = timestamp
                            exit_reason = 'TARGET_HIT'
                            break
                        
                        # Check if stop loss hit (exit with loss)
                        if row['low'] <= signal.stop_loss:
                            exit_price = signal.stop_loss
                            exit_timestamp = timestamp
                            exit_reason = 'STOP_HIT'
                            break
                    
                    # If no exit found by end of data, use last available price
                    # But mark as HOLDING (not expired!)
                    if exit_price is None:
                        exit_price = future_data.iloc[-1]['close']
                        exit_timestamp = future_data.index[-1]
                        if exit_price >= signal.target_price:
                            exit_reason = 'TARGET_HIT'
                        elif exit_price <= signal.stop_loss:
                            exit_reason = 'STOP_HIT'
                        else:
                            exit_reason = 'HOLDING'  # Still holding position
                    
                    # Calculate results
                    signal.exit_price = exit_price
                    signal.exit_timestamp = exit_timestamp
                    signal.exit_reason = exit_reason
                    signal.profit_loss = exit_price - signal.entry_price
                    signal.profit_loss_pct = (exit_price - signal.entry_price) / signal.entry_price
                    signal.leveraged_profit_loss_pct = signal.profit_loss_pct * signal.leverage
                    signal.duration_hours = (exit_timestamp - signal.timestamp).total_seconds() / 3600
                    
                    executed_signals.append(signal)
                    
                except Exception as e:
                    logger.error(f"Error executing signal: {str(e)}")
                    continue
            
            logger.info(f"Executed {len(executed_signals)} signals")
            return executed_signals
            
        except Exception as e:
            logger.error(f"Error executing signals: {str(e)}")
            return []
    
    def run_backtest(self, symbols: List[str] = None, days: int = 30, 
                    initial_capital: float = 20.0) -> Dict:
        """Run complete backtest"""
        try:
            if symbols is None:
                symbols = config.TRADING_PAIRS  # Test with first 5 pairs
            
            logger.info(f"Starting backtest: {len(symbols)} symbols, {days} days, ${initial_capital} capital")
            
            # Load model
            if not self.load_model():
                raise ValueError("Failed to load ML model")
            
            # Collect historical data
            historical_data = self.collect_historical_data(symbols, days)
            
            if not historical_data:
                raise ValueError("No historical data collected")
            
            # Generate and execute signals for each symbol
            all_signals = []
            
            for symbol, df in historical_data.items():
                # Generate signals
                signals = self.generate_signals(df, symbol)
                
                # Execute signals
                executed_signals = self.execute_signals(signals, df)
                
                all_signals.extend(executed_signals)
            
            # Calculate backtest results
            results = self._calculate_backtest_results(all_signals, initial_capital, days)
            
            logger.info(f"Backtest completed: {len(all_signals)} signals executed")
            return results
            
        except Exception as e:
            logger.error(f"Error running backtest: {str(e)}")
            return {}
    
    def _calculate_backtest_results(self, signals: List[BacktestSignal], initial_capital: float, days: int = 30) -> Dict:
        """Calculate comprehensive backtest results"""
        try:
            if not signals:
                return {'error': 'No signals to analyze'}
            
            # Basic statistics
            total_signals = len(signals)
            successful_signals = len([s for s in signals if s.exit_reason == 'TARGET_HIT'])
            failed_signals = len([s for s in signals if s.exit_reason == 'STOP_HIT'])
            holding_signals = len([s for s in signals if s.exit_reason == 'HOLDING'])
            
            success_rate = (successful_signals / total_signals) * 100 if total_signals > 0 else 0
            win_rate = (successful_signals / (successful_signals + failed_signals)) * 100 if (successful_signals + failed_signals) > 0 else 0
            
            # Portfolio simulation with compound interest (20% per trade)
            position_size_pct = 0.2
            
            # Normal trading simulation
            normal_capital = initial_capital
            for signal in signals:
                if signal.profit_loss_pct is not None:
                    trade_amount = normal_capital * position_size_pct
                    trade_return = trade_amount * signal.profit_loss_pct
                    normal_capital += trade_return
            
            # Leveraged trading simulation
            leveraged_capital = initial_capital
            for signal in signals:
                if signal.leveraged_profit_loss_pct is not None:
                    trade_amount = leveraged_capital * position_size_pct
                    trade_return = trade_amount * signal.leveraged_profit_loss_pct
                    leveraged_capital += trade_return
            
            # Calculate profits
            normal_profit = normal_capital - initial_capital
            leveraged_profit = leveraged_capital - initial_capital
            
            # Calculate ROI
            normal_roi = ((normal_capital / initial_capital) - 1) * 100
            leveraged_roi = ((leveraged_capital / initial_capital) - 1) * 100
            
            # Calculate annualized returns
            normal_monthly = ((normal_capital / initial_capital) ** (30/days) - 1) * 100 if days > 0 else 0
            normal_yearly = ((normal_capital / initial_capital) ** (365/days) - 1) * 100 if days > 0 else 0
            leveraged_monthly = ((leveraged_capital / initial_capital) ** (30/days) - 1) * 100 if days > 0 else 0
            leveraged_yearly = ((leveraged_capital / initial_capital) ** (365/days) - 1) * 100 if days > 0 else 0
            
            # Calculate scaling potential
            normal_multiplier = normal_capital / initial_capital
            leveraged_multiplier = leveraged_capital / initial_capital
            
            # Calculate averages
            winning_trades = [s.profit_loss_pct for s in signals if s.profit_loss_pct is not None and s.profit_loss_pct > 0]
            losing_trades = [s.profit_loss_pct for s in signals if s.profit_loss_pct is not None and s.profit_loss_pct < 0]
            leveraged_winning = [s.leveraged_profit_loss_pct for s in signals if s.leveraged_profit_loss_pct is not None and s.leveraged_profit_loss_pct > 0]
            leveraged_losing = [s.leveraged_profit_loss_pct for s in signals if s.leveraged_profit_loss_pct is not None and s.leveraged_profit_loss_pct < 0]
            
            avg_win = np.mean(winning_trades) * 100 if winning_trades else 0
            avg_loss = np.mean(losing_trades) * 100 if losing_trades else 0
            avg_leveraged_win = np.mean(leveraged_winning) * 100 if leveraged_winning else 0
            avg_leveraged_loss = np.mean(leveraged_losing) * 100 if leveraged_losing else 0
            
            # Leverage stats
            avg_leverage = np.mean([s.leverage for s in signals]) if signals else 0
            max_leverage = max([s.leverage for s in signals]) if signals else 0
            min_leverage = min([s.leverage for s in signals]) if signals else 0
            
            # Duration
            avg_duration = np.mean([s.duration_hours for s in signals if s.duration_hours is not None]) if signals else 0
            
            # Symbol performance
            symbol_stats = {}
            for symbol in set(s.symbol for s in signals):
                symbol_signals = [s for s in signals if s.symbol == symbol]
                symbol_success = len([s for s in symbol_signals if s.exit_reason == 'TARGET_HIT'])
                symbol_total = len(symbol_signals)
                symbol_stats[symbol] = {
                    'total_signals': symbol_total,
                    'successful': symbol_success,
                    'success_rate': (symbol_success / symbol_total * 100) if symbol_total > 0 else 0,
                    'total_profit_loss': sum(s.profit_loss for s in symbol_signals if s.profit_loss is not None)
                }
            
            # Build results
            results = {
                'backtest_period': f"{days} days",
                'initial_capital': initial_capital,
                
                'normal_trading': {
                    'final_capital': normal_capital,
                    'total_return': normal_profit,
                    'total_return_pct': normal_roi
                },
                
                'leveraged_trading': {
                    'final_capital': leveraged_capital,
                    'total_return': leveraged_profit,
                    'total_return_pct': leveraged_roi
                },
                
                'signal_stats': {
                    'total_signals': total_signals,
                    'successful_signals': successful_signals,
                    'failed_signals': failed_signals,
                    'holding_signals': holding_signals,
                    'success_rate': success_rate,
                    'win_rate': win_rate
                },
                
                'leverage_stats': {
                    'avg_leverage': avg_leverage,
                    'max_leverage': max_leverage,
                    'min_leverage': min_leverage
                },
                
                'profit_analysis': {
                    'position_size_pct': position_size_pct * 100,
                    'normal': {
                        'money_made': normal_profit,
                        'roi_total': normal_roi,
                        'roi_monthly': normal_monthly,
                        'roi_yearly': normal_yearly
                    },
                    'leveraged': {
                        'money_made': leveraged_profit,
                        'roi_total': leveraged_roi,
                        'roi_monthly': leveraged_monthly,
                        'roi_yearly': leveraged_yearly
                    },
                    'potential_earnings': {
                        'normal': {
                            '$100_start': normal_multiplier * 100,
                            '$1000_start': normal_multiplier * 1000,
                            '$10000_start': normal_multiplier * 10000
                        },
                        'leveraged': {
                            '$100_start': leveraged_multiplier * 100,
                            '$1000_start': leveraged_multiplier * 1000,
                            '$10000_start': leveraged_multiplier * 10000
                        }
                    }
                },
                
                'performance_metrics': {
                    'normal': {
                        'avg_win': avg_win,
                        'avg_loss': avg_loss,
                        'profit_factor': abs(sum(winning_trades) / sum(losing_trades)) if losing_trades else float('inf')
                    },
                    'leveraged': {
                        'avg_win': avg_leveraged_win,
                        'avg_loss': avg_leveraged_loss,
                        'profit_factor': abs(sum(leveraged_winning) / sum(leveraged_losing)) if leveraged_losing else float('inf')
                    },
                    'avg_duration_hours': avg_duration
                },
                
                'symbol_performance': symbol_stats,
                'detailed_signals': [s.to_dict() for s in signals]
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Error calculating backtest results: {str(e)}")
            return {'error': str(e)}
        
    def generate_backtest_report(self, results: Dict) -> str:
        """Generate human-readable backtest report"""
        try:
            if 'error' in results:
                return f"❌ Backtest Error: {results['error']}"
            
            report = f"""
📊 BACKTEST REPORT (COMPOUND TRADING)
{'='*50}

💰 FINANCIAL PERFORMANCE:
   Initial Capital: ${results['initial_capital']:.2f}
   Position Size: {results['profit_analysis']['position_size_pct']:.0f}% per trade
   
   📈 NORMAL TRADING (NO LEVERAGE):
   Final Capital: ${results['normal_trading']['final_capital']:.2f}
   Money Made: ${results['profit_analysis']['normal']['money_made']:.2f}
   Total ROI: {results['profit_analysis']['normal']['roi_total']:.2f}%
   Monthly ROI: {results['profit_analysis']['normal']['roi_monthly']:.2f}%
   Yearly ROI: {results['profit_analysis']['normal']['roi_yearly']:.2f}%
   
   🚀 LEVERAGED TRADING:
   Final Capital: ${results['leveraged_trading']['final_capital']:.2f}
   Money Made: ${results['profit_analysis']['leveraged']['money_made']:.2f}
   Total ROI: {results['profit_analysis']['leveraged']['roi_total']:.2f}%
   Monthly ROI: {results['profit_analysis']['leveraged']['roi_monthly']:.2f}%
   Yearly ROI: {results['profit_analysis']['leveraged']['roi_yearly']:.2f}%

📈 SIGNAL STATISTICS:
   Total Signals: {results['signal_stats']['total_signals']}
   Successful: {results['signal_stats']['successful_signals']} ({results['signal_stats']['success_rate']:.1f}%)
   Failed: {results['signal_stats']['failed_signals']}
   Still Holding: {results['signal_stats']['holding_signals']}
   Win Rate (Completed): {results['signal_stats']['win_rate']:.1f}%

💰 SCALING POTENTIAL (LEVERAGED):
   Start with $100 → End with ${results['profit_analysis']['potential_earnings']['leveraged']['$100_start']:.2f}
   Start with $1,000 → End with ${results['profit_analysis']['potential_earnings']['leveraged']['$1000_start']:.2f}
   Start with $10,000 → End with ${results['profit_analysis']['potential_earnings']['leveraged']['$10000_start']:.2f}

⚡ LEVERAGE STATISTICS:
   Average Leverage: {results['leverage_stats']['avg_leverage']:.1f}x
   Maximum Leverage: {results['leverage_stats']['max_leverage']:.1f}x
   Minimum Leverage: {results['leverage_stats']['min_leverage']:.1f}x

⚡ PERFORMANCE METRICS:
   📊 NORMAL TRADING:
   Average Win: {results['performance_metrics']['normal']['avg_win']:.2f}%
   Average Loss: {results['performance_metrics']['normal']['avg_loss']:.2f}%
   Sharpe Ratio: {results['performance_metrics']['normal']['sharpe_ratio']:.3f}
   Max Drawdown: ${results['performance_metrics']['normal']['max_drawdown']:.2f}
   Profit Factor: {results['performance_metrics']['normal']['profit_factor']:.2f}
   
   🚀 LEVERAGED TRADING:
   Average Win: {results['performance_metrics']['leveraged']['avg_win']:.2f}%
   Average Loss: {results['performance_metrics']['leveraged']['avg_loss']:.2f}%
   Sharpe Ratio: {results['performance_metrics']['leveraged']['sharpe_ratio']:.3f}
   Max Drawdown: ${results['performance_metrics']['leveraged']['max_drawdown']:.2f}
   Profit Factor: {results['performance_metrics']['leveraged']['profit_factor']:.2f}
   
   Average Duration: {results['performance_metrics']['avg_duration_hours']:.1f} hours

🪙 SYMBOL PERFORMANCE:
"""
            
            if 'symbol_performance' in results:
                for symbol, stats in results['symbol_performance'].items():
                    report += f"   {symbol}: {stats['total_signals']} signals, {stats['success_rate']:.1f}% success, ${stats['total_profit_loss']:.2f} P/L\n"
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating report: {str(e)}")
            return f"❌ Error generating report: {str(e)}"