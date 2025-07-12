#!/usr/bin/env python3
"""
Live Learning System - AI یادگیری زنده از بازار
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
import json
import os
from dataclasses import dataclass, asdict
import threading
import time
from collections import deque

logger = logging.getLogger(__name__)

@dataclass
class LiveSignalResult:
    """نتیجه سیگنال برای یادگیری"""
    signal_id: str
    symbol: str
    timestamp: datetime
    entry_price: float
    target_price: float
    stop_loss: float
    confidence: float
    
    # Features used for this signal
    features: Dict[str, float]
    
    # Results (filled later)
    actual_outcome: Optional[str] = None  # 'TARGET_HIT', 'STOP_HIT', 'EXPIRED'
    actual_return: Optional[float] = None
    max_profit: Optional[float] = None
    max_drawdown: Optional[float] = None
    exit_time: Optional[datetime] = None
    
    # Learning metrics
    was_correct: Optional[bool] = None
    confidence_accuracy: Optional[float] = None
    
    def to_dict(self):
        return asdict(self)

class LiveLearningSystem:
    """
    سیستم یادگیری زنده که از نتایج سیگنال‌ها یاد می‌گیره
    """
    
    def __init__(self, data_collector, feature_engineer, ml_model):
        self.data_collector = data_collector
        self.feature_engineer = feature_engineer
        self.ml_model = ml_model
        
        # Live tracking
        self.active_signals: Dict[str, LiveSignalResult] = {}
        self.completed_signals: deque = deque(maxlen=1000)
        
        # Learning database
        self.learning_data = []
        self.performance_history = []
        
        # 🚀 OPTIMIZED Adaptive thresholds (از بک‌تست)
        self.dynamic_threshold = 0.45  # شروع پایین‌تر
        self.min_threshold = 0.25      # حداقل کمتر
        self.max_threshold = 0.70      # حداکثر کمتر
        
        # 🚀 OPTIMIZED Learning settings (تندتر یاد بگیره)
        self.retrain_after_signals = 15  # کمتر از 30
        self.evaluation_window = 12      # کمتر از 24
        self.adaptation_sensitivity = 0.03  # بیشتر تغییر کنه
        
        # Create directories
        os.makedirs('data/live_learning', exist_ok=True)
        
        # Load previous learning data
        self.load_learning_history()
        
        # Start monitoring thread
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitor_signals, daemon=True)
        self.monitor_thread.start()
        
        logger.info("🚀 OPTIMIZED Live Learning System initialized")
        logger.info(f"   Initial threshold: {self.dynamic_threshold:.3f}")
        logger.info(f"   Learning sensitivity: {self.adaptation_sensitivity}")
    
    def register_signal(self, signal, features: pd.DataFrame) -> str:
        """ثبت سیگنال جدید برای tracking"""
        try:
            signal_id = f"{signal.symbol}_{int(signal.created_at.timestamp())}"
            
            # Extract features as dict
            features_dict = features.iloc[-1].to_dict()
            
            live_signal = LiveSignalResult(
                signal_id=signal_id,
                symbol=signal.symbol,
                timestamp=signal.created_at,
                entry_price=signal.entry_price,
                target_price=signal.target_price,
                stop_loss=signal.stop_loss,
                confidence=signal.confidence,
                features=features_dict
            )
            
            self.active_signals[signal_id] = live_signal
            
            logger.info(f"📊 Registered signal for live learning: {signal_id} (conf: {signal.confidence:.3f})")
            return signal_id
            
        except Exception as e:
            logger.error(f"Error registering signal: {str(e)}")
            return ""
    
    def _monitor_signals(self):
        """مانیتور کردن سیگنال‌های فعال"""
        while self.monitoring_active:
            try:
                if self.active_signals:
                    self._check_signal_outcomes()
                time.sleep(300)  # هر 5 دقیقه چک کن
                
            except Exception as e:
                logger.error(f"Error in signal monitoring: {str(e)}")
                time.sleep(60)
    
    def _check_signal_outcomes(self):
        """چک کردن نتایج سیگنال‌های فعال"""
        completed_count = 0
        
        for signal_id, signal in list(self.active_signals.items()):
            try:
                # Get current price
                current_price = self.data_collector.get_current_price(signal.symbol)
                if current_price is None:
                    continue
                
                # Check if signal completed
                outcome = self._evaluate_signal_outcome(signal, current_price)
                
                if outcome:
                    signal.actual_outcome = outcome['outcome']
                    signal.actual_return = outcome['return']
                    signal.max_profit = outcome['max_profit']
                    signal.max_drawdown = outcome['max_drawdown']
                    signal.exit_time = datetime.now()
                    signal.was_correct = outcome['was_correct']
                    signal.confidence_accuracy = outcome['confidence_accuracy']
                    
                    # Move to completed
                    self.completed_signals.append(signal)
                    del self.active_signals[signal_id]
                    completed_count += 1
                    
                    # Learn from this result
                    self._learn_from_signal(signal)
                    
                    logger.info(f"🎯 Signal completed: {signal.symbol} - {outcome['outcome']} (return: {outcome['return']*100:.1f}%)")
                
            except Exception as e:
                logger.error(f"Error checking signal {signal_id}: {str(e)}")
        
        if completed_count > 0:
            logger.info(f"✅ Processed {completed_count} completed signals")
    
    def _evaluate_signal_outcome(self, signal: LiveSignalResult, current_price: float) -> Optional[Dict]:
        """ارزیابی نتیجه سیگنال"""
        try:
            # Time elapsed
            time_elapsed = datetime.now() - signal.timestamp
            
            # Calculate returns
            actual_return = (current_price - signal.entry_price) / signal.entry_price
            
            # Check outcomes
            if current_price >= signal.target_price:
                outcome = 'TARGET_HIT'
                was_correct = True
                confidence_accuracy = signal.confidence
                
            elif current_price <= signal.stop_loss:
                outcome = 'STOP_HIT' 
                was_correct = False
                confidence_accuracy = 1.0 - signal.confidence
                
            elif time_elapsed.total_seconds() > self.evaluation_window * 3600:
                # Signal expired - evaluate based on actual return
                outcome = 'EXPIRED'
                was_correct = actual_return > 0.005  # بیش از 0.5% سود = موفق
                confidence_accuracy = signal.confidence if was_correct else (1.0 - signal.confidence)
                
            else:
                return None  # Still active
            
            # Calculate max profit/drawdown (simplified)
            max_profit = max(0, actual_return)
            max_drawdown = min(0, actual_return)
            
            return {
                'outcome': outcome,
                'return': actual_return,
                'max_profit': max_profit,
                'max_drawdown': max_drawdown,
                'was_correct': was_correct,
                'confidence_accuracy': confidence_accuracy
            }
            
        except Exception as e:
            logger.error(f"Error evaluating signal outcome: {str(e)}")
            return None
    
    def _learn_from_signal(self, signal: LiveSignalResult):
        """یادگیری از نتیجه سیگنال"""
        try:
            # Add to learning data
            learning_sample = {
                'features': signal.features,
                'predicted_confidence': signal.confidence,
                'actual_outcome': signal.was_correct,
                'actual_return': signal.actual_return,
                'timestamp': signal.timestamp.isoformat(),
                'symbol': signal.symbol
            }
            
            self.learning_data.append(learning_sample)
            
            # Update performance metrics
            self._update_performance_metrics()
            
            # Adapt threshold if needed
            self._adapt_threshold()
            
            # Check if we need retraining (disabled for now)
            # if len(self.learning_data) >= self.retrain_after_signals:
            #     self._trigger_incremental_learning()
            
            # Save learning data
            self._save_learning_data()
            
            logger.info(f"🧠 Learned from signal: {signal.symbol} ({'✅' if signal.was_correct else '❌'})")
            
        except Exception as e:
            logger.error(f"Error learning from signal: {str(e)}")
    
    def _update_performance_metrics(self):
        """به‌روزرسانی متریک‌های عملکرد"""
        try:
            if len(self.completed_signals) < 5:
                return
            
            recent_signals = list(self.completed_signals)[-20:]  # آخرین 20 سیگنال
            
            # Calculate metrics
            total_signals = len(recent_signals)
            correct_signals = sum(1 for s in recent_signals if s.was_correct)
            win_rate = correct_signals / total_signals
            
            avg_confidence = np.mean([s.confidence for s in recent_signals])
            avg_return = np.mean([s.actual_return for s in recent_signals if s.actual_return is not None])
            
            # Confidence calibration
            high_conf_signals = [s for s in recent_signals if s.confidence >= 0.65]
            high_conf_accuracy = np.mean([s.was_correct for s in high_conf_signals]) if high_conf_signals else 0
            
            performance = {
                'timestamp': datetime.now().isoformat(),
                'total_signals': total_signals,
                'win_rate': win_rate,
                'avg_confidence': avg_confidence,
                'avg_return': avg_return,
                'high_conf_accuracy': high_conf_accuracy,
                'current_threshold': self.dynamic_threshold
            }
            
            self.performance_history.append(performance)
            
            logger.info(f"📊 Performance updated: WR={win_rate:.1%}, AvgConf={avg_confidence:.3f}, Threshold={self.dynamic_threshold:.3f}")
            
        except Exception as e:
            logger.error(f"Error updating performance: {str(e)}")
    
    def _adapt_threshold(self):
        """🚀 تطبیق بهینه‌شده آستانه اطمینان"""
        try:
            if len(self.performance_history) < 2:
                return
            
            # استفاده از آخرین 2 نتیجه برای تصمیم‌گیری سریع‌تر
            recent_performance = self.performance_history[-2:]
            avg_win_rate = np.mean([p['win_rate'] for p in recent_performance])
            
            old_threshold = self.dynamic_threshold
            
            # 🚀 منطق تطبیق بهینه‌شده (حساس‌تر)
            if avg_win_rate < 0.45:  # کاهش از 0.5
                # آستانه رو بالا ببر (محافظه‌کارتر باش)
                self.dynamic_threshold = min(self.max_threshold, self.dynamic_threshold + self.adaptation_sensitivity)
                change_reason = "low_win_rate"
                
            elif avg_win_rate > 0.70:  # کاهش از 0.75
                # آستانه رو پایین بیار (بیشتر فرصت بگیر)
                self.dynamic_threshold = max(self.min_threshold, self.dynamic_threshold - (self.adaptation_sensitivity * 0.7))
                change_reason = "high_win_rate"
            else:
                return  # No change needed
            
            if old_threshold != self.dynamic_threshold:
                direction = "🔺" if self.dynamic_threshold > old_threshold else "🔻"
                logger.info(f"{direction} OPTIMIZED Threshold: {old_threshold:.3f} → {self.dynamic_threshold:.3f} ({change_reason})")
            
        except Exception as e:
            logger.error(f"Error adapting threshold: {str(e)}")
    
    def get_adaptive_threshold(self) -> float:
        """گرفتن آستانه تطبیقی فعلی"""
        return self.dynamic_threshold
    
    def get_live_performance(self) -> Dict:
        """گرفتن عملکرد زنده"""
        try:
            if not self.performance_history:
                return {
                    'status': 'initializing',
                    'current_threshold': self.dynamic_threshold,
                    'active_signals': len(self.active_signals),
                    'total_completed': len(self.completed_signals)
                }
            
            latest = self.performance_history[-1]
            
            return {
                'status': 'active',
                'current_threshold': self.dynamic_threshold,
                'recent_win_rate': latest['win_rate'],
                'avg_confidence': latest['avg_confidence'],
                'avg_return': latest.get('avg_return', 0),
                'high_conf_accuracy': latest['high_conf_accuracy'],
                'total_signals_learned': len(self.completed_signals),
                'active_signals': len(self.active_signals),
                'learning_samples': len(self.learning_data),
                'last_update': latest['timestamp']
            }
            
        except Exception as e:
            logger.error(f"Error getting live performance: {str(e)}")
            return {'status': 'error', 'error': str(e)}
    
    def _save_learning_data(self):
        """ذخیره داده‌های یادگیری"""
        try:
            # Save learning samples
            learning_file = 'data/live_learning/learning_data.json'
            with open(learning_file, 'w') as f:
                json.dump(self.learning_data[-50:], f, default=str, indent=2)  # آخرین 50 نمونه
            
            # Save performance history
            performance_file = 'data/live_learning/performance_history.json'
            with open(performance_file, 'w') as f:
                json.dump(self.performance_history[-20:], f, default=str, indent=2)  # آخرین 20 متریک
            
            # Save completed signals summary
            if self.completed_signals:
                completed_file = 'data/live_learning/completed_signals.json'
                completed_data = [s.to_dict() for s in list(self.completed_signals)[-20:]]
                with open(completed_file, 'w') as f:
                    json.dump(completed_data, f, default=str, indent=2)
            
        except Exception as e:
            logger.error(f"Error saving learning data: {str(e)}")
    
    def load_learning_history(self):
        """بارگذاری تاریخچه یادگیری"""
        try:
            # Load learning data
            learning_file = 'data/live_learning/learning_data.json'
            if os.path.exists(learning_file):
                with open(learning_file, 'r') as f:
                    self.learning_data = json.load(f)
                logger.info(f"📚 Loaded {len(self.learning_data)} learning samples")
            
            # Load performance history
            performance_file = 'data/live_learning/performance_history.json'
            if os.path.exists(performance_file):
                with open(performance_file, 'r') as f:
                    self.performance_history = json.load(f)
                logger.info(f"📊 Loaded {len(self.performance_history)} performance records")
                
                # Restore adaptive threshold
                if self.performance_history:
                    last_threshold = self.performance_history[-1].get('current_threshold', 0.55)
                    self.dynamic_threshold = max(self.min_threshold, min(self.max_threshold, last_threshold))
                    logger.info(f"🎯 Restored adaptive threshold: {self.dynamic_threshold:.3f}")
            
            # Load completed signals
            completed_file = 'data/live_learning/completed_signals.json'
            if os.path.exists(completed_file):
                with open(completed_file, 'r') as f:
                    completed_data = json.load(f)
                    for signal_data in completed_data:
                        # Convert back to LiveSignalResult
                        signal_data['timestamp'] = datetime.fromisoformat(signal_data['timestamp'])
                        if signal_data.get('exit_time'):
                            signal_data['exit_time'] = datetime.fromisoformat(signal_data['exit_time'])
                        signal = LiveSignalResult(**signal_data)
                        self.completed_signals.append(signal)
                logger.info(f"🏁 Loaded {len(self.completed_signals)} completed signals")
            
        except Exception as e:
            logger.error(f"Error loading learning history: {str(e)}")
    
    def get_learning_summary(self) -> str:
        """گزارش خلاصه یادگیری"""
        try:
            performance = self.get_live_performance()
            
            if performance['status'] == 'initializing':
                return f"""
🧠 LIVE LEARNING STATUS: Initializing
📊 Current Threshold: {performance['current_threshold']:.3f}
⚡ Active Signals: {performance['active_signals']}
🏁 Completed Signals: {performance['total_completed']}
"""
            
            return f"""
🧠 LIVE LEARNING STATUS: Active
========================
🎯 Adaptive Threshold: {performance['current_threshold']:.3f}
📈 Recent Win Rate: {performance['recent_win_rate']*100:.1f}%
🔥 Avg Confidence: {performance['avg_confidence']:.3f}
💰 Avg Return: {performance['avg_return']*100:.2f}%
⚡ Active Signals: {performance['active_signals']}
🏁 Total Learned: {performance['total_signals_learned']}
📚 Learning Samples: {performance['learning_samples']}
"""
        except Exception as e:
            return f"❌ Error generating summary: {str(e)}"
    
    def stop_monitoring(self):
        """توقف monitoring"""
        self.monitoring_active = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5)
        logger.info("🛑 Live learning monitoring stopped")