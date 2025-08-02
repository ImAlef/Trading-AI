import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass
from config import config
from backend.services.fallback_data_collector import FallbackDataCollector
from backend.services.feature_engineer import FeatureEngineer
from backend.services.ml_model import MLModel
from backend.services.live_learning import LiveLearningSystem

logger = logging.getLogger(__name__)

@dataclass
class TradingSignal:
    """
    Data class for trading signals
    """
    symbol: str
    signal_type: str  # 'BUY' or 'SELL'
    entry_price: float
    target_price: float
    stop_loss: float
    confidence: float
    created_at: datetime
    expires_at: datetime
    
    # Technical analysis data
    rsi: float
    macd: float
    macd_signal: float
    bb_upper: float
    bb_lower: float
    volume_ratio: float
    
    # Model information
    model_version: str
    features_used: int
    timeframe: str
    
    def to_dict(self) -> Dict:
        """Convert signal to dictionary"""
        return {
            'symbol': self.symbol,
            'signal_type': self.signal_type,
            'entry_price': self.entry_price,
            'target_price': self.target_price,
            'stop_loss': self.stop_loss,
            'confidence': self.confidence,
            'created_at': self.created_at.isoformat(),
            'expires_at': self.expires_at.isoformat(),
            'rsi': self.rsi,
            'macd': self.macd,
            'macd_signal': self.macd_signal,
            'bb_upper': self.bb_upper,
            'bb_lower': self.bb_lower,
            'volume_ratio': self.volume_ratio,
            'model_version': self.model_version,
            'features_used': self.features_used,
            'timeframe': self.timeframe
        }
    
    def get_profit_potential(self) -> float:
        """Calculate potential profit percentage for LONG and SHORT"""
        if self.signal_type == 'BUY':
            return (self.target_price - self.entry_price) / self.entry_price
        else:  # SELL (SHORT)
            return (self.entry_price - self.target_price) / self.entry_price
    
    def get_risk_ratio(self) -> float:
        """Calculate risk-to-reward ratio for LONG and SHORT"""
        if self.signal_type == 'BUY':
            risk = abs(self.entry_price - self.stop_loss) / self.entry_price
            reward = self.get_profit_potential()
        else:  # SELL (SHORT)
            risk = abs(self.stop_loss - self.entry_price) / self.entry_price
            reward = self.get_profit_potential()
        
        return reward / risk if risk > 0 else 0
    
    def get_signal_direction_emoji(self) -> str:
        """Get emoji for signal direction"""
        return "📈" if self.signal_type == 'BUY' else "📉"
    
    def get_detailed_info(self) -> str:
        """Get detailed signal information"""
        direction = "LONG" if self.signal_type == 'BUY' else "SHORT"
        emoji = self.get_signal_direction_emoji()
        
        return f"""
{emoji} {direction} Signal: {self.symbol}
Entry: ${self.entry_price:.6f}
Target: ${self.target_price:.6f} ({self.get_profit_potential()*100:+.2f}%)
Stop: ${self.stop_loss:.6f}
R/R: {self.get_risk_ratio():.2f}
Confidence: {self.confidence:.1%}
"""

class SignalDetector:
    """
    Main signal detection system با قابلیت یادگیری زنده
    """
    
    def __init__(self):
        self.data_collector = FallbackDataCollector()
        self.feature_engineer = FeatureEngineer()
        self.ml_model = MLModel()
        self.active_signals: List[TradingSignal] = []
        
        # Live Learning System
        self.live_learning: Optional[LiveLearningSystem] = None
        
    def load_model(self, model_path: str = None) -> bool:
        """Load the ML model"""
        try:
            if model_path:
                success = self.ml_model.load_model(model_path)
            else:
                # Try to load the latest model
                success = self.ml_model.load_latest_model()
            
            if success:
                logger.info(f"Model loaded successfully: {self.ml_model.model_info['version']}")
                
                # Initialize live learning system
                self.initialize_live_learning()
                return True
            else:
                logger.error("Failed to load ML model")
                return False
                
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False
    
    def initialize_live_learning(self) -> bool:
        """راه‌اندازی سیستم یادگیری زنده"""
        try:
            self.live_learning = LiveLearningSystem(
                self.data_collector,
                self.feature_engineer, 
                self.ml_model
            )
            logger.info("🧠 Live learning system initialized!")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize live learning: {str(e)}")
            return False
    
    def analyze_symbol(self, symbol: str, timeframe: str = "1h") -> Optional[TradingSignal]:
        """
        Analyze a single symbol and generate signal if conditions are met (LONG or SHORT)
        """
        try:
            logger.info(f"Analyzing {symbol} on {timeframe} timeframe...")
            
            # Step 1: Get market data
            df = self.data_collector.get_klines(symbol, timeframe, limit=config.LOOKBACK_PERIODS + 50)
            
            if df.empty:
                logger.warning(f"No data available for {symbol}")
                return None
            
            # Step 2: Feature engineering (for prediction, not training)
            df_features = self.feature_engineer.prepare_features_for_prediction(df)
            
            if df_features.empty:
                logger.warning(f"Feature engineering failed for {symbol}")
                return None
            
            # Step 3: Get latest features for prediction
            latest_features = df_features.iloc[-1:].copy()
            
            # Step 4: Make prediction
            prediction, confidence = self.ml_model.predict_single(latest_features)
            
            # Step 5: Use adaptive threshold if live learning is active
            threshold = config.CONFIDENCE_THRESHOLD
            if self.live_learning:
                threshold = self.live_learning.get_adaptive_threshold()
            
            # Step 6: Determine signal type based on market conditions
            signal_type = self._determine_signal_type(latest_features.iloc[-1])
            
            logger.info(f"{symbol}: Prediction={prediction}, Confidence={confidence:.3f}, Type={signal_type}, Threshold={threshold:.3f}")
            
            # Step 7: Check if signal meets criteria
            if prediction == 1 and confidence >= threshold:
                
                # Get current market data
                current_price = latest_features['close'].iloc[-1]
                
                # Calculate signal parameters
                signal = self._create_signal(
                    symbol=symbol,
                    current_price=current_price,
                    confidence=confidence,
                    features=latest_features,
                    timeframe=timeframe,
                    signal_type=signal_type
                )
                
                # Additional filters
                if self._validate_signal(signal):
                    # Register signal for live learning
                    if self.live_learning:
                        signal_id = self.live_learning.register_signal(signal, latest_features)
                        logger.info(f"🧠 Signal registered for live learning: {signal_id}")
                    
                    logger.info(f"✅ Valid {signal_type} signal generated for {symbol}")
                    return signal
                else:
                    logger.info(f"❌ Signal filtered out for {symbol}")
                    return None
            
            return None
            
        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {str(e)}")
            return None
        
    def _determine_signal_type(self, latest_row: pd.Series) -> str:
        """
        تشخیص نوع سیگنال بر اساس شرایط بازار
        """
        try:
            # Get technical indicators
            rsi = latest_row.get('rsi', 50)
            price_vs_ema9 = latest_row.get('price_vs_ema9', 0)
            price_vs_ema21 = latest_row.get('price_vs_ema21', 0)
            bb_position = latest_row.get('bb_position', 0.5)
            macd_histogram = latest_row.get('macd_histogram', 0)
            
            # Count bullish and bearish signals
            bullish_signals = 0
            bearish_signals = 0
            
            # RSI analysis
            if rsi < 40:  # Oversold - bullish
                bullish_signals += 1
            elif rsi > 70:  # Overbought - bearish
                bearish_signals += 1
            
            # Moving average trends
            if price_vs_ema9 > 0 and price_vs_ema21 > 0:  # Price above MAs - bullish
                bullish_signals += 1
            elif price_vs_ema9 < 0 and price_vs_ema21 < 0:  # Price below MAs - bearish
                bearish_signals += 1
            
            # Bollinger Bands position
            if bb_position < 0.2:  # Near lower band - bullish
                bullish_signals += 1
            elif bb_position > 0.8:  # Near upper band - bearish
                bearish_signals += 1
            
            # MACD momentum
            if macd_histogram > 0:  # Positive momentum - bullish
                bullish_signals += 1
            elif macd_histogram < 0:  # Negative momentum - bearish
                bearish_signals += 1
            
            # Determine signal type
            if bearish_signals > bullish_signals and bearish_signals >= 2:
                return 'SELL'  # SHORT signal
            else:
                return 'BUY'   # LONG signal (default)
            
        except Exception as e:
            logger.error(f"Error determining signal type: {str(e)}")
            return 'BUY'  # Default to LONG    
    
    def _create_signal(self, symbol: str, current_price: float, confidence: float, 
                  features: pd.DataFrame, timeframe: str, signal_type: str = 'BUY') -> TradingSignal:
        """
        Create a trading signal with dynamically calculated parameters (LONG or SHORT)
        """
        # Get latest market data for better calculations
        latest_row = features.iloc[-1]
        
        # Calculate ATR for volatility-based stops
        atr = latest_row.get('atr', current_price * 0.02)  # Default 2% if ATR not available
        
        # Calculate support/resistance levels
        support_level = self._calculate_support_level(features, current_price)
        resistance_level = self._calculate_resistance_level(features, current_price)
        
        # Dynamic entry strategy based on confidence and technical indicators
        entry_price = self._calculate_entry_price(current_price, latest_row, confidence, signal_type)
        
        # Dynamic stop loss and take profit based on signal type
        if signal_type == 'BUY':
            stop_loss = self._calculate_stop_loss_long(entry_price, atr, support_level, latest_row, confidence)
            target_price = self._calculate_take_profit_long(entry_price, stop_loss, resistance_level, latest_row, confidence)
        else:  # SELL (SHORT)
            stop_loss = self._calculate_stop_loss_short(entry_price, atr, resistance_level, latest_row, confidence)
            target_price = self._calculate_take_profit_short(entry_price, stop_loss, support_level, latest_row, confidence)
        
        # 📊 Log detailed price information with 6 decimal precision
        logger.info(f"💰 {signal_type} Signal Price Details for {symbol}:")
        logger.info(f"   Current Price: ${current_price:.6f}")
        logger.info(f"   Entry Price: ${entry_price:.6f}")
        logger.info(f"   Target Price: ${target_price:.6f}")
        logger.info(f"   Stop Loss: ${stop_loss:.6f}")
        
        if signal_type == 'BUY':
            risk_pct = ((entry_price - stop_loss) / entry_price * 100)
            reward_pct = ((target_price - entry_price) / entry_price * 100)
        else:  # SELL
            risk_pct = ((stop_loss - entry_price) / entry_price * 100)
            reward_pct = ((entry_price - target_price) / entry_price * 100)
        
        logger.info(f"   Risk: {risk_pct:.3f}%")
        logger.info(f"   Reward: {reward_pct:.3f}%")
        logger.info(f"   R/R Ratio: {(reward_pct / risk_pct):.2f}")
        
        # Create signal
        signal = TradingSignal(
            symbol=symbol,
            signal_type=signal_type,
            entry_price=round(entry_price, 6),  # 6 decimal places precision
            target_price=round(target_price, 6),
            stop_loss=round(stop_loss, 6),
            confidence=confidence,
            created_at=datetime.now(),
            expires_at=datetime.now() + timedelta(hours=config.SIGNAL_EXPIRY_HOURS),
            rsi=latest_row.get('rsi', 0),
            macd=latest_row.get('macd', 0),
            macd_signal=latest_row.get('macd_signal', 0),
            bb_upper=latest_row.get('bb_upper', 0),
            bb_lower=latest_row.get('bb_lower', 0),
            volume_ratio=latest_row.get('volume_ratio', 1),
            model_version=self.ml_model.model_info.get('version', 'unknown'),
            features_used=len(self.feature_engineer.get_feature_columns()),
            timeframe=timeframe
        )
        
        return signal
    
    def _calculate_stop_loss_long(self, entry_price: float, atr: float, support_level: float, 
                             latest_row: pd.Series, confidence: float) -> float:
        """🚀 محاسبه stop loss بهینه‌شده برای LONG با کنترل ایمنی"""
        try:
            # ATR-based stop loss بهینه‌شده
            atr_multiplier = 1.2 if confidence > 0.7 else 1.5
            atr_stop = entry_price - (atr * atr_multiplier)
            
            # Support-based stop loss
            support_buffer = entry_price * 0.003
            support_stop = support_level - support_buffer
            
            # استفاده از بالاتر (کمتر aggressive)
            calculated_stop = max(atr_stop, support_stop)
            
            # حداکثر ضرر بهینه‌شده
            max_loss_pct = config.MAX_STOP_LOSS
            max_stop = entry_price * (1 - max_loss_pct)
            
            # استفاده از بالاتر (کمتر ریسک)
            preliminary_stop = max(calculated_stop, max_stop)
            
            # 🛡️ SAFETY CHECK: Stop loss نباید بالاتر از entry price باشه
            if preliminary_stop >= entry_price:
                logger.warning(f"⚠️ LONG Stop loss {preliminary_stop:.6f} >= entry price {entry_price:.6f}, using safe fallback")
                safe_stop = entry_price * (1 - max(config.MAX_STOP_LOSS, 0.01))
                return safe_stop
            
            return preliminary_stop
            
        except Exception as e:
            logger.error(f"Error calculating LONG stop loss: {str(e)}")
            safe_stop = entry_price * (1 - config.MAX_STOP_LOSS)
            return safe_stop

    def _calculate_stop_loss_short(self, entry_price: float, atr: float, resistance_level: float, 
                                latest_row: pd.Series, confidence: float) -> float:
        """🚀 محاسبه stop loss بهینه‌شده برای SHORT با کنترل ایمنی"""
        try:
            # ATR-based stop loss بهینه‌شده
            atr_multiplier = 1.2 if confidence > 0.7 else 1.5
            atr_stop = entry_price + (atr * atr_multiplier)
            
            # Resistance-based stop loss
            resistance_buffer = entry_price * 0.003
            resistance_stop = resistance_level + resistance_buffer
            
            # استفاده از کمتر (کمتر aggressive)
            calculated_stop = min(atr_stop, resistance_stop)
            
            # حداکثر ضرر بهینه‌شده
            max_loss_pct = config.MAX_STOP_LOSS
            max_stop = entry_price * (1 + max_loss_pct)
            
            # استفاده از کمتر (کمتر ریسک)
            preliminary_stop = min(calculated_stop, max_stop)
            
            # 🛡️ SAFETY CHECK: Stop loss نباید پایین‌تر از entry price باشه
            if preliminary_stop <= entry_price:
                logger.warning(f"⚠️ SHORT Stop loss {preliminary_stop:.6f} <= entry price {entry_price:.6f}, using safe fallback")
                safe_stop = entry_price * (1 + max(config.MAX_STOP_LOSS, 0.01))
                return safe_stop
            
            return preliminary_stop
            
        except Exception as e:
            logger.error(f"Error calculating SHORT stop loss: {str(e)}")
            safe_stop = entry_price * (1 + config.MAX_STOP_LOSS)
            return safe_stop

    def _calculate_take_profit_long(self, entry_price: float, stop_loss: float, resistance_level: float,
                                latest_row: pd.Series, confidence: float) -> float:
        """🚀 محاسبه take profit بهینه‌شده برای LONG"""
        try:
            # محاسبه ریسک
            risk = entry_price - stop_loss
            
            # نسبت ریسک-سود بهینه‌شده
            if confidence > 0.8:
                risk_reward_ratio = 4.0
            elif confidence > 0.7:
                risk_reward_ratio = 3.5
            elif confidence > 0.6:
                risk_reward_ratio = 3.0
            else:
                risk_reward_ratio = 2.5
            
            # محاسبه target بر اساس ریسک-سود
            risk_reward_target = entry_price + (risk * risk_reward_ratio)
            
            # محاسبه target بر اساس resistance
            resistance_buffer = entry_price * 0.003
            resistance_target = resistance_level - resistance_buffer
            
            # استفاده از کمتر (محافظه‌کارانه‌تر)
            calculated_target = min(risk_reward_target, resistance_target)
            
            # حداقل سود بهینه‌شده
            min_profit_pct = config.MIN_PROFIT_TARGET
            min_target = entry_price * (1 + min_profit_pct)
            
            # استفاده از بالاتر (حداقل سود)
            final_target = max(calculated_target, min_target)
            
            return final_target
            
        except Exception as e:
            logger.error(f"Error calculating LONG take profit: {str(e)}")
            return entry_price * (1 + config.MIN_PROFIT_TARGET)

    def _calculate_take_profit_short(self, entry_price: float, stop_loss: float, support_level: float,
                                    latest_row: pd.Series, confidence: float) -> float:
        """🚀 محاسبه take profit بهینه‌شده برای SHORT"""
        try:
            # محاسبه ریسک
            risk = stop_loss - entry_price
            
            # نسبت ریسک-سود بهینه‌شده
            if confidence > 0.8:
                risk_reward_ratio = 4.0
            elif confidence > 0.7:
                risk_reward_ratio = 3.5
            elif confidence > 0.6:
                risk_reward_ratio = 3.0
            else:
                risk_reward_ratio = 2.5
            
            # محاسبه target بر اساس ریسک-سود
            risk_reward_target = entry_price - (risk * risk_reward_ratio)
            
            # محاسبه target بر اساس support
            support_buffer = entry_price * 0.003
            support_target = support_level + support_buffer
            
            # استفاده از بالاتر (محافظه‌کارانه‌تر)
            calculated_target = max(risk_reward_target, support_target)
            
            # حداقل سود بهینه‌شده
            min_profit_pct = config.MIN_PROFIT_TARGET
            min_target = entry_price * (1 - min_profit_pct)
            
            # استفاده از کمتر (حداقل سود)
            final_target = min(calculated_target, min_target)
            
            return final_target
            
        except Exception as e:
            logger.error(f"Error calculating SHORT take profit: {str(e)}")
            return entry_price * (1 - config.MIN_PROFIT_TARGET)
    def _calculate_support_level(self, features: pd.DataFrame, current_price: float) -> float:
        """Calculate dynamic support level"""
        try:
            # Get recent lows
            recent_lows = features['low'].tail(20)
            
            # Calculate support as recent swing low
            support = recent_lows.min()
            
            # Make sure support is reasonable (not too far from current price)
            max_support_distance = current_price * 0.05  # Max 5% below current price
            support = max(support, current_price - max_support_distance)
            
            return support
            
        except Exception as e:
            logger.error(f"Error calculating support level: {str(e)}")
            return current_price * 0.98  # Default 2% below current price
    
    def _calculate_resistance_level(self, features: pd.DataFrame, current_price: float) -> float:
        """Calculate dynamic resistance level"""
        try:
            # Get recent highs
            recent_highs = features['high'].tail(20)
            
            # Calculate resistance as recent swing high
            resistance = recent_highs.max()
            
            # Make sure resistance is reasonable (not too far from current price)
            min_resistance_distance = current_price * 0.02  # Min 2% above current price
            resistance = max(resistance, current_price + min_resistance_distance)
            
            return resistance
            
        except Exception as e:
            logger.error(f"Error calculating resistance level: {str(e)}")
            return current_price * 1.03  # Default 3% above current price
    
    def _calculate_entry_price(self, current_price: float, latest_row: pd.Series, 
                          confidence: float, signal_type: str) -> float:
        """🚀 محاسبه قیمت ورود بهینه‌شده برای LONG و SHORT"""
        try:
            if signal_type == 'BUY':
                # LONG: Wait for small pullback
                if confidence > 0.7:
                    return current_price  # ورود فوری برای confidence بالا
                elif confidence > 0.6:
                    pullback_pct = 0.003  # 0.3% pullback کوچک
                    return current_price * (1 - pullback_pct)
                else:
                    pullback_pct = 0.007  # 0.7% pullback متوسط
                    return current_price * (1 - pullback_pct)
            else:  # SELL (SHORT)
                # SHORT: Wait for small bounce
                if confidence > 0.7:
                    return current_price  # ورود فوری برای confidence بالا
                elif confidence > 0.6:
                    bounce_pct = 0.003  # 0.3% bounce کوچک
                    return current_price * (1 + bounce_pct)
                else:
                    bounce_pct = 0.007  # 0.7% bounce متوسط
                    return current_price * (1 + bounce_pct)
            
        except Exception as e:
            logger.error(f"Error calculating optimized entry price: {str(e)}")
            return current_price
    
    def _calculate_stop_loss(self, entry_price: float, atr: float, support_level: float, 
                       latest_row: pd.Series, confidence: float) -> float:
        """🚀 محاسبه stop loss بهینه‌شده با کنترل ایمنی"""
        try:
            # ATR-based stop loss بهینه‌شده
            atr_multiplier = 1.2 if confidence > 0.7 else 1.5  # تنگ‌تر
            atr_stop = entry_price - (atr * atr_multiplier)
            
            # Support-based stop loss
            support_buffer = entry_price * 0.003  # کمتر از 0.005
            support_stop = support_level - support_buffer
            
            # استفاده از بالاتر (کمتر aggressive)
            calculated_stop = max(atr_stop, support_stop)
            
            # 🚀 حداکثر ضرر بهینه‌شده
            max_loss_pct = config.MAX_STOP_LOSS  # 1.2%
            max_stop = entry_price * (1 - max_loss_pct)
            
            # استفاده از بالاتر (کمتر ریسک)
            preliminary_stop = max(calculated_stop, max_stop)
            
            # 🛡️ SAFETY CHECK: Stop loss نباید بالاتر از entry price باشه
            if preliminary_stop >= entry_price:
                logger.warning(f"⚠️ Stop loss {preliminary_stop:.6f} >= entry price {entry_price:.6f}, using safe fallback")
                # Use a safe percentage below entry price
                safe_stop = entry_price * (1 - max(config.MAX_STOP_LOSS, 0.01))  # حداقل 1% زیر قیمت ورود
                return safe_stop
            
            return preliminary_stop
            
        except Exception as e:
            logger.error(f"Error calculating optimized stop loss: {str(e)}")
            # Safe fallback
            safe_stop = entry_price * (1 - config.MAX_STOP_LOSS)
            return safe_stop
    
    def _calculate_take_profit(self, entry_price: float, stop_loss: float, resistance_level: float,
                              latest_row: pd.Series, confidence: float) -> float:
        """🚀 محاسبه take profit بهینه‌شده"""
        try:
            # محاسبه ریسک
            risk = entry_price - stop_loss
            
            # 🚀 نسبت ریسک-سود بهینه‌شده
            if confidence > 0.8:
                risk_reward_ratio = 4.0  # بالاتر از 3.0
            elif confidence > 0.7:
                risk_reward_ratio = 3.5  # بالاتر از 2.5
            elif confidence > 0.6:
                risk_reward_ratio = 3.0  # بالاتر از 2.0
            else:
                risk_reward_ratio = 2.5  # بالاتر از 2.0
            
            # محاسبه target بر اساس ریسک-سود
            risk_reward_target = entry_price + (risk * risk_reward_ratio)
            
            # محاسبه target بر اساس resistance
            resistance_buffer = entry_price * 0.003  # کمتر از 0.005
            resistance_target = resistance_level - resistance_buffer
            
            # استفاده از کمتر (محافظه‌کارانه‌تر)
            calculated_target = min(risk_reward_target, resistance_target)
            
            # 🚀 حداقل سود بهینه‌شده
            min_profit_pct = config.MIN_PROFIT_TARGET  # 1.5%
            min_target = entry_price * (1 + min_profit_pct)
            
            # استفاده از بالاتر (حداقل سود)
            final_target = max(calculated_target, min_target)
            
            return final_target
            
        except Exception as e:
            logger.error(f"Error calculating optimized take profit: {str(e)}")
            return entry_price * (1 + config.MIN_PROFIT_TARGET)
    
    def _validate_signal(self, signal: TradingSignal) -> bool:
        """🚀 اعتبارسنجی سیگنال بهینه‌شده برای LONG و SHORT"""
        try:
            # Calculate risk-reward ratio based on signal type
            if signal.signal_type == 'BUY':
                risk = abs(signal.entry_price - signal.stop_loss) / signal.entry_price
                reward = abs(signal.target_price - signal.entry_price) / signal.entry_price
            else:  # SELL (SHORT)
                risk = abs(signal.stop_loss - signal.entry_price) / signal.entry_price
                reward = abs(signal.entry_price - signal.target_price) / signal.entry_price
            
            risk_reward = reward / risk if risk > 0 else 0
            
            # 🚀 فیلتر 1: نسبت ریسک-سود کمتر
            if risk_reward < config.SIGNAL_VALIDATION["min_risk_reward"]:  # 1.2
                logger.info(f"Signal filtered: Poor risk-reward ratio ({risk_reward:.2f})")
                return False
            
            # 🚀 فیلتر 2: RSI checks based on signal type
            if signal.signal_type == 'BUY':
                # For LONG: Don't buy when extremely overbought
                if signal.rsi > config.SIGNAL_VALIDATION["max_rsi_overbought"]:  # 90
                    logger.info(f"Signal filtered: RSI too high for LONG ({signal.rsi:.1f})")
                    return False
            else:  # SELL (SHORT)
                # For SHORT: Don't sell when extremely oversold
                if signal.rsi < (100 - config.SIGNAL_VALIDATION["max_rsi_overbought"]):  # 10
                    logger.info(f"Signal filtered: RSI too low for SHORT ({signal.rsi:.1f})")
                    return False
            
            # 🚀 فیلتر 3: Volume کمتر سخت‌گیری
            if signal.volume_ratio < config.SIGNAL_VALIDATION["min_volume_ratio"]:  # 0.6
                logger.info(f"Signal filtered: Low volume ({signal.volume_ratio:.2f})")
                return False
            
            # فیلتر 4: تکراری نبودن
            if self._is_duplicate_signal(signal):
                logger.info(f"Signal filtered: Duplicate signal for {signal.symbol}")
                return False
            
            # 🚀 فیلتر 5: Safety check for price relationships
            if signal.signal_type == 'BUY':
                if signal.stop_loss >= signal.entry_price:
                    logger.error(f"❌ LONG Signal validation failed: Stop loss {signal.stop_loss:.6f} >= Entry {signal.entry_price:.6f}")
                    return False
                if signal.target_price <= signal.entry_price:
                    logger.error(f"❌ LONG Signal validation failed: Target {signal.target_price:.6f} <= Entry {signal.entry_price:.6f}")
                    return False
            else:  # SELL (SHORT)
                if signal.stop_loss <= signal.entry_price:
                    logger.error(f"❌ SHORT Signal validation failed: Stop loss {signal.stop_loss:.6f} <= Entry {signal.entry_price:.6f}")
                    return False
                if signal.target_price >= signal.entry_price:
                    logger.error(f"❌ SHORT Signal validation failed: Target {signal.target_price:.6f} >= Entry {signal.entry_price:.6f}")
                    return False
            
            # 🚀 فیلتر 6: Minimum profit check
            min_profit_pct = config.MIN_PROFIT_TARGET  # 1.5%
            if reward < min_profit_pct:
                logger.info(f"Signal filtered: Profit too low ({reward*100:.2f}% < {min_profit_pct*100:.1f}%)")
                return False
            
            # 🚀 فیلتر 7: Maximum risk check  
            max_risk_pct = config.MAX_STOP_LOSS  # 1.2%
            if risk > max_risk_pct:
                logger.info(f"Signal filtered: Risk too high ({risk*100:.2f}% > {max_risk_pct*100:.1f}%)")
                return False
            
            logger.info(f"✅ Signal validation passed: {signal.signal_type} {signal.symbol}")
            logger.info(f"   Risk: {risk*100:.2f}%, Reward: {reward*100:.2f}%, R/R: {risk_reward:.2f}")
            logger.info(f"   RSI: {signal.rsi:.1f}, Volume: {signal.volume_ratio:.2f}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating signal: {str(e)}")
            logger.error(f"Signal details: {signal.symbol} {signal.signal_type} Entry:{signal.entry_price:.6f} Target:{signal.target_price:.6f} Stop:{signal.stop_loss:.6f}")
            return False
    
    def _is_duplicate_signal(self, new_signal: TradingSignal) -> bool:
        """
        Check if we already have an active signal for this symbol
        """
        for active_signal in self.active_signals:
            if (active_signal.symbol == new_signal.symbol and 
                active_signal.expires_at > datetime.now()):
                return True
        return False
    
    def scan_markets(self, symbols: List[str] = None) -> List[TradingSignal]:
        """
        Scan multiple markets for trading signals
        """
        if symbols is None:
            symbols = config.TRADING_PAIRS
        
        logger.info(f"Scanning {len(symbols)} markets for signals...")
        
        signals = []
        
        for symbol in symbols:
            try:
                signal = self.analyze_symbol(symbol)
                if signal:
                    signals.append(signal)
                    self.active_signals.append(signal)
                    logger.info(f"📈 Signal found: {symbol} @ {signal.confidence:.3f}")
                
                # Rate limiting
                import time
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error scanning {symbol}: {str(e)}")
                continue
        
        # Clean up expired signals
        self._cleanup_expired_signals()
        
        logger.info(f"Market scan completed: {len(signals)} new signals found")
        return signals
    
    def _cleanup_expired_signals(self):
        """Remove expired signals from active list"""
        now = datetime.now()
        self.active_signals = [
            signal for signal in self.active_signals 
            if signal.expires_at > now
        ]
    
    def get_active_signals(self) -> List[TradingSignal]:
        """Get list of active signals"""
        self._cleanup_expired_signals()
        return self.active_signals.copy()
    
    def get_live_learning_status(self) -> Dict:
        """گرفتن وضعیت یادگیری زنده"""
        if not self.live_learning:
            return {'status': 'disabled'}
        
        return self.live_learning.get_live_performance()
    
    def get_signal_summary(self) -> Dict:
        """Get summary of signal detection system"""
        active_signals = self.get_active_signals()
        
        base_summary = {
            'active_signals': len(active_signals),
            'model_loaded': self.ml_model.is_trained,
            'model_version': self.ml_model.model_info.get('version', 'None'),
            'confidence_threshold': config.CONFIDENCE_THRESHOLD,
            'symbols_monitored': len(config.TRADING_PAIRS),
            'last_scan': datetime.now().isoformat(),
            'signals_summary': [
                {
                    'symbol': signal.symbol,
                    'confidence': signal.confidence,
                    'created_at': signal.created_at.isoformat(),
                    'expires_at': signal.expires_at.isoformat()
                }
                for signal in active_signals
            ]
        }
        
        # Add live learning info
        if self.live_learning:
            live_status = self.get_live_learning_status()
            base_summary.update({
                'live_learning': live_status,
                'adaptive_threshold': self.live_learning.get_adaptive_threshold()
            })
        
        return base_summary
    
    def get_enhanced_summary(self) -> Dict:
        """خلاصه پیشرفته با یادگیری زنده"""
        summary = self.get_signal_summary()
        
        if self.live_learning:
            learning_summary = self.live_learning.get_learning_summary()
            summary['learning_summary_text'] = learning_summary
        
        return summary
    
    def validate_signal_outcome(self, signal: TradingSignal) -> Dict:
        """
        Check the outcome of a signal after expiry
        """
        try:
            # Get current price
            current_price = self.data_collector.get_current_price(signal.symbol)
            
            if current_price is None:
                return {'status': 'error', 'message': 'Could not get current price'}
            
            # Calculate actual return
            actual_return = (current_price - signal.entry_price) / signal.entry_price
            
            # Determine outcome
            if current_price >= signal.target_price:
                outcome = 'target_hit'
            elif current_price <= signal.stop_loss:
                outcome = 'stop_loss_hit'
            else:
                outcome = 'expired'
            
            return {
                'status': 'success',
                'signal_id': f"{signal.symbol}_{signal.created_at.timestamp()}",
                'symbol': signal.symbol,
                'entry_price': signal.entry_price,
                'current_price': current_price,
                'target_price': signal.target_price,
                'stop_loss': signal.stop_loss,
                'actual_return': actual_return,
                'predicted_return': signal.get_profit_potential(),
                'confidence': signal.confidence,
                'outcome': outcome,
                'success': outcome == 'target_hit'
            }
            
        except Exception as e:
            logger.error(f"Error validating signal outcome: {str(e)}")
            return {'status': 'error', 'message': str(e)}
    
    def get_system_status(self) -> Dict:
        """Get overall system status"""
        return {
            'data_collector': 'online',
            'feature_engineer': 'ready',
            'ml_model': 'trained' if self.ml_model.is_trained else 'not_trained',
            'live_learning': 'active' if self.live_learning else 'disabled',
            'active_signals': len(self.get_active_signals()),
            'model_info': self.ml_model.model_info,
            'last_update': datetime.now().isoformat()
        }