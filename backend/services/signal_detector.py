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
from backend.services.multi_timeframe_detector import MultiTimeframeDetector

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
        """Calculate potential profit percentage"""
        return (self.target_price - self.entry_price) / self.entry_price
    
    def get_risk_ratio(self) -> float:
        """Calculate risk-to-reward ratio"""
        risk = abs(self.entry_price - self.stop_loss) / self.entry_price
        reward = self.get_profit_potential()
        return reward / risk if risk > 0 else 0

class SignalDetector:
    """
    Main signal detection system با قابلیت یادگیری زنده
    """
    
    def __init__(self):
        self.data_collector = FallbackDataCollector()
        self.feature_engineer = FeatureEngineer()
        self.ml_model = MLModel()
        self.active_signals: List[TradingSignal] = []
        
        # 🆕 اضافه کردن Multi-Timeframe Detector
        self.mtf_detector = MultiTimeframeDetector()
        
        # Live Learning System
        self.live_learning: Optional[LiveLearningSystem] = None

    def analyze_symbol_enhanced(self, symbol: str, use_multi_timeframe: bool = True) -> Optional[TradingSignal]:
        """
        تحلیل پیشرفته symbol با multi-timeframe
        """
        try:
            logger.info(f"🔍 Enhanced analysis for {symbol} (MTF: {use_multi_timeframe})")
            
            if use_multi_timeframe and self.mtf_detector:
                # استفاده از Multi-Timeframe Analysis
                mtf_result = self.mtf_detector.analyze_symbol_multi_timeframe(symbol)
                
                if mtf_result:
                    logger.info(f"✅ Multi-timeframe signal found for {symbol}")
                    
                    # دریافت داده‌های 1h برای محاسبه قیمت‌ها
                    df = self.data_collector.get_klines(symbol, "1h", limit=config.LOOKBACK_PERIODS + 50)
                    
                    if df.empty:
                        logger.warning(f"⚠️ No market data for {symbol}")
                        return None
                    
                    # Feature engineering برای محاسبه قیمت‌ها
                    df_features = self.feature_engineer.prepare_features_for_prediction(df)
                    
                    if df_features.empty:
                        return None
                    
                    # ساخت سیگنال با confidence از multi-timeframe
                    signal = self._create_signal_from_mtf(
                        symbol=symbol,
                        mtf_result=mtf_result,
                        features=df_features,
                        timeframe="multi"
                    )
                    
                    # اعتبارسنجی سیگنال
                    if self._validate_signal(signal):
                        # ثبت برای live learning
                        if self.live_learning:
                            signal_id = self.live_learning.register_signal(signal, df_features)
                            logger.info(f"🧠 MTF Signal registered for live learning: {signal_id}")
                        
                        return signal
                    else:
                        logger.info(f"❌ Multi-timeframe signal filtered out for {symbol}")
                        return None
                else:
                    logger.info(f"⚪ No multi-timeframe signal for {symbol}")
                    return None
            else:
                # استفاده از روش قدیمی (single timeframe)
                return self.analyze_symbol(symbol)
                
        except Exception as e:
            logger.error(f"❌ Error in enhanced analysis for {symbol}: {str(e)}")
            return None

    def _create_signal_from_mtf(self, symbol: str, mtf_result: Dict, 
                            features: pd.DataFrame, timeframe: str) -> TradingSignal:
        """
        ساخت سیگنال از نتیجه multi-timeframe
        """
        try:
            # استفاده از confidence از multi-timeframe
            confidence = mtf_result['confidence']
            
            # دریافت آخرین قیمت
            current_price = features['close'].iloc[-1]
            latest_row = features.iloc[-1]
            
            # محاسبه قیمت‌ها (همان روش قبلی)
            entry_price = self._calculate_entry_price(current_price, latest_row, confidence)
            
            # محاسبه ATR برای stop loss
            atr = latest_row.get('atr', current_price * 0.02)
            support_level = self._calculate_support_level(features, current_price)
            resistance_level = self._calculate_resistance_level(features, current_price)
            
            stop_loss = self._calculate_stop_loss(entry_price, atr, support_level, latest_row, confidence)
            target_price = self._calculate_take_profit(entry_price, stop_loss, resistance_level, latest_row, confidence)
            
            # ساخت سیگنال
            signal = TradingSignal(
                symbol=symbol,
                signal_type='BUY',
                entry_price=entry_price,
                target_price=target_price,
                stop_loss=stop_loss,
                confidence=confidence,
                created_at=datetime.now(),
                expires_at=datetime.now() + timedelta(hours=config.SIGNAL_EXPIRY_HOURS),
                rsi=latest_row.get('rsi', 0),
                macd=latest_row.get('macd', 0),
                macd_signal=latest_row.get('macd_signal', 0),
                bb_upper=latest_row.get('bb_upper', 0),
                bb_lower=latest_row.get('bb_lower', 0),
                volume_ratio=latest_row.get('volume_ratio', 1),
                model_version=f"MTF_{self.ml_model.model_info.get('version', 'unknown')}",
                features_used=len(self.feature_engineer.get_feature_columns()),
                timeframe=timeframe
            )
            
            # اضافه کردن اطلاعات multi-timeframe به سیگنال
            signal.mtf_info = {
                'timeframes_agreed': mtf_result['timeframes_agreed'],
                'total_timeframes': mtf_result['total_timeframes'],
                'trend_alignment': mtf_result['trend_alignment'],
                'htf_confirmation': mtf_result['htf_confirmation'],
                'base_confidence': mtf_result['base_confidence'],
                'timeframe_breakdown': mtf_result.get('timeframe_breakdown', {})
            }
            
            return signal
            
        except Exception as e:
            logger.error(f"❌ Error creating MTF signal: {str(e)}")
            # fallback به محاسبه ساده
            return self._create_signal(symbol, features['close'].iloc[-1], mtf_result['confidence'], features, timeframe)

    def scan_markets_enhanced(self, symbols: List[str] = None, use_multi_timeframe: bool = True) -> List[TradingSignal]:
        """
        اسکن پیشرفته بازارها با multi-timeframe
        """
        if symbols is None:
            symbols = config.TRADING_PAIRS
        
        scan_type = "Multi-Timeframe" if use_multi_timeframe else "Single-Timeframe"
        logger.info(f"🔍 {scan_type} market scan starting for {len(symbols)} symbols...")
        
        signals = []
        successful_scans = 0
        
        for symbol in symbols:
            try:
                signal = self.analyze_symbol_enhanced(symbol, use_multi_timeframe)
                
                if signal:
                    signals.append(signal)
                    self.active_signals.append(signal)
                    
                    # اطلاعات اضافی برای multi-timeframe
                    if hasattr(signal, 'mtf_info') and signal.mtf_info:
                        mtf_info = signal.mtf_info
                        logger.info(f"📈 {scan_type} Signal: {symbol} @ {signal.confidence:.3f}")
                        logger.info(f"   Timeframes: {mtf_info['timeframes_agreed']}/{mtf_info['total_timeframes']}")
                        logger.info(f"   Trend alignment: {mtf_info['trend_alignment']:.2f}")
                        logger.info(f"   HTF confirmation: {'✅' if mtf_info['htf_confirmation'] else '❌'}")
                    else:
                        logger.info(f"📈 Single-TF Signal: {symbol} @ {signal.confidence:.3f}")
                
                successful_scans += 1
                
                # Rate limiting
                import time
                time.sleep(0.15)  # کمی بیشتر برای multi-timeframe
                
            except Exception as e:
                logger.error(f"❌ Error scanning {symbol}: {str(e)}")
                continue
        
        # پاکسازی سیگنال‌های منقضی
        self._cleanup_expired_signals()
        
        logger.info(f"✅ {scan_type} scan completed:")
        logger.info(f"   Successful scans: {successful_scans}/{len(symbols)}")
        logger.info(f"   New signals: {len(signals)}")
        logger.info(f"   Active signals: {len(self.active_signals)}")
        
        return signals

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
        Analyze a single symbol and generate signal if conditions are met
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
                logger.info(f"{symbol}: Prediction={prediction}, Confidence={confidence:.3f}, Adaptive Threshold={threshold:.3f}")
            else:
                logger.info(f"{symbol}: Prediction={prediction}, Confidence={confidence:.3f}")
            
            # Step 6: Check if signal meets criteria
            if prediction == 1 and confidence >= threshold:
                
                # Get current market data
                current_price = latest_features['close'].iloc[-1]
                
                # Calculate signal parameters
                signal = self._create_signal(
                    symbol=symbol,
                    current_price=current_price,
                    confidence=confidence,
                    features=latest_features,
                    timeframe=timeframe
                )
                
                # Additional filters
                if self._validate_signal(signal):
                    # Register signal for live learning
                    if self.live_learning:
                        signal_id = self.live_learning.register_signal(signal, latest_features)
                        logger.info(f"🧠 Signal registered for live learning: {signal_id}")
                    
                    logger.info(f"✅ Valid signal generated for {symbol}")
                    return signal
                else:
                    logger.info(f"❌ Signal filtered out for {symbol}")
                    return None
            
            return None
            
        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {str(e)}")
            return None
    
    def _create_signal(self, symbol: str, current_price: float, confidence: float, 
                      features: pd.DataFrame, timeframe: str) -> TradingSignal:
        """
        Create a trading signal with dynamically calculated parameters
        """
        # Get latest market data for better calculations
        latest_row = features.iloc[-1]
        
        # Calculate ATR for volatility-based stops
        atr = latest_row.get('atr', current_price * 0.02)  # Default 2% if ATR not available
        
        # Calculate support/resistance levels
        support_level = self._calculate_support_level(features, current_price)
        resistance_level = self._calculate_resistance_level(features, current_price)
        
        # Dynamic entry strategy based on confidence and technical indicators
        entry_price = self._calculate_entry_price(current_price, latest_row, confidence)
        
        # Dynamic stop loss based on ATR and support levels
        stop_loss = self._calculate_stop_loss(entry_price, atr, support_level, latest_row, confidence)
        
        # Dynamic take profit based on resistance and risk-reward
        target_price = self._calculate_take_profit(entry_price, stop_loss, resistance_level, latest_row, confidence)
        
        # Create signal
        signal = TradingSignal(
            symbol=symbol,
            signal_type='BUY',  # Currently only BUY signals
            entry_price=entry_price,
            target_price=target_price,
            stop_loss=stop_loss,
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
    
    def _calculate_entry_price(self, current_price: float, latest_row: pd.Series, confidence: float) -> float:
        """🚀 محاسبه قیمت ورود بهینه‌شده"""
        try:
            # منطق بهینه‌شده از بک‌تست
            if confidence > 0.7:
                return current_price  # ورود فوری برای confidence بالا
            elif confidence > 0.6:
                pullback_pct = 0.003  # 0.3% pullback کوچک
                return current_price * (1 - pullback_pct)
            else:
                pullback_pct = 0.007  # 0.7% pullback متوسط
                return current_price * (1 - pullback_pct)
            
        except Exception as e:
            logger.error(f"Error calculating optimized entry price: {str(e)}")
            return current_price
    
    def _calculate_stop_loss(self, entry_price: float, atr: float, support_level: float, 
                           latest_row: pd.Series, confidence: float) -> float:
        """🚀 محاسبه stop loss بهینه‌شده"""
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
            final_stop = max(calculated_stop, max_stop)
            
            return final_stop
            
        except Exception as e:
            logger.error(f"Error calculating optimized stop loss: {str(e)}")
            return entry_price * (1 - config.MAX_STOP_LOSS)
    
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
        """🚀 اعتبارسنجی سیگنال بهینه‌شده"""
        try:
            # 🚀 فیلتر 1: نسبت ریسک-سود کمتر
            risk_reward = signal.get_risk_ratio()
            if risk_reward < config.SIGNAL_VALIDATION["min_risk_reward"]:  # 1.2
                logger.info(f"Signal filtered: Poor risk-reward ratio ({risk_reward:.2f})")
                return False
            
            # 🚀 فیلتر 2: RSI اجازه بالاتر
            if signal.rsi > config.SIGNAL_VALIDATION["max_rsi_overbought"]:  # 90
                logger.info(f"Signal filtered: RSI too high ({signal.rsi:.1f})")
                return False
            
            # 🚀 فیلتر 3: Volume کمتر سخت‌گیری
            if signal.volume_ratio < config.SIGNAL_VALIDATION["min_volume_ratio"]:  # 0.6
                logger.info(f"Signal filtered: Low volume ({signal.volume_ratio:.2f})")
                return False
            
            # فیلتر 4: تکراری نبودن
            if self._is_duplicate_signal(signal):
                logger.info(f"Signal filtered: Duplicate signal for {signal.symbol}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating signal: {str(e)}")
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
        """
        خلاصه پیشرفته با اطلاعات multi-timeframe
        """
        summary = self.get_signal_summary()
        
        # اضافه کردن اطلاعات multi-timeframe
        mtf_signals = [s for s in self.active_signals if hasattr(s, 'mtf_info') and s.mtf_info]
        
        summary.update({
            'multi_timeframe_signals': len(mtf_signals),
            'single_timeframe_signals': len(self.active_signals) - len(mtf_signals),
            'mtf_detector_loaded': self.mtf_detector.models_loaded if self.mtf_detector else False
        })
        
        # اطلاعات تفصیلی سیگنال‌های multi-timeframe
        if mtf_signals:
            summary['mtf_signal_details'] = []
            for signal in mtf_signals:
                if hasattr(signal, 'mtf_info') and signal.mtf_info:
                    summary['mtf_signal_details'].append({
                        'symbol': signal.symbol,
                        'confidence': signal.confidence,
                        'timeframes_agreed': signal.mtf_info['timeframes_agreed'],
                        'trend_alignment': signal.mtf_info['trend_alignment'],
                        'htf_confirmation': signal.mtf_info['htf_confirmation']
                    })
        
        # اضافه کردن خلاصه live learning
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