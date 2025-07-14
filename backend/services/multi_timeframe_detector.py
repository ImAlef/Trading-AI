#!/usr/bin/env python3
"""
Multi-Timeframe Signal Detection
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional, List
import logging
from datetime import datetime
from backend.services.fallback_data_collector import FallbackDataCollector
from backend.services.feature_engineer import FeatureEngineer
from backend.services.ml_model import MLModel
from config import config

logger = logging.getLogger(__name__)

class MultiTimeframeDetector:
    """
    تشخیص سیگنال در چندین timeframe برای دقت بالاتر
    """
    
    def __init__(self):
        self.data_collector = FallbackDataCollector()
        self.feature_engineer = FeatureEngineer()
        
        # تنظیمات timeframe ها
        self.timeframes = {
            '15m': {
                'weight': 0.15,        # وزن کم برای timeframe کوتاه
                'periods': 96,         # 24 ساعت در 15 دقیقه
                'threshold': 0.50      # آستانه کمتر
            },
            '1h': {
                'weight': 0.45,        # وزن اصلی
                'periods': 168,        # 1 هفته
                'threshold': 0.55      # آستانه اصلی
            },
            '4h': {
                'weight': 0.35,        # وزن بالا برای trend
                'periods': 168,        # 4 هفته
                'threshold': 0.60      # آستانه بالاتر
            },
            '1d': {
                'weight': 0.05,        # وزن کم ولی مهم برای trend کلی
                'periods': 90,         # 3 ماه
                'threshold': 0.65      # آستانه خیلی بالا
            }
        }
        
        # مدل‌های مختلف برای هر timeframe
        self.models = {}
        self.models_loaded = False
    
    def load_models(self) -> bool:
        """بارگذاری مدل‌ها برای هر timeframe - نسخه بهبود شده"""
        try:
            logger.info("🔄 Loading models for each timeframe...")
            
            # Import کردن مجدد MLModel برای اطمینان
            from backend.services.ml_model import MLModel
            
            # ایجاد instance جدید
            base_model = MLModel()
            
            # تلاش برای بارگذاری مدل
            try:
                model_loaded = base_model.load_latest_model()
                if not model_loaded:
                    logger.warning("⚠️ Latest model not found, trying any available model...")
                    
                    # تلاش برای یافتن هر مدلی در پوشه models
                    import os
                    models_dir = "data/models"
                    if os.path.exists(models_dir):
                        model_files = [f for f in os.listdir(models_dir) if f.endswith('.pkl')]
                        if model_files:
                            # استفاده از اولین مدل موجود
                            model_path = os.path.join(models_dir, model_files[0])
                            model_loaded = base_model.load_model(model_path)
                            logger.info(f"✅ Loaded model: {model_files[0]}")
                        else:
                            logger.error("❌ No model files found in data/models/")
                            return False
                    else:
                        logger.error("❌ Models directory not found")
                        return False
            except Exception as e:
                logger.error(f"❌ Error loading model: {str(e)}")
                return False
            
            if not model_loaded:
                logger.error("❌ Failed to load any model")
                return False
            
            # بررسی که مدل trained باشد
            if not base_model.is_trained:
                logger.error("❌ Model is not trained")
                return False
            
            # استفاده از همان مدل برای همه timeframe ها
            for tf in self.timeframes.keys():
                self.models[tf] = base_model
                logger.debug(f"   ✓ Model assigned to {tf}")
            
            self.models_loaded = True
            logger.info(f"✅ Models successfully loaded for {len(self.models)} timeframes")
            logger.info(f"   Model version: {base_model.model_info.get('version', 'unknown')}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Critical error loading models: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    # و این متد برای بهبود تحلیل:
    def analyze_symbol_multi_timeframe(self, symbol: str) -> Optional[Dict]:
        """
        تحلیل یک symbol در چندین timeframe - نسخه بهبود شده
        """
        try:
            # بررسی و بارگذاری مدل‌ها
            if not self.models_loaded:
                logger.info(f"🔄 Models not loaded, loading now for {symbol}...")
                if not self.load_models():
                    logger.error(f"❌ Failed to load models for {symbol}")
                    return None
            
            logger.info(f"🔍 Multi-timeframe analysis for {symbol}")
            
            timeframe_results = {}
            successful_analyses = 0
            
            # تحلیل در هر timeframe
            for tf, tf_config in self.timeframes.items():
                try:
                    logger.debug(f"   Analyzing {symbol} on {tf}...")
                    result = self.analyze_single_timeframe(symbol, tf, tf_config)
                    
                    if result:
                        timeframe_results[tf] = result
                        successful_analyses += 1
                        logger.info(f"   ✓ {tf}: prediction={result['prediction']}, confidence={result['confidence']:.3f}")
                    else:
                        logger.warning(f"   ⚠️ {tf}: No result")
                    
                except Exception as e:
                    logger.error(f"❌ Error analyzing {symbol} on {tf}: {str(e)}")
                    continue
            
            logger.info(f"📊 {symbol}: {successful_analyses}/{len(self.timeframes)} timeframes analyzed successfully")
            
            if not timeframe_results:
                logger.warning(f"⚠️ No timeframe analysis completed for {symbol}")
                return None
            
            # ترکیب نتایج
            combined_signal = self.combine_timeframe_signals(timeframe_results, symbol)
            
            if combined_signal:
                logger.info(f"✅ Multi-timeframe signal: {symbol}")
                logger.info(f"   Final confidence: {combined_signal['confidence']:.3f}")
                logger.info(f"   Timeframes agreed: {combined_signal['timeframes_agreed']}/{combined_signal['total_timeframes']}")
                logger.info(f"   Trend alignment: {combined_signal['trend_alignment']:.2f}")
            else:
                logger.info(f"❌ No combined signal for {symbol}")
            
            return combined_signal
            
        except Exception as e:
            logger.error(f"❌ Critical error in multi-timeframe analysis for {symbol}: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def analyze_single_timeframe(self, symbol: str, timeframe: str, config: Dict) -> Optional[Dict]:
        """
        تحلیل در یک timeframe مشخص
        """
        try:
            # دریافت داده
            df = self.data_collector.get_klines(symbol, timeframe, limit=config['periods'])
            
            if df.empty or len(df) < 50:
                logger.warning(f"⚠️ Insufficient data for {symbol} on {timeframe}")
                return None
            
            # feature engineering
            df_features = self.feature_engineer.prepare_features_for_prediction(df)
            
            if df_features.empty:
                logger.warning(f"⚠️ Feature engineering failed for {symbol} on {timeframe}")
                return None
            
            # پیش‌بینی
            latest_features = df_features.iloc[-1:].copy()
            model = self.models.get(timeframe)
            
            if not model:
                return None
            
            prediction, confidence = model.predict_single(latest_features)
            
            # محاسبه قدرت trend
            trend_strength = self.calculate_trend_strength(df_features)
            
            # بررسی آستانه
            threshold = config.get('threshold', 0.55)
            
            return {
                'prediction': prediction,
                'confidence': confidence,
                'trend_strength': trend_strength,
                'threshold': threshold,
                'weight': config['weight'],
                'periods_analyzed': len(df_features),
                'timeframe': timeframe,
                'passes_threshold': confidence >= threshold and prediction == 1
            }
            
        except Exception as e:
            logger.error(f"❌ Error analyzing {timeframe}: {str(e)}")
            return None
    
    def calculate_trend_strength(self, df: pd.DataFrame) -> float:
        """
        محاسبه قدرت trend در timeframe
        """
        try:
            if df.empty:
                return 0.5
            
            latest = df.iloc[-1]
            trend_factors = []
            
            # 1. EMA Trend Analysis
            try:
                if 'ema_9' in df.columns and 'ema_21' in df.columns:
                    if latest['ema_9'] > latest['ema_21']:
                        trend_factors.append(0.3)
                    
                if 'ema_21' in df.columns and 'sma_50' in df.columns:
                    if latest['ema_21'] > latest['sma_50']:
                        trend_factors.append(0.3)
            except:
                pass
            
            # 2. MACD Analysis
            try:
                if 'macd' in df.columns and 'macd_signal' in df.columns:
                    if latest['macd'] > latest['macd_signal']:
                        trend_factors.append(0.2)
                        
                if 'macd_histogram' in df.columns:
                    if latest['macd_histogram'] > 0:
                        trend_factors.append(0.1)
            except:
                pass
            
            # 3. Price vs Moving Averages
            try:
                if 'close' in df.columns and 'ema_21' in df.columns:
                    if latest['close'] > latest['ema_21']:
                        trend_factors.append(0.2)
            except:
                pass
            
            # 4. Volume Confirmation
            try:
                if 'volume_ratio' in df.columns:
                    volume_strength = min(latest['volume_ratio'] / 2.0, 0.3)
                    trend_factors.append(volume_strength)
            except:
                pass
            
            # محاسبه نهایی
            if trend_factors:
                trend_strength = sum(trend_factors)
                return min(trend_strength, 1.0)
            
            return 0.5  # Neutral
            
        except Exception as e:
            logger.error(f"❌ Error calculating trend strength: {str(e)}")
            return 0.5
    
    def combine_timeframe_signals(self, timeframe_results: Dict, symbol: str) -> Optional[Dict]:
        """
        ترکیب هوشمندانه سیگنال‌های multi-timeframe
        """
        try:
            if not timeframe_results:
                return None
            
            # 1. بررسی تعداد timeframe هایی که signal داده‌اند
            passing_timeframes = []
            total_timeframes = len(timeframe_results)
            
            for tf, result in timeframe_results.items():
                if result.get('passes_threshold', False):
                    passing_timeframes.append((tf, result))
            
            # حداقل 2 timeframe باید موافق باشند
            min_agreement = 2
            if len(passing_timeframes) < min_agreement:
                logger.info(f"❌ {symbol}: Only {len(passing_timeframes)} timeframes agreed (need {min_agreement})")
                return None
            
            # 2. محاسبه weighted confidence
            total_weight = 0
            weighted_confidence = 0
            
            for tf, result in passing_timeframes:
                weight = result['weight']
                confidence = result['confidence']
                
                weighted_confidence += confidence * weight
                total_weight += weight
            
            if total_weight == 0:
                return None
            
            base_confidence = weighted_confidence / total_weight
            
            # 3. محاسبه trend alignment bonus
            trend_scores = [result['trend_strength'] for tf, result in passing_timeframes]
            trend_alignment = np.mean(trend_scores) if trend_scores else 0.5
            
            # 4. Higher timeframe confirmation bonus
            htf_bonus = self.calculate_htf_bonus(passing_timeframes)
            
            # 5. محاسبه confidence نهایی
            alignment_bonus = (trend_alignment - 0.5) * 0.1  # تا 5% bonus یا penalty
            
            final_confidence = base_confidence + alignment_bonus + htf_bonus
            final_confidence = min(final_confidence, 0.95)  # حداکثر 95%
            
            # 6. بررسی نهایی
            if final_confidence >= config.CONFIDENCE_THRESHOLD:
                return {
                    'prediction': 1,
                    'confidence': final_confidence,
                    'timeframes_agreed': len(passing_timeframes),
                    'total_timeframes': total_timeframes,
                    'trend_alignment': trend_alignment,
                    'htf_confirmation': htf_bonus > 0,
                    'base_confidence': base_confidence,
                    'analysis_type': 'multi_timeframe',
                    'timeframe_breakdown': {
                        tf: {
                            'confidence': result['confidence'],
                            'trend_strength': result['trend_strength']
                        }
                        for tf, result in passing_timeframes
                    }
                }
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Error combining timeframe signals: {str(e)}")
            return None
    
    def calculate_htf_bonus(self, passing_timeframes: List) -> float:
        """
        محاسبه bonus برای تایید higher timeframe ها
        """
        try:
            htf_priority = {
                '1d': 4,    # بالاترین اولویت
                '4h': 3,
                '1h': 2,
                '15m': 1    # کمترین اولویت
            }
            
            confirmation_score = 0
            max_possible_score = 0
            
            for tf, result in passing_timeframes:
                priority = htf_priority.get(tf, 1)
                confidence = result['confidence']
                
                max_possible_score += priority
                confirmation_score += priority * confidence
            
            if max_possible_score > 0:
                htf_ratio = confirmation_score / max_possible_score
                
                # اگر higher timeframe ها تایید کنند، bonus بده
                higher_timeframes = ['4h', '1d']
                htf_count = sum(1 for tf, _ in passing_timeframes if tf in higher_timeframes)
                
                if htf_count > 0:
                    return min(htf_ratio * 0.05, 0.05)  # حداکثر 5% bonus
            
            return 0
            
        except Exception as e:
            logger.error(f"❌ Error calculating HTF bonus: {str(e)}")
            return 0
    
    def get_timeframe_summary(self, symbol: str) -> Dict:
        """
        خلاصه‌ای از وضعیت symbol در همه timeframe ها
        """
        try:
            summary = {
                'symbol': symbol,
                'analysis_time': datetime.now().isoformat(),
                'timeframes': {}
            }
            
            for tf, config in self.timeframes.items():
                try:
                    result = self.analyze_single_timeframe(symbol, tf, config)
                    if result:
                        summary['timeframes'][tf] = {
                            'confidence': result['confidence'],
                            'prediction': result['prediction'],
                            'trend_strength': result['trend_strength'],
                            'passes_threshold': result['passes_threshold']
                        }
                except:
                    summary['timeframes'][tf] = {
                        'confidence': 0,
                        'prediction': 0,
                        'trend_strength': 0,
                        'passes_threshold': False,
                        'error': True
                    }
            
            return summary
            
        except Exception as e:
            logger.error(f"❌ Error getting timeframe summary: {str(e)}")
            return {'symbol': symbol, 'error': str(e)}