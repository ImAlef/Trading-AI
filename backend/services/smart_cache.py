#!/usr/bin/env python3
"""
Smart Caching System for Better Performance
"""
import pickle
import hashlib
import time
import os
import pandas as pd
import numpy as np
from typing import Any, Optional, Dict, List
import logging
from datetime import datetime, timedelta
import json

logger = logging.getLogger(__name__)

class SmartCache:
    """
    سیستم کش هوشمند برای بهبود سرعت
    """
    
    def __init__(self, cache_dir: str = "data/cache"):
        self.cache_dir = cache_dir
        self.memory_cache = {}
        self.cache_stats = {
            'hits': 0,
            'misses': 0,
            'total_requests': 0
        }
        
        # ایجاد پوشه cache
        os.makedirs(cache_dir, exist_ok=True)
        
        # تنظیمات cache
        self.cache_durations = {
            'market_data': 60,        # 1 دقیقه
            'predictions': 300,       # 5 دقیقه  
            'features': 180,          # 3 دقیقه
            'sentiment': 600,         # 10 دقیقه
            'indicators': 120,        # 2 دقیقه
            'model_data': 3600        # 1 ساعت
        }
        
        # تنظیمات memory cache
        self.max_memory_items = 1000
        self.memory_ttl = 300  # 5 دقیقه
        
        logger.info(f"📋 Smart Cache initialized at {cache_dir}")
    
    def get(self, key: str, category: str = 'default') -> Optional[Any]:
        """
        دریافت داده از cache
        """
        self.cache_stats['total_requests'] += 1
        
        # ابتدا memory cache را بررسی کن
        memory_result = self._get_from_memory(key)
        if memory_result is not None:
            self.cache_stats['hits'] += 1
            logger.debug(f"💾 Memory cache hit: {key}")
            return memory_result
        
        # سپس disk cache را بررسی کن
        disk_result = self._get_from_disk(key, category)
        if disk_result is not None:
            # اضافه کردن به memory cache
            self._set_to_memory(key, disk_result)
            self.cache_stats['hits'] += 1
            logger.debug(f"💿 Disk cache hit: {key}")
            return disk_result
        
        self.cache_stats['misses'] += 1
        logger.debug(f"❌ Cache miss: {key}")
        return None
    
    def set(self, key: str, value: Any, category: str = 'default', 
            ttl: Optional[int] = None) -> bool:
        """
        ذخیره داده در cache
        """
        try:
            # ذخیره در memory cache
            self._set_to_memory(key, value, ttl)
            
            # ذخیره در disk cache
            self._set_to_disk(key, value, category, ttl)
            
            logger.debug(f"💾 Cached: {key} (category: {category})")
            return True
            
        except Exception as e:
            logger.error(f"❌ Cache set error: {str(e)}")
            return False
    
    def _get_from_memory(self, key: str) -> Optional[Any]:
        """دریافت از memory cache"""
        try:
            if key in self.memory_cache:
                item = self.memory_cache[key]
                
                # بررسی expiry
                if time.time() < item['expires']:
                    return item['data']
                else:
                    # حذف داده منقضی
                    del self.memory_cache[key]
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Memory cache get error: {str(e)}")
            return None
    
    def _set_to_memory(self, key: str, value: Any, ttl: Optional[int] = None):
        """ذخیره در memory cache"""
        try:
            # پاکسازی cache در صورت پر بودن
            if len(self.memory_cache) >= self.max_memory_items:
                self._cleanup_memory_cache()
            
            expires = time.time() + (ttl or self.memory_ttl)
            
            self.memory_cache[key] = {
                'data': value,
                'expires': expires,
                'created': time.time()
            }
            
        except Exception as e:
            logger.error(f"❌ Memory cache set error: {str(e)}")
    
    def _get_from_disk(self, key: str, category: str) -> Optional[Any]:
        """دریافت از disk cache"""
        try:
            cache_file = self._get_cache_file_path(key, category)
            
            if not os.path.exists(cache_file):
                return None
            
            # بررسی expiry
            ttl = self.cache_durations.get(category, 300)
            file_age = time.time() - os.path.getmtime(cache_file)
            
            if file_age > ttl:
                # حذف فایل منقضی
                try:
                    os.remove(cache_file)
                except:
                    pass
                return None
            
            # خواندن داده
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
                return data
            
        except Exception as e:
            logger.error(f"❌ Disk cache get error: {str(e)}")
            return None
    
    def _set_to_disk(self, key: str, value: Any, category: str, ttl: Optional[int] = None):
        """ذخیره در disk cache"""
        try:
            cache_file = self._get_cache_file_path(key, category)
            
            # ذخیره داده
            with open(cache_file, 'wb') as f:
                pickle.dump(value, f)
            
        except Exception as e:
            logger.error(f"❌ Disk cache set error: {str(e)}")
    
    def _get_cache_file_path(self, key: str, category: str) -> str:
        """مسیر فایل cache"""
        # ایجاد hash از key برای نام فایل امن
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return os.path.join(self.cache_dir, f"{category}_{key_hash}.cache")
    
    def _cleanup_memory_cache(self):
        """پاکسازی memory cache"""
        try:
            # حذف نصف داده‌های قدیمی‌تر
            items = list(self.memory_cache.items())
            items.sort(key=lambda x: x[1]['created'])
            
            items_to_remove = items[:len(items)//2]
            
            for key, _ in items_to_remove:
                del self.memory_cache[key]
            
            logger.debug(f"🧹 Memory cache cleaned: removed {len(items_to_remove)} items")
            
        except Exception as e:
            logger.error(f"❌ Memory cache cleanup error: {str(e)}")
    
    def clear(self, category: Optional[str] = None):
        """پاکسازی cache"""
        try:
            if category:
                # پاکسازی یک دسته خاص
                for filename in os.listdir(self.cache_dir):
                    if filename.startswith(f"{category}_"):
                        file_path = os.path.join(self.cache_dir, filename)
                        os.remove(file_path)
                logger.info(f"🧹 Cleared {category} cache")
            else:
                # پاکسازی کامل
                for filename in os.listdir(self.cache_dir):
                    if filename.endswith('.cache'):
                        file_path = os.path.join(self.cache_dir, filename)
                        os.remove(file_path)
                
                self.memory_cache.clear()
                logger.info("🧹 Cleared all cache")
                
        except Exception as e:
            logger.error(f"❌ Cache clear error: {str(e)}")
    
    def get_stats(self) -> Dict:
        """آمار cache"""
        total = self.cache_stats['total_requests']
        hits = self.cache_stats['hits']
        misses = self.cache_stats['misses']
        
        hit_rate = (hits / total * 100) if total > 0 else 0
        
        # محاسبه اندازه cache
        memory_size = len(self.memory_cache)
        disk_files = len([f for f in os.listdir(self.cache_dir) if f.endswith('.cache')])
        
        return {
            'hit_rate': hit_rate,
            'total_requests': total,
            'cache_hits': hits,
            'cache_misses': misses,
            'memory_items': memory_size,
            'disk_files': disk_files
        }

class CachedDataCollector:
    """
    Data Collector با قابلیت cache
    """
    
    def __init__(self, base_collector, cache_system: SmartCache):
        self.base_collector = base_collector
        self.cache = cache_system
    
    def get_klines_cached(self, symbol: str, interval: str, limit: int = 500) -> pd.DataFrame:
        """
        دریافت klines با cache
        """
        try:
            # ایجاد cache key
            cache_key = f"klines_{symbol}_{interval}_{limit}"
            
            # بررسی cache
            cached_data = self.cache.get(cache_key, 'market_data')
            
            if cached_data is not None:
                logger.debug(f"📋 Using cached market data for {symbol}")
                return cached_data
            
            # دریافت داده جدید
            logger.debug(f"🔄 Fetching fresh market data for {symbol}")
            df = self.base_collector.get_klines(symbol, interval, limit)
            
            if not df.empty:
                # ذخیره در cache
                self.cache.set(cache_key, df, 'market_data')
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Cached klines error: {str(e)}")
            # fallback به collector اصلی
            return self.base_collector.get_klines(symbol, interval, limit)
    
    def get_current_price_cached(self, symbol: str) -> Optional[float]:
        """
        دریافت قیمت فعلی با cache
        """
        try:
            cache_key = f"price_{symbol}"
            
            # cache کوتاه‌مدت برای قیمت (30 ثانیه)
            cached_price = self.cache.get(cache_key, 'market_data')
            
            if cached_price is not None:
                return cached_price
            
            # دریافت قیمت جدید
            price = self.base_collector.get_current_price(symbol)
            
            if price:
                # cache با TTL کوتاه
                self.cache.set(cache_key, price, 'market_data', ttl=30)
            
            return price
            
        except Exception as e:
            logger.error(f"❌ Cached price error: {str(e)}")
            return self.base_collector.get_current_price(symbol)

class CachedFeatureEngineer:
    """
    Feature Engineer با قابلیت cache
    """
    
    def __init__(self, base_engineer, cache_system: SmartCache):
        self.base_engineer = base_engineer
        self.cache = cache_system
    
    def prepare_features_cached(self, df: pd.DataFrame, for_prediction: bool = True) -> pd.DataFrame:
        """
        آماده‌سازی features با cache
        """
        try:
            # ایجاد hash از DataFrame
            df_hash = self._calculate_dataframe_hash(df)
            cache_key = f"features_{df_hash}_{for_prediction}"
            
            # بررسی cache
            cached_features = self.cache.get(cache_key, 'features')
            
            if cached_features is not None:
                logger.debug("📋 Using cached features")
                return cached_features
            
            # محاسبه features جدید
            logger.debug("🔄 Computing fresh features")
            
            if for_prediction:
                features_df = self.base_engineer.prepare_features_for_prediction(df)
            else:
                features_df = self.base_engineer.prepare_features(df)
            
            if not features_df.empty:
                # ذخیره در cache
                self.cache.set(cache_key, features_df, 'features')
            
            return features_df
            
        except Exception as e:
            logger.error(f"❌ Cached features error: {str(e)}")
            # fallback
            if for_prediction:
                return self.base_engineer.prepare_features_for_prediction(df)
            else:
                return self.base_engineer.prepare_features(df)
    
    def _calculate_dataframe_hash(self, df: pd.DataFrame) -> str:
        """محاسبه hash از DataFrame"""
        try:
            # استفاده از آخرین ردیف و اندازه DataFrame
            if df.empty:
                return "empty_df"
            
            # ترکیب timestamp آخرین ردیف + اندازه DataFrame
            last_timestamp = df.index[-1] if hasattr(df.index, 'min') else len(df)
            size_info = f"{len(df)}_{len(df.columns)}"
            
            # محاسبه hash
            hash_input = f"{last_timestamp}_{size_info}"
            return hashlib.md5(hash_input.encode()).hexdigest()[:16]
            
        except Exception as e:
            logger.error(f"❌ DataFrame hash error: {str(e)}")
            return f"hash_error_{time.time()}"

class CachedMLModel:
    """
    ML Model با قابلیت cache
    """
    
    def __init__(self, base_model, cache_system: SmartCache):
        self.base_model = base_model
        self.cache = cache_system
    
    def predict_single_cached(self, X: pd.DataFrame) -> tuple[int, float]:
        """
        پیش‌بینی تکی با cache
        """
        try:
            # ایجاد cache key از features
            features_hash = self._calculate_features_hash(X)
            cache_key = f"prediction_{features_hash}"
            
            # بررسی cache
            cached_result = self.cache.get(cache_key, 'predictions')
            
            if cached_result is not None:
                logger.debug("📋 Using cached prediction")
                return cached_result
            
            # پیش‌بینی جدید
            logger.debug("🔄 Computing fresh prediction")
            prediction, confidence = self.base_model.predict_single(X)
            
            result = (int(prediction), float(confidence))
            
            # ذخیره در cache
            self.cache.set(cache_key, result, 'predictions')
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Cached prediction error: {str(e)}")
            # fallback
            return self.base_model.predict_single(X)
    
    def _calculate_features_hash(self, X: pd.DataFrame) -> str:
        """محاسبه hash از features"""
        try:
            if X.empty:
                return "empty_features"
            
            # استفاده از مقادیر آخرین ردیف
            last_row = X.iloc[-1]
            
            # انتخاب features مهم برای hash
            important_features = ['close', 'rsi', 'macd', 'volume_ratio', 'bb_position']
            hash_values = []
            
            for feature in important_features:
                if feature in last_row:
                    # گرد کردن به 4 رقم اعشار
                    value = round(float(last_row[feature]), 4)
                    hash_values.append(str(value))
            
            hash_input = "_".join(hash_values)
            return hashlib.md5(hash_input.encode()).hexdigest()[:16]
            
        except Exception as e:
            logger.error(f"❌ Features hash error: {str(e)}")
            return f"features_error_{time.time()}"
    
    # سایر متدهای base_model را proxy کن
    def __getattr__(self, name):
        return getattr(self.base_model, name)

class PerformanceMonitor:
    """
    مانیتور عملکرد cache
    """
    
    def __init__(self, cache_system: SmartCache):
        self.cache = cache_system
        self.start_time = time.time()
        self.request_times = []
    
    def log_request_time(self, duration: float):
        """ثبت زمان درخواست"""
        self.request_times.append(duration)
        
        # نگهداری آخرین 1000 درخواست
        if len(self.request_times) > 1000:
            self.request_times = self.request_times[-1000:]
    
    def get_performance_stats(self) -> Dict:
        """آمار عملکرد"""
        cache_stats = self.cache.get_stats()
        
        uptime = time.time() - self.start_time
        
        avg_request_time = np.mean(self.request_times) if self.request_times else 0
        
        return {
            'cache_hit_rate': cache_stats['hit_rate'],
            'total_requests': cache_stats['total_requests'],
            'avg_request_time_ms': avg_request_time * 1000,
            'uptime_hours': uptime / 3600,
            'memory_cache_size': cache_stats['memory_items'],
            'disk_cache_files': cache_stats['disk_files'],
            'performance_gain': self._calculate_performance_gain(cache_stats['hit_rate'])
        }
    
    def _calculate_performance_gain(self, hit_rate: float) -> str:
        """محاسبه بهبود عملکرد"""
        if hit_rate == 0:
            return "No improvement"
        
        # فرض: هر cache hit 10x سریعتر از fetch جدید
        speedup = 1 + (hit_rate / 100) * 9
        return f"{speedup:.1f}x faster"
    
    def get_cache_health(self) -> Dict:
        """سلامت cache"""
        stats = self.cache.get_stats()
        
        # بررسی سلامت بر اساس hit rate
        hit_rate = stats['hit_rate']
        
        if hit_rate >= 70:
            health = "Excellent"
        elif hit_rate >= 50:
            health = "Good"
        elif hit_rate >= 30:
            health = "Fair"
        else:
            health = "Poor"
        
        return {
            'health_status': health,
            'hit_rate': hit_rate,
            'recommendations': self._get_cache_recommendations(hit_rate)
        }
    
    def _get_cache_recommendations(self, hit_rate: float) -> List[str]:
        """پیشنهادات بهبود cache"""
        recommendations = []
        
        if hit_rate < 30:
            recommendations.append("Consider increasing cache TTL")
            recommendations.append("Check if cache keys are too specific")
        
        if hit_rate < 50:
            recommendations.append("Monitor cache size limits")
            recommendations.append("Review cache categories")
        
        if hit_rate > 80:
            recommendations.append("Cache is performing well")
        
        return recommendations