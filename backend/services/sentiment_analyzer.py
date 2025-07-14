#!/usr/bin/env python3
"""
Real-time Sentiment Analysis for Crypto Signals
"""
import requests
import asyncio
import aiohttp
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging
import time
import re
from textblob import TextBlob

logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    """
    تحلیل sentiment از منابع مختلف برای بهبود دقت سیگنال‌ها
    """
    
    def __init__(self):
        self.sources = {
            'fear_greed': FearGreedAnalyzer(),
            'news': NewsAnalyzer(),
            'onchain': OnChainAnalyzer(),
            'social': SocialAnalyzer()
        }
        
        # وزن هر منبع
        self.source_weights = {
            'fear_greed': 0.25,  # Fear & Greed Index
            'news': 0.25,        # اخبار
            'onchain': 0.35,     # داده‌های on-chain
            'social': 0.15       # شبکه‌های اجتماعی
        }
        
        self.cache = {}
        self.cache_duration = 300  # 5 دقیقه
    
    async def get_market_sentiment(self, symbol: str = 'BTCUSDT') -> Dict:
        """
        دریافت sentiment کلی بازار
        """
        try:
            logger.info(f"🔍 Analyzing market sentiment for {symbol}")
            
            # بررسی cache
            cache_key = f"sentiment_{symbol}"
            if self._is_cached(cache_key):
                logger.info("📋 Using cached sentiment data")
                return self.cache[cache_key]['data']
            
            sentiment_data = {}
            
            # تحلیل از منابع مختلف
            for source_name, analyzer in self.sources.items():
                try:
                    result = await analyzer.analyze_sentiment(symbol)
                    sentiment_data[source_name] = result
                    logger.info(f"   {source_name}: {result.get('sentiment', 0):.2f}")
                    
                except Exception as e:
                    logger.error(f"❌ Error in {source_name} analysis: {str(e)}")
                    sentiment_data[source_name] = {'sentiment': 0, 'confidence': 0}
            
            # ترکیب نتایج
            combined_sentiment = self._combine_sentiments(sentiment_data)
            
            # ذخیره در cache
            self._cache_data(cache_key, combined_sentiment)
            
            logger.info(f"📊 Combined sentiment: {combined_sentiment['sentiment_score']:.2f} (conf: {combined_sentiment['confidence']:.2f})")
            
            return combined_sentiment
            
        except Exception as e:
            logger.error(f"❌ Error getting market sentiment: {str(e)}")
            return self._get_neutral_sentiment()
    
    def _combine_sentiments(self, sentiment_data: Dict) -> Dict:
        """
        ترکیب هوشمندانه sentiment از منابع مختلف
        """
        try:
            weighted_sentiment = 0
            total_weight = 0
            confidence_scores = []
            
            for source, data in sentiment_data.items():
                if data and 'sentiment' in data:
                    weight = self.source_weights.get(source, 0.1)
                    confidence = data.get('confidence', 0.5)
                    sentiment = data.get('sentiment', 0)
                    
                    # اعمال وزن و confidence
                    effective_weight = weight * confidence
                    weighted_sentiment += sentiment * effective_weight
                    total_weight += effective_weight
                    confidence_scores.append(confidence)
            
            if total_weight > 0:
                final_sentiment = weighted_sentiment / total_weight
                avg_confidence = np.mean(confidence_scores) if confidence_scores else 0
                
                return {
                    'sentiment_score': max(-1, min(1, final_sentiment)),  # محدود کردن به [-1, 1]
                    'confidence': avg_confidence,
                    'sources_used': len([s for s in sentiment_data.values() if s.get('sentiment') is not None]),
                    'breakdown': sentiment_data,
                    'analysis_time': datetime.now().isoformat()
                }
            
            return self._get_neutral_sentiment()
            
        except Exception as e:
            logger.error(f"❌ Error combining sentiments: {str(e)}")
            return self._get_neutral_sentiment()
    
    def _get_neutral_sentiment(self) -> Dict:
        """sentiment خنثی در صورت خطا"""
        return {
            'sentiment_score': 0,
            'confidence': 0,
            'sources_used': 0,
            'breakdown': {},
            'analysis_time': datetime.now().isoformat()
        }
    
    def _is_cached(self, cache_key: str) -> bool:
        """بررسی وجود داده در cache"""
        if cache_key not in self.cache:
            return False
        
        cache_time = self.cache[cache_key]['timestamp']
        return (time.time() - cache_time) < self.cache_duration
    
    def _cache_data(self, cache_key: str, data: Dict):
        """ذخیره داده در cache"""
        self.cache[cache_key] = {
            'data': data,
            'timestamp': time.time()
        }

class FearGreedAnalyzer:
    """
    تحلیل Fear & Greed Index
    """
    
    async def analyze_sentiment(self, symbol: str) -> Dict:
        """
        دریافت و تحلیل Fear & Greed Index
        """
        try:
            url = "https://api.alternative.me/fng/"
            
            timeout = aiohttp.ClientTimeout(total=10)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        if data.get('data'):
                            fng_value = int(data['data'][0]['value'])
                            classification = data['data'][0]['value_classification']
                            
                            # تبدیل به sentiment score (-1 تا 1)
                            sentiment_score = self._fng_to_sentiment(fng_value)
                            
                            return {
                                'sentiment': sentiment_score,
                                'confidence': 0.8,  # Fear & Greed Index معتبر است
                                'raw_value': fng_value,
                                'classification': classification
                            }
            
            return {'sentiment': 0, 'confidence': 0}
            
        except Exception as e:
            logger.error(f"❌ Fear & Greed analysis error: {str(e)}")
            return {'sentiment': 0, 'confidence': 0}
    
    def _fng_to_sentiment(self, fng_value: int) -> float:
        """
        تبدیل Fear & Greed Index به sentiment score
        """
        # فرمول: (value - 50) / 50
        # 0-20: خیلی ترسناک = -0.6 تا -1.0
        # 20-40: ترسناک = -0.2 تا -0.6  
        # 40-60: خنثی = -0.2 تا 0.2
        # 60-80: طمع = 0.2 تا 0.6
        # 80-100: خیلی طمع = 0.6 تا 1.0
        
        if fng_value <= 20:
            return -0.8  # خیلی ترسناک
        elif fng_value <= 40:
            return -0.4  # ترسناک
        elif fng_value <= 60:
            return 0.0   # خنثی
        elif fng_value <= 80:
            return 0.4   # طمع
        else:
            return 0.8   # خیلی طمع

class NewsAnalyzer:
    """
    تحلیل اخبار کریپتو
    """
    
    def __init__(self):
        self.crypto_keywords = {
            # کلمات مثبت
            'bullish': 0.6, 'bull': 0.5, 'pump': 0.7, 'moon': 0.8, 'rise': 0.4,
            'surge': 0.6, 'rally': 0.6, 'gain': 0.4, 'profit': 0.5, 'up': 0.3,
            'breakthrough': 0.7, 'adoption': 0.6, 'partnership': 0.5,
            
            # کلمات منفی  
            'bearish': -0.6, 'bear': -0.5, 'dump': -0.7, 'crash': -0.8, 'fall': -0.4,
            'drop': -0.4, 'decline': -0.4, 'loss': -0.5, 'down': -0.3, 'fud': -0.6,
            'correction': -0.3, 'pullback': -0.2, 'regulation': -0.4
        }
    
    async def analyze_sentiment(self, symbol: str) -> Dict:
        """
        تحلیل sentiment اخبار
        """
        try:
            # برای سادگی، از CoinGecko استفاده می‌کنیم
            coin_id = self._symbol_to_coingecko_id(symbol)
            
            if not coin_id:
                return {'sentiment': 0, 'confidence': 0}
            
            # دریافت اخبار اخیر
            news_data = await self._fetch_recent_news(coin_id)
            
            if not news_data:
                return {'sentiment': 0, 'confidence': 0}
            
            # تحلیل sentiment اخبار
            sentiment_scores = []
            
            for news_item in news_data[:10]:  # آخرین 10 خبر
                title = news_item.get('title', '')
                description = news_item.get('description', '')
                
                text = f"{title} {description}"
                sentiment = self._analyze_text_sentiment(text)
                sentiment_scores.append(sentiment)
            
            if sentiment_scores:
                avg_sentiment = np.mean(sentiment_scores)
                confidence = min(len(sentiment_scores) / 10, 0.8)  # حداکثر 80%
                
                return {
                    'sentiment': avg_sentiment,
                    'confidence': confidence,
                    'news_count': len(sentiment_scores)
                }
            
            return {'sentiment': 0, 'confidence': 0}
            
        except Exception as e:
            logger.error(f"❌ News analysis error: {str(e)}")
            return {'sentiment': 0, 'confidence': 0}
    
    def _symbol_to_coingecko_id(self, symbol: str) -> str:
        """تبدیل symbol به CoinGecko ID"""
        mapping = {
            'BTCUSDT': 'bitcoin',
            'ETHUSDT': 'ethereum', 
            'BNBUSDT': 'binancecoin',
            'ADAUSDT': 'cardano',
            'SOLUSDT': 'solana',
            'DOTUSDT': 'polkadot',
            'LINKUSDT': 'chainlink',
            'MATICUSDT': 'matic-network',
            'LTCUSDT': 'litecoin',
            'XRPUSDT': 'ripple'
        }
        return mapping.get(symbol, '')
    
    async def _fetch_recent_news(self, coin_id: str) -> List:
        """دریافت اخبار اخیر"""
        try:
            # برای سادگی، اخبار شبیه‌سازی شده
            # در پیاده‌سازی واقعی از API های خبری استفاده کنید
            return [
                {
                    'title': f'{coin_id} shows strong momentum today',
                    'description': 'Market analysis suggests positive outlook'
                },
                {
                    'title': f'Technical analysis bullish for {coin_id}',
                    'description': 'Indicators point to continued growth'
                }
            ]
            
        except Exception as e:
            logger.error(f"❌ Error fetching news: {str(e)}")
            return []
    
    def _analyze_text_sentiment(self, text: str) -> float:
        """تحلیل sentiment متن"""
        try:
            # پاکسازی متن
            text = text.lower()
            text = re.sub(r'[^\w\s]', '', text)
            
            # محاسبه sentiment با کلمات کریپتو
            sentiment_score = 0
            word_count = 0
            
            words = text.split()
            
            for word in words:
                if word in self.crypto_keywords:
                    sentiment_score += self.crypto_keywords[word]
                    word_count += 1
            
            # استفاده از TextBlob برای sentiment پایه
            try:
                blob = TextBlob(text)
                base_sentiment = blob.sentiment.polarity
            except:
                base_sentiment = 0
            
            # ترکیب sentiment ها
            if word_count > 0:
                crypto_sentiment = sentiment_score / word_count
                final_sentiment = (crypto_sentiment * 0.7) + (base_sentiment * 0.3)
            else:
                final_sentiment = base_sentiment
            
            return max(-1, min(1, final_sentiment))
            
        except Exception as e:
            logger.error(f"❌ Text sentiment analysis error: {str(e)}")
            return 0

class OnChainAnalyzer:
    """
    تحلیل داده‌های on-chain
    """
    
    async def analyze_sentiment(self, symbol: str) -> Dict:
        """
        تحلیل sentiment بر اساس داده‌های on-chain
        """
        try:
            if symbol != 'BTCUSDT':
                # فقط برای Bitcoin داده‌های on-chain داریم
                return {'sentiment': 0, 'confidence': 0.1}
            
            # دریافت معیارهای on-chain
            metrics = await self._fetch_onchain_metrics()
            
            if not metrics:
                return {'sentiment': 0, 'confidence': 0}
            
            # محاسبه sentiment
            sentiment_score = self._calculate_onchain_sentiment(metrics)
            
            return {
                'sentiment': sentiment_score,
                'confidence': 0.9,  # داده‌های on-chain قابل اعتماد هستند
                'metrics': metrics
            }
            
        except Exception as e:
            logger.error(f"❌ On-chain analysis error: {str(e)}")
            return {'sentiment': 0, 'confidence': 0}
    
    async def _fetch_onchain_metrics(self) -> Dict:
        """
        دریافت معیارهای on-chain (شبیه‌سازی شده)
        """
        try:
            # در پیاده‌سازی واقعی از API هایی مثل Glassnode استفاده کنید
            # فعلاً داده‌های شبیه‌سازی شده
            
            metrics = {
                'active_addresses_7d': np.random.randint(900000, 1100000),
                'transaction_volume_24h': np.random.uniform(10e9, 20e9),
                'exchange_flow_ratio': np.random.uniform(0.8, 1.2),
                'long_term_holder_supply': np.random.uniform(0.6, 0.75),
                'mvrv_ratio': np.random.uniform(0.8, 2.0),
                'nvt_ratio': np.random.uniform(50, 150)
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"❌ Error fetching on-chain metrics: {str(e)}")
            return {}
    
    def _calculate_onchain_sentiment(self, metrics: Dict) -> float:
        """
        محاسبه sentiment بر اساس معیارهای on-chain
        """
        try:
            sentiment_factors = []
            
            # 1. Active Addresses (بیشتر = مثبت)
            active_addresses = metrics.get('active_addresses_7d', 1000000)
            if active_addresses > 1000000:
                sentiment_factors.append(0.3)
            elif active_addresses < 800000:
                sentiment_factors.append(-0.3)
            
            # 2. Exchange Flow Ratio (کمتر = مثبت - hodling)
            exchange_flow = metrics.get('exchange_flow_ratio', 1.0)
            if exchange_flow < 0.9:
                sentiment_factors.append(0.4)  # کمتر جریان به صرافی = نگهداری
            elif exchange_flow > 1.1:
                sentiment_factors.append(-0.4)  # بیشتر جریان به صرافی = فروش
            
            # 3. Long-term Holder Supply (بیشتر = مثبت)
            lth_supply = metrics.get('long_term_holder_supply', 0.65)
            if lth_supply > 0.7:
                sentiment_factors.append(0.3)  # دست‌های قوی
            elif lth_supply < 0.6:
                sentiment_factors.append(-0.2)
            
            # 4. MVRV Ratio (نزدیک 1 = مثبت)
            mvrv = metrics.get('mvrv_ratio', 1.0)
            if 0.8 <= mvrv <= 1.2:
                sentiment_factors.append(0.2)  # fair value
            elif mvrv > 2.0:
                sentiment_factors.append(-0.4)  # overvalued
            elif mvrv < 0.5:
                sentiment_factors.append(0.3)  # undervalued
            
            # محاسبه میانگین
            if sentiment_factors:
                return np.mean(sentiment_factors)
            
            return 0
            
        except Exception as e:
            logger.error(f"❌ Error calculating on-chain sentiment: {str(e)}")
            return 0

class SocialAnalyzer:
    """
    تحلیل شبکه‌های اجتماعی (ساده شده)
    """
    
    async def analyze_sentiment(self, symbol: str) -> Dict:
        """
        تحلیل sentiment شبکه‌های اجتماعی
        """
        try:
            # در پیاده‌سازی واقعی از Twitter API، Reddit API استفاده کنید
            # فعلاً شبیه‌سازی ساده
            
            social_sentiment = np.random.uniform(-0.5, 0.5)
            confidence = 0.4  # کمتر قابل اعتماد
            
            return {
                'sentiment': social_sentiment,
                'confidence': confidence,
                'mentions_count': np.random.randint(50, 200)
            }
            
        except Exception as e:
            logger.error(f"❌ Social analysis error: {str(e)}")
            return {'sentiment': 0, 'confidence': 0}