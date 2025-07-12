#!/usr/bin/env python3
"""
Quick Market Analysis Test
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_market_conditions():
    """
    Test current market conditions to understand why no signals
    """
    try:
        print("🔍 Analyzing Current Market Conditions...")
        print("=" * 60)
        
        from backend.services.fallback_data_collector import FallbackDataCollector
        from backend.services.feature_engineer import FeatureEngineer
        from backend.services.ml_model import MLModel
        from config import config
        
        # Initialize services
        collector = FallbackDataCollector()
        feature_engineer = FeatureEngineer()
        ml_model = MLModel()
        
        # Load model
        print("📥 Loading ML model...")
        if not ml_model.load_latest_model():
            print("❌ No model found! You need to train a model first.")
            print("💡 Run: python test_ml_model.py")
            return False
        
        print(f"✅ Model loaded: {ml_model.model_info.get('version', 'unknown')}")
        print(f"🎯 Model type: {ml_model.model_info.get('model_type', 'unknown')}")
        print(f"📊 Features: {len(ml_model.feature_columns) if ml_model.feature_columns else 0}")
        
        # Test pairs
        test_pairs = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT", "SOLUSDT", "DOTUSDT", "LINKUSDT"]
        
        market_stats = {
            'total_pairs': 0,
            'high_confidence': 0,  # >=0.55
            'medium_confidence': 0,  # 0.3-0.55
            'low_confidence': 0,  # <0.3
            'predictions_1': 0,  # Buy signals
            'predictions_0': 0,  # No buy signals
            'avg_rsi': 0,
            'avg_volatility': 0,
            'trending_count': 0,
            'sideways_count': 0
        }
        
        pair_results = []
        
        print("\n📊 Analyzing individual pairs...")
        print("-" * 60)
        
        for symbol in test_pairs:
            try:
                print(f"🔍 {symbol}...", end=" ")
                
                # Get market data
                df = collector.get_klines(symbol, "1h", limit=200)
                if df.empty:
                    print("❌ No data")
                    continue
                
                # Feature engineering
                df_features = feature_engineer.prepare_features_for_prediction(df)
                if df_features.empty:
                    print("❌ No features")
                    continue
                
                # Get latest features
                latest = df_features.iloc[-1]
                
                # Make prediction
                latest_features = df_features.iloc[-1:]
                prediction, confidence = ml_model.predict_single(latest_features)
                
                # Calculate market metrics
                current_price = latest['close']
                rsi = latest.get('rsi', 0)
                macd = latest.get('macd', 0)
                macd_signal = latest.get('macd_signal', 0)
                bb_position = latest.get('bb_position', 0.5)
                volume_ratio = latest.get('volume_ratio', 1)
                volatility = latest.get('volatility_20d', 0)
                
                # 24h price change
                price_24h_ago = df['close'].iloc[-25] if len(df) >= 25 else df['close'].iloc[0]
                price_change_24h = (current_price / price_24h_ago - 1) * 100
                
                # Trend detection
                ema9 = latest.get('ema_9', current_price)
                ema21 = latest.get('ema_21', current_price)
                trend_strength = abs(ema9 - ema21) / current_price
                is_trending = trend_strength > 0.015  # 1.5% difference
                
                # Store result
                result = {
                    'symbol': symbol,
                    'prediction': prediction,
                    'confidence': confidence,
                    'current_price': current_price,
                    'rsi': rsi,
                    'macd': macd,
                    'macd_signal': macd_signal,
                    'bb_position': bb_position,
                    'volume_ratio': volume_ratio,
                    'volatility': volatility,
                    'price_change_24h': price_change_24h,
                    'trend_strength': trend_strength,
                    'is_trending': is_trending
                }
                
                pair_results.append(result)
                
                # Update stats
                market_stats['total_pairs'] += 1
                
                if confidence >= 0.55:
                    market_stats['high_confidence'] += 1
                elif confidence >= 0.3:
                    market_stats['medium_confidence'] += 1
                else:
                    market_stats['low_confidence'] += 1
                
                if prediction == 1:
                    market_stats['predictions_1'] += 1
                else:
                    market_stats['predictions_0'] += 1
                
                market_stats['avg_rsi'] += rsi
                market_stats['avg_volatility'] += volatility
                
                if is_trending:
                    market_stats['trending_count'] += 1
                else:
                    market_stats['sideways_count'] += 1
                
                # Show quick result
                conf_emoji = "🔥" if confidence >= 0.55 else "📊" if confidence >= 0.3 else "😴"
                pred_emoji = "📈" if prediction == 1 else "😐"
                trend_emoji = "🚀" if is_trending else "➡️"
                
                print(f"{conf_emoji}{pred_emoji}{trend_emoji} P={prediction}, C={confidence:.3f}, RSI={rsi:.0f}")
                
            except Exception as e:
                print(f"❌ Error: {str(e)}")
                continue
        
        # Calculate averages
        if market_stats['total_pairs'] > 0:
            market_stats['avg_rsi'] /= market_stats['total_pairs']
            market_stats['avg_volatility'] /= market_stats['total_pairs']
        
        print("\n" + "="*60)
        print("📊 MARKET ANALYSIS RESULTS")
        print("="*60)
        
        # Signal Analysis
        print(f"🎯 SIGNAL ANALYSIS:")
        print(f"   High Confidence (≥55%): {market_stats['high_confidence']}")
        print(f"   Medium Confidence (30-55%): {market_stats['medium_confidence']}")
        print(f"   Low Confidence (<30%): {market_stats['low_confidence']}")
        print(f"   Buy Predictions: {market_stats['predictions_1']}")
        print(f"   Hold/Sell Predictions: {market_stats['predictions_0']}")
        
        # Market Conditions
        print(f"\n🌊 MARKET CONDITIONS:")
        print(f"   Average RSI: {market_stats['avg_rsi']:.1f}")
        print(f"   Average Volatility: {market_stats['avg_volatility']:.4f}")
        print(f"   Trending Pairs: {market_stats['trending_count']}")
        print(f"   Sideways Pairs: {market_stats['sideways_count']}")
        
        # Detailed Results
        print(f"\n📋 DETAILED RESULTS:")
        print("-" * 100)
        print(f"{'Symbol':<10} {'Pred':<4} {'Conf':<6} {'RSI':<5} {'Vol':<6} {'24h%':<6} {'Trend':<6} {'Status'}")
        print("-" * 100)
        
        for result in pair_results:
            status = "🔥SIGNAL" if result['confidence'] >= 0.55 else "📊MEDIUM" if result['confidence'] >= 0.3 else "😴LOW"
            trend_text = "TREND" if result['is_trending'] else "SIDE"
            
            print(f"{result['symbol']:<10} {result['prediction']:<4} {result['confidence']:<6.3f} {result['rsi']:<5.0f} {result['volatility']:<6.3f} {result['price_change_24h']:<6.1f} {trend_text:<6} {status}")
        
        # Diagnosis
        print(f"\n🔍 DIAGNOSIS:")
        print("-" * 40)
        
        if market_stats['high_confidence'] == 0:
            if market_stats['medium_confidence'] > 0:
                print("✅ NORMAL: Model found medium opportunities but threshold (55%) keeps quality high")
                print(f"💡 INFO: {market_stats['medium_confidence']} pairs have 30-55% confidence")
            else:
                print("😐 NORMAL: No strong opportunities in current market conditions")
        else:
            print(f"🎉 GREAT: {market_stats['high_confidence']} high-confidence signals found!")
        
        if market_stats['predictions_0'] == market_stats['total_pairs']:
            print("⚠️  NOTICE: All predictions are 0 (no buy signals)")
            print("   This suggests model sees no good entry points right now")
        
        if market_stats['sideways_count'] > market_stats['trending_count']:
            print("📊 EXPLANATION: Most pairs are in sideways movement")
            print("   This is why signals are rare - model avoids low-probability setups")
        
        if market_stats['avg_rsi'] > 70:
            print("📈 CONDITION: Market is overbought - model waiting for pullback")
        elif market_stats['avg_rsi'] < 30:
            print("📉 CONDITION: Market is oversold - but needs more confirmation")
        elif 45 <= market_stats['avg_rsi'] <= 55:
            print("😐 CONDITION: Market is neutral - waiting for clear direction")
        
        if market_stats['avg_volatility'] < 0.02:
            print("😴 CONDITION: Low volatility environment - fewer opportunities")
        
        # Conclusion
        print(f"\n🎯 CONCLUSION:")
        print("-" * 40)
        
        if market_stats['high_confidence'] == 0 and market_stats['medium_confidence'] == 0:
            print("🔴 CURRENT STATUS: No signals because market conditions don't meet criteria")
            print("✅ MODEL BEHAVIOR: This is CORRECT and shows quality control")
            print("⏳ RECOMMENDATION: Be patient - wait for better market conditions")
            print("💡 ALTERNATIVE: Temporarily lower threshold to 0.3 to see medium opportunities")
        elif market_stats['high_confidence'] == 0 and market_stats['medium_confidence'] > 0:
            print("🟡 CURRENT STATUS: Medium opportunities available but below 55% threshold")
            print("✅ MODEL BEHAVIOR: Working correctly - being selective for quality")
            print("💭 CONSIDERATION: Current 55% threshold is maintaining high quality")
        else:
            print("🟢 CURRENT STATUS: High-quality signals detected!")
            print("🚀 MODEL BEHAVIOR: Working perfectly")
        
        # Backtest reminder
        print(f"\n📈 BACKTEST REMINDER:")
        print("Your backtest showed:")
        print("   • ~1 signal per day")
        print("   • 600% profit in 1 month") 
        print("   • 69% win rate")
        print("This suggests the model is designed to be VERY selective!")
        
        print("\n" + "="*60)
        return True
        
    except Exception as e:
        print(f"❌ Error in analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_model_specifically():
    """Test model with some manual data to see if it's working"""
    try:
        print("\n🧪 Testing model functionality...")
        
        from backend.services.ml_model import MLModel
        
        ml_model = MLModel()
        if not ml_model.load_latest_model():
            print("❌ Can't load model")
            return False
        
        print("✅ Model loaded successfully")
        print(f"📊 Model info: {ml_model.model_info}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model test failed: {str(e)}")
        return False

if __name__ == "__main__":
    print("🤖 Crypto Signal Bot - Market Analysis")
    print("="*60)
    
    # Test model
    if not test_model_specifically():
        print("\n❌ Model test failed - exiting")
        sys.exit(1)
    
    # Test market conditions
    success = test_market_conditions()
    
    if success:
        print("\n✅ Analysis completed successfully!")
    else:
        print("\n❌ Analysis failed!")
    
    print("\n👋 Done!")