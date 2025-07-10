#!/usr/bin/env python3
"""
Test script for Feature Engineer
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backend.services.fallback_data_collector import FallbackDataCollector
from backend.services.feature_engineer import FeatureEngineer
from config import config
import logging
import pandas as pd

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_feature_engineer():
    """Test the feature engineering functionality"""
    
    print("🔧 Testing Feature Engineer...")
    print("="*50)
    
    try:
        # Step 1: Get sample data
        print("\n📊 Step 1: Collecting sample data...")
        collector = FallbackDataCollector()
        df = collector.get_klines("BTCUSDT", "1h", limit=500)
        
        if df.empty:
            print("❌ Failed to get sample data")
            return False
        
        print(f"✅ Collected {len(df)} candles")
        print(f"   Date range: {df.index[0]} to {df.index[-1]}")
        print(f"   Columns: {list(df.columns)}")
        
        # Step 2: Initialize feature engineer
        print("\n🔧 Step 2: Initializing Feature Engineer...")
        fe = FeatureEngineer()
        print("✅ Feature Engineer initialized")
        
        # Step 3: Test technical indicators
        print("\n📈 Step 3: Testing technical indicators...")
        df_with_indicators = fe.calculate_technical_indicators(df)
        
        if len(df_with_indicators.columns) > len(df.columns):
            new_indicators = len(df_with_indicators.columns) - len(df.columns)
            print(f"✅ Added {new_indicators} technical indicators")
            
            # Show some indicators
            indicator_cols = ['rsi', 'macd', 'bb_upper', 'bb_lower', 'volume_ratio']
            for col in indicator_cols:
                if col in df_with_indicators.columns:
                    latest_val = df_with_indicators[col].iloc[-1]
                    print(f"   {col}: {latest_val:.4f}")
        else:
            print("❌ No indicators were added")
            return False
        
        # Step 4: Test price features
        print("\n💰 Step 4: Testing price features...")
        df_with_price = fe.calculate_price_features(df_with_indicators)
        
        price_features = ['price_change', 'hl_spread', 'volatility_20d', 'momentum_5']
        added_features = [col for col in price_features if col in df_with_price.columns]
        
        if added_features:
            print(f"✅ Added {len(added_features)} price features")
            for col in added_features[:3]:  # Show first 3
                latest_val = df_with_price[col].iloc[-1]
                print(f"   {col}: {latest_val:.4f}")
        else:
            print("❌ No price features were added")
            return False
        
        # Step 5: Test trend features
        print("\n📊 Step 5: Testing trend features...")
        df_with_trend = fe.calculate_trend_features(df_with_price)
        
        trend_features = ['ma_trend_short', 'rsi_oversold', 'bb_squeeze', 'volume_above_avg']
        added_trend = [col for col in trend_features if col in df_with_trend.columns]
        
        if added_trend:
            print(f"✅ Added {len(added_trend)} trend features")
            for col in added_trend[:3]:  # Show first 3
                latest_val = df_with_trend[col].iloc[-1]
                print(f"   {col}: {latest_val}")
        else:
            print("❌ No trend features were added")
            return False
        
        # Step 6: Test pattern features
        print("\n🕯️ Step 6: Testing pattern features...")
        df_with_patterns = fe.calculate_pattern_features(df_with_trend)
        
        pattern_features = ['body_size', 'bullish_candle', 'hammer', 'doji']
        added_patterns = [col for col in pattern_features if col in df_with_patterns.columns]
        
        if added_patterns:
            print(f"✅ Added {len(added_patterns)} pattern features")
            for col in added_patterns[:3]:  # Show first 3
                latest_val = df_with_patterns[col].iloc[-1]
                print(f"   {col}: {latest_val}")
        else:
            print("❌ No pattern features were added")
            return False
        
        # Step 7: Test target creation
        print("\n🎯 Step 7: Testing target variable creation...")
        df_with_target = fe.create_target_variable(df_with_patterns)
        
        if 'target' in df_with_target.columns:
            target_sum = df_with_target['target'].sum()
            target_pct = (target_sum / len(df_with_target)) * 100
            print(f"✅ Created target variable")
            print(f"   Positive samples: {target_sum}/{len(df_with_target)} ({target_pct:.1f}%)")
            print(f"   Target return threshold: {config.MIN_PROFIT_TARGET*100:.1f}%")
        else:
            print("❌ Failed to create target variable")
            return False
        
        # Step 8: Test complete pipeline
        print("\n🚀 Step 8: Testing complete feature engineering pipeline...")
        final_df = fe.prepare_features(df)
        
        if not final_df.empty:
            print(f"✅ Complete pipeline successful")
            print(f"   Final dataset: {len(final_df)} rows × {len(final_df.columns)} columns")
            print(f"   Original data: {len(df)} rows")
            print(f"   Data reduction: {((len(df) - len(final_df)) / len(df)) * 100:.1f}%")
            
            # Show feature categories
            feature_cols = fe.get_feature_columns()
            available_features = [col for col in feature_cols if col in final_df.columns]
            print(f"   Available features: {len(available_features)}")
            
            # Show sample of features
            print("\n📊 Sample of final features:")
            sample_features = available_features[:5]
            for col in sample_features:
                latest_val = final_df[col].iloc[-1]
                print(f"   {col}: {latest_val:.4f}")
                
        else:
            print("❌ Complete pipeline failed")
            return False
        
        # Step 9: Test feature importance names
        print("\n📝 Step 9: Testing feature importance names...")
        importance_names = fe.get_feature_importance_names()
        
        if importance_names:
            print(f"✅ Feature importance names ready ({len(importance_names)} features)")
            sample_names = list(importance_names.items())[:3]
            for key, name in sample_names:
                print(f"   {key} → {name}")
        else:
            print("❌ No feature importance names found")
            return False
        
        print("\n" + "="*50)
        print("🎉 All feature engineering tests passed!")
        print(f"✅ Feature engineering system is ready!")
        
        # Summary statistics
        print("\n📊 Summary:")
        print(f"   📈 Technical indicators: Working")
        print(f"   💰 Price features: Working")
        print(f"   📊 Trend features: Working")
        print(f"   🕯️ Pattern features: Working")
        print(f"   🎯 Target variable: Working")
        print(f"   📊 Final dataset: {len(final_df)} samples with {len(available_features)} features")
        
        return True
        
    except Exception as e:
        logger.error(f"Feature engineering test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🤖 Crypto Signal Bot - Feature Engineering Test")
    print("="*50)
    
    print(f"📋 Configuration:")
    print(f"   Target return: {config.MIN_PROFIT_TARGET*100:.1f}%")
    print(f"   Signal expiry: {config.SIGNAL_EXPIRY_HOURS} hours")
    print(f"   Lookback periods: {config.LOOKBACK_PERIODS}")
    
    try:
        success = test_feature_engineer()
        
        if success:
            print("\n🎉 Feature engineering test completed successfully!")
            print("✅ Ready for ML model training!")
            print("\n🚀 Next steps:")
            print("   1. Train the ML model")
            print("   2. Test signal generation")
            print("   3. Setup email alerts")
        else:
            print("\n❌ Feature engineering test failed")
            
    except KeyboardInterrupt:
        print("\n⚠️  Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()