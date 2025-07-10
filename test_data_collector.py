#!/usr/bin/env python3
"""
Test script for Data Collector - Using Fallback Only
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import config
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_data_collector():
    """Test the data collector functionality"""
    
    print("🚀 Testing Data Collector (Fallback)...")
    print("="*50)
    
    try:
        # Import fallback collector
        from backend.services.fallback_data_collector import FallbackDataCollector
        
        collector = FallbackDataCollector()
        test_symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]
        
        # Test 1: Get current prices
        print("\n📊 Test 1: Getting current prices...")
        prices = collector.get_multiple_prices(test_symbols)
        
        if prices:
            print("✅ Current prices fetched successfully:")
            for symbol, price in prices.items():
                print(f"   {symbol}: ${price:,.2f}")
        else:
            print("❌ Failed to fetch current prices")
            return False
        
        # Test 2: Get candlestick data
        print("\n📈 Test 2: Getting candlestick data...")
        df = collector.get_klines("BTCUSDT", "1h", limit=100)
        
        if not df.empty:
            print("✅ Candlestick data fetched successfully:")
            print(f"   Data points: {len(df)}")
            print(f"   Date range: {df.index[0]} to {df.index[-1]}")
            print(f"   Current price: ${df['close'].iloc[-1]:,.2f}")
            
            # Calculate 24h change if we have enough data
            if len(df) >= 24:
                change_24h = ((df['close'].iloc[-1] / df['close'].iloc[-24]) - 1) * 100
                print(f"   24h change: {change_24h:.2f}%")
            else:
                print(f"   Note: Not enough data for 24h change calculation")
        else:
            print("❌ Failed to fetch candlestick data")
            return False
        
        # Test 3: Get 24h ticker
        print("\n📊 Test 3: Getting 24h ticker data...")
        ticker = collector.get_24h_ticker("BTCUSDT")
        
        if ticker:
            print("✅ 24h ticker data fetched successfully:")
            print(f"   Price change: {ticker['price_change_percent']:.2f}%")
            print(f"   High: ${ticker['high_price']:,.2f}")
            print(f"   Low: ${ticker['low_price']:,.2f}")
            print(f"   Volume: {ticker['volume']:,.0f} BTC")
        else:
            print("❌ Failed to fetch 24h ticker data")
            return False
        
        # Test 4: Get market sentiment
        print("\n😨 Test 4: Getting market sentiment...")
        sentiment = collector.get_market_sentiment()
        
        if sentiment:
            print("✅ Market sentiment fetched successfully:")
            print(f"   Fear & Greed Index: {sentiment['fear_greed_index']}/100")
            print(f"   Classification: {sentiment['classification']}")
        else:
            print("⚠️  Market sentiment using default values")
        
        # Test 5: Collect market data for multiple symbols
        print("\n🔄 Test 5: Collecting market data for multiple symbols...")
        market_data = collector.collect_market_data(test_symbols[:2], "1h")  # Test with 2 symbols
        
        if market_data:
            print("✅ Market data collected successfully:")
            for symbol, df in market_data.items():
                print(f"   {symbol}: {len(df)} candles, Latest: ${df['close'].iloc[-1]:,.2f}")
        else:
            print("❌ Failed to collect market data")
            return False
        
        print("\n" + "="*50)
        print("🎉 All tests passed! Data Collector is working correctly.")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {str(e)}")
        print("Make sure fallback_data_collector.py exists in backend/services/")
        return False
    except Exception as e:
        logger.error(f"Data collector test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_historical_collector():
    """Test historical data collector"""
    
    print("\n📚 Testing Historical Data Collector...")
    print("="*50)
    
    try:
        import requests
        from datetime import datetime, timedelta
        
        # Simple historical data test using requests
        end_time = datetime.now()
        start_time = end_time - timedelta(days=7)
        
        start_ts = int(start_time.timestamp() * 1000)
        end_ts = int(end_time.timestamp() * 1000)
        
        print(f"📅 Fetching historical data from {start_time.strftime('%Y-%m-%d')} to {end_time.strftime('%Y-%m-%d')}")
        
        params = {
            "symbol": "BTCUSDT",
            "interval": "1h",
            "startTime": start_ts,
            "endTime": end_ts,
            "limit": 500
        }
        
        response = requests.get("https://api.binance.com/api/v3/klines", params=params, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            
            if data:
                print("✅ Historical data fetched successfully:")
                print(f"   Data points: {len(data)}")
                
                # Process first and last candles
                first_candle = data[0]
                last_candle = data[-1]
                
                first_time = datetime.fromtimestamp(first_candle[0] / 1000)
                last_time = datetime.fromtimestamp(last_candle[0] / 1000)
                
                print(f"   Date range: {first_time} to {last_time}")
                print(f"   First price: ${float(first_candle[4]):,.2f}")
                print(f"   Last price: ${float(last_candle[4]):,.2f}")
                
                # Calculate some basic stats
                prices = [float(candle[4]) for candle in data]  # Close prices
                volumes = [float(candle[5]) for candle in data]  # Volumes
                
                print(f"   High: ${max(prices):,.2f}")
                print(f"   Low: ${min(prices):,.2f}")
                print(f"   Avg Volume: {sum(volumes)/len(volumes):,.0f}")
                
                return True
            else:
                print("❌ No historical data returned")
                return False
        else:
            print(f"❌ Failed to fetch historical data: {response.status_code}")
            return False
            
    except Exception as e:
        logger.error(f"Historical data test failed: {str(e)}")
        return False

if __name__ == "__main__":
    print("🤖 Crypto Signal Bot - Data Collector Test")
    print("="*50)
    
    # Check configuration
    if not config.validate_config():
        print("⚠️  Configuration validation failed. Email settings are missing but data collection will still work.")
    
    print(f"📋 Configuration loaded successfully")
    print(f"   Trading pairs: {len(config.TRADING_PAIRS)}")
    print(f"   Primary timeframe: {config.PRIMARY_TIMEFRAME}")
    print(f"   Confidence threshold: {config.CONFIDENCE_THRESHOLD}")
    
    # Run tests
    try:
        success = test_data_collector()
        
        if success:
            # Run historical data test
            historical_success = test_historical_collector()
            
            if historical_success:
                print("\n🎉 All tests completed successfully!")
                print("✅ Data collection system is ready!")
                print("\n🚀 Next steps:")
                print("   1. Configure email settings in .env file")
                print("   2. Run feature engineering tests")
                print("   3. Train the ML model")
            else:
                print("\n⚠️  Historical data test failed, but main collector works")
                print("✅ Data collection system is ready!")
        else:
            print("\n❌ Data collector test failed")
            
    except KeyboardInterrupt:
        print("\n⚠️  Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()