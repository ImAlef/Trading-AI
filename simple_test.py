#!/usr/bin/env python3
"""
Simple test to check Binance API connection
"""
import requests
import json

def test_binance_connection():
    """Test basic connection to Binance API"""
    
    print("🔗 Testing Binance API Connection...")
    print("="*50)
    
    try:
        # Test 1: Simple ping
        print("\n🏓 Test 1: API Ping...")
        response = requests.get("https://api.binance.com/api/v3/ping", timeout=30)
        
        if response.status_code == 200:
            print("✅ Binance API is reachable")
        else:
            print(f"❌ Ping failed: {response.status_code}")
            return False
            
        # Test 2: Server time
        print("\n⏰ Test 2: Server Time...")
        response = requests.get("https://api.binance.com/api/v3/time", timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            server_time = data['serverTime']
            print(f"✅ Server time: {server_time}")
        else:
            print(f"❌ Time check failed: {response.status_code}")
            return False
            
        # Test 3: Get BTC price
        print("\n💰 Test 3: BTC Price...")
        response = requests.get(
            "https://api.binance.com/api/v3/ticker/price",
            params={"symbol": "BTCUSDT"},
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            price = float(data['price'])
            print(f"✅ BTC Price: ${price:,.2f}")
        else:
            print(f"❌ Price check failed: {response.status_code}")
            return False
            
        # Test 4: Get multiple prices
        print("\n📊 Test 4: Multiple Prices...")
        response = requests.get("https://api.binance.com/api/v3/ticker/price", timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]
            prices = {}
            
            for item in data:
                if item['symbol'] in symbols:
                    prices[item['symbol']] = float(item['price'])
            
            if prices:
                print("✅ Multiple prices fetched:")
                for symbol, price in prices.items():
                    print(f"   {symbol}: ${price:,.2f}")
            else:
                print("❌ No prices found")
                return False
        else:
            print(f"❌ Multiple prices failed: {response.status_code}")
            return False
            
        print("\n🎉 All connection tests passed!")
        return True
        
    except requests.exceptions.Timeout:
        print("❌ Connection timeout - check your internet connection")
        return False
    except requests.exceptions.ConnectionError:
        print("❌ Connection error - check your internet connection or proxy settings")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")
        return False

if __name__ == "__main__":
    print("🤖 Crypto Signal Bot - Simple Connection Test")
    print("="*50)
    
    success = test_binance_connection()
    
    if success:
        print("\n✅ Connection test successful!")
        print("🚀 You can now run the full data collector test")
    else:
        print("\n❌ Connection test failed")
        print("Please check your internet connection and try again")