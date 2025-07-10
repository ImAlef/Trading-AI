import asyncio
import aiohttp
import pandas as pd
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging
from config import config

logger = logging.getLogger(__name__)

class DataCollector:
    def __init__(self):
        self.base_url = "https://api.binance.com/api/v3"
        self.session = None
        self.rate_limit_delay = 60 / config.BINANCE_RATE_LIMIT  # Delay between requests
        self.timeout = aiohttp.ClientTimeout(total=30, connect=10)
        
    async def __aenter__(self):
        # Create session with timeout and SSL configuration
        connector = aiohttp.TCPConnector(
            limit=100,
            ttl_dns_cache=300,
            use_dns_cache=True,
            ssl=False  # For testing, disable SSL verification
        )
        
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=self.timeout,
            headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
        )
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    async def _make_request(self, url: str, params: dict = None, retries: int = 3) -> Optional[dict]:
        """
        Make HTTP request with retry logic
        """
        for attempt in range(retries):
            try:
                logger.info(f"Making request to {url} (attempt {attempt + 1}/{retries})")
                
                async with self.session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data
                    elif response.status == 429:  # Rate limit
                        logger.warning(f"Rate limit hit, waiting...")
                        await asyncio.sleep(2 ** attempt)
                        continue
                    else:
                        logger.error(f"HTTP {response.status}: {await response.text()}")
                        
            except asyncio.TimeoutError:
                logger.warning(f"Timeout on attempt {attempt + 1}, retrying...")
                await asyncio.sleep(2 ** attempt)
            except Exception as e:
                logger.error(f"Request failed on attempt {attempt + 1}: {str(e)}")
                if attempt < retries - 1:
                    await asyncio.sleep(2 ** attempt)
                else:
                    return None
        
        return None
    async def get_klines(self, symbol: str, interval: str, limit: int = 500) -> pd.DataFrame:
        """
        Get candlestick data from Binance
        """
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit
        }
        
        url = f"{self.base_url}/klines"
        data = await self._make_request(url, params)
        
        if data:
            try:
                # Convert to DataFrame
                df = pd.DataFrame(data, columns=[
                    'timestamp', 'open', 'high', 'low', 'close', 'volume',
                    'close_time', 'quote_asset_volume', 'number_of_trades',
                    'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
                ])
                
                # Convert data types
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df['open'] = df['open'].astype(float)
                df['high'] = df['high'].astype(float)
                df['low'] = df['low'].astype(float)
                df['close'] = df['close'].astype(float)
                df['volume'] = df['volume'].astype(float)
                
                # Keep only necessary columns
                df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
                df.set_index('timestamp', inplace=True)
                
                logger.info(f"Successfully fetched {len(df)} candles for {symbol}")
                return df
                
            except Exception as e:
                logger.error(f"Error processing klines data for {symbol}: {str(e)}")
                return pd.DataFrame()
        
        return pd.DataFrame()
    
    async def get_current_price(self, symbol: str) -> Optional[float]:
        """
        Get current price for a symbol
        """
        params = {"symbol": symbol}
        url = f"{self.base_url}/ticker/price"
        
        data = await self._make_request(url, params)
        if data:
            try:
                return float(data['price'])
            except (KeyError, ValueError) as e:
                logger.error(f"Error parsing price data for {symbol}: {str(e)}")
                return None
        
        return None
    
    async def get_24h_ticker(self, symbol: str) -> Dict:
        """
        Get 24h ticker statistics
        """
        params = {"symbol": symbol}
        url = f"{self.base_url}/ticker/24hr"
        
        data = await self._make_request(url, params)
        if data:
            try:
                return {
                    'price_change_percent': float(data['priceChangePercent']),
                    'high_price': float(data['highPrice']),
                    'low_price': float(data['lowPrice']),
                    'volume': float(data['volume']),
                    'quote_volume': float(data['quoteVolume']),
                    'count': int(data['count'])
                }
            except (KeyError, ValueError) as e:
                logger.error(f"Error parsing 24h ticker data for {symbol}: {str(e)}")
                return {}
        
        return {}
    
    async def get_multiple_prices(self, symbols: List[str]) -> Dict[str, float]:
        """
        Get current prices for multiple symbols
        """
        url = f"{self.base_url}/ticker/price"
        
        data = await self._make_request(url)
        if data:
            try:
                prices = {}
                for item in data:
                    if item['symbol'] in symbols:
                        prices[item['symbol']] = float(item['price'])
                
                return prices
            except (KeyError, ValueError) as e:
                logger.error(f"Error parsing multiple prices data: {str(e)}")
                return {}
        
        return {}
    
    async def collect_market_data(self, symbols: List[str], timeframe: str = "1h") -> Dict[str, pd.DataFrame]:
        """
        Collect market data for multiple symbols
        """
        market_data = {}
        
        for symbol in symbols:
            try:
                # Rate limiting
                await asyncio.sleep(self.rate_limit_delay)
                
                df = await self.get_klines(symbol, timeframe)
                if not df.empty:
                    market_data[symbol] = df
                    
            except Exception as e:
                logger.error(f"Error collecting data for {symbol}: {str(e)}")
                continue
        
        logger.info(f"Collected data for {len(market_data)} symbols")
        return market_data
    
    async def get_market_sentiment(self) -> Dict:
        """
        Get market sentiment from Fear & Greed Index (free API)
        """
        try:
            url = "https://api.alternative.me/fng/"
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    if data['data']:
                        return {
                            'fear_greed_index': int(data['data'][0]['value']),
                            'classification': data['data'][0]['value_classification'],
                            'timestamp': data['data'][0]['timestamp']
                        }
                    
        except Exception as e:
            logger.error(f"Error fetching market sentiment: {str(e)}")
            
        return {'fear_greed_index': 50, 'classification': 'Neutral', 'timestamp': str(int(time.time()))}

class HistoricalDataCollector:
    """
    Separate class for collecting historical data (for training)
    """
    def __init__(self):
        self.base_url = "https://api.binance.com/api/v3"
        
    def get_historical_klines(self, symbol: str, interval: str, start_time: datetime, end_time: datetime = None) -> pd.DataFrame:
        """
        Get historical candlestick data
        """
        import requests
        
        if end_time is None:
            end_time = datetime.now()
        
        start_ts = int(start_time.timestamp() * 1000)
        end_ts = int(end_time.timestamp() * 1000)
        
        all_data = []
        current_start = start_ts
        
        while current_start < end_ts:
            params = {
                "symbol": symbol,
                "interval": interval,
                "startTime": current_start,
                "endTime": end_ts,
                "limit": 1000
            }
            
            try:
                response = requests.get(f"{self.base_url}/klines", params=params)
                if response.status_code == 200:
                    data = response.json()
                    if not data:
                        break
                        
                    all_data.extend(data)
                    current_start = data[-1][0] + 1  # Next timestamp
                    
                    # Rate limiting
                    time.sleep(0.5)
                    
                else:
                    logger.error(f"Failed to fetch historical data: {response.status_code}")
                    break
                    
            except Exception as e:
                logger.error(f"Error fetching historical data: {str(e)}")
                break
        
        if all_data:
            df = pd.DataFrame(all_data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            # Convert data types
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['open'] = df['open'].astype(float)
            df['high'] = df['high'].astype(float)
            df['low'] = df['low'].astype(float)
            df['close'] = df['close'].astype(float)
            df['volume'] = df['volume'].astype(float)
            
            # Keep only necessary columns
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            df.set_index('timestamp', inplace=True)
            
            # Remove duplicates
            df = df[~df.index.duplicated(keep='first')]
            df.sort_index(inplace=True)
            
            logger.info(f"Collected {len(df)} historical candles for {symbol}")
            return df
        
        return pd.DataFrame()
    
    def collect_training_data(self, symbols: List[str], days: int = 180) -> Dict[str, pd.DataFrame]:
        """
        Collect training data for multiple symbols
        """
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)
        
        training_data = {}
        
        for symbol in symbols:
            try:
                df = self.get_historical_klines(symbol, "1h", start_time, end_time)
                if not df.empty and len(df) > 100:  # Minimum data requirement
                    training_data[symbol] = df
                    logger.info(f"Collected training data for {symbol}: {len(df)} candles")
                else:
                    logger.warning(f"Insufficient data for {symbol}")
                    
            except Exception as e:
                logger.error(f"Error collecting training data for {symbol}: {str(e)}")
                continue
        
        return training_data