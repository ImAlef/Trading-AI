import requests
import pandas as pd
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging
from config import config

logger = logging.getLogger(__name__)

class FallbackDataCollector:
    """
    Synchronous fallback data collector using requests library
    """
    def __init__(self):
        self.base_url = "https://api.binance.com/api/v3"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        self.rate_limit_delay = 60 / config.BINANCE_RATE_LIMIT
        
    def _make_request(self, url: str, params: dict = None, retries: int = 3) -> Optional[dict]:
        """
        Make HTTP request with retry logic
        """
        for attempt in range(retries):
            try:
                logger.info(f"Making request to {url} (attempt {attempt + 1}/{retries})")
                
                response = self.session.get(url, params=params, timeout=30)
                
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 429:  # Rate limit
                    logger.warning(f"Rate limit hit, waiting...")
                    time.sleep(2 ** attempt)
                    continue
                else:
                    logger.error(f"HTTP {response.status_code}: {response.text}")
                    
            except requests.exceptions.Timeout:
                logger.warning(f"Timeout on attempt {attempt + 1}, retrying...")
                time.sleep(2 ** attempt)
            except requests.exceptions.ConnectionError:
                logger.warning(f"Connection error on attempt {attempt + 1}, retrying...")
                time.sleep(2 ** attempt)
            except Exception as e:
                logger.error(f"Request failed on attempt {attempt + 1}: {str(e)}")
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    return None
        
        return None
    
    def get_klines(self, symbol: str, interval: str, limit: int = 500) -> pd.DataFrame:
        """
        Get candlestick data from Binance
        """
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit
        }
        
        url = f"{self.base_url}/klines"
        data = self._make_request(url, params)
        
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
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        """
        Get current price for a symbol
        """
        params = {"symbol": symbol}
        url = f"{self.base_url}/ticker/price"
        
        data = self._make_request(url, params)
        if data:
            try:
                return float(data['price'])
            except (KeyError, ValueError) as e:
                logger.error(f"Error parsing price data for {symbol}: {str(e)}")
                return None
        
        return None
    
    def get_24h_ticker(self, symbol: str) -> Dict:
        """
        Get 24h ticker statistics
        """
        params = {"symbol": symbol}
        url = f"{self.base_url}/ticker/24hr"
        
        data = self._make_request(url, params)
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
    
    def get_multiple_prices(self, symbols: List[str]) -> Dict[str, float]:
        """
        Get current prices for multiple symbols
        """
        url = f"{self.base_url}/ticker/price"
        
        data = self._make_request(url)
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
    
    def get_market_sentiment(self) -> Dict:
        """
        Get market sentiment from Fear & Greed Index
        """
        try:
            url = "https://api.alternative.me/fng/"
            response = self.session.get(url, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                if data['data']:
                    return {
                        'fear_greed_index': int(data['data'][0]['value']),
                        'classification': data['data'][0]['value_classification'],
                        'timestamp': data['data'][0]['timestamp']
                    }
                    
        except Exception as e:
            logger.error(f"Error fetching market sentiment: {str(e)}")
            
        return {'fear_greed_index': 50, 'classification': 'Neutral', 'timestamp': str(int(time.time()))}
    
    def collect_market_data(self, symbols: List[str], timeframe: str = "1h") -> Dict[str, pd.DataFrame]:
        """
        Collect market data for multiple symbols
        """
        market_data = {}
        
        for symbol in symbols:
            try:
                # Rate limiting
                time.sleep(self.rate_limit_delay)
                
                df = self.get_klines(symbol, timeframe)
                if not df.empty:
                    market_data[symbol] = df
                    
            except Exception as e:
                logger.error(f"Error collecting data for {symbol}: {str(e)}")
                continue
        
        logger.info(f"Collected data for {len(market_data)} symbols")
        return market_data
    
    def get_historical_klines(self, symbol: str, interval: str, start_time: datetime, end_time: datetime = None) -> pd.DataFrame:
        """
        Get historical candlestick data
        """
        if end_time is None:
            end_time = datetime.now()
        
        start_ts = int(start_time.timestamp() * 1000)
        end_ts = int(end_time.timestamp() * 1000)
        
        all_data = []
        current_start = start_ts
        
        logger.info(f"Collecting historical data for {symbol} from {start_time} to {end_time}")
        
        while current_start < end_ts:
            params = {
                "symbol": symbol,
                "interval": interval,
                "startTime": current_start,
                "endTime": end_ts,
                "limit": 1000
            }
            
            data = self._make_request(f"{self.base_url}/klines", params)
            
            if not data:
                logger.warning(f"No data returned for {symbol} at timestamp {current_start}")
                break
                
            all_data.extend(data)
            current_start = data[-1][0] + 1  # Next timestamp
            
            # Rate limiting
            time.sleep(0.1)
            
            # Progress logging
            if len(all_data) % 1000 == 0:
                logger.info(f"Collected {len(all_data)} candles for {symbol}...")
        
        if all_data:
            try:
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
                
                logger.info(f"Successfully collected {len(df)} historical candles for {symbol}")
                return df
                
            except Exception as e:
                logger.error(f"Error processing historical data for {symbol}: {str(e)}")
                return pd.DataFrame()
        
        logger.warning(f"No historical data collected for {symbol}")
        return pd.DataFrame()
    
    def get_server_time(self) -> Optional[int]:
        """
        Get Binance server time
        """
        url = f"{self.base_url}/time"
        data = self._make_request(url)
        
        if data:
            try:
                return int(data['serverTime'])
            except (KeyError, ValueError) as e:
                logger.error(f"Error parsing server time: {str(e)}")
                return None
        
        return None
    
    def ping_server(self) -> bool:
        """
        Ping Binance server to check connectivity
        """
        url = f"{self.base_url}/ping"
        data = self._make_request(url)
        
        return data is not None
    
    def get_exchange_info(self) -> Dict:
        """
        Get exchange information
        """
        url = f"{self.base_url}/exchangeInfo"
        data = self._make_request(url)
        
        if data:
            return data
        
        return {}
    
    def get_symbol_info(self, symbol: str) -> Dict:
        """
        Get information about a specific symbol
        """
        exchange_info = self.get_exchange_info()
        
        if exchange_info and 'symbols' in exchange_info:
            for symbol_info in exchange_info['symbols']:
                if symbol_info['symbol'] == symbol:
                    return symbol_info
        
        return {}
    
    def validate_symbol(self, symbol: str) -> bool:
        """
        Validate if a symbol exists and is tradeable
        """
        symbol_info = self.get_symbol_info(symbol)
        
        if symbol_info:
            return symbol_info.get('status') == 'TRADING'
        
        return False
    
    def get_recent_trades(self, symbol: str, limit: int = 500) -> List[Dict]:
        """
        Get recent trades for a symbol
        """
        params = {
            "symbol": symbol,
            "limit": limit
        }
        
        url = f"{self.base_url}/trades"
        data = self._make_request(url, params)
        
        if data:
            return data
        
        return []
    
    def get_order_book(self, symbol: str, limit: int = 100) -> Dict:
        """
        Get order book for a symbol
        """
        params = {
            "symbol": symbol,
            "limit": limit
        }
        
        url = f"{self.base_url}/depth"
        data = self._make_request(url, params)
        
        if data:
            return data
        
        return {}
    
    def get_avg_price(self, symbol: str) -> Optional[float]:
        """
        Get average price for a symbol
        """
        params = {"symbol": symbol}
        url = f"{self.base_url}/avgPrice"
        
        data = self._make_request(url, params)
        if data:
            try:
                return float(data['price'])
            except (KeyError, ValueError) as e:
                logger.error(f"Error parsing average price for {symbol}: {str(e)}")
                return None
        
        return None
    
    def health_check(self) -> Dict:
        """
        Perform comprehensive health check
        """
        health_status = {
            'ping': False,
            'server_time': None,
            'exchange_info': False,
            'sample_price': None,
            'overall_status': 'DOWN'
        }
        
        try:
            # Test ping
            health_status['ping'] = self.ping_server()
            
            # Test server time
            health_status['server_time'] = self.get_server_time()
            
            # Test exchange info
            exchange_info = self.get_exchange_info()
            health_status['exchange_info'] = bool(exchange_info)
            
            # Test sample price
            health_status['sample_price'] = self.get_current_price('BTCUSDT')
            
            # Overall status
            if all([
                health_status['ping'],
                health_status['server_time'],
                health_status['exchange_info'],
                health_status['sample_price']
            ]):
                health_status['overall_status'] = 'UP'
            else:
                health_status['overall_status'] = 'PARTIAL'
                
        except Exception as e:
            logger.error(f"Error during health check: {str(e)}")
            health_status['error'] = str(e)
        
        return health_status