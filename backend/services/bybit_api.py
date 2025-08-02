import requests
import hashlib
import hmac
import time
import json
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class Position:
    """Active position data"""
    symbol: str
    side: str  # 'Buy' or 'Sell'
    size: float
    entry_price: float
    mark_price: float
    pnl: float
    margin: float
    leverage: int
    position_id: str
    created_at: datetime

@dataclass
class OrderResult:
    """Order execution result"""
    success: bool
    order_id: Optional[str] = None
    message: str = ""
    price: Optional[float] = None
    quantity: Optional[float] = None

class BybitAPI:
    """
    Bybit Futures API Integration
    """
    
    def __init__(self, api_key: str, secret_key: str, testnet: bool = True):
        self.api_key = api_key
        self.secret_key = secret_key
        self.testnet = testnet
        
        # API endpoints
        if testnet:
            self.base_url = "https://api-testnet.bybit.com"
        else:
            self.base_url = "https://api.bybit.com"
            
        self.session = requests.Session()
        self.session.headers.update({
            'X-BAPI-API-KEY': self.api_key,
            'Content-Type': 'application/json'
        })
        
        logger.info(f"Bybit API initialized ({'TESTNET' if testnet else 'LIVE'})")
    
    def _generate_signature(self, params: str, timestamp: str) -> str:
        """Generate API signature"""
        try:
            # For Bybit V5 API
            param_str = f"{timestamp}{self.api_key}5000{params}"
            signature = hmac.new(
                self.secret_key.encode('utf-8'),
                param_str.encode('utf-8'),
                hashlib.sha256
            ).hexdigest()
            return signature
        except Exception as e:
            logger.error(f"Error generating signature: {str(e)}")
            return ""
    
    def _make_request(self, method: str, endpoint: str, params: Dict = None) -> Optional[Dict]:
        """Make API request with authentication"""
        try:
            url = f"{self.base_url}{endpoint}"
            timestamp = str(int(time.time() * 1000))
            
            if params is None:
                params = {}
            
            # Prepare request data
            if method.upper() == 'GET':
                query_string = '&'.join([f"{k}={v}" for k, v in sorted(params.items())])
                signature = self._generate_signature(query_string, timestamp)
                
                headers = {
                    'X-BAPI-API-KEY': self.api_key,
                    'X-BAPI-SIGN': signature,
                    'X-BAPI-TIMESTAMP': timestamp,
                    'X-BAPI-RECV-WINDOW': '5000'
                }
                
                response = self.session.get(url, params=params, headers=headers, timeout=30)
                
            else:  # POST, PUT, DELETE
                body = json.dumps(params) if params else ""
                signature = self._generate_signature(body, timestamp)
                
                headers = {
                    'X-BAPI-API-KEY': self.api_key,
                    'X-BAPI-SIGN': signature,
                    'X-BAPI-TIMESTAMP': timestamp,
                    'X-BAPI-RECV-WINDOW': '5000',
                    'Content-Type': 'application/json'
                }
                
                if method.upper() == 'POST':
                    response = self.session.post(url, data=body, headers=headers, timeout=30)
                elif method.upper() == 'DELETE':
                    response = self.session.delete(url, data=body, headers=headers, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                if result.get('retCode') == 0:
                    return result
                else:
                    logger.error(f"API Error: {result.get('retMsg', 'Unknown error')}")
                    return None
            else:
                logger.error(f"HTTP Error {response.status_code}: {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Request failed: {str(e)}")
            return None
    
    def get_account_info(self) -> Optional[Dict]:
        """Get futures account information"""
        try:
            result = self._make_request('GET', '/v5/account/wallet-balance', {'accountType': 'UNIFIED'})
            if result:
                logger.info("Account info retrieved successfully")
                return result
            return None
        except Exception as e:
            logger.error(f"Error getting account info: {str(e)}")
            return None
    
    def get_balance(self) -> Optional[float]:
        """Get USDT balance"""
        try:
            account_info = self.get_account_info()
            if account_info and account_info.get('result'):
                wallet_list = account_info['result'].get('list', [])
                for wallet in wallet_list:
                    if wallet.get('accountType') == 'UNIFIED':
                        coins = wallet.get('coin', [])
                        for coin in coins:
                            if coin.get('coin') == 'USDT':
                                balance = float(coin.get('walletBalance', 0))
                                logger.info(f"USDT Balance: ${balance:.2f}")
                                return balance
            return None
        except Exception as e:
            logger.error(f"Error getting balance: {str(e)}")
            return None
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current mark price"""
        try:
            params = {'category': 'linear', 'symbol': symbol}
            result = self._make_request('GET', '/v5/market/tickers', params)
            if result and result.get('result'):
                tickers = result['result'].get('list', [])
                if tickers:
                    price = float(tickers[0].get('markPrice', 0))
                    return price
            return None
        except Exception as e:
            logger.error(f"Error getting price for {symbol}: {str(e)}")
            return None
    
    def set_leverage(self, symbol: str, leverage: int) -> bool:
        """Set leverage for symbol"""
        try:
            params = {
                'category': 'linear',
                'symbol': symbol,
                'buyLeverage': str(leverage),
                'sellLeverage': str(leverage)
            }
            
            result = self._make_request('POST', '/v5/position/set-leverage', params)
            if result:
                logger.info(f"Leverage set to {leverage}x for {symbol}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error setting leverage for {symbol}: {str(e)}")
            return False
    
    def place_market_order(self, symbol: str, side: str, quantity: float, 
                          leverage: int = 1) -> OrderResult:
        """Place market order"""
        try:
            # Set leverage first
            if not self.set_leverage(symbol, leverage):
                return OrderResult(False, message="Failed to set leverage")
            
            params = {
                'category': 'linear',
                'symbol': symbol,
                'side': side,  # Buy or Sell
                'orderType': 'Market',
                'qty': str(quantity),
                'timeInForce': 'IOC'  # Immediate or Cancel
            }
            
            result = self._make_request('POST', '/v5/order/create', params)
            
            if result and result.get('result'):
                order_data = result['result']
                logger.info(f"Market order placed: {symbol} {side} {quantity}")
                return OrderResult(
                    success=True,
                    order_id=order_data.get('orderId'),
                    message="Order placed successfully",
                    quantity=float(quantity)
                )
            else:
                error_msg = result.get('retMsg', 'Unknown error') if result else 'API request failed'
                return OrderResult(False, message=error_msg)
                
        except Exception as e:
            logger.error(f"Error placing market order: {str(e)}")
            return OrderResult(False, message=str(e))
    
    def place_limit_order(self, symbol: str, side: str, quantity: float, 
                         price: float, leverage: int = 1) -> OrderResult:
        """Place limit order"""
        try:
            # Set leverage first
            if not self.set_leverage(symbol, leverage):
                return OrderResult(False, message="Failed to set leverage")
            
            params = {
                'category': 'linear',
                'symbol': symbol,
                'side': side,
                'orderType': 'Limit',
                'qty': str(quantity),
                'price': str(price),
                'timeInForce': 'GTC'  # Good Till Cancel
            }
            
            result = self._make_request('POST', '/v5/order/create', params)
            
            if result and result.get('result'):
                order_data = result['result']
                logger.info(f"Limit order placed: {symbol} {side} {quantity} @ {price}")
                return OrderResult(
                    success=True,
                    order_id=order_data.get('orderId'),
                    message="Order placed successfully",
                    price=price,
                    quantity=float(quantity)
                )
            else:
                error_msg = result.get('retMsg', 'Unknown error') if result else 'API request failed'
                return OrderResult(False, message=error_msg)
                
        except Exception as e:
            logger.error(f"Error placing limit order: {str(e)}")
            return OrderResult(False, message=str(e))
    
    def place_stop_order(self, symbol: str, side: str, quantity: float, 
                        stop_price: float, leverage: int = 1) -> OrderResult:
        """Place stop market order"""
        try:
            params = {
                'category': 'linear',
                'symbol': symbol,
                'side': side,
                'orderType': 'Market',
                'qty': str(quantity),
                'triggerPrice': str(stop_price),
                'timeInForce': 'IOC'
            }
            
            result = self._make_request('POST', '/v5/order/create', params)
            
            if result and result.get('result'):
                order_data = result['result']
                logger.info(f"Stop order placed: {symbol} {side} {quantity} @ {stop_price}")
                return OrderResult(
                    success=True,
                    order_id=order_data.get('orderId'),
                    message="Stop order placed successfully",
                    price=stop_price,
                    quantity=float(quantity)
                )
            else:
                error_msg = result.get('retMsg', 'Unknown error') if result else 'API request failed'
                return OrderResult(False, message=error_msg)
                
        except Exception as e:
            logger.error(f"Error placing stop order: {str(e)}")
            return OrderResult(False, message=str(e))
    
    def cancel_order(self, symbol: str, order_id: str) -> bool:
        """Cancel open order"""
        try:
            params = {
                'category': 'linear',
                'symbol': symbol,
                'orderId': order_id
            }
            
            result = self._make_request('POST', '/v5/order/cancel', params)
            if result:
                logger.info(f"Order cancelled: {order_id}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error cancelling order {order_id}: {str(e)}")
            return False
    
    def get_open_positions(self) -> List[Position]:
        """Get all open positions"""
        try:
            params = {'category': 'linear'}
            result = self._make_request('GET', '/v5/position/list', params)
            positions = []
            
            if result and result.get('result'):
                position_list = result['result'].get('list', [])
                for pos_data in position_list:
                    if abs(float(pos_data.get('size', 0))) > 0:  # Only active positions
                        position = Position(
                            symbol=pos_data.get('symbol'),
                            side=pos_data.get('side'),
                            size=abs(float(pos_data.get('size', 0))),
                            entry_price=float(pos_data.get('avgPrice', 0)),
                            mark_price=float(pos_data.get('markPrice', 0)),
                            pnl=float(pos_data.get('unrealisedPnl', 0)),
                            margin=float(pos_data.get('positionIM', 0)),
                            leverage=int(float(pos_data.get('leverage', 1))),
                            position_id=pos_data.get('symbol'),  # Use symbol as ID
                            created_at=datetime.now()  # API doesn't provide this
                        )
                        positions.append(position)
                        
            logger.info(f"Found {len(positions)} open positions")
            return positions
            
        except Exception as e:
            logger.error(f"Error getting open positions: {str(e)}")
            return []
    
    def close_position(self, symbol: str) -> OrderResult:
        """Close position by market order"""
        try:
            # Get current position
            positions = self.get_open_positions()
            target_position = None
            
            for pos in positions:
                if pos.symbol == symbol:
                    target_position = pos
                    break
            
            if not target_position:
                return OrderResult(False, message="No open position found")
            
            # Determine opposite side
            close_side = 'Sell' if target_position.side == 'Buy' else 'Buy'
            
            # Place closing order
            return self.place_market_order(
                symbol=symbol,
                side=close_side,
                quantity=target_position.size,
                leverage=target_position.leverage
            )
            
        except Exception as e:
            logger.error(f"Error closing position for {symbol}: {str(e)}")
            return OrderResult(False, message=str(e))
    
    def get_open_orders(self, symbol: str = None) -> List[Dict]:
        """Get open orders"""
        try:
            params = {'category': 'linear'}
            if symbol:
                params['symbol'] = symbol
                
            result = self._make_request('GET', '/v5/order/realtime', params)
            if result and result.get('result'):
                return result['result'].get('list', [])
            return []
        except Exception as e:
            logger.error(f"Error getting open orders: {str(e)}")
            return []
    
    def calculate_position_size(self, balance: float, entry_price: float, 
                              leverage: int, risk_percent: float = 0.1) -> float:
        """Calculate position size based on risk management"""
        try:
            # Calculate position size for risk percentage of balance
            position_value = balance * risk_percent * leverage
            quantity = position_value / entry_price
            
            # Round to appropriate decimal places (symbol-specific)
            quantity = round(quantity, 4)
            
            logger.info(f"Calculated position size: {quantity} (${position_value:.2f} notional)")
            return quantity
            
        except Exception as e:
            logger.error(f"Error calculating position size: {str(e)}")
            return 0.0
    
    def health_check(self) -> Dict:
        """Check API connectivity and account status"""
        try:
            # Test API connectivity
            server_time = self._make_request('GET', '/v5/market/time')
            if not server_time:
                return {'status': 'ERROR', 'message': 'Cannot connect to API'}
            
            # Test account access
            account_info = self.get_account_info()
            if not account_info:
                return {'status': 'ERROR', 'message': 'Cannot access account'}
            
            # Get balance
            balance = self.get_balance()
            
            return {
                'status': 'OK',
                'server_time': server_time.get('result', {}).get('timeSecond'),
                'balance': balance,
                'testnet': self.testnet
            }
            
        except Exception as e:
            return {'status': 'ERROR', 'message': str(e)}
    
    def get_trading_fees(self, symbol: str) -> Optional[Dict]:
        """Get trading fees for symbol"""
        try:
            result = self._make_request('GET', '/v5/account/fee-rate', {'category': 'linear', 'symbol': symbol})
            if result and result.get('result'):
                fee_data = result['result'].get('list', [])
                if fee_data:
                    return {
                        'maker_fee': float(fee_data[0].get('makerFeeRate', 0)),
                        'taker_fee': float(fee_data[0].get('takerFeeRate', 0))
                    }
            return None
        except Exception as e:
            logger.error(f"Error getting trading fees: {str(e)}")
            return None
    
    def get_klines(self, symbol: str, interval: str, limit: int = 200) -> List[Dict]:
        """Get candlestick data"""
        try:
            params = {
                'category': 'linear',
                'symbol': symbol,
                'interval': interval,
                'limit': limit
            }
            
            result = self._make_request('GET', '/v5/market/kline', params)
            if result and result.get('result'):
                return result['result'].get('list', [])
            return []
        except Exception as e:
            logger.error(f"Error getting klines for {symbol}: {str(e)}")
            return []
    
    def get_symbol_info(self, symbol: str) -> Optional[Dict]:
        """Get symbol trading information"""
        try:
            params = {'category': 'linear', 'symbol': symbol}
            result = self._make_request('GET', '/v5/market/instruments-info', params)
            if result and result.get('result'):
                instruments = result['result'].get('list', [])
                if instruments:
                    return instruments[0]
            return None
        except Exception as e:
            logger.error(f"Error getting symbol info for {symbol}: {str(e)}")
            return None