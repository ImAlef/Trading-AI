import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging
from config import config
import ta

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """
    Feature Engineering for Technical Analysis
    """
    
    def __init__(self):
        self.indicators_config = config.TECHNICAL_INDICATORS
        
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all technical indicators for the given DataFrame
        """
        try:
            # Make a copy to avoid modifying original data
            data = df.copy()
            
            # Ensure we have required columns
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in data.columns for col in required_cols):
                raise ValueError(f"DataFrame must contain columns: {required_cols}")
            
            # RSI
            data['rsi'] = ta.momentum.RSIIndicator(
                close=data['close'], 
                window=self.indicators_config['RSI']['period']
            ).rsi()
            
            # MACD
            macd = ta.trend.MACD(
                close=data['close'],
                window_fast=self.indicators_config['MACD']['fast'],
                window_slow=self.indicators_config['MACD']['slow'],
                window_sign=self.indicators_config['MACD']['signal']
            )
            data['macd'] = macd.macd()
            data['macd_signal'] = macd.macd_signal()
            data['macd_histogram'] = macd.macd_diff()
            
            # Bollinger Bands
            bollinger = ta.volatility.BollingerBands(
                close=data['close'],
                window=self.indicators_config['BOLLINGER']['period'],
                window_dev=self.indicators_config['BOLLINGER']['std']
            )
            data['bb_upper'] = bollinger.bollinger_hband()
            data['bb_middle'] = bollinger.bollinger_mavg()
            data['bb_lower'] = bollinger.bollinger_lband()
            data['bb_width'] = (data['bb_upper'] - data['bb_lower']) / data['bb_middle']
            data['bb_position'] = (data['close'] - data['bb_lower']) / (data['bb_upper'] - data['bb_lower'])
            
            # Moving Averages
            for period in self.indicators_config['SMA']['periods']:
                data[f'sma_{period}'] = ta.trend.SMAIndicator(
                    close=data['close'], window=period
                ).sma_indicator()
            
            for period in self.indicators_config['EMA']['periods']:
                data[f'ema_{period}'] = ta.trend.EMAIndicator(
                    close=data['close'], window=period
                ).ema_indicator()
            
            # Stochastic
            stoch = ta.momentum.StochasticOscillator(
                high=data['high'],
                low=data['low'],
                close=data['close'],
                window=self.indicators_config['STOCH']['k_period'],
                smooth_window=self.indicators_config['STOCH']['d_period']
            )
            data['stoch_k'] = stoch.stoch()
            data['stoch_d'] = stoch.stoch_signal()
            
            # ATR (Average True Range)
            data['atr'] = ta.volatility.AverageTrueRange(
                high=data['high'],
                low=data['low'],
                close=data['close'],
                window=self.indicators_config['ATR']['period']
            ).average_true_range()
            
            # OBV (On Balance Volume)
            data['obv'] = ta.volume.OnBalanceVolumeIndicator(
                close=data['close'],
                volume=data['volume']
            ).on_balance_volume()
            
            # Volume SMA
            data['volume_sma'] = ta.trend.SMAIndicator(
                close=data['volume'],
                window=self.indicators_config['VOLUME_SMA']['period']
            ).sma_indicator()
            
            # Volume ratio
            data['volume_ratio'] = data['volume'] / data['volume_sma']
            # Clean inf values
            data = data.replace([np.inf, -np.inf], np.nan)
            data = data.fillna(0)
            logger.info(f"Successfully calculated technical indicators for {len(data)} rows")
            return data
            
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {str(e)}")
            return df
    
    def calculate_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate price-based features
        """
        try:
            data = df.copy()
            
            # Price changes
            data['price_change'] = data['close'].pct_change()
            data['price_change_abs'] = data['price_change'].abs()
            
            # High-Low spread
            data['hl_spread'] = (data['high'] - data['low']) / data['close']
            
            # Open-Close spread
            data['oc_spread'] = (data['close'] - data['open']) / data['open']
            
            # Price position within the day's range
            data['price_position'] = (data['close'] - data['low']) / (data['high'] - data['low'])
            
            # Rolling statistics
            for window in [5, 10, 20]:
                data[f'returns_{window}d'] = data['close'].pct_change(window)
                data[f'volatility_{window}d'] = data['price_change'].rolling(window).std()
                data[f'high_{window}d'] = data['high'].rolling(window).max()
                data[f'low_{window}d'] = data['low'].rolling(window).min()
            
            # Price momentum
            data['momentum_1'] = data['close'] / data['close'].shift(1) - 1
            data['momentum_3'] = data['close'] / data['close'].shift(3) - 1
            data['momentum_5'] = data['close'] / data['close'].shift(5) - 1
            # Clean inf values  
            data = data.replace([np.inf, -np.inf], np.nan)
            data = data.fillna(0)
            logger.info(f"Successfully calculated price features for {len(data)} rows")
            return data
            
        except Exception as e:
            logger.error(f"Error calculating price features: {str(e)}")
            return df
    
    def calculate_trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate trend-based features
        """
        try:
            data = df.copy()
            
            # Moving average trends
            data['ma_trend_short'] = np.where(data['ema_9'] > data['ema_21'], 1, 0)
            data['ma_trend_long'] = np.where(data['sma_50'] > data['sma_200'], 1, 0)
            
            # Price vs MA position
            data['price_vs_ema9'] = (data['close'] - data['ema_9']) / data['ema_9']
            data['price_vs_ema21'] = (data['close'] - data['ema_21']) / data['ema_21']
            data['price_vs_sma50'] = (data['close'] - data['sma_50']) / data['sma_50']
            
            # MACD signals
            data['macd_signal_line'] = np.where(data['macd'] > data['macd_signal'], 1, 0)
            data['macd_histogram_positive'] = np.where(data['macd_histogram'] > 0, 1, 0)
            
            # RSI levels
            data['rsi_oversold'] = np.where(data['rsi'] < 30, 1, 0)
            data['rsi_overbought'] = np.where(data['rsi'] > 70, 1, 0)
            
            # Bollinger Bands position
            data['bb_squeeze'] = np.where(data['bb_width'] < data['bb_width'].rolling(20).mean(), 1, 0)
            data['bb_breakout_upper'] = np.where(data['close'] > data['bb_upper'], 1, 0)
            data['bb_breakout_lower'] = np.where(data['close'] < data['bb_lower'], 1, 0)
            
            # Volume trends
            data['volume_above_avg'] = np.where(data['volume_ratio'] > 1.5, 1, 0)
            # Clean inf values
            data = data.replace([np.inf, -np.inf], np.nan)
            data = data.fillna(0)
            logger.info(f"Successfully calculated trend features for {len(data)} rows")
            return data
            
        except Exception as e:
            logger.error(f"Error calculating trend features: {str(e)}")
            return df
    
    def calculate_pattern_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate candlestick pattern features
        """
        try:
            data = df.copy()
            
            # Body and shadow sizes
            data['body_size'] = abs(data['close'] - data['open']) / data['close']
            data['upper_shadow'] = (data['high'] - data[['open', 'close']].max(axis=1)) / data['close']
            data['lower_shadow'] = (data[['open', 'close']].min(axis=1) - data['low']) / data['close']
            
            # Candle types
            data['bullish_candle'] = np.where(data['close'] > data['open'], 1, 0)
            data['bearish_candle'] = np.where(data['close'] < data['open'], 1, 0)
            data['doji'] = np.where(data['body_size'] < 0.001, 1, 0)
            
            # Simple patterns
            data['hammer'] = np.where(
                (data['lower_shadow'] > 2 * data['body_size']) & 
                (data['upper_shadow'] < 0.5 * data['body_size']), 1, 0
            )
            
            data['shooting_star'] = np.where(
                (data['upper_shadow'] > 2 * data['body_size']) & 
                (data['lower_shadow'] < 0.5 * data['body_size']), 1, 0
            )
            
            data['long_body'] = np.where(data['body_size'] > data['body_size'].rolling(20).mean() * 1.5, 1, 0)
            
            # Gap detection
            data['gap_up'] = np.where(data['low'] > data['high'].shift(1), 1, 0)
            data['gap_down'] = np.where(data['high'] < data['low'].shift(1), 1, 0)
            # Clean inf values
            data = data.replace([np.inf, -np.inf], np.nan)
            data = data.fillna(0)
            logger.info(f"Successfully calculated pattern features for {len(data)} rows")
            return data
            
        except Exception as e:
            logger.error(f"Error calculating pattern features: {str(e)}")
            return df
    
    def create_target_variable(self, df: pd.DataFrame, target_hours: int = 4, target_return: float = 0.015) -> pd.DataFrame:
        """
        Create target variable for ML model
        """
        try:
            data = df.copy()
            
            # Calculate future returns
            data['future_return'] = data['close'].shift(-target_hours) / data['close'] - 1
            
            # Create binary target (1 if price increases by target_return or more)
            data['target'] = np.where(data['future_return'] >= target_return, 1, 0)
            
            # Also create regression target
            data['target_return'] = data['future_return']
            
            # Instead of dropping all NaN rows, only drop the last few rows that don't have future data
            # Keep rows where we have enough historical data for indicators
            valid_rows = data.dropna(subset=['future_return']).index
            if len(valid_rows) > 0:
                last_valid_idx = valid_rows.max()
                data = data.loc[:last_valid_idx]
            
            logger.info(f"Created target variable with {data['target'].sum()} positive samples out of {len(data)} total")
            return data
            
        except Exception as e:
            logger.error(f"Error creating target variable: {str(e)}")
            return df
    
    def prepare_features_for_prediction(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Complete feature engineering pipeline for prediction (without target)
        """
        try:
            logger.info("Starting feature engineering pipeline for prediction...")
            
            # Step 1: Calculate technical indicators
            data = self.calculate_technical_indicators(df)
            
            # Step 2: Calculate price features
            data = self.calculate_price_features(data)
            
            # Step 3: Calculate trend features
            data = self.calculate_trend_features(data)
            
            # Step 4: Calculate pattern features
            data = self.calculate_pattern_features(data)
            
            # Remove rows with NaN values (from indicators that need warmup)
            # But keep enough data for prediction
            original_len = len(data)
            
            # Only drop rows where essential features are NaN
            essential_features = ['rsi', 'macd', 'bb_upper', 'bb_lower', 'volume_ratio', 'price_change']
            data = data.dropna(subset=essential_features)
            
            final_len = len(data)
            
            logger.info(f"Feature engineering for prediction completed: {original_len} -> {final_len} rows")
            logger.info(f"Created {len(data.columns)} features")
            
            return data
            
        except Exception as e:
            logger.error(f"Error in feature engineering pipeline for prediction: {str(e)}")
            return df
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Complete feature engineering pipeline for training (with target)
        """
        try:
            logger.info("Starting feature engineering pipeline...")
            
            # Step 1: Calculate technical indicators
            data = self.calculate_technical_indicators(df)
            
            # Step 2: Calculate price features
            data = self.calculate_price_features(data)
            
            # Step 3: Calculate trend features
            data = self.calculate_trend_features(data)
            
            # Step 4: Calculate pattern features
            data = self.calculate_pattern_features(data)
            
            # Step 5: Create target variable
            data = self.create_target_variable(data)
            
            # Remove rows with NaN values (from indicators that need warmup)
            original_len = len(data)
            data = data.dropna()
            final_len = len(data)
            
            logger.info(f"Feature engineering completed: {original_len} -> {final_len} rows")
            logger.info(f"Created {len(data.columns)} features")
            
            return data
            
        except Exception as e:
            logger.error(f"Error in feature engineering pipeline: {str(e)}")
            return df
    
    def get_feature_columns(self) -> List[str]:
        """
        Get list of feature columns (excluding OHLCV and target)
        """
        # Base columns to exclude
        exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'target', 'target_return', 'future_return']
        
        # Get all possible feature columns
        feature_cols = []
        
        # Technical indicators
        feature_cols.extend(['rsi', 'macd', 'macd_signal', 'macd_histogram'])
        feature_cols.extend(['bb_upper', 'bb_middle', 'bb_lower', 'bb_width', 'bb_position'])
        feature_cols.extend(['stoch_k', 'stoch_d', 'atr', 'obv', 'volume_sma', 'volume_ratio'])
        
        # Moving averages
        for period in self.indicators_config['SMA']['periods']:
            feature_cols.append(f'sma_{period}')
        for period in self.indicators_config['EMA']['periods']:
            feature_cols.append(f'ema_{period}')
        
        # Price features
        feature_cols.extend(['price_change', 'price_change_abs', 'hl_spread', 'oc_spread', 'price_position'])
        
        # Rolling features
        for window in [5, 10, 20]:
            feature_cols.extend([f'returns_{window}d', f'volatility_{window}d', f'high_{window}d', f'low_{window}d'])
        
        # Momentum features
        feature_cols.extend(['momentum_1', 'momentum_3', 'momentum_5'])
        
        # Trend features
        feature_cols.extend(['ma_trend_short', 'ma_trend_long', 'price_vs_ema9', 'price_vs_ema21', 'price_vs_sma50'])
        feature_cols.extend(['macd_signal_line', 'macd_histogram_positive', 'rsi_oversold', 'rsi_overbought'])
        feature_cols.extend(['bb_squeeze', 'bb_breakout_upper', 'bb_breakout_lower', 'volume_above_avg'])
        
        # Pattern features
        feature_cols.extend(['body_size', 'upper_shadow', 'lower_shadow', 'bullish_candle', 'bearish_candle', 'doji'])
        feature_cols.extend(['hammer', 'shooting_star', 'long_body', 'gap_up', 'gap_down'])
        
        return feature_cols
    
    def get_feature_importance_names(self) -> Dict[str, str]:
        """
        Get human-readable names for features
        """
        return {
            'rsi': 'RSI (14)',
            'macd': 'MACD Line',
            'macd_signal': 'MACD Signal',
            'macd_histogram': 'MACD Histogram',
            'bb_width': 'Bollinger Band Width',
            'bb_position': 'BB Position',
            'volume_ratio': 'Volume Ratio',
            'price_change': 'Price Change',
            'hl_spread': 'High-Low Spread',
            'volatility_20d': '20-Day Volatility',
            'momentum_5': '5-Period Momentum',
            'ma_trend_short': 'Short MA Trend',
            'price_vs_ema9': 'Price vs EMA9',
            'rsi_oversold': 'RSI Oversold',
            'bb_squeeze': 'BB Squeeze',
            'volume_above_avg': 'High Volume'
        }