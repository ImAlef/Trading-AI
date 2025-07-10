import os
from typing import List
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Database
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./crypto_bot.db")
    
    # API Keys
    BINANCE_API_KEY = os.getenv("BINANCE_API_KEY", "")
    BINANCE_SECRET_KEY = os.getenv("BINANCE_SECRET_KEY", "")
    
    # Email Configuration
    SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
    SMTP_PORT = int(os.getenv("SMTP_PORT", 587))
    SMTP_USERNAME = os.getenv("SMTP_USERNAME", "")
    SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
    FROM_EMAIL = os.getenv("FROM_EMAIL", "")
    TO_EMAIL = os.getenv("TO_EMAIL", "")
    
    # Trading Configuration
    TRADING_PAIRS: List[str] = [
        "BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT", 
        "SOLUSDT", "DOTUSDT", "LINKUSDT", "MATICUSDT",
        "LTCUSDT", "TRXUSDT","XRPUSDT"
    ]
    
    TIMEFRAMES = ["15m", "30m", "1h", "4h"]
    PRIMARY_TIMEFRAME = "1h"
    
    # Model Configuration
    CONFIDENCE_THRESHOLD = 0.55  # کم کردیم از 0.6 به 0.55
    MODEL_RETRAIN_HOURS = 24 * 7  # Weekly retrain
    LOOKBACK_PERIODS = 100
    
    # Signal Configuration
    MIN_PROFIT_TARGET = 0.02  # 2% minimum profit
    MAX_STOP_LOSS = 0.01      # 1% maximum loss
    SIGNAL_EXPIRY_HOURS = 999
    
    # System Configuration
    DATA_COLLECTION_INTERVAL = 900  # 15 minutes in seconds
    MAX_CONCURRENT_PAIRS = 15
    LOG_LEVEL = "INFO"
    
    # Security
    SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-here")
    ALGORITHM = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES = 30
    
    # WebSocket
    WS_HOST = "0.0.0.0"
    WS_PORT = 8001
    
    # API Rate Limits
    BINANCE_RATE_LIMIT = 1200  # requests per minute
    COINGECKO_RATE_LIMIT = 50  # requests per minute
    
    # Feature Engineering
    TECHNICAL_INDICATORS = {
        "RSI": {"period": 14},
        "MACD": {"fast": 12, "slow": 26, "signal": 9},
        "BOLLINGER": {"period": 20, "std": 2},
        "SMA": {"periods": [20, 50, 100, 200]},
        "EMA": {"periods": [9, 21, 50]},
        "STOCH": {"k_period": 14, "d_period": 3},
        "ATR": {"period": 14},
        "OBV": {},
        "VOLUME_SMA": {"period": 20}
    }
    
    # Backtesting
    BACKTEST_DAYS = 180
    TRAIN_TEST_SPLIT = 0.8
    
    @classmethod
    def get_trading_pairs_with_suffix(cls) -> List[str]:
        """Get trading pairs with proper suffix for different exchanges"""
        return [f"{pair}" for pair in cls.TRADING_PAIRS]
    
    @classmethod
    def validate_config(cls) -> bool:
        """Validate critical configuration"""
        required_vars = ["SMTP_USERNAME", "SMTP_PASSWORD", "FROM_EMAIL", "TO_EMAIL"]
        missing_vars = [var for var in required_vars if not getattr(cls, var)]
        
        if missing_vars:
            print(f"Missing required environment variables: {missing_vars}")
            return False
        
        return True

# Development/Production specific configs
class DevelopmentConfig(Config):
    DEBUG = True
    LOG_LEVEL = "DEBUG"

class ProductionConfig(Config):
    DEBUG = False
    LOG_LEVEL = "INFO"

# Choose config based on environment
config = DevelopmentConfig() if os.getenv("ENVIRONMENT") == "development" else ProductionConfig()