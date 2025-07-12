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
    
    # 🔥 FIXED: Back to 55% threshold
    CONFIDENCE_THRESHOLD = 0.55  # برگشت به 55%
    MODEL_RETRAIN_HOURS = 24 * 7  # Weekly retrain
    LOOKBACK_PERIODS = 100
    
    # 🔥 FIXED: Better risk/reward ratios
    MIN_PROFIT_TARGET = 0.025    # 2.5% minimum profit (بالاتر از 1.5%)
    MAX_STOP_LOSS = 0.015        # 1.5% maximum loss
    SIGNAL_EXPIRY_HOURS = 24     # 24 hours expiry
    
    # System Configuration
    DATA_COLLECTION_INTERVAL = 900  # 15 minutes (back to normal)
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
    
    # 🔥 FIXED: Live Learning Settings
    LIVE_LEARNING = {
        "initial_threshold": 0.55,  # شروع از 55%
        "min_threshold": 0.45,      # حداقل 45%
        "max_threshold": 0.75,      # حداکثر 75%
        "adaptation_sensitivity": 0.02,
        "min_signals_for_adaptation": 8,
        "learning_window": 15
    }
    
    # 🔥 FIXED: Conservative Risk Management
    RISK_MANAGEMENT = {
        "position_size_pct": 0.25,  # 25% (کمتر از 30%)
        "max_concurrent_trades": 3,  # کمتر trades همزمان
        "max_daily_trades": 8,       # کمتر trades روزانه
        "max_drawdown_limit": 0.20,
        "emergency_stop_loss": 0.15
    }
    
    # 🔥 FIXED: Stricter Signal Validation
    SIGNAL_VALIDATION = {
        "min_risk_reward": 1.8,     # بالاتر از 1.2
        "max_rsi_overbought": 80,   # کمتر از 90
        "min_volume_ratio": 0.8,    # بالاتر از 0.6
        "require_multiple_confirmations": True  # تایید چندگانه
    }
    
    # 🔥 NEW: Price Display Settings
    PRICE_DISPLAY = {
        "decimal_places": {
            # تعداد اعشار برای نمایش قیمت بر اساس قیمت
            "high_price": 2,      # > $100: 2 decimal
            "medium_price": 3,    # $1-100: 3 decimal  
            "low_price": 4,       # $0.1-1: 4 decimal
            "micro_price": 5      # < $0.1: 5 decimal
        },
        "minimum_difference_pct": 0.8  # حداقل 0.8% فاصله بین entry/stop
    }
    
    @classmethod
    def get_price_decimals(cls, price: float) -> int:
        """تعداد اعشار مناسب برای نمایش قیمت"""
        if price >= 100:
            return cls.PRICE_DISPLAY["decimal_places"]["high_price"]
        elif price >= 1:
            return cls.PRICE_DISPLAY["decimal_places"]["medium_price"]
        elif price >= 0.1:
            return cls.PRICE_DISPLAY["decimal_places"]["low_price"]
        else:
            return cls.PRICE_DISPLAY["decimal_places"]["micro_price"]
    
    @classmethod
    def format_price(cls, price: float) -> str:
        """فرمت کردن قیمت با تعداد اعشار مناسب"""
        decimals = cls.get_price_decimals(price)
        return f"${price:.{decimals}f}"
    
    @classmethod
    def validate_price_differences(cls, entry: float, target: float, stop: float) -> bool:
        """بررسی اینکه فاصله قیمت‌ها کافی باشه"""
        entry_stop_diff = abs(entry - stop) / entry
        entry_target_diff = abs(target - entry) / entry
        
        min_diff = cls.PRICE_DISPLAY["minimum_difference_pct"] / 100
        
        return (entry_stop_diff >= min_diff and entry_target_diff >= min_diff)
    
    # Leverage Settings (محافظه‌کارانه‌تر)
    LEVERAGE_SETTINGS = {
        "BTCUSDT": {"max": 8.0, "confidence_multiplier": 1.0},
        "ETHUSDT": {"max": 6.0, "confidence_multiplier": 0.9},
        "BNBUSDT": {"max": 5.0, "confidence_multiplier": 0.8},
        "ADAUSDT": {"max": 4.0, "confidence_multiplier": 0.7},
        "SOLUSDT": {"max": 4.0, "confidence_multiplier": 0.7},
        "DOTUSDT": {"max": 4.0, "confidence_multiplier": 0.7},
        "LINKUSDT": {"max": 4.0, "confidence_multiplier": 0.7},
        "MATICUSDT": {"max": 3.0, "confidence_multiplier": 0.6},
        "LTCUSDT": {"max": 5.0, "confidence_multiplier": 0.8},
        "TRXUSDT": {"max": 3.0, "confidence_multiplier": 0.6},
        "XRPUSDT": {"max": 4.0, "confidence_multiplier": 0.7},
        "default": {"max": 3.0, "confidence_multiplier": 0.6}
    }
    
    @classmethod
    def get_conservative_leverage(cls, symbol: str, confidence: float) -> float:
        """محاسبه leverage محافظه‌کارانه"""
        settings = cls.LEVERAGE_SETTINGS.get(symbol, cls.LEVERAGE_SETTINGS["default"])
        
        # محاسبه leverage بر اساس confidence (محافظه‌کارانه‌تر)
        if confidence >= 0.85:
            base_leverage = 6.0
        elif confidence >= 0.75:
            base_leverage = 4.0
        elif confidence >= 0.65:
            base_leverage = 3.0
        elif confidence >= 0.55:
            base_leverage = 2.0
        else:
            base_leverage = 1.0
        
        # اعمال ضریب symbol
        adjusted_leverage = base_leverage * settings["confidence_multiplier"]
        
        # محدود کردن به حداکثر symbol
        final_leverage = min(adjusted_leverage, settings["max"])
        
        return final_leverage
    
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