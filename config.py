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
    
    # 🚀 OPTIMIZED Model Configuration (از بک‌تست)
    CONFIDENCE_THRESHOLD = 0.55  # کم کردیم از 0.55 به 0.45
    MODEL_RETRAIN_HOURS = 24 * 7  # Weekly retrain
    LOOKBACK_PERIODS = 100
    
    # 🚀 OPTIMIZED Signal Configuration (از بک‌تست)
    MIN_PROFIT_TARGET = 0.015    # کم کردیم از 0.02 به 0.015
    MAX_STOP_LOSS = 0.012        # کم کردیم از 0.015 به 0.012
    SIGNAL_EXPIRY_HOURS = 18     # کم کردیم از 999 به 18
    
    # System Configuration
    DATA_COLLECTION_INTERVAL = 600  # 10 دقیقه (کم کردیم از 15)
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
    
    # 🚀 NEW: Optimized Live Learning Settings
    LIVE_LEARNING = {
        "initial_threshold": 0.45,
        "min_threshold": 0.25,
        "max_threshold": 0.70,
        "adaptation_sensitivity": 0.03,  # هر بار چقدر تغییر کنه
        "min_signals_for_adaptation": 5,
        "learning_window": 20  # آخرین 20 سیگنال رو نگاه کن
    }
    
    # 🚀 NEW: Optimized Risk Management
    RISK_MANAGEMENT = {
        "position_size_pct": 0.30,  # 30% از سرمایه در هر معامله
        "max_concurrent_trades": 5,
        "max_daily_trades": 15,
        "max_drawdown_limit": 0.25,  # 25% max drawdown
        "emergency_stop_loss": 0.20  # اگر 20% ضرر کردی همه رو ببند
    }
    
    # 🚀 NEW: Optimized Signal Validation
    SIGNAL_VALIDATION = {
        "min_risk_reward": 1.2,  # کم کردیم از 1.5
        "max_rsi_overbought": 90,  # بالا بردیم از 85
        "min_volume_ratio": 0.6,   # کم کردیم از 0.8
        "require_multiple_confirmations": False
    }
    
    # 🚀 NEW: Leverage Settings (از بک‌تست)
    LEVERAGE_SETTINGS = {
        "BTCUSDT": {"max": 15.0, "confidence_multiplier": 1.2},
        "ETHUSDT": {"max": 12.0, "confidence_multiplier": 1.1},
        "BNBUSDT": {"max": 10.0, "confidence_multiplier": 1.0},
        "ADAUSDT": {"max": 8.0, "confidence_multiplier": 0.9},
        "SOLUSDT": {"max": 8.0, "confidence_multiplier": 0.9},
        "DOTUSDT": {"max": 8.0, "confidence_multiplier": 0.9},
        "LINKUSDT": {"max": 8.0, "confidence_multiplier": 0.9},
        "MATICUSDT": {"max": 6.0, "confidence_multiplier": 0.8},
        "LTCUSDT": {"max": 10.0, "confidence_multiplier": 1.0},
        "TRXUSDT": {"max": 6.0, "confidence_multiplier": 0.8},
        "XRPUSDT": {"max": 8.0, "confidence_multiplier": 0.9},
        "default": {"max": 5.0, "confidence_multiplier": 0.8}
    }
    
    @classmethod
    def get_optimized_leverage(cls, symbol: str, confidence: float) -> float:
        """محاسبه leverage بهینه بر اساس symbol و confidence"""
        settings = cls.LEVERAGE_SETTINGS.get(symbol, cls.LEVERAGE_SETTINGS["default"])
        
        # محاسبه leverage بر اساس confidence
        if confidence >= 0.9:
            base_leverage = 15.0
        elif confidence >= 0.8:
            base_leverage = 10.0
        elif confidence >= 0.7:
            base_leverage = 6.0
        elif confidence >= 0.6:
            base_leverage = 4.0
        elif confidence >= 0.5:
            base_leverage = 2.5
        else:
            base_leverage = 1.5
        
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

# 🚀 تابع کمکی برای گرفتن تنظیمات optimized
def get_optimized_settings():
    """گرفتن تمام تنظیمات بهینه‌شده"""
    return {
        "confidence_threshold": config.CONFIDENCE_THRESHOLD,
        "position_size": config.RISK_MANAGEMENT["position_size_pct"],
        "min_profit": config.MIN_PROFIT_TARGET,
        "max_loss": config.MAX_STOP_LOSS,
        "scan_interval_minutes": config.DATA_COLLECTION_INTERVAL / 60,
        "signal_expiry_hours": config.SIGNAL_EXPIRY_HOURS,
        "live_learning_enabled": True,
        "optimized_leverage": True,
        "aggressive_targets": True
    }