# backend/database.py
from sqlalchemy import create_engine, MetaData
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from config import config

# Database connection
engine = create_engine(
    config.DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in config.DATABASE_URL else {}
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Dependency to get database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# backend/models/signal.py
from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, Text
from sqlalchemy.sql import func
from database import Base

class Signal(Base):
    __tablename__ = "signals"
    
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(20), index=True)
    signal_type = Column(String(10))  # BUY or SELL
    entry_price = Column(Float)
    target_price = Column(Float)
    stop_loss = Column(Float)
    confidence = Column(Float)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    expires_at = Column(DateTime(timezone=True))
    
    # Status tracking
    status = Column(String(20), default="ACTIVE")  # ACTIVE, EXPIRED, HIT_TARGET, HIT_STOP
    actual_result = Column(Float, nullable=True)
    
    # Technical analysis data
    rsi = Column(Float)
    macd = Column(Float)
    macd_signal = Column(Float)
    bb_upper = Column(Float)
    bb_lower = Column(Float)
    volume_ratio = Column(Float)
    
    # Additional metadata
    timeframe = Column(String(10))
    notes = Column(Text)
    model_version = Column(String(50))
    
    def to_dict(self):
        return {
            "id": self.id,
            "symbol": self.symbol,
            "signal_type": self.signal_type,
            "entry_price": self.entry_price,
            "target_price": self.target_price,
            "stop_loss": self.stop_loss,
            "confidence": self.confidence,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "status": self.status,
            "actual_result": self.actual_result,
            "timeframe": self.timeframe,
            "model_version": self.model_version
        }

# backend/models/trading_pair.py
from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, Text
from sqlalchemy.sql import func
from database import Base

class TradingPair(Base):
    __tablename__ = "trading_pairs"
    
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(20), unique=True, index=True)
    base_asset = Column(String(10))
    quote_asset = Column(String(10))
    
    # Configuration
    is_active = Column(Boolean, default=True)
    min_profit_target = Column(Float, default=0.02)
    max_stop_loss = Column(Float, default=0.01)
    
    # Statistics
    total_signals = Column(Integer, default=0)
    successful_signals = Column(Integer, default=0)
    success_rate = Column(Float, default=0.0)
    avg_profit = Column(Float, default=0.0)
    
    # Last update info
    last_price = Column(Float)
    last_update = Column(DateTime(timezone=True), server_default=func.now())
    
    # Metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    notes = Column(Text)
    
    def calculate_success_rate(self):
        if self.total_signals > 0:
            self.success_rate = self.successful_signals / self.total_signals
        else:
            self.success_rate = 0.0
    
    def to_dict(self):
        return {
            "id": self.id,
            "symbol": self.symbol,
            "base_asset": self.base_asset,
            "quote_asset": self.quote_asset,
            "is_active": self.is_active,
            "min_profit_target": self.min_profit_target,
            "max_stop_loss": self.max_stop_loss,
            "total_signals": self.total_signals,
            "successful_signals": self.successful_signals,
            "success_rate": self.success_rate,
            "avg_profit": self.avg_profit,
            "last_price": self.last_price,
            "last_update": self.last_update.isoformat() if self.last_update else None
        }

# backend/models/model_performance.py
from sqlalchemy import Column, Integer, String, Float, DateTime, Text, JSON
from sqlalchemy.sql import func
from database import Base

class ModelPerformance(Base):
    __tablename__ = "model_performance"
    
    id = Column(Integer, primary_key=True, index=True)
    model_version = Column(String(50))
    
    # Training metrics
    accuracy = Column(Float)
    precision = Column(Float)
    recall = Column(Float)
    f1_score = Column(Float)
    
    # Backtesting results
    backtest_returns = Column(Float)
    max_drawdown = Column(Float)
    sharpe_ratio = Column(Float)
    
    # Training data info
    training_samples = Column(Integer)
    training_start = Column(DateTime(timezone=True))
    training_end = Column(DateTime(timezone=True))
    
    # Feature importance (stored as JSON)
    feature_importance = Column(JSON)
    
    # Model metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    notes = Column(Text)
    
    def to_dict(self):
        return {
            "id": self.id,
            "model_version": self.model_version,
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "backtest_returns": self.backtest_returns,
            "max_drawdown": self.max_drawdown,
            "sharpe_ratio": self.sharpe_ratio,
            "training_samples": self.training_samples,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "feature_importance": self.feature_importance
        }

# backend/models/__init__.py
from .signal import Signal
from .trading_pair import TradingPair
from .model_performance import ModelPerformance

__all__ = ["Signal", "TradingPair", "ModelPerformance"]