#!/usr/bin/env python3
"""
Advanced Portfolio and Risk Management System
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from scipy.optimize import minimize
import json
import os

logger = logging.getLogger(__name__)

@dataclass
class Position:
    """کلاس موقعیت معاملاتی"""
    symbol: str
    entry_price: float
    target_price: float
    stop_loss: float
    position_size: float  # مقدار دلاری
    confidence: float
    risk_amount: float    # مقدار ریسک
    created_at: datetime
    signal_id: str
    
    # نتایج (بعداً پر می‌شود)
    current_price: Optional[float] = None
    unrealized_pnl: Optional[float] = None
    unrealized_pnl_pct: Optional[float] = None
    
    def calculate_current_metrics(self, current_price: float):
        """محاسبه معیارهای فعلی"""
        self.current_price = current_price
        self.unrealized_pnl = (current_price - self.entry_price) * (self.position_size / self.entry_price)
        self.unrealized_pnl_pct = (current_price - self.entry_price) / self.entry_price
    
    def to_dict(self) -> Dict:
        return asdict(self)

class PortfolioManager:
    """
    مدیریت پیشرفته پورتفولیو و ریسک
    """
    
    def __init__(self, initial_capital: float = 10000):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.positions: List[Position] = []
        self.closed_positions: List[Position] = []
        
        # تنظیمات ریسک
        self.max_risk_per_trade = 0.02        # 2% ریسک حداکثر هر معامله
        self.max_portfolio_risk = 0.15        # 15% ریسک کل پورتفولیو
        self.max_positions = 5                # حداکثر 5 موقعیت همزمان
        self.max_correlation = 0.8            # حداکثر همبستگی بین موقعیت‌ها
        
        # ضرایب Kelly
        self.kelly_fraction = 0.25            # محافظه‌کارانه: 25% Kelly
        
        # correlation matrix برای crypto ها
        self.correlation_matrix = self._load_correlation_matrix()
        
        # آمار عملکرد
        self.performance_stats = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0,
            'max_drawdown': 0,
            'current_drawdown': 0
        }
        
        # ایجاد پوشه برای ذخیره داده‌ها
        os.makedirs('data/portfolio', exist_ok=True)
        
        # بارگذاری تاریخچه
        self._load_portfolio_history()
        
        logger.info(f"💼 Portfolio Manager initialized with ${initial_capital:,.2f}")
    
    def calculate_position_size(self, signal, historical_performance: Dict = None) -> Tuple[float, str]:
        """
        محاسبه اندازه موقعیت با Kelly Criterion
        """
        try:
            # دریافت آمار عملکرد تاریخی
            if historical_performance:
                win_rate = historical_performance.get('win_rate', 0.6)
                avg_win = historical_performance.get('avg_win', 0.025)
                avg_loss = historical_performance.get('avg_loss', 0.015)
            else:
                # مقادیر پیش‌فرض محافظه‌کارانه
                win_rate = 0.55
                avg_win = 0.025  # 2.5%
                avg_loss = 0.015  # 1.5%
            
            # محاسبه Kelly Criterion
            kelly_f = self._calculate_kelly_fraction(win_rate, avg_win, avg_loss)
            
            # محاسبه ریسک معامله
            risk_pct = abs(signal.entry_price - signal.stop_loss) / signal.entry_price
            
            # محاسبه اندازه موقعیت بر اساس ریسک
            risk_amount = self.current_capital * self.max_risk_per_trade
            position_size_risk = risk_amount / risk_pct
            
            # محاسبه اندازه موقعیت بر اساس Kelly
            kelly_position_size = self.current_capital * kelly_f * self.kelly_fraction
            
            # انتخاب کمتر (محافظه‌کارانه‌تر)
            final_position_size = min(position_size_risk, kelly_position_size)
            
            # محدودیت‌های اضافی
            max_position_size = self.current_capital * 0.3  # حداکثر 30% capital
            final_position_size = min(final_position_size, max_position_size)
            
            # حداقل اندازه موقعیت
            min_position_size = self.current_capital * 0.01  # حداقل 1%
            final_position_size = max(final_position_size, min_position_size)
            
            # توضیح محاسبه
            calculation_reason = f"Kelly: ${kelly_position_size:.0f}, Risk: ${position_size_risk:.0f}, Final: ${final_position_size:.0f}"
            
            logger.info(f"💰 Position size calculated: ${final_position_size:.2f}")
            logger.info(f"   {calculation_reason}")
            
            return final_position_size, calculation_reason
            
        except Exception as e:
            logger.error(f"❌ Position size calculation error: {str(e)}")
            # fallback محافظه‌کارانه
            fallback_size = self.current_capital * 0.05  # 5%
            return fallback_size, f"Fallback: ${fallback_size:.0f}"
    
    def _calculate_kelly_fraction(self, win_rate: float, avg_win: float, avg_loss: float) -> float:
        """
        محاسبه Kelly Criterion
        f = (bp - q) / b
        """
        try:
            if avg_loss == 0 or win_rate == 0:
                return 0.1  # 10% محافظه‌کارانه
            
            b = avg_win / avg_loss  # odds
            p = win_rate           # احتمال برد
            q = 1 - win_rate       # احتمال باخت
            
            kelly_f = (b * p - q) / b
            
            # محدود کردن Kelly به مقادیر منطقی
            kelly_f = max(0, min(kelly_f, 0.5))  # حداکثر 50%
            
            return kelly_f
            
        except Exception as e:
            logger.error(f"❌ Kelly calculation error: {str(e)}")
            return 0.1
    
    def can_open_position(self, signal) -> Tuple[bool, str]:
        """
        بررسی امکان باز کردن موقعیت جدید
        """
        try:
            # بررسی تعداد موقعیت‌ها
            if len(self.positions) >= self.max_positions:
                return False, f"Maximum positions reached ({self.max_positions})"
            
            # بررسی ریسک کل پورتفولیو
            current_risk = self._calculate_portfolio_risk()
            
            # محاسبه ریسک موقعیت جدید
            position_size, _ = self.calculate_position_size(signal)
            new_position_risk = abs(signal.entry_price - signal.stop_loss) / signal.entry_price
            additional_risk = (position_size / self.current_capital) * new_position_risk
            
            if current_risk + additional_risk > self.max_portfolio_risk:
                return False, f"Portfolio risk limit exceeded ({current_risk + additional_risk:.1%} > {self.max_portfolio_risk:.1%})"
            
            # بررسی همبستگی
            correlation_risk = self._calculate_correlation_risk(signal.symbol)
            
            if correlation_risk > self.max_correlation:
                return False, f"High correlation with existing positions ({correlation_risk:.1%})"
            
            # بررسی capital کافی
            if position_size > self.current_capital * 0.95:
                return False, "Insufficient capital"
            
            # بررسی موقعیت تکراری
            for position in self.positions:
                if position.symbol == signal.symbol:
                    return False, f"Already have position in {signal.symbol}"
            
            return True, "Position approved"
            
        except Exception as e:
            logger.error(f"❌ Position validation error: {str(e)}")
            return False, f"Validation error: {str(e)}"
    
    def open_position(self, signal) -> Optional[Position]:
        """
        باز کردن موقعیت جدید
        """
        try:
            # بررسی امکان باز کردن
            can_open, reason = self.can_open_position(signal)
            
            if not can_open:
                logger.warning(f"❌ Cannot open position for {signal.symbol}: {reason}")
                return None
            
            # محاسبه اندازه موقعیت
            position_size, calculation_reason = self.calculate_position_size(signal)
            
            # محاسبه ریسک
            risk_pct = abs(signal.entry_price - signal.stop_loss) / signal.entry_price
            risk_amount = position_size * risk_pct
            
            # ایجاد موقعیت
            position = Position(
                symbol=signal.symbol,
                entry_price=signal.entry_price,
                target_price=signal.target_price,
                stop_loss=signal.stop_loss,
                position_size=position_size,
                confidence=signal.confidence,
                risk_amount=risk_amount,
                created_at=datetime.now(),
                signal_id=f"{signal.symbol}_{int(signal.created_at.timestamp())}"
            )
            
            # اضافه کردن به لیست
            self.positions.append(position)
            
            # بروزرسانی capital
            self.current_capital -= position_size
            
            # آمار
            self.performance_stats['total_trades'] += 1
            
            # ذخیره
            self._save_portfolio_state()
            
            logger.info(f"✅ Position opened: {signal.symbol}")
            logger.info(f"   Size: ${position_size:.2f} (Risk: ${risk_amount:.2f})")
            logger.info(f"   Remaining capital: ${self.current_capital:.2f}")
            
            return position
            
        except Exception as e:
            logger.error(f"❌ Error opening position: {str(e)}")
            return None
    
    def close_position(self, position: Position, exit_price: float, exit_reason: str) -> Dict:
        """
        بستن موقعیت
        """
        try:
            # محاسبه P&L
            pnl = (exit_price - position.entry_price) * (position.position_size / position.entry_price)
            pnl_pct = (exit_price - position.entry_price) / position.entry_price
            
            # بروزرسانی capital
            final_amount = position.position_size + pnl
            self.current_capital += final_amount
            
            # آمار
            if pnl > 0:
                self.performance_stats['winning_trades'] += 1
            else:
                self.performance_stats['losing_trades'] += 1
            
            self.performance_stats['total_pnl'] += pnl
            
            # بروزرسانی drawdown
            self._update_drawdown_stats()
            
            # حذف از موقعیت‌های فعال
            self.positions.remove(position)
            
            # اضافه کردن به موقعیت‌های بسته شده
            position.current_price = exit_price
            position.unrealized_pnl = pnl
            position.unrealized_pnl_pct = pnl_pct
            self.closed_positions.append(position)
            
            # ذخیره
            self._save_portfolio_state()
            
            result = {
                'symbol': position.symbol,
                'entry_price': position.entry_price,
                'exit_price': exit_price,
                'pnl': pnl,
                'pnl_pct': pnl_pct,
                'position_size': position.position_size,
                'exit_reason': exit_reason,
                'holding_period': datetime.now() - position.created_at
            }
            
            logger.info(f"🔒 Position closed: {position.symbol}")
            logger.info(f"   P&L: ${pnl:.2f} ({pnl_pct:.1%})")
            logger.info(f"   Reason: {exit_reason}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error closing position: {str(e)}")
            return {}
    
    def update_positions(self, current_prices: Dict[str, float]):
        """
        بروزرسانی موقعیت‌های فعال
        """
        try:
            for position in self.positions:
                if position.symbol in current_prices:
                    current_price = current_prices[position.symbol]
                    position.calculate_current_metrics(current_price)
            
            # بروزرسانی آمار کلی
            self._update_portfolio_metrics()
            
        except Exception as e:
            logger.error(f"❌ Error updating positions: {str(e)}")
    
    def _calculate_portfolio_risk(self) -> float:
        """محاسبه ریسک کل پورتفولیو"""
        try:
            total_risk = 0
            
            for position in self.positions:
                position_risk_pct = position.risk_amount / self.initial_capital
                total_risk += position_risk_pct
            
            return total_risk
            
        except Exception as e:
            logger.error(f"❌ Portfolio risk calculation error: {str(e)}")
            return 0
    
    def _calculate_correlation_risk(self, new_symbol: str) -> float:
        """محاسبه ریسک همبستگی"""
        try:
            if not self.positions:
                return 0
            
            correlations = []
            
            for position in self.positions:
                correlation = self._get_asset_correlation(new_symbol, position.symbol)
                correlations.append(correlation)
            
            # میانگین همبستگی با موقعیت‌های موجود
            return np.mean(correlations) if correlations else 0
            
        except Exception as e:
            logger.error(f"❌ Correlation risk calculation error: {str(e)}")
            return 0
    
    def _get_asset_correlation(self, symbol1: str, symbol2: str) -> float:
        """دریافت همبستگی بین دو دارایی"""
        try:
            if symbol1 == symbol2:
                return 1.0
            
            pair = tuple(sorted([symbol1, symbol2]))
            return self.correlation_matrix.get(pair, 0.3)  # پیش‌فرض 30%
            
        except Exception as e:
            return 0.3
    
    def _load_correlation_matrix(self) -> Dict:
        """بارگذاری ماتریس همبستگی"""
        try:
            # ماتریس همبستگی ساده شده برای crypto های اصلی
            return {
                ('BTCUSDT', 'ETHUSDT'): 0.75,
                ('BTCUSDT', 'BNBUSDT'): 0.65,
                ('BTCUSDT', 'ADAUSDT'): 0.60,
                ('BTCUSDT', 'SOLUSDT'): 0.70,
                ('BTCUSDT', 'DOTUSDT'): 0.55,
                ('BTCUSDT', 'LINKUSDT'): 0.60,
                ('BTCUSDT', 'MATICUSDT'): 0.55,
                ('BTCUSDT', 'LTCUSDT'): 0.70,
                ('BTCUSDT', 'XRPUSDT'): 0.50,
                
                ('ETHUSDT', 'BNBUSDT'): 0.60,
                ('ETHUSDT', 'ADAUSDT'): 0.55,
                ('ETHUSDT', 'SOLUSDT'): 0.65,
                ('ETHUSDT', 'DOTUSDT'): 0.60,
                ('ETHUSDT', 'LINKUSDT'): 0.65,
                ('ETHUSDT', 'MATICUSDT'): 0.60,
                
                ('ADAUSDT', 'DOTUSDT'): 0.50,
                ('SOLUSDT', 'DOTUSDT'): 0.45,
                # ادامه ماتریس...
            }
            
        except Exception as e:
            logger.error(f"❌ Error loading correlation matrix: {str(e)}")
            return {}
    
    def _update_drawdown_stats(self):
        """بروزرسانی آمار drawdown"""
        try:
            current_value = self.get_portfolio_value()
            peak_value = max(current_value, getattr(self, 'peak_value', self.initial_capital))
            
            self.peak_value = peak_value
            
            current_drawdown = (peak_value - current_value) / peak_value
            self.performance_stats['current_drawdown'] = current_drawdown
            self.performance_stats['max_drawdown'] = max(
                self.performance_stats['max_drawdown'], 
                current_drawdown
            )
            
        except Exception as e:
            logger.error(f"❌ Drawdown calculation error: {str(e)}")
    
    def _update_portfolio_metrics(self):
        """بروزرسانی معیارهای پورتفولیو"""
        try:
            # محاسبه P&L غیرمحقق
            total_unrealized_pnl = sum(
                pos.unrealized_pnl for pos in self.positions 
                if pos.unrealized_pnl is not None
            )
            
            # بروزرسانی drawdown
            self._update_drawdown_stats()
            
        except Exception as e:
            logger.error(f"❌ Portfolio metrics update error: {str(e)}")
    
    def get_portfolio_value(self) -> float:
        """محاسبه ارزش کل پورتفولیو"""
        try:
            # capital نقدی
            total_value = self.current_capital
            
            # ارزش موقعیت‌های فعال
            for position in self.positions:
                if position.current_price:
                    position_value = (position.current_price / position.entry_price) * position.position_size
                    total_value += position_value
                else:
                    # اگر قیمت فعلی نداریم، از قیمت ورود استفاده کن
                    total_value += position.position_size
            
            return total_value
            
        except Exception as e:
            logger.error(f"❌ Portfolio value calculation error: {str(e)}")
            return self.current_capital
    
    def get_portfolio_summary(self) -> Dict:
        """خلاصه پورتفولیو"""
        try:
            total_value = self.get_portfolio_value()
            total_return = total_value - self.initial_capital
            total_return_pct = total_return / self.initial_capital
            
            # محاسبه win rate
            total_closed = len(self.closed_positions)
            win_rate = (self.performance_stats['winning_trades'] / total_closed * 100) if total_closed > 0 else 0
            
            # P&L غیرمحقق
            unrealized_pnl = sum(
                pos.unrealized_pnl for pos in self.positions 
                if pos.unrealized_pnl is not None
            )
            
            return {
                'portfolio_value': total_value,
                'initial_capital': self.initial_capital,
                'cash_available': self.current_capital,
                'total_return': total_return,
                'total_return_pct': total_return_pct,
                'unrealized_pnl': unrealized_pnl,
                'active_positions': len(self.positions),
                'closed_positions': total_closed,
                'win_rate': win_rate,
                'total_trades': self.performance_stats['total_trades'],
                'current_drawdown': self.performance_stats['current_drawdown'],
                'max_drawdown': self.performance_stats['max_drawdown'],
                'portfolio_risk': self._calculate_portfolio_risk(),
                'capital_utilization': (1 - self.current_capital / total_value) * 100
            }
            
        except Exception as e:
            logger.error(f"❌ Portfolio summary error: {str(e)}")
            return {}
    
    def _save_portfolio_state(self):
        """ذخیره وضعیت پورتفولیو"""
        try:
            state = {
                'current_capital': self.current_capital,
                'positions': [pos.to_dict() for pos in self.positions],
                'closed_positions': [pos.to_dict() for pos in self.closed_positions[-50:]],  # آخرین 50
                'performance_stats': self.performance_stats,
                'last_update': datetime.now().isoformat()
            }
            
            with open('data/portfolio/portfolio_state.json', 'w') as f:
                json.dump(state, f, default=str, indent=2)
                
        except Exception as e:
            logger.error(f"❌ Portfolio save error: {str(e)}")
    
    def _load_portfolio_history(self):
        """بارگذاری تاریخچه پورتفولیو"""
        try:
            state_file = 'data/portfolio/portfolio_state.json'
            
            if os.path.exists(state_file):
                with open(state_file, 'r') as f:
                    state = json.load(f)
                
                self.current_capital = state.get('current_capital', self.initial_capital)
                self.performance_stats = state.get('performance_stats', self.performance_stats)
                
                # بارگذاری موقعیت‌های بسته شده
                closed_data = state.get('closed_positions', [])
                for pos_data in closed_data:
                    # تبدیل string datetime به datetime object
                    pos_data['created_at'] = datetime.fromisoformat(pos_data['created_at'])
                    position = Position(**pos_data)
                    self.closed_positions.append(position)
                
                logger.info(f"📊 Portfolio history loaded: {len(self.closed_positions)} past trades")
                
        except Exception as e:
            logger.error(f"❌ Portfolio history load error: {str(e)}")

class RiskMonitor:
    """
    مانیتور ریسک realtime
    """
    
    def __init__(self, portfolio_manager: PortfolioManager):
        self.portfolio = portfolio_manager
        self.alerts = []
        
        # آستانه‌های هشدار
        self.alert_thresholds = {
            'max_drawdown': 0.10,      # 10% drawdown
            'portfolio_risk': 0.12,    # 12% portfolio risk
            'single_position_loss': 0.05,  # 5% loss در یک position
            'correlation_warning': 0.7  # 70% correlation
        }
    
    def check_risk_alerts(self) -> List[Dict]:
        """بررسی هشدارهای ریسک"""
        try:
            alerts = []
            
            # بررسی drawdown
            current_drawdown = self.portfolio.performance_stats['current_drawdown']
            if current_drawdown > self.alert_thresholds['max_drawdown']:
                alerts.append({
                    'type': 'HIGH_DRAWDOWN',
                    'severity': 'HIGH',
                    'message': f"Portfolio drawdown: {current_drawdown:.1%}",
                    'timestamp': datetime.now().isoformat()
                })
            
            # بررسی ریسک پورتفولیو
            portfolio_risk = self.portfolio._calculate_portfolio_risk()
            if portfolio_risk > self.alert_thresholds['portfolio_risk']:
                alerts.append({
                    'type': 'HIGH_PORTFOLIO_RISK',
                    'severity': 'MEDIUM',
                    'message': f"Portfolio risk: {portfolio_risk:.1%}",
                    'timestamp': datetime.now().isoformat()
                })
            
            # بررسی ضرر موقعیت‌های تکی
            for position in self.portfolio.positions:
                if position.unrealized_pnl_pct and position.unrealized_pnl_pct < -self.alert_thresholds['single_position_loss']:
                    alerts.append({
                        'type': 'POSITION_LOSS',
                        'severity': 'MEDIUM',
                        'message': f"{position.symbol}: {position.unrealized_pnl_pct:.1%} loss",
                        'timestamp': datetime.now().isoformat()
                    })
            
            # بررسی همبستگی بالا
            symbols = [pos.symbol for pos in self.portfolio.positions]
            high_correlation_pairs = self._find_high_correlation_pairs(symbols)
            
            if high_correlation_pairs:
                alerts.append({
                    'type': 'HIGH_CORRELATION',
                    'severity': 'LOW',
                    'message': f"High correlation detected: {high_correlation_pairs}",
                    'timestamp': datetime.now().isoformat()
                })
            
            return alerts
            
        except Exception as e:
            logger.error(f"❌ Risk alerts check error: {str(e)}")
            return []
    
    def _find_high_correlation_pairs(self, symbols: List[str]) -> List[str]:
        """یافتن جفت‌های با همبستگی بالا"""
        try:
            high_corr_pairs = []
            
            for i, symbol1 in enumerate(symbols):
                for symbol2 in symbols[i+1:]:
                    correlation = self.portfolio._get_asset_correlation(symbol1, symbol2)
                    
                    if correlation > self.alert_thresholds['correlation_warning']:
                        high_corr_pairs.append(f"{symbol1}-{symbol2}")
            
            return high_corr_pairs
            
        except Exception as e:
            return []
    
    def get_risk_report(self) -> Dict:
        """گزارش ریسک کامل"""
        try:
            portfolio_summary = self.portfolio.get_portfolio_summary()
            risk_alerts = self.check_risk_alerts()
            
            # محاسبه Risk Score
            risk_score = self._calculate_risk_score(portfolio_summary)
            
            return {
                'risk_score': risk_score,
                'risk_level': self._get_risk_level(risk_score),
                'portfolio_metrics': portfolio_summary,
                'active_alerts': risk_alerts,
                'recommendations': self._get_risk_recommendations(risk_score, risk_alerts),
                'generated_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Risk report error: {str(e)}")
            return {}
    
    def _calculate_risk_score(self, portfolio_summary: Dict) -> float:
        """محاسبه امتیاز ریسک (0-100)"""
        try:
            score = 0
            
            # Portfolio Risk (40% از امتیاز)
            portfolio_risk = portfolio_summary.get('portfolio_risk', 0)
            risk_score = min(portfolio_risk / 0.2 * 40, 40)  # حداکثر 40 امتیاز
            score += risk_score
            
            # Drawdown (30% از امتیاز)
            drawdown = portfolio_summary.get('current_drawdown', 0)
            drawdown_score = min(drawdown / 0.15 * 30, 30)
            score += drawdown_score
            
            # Capital Utilization (20% از امتیاز)
            capital_util = portfolio_summary.get('capital_utilization', 0) / 100
            util_score = min(capital_util / 0.8 * 20, 20)
            score += util_score
            
            # Position Count (10% از امتیاز)
            position_count = portfolio_summary.get('active_positions', 0)
            position_score = min(position_count / 5 * 10, 10)
            score += position_score
            
            return min(score, 100)
            
        except Exception as e:
            return 0
    
    def _get_risk_level(self, risk_score: float) -> str:
        """تعیین سطح ریسک"""
        if risk_score <= 25:
            return "LOW"
        elif risk_score <= 50:
            return "MODERATE"
        elif risk_score <= 75:
            return "HIGH"
        else:
            return "VERY_HIGH"
    
    def _get_risk_recommendations(self, risk_score: float, alerts: List[Dict]) -> List[str]:
        """پیشنهادات کاهش ریسک"""
        recommendations = []
        
        if risk_score > 75:
            recommendations.append("Consider closing some positions to reduce risk")
            recommendations.append("Avoid opening new positions until risk decreases")
        
        if risk_score > 50:
            recommendations.append("Monitor positions closely")
            recommendations.append("Consider tighter stop losses")
        
        # پیشنهادات بر اساس alerts
        alert_types = [alert['type'] for alert in alerts]
        
        if 'HIGH_DRAWDOWN' in alert_types:
            recommendations.append("Review trading strategy - high drawdown detected")
        
        if 'HIGH_CORRELATION' in alert_types:
            recommendations.append("Diversify portfolio - reduce correlated positions")
        
        if 'POSITION_LOSS' in alert_types:
            recommendations.append("Review stop loss levels for losing positions")
        
        if not recommendations:
            recommendations.append("Portfolio risk is within acceptable limits")
        
        return recommendations