#!/usr/bin/env python3
"""
Test script for Backtester
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backend.services.backtester import Backtester
from config import config
import logging
import json

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

#!/usr/bin/env python3
"""
Test script for Backtester
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backend.services.backtester import Backtester
from config import config
import logging
import json

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_backtester():
    """Test the backtester functionality"""
    
    print("📊 Testing Backtester...")
    print("="*50)
    
    try:
        # Step 1: Initialize backtester
        print("\n🔧 Step 1: Initializing Backtester...")
        backtester = Backtester()
        print("✅ Backtester initialized")
        
        # Step 2: Load ML model
        print("\n🧠 Step 2: Loading ML model...")
        model_loaded = backtester.load_model()
        
        if model_loaded:
            print("✅ ML model loaded successfully")
            print(f"   Model version: {backtester.ml_model.model_info.get('version', 'unknown')}")
            print(f"   Features: {backtester.ml_model.model_info.get('features', 'unknown')}")
        else:
            print("❌ Failed to load ML model")
            print("   Please run test_ml_model.py first to train a model")
            return False
        
        # Step 3: Test historical data collection
        print("\n📊 Step 3: Testing historical data collection...")
        test_symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT", "SOLUSDT" , "XRPUSDT"]
        test_days = 7  # Start with 7 days for testing
        
        print(f"   Collecting {test_days} days of data for {len(test_symbols)} symbols...")
        historical_data = backtester.collect_historical_data(test_symbols, test_days)
        
        if historical_data:
            print("✅ Historical data collected successfully")
            for symbol, df in historical_data.items():
                print(f"   {symbol}: {len(df)} candles")
        else:
            print("❌ Failed to collect historical data")
            return False
        
        # Step 4: Test signal generation
        print("\n📈 Step 4: Testing signal generation...")
        all_signals = []
        
        for symbol, df in historical_data.items():
            print(f"   Generating signals for {symbol}...")
            signals = backtester.generate_signals(df, symbol)
            all_signals.extend(signals)
            print(f"   Generated {len(signals)} signals for {symbol}")
        
        if all_signals:
            print(f"✅ Total signals generated: {len(all_signals)}")
            
            # Show sample signals
            print("\n📋 Sample signals:")
            for i, signal in enumerate(all_signals[:3]):  # Show first 3
                print(f"   {i+1}. {signal.symbol} @ {signal.timestamp.strftime('%Y-%m-%d %H:%M')}")
                print(f"      Entry: ${signal.entry_price:.2f}, Target: ${signal.target_price:.2f}, Stop: ${signal.stop_loss:.2f}")
                print(f"      Confidence: {signal.confidence:.3f} | Leverage: {signal.leverage:.1f}x")
        else:
            print("⚪ No signals generated (this can happen in calm markets)")
        
        # Step 5: Test signal execution
        print("\n🎯 Step 5: Testing signal execution...")
        executed_signals = []
        
        for symbol, df in historical_data.items():
            symbol_signals = [s for s in all_signals if s.symbol == symbol]
            if symbol_signals:
                print(f"   Executing {len(symbol_signals)} signals for {symbol}...")
                executed = backtester.execute_signals(symbol_signals, df)
                executed_signals.extend(executed)
        
        if executed_signals:
            print(f"✅ Executed {len(executed_signals)} signals")
            
            # Show execution results
            successful = len([s for s in executed_signals if s.exit_reason == 'TARGET_HIT'])
            failed = len([s for s in executed_signals if s.exit_reason == 'STOP_HIT'])
            expired = len([s for s in executed_signals if s.exit_reason == 'EXPIRED'])
            
            print(f"   Results: {successful} successful, {failed} failed, {expired} expired")
            
            # Show sample results
            print("\n📋 Sample execution results:")
            for i, signal in enumerate(executed_signals[:3]):  # Show first 3
                profit_pct = signal.profit_loss_pct * 100 if signal.profit_loss_pct else 0
                leveraged_profit_pct = signal.leveraged_profit_loss_pct * 100 if signal.leveraged_profit_loss_pct else 0
                print(f"   {i+1}. {signal.symbol}: {signal.exit_reason}")
                print(f"      Entry: ${signal.entry_price:.2f} → Exit: ${signal.exit_price:.2f}")
                print(f"      P/L: {profit_pct:.2f}% | Leveraged P/L: {leveraged_profit_pct:.2f}% ({signal.leverage:.1f}x)")
                print(f"      Duration: {signal.duration_hours:.1f}h")
        else:
            print("⚪ No signals executed")
        
        # Step 6: Test backtest results calculation
        print("\n📊 Step 6: Testing backtest results calculation...")
        
        if executed_signals:
            initial_capital = 20.0
            test_days = 7
            results = backtester._calculate_backtest_results(executed_signals, initial_capital, test_days)
            
            if 'error' not in results:
                print("✅ Backtest results calculated successfully")
                print(f"   Initial capital: ${results['initial_capital']:.2f}")
                print(f"   Normal final capital: ${results['normal_trading']['final_capital']:.2f}")
                print(f"   Leveraged final capital: ${results['leveraged_trading']['final_capital']:.2f}")
                print(f"   Normal return: {results['normal_trading']['total_return_pct']:.2f}%")
                print(f"   Leveraged return: {results['leveraged_trading']['total_return_pct']:.2f}%")
                print(f"   Success rate: {results['signal_stats']['success_rate']:.1f}%")
                print(f"   Average leverage: {results['leverage_stats']['avg_leverage']:.1f}x")
            else:
                print(f"❌ Error calculating results: {results['error']}")
                return False
        else:
            print("⚪ No executed signals to analyze")
        
        # Step 7: Test report generation
        print("\n📝 Step 7: Testing report generation...")
        
        if executed_signals:
            report = backtester.generate_backtest_report(results)
            print("✅ Backtest report generated:")
            print(report)
        else:
            print("⚪ No data for report generation")
        
        print("\n" + "="*50)
        print("🎉 All backtester tests completed!")
        
        return True
        
    except Exception as e:
        logger.error(f"Backtester test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def run_full_backtest():
    """Run a full 30-day backtest"""
    print("\n🚀 Running Full Backtest (30 days)...")
    print("="*50)
    
    try:
        backtester = Backtester()
        
        # Configuration
        symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT"]  # 4 symbols for faster testing
        days = 30
        initial_capital = 20.0
        
        print(f"📋 Backtest Configuration:")
        print(f"   Symbols: {symbols}")
        print(f"   Period: {days} days")
        print(f"   Initial capital: ${initial_capital}")
        print(f"   Confidence threshold: {config.CONFIDENCE_THRESHOLD}")
        
        # Run backtest
        results = backtester.run_backtest(symbols, days, initial_capital)
        
        if results and 'error' not in results:
            print("\n✅ Full backtest completed successfully!")
            
            # Generate and display report
            report = backtester.generate_backtest_report(results)
            print(report)
            
            # Save detailed results
            output_file = f"backtest_results_{days}days.json"
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            print(f"\n💾 Detailed results saved to: {output_file}")
            
            # Key insights
            print("\n🔍 Key Insights:")
            normal_return = results['normal_trading']['total_return_pct']
            leveraged_return = results['leveraged_trading']['total_return_pct']
            
            print(f"   📈 Normal trading: {normal_return:.2f}%")
            print(f"   🚀 Leveraged trading: {leveraged_return:.2f}%")
            print(f"   ⚡ Leverage amplification: {leveraged_return - normal_return:.2f}%")
            
            if leveraged_return > 0:
                print(f"   📈 Profitable leveraged strategy: +{leveraged_return:.2f}%")
            else:
                print(f"   📉 Loss-making leveraged strategy: {leveraged_return:.2f}%")
            
            if results['signal_stats']['success_rate'] > 60:
                print(f"   ✅ Good success rate: {results['signal_stats']['success_rate']:.1f}%")
            else:
                print(f"   ⚠️  Low success rate: {results['signal_stats']['success_rate']:.1f}%")
            
            avg_leverage = results['leverage_stats']['avg_leverage']
            print(f"   ⚡ Average leverage used: {avg_leverage:.1f}x")
            
            return True
            
        else:
            print(f"❌ Full backtest failed: {results.get('error', 'Unknown error')}")
            return False
        
    except Exception as e:
        logger.error(f"Full backtest failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🤖 Crypto Signal Bot - Backtester Test")
    print("="*50)
    
    print(f"📋 Configuration:")
    print(f"   Confidence threshold: {config.CONFIDENCE_THRESHOLD}")
    print(f"   Min profit target: {config.MIN_PROFIT_TARGET*100:.1f}%")
    print(f"   Max stop loss: {config.MAX_STOP_LOSS*100:.1f}%")
    print(f"   Signal expiry: {config.SIGNAL_EXPIRY_HOURS} hours")
    
    try:
        # Run basic tests
        success = test_backtester()
        
        if success:
            print("\n🎉 Backtester test completed successfully!")
            
            # Ask user if they want to run full backtest
            print("\n" + "="*50)
            user_input = input("🚀 Run full 30-day backtest with $20 starting capital? (y/N): ").lower().strip()
            
            if user_input == 'y':
                full_success = run_full_backtest()
                
                if full_success:
                    print("\n🎊 Full backtest completed successfully!")
                    print("   Check the generated JSON file for detailed results")
                else:
                    print("\n❌ Full backtest failed")
            
            print("\n🚀 Next steps:")
            print("   1. Analyze backtest results")
            print("   2. Optimize model parameters")
            print("   3. Deploy automated scanner")
            
        else:
            print("\n❌ Backtester test failed")
            
    except KeyboardInterrupt:
        print("\n⚠️  Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()