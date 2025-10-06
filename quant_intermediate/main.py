import matplotlib.pyplot as plt
import seaborn as sns
from core.data_manager import data_manager
from core.backtest import BacktestEngine
from strategies.buy_and_hold import BuyAndHoldStrategy
from strategies.moving_average_cross import MovingAverageCrossStrategy
from strategies.multi_factor_strategy import MultiFactorStrategy, MeanVarianceStrategy
from utils.config import config

# Set up plotting style
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")


def main():
    print("=== A-Share Quantitative Backtest System - Phase 2 ===")
    print("Multi-Factor Models & Portfolio Optimization")

    try:
        # 1. Get data
        print("\n1. Fetching data...")
        print(f"Stock pool: {config.STOCK_POOL}")

        # Test single stock for comparison
        single_stock_data = data_manager.get_stock_data(config.TEST_STOCK_CODE)
        print(f"Single stock data: {len(single_stock_data)} records")

        # 2. Initialize strategies
        print("\n2. Initializing strategies...")
        strategies = [
            BuyAndHoldStrategy().set_data(single_stock_data),
            MovingAverageCrossStrategy(20, 60).set_data(single_stock_data),
            MultiFactorStrategy(config.STOCK_POOL[:8], rebalance_freq='M'),  # Use first 8 stocks
            MeanVarianceStrategy(config.STOCK_POOL[:6], lookback_period=126)  # Use first 6 stocks
        ]

        # 3. Run backtests
        print("\n3. Running backtests...")
        backtest_engine = BacktestEngine()

        for strategy in strategies:
            print(f"\n" + "=" * 50)
            print(f"Backtesting: {strategy.name}")
            print("=" * 50)
            try:
                result_data = backtest_engine.run_backtest(strategy)
                print(f"✓ {strategy.name} backtest successful")

                # Show portfolio info for multi-stock strategies
                if hasattr(strategy, 'portfolio_weights'):
                    rebalance_dates = list(strategy.portfolio_weights.keys())
                    print(f"Portfolio rebalances: {len(rebalance_dates)}")
                    if rebalance_dates:
                        last_rebalance = rebalance_dates[-1]
                        last_weights = strategy.portfolio_weights[last_rebalance]
                        print(f"Last portfolio ({last_rebalance.date()}): {len(last_weights)} stocks")

            except Exception as e:
                print(f"✗ {strategy.name} backtest failed: {e}")
                import traceback
                traceback.print_exc()

        # 4. Performance analysis and visualization
        if backtest_engine.results:
            print("\n4. Performance analysis and chart generation...")
            plot_results(backtest_engine)
            print_results(backtest_engine)
        else:
            print("No successful backtest results to display")

    except Exception as e:
        print(f"Program execution error: {e}")
        import traceback
        traceback.print_exc()


def plot_results(backtest_engine):
    """Plot results with enhanced visualization"""
    try:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))

        # Plot cumulative returns
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        color_idx = 0

        for strategy_name, result in backtest_engine.results.items():
            data = result['data']
            ax1.plot(data.index, data['cumulative_returns'],
                     label=strategy_name, linewidth=2, color=colors[color_idx % len(colors)])
            color_idx += 1

        # Plot benchmark returns (use single stock benchmark)
        benchmark_data = None
        for result in backtest_engine.results.values():
            if 'close' in result['data'].columns:
                benchmark_data = result['data']
                break

        if benchmark_data is not None:
            ax1.plot(benchmark_data.index, benchmark_data['benchmark_cumulative'],
                     label='Price Benchmark', linewidth=2, linestyle='--', alpha=0.7, color='black')

        ax1.set_title('Strategy Performance Comparison', fontsize=15, pad=20)
        ax1.set_ylabel('Cumulative Returns')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot number of stocks for portfolio strategies
        ax2.set_title('Portfolio Composition', fontsize=15, pad=20)
        for strategy_name, result in backtest_engine.results.items():
            if hasattr(result['strategy'], 'portfolio_weights'):
                data = result['data']
                if 'num_stocks' in data.columns:
                    ax2.plot(data.index, data['num_stocks'],
                             label=f'{strategy_name} - Stocks', alpha=0.7)

        ax2.set_ylabel('Number of Stocks')
        ax2.set_xlabel('Date')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        fig.savefig("result.png", dpi=300, bbox_inches="tight")
        plt.show()

    except Exception as e:
        print(f"Plotting error: {e}")


def print_results(backtest_engine):
    """Print enhanced backtest results"""
    print("\n" + "=" * 80)
    print("BACKTEST RESULTS SUMMARY - MULTI-FACTOR & PORTFOLIO STRATEGIES")
    print("=" * 80)

    for strategy_name, result in backtest_engine.results.items():
        data = result['data']
        total_return = data['cumulative_returns'].iloc[-1] - 1

        # Get appropriate benchmark
        if 'benchmark_cumulative' in data.columns:
            benchmark_return = data['benchmark_cumulative'].iloc[-1] - 1
        else:
            # For portfolio strategies, use first available benchmark
            benchmark_return = total_return  # Default fallback

        days = len(data)

        # Annualized return
        years = days / 252
        annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

        # Maximum drawdown
        cumulative_max = data['cumulative_returns'].cummax()
        drawdown = (data['cumulative_returns'] - cumulative_max) / cumulative_max
        max_drawdown = drawdown.min()

        # Volatility (annualized)
        volatility = data['strategy_returns'].std() * np.sqrt(252)

        # Sharpe ratio (assuming 0 risk-free rate)
        sharpe_ratio = annual_return / volatility if volatility > 0 else 0

        print(f"\n{strategy_name}:")
        print(f"  Total Return:     {total_return:>8.2%}")
        print(f"  Annual Return:    {annual_return:>8.2%}")
        print(f"  Max Drawdown:     {max_drawdown:>8.2%}")
        print(f"  Volatility:       {volatility:>8.2%}")
        print(f"  Sharpe Ratio:     {sharpe_ratio:>8.2f}")
        print(f"  Final NAV:        {data['cumulative_returns'].iloc[-1]:.2f}")
        print(f"  Trading Days:     {days}")

        # Additional info for portfolio strategies
        if hasattr(result['strategy'], 'portfolio_weights'):
            weights = result['strategy'].portfolio_weights
            rebalance_count = len(weights)
            if weights:
                last_date = list(weights.keys())[-1]
                stock_count = len(weights[last_date])
                print(f"  Portfolio Info:   {stock_count} stocks, {rebalance_count} rebalances")

        print("-" * 50)


if __name__ == "__main__":
    import numpy as np  # Add this import for Sharpe ratio calculation

    main()