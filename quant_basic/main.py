import matplotlib.pyplot as plt
import seaborn as sns
from core.data_manager import data_manager
from core.backtest import BacktestEngine
from strategies.buy_and_hold import BuyAndHoldStrategy
from strategies.moving_average_cross import MovingAverageCrossStrategy
from utils.config import config

# Set up plotting style
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")


def main():
    print("=== A-Share Quantitative Backtest System Started ===")

    try:
        # 1. Get data
        print("\n1. Fetching data...")
        stock_data = data_manager.get_stock_data(config.TEST_STOCK_CODE)

        if stock_data is None or len(stock_data) == 0:
            print("Error: No data retrieved")
            return

        print(f"Data sample:")
        print(stock_data.head())
        print(f"\nData basic info:")
        print(f"Time range: {stock_data.index[0]} to {stock_data.index[-1]}")
        print(f"Data length: {len(stock_data)}")
        print(f"Close price range: {stock_data['close'].min():.2f} - {stock_data['close'].max():.2f}")

        # 2. Initialize strategies
        print("\n2. Initializing strategies...")
        strategies = [
            BuyAndHoldStrategy().set_data(stock_data),
            MovingAverageCrossStrategy(20, 60).set_data(stock_data)
        ]

        # 3. Run backtests
        print("\n3. Running backtests...")
        backtest_engine = BacktestEngine()

        for strategy in strategies:
            print(f"\nBacktesting: {strategy.name}")
            try:
                backtest_engine.run_backtest(strategy)
                print(f"✓ {strategy.name} backtest successful")
            except Exception as e:
                print(f"✗ {strategy.name} backtest failed: {e}")

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
    """Plot results"""
    try:
        fig = plt.figure(figsize=(14, 10))

        # Plot cumulative returns
        plt.subplot(2, 1, 1)
        colors = ['blue', 'red', 'green']
        color_idx = 0

        for strategy_name, result in backtest_engine.results.items():
            data = result['data']
            plt.plot(data.index, data['cumulative_returns'],
                     label=strategy_name, linewidth=2, color=colors[color_idx % len(colors)])
            color_idx += 1

        # Plot benchmark returns
        benchmark_data = list(backtest_engine.results.values())[0]['data']
        plt.plot(benchmark_data.index, benchmark_data['benchmark_cumulative'],
                 label='Price Benchmark', linewidth=2, linestyle='--', alpha=0.7, color='black')

        plt.title('Strategy Performance Comparison', fontsize=15, pad=20)
        plt.ylabel('Cumulative Returns')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Plot price and moving averages
        plt.subplot(2, 1, 2)
        ma_strategy_name = None
        for name in backtest_engine.results.keys():
            if 'MA Crossover' in name:
                ma_strategy_name = name
                break

        if ma_strategy_name:
            ma_data = backtest_engine.results[ma_strategy_name]['data']
            plt.plot(ma_data.index, ma_data['close'], label='Price', alpha=0.7, linewidth=1)
            if 'short_ma' in ma_data.columns:
                plt.plot(ma_data.index, ma_data['short_ma'], label='20-day MA', alpha=0.8, linewidth=1.5)
            if 'long_ma' in ma_data.columns:
                plt.plot(ma_data.index, ma_data['long_ma'], label='60-day MA', alpha=0.8, linewidth=1.5)

            plt.title('Price and Moving Averages', fontsize=15, pad=20)
            plt.ylabel('Price')
            plt.legend()
            plt.grid(True, alpha=0.3)

        plt.tight_layout()
        fig.savefig("result.png", dpi=300, bbox_inches="tight")
        plt.show()

    except Exception as e:
        print(f"Plotting error: {e}")


def print_results(backtest_engine):
    """Print backtest results"""
    print("\n" + "=" * 60)
    print("BACKTEST RESULTS SUMMARY")
    print("=" * 60)

    for strategy_name, result in backtest_engine.results.items():
        data = result['data']
        total_return = data['cumulative_returns'].iloc[-1] - 1
        benchmark_return = data['benchmark_cumulative'].iloc[-1] - 1
        days = len(data)

        # Annualized return
        years = days / 252
        annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

        # Maximum drawdown
        cumulative_max = data['cumulative_returns'].cummax()
        drawdown = (data['cumulative_returns'] - cumulative_max) / cumulative_max
        max_drawdown = drawdown.min()

        print(f"\n{strategy_name}:")
        print(f"  Total Return:    {total_return:>8.2%} (Benchmark: {benchmark_return:.2%})")
        print(f"  Annual Return:   {annual_return:>8.2%}")
        print(f"  Max Drawdown:    {max_drawdown:>8.2%}")
        print(f"  Final NAV:       {data['cumulative_returns'].iloc[-1]:.2f}")
        print(f"  Trading Days:    {days}")


if __name__ == "__main__":
    main()