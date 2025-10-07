import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from core.data_manager import data_manager
from core.backtest import BacktestEngine
from strategies.buy_and_hold import BuyAndHoldStrategy
from strategies.moving_average_cross import MovingAverageCrossStrategy
from strategies.multi_factor_strategy import MultiFactorStrategy, MeanVarianceStrategy
from strategies.ml_strategy import MLStrategy, EnsembleMLStrategy
from strategies.pair_trading_strategy import PairTradingStrategy, MeanReversionStrategy
from utils.config import config

# Set up plotting style
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")


def main():
    print("=== A-Share Quantitative Backtest System - Phase 3 ===")
    print("Statistical Arbitrage & Machine Learning")

    try:
        # 1. Get data
        print("\n1. Fetching data...")
        single_stock_data = data_manager.get_stock_data(config.TEST_STOCK_CODE)
        print(f"Single stock data: {len(single_stock_data)} records")

        # 2. Initialize strategies
        print("\n2. Initializing strategies...")
        strategies = [
            # Traditional strategies for comparison
            BuyAndHoldStrategy().set_data(single_stock_data),
            MovingAverageCrossStrategy(20, 60).set_data(single_stock_data),

            # Phase 2: Multi-factor strategies
            MultiFactorStrategy(config.STOCK_POOL[:6], rebalance_freq='M'),

            # Phase 3: Statistical arbitrage
            PairTradingStrategy([('sz.000001', 'sh.600036')]),  # Ping An Bank vs China Merchants Bank
            MeanReversionStrategy(lookback_period=20, entry_z=1.5, exit_z=0.5)
            .set_data(single_stock_data),

            # Phase 3: Machine learning - 调整参数增加交易频率
            MLStrategy(
                model_type='random_forest',
                lookforward_days=5,
                retrain_freq=21,  # 从63天改为21天（月度）
                min_train_size=0.2  # 降低最小训练集要求
            ).set_data(single_stock_data),

            EnsembleMLStrategy(
                lookforward_days=5,
                retrain_freq=21,
                min_train_size=0.2,
                voting_threshold=0.55  # 降低集成投票阈值
            ).set_data(single_stock_data)
        ]

        # 3. Run backtests
        print("\n3. Running backtests...")
        backtest_engine = BacktestEngine()

        for strategy in strategies:
            print(f"\n" + "=" * 60)
            print(f"Backtesting: {strategy.name}")
            print("=" * 60)
            try:
                result_data = backtest_engine.run_backtest(strategy)
                print(f"✓ {strategy.name} backtest successful")

                # 对于ML策略，显示训练摘要
                if 'ML' in strategy.name and hasattr(strategy, 'training_history'):
                    print(f"ML训练摘要: {len(strategy.training_history)} 次训练")

            except Exception as e:
                print(f"✗ {strategy.name} backtest failed: {e}")
                import traceback
                traceback.print_exc()

        # 4. Performance analysis and visualization
        if backtest_engine.results:
            print("\n4. Performance analysis and chart generation...")
            plot_results(backtest_engine)
            print_advanced_results(backtest_engine)

            # 生成ML训练报告
            print("\n5. ML策略详细分析...")
            for strategy_name in backtest_engine.results.keys():
                if 'ML' in strategy_name:
                    backtest_engine.generate_ml_training_report(strategy_name)
        else:
            print("No successful backtest results to display")

    except Exception as e:
        print(f"Program execution error: {e}")
        import traceback
        traceback.print_exc()


def plot_results(backtest_engine):
    """Plot results with enhanced visualization for Phase 3"""
    try:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        ax1, ax2, ax3, ax4 = axes.flatten()

        # Plot 1: Cumulative returns comparison
        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink']
        color_idx = 0

        for strategy_name, result in backtest_engine.results.items():
            data = result['data']
            ax1.plot(data.index, data['cumulative_returns'],
                     label=strategy_name, linewidth=2, color=colors[color_idx % len(colors)])
            color_idx += 1

        ax1.set_title('Strategy Performance Comparison', fontsize=14)
        ax1.set_ylabel('Cumulative Returns')
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)

        # Plot 2: Drawdown analysis
        for strategy_name, result in backtest_engine.results.items():
            data = result['data']
            cumulative_max = data['cumulative_returns'].cummax()
            drawdown = (data['cumulative_returns'] - cumulative_max) / cumulative_max
            ax2.plot(data.index, drawdown, label=strategy_name, alpha=0.7)

        ax2.set_title('Strategy Drawdowns', fontsize=14)
        ax2.set_ylabel('Drawdown')
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)

        # Plot 3: Rolling Sharpe ratio (6-month)
        for strategy_name, result in backtest_engine.results.items():
            data = result['data']
            returns = data['strategy_returns']
            rolling_sharpe = returns.rolling(126).mean() / returns.rolling(126).std() * np.sqrt(252)
            ax3.plot(data.index, rolling_sharpe, label=strategy_name, alpha=0.7)

        ax3.set_title('6-Month Rolling Sharpe Ratio', fontsize=14)
        ax3.set_ylabel('Sharpe Ratio')
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3)

        # Plot 4: Strategy correlation heatmap (placeholder)
        strategies_list = list(backtest_engine.results.keys())
        correlation_matrix = np.eye(len(strategies_list))  # Placeholder

        im = ax4.imshow(correlation_matrix, cmap='coolwarm', aspect='auto',
                        vmin=-1, vmax=1)
        ax4.set_xticks(range(len(strategies_list)))
        ax4.set_yticks(range(len(strategies_list)))
        ax4.set_xticklabels([s[:15] for s in strategies_list], rotation=45, ha='right')
        ax4.set_yticklabels([s[:15] for s in strategies_list])
        ax4.set_title('Strategy Returns Correlation', fontsize=14)

        # Add colorbar
        plt.colorbar(im, ax=ax4)

        plt.tight_layout()
        fig.savefig("result.png", dpi=300, bbox_inches="tight")
        plt.show()

    except Exception as e:
        print(f"Plotting error: {e}")


def print_advanced_results(backtest_engine):
    """Print advanced performance metrics for Phase 3"""
    print("\n" + "=" * 100)
    print("ADVANCED BACKTEST RESULTS - STATISTICAL ARBITRAGE & MACHINE LEARNING")
    print("=" * 100)

    metrics_data = []

    for strategy_name, result in backtest_engine.results.items():
        data = result['data']
        returns = data['strategy_returns']
        cumulative_returns = data['cumulative_returns']

        # Basic metrics
        total_return = cumulative_returns.iloc[-1] - 1
        annual_return = (1 + total_return) ** (252 / len(data)) - 1
        volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = annual_return / volatility if volatility > 0 else 0

        # Advanced metrics
        # Maximum drawdown
        cumulative_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns - cumulative_max) / cumulative_max
        max_drawdown = drawdown.min()

        # Calmar ratio
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0

        # Win rate
        winning_trades = (returns > 0).sum()
        total_trades = (returns != 0).sum()
        win_rate = winning_trades / total_trades if total_trades > 0 else 0

        # Profit factor
        gross_profit = returns[returns > 0].sum()
        gross_loss = abs(returns[returns < 0].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf

        # Store metrics
        metrics = {
            'Strategy': strategy_name,
            'Total Return': total_return,
            'Annual Return': annual_return,
            'Volatility': volatility,
            'Sharpe Ratio': sharpe_ratio,
            'Max Drawdown': max_drawdown,
            'Calmar Ratio': calmar_ratio,
            'Win Rate': win_rate,
            'Profit Factor': profit_factor
        }
        metrics_data.append(metrics)

        # Print individual strategy results
        print(f"\n{strategy_name}:")
        print(f"  Total Return:     {total_return:>8.2%}")
        print(f"  Annual Return:    {annual_return:>8.2%}")
        print(f"  Volatility:       {volatility:>8.2%}")
        print(f"  Sharpe Ratio:     {sharpe_ratio:>8.2f}")
        print(f"  Max Drawdown:     {max_drawdown:>8.2%}")
        print(f"  Calmar Ratio:     {calmar_ratio:>8.2f}")
        print(f"  Win Rate:         {win_rate:>8.2%}")
        print(f"  Profit Factor:    {profit_factor:>8.2f}")

        # 添加ML特定分析
        if 'ML' in strategy_name:
            # 计算ML特定指标
            ml_metrics = backtest_engine.calculate_ml_specific_metrics(strategy_name)
            if ml_metrics:
                print(f"  ML特定指标:")
                print(f"  换手率:        {ml_metrics.get('turnover_rate', 0):>8.2%}")
                print(f"  平均持仓周期:  {ml_metrics.get('avg_holding_period', 0):>8.1f}天")

        print("-" * 80)

    # Print summary table
    print("\n" + "=" * 100)
    print("PERFORMANCE SUMMARY")
    print("=" * 100)

    # Create summary DataFrame for better visualization
    metrics_df = pd.DataFrame(metrics_data)
    metrics_df = metrics_df.round(4)

    # Convert to percentage for display
    percent_columns = ['Total Return', 'Annual Return', 'Volatility', 'Max Drawdown', 'Win Rate']
    for col in percent_columns:
        metrics_df[col] = metrics_df[col].apply(lambda x: f"{x:.2%}")

    # Format other columns
    metrics_df['Sharpe Ratio'] = metrics_df['Sharpe Ratio'].apply(lambda x: f"{x:.2f}")
    metrics_df['Calmar Ratio'] = metrics_df['Calmar Ratio'].apply(lambda x: f"{x:.2f}")
    metrics_df['Profit Factor'] = metrics_df['Profit Factor'].apply(lambda x: f"{x:.2f}")

    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()