import pandas as pd
import numpy as np
from core.strategy import Strategy
from core.statistical_arbitrage import statistical_arbitrage
from core.data_manager import data_manager


class PairTradingStrategy(Strategy):
    """Statistical Arbitrage Pair Trading Strategy with Walk-Forward"""

    def __init__(self, stock_pairs: list = None, entry_z: float = 2.0, exit_z: float = 0.5,
                 lookback: int = 60, min_lookback: int = 30):
        super().__init__("Pair Trading Strategy")
        self.stock_pairs = stock_pairs or []
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.lookback = lookback  # 用于计算对冲比率
        self.min_lookback = min_lookback  # 最小数据量
        self.pair_data = {}
        self.training_history = []

    def prepare_data(self):
        """Prepare data for pair trading"""
        if not self.stock_pairs:
            # Auto-detect cointegrated pairs
            self._find_best_pairs()

        # Get data for all stocks in pairs
        all_stocks = set()
        for pair in self.stock_pairs:
            all_stocks.update(pair)

        self.all_data, common_dates = data_manager.get_multiple_stocks_data(list(all_stocks))

        # Use first stock as base data
        if self.all_data:
            first_stock = list(self.all_data.keys())[0]
            self.data = self.all_data[first_stock].copy()

        return self

    def _find_best_pairs(self, max_pairs: int = 3):
        """Find best cointegrated pairs automatically"""
        print("Finding cointegrated pairs...")

        # Use a predefined set of stocks for pair searching
        search_pool = [
            'sz.000001', 'sz.000002', 'sh.600036', 'sh.600519',
            'sz.000858', 'sh.601318', 'sz.000333'
        ]

        # Get price data
        price_data = {}
        for stock in search_pool:
            try:
                data = data_manager.get_stock_data(stock)
                price_data[stock] = data
            except:
                continue

        # Find cointegrated pairs
        pairs = statistical_arbitrage.find_cointegrated_pairs(price_data)

        # Select best pairs
        selected_pairs = []
        for pair_info in pairs[:max_pairs]:
            stock1, stock2 = pair_info['pair']
            selected_pairs.append((stock1, stock2))
            print(f"Selected pair: {stock1} - {stock2} (p-value: {pair_info['pvalue']:.4f})")

        self.stock_pairs = selected_pairs

    def _calculate_rolling_hedge_ratio(self, series1, series2, window=60):
        """滚动计算对冲比率，避免数据泄露"""
        hedge_ratios = pd.Series(index=series1.index, dtype=float)

        for i in range(window, len(series1)):
            # 只用历史数据计算对冲比率
            hist_series1 = series1.iloc[i - window:i]
            hist_series2 = series2.iloc[i - window:i]

            if len(hist_series1) >= window // 2:  # 至少有一半数据
                try:
                    hedge_ratio = np.polyfit(hist_series1, hist_series2, 1)[0]
                    hedge_ratios.iloc[i] = hedge_ratio
                except:
                    hedge_ratios.iloc[i] = np.nan

        return hedge_ratios.ffill()

    def generate_signals(self):
        """Generate pair trading signals with Walk-Forward"""
        if not self.stock_pairs:
            raise ValueError("No stock pairs defined")

        all_dates = self.data.index
        signals = pd.Series(0, index=all_dates)
        self.pair_signals = {}

        print(f"Generating pair trading signals with Walk-Forward (lookback: {self.lookback})")

        for pair in self.stock_pairs:
            stock1, stock2 = pair
            if stock1 not in self.all_data or stock2 not in self.all_data:
                continue

            # Get price series
            series1 = self.all_data[stock1]['close']
            series2 = self.all_data[stock2]['close']

            # Align series
            common_dates = series1.index.intersection(series2.index)
            series1_aligned = series1.loc[common_dates]
            series2_aligned = series2.loc[common_dates]

            # 滚动计算对冲比率（避免数据泄露）
            hedge_ratios = self._calculate_rolling_hedge_ratio(
                series1_aligned, series2_aligned, self.lookback
            )

            # 计算价差和z-score
            spread = series2_aligned - hedge_ratios * series1_aligned

            # 滚动计算z-score
            zscore = pd.Series(index=spread.index, dtype=float)
            for i in range(self.lookback, len(spread)):
                hist_spread = spread.iloc[i - self.lookback:i]
                if len(hist_spread) >= self.min_lookback:
                    mean = hist_spread.mean()
                    std = hist_spread.std()
                    if std > 0:
                        zscore.iloc[i] = (spread.iloc[i] - mean) / std

            zscore = zscore.ffill()

            # 生成信号
            pair_signals = pd.Series(0, index=common_dates)

            # Long spread (buy stock2, sell stock1) when spread is low
            pair_signals[(zscore < -self.entry_z) & (zscore.notna())] = 1
            # Short spread (sell stock2, buy stock1) when spread is high
            pair_signals[(zscore > self.entry_z) & (zscore.notna())] = -1
            # Exit when spread returns to normal
            pair_signals[(zscore >= -self.exit_z) & (zscore <= self.exit_z) & (zscore.notna())] = 0

            self.pair_signals[pair] = {
                'signals': pair_signals,
                'zscore': zscore,
                'spread': spread,
                'hedge_ratios': hedge_ratios
            }

            print(f"Pair {stock1}-{stock2}: {pair_signals.abs().sum()} trading signals")

        # 智能组合多个对的信号
        signals = self._combine_pair_signals(all_dates)

        # 避免前视偏差
        signals = signals.shift(1).fillna(0)

        self.signals = signals
        print(f"Final combined signals: Long: {(signals == 1).sum()}, Short: {(signals == -1).sum()}")

        return self

    def _combine_pair_signals(self, all_dates):
        """智能组合多个对的信号"""
        if not self.pair_signals:
            return pd.Series(0, index=all_dates)

        # 方法1: 投票机制
        all_signals = pd.DataFrame(index=all_dates)

        for pair, data in self.pair_signals.items():
            signal_name = f"{pair[0]}_{pair[1]}"
            aligned_signals = data['signals'].reindex(all_dates, fill_value=0)
            all_signals[signal_name] = aligned_signals

        # 多数投票
        combined_signals = all_signals.sum(axis=1)
        final_signals = pd.Series(0, index=all_dates)

        # 设置阈值：至少一半的配对同意
        threshold = len(self.pair_signals) // 2
        final_signals[combined_signals > threshold] = 1
        final_signals[combined_signals < -threshold] = -1

        return final_signals


class MeanReversionStrategy(Strategy):
    """Mean Reversion Strategy with Walk-Forward Z-score"""

    def __init__(self, lookback_period: int = 20, entry_z: float = 2.0, exit_z: float = 0.5):
        super().__init__(f"Mean Reversion ({lookback_period}d)")
        self.lookback_period = lookback_period
        self.entry_z = entry_z
        self.exit_z = exit_z

    def generate_signals(self):
        """Generate mean reversion signals with proper rolling calculation"""
        if self.data is None:
            raise ValueError("No data available")

        price = self.data['close']

        # 滚动计算z-score（避免数据泄露）
        zscore = pd.Series(index=price.index, dtype=float)

        for i in range(self.lookback_period, len(price)):
            # 只用历史数据计算
            hist_prices = price.iloc[i - self.lookback_period:i]
            mean = hist_prices.mean()
            std = hist_prices.std()

            if std > 0:
                zscore.iloc[i] = (price.iloc[i] - mean) / std

        zscore = zscore.ffill()

        # Generate signals
        signals = pd.Series(0, index=self.data.index)

        # Buy when oversold
        signals[(zscore < -self.entry_z) & (zscore.notna())] = 1
        # Sell when overbought
        signals[(zscore > self.entry_z) & (zscore.notna())] = -1
        # Exit when returns to normal
        signals[(zscore >= -self.exit_z) & (zscore <= self.exit_z) & (zscore.notna())] = 0

        # Shift to avoid look-ahead bias
        signals = signals.shift(1).fillna(0)

        self.signals = signals
        print(f"Mean Reversion: {len(signals)} signals, "
              f"Long: {(signals == 1).sum()}, Short: {(signals == -1).sum()}")

        return self