import pandas as pd
import numpy as np
from core.strategy import Strategy
from core.statistical_arbitrage import statistical_arbitrage
from core.data_manager import data_manager


class PairTradingStrategy(Strategy):
    """Statistical Arbitrage Pair Trading Strategy"""

    def __init__(self, stock_pairs: list = None, entry_z: float = 2.0, exit_z: float = 0.5):
        super().__init__("Pair Trading Strategy")
        self.stock_pairs = stock_pairs or []
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.lookback = 20
        self.pair_data = {}
        self.current_positions = {}

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

    def generate_signals(self):
        """Generate pair trading signals"""
        if not self.stock_pairs:
            raise ValueError("No stock pairs defined")

        all_dates = self.data.index
        signals = pd.Series(0, index=all_dates)
        self.pair_signals = {}

        for pair in self.stock_pairs:
            stock1, stock2 = pair
            if stock1 not in self.all_data or stock2 not in self.all_data:
                continue

            # Calculate spread and z-score
            series1 = self.all_data[stock1]['close']
            series2 = self.all_data[stock2]['close']

            # Align series
            common_dates = series1.index.intersection(series2.index)
            series1 = series1.loc[common_dates]
            series2 = series2.loc[common_dates]

            # Calculate hedge ratio (simplified)
            hedge_ratio = np.polyfit(series1, series2, 1)[0]
            spread = statistical_arbitrage.calculate_spread(series1, series2, hedge_ratio)
            zscore = statistical_arbitrage.calculate_zscore(spread, self.lookback)

            # Generate signals for this pair
            pair_signals = pd.Series(0, index=common_dates)

            # Long spread (buy stock1, sell stock2) when zscore < -entry_z
            pair_signals[zscore < -self.entry_z] = 1
            # Short spread (sell stock1, buy stock2) when zscore > entry_z
            pair_signals[zscore > self.entry_z] = -1
            # Exit when zscore crosses exit_z
            pair_signals[(zscore >= -self.exit_z) & (zscore <= self.exit_z)] = 0

            self.pair_signals[pair] = {
                'signals': pair_signals,
                'zscore': zscore,
                'spread': spread,
                'hedge_ratio': hedge_ratio
            }

        # Combine signals from all pairs (simple approach: use first pair's signals)
        if self.pair_signals:
            first_pair = list(self.pair_signals.keys())[0]
            combined_signals = self.pair_signals[first_pair]['signals']
            signals = combined_signals.reindex(all_dates, fill_value=0)

            print(f"Pair Trading: {len(self.pair_signals)} pairs, signals range: {signals.min()} to {signals.max()}")

        self.signals = signals
        return self


class MeanReversionStrategy(Strategy):
    """Mean Reversion Strategy based on Z-score"""

    def __init__(self, lookback_period: int = 20, entry_z: float = 2.0, exit_z: float = 0.5):
        super().__init__(f"Mean Reversion ({lookback_period}d)")
        self.lookback_period = lookback_period
        self.entry_z = entry_z
        self.exit_z = exit_z

    def generate_signals(self):
        """Generate mean reversion signals"""
        if self.data is None:
            raise ValueError("No data available")

        # Calculate z-score of price
        price = self.data['close']
        rolling_mean = price.rolling(window=self.lookback_period).mean()
        rolling_std = price.rolling(window=self.lookback_period).std()
        zscore = (price - rolling_mean) / rolling_std

        # Generate signals
        signals = pd.Series(0, index=self.data.index)

        # Buy when price is significantly below mean (oversold)
        signals[zscore < -self.entry_z] = 1
        # Sell when price is significantly above mean (overbought)
        signals[zscore > self.entry_z] = -1
        # Exit when price returns to normal range
        signals[(zscore >= -self.exit_z) & (zscore <= self.exit_z)] = 0

        # Ensure we have enough data
        signals.iloc[:self.lookback_period] = 0

        # Shift to avoid look-ahead bias
        signals = signals.shift(1).fillna(0)

        self.signals = signals
        print(f"Mean Reversion: {len(signals)} signals, "
              f"Long: {(signals == 1).sum()}, Short: {(signals == -1).sum()}")

        return self