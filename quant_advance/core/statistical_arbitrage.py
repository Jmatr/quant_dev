import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import coint
from scipy import stats
import warnings

warnings.filterwarnings('ignore')


class StatisticalArbitrage:
    """Statistical Arbitrage for Pair Trading"""

    @staticmethod
    def find_cointegrated_pairs(price_data: dict, significance: float = 0.05) -> list:
        """Find cointegrated pairs using Engle-Granger test"""
        pairs = []
        symbols = list(price_data.keys())

        for i in range(len(symbols)):
            for j in range(i + 1, len(symbols)):
                stock1 = symbols[i]
                stock2 = symbols[j]

                # Get price series
                series1 = price_data[stock1]['close']
                series2 = price_data[stock2]['close']

                # Align dates
                common_dates = series1.index.intersection(series2.index)
                if len(common_dates) < 100:  # Need sufficient data
                    continue

                series1_aligned = series1.loc[common_dates]
                series2_aligned = series2.loc[common_dates]

                # Perform cointegration test
                try:
                    score, pvalue, _ = coint(series1_aligned, series2_aligned)

                    if pvalue < significance:
                        # Calculate hedge ratio using OLS
                        hedge_ratio = np.polyfit(series1_aligned, series2_aligned, 1)[0]

                        pairs.append({
                            'pair': (stock1, stock2),
                            'pvalue': pvalue,
                            'score': score,
                            'hedge_ratio': hedge_ratio,
                            'num_observations': len(common_dates)
                        })
                except:
                    continue

        # Sort by p-value (most significant first)
        pairs.sort(key=lambda x: x['pvalue'])
        return pairs

    @staticmethod
    def calculate_spread(series1: pd.Series, series2: pd.Series, hedge_ratio: float) -> pd.Series:
        """Calculate spread between two series"""
        return series1 - hedge_ratio * series2

    @staticmethod
    def calculate_zscore(spread: pd.Series, lookback: int = 20) -> pd.Series:
        """Calculate Z-score of spread"""
        spread_mean = spread.rolling(window=lookback).mean()
        spread_std = spread.rolling(window=lookback).std()
        return (spread - spread_mean) / spread_std

    @staticmethod
    def find_optimal_parameters(series1: pd.Series, series2: pd.Series,
                                hedge_ratio: float, max_lookback: int = 60) -> dict:
        """Find optimal parameters for pair trading"""
        spread = StatisticalArbitrage.calculate_spread(series1, series2, hedge_ratio)

        best_sharpe = -np.inf
        best_params = {}

        for lookback in range(10, max_lookback + 1, 5):
            for entry_z in np.arange(1.0, 2.5, 0.25):
                for exit_z in np.arange(0.0, 1.0, 0.25):

                    # Generate signals
                    zscore = StatisticalArbitrage.calculate_zscore(spread, lookback)
                    signals = pd.Series(0, index=zscore.index)

                    # Long spread (buy series1, sell series2) when zscore < -entry_z
                    signals[zscore < -entry_z] = 1
                    # Short spread (sell series1, buy series2) when zscore > entry_z
                    signals[zscore > entry_z] = -1
                    # Exit when zscore crosses exit_z
                    signals[(zscore >= -exit_z) & (zscore <= exit_z)] = 0

                    # Calculate returns (simplified)
                    returns = signals.shift(1) * spread.pct_change()
                    sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else -np.inf

                    if sharpe > best_sharpe and not np.isinf(sharpe):
                        best_sharpe = sharpe
                        best_params = {
                            'lookback': lookback,
                            'entry_z': entry_z,
                            'exit_z': exit_z,
                            'sharpe': sharpe
                        }

        return best_params


# Singleton instance
statistical_arbitrage = StatisticalArbitrage()