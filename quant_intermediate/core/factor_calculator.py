import pandas as pd
import numpy as np
from typing import Dict, List


class FactorCalculator:
    """Factor Calculator for multi-factor models"""

    @staticmethod
    def calculate_momentum_factor(data: pd.DataFrame, lookback_period: int = 20) -> pd.Series:
        """Calculate momentum factor: past N-day return"""
        return data['close'].pct_change(lookback_period)

    @staticmethod
    def calculate_volume_factor(data: pd.DataFrame, lookback_period: int = 10) -> pd.Series:
        """Calculate volume factor: volume relative to N-day average"""
        avg_volume = data['volume'].rolling(window=lookback_period).mean()
        return data['volume'] / avg_volume

    @staticmethod
    def calculate_volatility_factor(data: pd.DataFrame, lookback_period: int = 20) -> pd.Series:
        """Calculate volatility factor: N-day price volatility"""
        returns = data['close'].pct_change()
        return returns.rolling(window=lookback_period).std()

    @staticmethod
    def calculate_moving_average_factor(data: pd.DataFrame, short_window: int = 10, long_window: int = 30) -> pd.Series:
        """Calculate MA factor: short MA / long MA ratio"""
        short_ma = data['close'].rolling(window=short_window).mean()
        long_ma = data['close'].rolling(window=long_window).mean()
        return short_ma / long_ma - 1  # Relative strength

    @staticmethod
    def calculate_rsi_factor(data: pd.DataFrame, window: int = 14) -> pd.Series:
        """Calculate RSI factor"""
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi / 100  # Normalize to 0-1

    @staticmethod
    def calculate_all_factors(data: pd.DataFrame) -> pd.DataFrame:
        """Calculate all factors for a single stock"""
        factors_df = pd.DataFrame(index=data.index)

        # Momentum factors
        factors_df['momentum_1m'] = FactorCalculator.calculate_momentum_factor(data, 20)
        factors_df['momentum_3m'] = FactorCalculator.calculate_momentum_factor(data, 60)

        # Volume factors
        factors_df['volume_ratio'] = FactorCalculator.calculate_volume_factor(data, 10)

        # Volatility factors
        factors_df['volatility_1m'] = FactorCalculator.calculate_volatility_factor(data, 20)

        # Trend factors
        factors_df['ma_ratio'] = FactorCalculator.calculate_moving_average_factor(data, 10, 30)

        # Mean reversion factors
        factors_df['rsi'] = FactorCalculator.calculate_rsi_factor(data, 14)

        return factors_df

    @staticmethod
    def normalize_factors(factors_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Normalize factors cross-sectionally (Z-score normalization)"""
        all_dates = list(factors_dict.values())[0].index
        normalized_factors = {}

        for date in all_dates:
            # Collect all factor values for this date
            date_factors = {}
            for stock, factors_df in factors_dict.items():
                if date in factors_df.index:
                    date_factors[stock] = factors_df.loc[date]

            if not date_factors:
                continue

            # Create DataFrame for this date's cross-sectional data
            cross_section = pd.DataFrame(date_factors).T

            # Z-score normalization
            normalized_cross_section = (cross_section - cross_section.mean()) / cross_section.std()

            # Store normalized values
            for stock in normalized_cross_section.index:
                if stock not in normalized_factors:
                    normalized_factors[stock] = pd.DataFrame(index=all_dates, columns=normalized_cross_section.columns)
                normalized_factors[stock].loc[date] = normalized_cross_section.loc[stock]

        return normalized_factors


# Singleton instance
factor_calculator = FactorCalculator()