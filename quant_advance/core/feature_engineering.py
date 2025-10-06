import pandas as pd
import numpy as np
from typing import List, Dict
from core.factor_calculator import factor_calculator


class FeatureEngineer:
    """Feature Engineering for Machine Learning Models"""

    @staticmethod
    def create_technical_features(data: pd.DataFrame, lookback_periods: List[int] = None) -> pd.DataFrame:
        """Create comprehensive technical features"""
        if lookback_periods is None:
            lookback_periods = [5, 10, 20, 30, 60]

        features_df = pd.DataFrame(index=data.index)

        # Price-based features
        features_df['returns_1d'] = data['close'].pct_change()
        features_df['returns_5d'] = data['close'].pct_change(5)
        features_df['returns_20d'] = data['close'].pct_change(20)

        # Volatility features
        for period in lookback_periods:
            features_df[f'volatility_{period}d'] = data['close'].pct_change().rolling(period).std()

        # Moving average features
        features_df['ma_5'] = data['close'].rolling(5).mean()
        features_df['ma_20'] = data['close'].rolling(20).mean()
        features_df['ma_ratio'] = features_df['ma_5'] / features_df['ma_20'] - 1

        # Price position relative to range
        features_df['high_20'] = data['high'].rolling(20).max()
        features_df['low_20'] = data['low'].rolling(20).min()
        features_df['price_position'] = (data['close'] - features_df['low_20']) / (
                    features_df['high_20'] - features_df['low_20'])

        # Volume features
        features_df['volume_ma_5'] = data['volume'].rolling(5).mean()
        features_df['volume_ratio'] = data['volume'] / features_df['volume_ma_5']
        features_df['volume_price_trend'] = data['volume'] * data['close'].pct_change()

        # Momentum indicators
        features_df['rsi_14'] = FeatureEngineer.calculate_rsi(data['close'], 14)
        features_df['macd'] = FeatureEngineer.calculate_macd(data['close'])

        # Mean reversion features
        features_df['price_zscore_20'] = (data['close'] - data['close'].rolling(20).mean()) / data['close'].rolling(
            20).std()

        # Drop NaN values
        features_df = features_df.dropna()

        return features_df

    @staticmethod
    def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    @staticmethod
    def calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
        """Calculate MACD indicator"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        return macd

    @staticmethod
    def create_target_variable(data: pd.DataFrame, forward_periods: int = 5) -> pd.Series:
        """Create target variable for prediction"""
        # Future returns (for regression)
        future_returns = data['close'].pct_change(forward_periods).shift(-forward_periods)

        # Binary classification: 1 if future return > 0, else 0
        binary_target = (future_returns > 0).astype(int)

        return binary_target

    @staticmethod
    def prepare_ml_dataset(data: pd.DataFrame, forward_periods: int = 5) -> tuple:
        """Prepare dataset for machine learning"""
        features = FeatureEngineer.create_technical_features(data)
        target = FeatureEngineer.create_target_variable(data, forward_periods)

        # Align features and target
        common_index = features.index.intersection(target.index)
        features = features.loc[common_index]
        target = target.loc[common_index]

        return features, target


# Singleton instance
feature_engineer = FeatureEngineer()