import pandas as pd
import numpy as np
from core.strategy import Strategy
from core.factor_calculator import factor_calculator
from core.portfolio_optimizer import portfolio_optimizer
from typing import Dict, List


class MultiFactorStrategy(Strategy):
    """Multi-Factor Strategy with Portfolio Optimization"""

    def __init__(self, stock_codes: List[str], rebalance_freq: str = 'M'):
        super().__init__("Multi-Factor Strategy")
        self.stock_codes = stock_codes
        self.rebalance_freq = rebalance_freq  # 'M' for monthly, 'Q' for quarterly
        self.factors_data = {}
        self.portfolio_weights = {}

    def prepare_data(self):
        """Prepare multi-stock data and calculate factors"""
        if self.data is not None:
            return self  # Already prepared

        # Get data for all stocks
        from core.data_manager import data_manager
        all_data, common_dates = data_manager.get_multiple_stocks_data(self.stock_codes)

        # Calculate factors for each stock
        self.factors_data = {}
        for code, data in all_data.items():
            factors_df = factor_calculator.calculate_all_factors(data)
            self.factors_data[code] = factors_df

        # Use first stock's data as base (all have same dates after alignment)
        if all_data:
            first_code = list(all_data.keys())[0]
            self.data = all_data[first_code].copy()

        return self

    def calculate_composite_score(self, factors_df: pd.DataFrame) -> float:
        """Calculate composite factor score using equal weighting"""
        # Simple equal-weighted composite score
        valid_factors = factors_df.dropna()
        if len(valid_factors) == 0:
            return 0
        return valid_factors.mean()

    def generate_signals(self):
        """Generate portfolio rebalancing signals - Improved version"""
        if not self.factors_data:
            raise ValueError("No factors data available. Call prepare_data() first.")

        all_dates = self.data.index

        # Initialize signals
        signals = pd.Series(0, index=all_dates)
        self.portfolio_weights = {}

        # Improved rebalance logic - rebalance monthly on first trading day of each month
        rebalance_dates = []
        current_month = None

        for date in all_dates:
            if date.month != current_month:
                rebalance_dates.append(date)
                current_month = date.month

        print(f"Rebalance frequency: Monthly, Rebalance dates: {len(rebalance_dates)}")

        last_rebalance_date = None

        for i, current_date in enumerate(all_dates):
            if current_date in rebalance_dates:
                try:
                    # Use lookback period for factor calculation to avoid look-ahead bias
                    lookback_days = 20
                    lookback_date_idx = max(0, i - lookback_days)
                    lookback_date = all_dates[lookback_date_idx]

                    # Calculate composite scores for all stocks
                    scores = {}
                    valid_stocks = []

                    for code in self.stock_codes:
                        if (code in self.factors_data and
                                lookback_date in self.factors_data[code].index):
                            # Use lookback date factors, not current date
                            factors = self.factors_data[code].loc[lookback_date]
                            score = self.calculate_composite_score(factors)
                            if not pd.isna(score):
                                scores[code] = score
                                valid_stocks.append(code)

                    if len(valid_stocks) >= 3:
                        # Rank stocks by composite score
                        ranked_stocks = sorted(scores.items(), key=lambda x: x[1], reverse=True)

                        # Select top 30-50% of stocks
                        n_select = max(3, min(6, int(len(ranked_stocks) * 0.4)))
                        selected_stocks = [stock for stock, score in ranked_stocks[:n_select]]

                        # Equal weight portfolio
                        weight = 1.0 / len(selected_stocks)
                        weights_dict = {stock: weight for stock in selected_stocks}

                        self.portfolio_weights[current_date] = weights_dict
                        signals.loc[current_date] = 1
                        last_rebalance_date = current_date

                        print(f"{current_date}: Selected {len(selected_stocks)} stocks: {selected_stocks}")

                except Exception as e:
                    print(f"Error in portfolio construction on {current_date}: {e}")

            elif last_rebalance_date is not None:
                # Carry forward last portfolio weights
                self.portfolio_weights[current_date] = self.portfolio_weights[last_rebalance_date]
                signals.loc[current_date] = 0

        self.signals = signals
        return self


class MeanVarianceStrategy(Strategy):
    """Mean-Variance Optimization Strategy"""

    def __init__(self, stock_codes: List[str], lookback_period: int = 126):  # 6 months
        super().__init__("Mean-Variance Strategy")
        self.stock_codes = stock_codes
        self.lookback_period = lookback_period
        self.portfolio_weights = {}

    def prepare_data(self):
        """Prepare multi-stock data"""
        if self.data is not None:
            return self

        from core.data_manager import data_manager
        all_data, common_dates = data_manager.get_multiple_stocks_data(self.stock_codes)

        if all_data:
            first_code = list(all_data.keys())[0]
            self.data = all_data[first_code].copy()

        self.all_data = all_data
        return self

    def generate_signals(self):
        """Generate signals using mean-variance optimization - Improved version"""
        if not hasattr(self, 'all_data'):
            raise ValueError("No data available. Call prepare_data() first.")

        all_dates = self.data.index
        signals = pd.Series(0, index=all_dates)
        self.portfolio_weights = {}

        # Rebalance quarterly instead of daily
        rebalance_dates = []
        current_quarter = None

        for date in all_dates:
            quarter = (date.year, (date.month - 1) // 3 + 1)
            if quarter != current_quarter:
                rebalance_dates.append(date)
                current_quarter = quarter

        print(f"Mean-Variance: Quarterly rebalance, {len(rebalance_dates)} rebalance dates")

        last_rebalance_date = None

        for i, current_date in enumerate(all_dates):
            if current_date in rebalance_dates and i >= self.lookback_period:
                lookback_start = all_dates[i - self.lookback_period]

                try:
                    # Calculate returns for lookback period
                    returns_data = {}
                    valid_stocks = []

                    for code, data in self.all_data.items():
                        if current_date in data.index and lookback_start in data.index:
                            lookback_data = data.loc[lookback_start:current_date]
                            if len(lookback_data) >= self.lookback_period * 0.8:
                                returns = lookback_data['close'].pct_change().dropna()
                                if len(returns) > 10:  # At least 10 trading days
                                    returns_data[code] = returns
                                    valid_stocks.append(code)

                    if len(valid_stocks) >= 3:  # Reduced from 5 to 3
                        # Create returns DataFrame
                        returns_df = pd.DataFrame(returns_data)

                        # Remove any stocks with NaN returns
                        returns_df = returns_df.dropna(axis=1)
                        valid_stocks = list(returns_df.columns)

                        if len(valid_stocks) >= 3:
                            # Calculate expected returns and covariance
                            expected_returns = portfolio_optimizer.calculate_expected_returns(returns_df)
                            cov_matrix = portfolio_optimizer.calculate_covariance_matrix(returns_df)

                            # Mean-variance optimization
                            result = portfolio_optimizer.maximize_sharpe_ratio(expected_returns, cov_matrix)

                            # Store weights (only significant allocations > 5%)
                            weights_dict = {}
                            for j, code in enumerate(valid_stocks):
                                if result['weights'][j] > 0.05:  # Increased from 1% to 5%
                                    weights_dict[code] = result['weights'][j]

                            if weights_dict:
                                # Normalize weights to sum to 1
                                total_weight = sum(weights_dict.values())
                                weights_dict = {k: v / total_weight for k, v in weights_dict.items()}

                                self.portfolio_weights[current_date] = weights_dict
                                signals.loc[current_date] = 1
                                last_rebalance_date = current_date

                                print(f"{current_date}: Optimized portfolio with {len(weights_dict)} stocks")
                                print(
                                    f"  Expected return: {result['expected_return']:.2%}, Volatility: {result['volatility']:.2%}")

                except Exception as e:
                    print(f"Error in mean-variance optimization on {current_date}: {e}")

            elif last_rebalance_date is not None:
                # Carry forward last portfolio weights
                self.portfolio_weights[current_date] = self.portfolio_weights[last_rebalance_date]
                signals.loc[current_date] = 0

        self.signals = signals
        return self