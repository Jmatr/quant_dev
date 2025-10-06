import pandas as pd
import numpy as np
from scipy.optimize import minimize
import warnings


class PortfolioOptimizer:
    """Portfolio Optimizer using Modern Portfolio Theory"""

    @staticmethod
    def calculate_expected_returns(returns_df: pd.DataFrame, method: str = 'historical') -> pd.Series:
        """Calculate expected returns for each asset"""
        if method == 'historical':
            return returns_df.mean() * 252  # Annualized
        else:
            raise ValueError(f"Unknown method: {method}")

    @staticmethod
    def calculate_covariance_matrix(returns_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate covariance matrix (annualized)"""
        return returns_df.cov() * 252

    @staticmethod
    def portfolio_volatility(weights: np.array, cov_matrix: pd.DataFrame) -> float:
        """Calculate portfolio volatility"""
        return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

    @staticmethod
    def portfolio_return(weights: np.array, expected_returns: pd.Series) -> float:
        """Calculate portfolio expected return"""
        return np.dot(weights, expected_returns)

    @staticmethod
    def sharpe_ratio(weights: np.array, expected_returns: pd.Series, cov_matrix: pd.DataFrame) -> float:
        """Calculate Sharpe ratio (negative for minimization)"""
        port_return = PortfolioOptimizer.portfolio_return(weights, expected_returns)
        port_vol = PortfolioOptimizer.portfolio_volatility(weights, cov_matrix)
        return - (port_return / port_vol) if port_vol > 0 else 0

    @staticmethod
    def minimize_volatility(expected_returns: pd.Series, cov_matrix: pd.DataFrame,
                            target_return: float = None) -> dict:
        """Minimize portfolio volatility"""
        n_assets = len(expected_returns)

        # Constraints
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]  # weights sum to 1

        if target_return is not None:
            constraints.append({'type': 'eq', 'fun': lambda x:
            PortfolioOptimizer.portfolio_return(x, expected_returns) - target_return})

        # Bounds (no short selling)
        bounds = tuple((0, 1) for _ in range(n_assets))

        # Initial guess (equal weights)
        init_guess = np.array([1 / n_assets] * n_assets)

        # Optimization
        result = minimize(
            fun=PortfolioOptimizer.portfolio_volatility,
            x0=init_guess,
            args=(cov_matrix,),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )

        return {
            'weights': result.x,
            'expected_return': PortfolioOptimizer.portfolio_return(result.x, expected_returns),
            'volatility': result.fun,
            'sharpe_ratio': -PortfolioOptimizer.sharpe_ratio(result.x, expected_returns, cov_matrix)
        }

    @staticmethod
    def maximize_sharpe_ratio(expected_returns: pd.Series, cov_matrix: pd.DataFrame) -> dict:
        """Maximize Sharpe ratio"""
        n_assets = len(expected_returns)

        # Constraints
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]

        # Bounds (no short selling)
        bounds = tuple((0, 1) for _ in range(n_assets))

        # Initial guess
        init_guess = np.array([1 / n_assets] * n_assets)

        # Optimization
        result = minimize(
            fun=PortfolioOptimizer.sharpe_ratio,
            x0=init_guess,
            args=(expected_returns, cov_matrix),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )

        optimal_weights = result.x
        expected_return = PortfolioOptimizer.portfolio_return(optimal_weights, expected_returns)
        volatility = PortfolioOptimizer.portfolio_volatility(optimal_weights, cov_matrix)

        return {
            'weights': optimal_weights,
            'expected_return': expected_return,
            'volatility': volatility,
            'sharpe_ratio': -result.fun
        }

    @staticmethod
    def equal_weight_portfolio(n_assets: int) -> np.array:
        """Generate equal weight portfolio"""
        return np.array([1 / n_assets] * n_assets)


# Singleton instance
portfolio_optimizer = PortfolioOptimizer()