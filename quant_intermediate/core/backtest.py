import pandas as pd
import numpy as np
from utils.config import config


class BacktestEngine:
    """Backtest Engine - Enhanced for Portfolio Strategies"""

    def __init__(self, initial_cash=None, commission_rate=None):
        self.initial_cash = initial_cash or config.INITIAL_CASH
        self.commission_rate = commission_rate or config.COMMISSION_RATE
        self.results = {}

    def run_backtest(self, strategy):
        """Run backtest - Enhanced for portfolio strategies"""
        if strategy.signals is None:
            strategy.prepare_data().generate_signals()

        # Check if this is a portfolio strategy
        is_portfolio_strategy = hasattr(strategy, 'portfolio_weights')

        if is_portfolio_strategy:
            return self._run_portfolio_backtest(strategy)
        else:
            return self._run_single_stock_backtest(strategy)

    def _run_single_stock_backtest(self, strategy):
        """Run backtest for single stock strategy"""
        data = strategy.data.copy()
        signals = strategy.signals.copy()

        # Initialize backtest data columns
        data['position'] = 0
        data['cash'] = self.initial_cash
        data['holdings'] = 0
        data['total'] = self.initial_cash
        data['returns'] = 0.0

        position = 0
        cash = self.initial_cash

        print(f"Single stock backtest started: data length={len(data)}")

        for i in range(len(data)):
            current_date = data.index[i]
            current_price = data['close'].iloc[i]
            current_signal = signals.iloc[i]

            # Execute trading signals
            if current_signal == 1 and position == 0:
                available_cash = cash * (1 - self.commission_rate)
                new_position = int(available_cash // current_price)

                if new_position > 0:
                    trade_value = new_position * current_price
                    trade_commission = trade_value * self.commission_rate

                    position = new_position
                    cash -= (trade_value + trade_commission)

            elif current_signal == 0 and position > 0:
                trade_value = position * current_price
                trade_commission = trade_value * self.commission_rate
                cash += (trade_value - trade_commission)
                position = 0

            # Update current row status
            data.iloc[i, data.columns.get_loc('position')] = float(position)
            data.iloc[i, data.columns.get_loc('cash')] = float(cash)
            data.iloc[i, data.columns.get_loc('holdings')] = float(position * current_price)
            data.iloc[i, data.columns.get_loc('total')] = float(cash + position * current_price)

        # Calculate returns
        self._calculate_performance(data)

        self.results[strategy.name] = {
            'data': data,
            'strategy': strategy
        }

        final_value = data['total'].iloc[-1]
        total_return = (final_value - self.initial_cash) / self.initial_cash
        print(f"Backtest completed: Final value={final_value:.2f}, Total return={total_return:.2%}")

        return data

    def _run_portfolio_backtest(self, strategy):
        """Run backtest for portfolio strategy"""
        from core.data_manager import data_manager

        # Get all stock data
        all_data, common_dates = data_manager.get_multiple_stocks_data(strategy.stock_codes)
        portfolio_weights = strategy.portfolio_weights

        # Use first stock's data as base
        base_data = list(all_data.values())[0].copy()
        data = base_data[['close']].copy()

        # Initialize portfolio tracking
        data['cash'] = self.initial_cash
        data['holdings'] = 0
        data['total'] = self.initial_cash
        data['num_stocks'] = 0

        current_weights = {}
        cash = self.initial_cash
        positions = {}  # stock -> shares

        print(f"Portfolio backtest started: {len(strategy.stock_codes)} stocks")

        for i in range(len(data)):
            current_date = data.index[i]

            # Check if rebalance needed
            if current_date in portfolio_weights:
                new_weights = portfolio_weights[current_date]

                # Sell existing positions not in new portfolio
                stocks_to_sell = [s for s in positions.keys() if s not in new_weights]
                for stock in stocks_to_sell:
                    if stock in all_data and current_date in all_data[stock].index:
                        price = all_data[stock].loc[current_date, 'close']
                        trade_value = positions[stock] * price
                        trade_commission = trade_value * self.commission_rate
                        cash += (trade_value - trade_commission)
                        del positions[stock]

                # Calculate total portfolio value
                portfolio_value = cash
                for stock, shares in positions.items():
                    if stock in all_data and current_date in all_data[stock].index:
                        price = all_data[stock].loc[current_date, 'close']
                        portfolio_value += shares * price

                # Rebalance to new weights
                if portfolio_value > 0:
                    for stock, target_weight in new_weights.items():
                        if stock in all_data and current_date in all_data[stock].index:
                            price = all_data[stock].loc[current_date, 'close']
                            target_value = portfolio_value * target_weight

                            # Calculate target shares
                            target_shares = int(target_value / price)
                            current_shares = positions.get(stock, 0)

                            # Execute trade
                            if target_shares != current_shares:
                                shares_diff = target_shares - current_shares
                                trade_value = abs(shares_diff) * price
                                trade_commission = trade_value * self.commission_rate

                                if shares_diff > 0:  # Buy
                                    if cash >= (trade_value + trade_commission):
                                        cash -= (trade_value + trade_commission)
                                        positions[stock] = target_shares
                                else:  # Sell
                                    cash += (trade_value - trade_commission)
                                    positions[stock] = target_shares

                    current_weights = new_weights

            # Update portfolio value
            holdings_value = 0
            for stock, shares in positions.items():
                if stock in all_data and current_date in all_data[stock].index:
                    price = all_data[stock].loc[current_date, 'close']
                    holdings_value += shares * price

            data.loc[current_date, 'cash'] = float(cash)
            data.loc[current_date, 'holdings'] = float(holdings_value)
            data.loc[current_date, 'total'] = float(cash + holdings_value)
            data.loc[current_date, 'num_stocks'] = int(len(positions))

        # Calculate returns
        self._calculate_performance(data)

        self.results[strategy.name] = {
            'data': data,
            'strategy': strategy,
            'positions': positions
        }

        final_value = data['total'].iloc[-1]
        total_return = (final_value - self.initial_cash) / self.initial_cash
        print(f"Portfolio backtest completed: Final value={final_value:.2f}, Total return={total_return:.2%}")

        return data

    def _calculate_performance(self, data):
        """Calculate performance metrics"""
        data['portfolio_value'] = data['total']
        data['strategy_returns'] = data['portfolio_value'].pct_change().fillna(0)
        data['cumulative_returns'] = (1 + data['strategy_returns']).cumprod()

        # Calculate benchmark returns (price itself for single stock, need adjustment for portfolio)
        if 'close' in data.columns:
            data['benchmark_returns'] = data['close'].pct_change().fillna(0)
            data['benchmark_cumulative'] = (1 + data['benchmark_returns']).cumprod()