import pandas as pd
import numpy as np
from utils.config import config


class BacktestEngine:
    """Backtest Engine"""

    def __init__(self, initial_cash=None, commission_rate=None):
        self.initial_cash = initial_cash or config.INITIAL_CASH
        self.commission_rate = commission_rate or config.COMMISSION_RATE
        self.results = {}

    def run_backtest(self, strategy):
        """Run backtest"""
        if strategy.signals is None:
            strategy.prepare_data().generate_signals()

        data = strategy.data.copy()
        signals = strategy.signals.copy()

        # Initialize backtest data columns
        data['position'] = 0  # Position size
        data['cash'] = self.initial_cash  # Cash
        data['holdings'] = 0  # Holdings value
        data['total'] = self.initial_cash  # Total assets
        data['returns'] = 0.0  # Daily returns

        position = 0
        cash = self.initial_cash

        print(f"Backtest started: data length={len(data)}, initial capital={self.initial_cash}")

        for i in range(len(data)):
            current_date = data.index[i]
            current_price = data['close'].iloc[i]
            current_signal = signals.iloc[i]

            # Execute trading signals
            if current_signal == 1 and position == 0:  # Buy signal and no position
                # Calculate buy quantity (considering commission)
                available_cash = cash * (1 - self.commission_rate)
                new_position = int(available_cash // current_price)

                if new_position > 0:
                    trade_value = new_position * current_price
                    trade_commission = trade_value * self.commission_rate

                    position = new_position
                    cash -= (trade_value + trade_commission)
                    print(f"{current_date}: BUY {new_position} shares at {current_price:.2f}")

            elif current_signal == 0 and position > 0:  # Sell signal and has position
                trade_value = position * current_price
                trade_commission = trade_value * self.commission_rate
                cash += (trade_value - trade_commission)
                print(f"{current_date}: SELL {position} shares at {current_price:.2f}")
                position = 0

            # Update current row status
            data.iloc[i, data.columns.get_loc('position')] = position
            data.iloc[i, data.columns.get_loc('cash')] = cash
            data.iloc[i, data.columns.get_loc('holdings')] = position * current_price
            data.iloc[i, data.columns.get_loc('total')] = cash + position * current_price

        # Calculate returns
        data['portfolio_value'] = data['total']
        data['strategy_returns'] = data['portfolio_value'].pct_change().fillna(0)
        data['cumulative_returns'] = (1 + data['strategy_returns']).cumprod()

        # Calculate benchmark returns (price itself)
        data['benchmark_returns'] = data['close'].pct_change().fillna(0)
        data['benchmark_cumulative'] = (1 + data['benchmark_returns']).cumprod()

        self.results[strategy.name] = {
            'data': data,
            'strategy': strategy
        }

        final_value = data['total'].iloc[-1]
        total_return = (final_value - self.initial_cash) / self.initial_cash
        print(f"Backtest completed: Final value={final_value:.2f}, Total return={total_return:.2%}")

        return data