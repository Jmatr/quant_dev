import pandas as pd
import numpy as np
from utils.config import config


class BacktestEngine:
    """Backtest Engine - Enhanced with realistic T+1 execution"""

    def __init__(self, initial_cash=None, commission_rate=None, slippage=0.001):
        self.initial_cash = initial_cash or config.INITIAL_CASH
        self.commission_rate = commission_rate or config.COMMISSION_RATE
        self.slippage = slippage  # 0.1% slippage cost
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
        """Run backtest for single stock strategy with realistic execution"""
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
        print(f"Execution: T+1 with open price, slippage={self.slippage:.1%}")

        # Start from day 1 to ensure we have open price for execution
        for i in range(1, len(data)):
            current_date = data.index[i]
            current_open = data['open'].iloc[i]  # Use open price for execution
            current_signal = signals.iloc[i]  # Signal generated on previous day

            # Apply slippage to execution prices
            buy_price = current_open * (1 + self.slippage)  # Buy at higher price
            sell_price = current_open * (1 - self.slippage)  # Sell at lower price

            # Execute trading signals
            if current_signal == 1 and position == 0:  # Buy signal and no position
                # Calculate buy quantity (considering commission and slippage)
                available_cash = cash * (1 - self.commission_rate)
                new_position = int(available_cash // buy_price)

                if new_position > 0:
                    trade_value = new_position * buy_price
                    trade_commission = trade_value * self.commission_rate

                    position = new_position
                    cash -= (trade_value + trade_commission)

                    if i % 250 == 0:  # Log quarterly
                        print(f"{current_date}: BUY {new_position} shares at {buy_price:.2f} (slippage included)")

            elif current_signal == 0 and position > 0:  # Sell signal and has position
                trade_value = position * sell_price
                trade_commission = trade_value * self.commission_rate
                cash += (trade_value - trade_commission)

                if i % 250 == 0:  # Log quarterly
                    print(f"{current_date}: SELL {position} shares at {sell_price:.2f} (slippage included)")
                position = 0

            # Calculate current holdings value using close price
            current_close = data['close'].iloc[i]
            holdings_value = position * current_close

            # Update current row status
            data.iloc[i, data.columns.get_loc('position')] = float(position)
            data.iloc[i, data.columns.get_loc('cash')] = float(cash)
            data.iloc[i, data.columns.get_loc('holdings')] = float(holdings_value)
            data.iloc[i, data.columns.get_loc('total')] = float(cash + holdings_value)

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
        """Run backtest for portfolio strategy with realistic execution"""
        from core.data_manager import data_manager

        # Get all stock data
        all_data, common_dates = data_manager.get_multiple_stocks_data(strategy.stock_codes)
        portfolio_weights = strategy.portfolio_weights

        # Use first stock's data as base
        base_data = list(all_data.values())[0].copy()
        data = base_data[['open', 'close']].copy()  # Keep both open and close

        # Initialize portfolio tracking
        data['cash'] = self.initial_cash
        data['holdings'] = 0
        data['total'] = self.initial_cash
        data['num_stocks'] = 0

        current_weights = {}
        cash = self.initial_cash
        positions = {}  # stock -> shares

        print(f"Portfolio backtest started: {len(strategy.stock_codes)} stocks")
        print(f"Execution: T+1 with open price, slippage={self.slippage:.1%}")

        for i in range(len(data)):
            current_date = data.index[i]

            # Check if rebalance needed (using previous day's signal)
            if current_date in portfolio_weights:
                new_weights = portfolio_weights[current_date]

                # Sell existing positions not in new portfolio
                stocks_to_sell = [s for s in positions.keys() if s not in new_weights]
                for stock in stocks_to_sell:
                    if stock in all_data and current_date in all_data[stock].index:
                        open_price = all_data[stock].loc[current_date, 'open']
                        sell_price = open_price * (1 - self.slippage)  # Apply slippage
                        trade_value = positions[stock] * sell_price
                        trade_commission = trade_value * self.commission_rate
                        cash += (trade_value - trade_commission)
                        del positions[stock]

                # Calculate total portfolio value (using previous close prices)
                portfolio_value = cash
                for stock, shares in positions.items():
                    if stock in all_data and current_date in all_data[stock].index:
                        # Use previous close for valuation (since we're at open)
                        prev_date = data.index[i - 1] if i > 0 else current_date
                        if prev_date in all_data[stock].index:
                            close_price = all_data[stock].loc[prev_date, 'close']
                            portfolio_value += shares * close_price

                # Rebalance to new weights using open prices
                if portfolio_value > 0:
                    for stock, target_weight in new_weights.items():
                        if stock in all_data and current_date in all_data[stock].index:
                            open_price = all_data[stock].loc[current_date, 'open']
                            buy_price = open_price * (1 + self.slippage)  # Apply slippage
                            target_value = portfolio_value * target_weight

                            # Calculate target shares
                            target_shares = int(target_value / buy_price)
                            current_shares = positions.get(stock, 0)

                            # Execute trade
                            if target_shares != current_shares:
                                shares_diff = target_shares - current_shares
                                if shares_diff > 0:  # Buy
                                    trade_value = shares_diff * buy_price
                                    trade_commission = trade_value * self.commission_rate

                                    if cash >= (trade_value + trade_commission):
                                        cash -= (trade_value + trade_commission)
                                        positions[stock] = target_shares
                                else:  # Sell
                                    trade_value = abs(shares_diff) * buy_price
                                    trade_commission = trade_value * self.commission_rate
                                    cash += (trade_value - trade_commission)
                                    positions[stock] = target_shares

                    current_weights = new_weights

            # Update portfolio value using current close prices
            holdings_value = 0
            for stock, shares in positions.items():
                if stock in all_data and current_date in all_data[stock].index:
                    close_price = all_data[stock].loc[current_date, 'close']
                    holdings_value += shares * close_price

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

        # Calculate benchmark returns (price itself for single stock)
        if 'close' in data.columns:
            data['benchmark_returns'] = data['close'].pct_change().fillna(0)
            data['benchmark_cumulative'] = (1 + data['benchmark_returns']).cumprod()

    def calculate_turnover(self, strategy_name):
        """Calculate portfolio turnover rate"""
        if strategy_name not in self.results:
            return 0

        result = self.results[strategy_name]
        data = result['data']

        if 'position' not in data.columns:
            return 0

        # Calculate turnover as absolute position changes
        position_changes = data['position'].diff().abs()
        avg_position = data['position'].abs().mean()

        if avg_position > 0:
            turnover = position_changes.sum() / (avg_position * len(data)) * 252
            return turnover
        return 0

    def generate_trade_report(self, strategy_name):
        """Generate detailed trade report"""
        if strategy_name not in self.results:
            return None

        result = self.results[strategy_name]
        data = result['data']

        # Identify trade entries and exits
        trades = []
        position = 0
        entry_price = 0
        entry_date = None

        for i in range(1, len(data)):
            current_position = data['position'].iloc[i]
            prev_position = data['position'].iloc[i - 1]

            # Entry signal
            if current_position > 0 and prev_position == 0:
                entry_price = data['open'].iloc[i]  # Entry at open price
                entry_date = data.index[i]

            # Exit signal
            elif current_position == 0 and prev_position > 0:
                exit_price = data['open'].iloc[i]  # Exit at open price
                exit_date = data.index[i]
                pnl = (exit_price - entry_price) / entry_price

                trades.append({
                    'entry_date': entry_date,
                    'exit_date': exit_date,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'return': pnl,
                    'holding_days': (exit_date - entry_date).days
                })

        return trades


# Enhanced configuration with execution parameters
class EnhancedBacktestEngine(BacktestEngine):
    """Enhanced backtest engine with additional features"""

    def __init__(self, initial_cash=None, commission_rate=None, slippage=0.001,
                 execution_lag=1, use_vwap=False):
        super().__init__(initial_cash, commission_rate, slippage)
        self.execution_lag = execution_lag  # T+1, T+2, etc.
        self.use_vwap = use_vwap  # Whether to use VWAP for execution

    def set_market_impact_model(self, impact_model):
        """Set custom market impact model"""
        self.market_impact_model = impact_model

    def calculate_slippage(self, order_size, daily_volume, volatility):
        """Calculate dynamic slippage based on order characteristics"""
        # Basic slippage model: larger orders and higher volatility cause more slippage
        volume_impact = order_size / daily_volume * 0.1  # 10% impact for 100% volume
        volatility_impact = volatility * 0.5  # 50% of daily volatility

        total_slippage = volume_impact + volatility_impact
        return min(total_slippage, 0.05)  # Cap at 5% slippage