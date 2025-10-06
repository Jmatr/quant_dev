import pandas as pd
from core.strategy import Strategy


class MovingAverageCrossStrategy(Strategy):
    """Moving Average Crossover Strategy"""

    def __init__(self, short_window=20, long_window=60):
        super().__init__(f"MA Crossover({short_window}/{long_window})")
        self.short_window = short_window
        self.long_window = long_window

    def generate_signals(self):
        """Generate trading signals"""
        if self.data is None or len(self.data) < self.long_window:
            raise ValueError("Insufficient data for moving average calculation")

        # Calculate moving averages
        self.data['short_ma'] = self.data['close'].rolling(window=self.short_window, min_periods=1).mean()
        self.data['long_ma'] = self.data['close'].rolling(window=self.long_window, min_periods=1).mean()

        # Generate signals: buy when short MA crosses above long MA, sell when crosses below
        signals = pd.Series(0, index=self.data.index)

        # Calculate golden cross and death cross
        golden_cross = (self.data['short_ma'] > self.data['long_ma']) & \
                       (self.data['short_ma'].shift(1) <= self.data['long_ma'].shift(1))

        death_cross = (self.data['short_ma'] < self.data['long_ma']) & \
                      (self.data['short_ma'].shift(1) >= self.data['long_ma'].shift(1))

        # Set signals
        current_signal = 0
        for i in range(len(signals)):
            if i < self.long_window:  # No trading in first long_window days
                signals.iloc[i] = 0
                continue

            if golden_cross.iloc[i]:
                current_signal = 1
            elif death_cross.iloc[i]:
                current_signal = 0

            signals.iloc[i] = current_signal

        # Shift signals forward by one day (avoid look-ahead bias)
        signals = signals.shift(1).fillna(0)

        self.signals = signals
        print(
            f"Signal Stats - Total days: {len(signals)}, Long positions: {(signals == 1).sum()}, Cash positions: {(signals == 0).sum()}")
        return self