import pandas as pd
from core.strategy import Strategy


class BuyAndHoldStrategy(Strategy):
    """Buy and Hold Strategy"""

    def __init__(self):
        super().__init__("Buy and Hold")

    def generate_signals(self):
        """Generate trading signals: buy at beginning, hold forever"""
        signals = pd.Series(0, index=self.data.index)

        # Buy at the first available trading day
        if len(signals) > 0:
            # Use 60 days as safe period to ensure moving averages are calculated
            start_index = min(60, len(signals) - 1)
            signals.iloc[start_index:] = 1

        self.signals = signals
        print(f"Buy and Hold - Total days: {len(signals)}, Holding days: {(signals == 1).sum()}")
        return self