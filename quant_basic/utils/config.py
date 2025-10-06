# Unified Configuration Management
class Config:
    # Data configuration
    START_DATE = "2010-01-01"
    END_DATE = "2023-12-31"

    # Backtest configuration
    INITIAL_CASH = 1000000  # Initial capital 1 million
    COMMISSION_RATE = 0.0003  # Commission rate 0.03%

    # Stock pool
    BENCHMARK_CODE = "sh.000300"  # CSI 300 as benchmark
    TEST_STOCK_CODE = "sz.000001"  # Ping An Bank as test stock


config = Config()