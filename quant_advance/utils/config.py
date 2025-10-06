# Unified Configuration Management
class Config:
    # Data configuration
    START_DATE = "2018-01-01"  # Shorter period for faster testing
    END_DATE = "2023-12-31"

    # Backtest configuration
    INITIAL_CASH = 1000000  # Initial capital 1 million
    COMMISSION_RATE = 0.0003  # Commission rate 0.03%

    # Stock pool
    BENCHMARK_CODE = "sh.000300"  # CSI 300 as benchmark
    TEST_STOCK_CODE = "sz.000001"  # Ping An Bank as test stock

    # Multi-stock portfolio (example A-shares)
    STOCK_POOL = [
        "sz.000001",  # Ping An Bank
        "sz.000002",  # Vanke A
        "sh.600036",  # China Merchants Bank
        "sh.600519",  # Kweichow Moutai
        "sz.000858",  # Wuliangye
        "sh.601318",  # Ping An Insurance
        "sh.601888",  # China Tourism Group
        "sz.000333",  # Midea Group
        "sh.600276",  # Hengrui Medicine
        "sz.002415"  # Hangzhou Hikvision
    ]


config = Config()