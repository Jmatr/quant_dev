import baostock as bs
import pandas as pd
import atexit
from utils.config import config
import time


class DataManager:
    def __init__(self):
        self.lg = None
        self._connect()
        # Register exit handler
        atexit.register(self._safe_disconnect)

    def _connect(self):
        """Connect to Baostock"""
        try:
            # First try to logout any existing connection
            try:
                bs.logout()
            except:
                pass

            self.lg = bs.login()
            if self.lg.error_code != '0':
                raise Exception(f"Login failed: {self.lg.error_msg}")
            print("Baostock login successful")
        except Exception as e:
            print(f"Login exception: {e}")
            # Retry once
            time.sleep(1)
            self.lg = bs.login()
            if self.lg.error_code != '0':
                raise Exception(f"Retry login failed: {self.lg.error_msg}")
            print("Baostock retry login successful")

    def _safe_disconnect(self):
        """Safely disconnect"""
        try:
            bs.logout()
            print("Baostock safely logged out")
        except Exception as e:
            print(f"Error during logout: {e}")

    def get_stock_data(self, code, start_date=None, end_date=None):
        """Get stock daily data"""
        start_date = start_date or config.START_DATE
        end_date = end_date or config.END_DATE

        print(f"Downloading {code} data ({start_date} to {end_date})...")

        # Get adjusted data directly
        rs = bs.query_history_k_data_plus(
            code,
            "date,code,open,high,low,close,volume,amount,turn",
            start_date=start_date,
            end_date=end_date,
            frequency="d",
            adjustflag="3"  # Adjusted close
        )

        if rs.error_code != '0':
            raise Exception(f"Data retrieval failed: {rs.error_msg}")

        data_list = []
        while (rs.error_code == '0') & rs.next():
            data_list.append(rs.get_row_data())

        if not data_list:
            raise Exception(f"No data retrieved for {code}")

        df = pd.DataFrame(data_list, columns=rs.fields)

        # Data type conversion
        numeric_columns = ['open', 'high', 'low', 'close', 'volume', 'amount', 'turn']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
        df = df.sort_index()

        # Clean invalid data
        df = df.dropna(subset=['close'])

        print(f"Download completed: {len(df)} records, time range: {df.index[0]} to {df.index[-1]}")
        print(f"Data columns: {list(df.columns)}")

        return df

    def get_benchmark_data(self):
        """Get benchmark data"""
        return self.get_stock_data(config.BENCHMARK_CODE)


# Singleton pattern
data_manager = DataManager()