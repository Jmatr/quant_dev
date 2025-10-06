import baostock as bs
import pandas as pd
import atexit
import numpy as np
from utils.config import config
import time


class DataManager:
    def __init__(self):
        self.lg = None
        self._connect()
        atexit.register(self._safe_disconnect)
        self._stock_pool = {}  # 缓存股票数据

    def _connect(self):
        """Connect to Baostock"""
        try:
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

    def get_stock_data(self, code, start_date=None, end_date=None, use_cache=True):
        """Get stock daily data"""
        if use_cache and code in self._stock_pool:
            return self._stock_pool[code].copy()

        start_date = start_date or config.START_DATE
        end_date = end_date or config.END_DATE

        print(f"Downloading {code} data ({start_date} to {end_date})...")

        rs = bs.query_history_k_data_plus(
            code,
            "date,code,open,high,low,close,volume,amount,turn,pctChg",
            start_date=start_date,
            end_date=end_date,
            frequency="d",
            adjustflag="3"
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
        numeric_columns = ['open', 'high', 'low', 'close', 'volume', 'amount', 'turn', 'pctChg']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
        df = df.sort_index()

        # Clean invalid data
        df = df.dropna(subset=['close'])

        print(f"Download completed: {len(df)} records")

        if use_cache:
            self._stock_pool[code] = df

        return df

    def get_multiple_stocks_data(self, codes, start_date=None, end_date=None):
        """Get multiple stocks data and align dates"""
        all_data = {}
        common_dates = None

        for code in codes:
            try:
                data = self.get_stock_data(code, start_date, end_date)
                all_data[code] = data

                if common_dates is None:
                    common_dates = set(data.index)
                else:
                    common_dates = common_dates.intersection(set(data.index))
            except Exception as e:
                print(f"Error getting data for {code}: {e}")

        # Filter to common dates only
        common_dates = sorted(common_dates)
        for code in all_data:
            all_data[code] = all_data[code].loc[common_dates]

        return all_data, common_dates

    def get_benchmark_data(self):
        """Get benchmark data"""
        return self.get_stock_data(config.BENCHMARK_CODE)


# Singleton pattern
data_manager = DataManager()