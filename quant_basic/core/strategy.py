from abc import ABC, abstractmethod
import pandas as pd


class Strategy(ABC):
    """策略抽象基类"""

    def __init__(self, name, initial_cash=1000000):
        self.name = name
        self.initial_cash = initial_cash
        self.data = None
        self.signals = None

    def set_data(self, data):
        """设置数据"""
        self.data = data.copy()
        return self

    @abstractmethod
    def generate_signals(self):
        """生成交易信号 - 子类必须实现"""
        pass

    def prepare_data(self):
        """数据预处理 - 子类可重写"""
        if self.data is not None:
            self.data['returns'] = self.data['close'].pct_change()
        return self