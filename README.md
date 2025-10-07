# A-Share Quantitative Trading Research Platform

[![Python 3.8+](https://img.shields.io/badge/Python-3.8%252B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](./LICENSE)
![Quant Trading](https://img.shields.io/badge/Quant-Trading-orange)

A complete quantitative trading research framework for A-shares, implementing everything from basic backtesting to advanced machine learning strategies.

---

## Project Overview
This project builds a complete quantitative trading system through three progressive stages:

- **Phase 1:** Basic framework and simple strategies  
- **Phase 2:** Multi-factor models and portfolio optimization  
- **Phase 3:** Statistical arbitrage and machine learning strategies

---

## Core Features
### Strategy System
- **Traditional Strategies:** Buy & Hold, Moving Average Crossover
- **Multi-Factor Strategies:** Momentum, Volume, Volatility factor combinations
- **Statistical Arbitrage:** Pair Trading, Mean Reversion
- **Machine Learning:** Random Forest, Ensemble Learning prediction models

### Professional Features
- **Realistic Backtesting:** T+1 mechanism, slippage costs, commission fees
- **Risk Management:** Max drawdown, Sharpe ratio, Calmar ratio
- **Performance Analysis:** Multi-dimensional strategy evaluation metrics
- **Modular Design:** Easy to extend with new strategies

---

## Quick Start
### Environment Requirements
- Python **3.8+**
- Dependencies listed in `requirements.txt`

### Installation Steps
**Clone the project**
```bash
git clone https://github.com/yourusername/a-share-quant-platform.git
cd a-share-quant-platform
```

**Create virtual environment**
```bash
python -m venv .venv
# Linux/Mac
source .venv/bin/activate
# Windows
.venv\Scripts\activate
```

**Install dependencies**
```bash
pip install -r requirements.txt
```

**Run example**
```bash
python main.py
```

---

## Project Structure
```text
quant_project/
│
├── core/                         # Core engines
│   ├── data_manager.py             # Data management
│   ├── backtest.py                 # Backtest engine
│   ├── strategy.py                 # Strategy base class
│   ├── factor_calculator.py        # Factor calculator
│   ├── portfolio_optimizer.py      # Portfolio optimizer
│   ├── statistical_arbitrage.py    # Statistical arbitrage
│   └── feature_engineering.py      # Feature engineering
│
├── strategies/                   # Strategy collection
│   ├── buy_and_hold.py             # Buy and hold
│   ├── moving_average_cross.py     # Moving average crossover
│   ├── multi_factor_strategy.py    # Multi-factor strategy
│   ├── pair_trading_strategy.py    # Pair trading
│   └── ml_strategy.py              # Machine learning
│
├── utils/                        # Utilities
│   ├── config.py                    # Configuration management
│   └── logger.py                    # Logging configuration
│
├── figs/                         # Saved figures (e.g., result.png)
├── outputs/                      # Backtest outputs
├── data/                         # Raw/processed data (gitignored)
├── main.py                       # Main program
├── requirements.txt              # Dependencies
└── README.md                     # Project documentation
```

---

## Usage Examples
**Basic Usage**
```python
from core.data_manager import data_manager
from core.backtest import BacktestEngine
from strategies.multi_factor_strategy import MultiFactorStrategy

# Get data
data = data_manager.get_stock_data("sz.000001")

# Create strategy
strategy = MultiFactorStrategy().set_data(data)

# Run backtest
engine = BacktestEngine()
results = engine.run_backtest(strategy)
```

**Custom Strategy**
```python
from core.strategy import Strategy
import pandas as pd

class MyCustomStrategy(Strategy):
    def __init__(self):
        super().__init__("Custom Strategy")
    
    def generate_signals(self):
        # Implement your trading logic
        signals = pd.Series(0, index=self.data.index)
        # ... your strategy code
        self.signals = signals
        return self
```

---

---

## Configuration
### Stock Pool Configuration
Modify in `utils/config.py`:
```python
STOCK_POOL = [
    "sz.000001",  # Ping An Bank
    "sz.000002",  # Vanke A
    "sh.600036",  # China Merchants Bank
    "sh.600519",  # Kweichow Moutai
    # ... add more stocks
]
```

### Backtest Parameters
```python
# Initial capital
INITIAL_CASH = 1000000

# Trading costs
COMMISSION_RATE = 0.0003  # 0.03%
SLIPPAGE = 0.001          # 0.1% slippage

# Time period
START_DATE = "2018-01-01"
END_DATE = "2023-12-31"
```

---

## Learning Path
**Phase 1: Basic Framework**
- Understand data acquisition and backtest engine
- Implement simple technical indicator strategies
- Learn basic performance analysis

**Phase 2: Multi-Factor Models**
- Learn factor investment theory
- Implement portfolio optimization algorithms
- Understand risk management concepts

**Phase 3: Advanced Strategies**
- Master statistical arbitrage principles
- Learn machine learning in quantitative trading
- Implement complex multi-strategy systems

---

## Contributing
We welcome contributions of all kinds!

1. Fork the project  
2. Create feature branch (`git checkout -b feature/AmazingFeature`)  
3. Commit changes (`git commit -m 'Add some AmazingFeature'`)  
4. Push to branch (`git push origin feature/AmazingFeature`)  
5. Open Pull Request

---

## Changelog
**v1.0.0 (2024-01-XX)**
- Complete three-phase quantitative framework
- Implement multiple trading strategies
- Professional backtesting and performance analysis
- Complete documentation and examples

---

## Risk Disclaimer
- This project is for research and educational purposes only
- Backtest results do not guarantee future performance
- Live trading involves risks; trade cautiously

---

## License
This project is licensed under the **MIT License** — see the [LICENSE](./LICENSE) file for details.

---

## Acknowledgments
- Thanks to Baostock for A-share data
- Inspired by open-source quantitative projects and community contributions

---

## Contact
For questions or suggestions:
- Submit an **Issue**
- Email: **your-email@example.com**

If this project helps you, please give it a ⭐️!

<div align="center">
Learn quantitative trading from scratch, make investing more scientific!
</div>