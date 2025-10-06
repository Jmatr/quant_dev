\# A-Share Quantitative Trading Research Platform



\[!\[Python 3.8+](https://img.shields.io/badge/Python-3.8%252B-blue)](https://www.python.org/)

\[!\[License: MIT](https://img.shields.io/badge/License-MIT-green)](./LICENSE)

!\[Quant Trading](https://img.shields.io/badge/Quant-Trading-orange)



A complete quantitative trading research framework for A-shares, implementing everything from basic backtesting to advanced machine learning strategies.



---



\## Project Overview

This project builds a complete quantitative trading system through three progressive stages:



\- \*\*Phase 1:\*\* Basic framework and simple strategies  

\- \*\*Phase 2:\*\* Multi-factor models and portfolio optimization  

\- \*\*Phase 3:\*\* Statistical arbitrage and machine learning strategies



---



\## Core Features

\### Strategy System

\- \*\*Traditional Strategies:\*\* Buy \& Hold, Moving Average Crossover

\- \*\*Multi-Factor Strategies:\*\* Momentum, Volume, Volatility factor combinations

\- \*\*Statistical Arbitrage:\*\* Pair Trading, Mean Reversion

\- \*\*Machine Learning:\*\* Random Forest, Ensemble Learning prediction models



\### Professional Features

\- \*\*Realistic Backtesting:\*\* T+1 mechanism, slippage costs, commission fees

\- \*\*Risk Management:\*\* Max drawdown, Sharpe ratio, Calmar ratio

\- \*\*Performance Analysis:\*\* Multi-dimensional strategy evaluation metrics

\- \*\*Modular Design:\*\* Easy to extend with new strategies



---



\## Quick Start

\### Environment Requirements

\- Python \*\*3.8+\*\*

\- Dependencies listed in `requirements.txt`



\### Installation Steps

\*\*Clone the project\*\*

```bash

git clone https://github.com/yourusername/a-share-quant-platform.git

cd a-share-quant-platform

```



\*\*Create virtual environment\*\*

```bash

python -m venv .venv

\# Linux/Mac

source .venv/bin/activate

\# Windows

.venv\\Scripts\\activate

```



\*\*Install dependencies\*\*

```bash

pip install -r requirements.txt

```



\*\*Run example\*\*

```bash

python main.py

```



---



\## Project Structure

```text

quant\_project/

│

├── core/                         # Core engines

│   ├── data\_manager.py             # Data management

│   ├── backtest.py                 # Backtest engine

│   ├── strategy.py                 # Strategy base class

│   ├── factor\_calculator.py        # Factor calculator

│   ├── portfolio\_optimizer.py      # Portfolio optimizer

│   ├── statistical\_arbitrage.py    # Statistical arbitrage

│   └── feature\_engineering.py      # Feature engineering

│

├── strategies/                   # Strategy collection

│   ├── buy\_and\_hold.py             # Buy and hold

│   ├── moving\_average\_cross.py     # Moving average crossover

│   ├── multi\_factor\_strategy.py    # Multi-factor strategy

│   ├── pair\_trading\_strategy.py    # Pair trading

│   └── ml\_strategy.py              # Machine learning

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



\## Usage Examples

\*\*Basic Usage\*\*

```python

from core.data\_manager import data\_manager

from core.backtest import BacktestEngine

from strategies.multi\_factor\_strategy import MultiFactorStrategy



\# Get data

data = data\_manager.get\_stock\_data("sz.000001")



\# Create strategy

strategy = MultiFactorStrategy().set\_data(data)



\# Run backtest

engine = BacktestEngine()

results = engine.run\_backtest(strategy)

```



\*\*Custom Strategy\*\*

```python

from core.strategy import Strategy

import pandas as pd



class MyCustomStrategy(Strategy):

&nbsp;   def \_\_init\_\_(self):

&nbsp;       super().\_\_init\_\_("Custom Strategy")

&nbsp;   

&nbsp;   def generate\_signals(self):

&nbsp;       # Implement your trading logic

&nbsp;       signals = pd.Series(0, index=self.data.index)

&nbsp;       # ... your strategy code

&nbsp;       self.signals = signals

&nbsp;       return self

```



---



\## Strategy Performance

\_Backtest results on 2018–2023 A-share data:\_



| Strategy          | Total Return | Annual Return | Sharpe Ratio | Max Drawdown |

|-------------------|-------------:|--------------:|-------------:|-------------:|

| Buy \& Hold        |      -11.53% |        -2.10% |        -0.07 |       -63.89%|

| MA Crossover      |       16.48% |         2.67% |         0.12 |       -35.37%|

| Multi-Factor      |       12.40% |         2.04% |         0.07 |       -55.85%|

| Pair Trading      |      257.48% |        24.65% |         2.59 |        -1.72%|

| ML Random Forest  |     1642.98% |        63.94% |         3.31 |       -10.10%|



> \*\*Note:\*\* Results are illustrative; performance varies by configuration, universe, and period.



---



\## Configuration

\### Stock Pool Configuration

Modify in `utils/config.py`:

```python

STOCK\_POOL = \[

&nbsp;   "sz.000001",  # Ping An Bank

&nbsp;   "sz.000002",  # Vanke A

&nbsp;   "sh.600036",  # China Merchants Bank

&nbsp;   "sh.600519",  # Kweichow Moutai

&nbsp;   # ... add more stocks

]

```



\### Backtest Parameters

```python

\# Initial capital

INITIAL\_CASH = 1000000



\# Trading costs

COMMISSION\_RATE = 0.0003  # 0.03%

SLIPPAGE = 0.001          # 0.1% slippage



\# Time period

START\_DATE = "2018-01-01"

END\_DATE = "2023-12-31"

```



---



\## Learning Path

\*\*Phase 1: Basic Framework\*\*

\- Understand data acquisition and backtest engine

\- Implement simple technical indicator strategies

\- Learn basic performance analysis



\*\*Phase 2: Multi-Factor Models\*\*

\- Learn factor investment theory

\- Implement portfolio optimization algorithms

\- Understand risk management concepts



\*\*Phase 3: Advanced Strategies\*\*

\- Master statistical arbitrage principles

\- Learn machine learning in quantitative trading

\- Implement complex multi-strategy systems



---



\## Contributing

We welcome contributions of all kinds!



1\. Fork the project  

2\. Create feature branch (`git checkout -b feature/AmazingFeature`)  

3\. Commit changes (`git commit -m 'Add some AmazingFeature'`)  

4\. Push to branch (`git push origin feature/AmazingFeature`)  

5\. Open Pull Request



---



\## Changelog

\*\*v1.0.0 (2024-01-XX)\*\*

\- Complete three-phase quantitative framework

\- Implement multiple trading strategies

\- Professional backtesting and performance analysis

\- Complete documentation and examples



---



\## Risk Disclaimer

\- This project is for research and educational purposes only

\- Backtest results do not guarantee future performance

\- Live trading involves risks; trade cautiously



---



\## License

This project is licensed under the \*\*MIT License\*\* — see the \[LICENSE](./LICENSE) file for details.



---



\## Acknowledgments

\- Thanks to Baostock for A-share data

\- Inspired by open-source quantitative projects and community contributions



---



\## Contact

For questions or suggestions:

\- Submit an \*\*Issue\*\*

\- Email: \*\*shengliu.bd@gmail.com.com\*\*



If this project helps you, please give it a ⭐️!



<div align="center">

Learn quantitative trading from scratch, make investing more scientific!

</div>

