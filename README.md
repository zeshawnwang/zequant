# zequant

Personal stock quantitative trading system.

## Features

- **Data Layer**: Incremental data fetching via AKShare/Tushare, DuckDB storage
- **Factor Layer**: Polars-powered factor computation, IC/IR analysis, factor registry
- **Strategy Layer**: Modular selector + timing + portfolio architecture
- **Backtest Layer**: Backtrader-powered backtesting with full fee simulation
- **Live Layer**: Simulated trading with real-time monitoring

## Quick Start

```bash
pip install -r requirements.txt

# Initialize database
python scripts/init_db.py

# Fetch data
python scripts/fetch_data.py

# Compute factors
python scripts/compute_factors.py

# Run backtest
python scripts/run_backtest.py
```

## Project Structure

```
config/          - YAML configuration
core/            - Core modules (DB, factors, strategies, backtest, broker, risk)
factors/         - Factor implementations
selectors/       - Stock selectors
timings/         - Timing generators
portfolios/      - Portfolio builders
strategies/      - Strategy instances
scripts/         - Operational scripts
tests/           - Unit tests
```

## Configuration

Edit `config/config.yaml` to customize:
- Data source and fetch schedule
- Fee rates (stamp tax, commission, slippage)
- Risk parameters
- Factor definitions
- Strategy parameters
