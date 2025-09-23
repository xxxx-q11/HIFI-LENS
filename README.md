
# HIFI-LENS: High-Fidelity Large-Scale Market Simulation Framework

> Stock markets are complex adaptive systems whose dynamics emerge from decentralized interactions among diverse participants responding to an ever-shifting informational landscape. Despite decades of progress, existing simulation frameworks remain constrained by either low behavioral fidelity, due to non-intelligent agents, or low structural fidelity, due to oversimplified trading mechanisms. To bridge this gap, we introduce HIFI-LENS, a large-scale simulation framework that integrates structural realism with generative agent intelligence. HIFI-LENS captures market complexity across three interconnected levels: (i) at the micro-level, it simulates over 15k heterogeneous agents, featuring a novel hierarchical architecture for LLM-powered institutional investors; (ii) at the meso-level, these agents interact in a nanosecond-resolution, NASDAQ-like continuous double auction market; and (iii) at the macro-level, their behavior is grounded in a rich information stream of both endogenous market data and over 12k real-world news articles and reports. We validate HIFI-LENS across eight GICS sectors and three representative real-world scenarios, showing that it not only reproduces five key stylized facts, but also accurately tracks real-world high-frequency price dynamics with an average MAPE of 3.48%. Our work paves the way for a new generation of market simulation, enabling high-fidelity studies of emergent phenomena, policy impacts, and AI-in-the-market dynamics.

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- pip package manager

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/zhentao-liu-JLU/HIFI_LENS.git
   cd HIFI_LENS
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure LLM API keys:**
   Edit `config_LLM.py` and add your API keys:
   ```python
   apikeys['deepseek_api_key']="your_api_key_here"
   ```

4. **Run the simulation:**
   ```bash
   python -u abides.py -c rsmtry_LLM3 -t JNJ -d 20250402 -s 1234 -l rmsctry_LLM3 --enable-llm-cache
   ```

## 📋 Command Line Arguments

The main simulation script accepts the following arguments:

| Argument | Description | Example |
|----------|-------------|---------|
| `-c, --config` | Configuration file name | `rsmtry_LLM3` |
| `-t, --ticker` | Stock symbol to simulate | `JNJ` |
| `-d, --historical-date` | Trading date (YYYYMMDD) | `20250402` |
| `-s, --seed` | Random seed for reproducibility | `1234` |
| `-l, --log_dir` | Log directory name | `rmsctry_LLM3` |
| `--enable-llm-cache` | Enable LLM result caching | Flag |
| `--start-time` | Simulation start time | `09:30:00` (default) |
| `--end-time` | Simulation end time | `16:00:00` (default) |

## ��️ Architecture Overview

### Agent Types

HIFI-LENS simulates multiple types of trading agents:

- **LLM Agents(Manager Agents)**: Intelligent institutional investors powered by Large Language Models
- **Noise Agents**: 12,000 agents representing retail traders with random behavior
- **Value Agents**: 100 agents that trade based on fundamental value analysis
- **Trade Agents**: 2,950 agents guided by Manager Agents
- **Market Maker Agents**: Adaptive market makers providing liquidity
- **Momentum Agents**: 50 agents that follow price momentum trends

### Market Infrastructure

- **Exchange Agent**: NASDAQ-like continuous double auction market
- **Order Book**: Nanosecond-resolution order matching
- **Oracle**: LLM-powered data provider

## 🔧 Configuration

### LLM Configuration (`config_LLM.py`)

```python
apikeys = {}
apikeys['deepseek_api_key'] = "your_api_key_here"
apikeys['think_model_name'] = "deepseek-reasoner"
apikeys['generate_model_name'] = "deepseek-chat"

base_agentconfig = {
    "company": "JNJ",
    "event": "Trump"
}
```

### Simulation Parameters

Key simulation parameters can be adjusted in the configuration files:

- **Market Hours**: 09:30:00 - 16:00:00 (configurable)
- **Starting Cash**: $10,000,000 per agent
- **Market Maker Parameters**: POV, spread settings, wake-up frequency

## 📊 Output and Analysis

The simulation generates comprehensive market data including:

- **Order Book Data**: Bid/ask prices and volumes
- **Trade Data**: Transaction records with timestamps
- **Agent Logs**: Individual agent trading behavior
- **Market Statistics**: Volume, volatility, and price dynamics

### Visualization Tools

- `real_time_visualization.py`: Real-time market visualization
- `plot_trade_price.py`: Price movement analysis
- `plot_trade_Comparison.py`: Comparative analysis tools

## 🧪 Experimental Features

### LLM Cache System

Enable caching to improve performance and reduce API costs:
```bash
python -u abides.py -c rsmtry_LLM3 -t JNJ -d 20250402 -s 1234 -l rmsctry_LLM3 --enable-llm-cache
```
If the llm_cache.json file does not exist, create the file and save the return result and its status after each invocation of the Manager Agents for breakpoint recovery
If the file already exists, the previous experiment can be reproduced
### Multiple Trading Days

The system supports simulating multiple consecutive trading days:
```python
trading_days = ["2025-04-02", "2025-04-03", "2025-04-04"]
```

## �� Project Structure


