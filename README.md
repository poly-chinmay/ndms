# Narrative-Driven Market Simulator (NDMS)

NDMS is a minimal agent-based market simulator where price emerges from conflicting agent trades under capital constraints.

## Core Idea

- Narratives are structured multi-signal objects (not scalar sentiment).
- Agents interpret the same narrative differently.
- Agents have finite cash, positions, and P&L dynamics.
- Price updates from net order flow:
	- $\Delta P = \lambda \times \text{net\_flow}$

## Repository Structure

ndms/
├── agent_engine/
├── narrative_engine/
├── market_engine/
├── simulation/
├── evaluation/
├── experiments/
├── main.py
└── README.md

## Setup

Use Python 3.10+.

Install dependency:

- numpy

Example:

```bash
python3 -m pip install numpy
```

## Run Simulation

Primary runner:

```bash
python3 simulation/runner.py
```

This prints per-step:

- Step
- Price
- Net Flow

and end-of-run metrics:

- kurtosis of log returns
- autocorrelation of returns
- autocorrelation of absolute returns

## Baseline vs Narrative Comparison

The runner includes both modes:

- `narrative` (NDMS)
- `random_shock` (baseline)

Compare directly via:

```python
from simulation.runner import compare_ndms_vs_baseline
compare_ndms_vs_baseline(steps=60, sim_seed=7)
```

## Notes

- The design is intentionally minimal and interpretable.
- Forced liquidation is included to produce endogenous crash-like flow spikes.
