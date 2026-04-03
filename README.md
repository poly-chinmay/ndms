# Narrative Driven Market Simulator NDMS

I built this project to answer a simple question.

Do narratives actually change how markets behave, or are they just storytelling layered on top of randomness?

## Core Idea

Traditional models assume data leads directly to price.

This project explores a different loop.

narrative to belief to action to price to narrative

Instead of treating market movement as purely statistical, I modeled it as a system where agents interpret narratives differently, beliefs evolve with memory, capital constraints shape decisions, and price emerges from conflicting actions.

## What I Built

The system has four core components.

### 1) Narrative Engine

Generates structured narratives instead of a single sentiment value.

Each narrative contains multiple signals such as macro and technology related features.

Different agents can extract different meanings from the same narrative.

### 2) Agent System

Uses heterogeneous agents including trend following, contrarian, risk averse, and risk seeking profiles.

Each agent has belief with memory, capital and positions, and risk aware decision making.

Agents do not just react, they disagree.

### 3) Market Mechanism

Price is driven by excess demand flow.

The model started with a linear impact form and now includes nonlinear impact scaling.

There are no hardcoded price paths.

Price emerges from behavior.

### 4) Simulation Loop

At each step, a signal source is generated, agents interpret it differently, beliefs update, trades are placed, and price moves.

## Key Experiment

To test whether narratives matter, I ran two systems with all other mechanics held constant.

1) NDMS with narratives

2) Baseline with random shocks and no narrative engine

## Results

### Kurtosis of Returns

NDMS is about 3.66.

Baseline is about 2.15.

This indicates more extreme return events under narrative driven dynamics.

### Volatility Clustering

NDMS absolute return autocorrelation lag 1 is about 0.81.

Baseline absolute return autocorrelation lag 1 is about 0.16.

This indicates stronger volatility persistence in NDMS.

### Behavior

NDMS shows trend buildup, sharp reversals, and regime like transitions.

Baseline is more shock driven and less structured.

## Interpretation

Routing market signals through a narrative layer produces meaningfully different dynamics from raw random shocks.

This does not prove narratives fully explain markets, but it does show narratives are a meaningful modeling layer.

## Limitations

The system is still simplified.

1) Impact is stylized and only partially nonlinear.

2) There is no explicit liquidity depth or full order book microstructure.

3) Outcomes are sensitive to parameter choices.

4) Agent strategies are intentionally minimal.

## Next Steps

1) Further calibrate nonlinear market impact.

2) Improve volatility persistence calibration.

3) Enrich narrative evolution dynamics.

4) Extend to multi asset settings.

## Why This Project Matters

Most projects tune models or reproduce known patterns.

This project asks whether changing how information is represented changes the system itself.

In practice, this means testing narratives versus numbers as distinct drivers of market dynamics.

## Tech

Python and NumPy with an agent based simulation architecture.

## Closing

This project is less about point prediction and more about understanding how information and belief shape markets.

If you are working on agent based systems, market microstructure, or AI and economics, I would love to connect.
