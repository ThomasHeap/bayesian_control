# Bayesian Control

A Python library for implementing Bayesian control systems and decision making under uncertainty.

## Overview

This library provides tools for:
- Bayesian state estimation
- Control system simulation
- Uncertainty-aware decision making
- Visualization of control systems and their performance

## Installation

```bash
pip install -e .
```

## Quick Start

```python
from bayesian_control import Model, Simulation

# Create a control system model
model = Model()

# Run a simulation
sim = Simulation(model)
results = sim.run()

# Visualize results
model.plot_results(results)
```

## Development

This project uses Python 3.8+. To set up the development environment:

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install development dependencies:
```bash
pip install -e ".[dev]"
```

## License

MIT License - see LICENSE file for details
