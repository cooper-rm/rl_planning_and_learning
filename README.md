# Lab 7: Planning and Learning Integration

**Morgan Cooper -- MSDS 684, Reinforcement Learning**

## Overview

This lab implements Dyna-Q on Gymnasium's Taxi-v3 to study how integrating a learned model with direct RL changes sample efficiency. The agent learns from real experience like Q-learning, but it also stores every transition in a Python dictionary and uses that model to run extra planning updates between real steps. The lab also extends Dyna-Q with the κ√τ exploration bonus (Dyna-Q+) for non-stationary environments and prioritized sweeping (heapq priority queue) for focused planning.

## Structure

- `Cooper_Morgan_Lab7.ipynb` -- Main notebook with all implementation and experiments
- `generate_report.py` -- Generates the PDF report from LaTeX
- `Cooper_Morgan_Lab7.pdf` -- Final report
- `figures/` -- Saved visualizations used in the report
- `requirements.txt` -- Python dependencies

## Key Components

- **Dyna-Q (`DynaQAgent`)**: NumPy Q-table for direct RL, Python dict `model[(s,a)] = (r, s')` updated after every real step, planning loop sampling n pairs from `model.keys()`.
- **Dyna-Q+ (`DynaQPlusAgent`)**: Adds the κ√τ exploration bonus on planning rewards using a `time_since[(s,a)]` dict.
- **Prioritized sweeping (`PrioritizedSweepingAgent`)**: heapq-backed priority queue keyed on TD-error magnitude, with predecessor propagation.
- **`ChangingTaxi` wrapper**: Adds a -10 penalty to row 0 transitions after step 1000 to make the learned model wrong on purpose.

## Results

| Comparison | Setting | Result |
|---|---|---|
| n-sweep | n=0 (pure Q-learning) | -33,684 cum reward, 0/10 seeds solved |
| n-sweep | n=5 | -29,789, 0/10 solved |
| n-sweep | n=10 | -26,527, 2/10 solved |
| n-sweep | n=50 | -16,247, 10/10 solved (median 215 episodes) |
| ChangingTaxi | Dyna-Q+ post-change rate | -1.93/step (16% better than Dyna-Q) |
| Planning order | Prioritized vs uniform (n=10) | -15,293 vs -16,443 cum reward |

## Usage

```bash
# Install dependencies
pip install -r requirements.txt

# Run the notebook end-to-end
jupyter notebook Cooper_Morgan_Lab7.ipynb

# Generate the PDF report
python generate_report.py
```

## References

- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press. Chapter 8.
