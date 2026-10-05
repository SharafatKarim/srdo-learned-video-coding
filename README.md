# Code Release: Semantic Bit Allocation and Control in Learned Video Coding

This repository contains the official code release to reproduce the experiments, tables, and figures from our manuscript on Semantic Rate-Distortion Optimization (SRDO) and Dual-Ascent Control in learned video compression.

## Overview

The codebase is built on [CompressAI](https://github.com/InterDigitalInc/CompressAI) and PyTorch. It provides the full pipeline used in the paper, including:
- **Semantic Weighting:** The $(1 + \mu\Omega_i)$ spatial weighting mechanism applied to the Scale-Space Flow codec.
- **Dual-Ascent Controller:** The controller that dynamically adjusts the Lagrange multiplier $\mu$ to satisfy a downstream task-utility constraint.
- **Experimental Runners:** Standalone scripts for 5-fold cross-validation and dual-ascent controller trajectories.
- **Figure Generation:** Scripts to reproduce the figures used in the manuscript.

## Prerequisites & Installation

We recommend using [`uv`](https://github.com/astral-sh/uv) to manage the Python environment, as it guarantees deterministic installations. 

1. Install `uv` if you haven't already:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
2. In the root of this release directory, sync the environment:
   ```bash
   uv sync
   ```
   *Note: All commands below assume you are running them via `uv run`.*

## Directory Structure

- `src/srdo/`: Core library code.
  - `loss.py`: Implements the SRDO loss (Eq. 4).
  - `omega.py`: Constructs the semantic prior $\Omega$.
  - `dual.py`: Implements the outer-loop dual-ascent controller.
  - `bdrate.py`: Bjontegaard delta-rate (BD-Rate) computation.
- `scripts/`: Evaluation, parsing, and ablation scripts.
  - `generate_journal_figures.py`: Rebuilds the manuscript's PDFs/PNGs.
- `scripts/kaggle/`: Self-contained heavy-compute scripts used for the core results.

## Reproducing Core Results

The primary quantitative claims in the paper come from the standalone multi-camera (Kaggle) experiments.

### 1. 5-Fold Cross-Validation (Section 6.5)

To reproduce the rate-distortion operating point shift (50.6% BD-Rate saving at matched constraint settings), run the CV script. This script implements a strict leave-one-scene-out cross-validation across 5 scenes.

```bash
uv run python scripts/kaggle/run_kaggle_standalone_all_in_one.py
```
*Note: Depending on your hardware, running all 5 folds with both $\mu=0$ and $\mu=60$ configurations can take multiple GPU-hours.*

### 2. Dual-Ascent Controller Mechanics (Section 6.3 & 7)

To reproduce the controller's two-phase behavior (approach from above and below) and observe the identifiability/drift limits discussed in the Conclusion:

```bash
uv run python scripts/kaggle/run_continuous_dual_ascent.py
```
This runs the dual controller against the validation pool and generates the trajectory CSVs (`continuous_dual_ascent_mu0_5.csv` and `continuous_dual_ascent_mu0_30.csv`) which track task utility and the multiplier $\mu$ over time.

### 3. Generating Figures

Once the experiments have been run and their CSV outputs are populated in the `outputs/` folder, you can rebuild all manuscript figures:

```bash
uv run python scripts/generate_journal_figures.py
```
The figures will be saved in `outputs/figures/`. 

## Data Availability

The multi-camera surveillance dataset used for the core experiments is available as `maadaaai/mulit-camera-real-scene-video-dataset` on Kaggle. The dataset loaders in `src/srdo/domain_d.py` are built to ingest this format natively.
