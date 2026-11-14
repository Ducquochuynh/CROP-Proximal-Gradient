# Nonnegative Ridge Regression (NRR)

## Description

This folder contains MATLAB codes for numerical experiments on nonnegative ridge regression (NRR) using proximal gradient methods and the proposed CROP-PG acceleration framework.

## Requirements

- MATLAB
- Required toolboxes, if any, are specified in the implementation.

## Methods

The experiments compare several proximal gradient algorithms, including:

- Proximal Gradient (PG)
- Fast Iterative Shrinkage-Thresholding Algorithm (FISTA)
- Anderson Acceleration for Proximal Gradient (AA-PG)
- CROP-PG (Exact and Inexact)

Additional comparison methods may be included in the experiment scripts.

## How to Run

1. Download or clone the repository.
2. Open MATLAB.
3. Set the current folder to `NRR_GITHUB`.
4. Run the main NRR experiment script.

## Experimental Setup

Synthetic data are generated using the settings specified in the MATLAB scripts. The experiments evaluate convergence behavior, iteration counts, and computational time.

## Reproducibility

All algorithmic parameters and stopping criteria are specified in the corresponding MATLAB implementation files.
