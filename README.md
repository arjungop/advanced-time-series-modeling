# Advanced Time Series Modeling Suite

<p align="left">
  <img src="https://img.shields.io/badge/Python-3.9+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-Deep_Learning-EE4C2C.svg" alt="PyTorch">
  <img src="https://img.shields.io/badge/Statsmodels-Time_Series-orange.svg" alt="Statsmodels">
  <img src="https://img.shields.io/badge/Machine_Learning-Scikit_Learn-yellow.svg" alt="Scikit-Learn">
</p>

## Overview
An industry-grade, end-to-end framework for **Multivariate Time Series Analysis and Forecasting**. This repository processes high-frequency, massive-scale sensor data (originally 5 million+ points) and applies a rigorous analytical pipeline spanning statistical methodologies, ensemble machine learning, and recurrent neural networks (PyTorch). 

It is designed to evaluate, analyze, and forecast complex temporal dynamics through multi-horizon predictive modeling.

## Key Capabilities

- **Deep Learning Forecasting:** Implements Gated Recurrent Units (GRU) leveraging PyTorch with MPS/GPU acceleration for sequence-to-sequence prediction.
- **Statistical & ML Modeling:** Evaluates classic parametric models (ARIMA/SARIMAX), non-linear machine learning ensembles (Random Forest Regressor), and volatility models (GARCH).
- **Rigorous Stationarity Hypotheses:** Advanced implementations of Augmented Dickey-Fuller (ADF), Kwiatkowski-Phillips-Schmidt-Shin (KPSS), and Phillips-Perron (PP) tests.
- **Advanced Preprocessing:** Handles non-linear transformations, sequence structurization (lookback windows), feature scaling (Standard & Min-Max), and high-frequency resampling.
- **Automated Diagnostic EDA:** Generates detailed Autocorrelation (ACF), Partial Autocorrelation (PACF), seasonal decomposition, and rolling statistics plots.

## Project Architecture

```text
├── data/                   # Large-scale datasets and sensor logs (ignored from git)
├── notebooks/              # Interactive environments for model experimentation
│   ├── TSA_Project.ipynb   # Comprehensive statistical and Deep Learning (GRU) modeling
│   └── TSA_ISP_Traffic.ipynb # ISP-specific sub-domain modeling
├── src/                    # Core, reproducible Python modules
│   ├── TSA_Updated.py      # Main pipeline script for time series preprocessing & modeling
│   └── TSA_ISP_Updated.py  # Specialized flow metrics processing pipeline
├── requirements.txt        # Virtual environment dependency lockfile
└── README.md
```

## Getting Started

### 1. Environment Initialization

```bash
# Clone the repository
git clone https://github.com/arjungop/advanced-time-series-modeling.git
cd advanced-time-series-modeling

# Create and activate your virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies (includes PyTorch, Statsmodels, Scikit-Learn, etc.)
pip install -r requirements.txt
```

### 2. Execution

To run the primary forecasting pipeline natively:

```bash
python src/TSA_Updated.py
```

To explore the PyTorch GRU architecture and deep statistical breakdowns interactively:

```bash
jupyter notebook notebooks/TSA_Project.ipynb
```
