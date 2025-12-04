# Attention Map Analysis: Input Distribution Effects

This project explores how assumptions on input distribution affect learned attention maps in transformer models for time series forecasting.

## Overview

The project trains transformer models on various time series with different distributional characteristics and analyzes how attention patterns differ across these distributions.

## Project Structure

- `data_generators.py`: Time series generators with different characteristics
- `transformer_model.py`: Transformer model implementation for time series forecasting
- `trainer.py`: Training utilities and attention extraction functions
- `visualization.py`: Attention map visualization and analysis tools
- `main_experiment.py`: Main script to run all experiments

## Time Series Generators

### 1. Stochastic & Volatile (Noise-Driven)
- **HeavyTailedAR1**: AR(1) process with Student-t noise (fat-tailed outliers)
- **GARCH11**: Time-varying volatility with clustering
- **JumpDiffusion**: Drift-diffusion with random massive jumps

### 2. Non-Stationary & Structural (Shift-Driven)
- **RegimeSwitching**: Mean shifts based on hidden Markov state
- **TrendBreaks**: Linear trend with abrupt slope changes
- **RandomWalk**: Non-stationary process with no mean reversion

### 3. Periodic & Deterministic (Pattern-Driven)
- **SeasonTrendOutliers**: Sinusoidal wave with trend and sparse outliers
- **MultiSeasonality**: Complex overlapping cycles

### 4. Long Memory
- **OneOverFNoise**: 1/f noise with persistent correlations

## Model Configuration

Fixed hyperparameters (as specified):
- `d_model`: 64
- `nhead`: 4
- `num_layers`: 2
- `dim_feedforward`: 256
- `dropout`: 0.1
- `input_len` (L): 100 past points
- `output_len` (H): 20 future points
- Loss: MSE between predicted and true future

## Attention Analysis

The project provides multiple ways to analyze attention maps:

1. **Visual Heatmaps**: 
   - X-axis: Time being attended to
   - Y-axis: Time doing the attending
   - Color: Attention strength
   - Patterns to look for: bright diagonals, repeated stripes, horizontal/vertical patterns

2. **Entropy Analysis**: Measures how focused vs. uniform attention is
   - Higher entropy = more uniform attention
   - Lower entropy = more focused attention

3. **Sparsity Analysis**: Gini coefficient measuring concentration
   - Higher sparsity = more concentrated attention

4. **Pattern Detection**:
   - Diagonal patterns (local attention)
   - Horizontal patterns (attending to specific time points)
   - Vertical patterns (specific positions focus heavily)

5. **Comprehensive Metrics**: Includes entropy, sparsity, concentration, and pattern strengths

## Usage

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run experiments:
```bash
python main_experiment.py
```

This will:
- Generate time series for each generator
- Train a transformer model on each
- Extract attention weights
- Create visualizations
- Compute analysis metrics
- Save all results to `results/` directory

## Results Structure

Results are saved in `results/<generator_name>/`:
- `training_curves.png`: Training and validation loss curves
- `attention_summary_<generator_name>.png`: Overview of all attention maps
- `entropy_analysis.png`: Entropy analysis across layers and heads
- `attention_L<layer>_H<head>_S<sample>.png`: Individual attention heatmaps
- `attention_metrics.json`: Quantitative metrics for each layer/head
- `summary.json`: Overall experiment summary

## Customization

You can modify the configuration in `main_experiment.py`:
- Adjust model architecture (d_model, nhead, num_layers, etc.)
- Change input/output lengths (L and H)
- Modify training parameters (epochs, learning rate, batch size)
- Adjust generator parameters in `data_generators.py`

## Research Questions

This project helps answer:
- Do attention heads overreact to spikes in heavy-tailed distributions?
- Do models learn persistence or treat everything as noise?
- Does attention concentrate during turbulent periods (GARCH)?
- Can models recover after sudden regime changes?
- How do attention patterns differ between stationary and non-stationary processes?
- Do predictable patterns lead to different attention structures?

