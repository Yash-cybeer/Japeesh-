# mini-gencast

Mini GenCast-style probabilistic weather forecasting project.

## Overview
This project implements a scaled-down version of GenCast for weather prediction using:
- Conditional diffusion models
- Icosahedral mesh processing with graph transformers
- Synthetic ERA5-like weather data (T2M, U10, V10)

## Quick Start
1. Install dependencies: `pip install -r requirements.txt`
2. Run training: See demo code below
3. Sample predictions and evaluate

## Structure
- `mini_gencast/dataset.py` - Data loading (synthetic & ERA5)
- `mini_gencast/mesh.py` - Icosahedral mesh generation
- `mini_gencast/model.py` - Neural network architecture
- `mini_gencast/diffusion.py` - Diffusion process
- `mini_gencast/train.py` - Training loop
- `mini_gencast/sample.py` - Sampling/inference
- `mini_gencast/evaluate.py` - Evaluation metrics

## Notes
This is a toy implementation for learning. For production use:
- Use real ERA5 data (0.25° resolution)
- Larger mesh (~41k nodes)
- Advanced samplers (DPMSolver++2S)
- Multi-GPU training
