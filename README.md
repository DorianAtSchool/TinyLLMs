# TinyLLMs

Small JAX experiments for a character-level language model and a short
regression warm-up. The script builds data generators, runs signed gradient
descent, and compares progressively larger models: constant, linear, MLP, and
two-layer variants. The report collects plots, timing tables, and sample text.

## Contents
- `code/run_me.py`: single script that executes all experiments end-to-end
- `TinyLLMs_Report.pdf`: report with figures, tables, and generated samples
- `report_src/`: source snippets and the working notebook used to build the report

## Requirements
- Python 3.9+
- JAX + jaxlib
- numpy
- matplotlib

## Data
The language model sections expect a `data.npz` file in the repo root with an
array named `data` containing integer-encoded characters. The encoding must
match the `chars` alphabet defined in `code/run_me.py`.

## Experiments
- Regression warm-up: fit a quadratic to synthetic data and compare learning
  rates and batch sizes using signed gradient descent.
- Data pipeline: sample fixed-length character contexts and next-token targets
  from the `data.npz` corpus.
- Baselines: constant model and linear context model with cross-entropy loss.
- Nonlinear models: single-hidden-layer MLP and a two-layer variant with ReLU.
- Generation: sample character sequences from each trained model to compare
  coherence and structure.

## Results
- Training logs include smooth-loss tracking and timing per experiment.
- Generated samples show a clear quality jump from constant to linear models,
  with additional gains from MLP and two-layer variants.
- Full outputs, plots, and example generations are in `TinyLLMs_Report.pdf`.

## Run
From the repo root:

```bash
python code/run_me.py
```

The script prints training logs and sample generations and opens a plot window
for the early regression task.
