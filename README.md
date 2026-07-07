# PhyCoFlow Graph Spectral Coherence Diagnostics

Graph spectral diagnostics for evaluating whether scientific machine learning models preserve physically meaningful structure on irregular point-cloud fields.

This repository implements a post-training evaluation framework for PhyCoFlow-style turbulent-combustion models. Instead of evaluating predictions only with pointwise metrics such as MSE or relative error, this project analyzes model outputs in the graph spectral domain to measure whether predictions preserve spatial-frequency structure, field coherence, and cross-scale physical coupling.

This work is part of undergraduate research associated with the **MIT DENG Energy and Nanotechnology Group**, focused on improving how learned physical models are evaluated for turbulent-combustion and other prediction tasks.

> **Note:** This repository is a companion diagnostics project.  
> The actual PhyCoFlow model implementation, training code, and demo tasks are maintained in the upstream repository:  
> https://github.com/cosmos2w/PhyCoFlow_demo

---

## Project Overview

Scientific machine learning models can achieve low pointwise error while still failing to preserve important physical structure. In turbulent-combustion systems, this can mean losing high-frequency spatial variation, weakening coupling between physical fields, or producing predictions that look accurate numerically but are less physically coherent.

This project builds graph spectral tools to evaluate those failures more directly.

The core idea is to represent the point-cloud geometry as a graph, compute a graph Laplacian basis, project physical fields into graph-frequency space, and compare predicted fields against ground truth across low-, mid-, and high-frequency bands.

This repository was developed as part of research associated with the **MIT DENG Energy and Nanotechnology Group**, where the broader goal is to evaluate and improve learned models for complex physical systems.

---

## Why This Matters

Standard evaluation metrics answer questions like:

```text
How close is the prediction to the target at each point?
```

This project asks more diagnostic questions:

```text
Does the model preserve global low-frequency structure?
Does it recover localized high-frequency variation?
Are errors concentrated in certain spatial-frequency bands?
Do predicted fields remain coherent with the ground truth?
Does the model preserve coupling between different physical variables?
```

For physics-informed machine learning, these questions are often just as important as raw reconstruction error.

---

## Technical Approach

The evaluation pipeline is based on graph signal processing.

```text
Point-cloud coordinates
        ↓
k-nearest-neighbor graph
        ↓
weighted adjacency matrix
        ↓
graph Laplacian
        ↓
graph Fourier basis
        ↓
spectral diagnostics on predicted vs. target fields
```

Physical fields are projected onto the graph Fourier basis, allowing predictions and targets to be compared by spatial scale.

Low graph frequencies capture smoother global structure, while higher graph frequencies capture more localized variation. This makes it possible to diagnose where a model is preserving or losing physically meaningful information.

---

## What This Repository Implements

This repository includes tools for:

- constructing graph representations of irregular point-cloud geometries,
- building weighted k-nearest-neighbor adjacency matrices,
- computing graph Laplacians and graph spectral bases,
- projecting physical fields into graph-frequency space,
- comparing predicted and target fields by graph-frequency band,
- evaluating same-frequency spectral coherence,
- comparing low-, mid-, and high-frequency band energy,
- analyzing cross-field and cross-frequency coupling behavior,
- visualizing spectral differences between model predictions and ground truth.

The framework is designed for post-training evaluation of learned physical models, especially PhyCoFlow-style models trained on turbulent-combustion data.

---

## Relationship to PhyCoFlow

This repository does **not** contain the canonical PhyCoFlow model implementation.

The actual PhyCoFlow model, training code, and demo tasks are maintained here:

```text
https://github.com/cosmos2w/PhyCoFlow_demo
```

This repository is intended to sit alongside the upstream model code as a diagnostics and evaluation layer.

```text
PhyCoFlow_demo
    → trains and runs the physical model

PhyCoFlow Graph Spectral Coherence Diagnostics
    → evaluates whether model outputs preserve graph-frequency structure,
      spectral coherence, and cross-scale physical coupling
```

---

## Repository Structure

```text
.
├── README.md
├── LICENSE
├── .gitignore
└── src/
    ├── graph_spectral_coherence/
    │   ├── __init__.py
    │   ├── graph_basis.py
    │   ├── cross_spectral.py
    │   ├── direct_cross_spectral_loss.py
    │   └── eval_coherence.py
    │
    └── visuals/
        ├── compare_band_energy.py
        └── visualizations.py
```

### Core Package: `graph_spectral_coherence`

The `graph_spectral_coherence` package contains the main graph spectral evaluation logic.

- `graph_basis.py`  
  Builds graph representations of point-cloud domains, including k-nearest-neighbor graph construction, weighted adjacency matrices, graph Laplacians, graph spectral bases, and low/mid/high graph-frequency bands.

- `cross_spectral.py`  
  Implements graph Fourier-domain diagnostics for comparing predicted and target physical fields, including spectral energy comparisons, same-frequency coherence, and cross-field or cross-frequency relationships.

- `direct_cross_spectral_loss.py`  
  Provides loss-style utilities for using graph spectral coherence terms during direct post-training or diagnostic optimization experiments.

- `eval_coherence.py`  
  Runs the main coherence evaluation pipeline on saved model outputs, graph bases, and turbulent-combustion field predictions.

- `__init__.py`  
  Marks the directory as a Python package and exposes reusable graph spectral coherence utilities.

### Visualization Utilities: `visuals`

The `visuals` directory contains scripts for converting spectral diagnostics into interpretable figures.

- `compare_band_energy.py`  
  Generates comparison plots for low-, mid-, and high-frequency band energy between predicted and ground-truth physical fields.

- `visualizations.py`  
  Produces higher-level diagnostic visualizations for coherence, spectral energy, and cross-frequency or cross-field behavior.

---

## Example Diagnostics

The framework is designed to support diagnostics such as:

```text
Prediction vs. Ground Truth
├── Low-frequency energy preservation
├── Mid-frequency structure preservation
├── High-frequency/local detail preservation
├── Same-frequency spectral coherence
├── Cross-field spectral relationships
└── Cross-frequency coupling
```

These diagnostics help identify cases where a model appears accurate under standard metrics but fails to preserve physically meaningful structure at specific spatial scales.

---

## Intended Workflow

A typical workflow is:

```text
1. Train or load a PhyCoFlow-style model.
2. Generate predicted physical fields.
3. Load the corresponding ground-truth fields.
4. Build or load a graph spectral basis for the point-cloud geometry.
5. Project predictions and targets into graph-frequency space.
6. Compute spectral coherence and band-energy diagnostics.
7. Generate plots comparing model predictions against ground truth.
```

This workflow is intended for evaluating trained physical models after inference or training, rather than replacing the upstream model training pipeline.

---

## Research Context

This repository is part of undergraduate research associated with the **MIT DENG Energy and Nanotechnology Group**.

The research focuses on improving evaluation methods for scientific machine learning models used in physical prediction tasks. In particular, this project explores whether graph spectral diagnostics can reveal structure-preservation failures that are not captured by standard pointwise metrics.

The broader motivation is to make evaluation more physically meaningful by studying not only whether predictions are numerically close to ground truth, but also whether they preserve spatial-frequency behavior, field coherence, and cross-scale relationships.

---

## Skills Demonstrated

This project demonstrates work across scientific machine learning, graph signal processing, and research software engineering:

- Python research software development
- graph construction for irregular point-cloud data
- sparse linear algebra and graph Laplacian eigendecomposition
- graph Fourier analysis of physical fields
- spectral evaluation of learned physical models
- post-training model diagnostics
- physics-informed ML evaluation
- modular code organization for metrics and visualizations
- integration with an external ML research codebase
- research software development in an MIT lab setting

---

## Current Status

This repository is under active research development.

The current focus is on building reliable post-training diagnostics for PhyCoFlow-style turbulent-combustion models, including graph-frequency band energy analysis, same-frequency coherence, and cross-frequency coupling.

Planned improvements include cleaner command-line interfaces, expanded visualization outputs, and tighter integration with the upstream PhyCoFlow demo repository.

---

## Upstream Model Repository

The original PhyCoFlow model and demo code are available here:

```text
https://github.com/cosmos2w/PhyCoFlow_demo
```

This repository builds complementary graph spectral evaluation tools around that modeling framework.