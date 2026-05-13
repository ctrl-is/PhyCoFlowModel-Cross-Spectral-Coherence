# PhyCoFlowModel Cross-Spectral Coherence

This repository develops a graph-based cross-spectral coherence framework for evaluating physical field predictions on irregular point-cloud geometries. The goal is to move beyond pointwise error metrics and analyze whether a learned model preserves the spatial-frequency structure of turbulent combustion fields.

The project currently builds graph tools for point-cloud data, computes graph Laplacian spectra, projects physical fields into the graph Fourier domain, and sets up coherence-style comparisons between predicted and target fields.

---

## Motivation

Many physics-informed and operator-learning models are evaluated using standard reconstruction metrics such as mean squared error or relative error. These metrics are useful, but they do not fully capture whether a model preserves important spatial structures across scales.

For turbulent combustion and other physical systems on irregular geometries, we often care about questions such as:

- Does the prediction preserve low-frequency global structure?
- Does it recover high-frequency localized variation?
- Are errors concentrated in particular spectral bands?
- Do predicted and target fields remain coherent across graph frequencies?

To answer these questions, this project represents the point cloud as a graph and analyzes physical fields using graph spectral methods.

---

## Core Idea

Given point-cloud coordinates, we construct a weighted graph whose nodes correspond to spatial locations. A graph Laplacian is then computed from the weighted adjacency matrix.

For a graph signal \(x\), such as temperature, velocity, or another scalar field over the point cloud, the graph Fourier transform is defined by projecting the signal onto the eigenvectors of the graph Laplacian:

\[
\hat{x} = U^\top x
\]

where:

- \(U\) contains the graph Laplacian eigenvectors,
- \(x\) is a field defined over graph nodes,
- \(\hat{x}\) is the graph-frequency representation of the field.

For predicted and target fields, the project compares their spectral behavior using graph power spectra and cross-spectral coherence.

A simplified coherence quantity is:

\[
C_{xy}(k) =
\frac{|S_{xy}(k)|^2}{S_{xx}(k)S_{yy}(k)}
\]

where:

- \(S_{xx}(k)\) is the target auto-spectrum,
- \(S_{yy}(k)\) is the prediction auto-spectrum,
- \(S_{xy}(k)\) is the cross-spectrum between target and prediction,
- \(k\) indexes graph frequencies.

High coherence means the predicted and target fields are aligned in that graph-frequency band.

---

## Current Features

The repository currently includes tools for:

- Extracting graph node coordinates from a turbulent combustion dataset
- Building k-nearest-neighbor graphs from point-cloud coordinates
- Computing edge distances
- Constructing weighted graph adjacency matrices
- Building graph Laplacians
- Computing graph spectra using sparse eigensolvers
- Preparing predicted and target fields for graph spectral analysis
- Supporting downstream coherence evaluation between model outputs and ground-truth physical fields
