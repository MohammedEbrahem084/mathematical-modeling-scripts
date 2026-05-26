# Mathematical Modeling and Data Analysis

This repository contains MATLAB scripts demonstrating applied mathematical reasoning, algorithm optimization, and complex system modeling.

## Files Included:

* **`lif_sunscreen_detection_svm.m`**: 
  * **Objective**: Detects microscopic contamination in food-grade materials using Laser-Induced Fluorescence Hyperspectral Imaging (LIF-HSI).
  * **Mathematical Approach**: Integrates a complete machine learning classification pipeline. Features mathematical data preprocessing (baseline correction, Savitzky-Golay smoothing), Hyperparameter Optimization, L1-Regularized Sparse Principal Component Analysis (SPCA) for feature extraction, and Support Vector Machine (SVM) modeling with k-fold cross-validation.
  * **Outcome**: Evaluates spatial and statistical performance metrics (F1-score, precision, recall) against ground-truth arrays to generate high-accuracy pixel-wise classification maps.

* **`hyperspectral_spca_fusion.m`**: 
  * **Objective**: Automates the sorting of materials using complex hyperspectral data under extreme industrial noise conditions (Photon starvation + 30% Gaussian noise).
  * **Mathematical Approach**: Implements Probability Density Function (PDF) divergence to extract optimal spectral bands. Features a custom-built L1-Regularized Sparse Principal Component Analysis (SPCA) algorithm to fuse multidimensional data matrices, significantly improving Contrast-to-Noise Ratio (CNR) compared to standard PCA.
  * **Outcome**: The mathematical models validated by this code were published in *Environmental Geochemistry and Health* (2026).

## Core Competencies Demonstrated:
* End-to-end Machine Learning pipeline development (SVM, cross-validation).
* Step-by-step mathematical logic translation into scientific code.
* Custom algorithm design for dimensionality reduction and statistical classification.
* Handling and optimization of massive, multi-dimensional data arrays.
