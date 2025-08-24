# MTGNet: A Multi-Modal Transformer-Graph Network for Real-Time EEG Analysis

This repository contains the work-in-progress code for **MTGNet**, a novel deep learning framework for real-time EEG signal analysis. This project is being developed as part of my Master's Thesis in [Your Field, e.g., Computer Science, Biomedical Engineering].

## Abstract

This project presents MTGNet, a novel framework for real-time EEG signal generation and cognitive state decoding that leverages a fusion of transformer encoders and graph neural networks. Using established datasets—such as the BETA dataset for steady-state visual evoked potentials and the SEED dataset for emotional state recognition—the system first preprocesses EEG signals with standard artifact removal and segmentation techniques. A 1D convolutional neural network extracts channel-wise temporal features, which are then processed by a transformer encoder to capture long-range temporal dependencies and dynamic patterns. In parallel, a graph is constructed from EEG channels using functional connectivity measures, and a graph neural network refines these spatial relationships. The resulting multi-modal features (with optional integration of complementary data like fNIRS) are fused and fed into a classifier for accurate, real-time cognitive state decoding. Visualization modules display attention maps and dynamic inter-channel connectivity to provide interpretable neurofeedback. MTGNet thus offers a robust, adaptable solution for advanced brain–computer interface applications and real-time neurofeedback systems without relying on reinforcement learning.

## Core Architecture

MTGNet employs a hybrid, multi-modal architecture to simultaneously process the temporal and spatial dimensions of EEG data. The Temporal Path captures dynamic patterns within individual channels over time, while the Spatial Path models the complex, interconnected relationships between different brain regions.


* **Temporal Path:** Raw EEG signals are processed by a 1D CNN to extract low-level temporal features from each channel. These features are then fed into a Transformer Encoder, which uses its self-attention mechanism to model long-range dependencies and dynamic patterns over time.
* **Spatial Path:** A Graph Neural Network (GNN) directly models the spatial relationships between EEG channels. The GNN constructs a graph where nodes represent EEG electrodes and edges represent functional connectivity, allowing it to learn features from the brain's network topology.
* **Feature Fusion:** The learned temporal and spatial feature representations are fused into a single, comprehensive vector, which is then passed to a classifier head for the final cognitive state decoding task.

## Current Status

**⚠️ This is an active research repository for an ongoing Master's Thesis.**

The code is currently under active development and should be considered a **work-in-progress**. The initial scripts for the model architecture, data loaders, and preprocessing steps have been pushed. However, the full training and validation pipelines are not yet finalized, and the code has not been optimized for production use.

## Project Roadmap

The following is a planned roadmap for the completion of the MTGNet project:

-   [ ] **Model Implementation:** Finalize and validate the core Transformer-GNN architecture.
-   [ ] **Dataset Integration:** Complete data loading and preprocessing pipelines for the BETA and SEED datasets.
-   [ ] **Model Training & Validation:** Train and benchmark the model's performance on SSVEP and emotion recognition tasks.
-   [ ] **Interpretability Module:** Develop the visualization components for generating Transformer attention maps and dynamic GNN connectivity graphs.
-   [ ] **Code Refactoring:** Clean and document the codebase for the final thesis submission.

## Tech Stack

* **Frameworks:** PyTorch / TensorFlow
* **Libraries:** Scikit-learn, MNE-Python (for EEG processing), PyTorch Geometric / DGL (for GNNs), NumPy, Matplotlib