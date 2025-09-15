# StepNav: Efficient Planning with Structured Trajectory Priors

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)

> **StepNav** is an efficient planning framework for visual navigation that generates reliable trajectories in complex and uncertain environments using structured trajectory priors.

## 🚀 Overview

Visual navigation is fundamental for autonomous systems, but generating reliable trajectories in complex environments remains challenging. Existing generative approaches often rely on unstructured noise priors, leading to unsafe or inefficient plans that require extensive refinement.

**StepNav** addresses these limitations through:

- **🎯 Structured Trajectory Priors**: Multi-modal, high-quality trajectory initialization instead of random noise
- **⚡ Real-time Efficiency**: Lightweight architecture suitable for real-time autonomous navigation  
- **🛡️ Safety-Aware Planning**: Success probability fields highlight safe and viable regions
- **🌟 Physical Continuity**: Dynamics-Inspired Feature Projection (DIFP) enforces spatiotemporal consistency

## 🏗️ Architecture

StepNav consists of three key components:

1. **Dynamics-Inspired Feature Projection (DIFP)**: Refines encoder outputs to enforce physical continuity in spatiotemporal features
2. **Success Probability Field**: Lightweight network predicting energy-based safety maps from egocentric visual inputs
3. **Conditional Flow-Matching Generator**: Generates final trajectories initialized from structured priors

## 📦 Installation

### Prerequisites
- Python 3.8+
- PyTorch 1.12+
- CUDA 11.3+ (for GPU acceleration)

### Quick Start
```bash
# Clone the repository
git clone https://github.com/yourusername/stepnav.git
cd stepnav

# Create conda environment
conda create -n stepnav python=3.8
conda activate stepnav

# Install dependencies
pip install -r requirements.txt

# Install StepNav
pip install -e .
```

## Acknowledgements
This project is inspired by and builds upon the following works:
- [NoMaD](https://github.com/robodhruv/visualnav-transformer)
- [FlowNav](https://github.com/utn-air/flownav)
- [NaviBridger](https://github.com/hren20/NaiviBridger)