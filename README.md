# Machine Learning Implementations

<!-- One or two sentences: what this repo is. e.g. "Implementations and notes from
Georgia Tech CS3600 (Intro to AI). Each folder is a self-contained project where I
build a core ML algorithm from scratch and compare it against library versions." -->
This repo consists of machine learning and AI projects that I have done both at GT and on my own. Each folder is a self-contained project where I build a core ML algorithm from scratch, then compare it against popular library versions.

---

## Table of Contents

<!-- These links jump to a folder on GitHub. Spaces in folder names become %20.
Delete rows for anything that isn't a real project (or that you don't want public). -->

- [Nearest Neighbor](NearestNeighbor/) — k-NN from scratch, sklearn, FAISS, XGBoost
- [Linear Regression](LinearRegression/) — normal equation, hand-derived gradient descent, JAX autodiff
- [Neural Network](NeuralNet/) — MLP for MNIST digits: from scratch with manual backprop, then the same net in Flax NNX
---

## Overview

<!-- A short paragraph on the theme tying these together: the course, the language/
libraries (JAX, NumPy, scikit-learn, XGBoost, FAISS), and your general approach
(e.g. "build the core algorithm by hand first, then benchmark against libraries"). -->
For these projects, I like to build out the core algorithm by hand, or in scratch, then benchmark against multiple libraries. The projects are coded with JAX, a tensor library that is better than numpy:
- JAX includes automatic differentation
- JAX has a JIT compiler that makes it faster than numpy or Pytorch
- JAX is built on XLA, so it can efficiently utilize CPUs, GPUs, TPUs
---

## Tech Stack

<!-- Bullet the main tools so a reader knows what they're looking at. -->
- **Language:** Python 3.14
- **Core libraries:** JAX / NumPy, scikit-learn, XGBoost, FAISS, Flax NNX, optax
- **GT PACE:** HPC resource @ GT
---

## Setup

<!-- How someone (including future-you) gets this running. Fill in the real commands. -->

```bash
# clone the repo
git clone https://github.com/rurup1/Machine-Learning-Implementations.git
cd Machine-Learning-Implementations

# create + activate a virtual environment
python -m venv venv
source venv/bin/activate        # macOS / Linux

# install dependencies
pip install -r requirements.txt
```

<!-- Note: you don't have a requirements.txt yet. Either create one with
`pip freeze > requirements.txt`, or just list the key packages here manually. -->

---

## Repository Structure

<!-- A quick map so readers know where things live. Trim to match reality. -->

```
Machine-Learning-Implementations/
├── NearestNeighbor/       # k-NN project (see its own README for deep dive)
├── LinearRegression/      # linear regression 3 ways: matrix inversion, gradient descent, autodiff
└── NeuralNet/             # MNIST MLP: from-scratch backprop vs. Flax NNX
```

---

## Projects

<!-- OPTIONAL: one short subsection per project as a teaser. The DEEP technical
writeup lives in each project's own README (e.g. NearestNeighbor/README.md),
not here. Keep these to a few sentences and link into the folder. -->

### Nearest Neighbor → [details](NearestNeighbor/)

### Linear Regression → [details](LinearRegression/)

### Neural Network → [details](NeuralNet/)

A two-layer MLP (784 → 128 → 10) that classifies MNIST handwritten digits, built two ways. First **from scratch** in pure JAX — hand-implemented forward pass, softmax/cross-entropy, and manual backpropagation through every layer (He initialization, ReLU, gradient descent). Then the **same network in Flax NNX**, leaning on the library for autodiff and an optax optimizer. Both reach ~97.5% test accuracy, illustrating how much the modern tooling collapses while producing equivalent results.
---
<!-- OPTIONAL closing section: what you learned, license, contact, etc. -->
