<!-- Shields: -->
[![Python](https://img.shields.io/badge/Python-3.11.3-informational)](https://www.python.org/downloads/source/)

<!-- Title: -->
<div align="center">
  <h1> Vanilla NumPy MLP Neural Network with SGD and Backpropagation </h1>
</div>
    
# Dependencies

<a href="https://www.python.org/" target="_blank" rel="noopener noreferrer"><img height="27" src="https://www.python.org/static/img/python-logo.png"></a>
<a href="https://matplotlib.org" target="_blank" rel="noopener noreferrer"><img height="27" src="https://matplotlib.org/_static/images/logo2.svg"></a>
<a href="https://numpy.org/" target="_blank" rel="noopener noreferrer"><img height="27" src="https://numpy.org/images/logo.svg"></a>
<br>

# About

> This material is designed to educate by bridging the gap between theory and implementation. If there is a blunder, do not hesitate to open an issue in the issue tracker.

Vanilla NumPy [implementation](fmnist_backprop_numpy.ipynb) of multilayer perceptron (MLP), a fully connected feedforward neural network, including Stochastic Gradient Descent (SGD) optimizer and backpropagation algorithm. The implementation was trained/tested using the Fashion-MNIST dataset for image classification. Resort to the [theory.ipynb](theory.ipynb) notebook for a theoretical background and derivation of the backpropagation algorithm, considering `Sigmoid` and `Softmax` activation functions, as well as `MSE` and `Categorical Cross-Entropy` loss functions.

To comply with GitHub's limit on large files, the dataset folder was git ignored. Download the [Fashion MNIST](https://www.kaggle.com/datasets/zalando-research/fashionmnist) dataset, extract it, then yank and paste the train and test files to a `dataset/` folder in the root directory of your local clone.

<div align="center">
  <a href="#"><img src="assets/predictions.png"/></a>
  <a href="#"><img src="assets/training_plot.png"/></a>
</div> 

# Technologies

- `NumPy`: data preprocessing (reading from CSV, decoding, rescaling, data splitting, data augmentation), feature engineering (one-hot encoding for labels), activation/loss functions, forward pass, backpropagation, etc.
- `Matplotlib`: plotting.
    
# Implementation details

- Architecture: MLP with two fully-connected hidden layers. Has support for two loss functions (**mse** and **cross-entropy**), three weight initializations (**xavier**, **he** or **normal**), **L2 regularization**, and training by batches.
- Dataset: [Fashion MNIST](https://www.kaggle.com/datasets/zalando-research/fashionmnist).

# Setting up the environment

I recommend Conda for environment management.

```bash
conda create -yn mlp python==3.11.3 && conda activate mlp \
&& conda install -yc conda-forge pip==23.2.1 && python -m pip install --user --upgrade pip \
&& python -m pip install -r requirements.txt
```
