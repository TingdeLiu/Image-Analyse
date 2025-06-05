<p align="center">
<h1 align="center"><strong>🧠 Image Analysis I: Lab Series (ST 2023)</strong></h1>
  <p align="center">
    <a href='mailto:kanyamahanga@ipi.uni-hannover.de' target='_blank'>Hubert Kanyamahanga</a>&emsp;
    <a href='mailto:tingde.liu@gmail.com' target='_blank'>Tingde Liu</a>&emsp;
    Chengliang Li&emsp;
    Yahui Zhuang
    <br>
    Institute of Photogrammetry and GeoInformation (IPI), Leibniz Universität Hannover
  </p>
</p>

---

## 🧪 Overview

This is a course I took at Leibniz University and the code is an exercise for my accompanying course. This repository contains practical lab exercises from the **Image Analysis I** course at IPI Hannover, focusing on classical and modern computer vision and deep learning techniques using Python, Scikit-learn, and PyTorch.

---

## 🧭 Labs Summary

### 🔬 Lab 1: Generative Probabilistic Models

- Understand generative classification (Bayesian decision theory).
- Work with Gaussian distributions and Bayes’ theorem.
- Implement:
  - Naive Bayes Classifier
  - Gaussian Mixture Models (GMM)
- Visual evaluation of classifiers using synthetic datasets.

### 🧮 Lab 2: Discriminative Probabilistic Models & Neural Networks

- Implement **Logistic Regression** for binary and multi-class classification.
- Understand and apply:
  - Feature space mapping
  - Softmax function
  - Maximum Likelihood estimation
- Learn fundamentals of **Neural Networks**:
  - Perceptron and MLP
  - Training with PyTorch (Loss, Gradient Descent, Regularization)
- Introduction to PyTorch API:
  - `torch.Tensor`, `torch.nn`, `torch.optim`

### 🧠 Lab 3: Deep Learning for Image Classification & Segmentation

- Apply **Convolutional Neural Networks (CNN)** for image classification.
- Build **Fully Convolutional Networks (FCNs)** for pixel-wise semantic segmentation.
- Understand:
  - CNN layers: Convolution, Pooling, Fully Connected
  - Encoder-Decoder structures (e.g. U-Net)
  - Loss functions like Cross-Entropy
- Use real datasets:
  - **MNIST** for classification
  - **UAVid** for segmentation

---

## 📦 Requirements

- Python 3.8+
- Jupyter Notebook
- `numpy`, `matplotlib`, `scikit-learn`, `torch`, `torchvision`

---

## 📁 Folder Structure

```bash
.
├── lab1/               # Generative models
├── lab2/               # Logistic Regression and Neural Nets
├── lab3/               # CNN and FCN
└── README.md
