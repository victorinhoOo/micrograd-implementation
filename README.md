# Micrograd Implementation with a Classification Test

[🇫🇷 Lire en français](README_fr.md)

This project is an implementation of **Micrograd**, a minimalist framework for automatic differentiation, created using Andrej Karpathy's tutorial:  
https://www.youtube.com/watch?v=VMj-3S1tku0&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ  

The goal is to deeply understand the fundamental concepts of the **forward pass**, **backward pass** (backpropagation), and their application in a neural network.

---

## 🧠 Micrograd

Micrograd is a simple framework for automatic differentiation (autograd) using the construction of a computational graph. Every operation is recorded in this graph, enabling the calculation of derivatives necessary for optimizing parameters in a machine learning model.

Key concepts covered in this project:
- Forward and backward passes.
- Construction of a simple neural network (MLP - Multi-Layer Perceptron).
- Supervised learning with artificial data.
- Application to a binary classification task.

---

## 📁 Structure 

The notebook contains several sections:
1. **Recreating Micrograd:** Development of the `Value` class to represent scalars with gradients.
2. **Building a Neural Network:** Creation of `Neuron`, `Layer`, and `MLP` classes to model a multilayer perceptron.
3. **Classification Test:** Application to a simple binary classification task with artificial data.
4. **Visualization of Results:** Displaying the decision boundary of the model.

---

## 🖼️ Visualization

Here is a visualization of the model's decision boundary after training:

![Decision Boundary](https://github.com/user-attachments/assets/6dc0d339-9e88-4728-8786-c9c5ad22a514)

---

## 🌐 Languages

- [🇫🇷 Version française](README_fr.md)
