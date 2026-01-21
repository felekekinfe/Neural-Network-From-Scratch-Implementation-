```markdown
# Neural Network From Scratch (NumPy)

A minimal, **from-scratch implementation of a feedforward neural network** using only **NumPy**.  
This project demonstrates how core deep learning components work *under the hood* without relying on high-level frameworks like TensorFlow or PyTorch.

The model is trained on a **2D spiral dataset** for multi-class classification.

---

## 📌 Features

- Fully connected (Dense) layers
- ReLU activation
- Softmax activation
- Categorical Cross-Entropy loss
- Optimized **Softmax + Cross-Entropy** backward pass
- Stochastic Gradient Descent (SGD) optimizer
- End-to-end forward & backward propagation
- Training loop with accuracy and loss logging

---

## 🧠 Model Architecture

```

Input (2)
↓
Dense (2 → 64)
↓
ReLU
↓
Dense (64 → 3)
↓
Softmax + Categorical Cross-Entropy

```

---

## 🧮 Mathematical Formulation

**Dense layer**
```

Z = XW + b

```

**ReLU**
```

ReLU(x) = max(0, x)

```

**Softmax**
```

Softmax(z_i) = exp(z_i) / Σ exp(z_j)

```

**Categorical Cross-Entropy Loss**
```

L = -log(p_correct)

```

**Optimized Gradient (Softmax + Cross-Entropy)**
```

∂L/∂z = (y_pred - y_true) / N

````

---

## 🚀 Getting Started

### 1. Install Dependencies
```bash
pip install numpy nnfs
````

### 2. Run the Script

```bash
python main.py
```

---

## 📊 Training Output

The training loop runs for **10,001 epochs** and prints progress every 1000 epochs:

```
epoch: 0, acc: 0.340, loss: 1.099
epoch: 1000, acc: 0.810, loss: 0.465
epoch: 10000, acc: 0.950, loss: 0.120
```

---

## 🧩 Code Structure

* `Layer_Dense` – Fully connected layer
* `Activation_ReLU` – Non-linear activation
* `Activation_Softmax` – Output activation
* `Loss_CategoricalCrossentropy` – Classification loss
* `Activation_Softmax_Loss_CategoricalCrossentropy` – Optimized combined layer
* `Optimizer_SGD` – Parameter update rule
* Training loop – Forward pass → Backward pass → Update

---

## 🎯 Learning Objectives

This project is ideal if you want to:

* Understand **backpropagation mathematically**
* Learn how gradients flow through a network
* See how optimizations like **Softmax + CE fusion** work
* Build intuition before using deep learning frameworks

---

## 📚 Inspiration

Inspired by *Neural Networks from Scratch* (NNFS) by Harrison Kinsley & Daniel Kukieła.

---

## 📄 License

MIT License – free to use, modify, and learn from.

---

**Built for learning, not abstraction.**

```
```
