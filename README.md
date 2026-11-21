# RNN from Scratch with NumPy

A small Recurrent Neural Network (RNN) implemented from scratch using NumPy, with the goal of understanding how recurrent neural networks work internally.

Instead of relying on deep learning frameworks such as PyTorch or TensorFlow, this project builds the core training process manually — from processing a sequence to calculating gradients and updating parameters.

##  Project Objective

The goal is to understand the complete learning process of a vanilla RNN:

- How a hidden state carries information across timesteps
- How the network processes sequential inputs
- How predictions and loss are calculated
- How gradients flow backward through time
- How shared parameters are updated using gradient descent

##  RNN Architecture

This project uses a tiny RNN with scalar inputs, scalar hidden states, and a scalar output.

At each timestep, the RNN calculates:

\[
z_t = W_x x_t + W_h h_{t-1} + b_h
\]

\[
h_t = \tanh(z_t)
\]

The final hidden state is used to produce a prediction:

\[
\hat{y} = W_y h_T + b_y
\]

Where:

| Parameter | Meaning |
|---|---|
| \(W_x\) | Input-to-hidden weight |
| \(W_h\) | Hidden-to-hidden recurrent weight |
| \(b_h\) | Hidden-layer bias |
| \(W_y\) | Hidden-to-output weight |
| \(b_y\) | Output-layer bias |
| \(h_t\) | Hidden state at timestep \(t\) |

The same recurrent parameters are reused at every timestep.

##  Training Pipeline

The training process consists of four main stages.

### 1. Forward Pass

The network processes the input sequence one timestep at a time.

At each timestep, it:
- Combines the current input with the previous hidden state.
- Applies the `tanh` activation.
- Stores the current hidden state for later use.

The final hidden state is used to calculate the prediction.

### 2. Loss Calculation

The model uses a squared-error loss:

\[
L = \frac{1}{2}(\hat{y}-y)^2
\]

This measures the difference between the predicted value and the target.

### 3. Backpropagation Through Time (BPTT)

The gradients are calculated by moving backward through the sequence.

Starting from the final hidden state, the gradient is propagated through each timestep using the chain rule.

The derivative of the `tanh` activation is:

\[
\frac{\partial h_t}{\partial z_t}=1-h_t^2
\]

Because the recurrent weights are shared across timesteps, their gradients are accumulated over the entire sequence.

### 4. Parameter Update

The parameters are updated using Stochastic Gradient Descent (SGD):

\[
\theta \leftarrow \theta-\eta\frac{\partial L}{\partial\theta}
\]

Where:
- \(\theta\) represents a model parameter.
- \(\eta\) is the learning rate.

The training loop repeats this process for multiple epochs.

##  Technologies

- Python
- NumPy

## Motivation

The purpose of this project is not simply to use an RNN, but to understand how it learns — one timestep, one gradient, and one parameter update at a time.
