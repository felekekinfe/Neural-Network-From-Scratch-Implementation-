import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

# --- 1. DENSE LAYER ---
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        # Initialize weights with a Gaussian distribution scaled by 0.01
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
        # Gradients on parameters
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        # Gradient on input values (to pass to previous layer)
        self.dinputs = np.dot(dvalues, self.weights.T)

# --- 2. ACTIVATION FUNCTIONS ---
class Activation_ReLU:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        # Zero gradient where input values were negative
        self.dinputs[self.inputs <= 0] = 0

class Activation_Softmax:
    def forward(self, inputs):
        # Unnormalized probabilities (with overflow protection)
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)




# --- 3. LOSS FUNCTIONS ---
class Loss:
    """Base Loss class for handling mean calculation."""
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        return np.mean(sample_losses)

class Loss_CategoricalCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        # Clip data to prevent log(0) errors
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # Probabilities for target values
        if len(y_true.shape) == 1:  # Categorical labels
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:  # One-hot encoded labels
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)

        return -np.log(correct_confidences)


# --- 4. COMBINED ACTIVATION & LOSS (Optimized) ---
class Activation_Softmax_Loss_CategoricalCrossentropy:
    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss = Loss_CategoricalCrossentropy()

    def forward(self, inputs, y_true):
        self.activation.forward(inputs)
        self.output = self.activation.output
        return self.loss.calculate(self.output, y_true)

    def backward(self, dvalues, y_true):
        """Mathematically optimized backward pass for Softmax + CrossEntropy."""
        samples = len(dvalues)
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)

        self.dinputs = dvalues.copy()
        # Calculate gradient: (Predicted - Ground Truth)
        self.dinputs[range(samples), y_true] -= 1
        self.dinputs = self.dinputs / samples

# --- 5. OPTIMIZER ---
class Optimizer_SGD:
    def __init__(self, learning_rate=1.0):
        self.learning_rate = learning_rate

    def update_params(self, layer):
        layer.weights += -self.learning_rate * layer.dweights
        layer.biases  += -self.learning_rate * layer.dbiases


# --- 6. TRAINING LOOP ---

# Initialize data
X, y = spiral_data(samples=100, classes=3)

# Build Model
dense1 = Layer_Dense(2, 64)
activation1 = Activation_ReLU()
dense2 = Layer_Dense(64, 3)
loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()

# Initialize Optimizer
optimizer = Optimizer_SGD(learning_rate=0.8)



for epoch in range(10001):
    # Forward Pass
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    loss = loss_activation.forward(dense2.output, y)

    # Accuracy Calculation
    predictions = np.argmax(loss_activation.output, axis=1)
    accuracy = np.mean(predictions == y)

    # Backward Pass
    loss_activation.backward(loss_activation.output, y)
    dense2.backward(loss_activation.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

    # Optimization Step
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)

    # Logging
    if not epoch % 1000:
        print(f'epoch: {epoch}, acc: {accuracy:.3f}, loss: {loss:.3f}')


