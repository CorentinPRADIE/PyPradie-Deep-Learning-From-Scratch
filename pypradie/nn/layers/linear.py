import numpy as np
from pypradie.core.tensor import Tensor
from .base_layer import Layer

class Linear(Layer):
    def __init__(self, input_size, output_size):
        # Xavier initialization for weights
        limit = np.sqrt(6 / (input_size + output_size))
        self.weights = Tensor(
            np.random.uniform(-limit, limit, (input_size, output_size)),
            requires_grad=True
        )
        # Corrected bias shape
        self.bias = Tensor(np.zeros((output_size,)), requires_grad=True)

    def forward(self, x):
        """Forward pass: y = xW + b"""
        return x @ self.weights + self.bias

    def __call__(self, x):
        """Make the layer callable so it works like a PyTorch module"""
        return self.forward(x)

    def parameters(self):
        """Return the list of trainable parameters."""
        return [self.weights, self.bias]

    def zero_grad(self):
        """Zero out the gradients for all trainable parameters."""
        self.weights.zero_grad()
        self.bias.zero_grad()
