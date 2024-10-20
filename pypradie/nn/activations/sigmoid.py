from ..layers.base_layer import Layer
from ...core.tensor import Tensor
import numpy as np

class Sigmoid(Layer):
    def forward(self, x):
        """Forward pass for Sigmoid: 1 / (1 + exp(-x))"""
        sigmoid_output = 1 / (1 + np.exp(-x.data))
        result = Tensor(sigmoid_output, requires_grad=x.requires_grad)
        
        if x.requires_grad:
            result.operation = self
            self.inputs = [x]
        
        return result

    def backward(self, grad_output):
        """Backward pass for Sigmoid"""
        x = self.inputs[0]
        sigmoid_output = 1 / (1 + np.exp(-x.data))  # Compute sigmoid again
        grad_input = grad_output * sigmoid_output * (1 - sigmoid_output)  # Derivative of sigmoid
        return [grad_input]

    def parameters(self):
        """Sigmoid has no trainable parameters."""
        return []

    def zero_grad(self):
        """No gradients to zero for Sigmoid since there are no parameters."""
        pass
