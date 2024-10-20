from ..layers.base_layer import Layer
from ...core.tensor import Tensor
import numpy as np

class Tanh(Layer):
    def forward(self, x):
        """Forward pass for Tanh: tanh(x)"""
        tanh_output = np.tanh(x.data)
        result = Tensor(tanh_output, requires_grad=x.requires_grad)
        
        if x.requires_grad:
            result.operation = self
            self.inputs = [x]
        
        return result

    def backward(self, grad_output):
        """Backward pass for Tanh"""
        x = self.inputs[0]
        tanh_output = np.tanh(x.data)  # Compute tanh again
        grad_input = grad_output * (1 - tanh_output ** 2)  # Derivative of tanh
        return [grad_input]

    def parameters(self):
        """Tanh has no trainable parameters."""
        return []

    def zero_grad(self):
        """No gradients to zero for Tanh since there are no parameters."""
        pass
