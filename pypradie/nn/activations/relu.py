from ..layers.base_layer import Layer
from ...core.tensor import Tensor

class ReLU(Layer):  # Inherit from Layer base class
    def forward(self, x):
        """Forward pass for ReLU: max(0, x)"""
        result = Tensor(x.data * (x.data > 0), requires_grad=x.requires_grad)
        
        if x.requires_grad:
            result.operation = self
            self.inputs = [x]
        
        return result

    def backward(self, grad_output):
        """Backward pass for ReLU"""
        x = self.inputs[0]
        grad_input = grad_output * (x.data > 0)  # Propagate gradient only where x > 0
        return [grad_input]  # Return as a list since backprop expects a list of gradients

    def parameters(self):
        """ReLU has no trainable parameters."""
        return []

    def zero_grad(self):
        """No gradients to zero for ReLU since there are no parameters."""
        pass
