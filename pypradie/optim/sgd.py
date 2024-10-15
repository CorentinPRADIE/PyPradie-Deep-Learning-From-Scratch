import numpy as np

class SGD:
    def __init__(self, params, lr=0.01):
        self.params = params
        self.lr = lr

    def step(self):
        for param in self.params:
            if param.requires_grad and param.grad is not None:
                grad_array = np.array(param.grad.data)
                param_array = np.array(param.data)
                if grad_array.shape != param_array.shape:
                    raise ValueError(f"Shape mismatch: param.data shape {param_array.shape}, grad.data shape {grad_array.shape}")
                param.data -= self.lr * grad_array

    def zero_grad(self):
        """Set the gradients of all parameters to zero."""
        for param in self.params:
            if param.grad is not None:
                param.zero_grad()
