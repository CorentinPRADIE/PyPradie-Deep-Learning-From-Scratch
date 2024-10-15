class Layer:
    def forward(self, *inputs):
        """Forward pass for the layer."""
        raise NotImplementedError

    def backward(self, grad_output):
        """Backward pass for the layer."""
        raise NotImplementedError

    def parameters(self):
        """Return a list of trainable parameters (if any)."""
        return []

    def zero_grad(self):
        """Zero the gradients of all trainable parameters (if any)."""
        for param in self.parameters():
            param.zero_grad()
