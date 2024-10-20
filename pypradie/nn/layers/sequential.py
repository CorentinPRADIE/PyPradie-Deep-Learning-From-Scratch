from .base_layer import Layer

class Sequential(Layer):  # Inherit from Layer base class
    def __init__(self, *layers):
        self.layers = layers

    def forward(self, x):
        """Forward pass through all layers."""
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def __call__(self, x):
        """Make Sequential object callable."""
        return self.forward(x)

    def parameters(self):
        """Return all trainable parameters from layers."""
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params

    def zero_grad(self):
        """Zero the gradients of all the layers' parameters."""
        for layer in self.layers:
            layer.zero_grad()
            
    def __getitem__(self, idx):
        return self.layers[idx]
