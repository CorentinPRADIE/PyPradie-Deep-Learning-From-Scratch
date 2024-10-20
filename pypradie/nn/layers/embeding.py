from ...core.tensor import Tensor
from .base_layer import Layer
import numpy as np

class Embedding(Layer):
    def __init__(self, vocab_size, dim):
        """
        Embedding layer that maps token indices to dense vectors.
        
        Args:
        - vocab_size (int): Number of unique tokens in the vocabulary.
        - dim (int): Dimensionality of the embeddings.
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.dim = dim
        
        # Initialize the embedding matrix with random values
        self.weight = Tensor((np.random.rand(vocab_size, dim) - 0.5) / dim, requires_grad=True)
    
    def forward(self, input):
        """
        Forward pass for the embedding layer.
        
        Args:
        - input (int): A token index (an integer).
        
        Returns:
        - Tensor: Embeddings corresponding to the input token index.
        """
        # Use input directly as the index
        return self.weight[input]

    def parameters(self):
        """Return the list of trainable parameters."""
        return [self.weight]
    
    def zero_grad(self):
        """Zero out the gradients for the embedding layer."""
        self.weight.zero_grad()
