from ...core.tensor import Tensor
from .base_layer import Layer
from .linear import Linear
from ..activations.sigmoid import Sigmoid
from ..activations.tanh import Tanh
import numpy as np

class RNNCell(Layer):
    
    def __init__(self, n_inputs, n_hidden, n_output, activation='sigmoid'):
        """
        Basic RNN Cell that processes one time step of the input sequence.
        """
        super().__init__()

        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_output = n_output
        
        # Select activation function
        if activation == 'sigmoid':
            self.activation = Sigmoid()
        elif activation == 'tanh':
            self.activation = Tanh()
        else:
            raise ValueError("Non-linearity not found")

        # Initialize linear layers for input-hidden, hidden-hidden, and hidden-output connections
        self.w_ih = Linear(n_inputs, n_hidden)  # Input to hidden
        self.w_hh = Linear(n_hidden, n_hidden)  # Hidden to hidden
        self.w_ho = Linear(n_hidden, n_output)  # Hidden to output
        
        # Collect parameters from the sub-layers
        self._parameters = []
        self._parameters += self.w_ih.parameters()  # Add parameters of w_ih
        self._parameters += self.w_hh.parameters()  # Add parameters of w_hh
        self._parameters += self.w_ho.parameters()  # Add parameters of w_ho
    
    def forward(self, input, hidden):
        """
        Forward pass for one time step of the RNN cell.
        
        Args:
        - input (Tensor): Input at the current time step.
        - hidden (Tensor): Hidden state from the previous time step.
        
        Returns:
        - output (Tensor): Output at the current time step.
        - new_hidden (Tensor): Updated hidden state.
        """
        print(f"Input shape: {input.data.shape}, Hidden shape: {hidden.data.shape}")
        
        # Compute new hidden state
        from_prev_hidden = self.w_hh.forward(hidden)
        combined = self.w_ih.forward(input) + from_prev_hidden
        new_hidden = self.activation.forward(combined)
        
        print(f"New hidden shape: {new_hidden.data.shape}")
        
        # Compute output
        output = self.w_ho.forward(new_hidden)
        
        print(f"Output shape: {output.data.shape}")
        
        return output, new_hidden

    
    def init_hidden(self, batch_size=1):
        """
        Initialize the hidden state for the RNN cell (zero vector).
        
        Args:
        - batch_size (int): Number of samples in a batch.
        
        Returns:
        - Tensor: Initialized hidden state of shape (batch_size, n_hidden).
        """
        # Return hidden state of shape (batch_size, n_hidden)
        return Tensor(np.zeros((batch_size, self.n_hidden)), requires_grad=True)

    def parameters(self):
        """Return all the trainable parameters of the RNNCell."""
        return self._parameters
