import numpy as np  # Import NumPy for handling data types

# Expose core Tensor functionality
from .core.tensor import Tensor

# Expose layers and loss functions from the nn module
from .nn import Sequential, Linear, ReLU, CrossEntropyLoss

# Expose optimizer
from .optim import SGD

# Expose utility components
from .utils.data import TensorDataset, DataLoader

# Expose no_grad context manager
from .utils.no_grad import no_grad

# Add Tensor as "tensor" for seamless replacement with PyTorch
tensor = Tensor

# Correct the type definitions as data types, not methods
float32 = np.float32
long = np.int64

def argmax(tensor, dim=None):
    """Global argmax function to mimic torch.argmax."""
    return tensor.argmax(dim=dim)
