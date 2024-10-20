from ..core.tensor import Tensor

class no_grad:
    def __enter__(self):
        """Disable gradient computation."""
        Tensor._grad_enabled = False

    def __exit__(self, exc_type, exc_value, traceback):
        """Re-enable gradient computation."""
        Tensor._grad_enabled = True
