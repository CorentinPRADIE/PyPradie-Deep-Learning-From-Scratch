import numpy as np
import logging
from .operation import Operation, Add, Multiply, MatMul, Divide, Sum

class Tensor:
    _grad_enabled = True  # Global flag to track if gradients are enabled

    def __init__(self, data, dtype=np.float32, requires_grad=False):
        self.data = np.array(data, dtype=dtype)
        self.requires_grad = requires_grad
        self.grad = None
        self.operation = None
        self._backward_called = False

    def backward(self, grad_output=None):
        """Performs the backward pass and computes gradients."""
        
        if not Tensor._grad_enabled:
            return

        if grad_output is None:
            grad_output = Tensor(np.ones_like(self.data))
            

        # Ensure grad_output is a Tensor
        if not isinstance(grad_output, Tensor):
            grad_output = Tensor(grad_output)

        # Accumulate gradients if required
        if self.requires_grad:

            # Initialize or accumulate gradients
            if self.grad is None:
                self.grad = Tensor(grad_output.data.copy())
            else:
                self.grad.data += grad_output.data

        # Proceed with backpropagation if this tensor was produced by an operation
        if self.operation and not self._backward_called:
            self._backward_called = True  # Prevent multiple backward calls

            # Call the backward method of the associated operation
            grad_inputs = self.operation.backward(grad_output)

            # Propagate gradients to each input tensor
            for input_tensor, grad_input in zip(self.operation.inputs, grad_inputs):
                if grad_input is not None:
                    input_tensor.backward(grad_input)
                
    def zero_grad(self):
        """Zero the gradient."""
        self.grad = None
        self._backward_called = False  # Reset for future backward passes

    def view(self, *shape):
        """Reshape the tensor."""
        self.data = self.data.reshape(shape)
        return self

    def size(self, dim=None):
        """Return the size of the tensor or size along a specific dimension."""
        if dim is None:
            return self.data.shape
        return self.data.shape[dim]

    def detach(self):
        """Returns a tensor without gradient tracking."""
        return Tensor(self.data.copy(), requires_grad=False)

    def sum(self):
        result_data = np.sum(self.data)
        result = Tensor(result_data, requires_grad=self.requires_grad)
        if self.requires_grad:
            result.operation = Sum(self)
        return result

    def argmax(self, dim=None):
        """Return the indices of the maximum value along a given dimension."""
        return np.argmax(self.data, axis=dim)

    # Overload basic arithmetic operators

    def __add__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)

        # Perform addition and create the result tensor
        result = Tensor(self.data + other.data, requires_grad=self.requires_grad or other.requires_grad)
        
        # Associate the Add operation with the result tensor
        result.operation = Add(self, other)  # Use Add operation

        return result

    def __sub__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        result = Tensor(self.data - other.data, requires_grad=self.requires_grad or other.requires_grad)
        
        # Use a Subtract operation for clarity (could define a Subtract operation)
        result.operation = Add(self, Tensor(-1 * other.data))  # Subtraction as negative addition
        return result

    def __mul__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        result = Tensor(self.data * other.data, requires_grad=self.requires_grad or other.requires_grad)
        result.operation = Multiply(self, other)
        return result

    def __pow__(self, power):
        result = Tensor(self.data ** power, requires_grad=self.requires_grad)
        
        # For exponentiation, register the corresponding operation (you may need to implement it)
        if self.requires_grad:
            # Assuming power is a scalar
            grad = power * self.data ** (power - 1)
            result.operation = Multiply(Tensor(grad), self)
        return result

    def __matmul__(self, other):
        if isinstance(other, Tensor):
            result = np.matmul(self.data, other.data)
            output = Tensor(result, requires_grad=self.requires_grad or other.requires_grad)
            output.operation = MatMul(self, other)  # Use MatMul operation for matrix multiplication
            return output
        else:
            raise TypeError(f"Matrix multiplication is only supported between Tensor objects, got {type(other)}")

    def __repr__(self):
        return f"Tensor(data={self.data}, requires_grad={self.requires_grad})"

    # Convenience casting methods
    def float(self):
        return Tensor(self.data.astype(np.float32), requires_grad=self.requires_grad)

    def long(self):
        return Tensor(self.data.astype(np.int64), requires_grad=self.requires_grad)
    
    def argmax(self, dim=None):
        """Return the indices of the maximum values along a given dimension."""
        if dim is None:
            return Tensor(np.argmax(self.data))
        return Tensor(np.argmax(self.data, axis=dim))
    
    def view(self, *shape):
        """Reshape the tensor and preserve the computational graph."""
        reshaped_data = self.data.reshape(shape)  # Reshape the data
        result = Tensor(reshaped_data, requires_grad=self.requires_grad)
        result.operation = self.operation  # Maintain the same operation to preserve the computational graph
        return result

    
    def item(self):
        """Return the scalar value of a Tensor if it contains a single value."""
        if self.data.size == 1:
            return self.data.item()  # Use NumPy's item method to return a scalar
        else:
            raise ValueError("Tensor contains more than one element, cannot convert to scalar.")

    @property
    def data(self):
        return self._data  # Always return NumPy array to avoid memoryview issues

    @data.setter
    def data(self, value):
        self._data = value  # Ensure underlying data is stored as NumPy array
    
    def __truediv__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other, requires_grad=False)
        result = Tensor(self.data / other.data, requires_grad=self.requires_grad or other.requires_grad)
        if self.requires_grad or other.requires_grad:
            result.operation = Divide(self, other)
        return result

    # Optionally, implement __rtruediv__ if needed
    def __rtruediv__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other, requires_grad=False)
        result = Tensor(other.data / self.data, requires_grad=self.requires_grad or other.requires_grad)
        if self.requires_grad or other.requires_grad:
            result.operation = Divide(other, self)
        return result

    def __getitem__(self, index):
        """Allows indexing and slicing of the tensor like a NumPy array."""
        return Tensor(self.data[index], requires_grad=self.requires_grad)
    
    def __eq__(self, other):
        """Element-wise equality comparison between tensors."""
        if isinstance(other, Tensor):
            comparison = self.data == other.data
        else:
            comparison = self.data == other
        return Tensor(comparison, requires_grad=False)
    
    def tolist(self):
        """Convert tensor data to a Python list."""
        return self.data.tolist()

    def __ne__(self, other):
        """Element-wise inequality comparison between tensors."""
        if isinstance(other, Tensor):
            comparison = self.data != other.data
        else:
            comparison = self.data != other
        return Tensor(comparison, requires_grad=False)

    def nonzero(self, as_tuple=False):
        """Return indices of non-zero elements."""
        non_zero_indices = np.nonzero(self.data)
        if as_tuple:
            return tuple(Tensor(idx) for idx in non_zero_indices)
        return Tensor(non_zero_indices)
