import numpy as np

class Operation:
    def __init__(self, *inputs):
        self.inputs = inputs

    def forward(self):
        raise NotImplementedError

    def backward(self, grad_output):
        raise NotImplementedError

def adjust_grad(grad, target_shape):
    grad_shape = grad.shape
    target_shape = tuple(target_shape)
    while len(grad_shape) > len(target_shape):
        grad = grad.sum(axis=0)
        grad_shape = grad.shape
    for axis, (g_dim, t_dim) in enumerate(zip(grad_shape, target_shape)):
        if t_dim == 1 and g_dim > 1:
            grad = grad.sum(axis=axis, keepdims=True)
            grad_shape = grad.shape
    grad = grad.reshape(target_shape)
    return grad

class Add(Operation):
    def forward(self, x, y):
        return x + y

    def backward(self, grad_output):
        from .tensor import Tensor
        x, y = self.inputs
        grad_x = grad_output
        grad_y = grad_output

        # Adjust gradients for broadcasting
        grad_x_data = adjust_grad(grad_x.data, x.data.shape)
        grad_y_data = adjust_grad(grad_y.data, y.data.shape)

        return Tensor(grad_x_data), Tensor(grad_y_data)

class Multiply(Operation):
    def forward(self, x, y):
        return x * y

    def backward(self, grad_output):
        from .tensor import Tensor
        x, y = self.inputs

        grad_x = grad_output.data * y.data
        grad_y = grad_output.data * x.data

        return Tensor(grad_x), Tensor(grad_y)
        
class MatMul(Operation):
    def forward(self, x, y):
        return np.matmul(x.data, y.data)

    def backward(self, grad_output):
        from .tensor import Tensor
        x, y = self.inputs

        # Compute gradients
        grad_x_data = np.matmul(grad_output.data, y.data.T)
        grad_y_data = np.matmul(x.data.T, grad_output.data)

        grad_x = Tensor(grad_x_data)
        grad_y = Tensor(grad_y_data)

        return grad_x, grad_y
        
class Divide(Operation):
    def forward(self, x, y):
        return x / y

    def backward(self, grad_output):
        from .tensor import Tensor
        x, y = self.inputs

        # Compute gradients w.r.t. inputs
        grad_x = grad_output.data / y.data
        grad_y = -grad_output.data * x.data / (y.data ** 2)

        return Tensor(grad_x), Tensor(grad_y)

class Sum(Operation):
    def __init__(self, input_tensor):
        super().__init__(input_tensor)

    def forward(self):
        x = self.inputs[0]
        return np.sum(x.data)

    def backward(self, grad_output):
        from .tensor import Tensor
        x = self.inputs[0]
        # grad_output is scalar, expand it to the shape of x.data
        grad_input = np.ones_like(x.data) * grad_output.data
        return (Tensor(grad_input),)

def sum_grad_over_broadcasted_dims(grad, target_shape):
    """
    Sum the gradient over dimensions where the target shape has size 1 or the dimension was added due to broadcasting.
    """
    grad_shape = grad.shape
    if len(grad_shape) > len(target_shape):
        target_shape = (1,) * (len(grad_shape) - len(target_shape)) + target_shape

    axes = tuple(
        i for i, (grad_dim, target_dim) in enumerate(zip(grad_shape, target_shape))
        if target_dim == 1 and grad_dim > 1
    )

    grad = grad.sum(axis=axes, keepdims=True)
    grad = grad.reshape(target_shape)

    return grad
