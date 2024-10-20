import numpy as np
from pypradie.core.tensor import Tensor

class CrossEntropyLoss:
    def forward(self, predictions, targets):
        """
        Compute the cross-entropy loss.
        Predictions should be the raw scores (logits) from the model, not probabilities.
        """
        epsilon = 1e-12  # Small value to prevent log(0)
        
        # Apply softmax to convert logits to probabilities
        exp_logits = np.exp(predictions.data - np.max(predictions.data, axis=1, keepdims=True))
        probabilities = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        
        self.probabilities = probabilities
        self.targets = targets.data.astype(int)  # Ensure integer type for indexing
        
        num_samples = predictions.data.shape[0]
        correct_log_probs = -np.log(probabilities[np.arange(num_samples), self.targets] + epsilon)
        loss_value = np.sum(correct_log_probs) / num_samples

        loss = Tensor(loss_value, requires_grad=True)
        loss.operation = self  # Associate this loss with the CrossEntropyLoss operation
        self.inputs = (predictions,)  # Store the inputs for backward pass
        return loss

    def backward(self, grad_output=None):
        """
        Compute the gradient of the cross-entropy loss with respect to predictions.
        """
        predictions = self.inputs[0]
        num_samples = predictions.data.shape[0]
        grad = self.probabilities.copy()
        grad[np.arange(num_samples), self.targets] -= 1
        grad /= num_samples

        return (Tensor(grad),)

    def __call__(self, predictions, targets):
        return self.forward(predictions, targets)
