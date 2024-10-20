# pypradie/utils/data.py

import numpy as np
from pypradie.core.tensor import Tensor

class TensorDataset:
    """Dataset wrapping tensors. Each sample will be retrieved by indexing tensors."""
    def __init__(self, *tensors):
        assert all(isinstance(t, Tensor) for t in tensors), "All inputs must be Tensors"
        self.tensors = tensors

    def __getitem__(self, index):
        # Support integer, slice, list, or array indices
        return tuple(tensor.data[index] for tensor in self.tensors)

    def __len__(self):
        return len(self.tensors[0].data)

class DataLoader:
    """Data loader that batches and shuffles data."""
    def __init__(self, dataset, batch_size=1, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(self.dataset))

    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
        for i in range(0, len(self.indices), self.batch_size):
            batch_indices = self.indices[i:i + self.batch_size]
            # Retrieve batches directly using batch indices
            batch = self.dataset[batch_indices]
            # Unpack the features and labels
            batch_features, batch_labels = batch
            # Create Tensors without unnecessary copying
            batch_features_tensor = Tensor(batch_features, requires_grad=False)
            batch_labels_tensor = Tensor(batch_labels, requires_grad=False)
            yield batch_features_tensor, batch_labels_tensor

    def __len__(self):
        return len(self.dataset) // self.batch_size
