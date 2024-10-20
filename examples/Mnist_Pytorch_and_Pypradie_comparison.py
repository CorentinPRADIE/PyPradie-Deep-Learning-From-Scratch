#!/usr/bin/env python
# coding: utf-8

# ## Introduction
# 
# The goal of building **PyPradie** is not to replace PyTorch, but to gain a deeper understanding of the core concepts behind a deep learning library. These core concepts include:
# 
# - **Autograd**: Automatic differentiation and backpropagation.
# - **Optimization**: Gradient-based optimization methods like SGD.
# - **Neural Network Layers**: Basic building blocks such as linear layers and activation functions.
# - **Tensors**: Efficient multidimensional arrays with support for mathematical operations.
# 
# Although PyPradie is not intended to replace PyTorch, it replicates the PyTorch API, allowing easy switching between the two libraries. This makes it interesting to compare their performance side by side to see if PyPradie can come close to PyTorch in terms of efficiency and accuracy.
# 
# In this notebook, we will perform a comparison between PyTorch and PyPradie on the **MNIST digit classification task** to see how PyPradie measures up.
# 

# ### Step 1: Loading the Library
# 
# The `get_framework` function dynamically loads either PyTorch or PyPradie based on the library name provided. This allows us to easily switch between the two frameworks in the same code.
# 
# - **PyTorch** is a well-known deep learning framework used widely in research and production.
# - **PyPradie** is a custom library built to mimic PyTorch's structure and functionality.
# 

# In[1]:


import time
import psutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, Markdown
np.random.seed(42)

# Function to dynamically load the correct library (PyTorch or PyPradie)
def get_framework(library_name):
    if library_name == "PyPradie":
        import pypradie as dl_framework
    else:
        import torch as dl_framework
    return dl_framework


# ### Step 2: Loading and Preprocessing the Data
# 
# The `load_data` function reads the MNIST dataset and prepares it for training. We normalize the pixel values and convert the features and labels into tensors. 
# 
# This function also uses `DataLoader` to create iterable batches for training and testing, which helps in efficiently handling large datasets.
# 
# - **Train Data**: 60,000 images and corresponding labels.
# - **Test Data**: 10,000 images and corresponding labels.
# 

# In[2]:


# Function to load and preprocess the MNIST data
def load_data(dl_framework, batch_size=64):
    train_df = pd.read_csv('../datasets/mnist/mnist_train.csv')
    test_df = pd.read_csv('../datasets/mnist/mnist_test.csv')
    
    # Extract labels and normalize features
    train_labels = train_df['label'].values
    train_features = train_df.drop('label', axis=1).values / 255.0  # Normalize pixel values
    test_labels = test_df['label'].values
    test_features = test_df.drop('label', axis=1).values / 255.0
    
    # Convert to framework-specific tensors
    tensor = dl_framework.tensor
    train_features_tensor = tensor(train_features, dtype=dl_framework.float32)
    train_labels_tensor = tensor(train_labels, dtype=dl_framework.long)
    test_features_tensor = tensor(test_features, dtype=dl_framework.float32)
    test_labels_tensor = tensor(test_labels, dtype=dl_framework.long)
    
    # Use DataLoader for batching and shuffling
    DataLoader = dl_framework.utils.data.DataLoader
    TensorDataset = dl_framework.utils.data.TensorDataset
    train_dataset = TensorDataset(train_features_tensor, train_labels_tensor)
    test_dataset = TensorDataset(test_features_tensor, test_labels_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader


# ### Step 3: Defining the Model
# 
# The `build_model` function defines a simple fully connected neural network (Multi-Layer Perceptron) with the following architecture:
# - Input layer: 784 input units (28 x 28 pixels).
# - Two hidden layers with 128 and 64 units, each followed by ReLU activation.
# - Output layer: 10 units for the 10 possible digit classes (0-9).
# 
# The loss function is **CrossEntropyLoss**, and the optimizer is **Stochastic Gradient Descent (SGD)**.
# 

# In[3]:


# Function to build the model, loss function, and optimizer
def build_model(dl_framework):
    nn = dl_framework.nn
    model = nn.Sequential(
        nn.Linear(28 * 28, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 10),
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = dl_framework.optim.SGD(model.parameters(), lr=0.01)
    return model, criterion, optimizer


# ### Step 4: Tracking Memory Usage
# 
# We track the memory usage during training using the `psutil` library. The `memory_usage_psutil` function returns the resident memory size (RSS) in megabytes, allowing us to measure the memory consumption of each framework.
# 

# In[4]:


# Memory usage tracking
def memory_usage_psutil():
    """Return the memory usage in MB."""
    process = psutil.Process()
    mem_info = process.memory_info()
    return mem_info.rss / 1024 ** 2  # Memory in MB


# ### Step 5: Training the Model
# 
# The `train_model` function trains the model over multiple epochs, tracking important metrics like training accuracy, loss, epoch time, and memory usage.
# 
# - For each epoch, we:
#   1. Forward pass: Compute the output of the model.
#   2. Backpropagation: Calculate the gradients.
#   3. Optimization: Update the model's parameters using the gradients.
# 
# We also measure how much memory the framework consumes during each epoch and record it.
# 

# In[5]:


# Training function with memory usage tracking
def train_model(dl_framework, model, criterion, optimizer, train_loader, epochs=5):
    tracking_data = {'train_accuracy': [], 'train_loss': [], 'epoch_time': [], 'memory_usage': []}
    tensor = dl_framework.tensor
    
    for epoch in range(epochs):
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        start_time = time.time()
        
        # Capture memory before training
        memory_before = memory_usage_psutil()
        
        for images, labels in train_loader:
            images = images.view(images.size(0), -1)  # Flatten images
            optimizer.zero_grad()  # Zero gradients
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            loss.backward()  # Backpropagation
            optimizer.step()  # Optimization
            
            running_loss += loss.item()
            
            predicted = dl_framework.argmax(outputs.detach(), dim=1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
        
        # Capture memory after training
        memory_after = memory_usage_psutil()
        epoch_memory_usage = memory_after - memory_before  # Measure memory used during training
        
        epoch_time = time.time() - start_time
        train_accuracy = 100 * correct_train / total_train
        tracking_data['train_accuracy'].append(train_accuracy)
        tracking_data['train_loss'].append(running_loss / len(train_loader))
        tracking_data['epoch_time'].append(epoch_time)
        tracking_data['memory_usage'].append(epoch_memory_usage)
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.6f}, Time: {epoch_time:.2f}s, Accuracy: {train_accuracy:.2f}%, Memory: {epoch_memory_usage:.2f} MB")
    
    return tracking_data


# ### Step 6: Testing the Model
# 
# The `test_model` function evaluates the trained model on the test dataset. It calculates the test accuracy by comparing the predicted labels with the true labels and disables gradient calculation to improve performance during inference.
# 

# In[6]:


# Testing function
def test_model(dl_framework, model, test_loader):
    correct_test = 0
    total_test = 0
    tensor = dl_framework.tensor
    
    with dl_framework.no_grad():  # Disable gradients for testing
        for images, labels in test_loader:
            images = images.view(images.size(0), -1)  # Flatten images
            outputs = model(images)
            predicted = dl_framework.argmax(outputs, dim=1)
            total_test += labels.size(0)
            correct_test += (predicted == labels).sum().item()
    
    test_accuracy = 100 * correct_test / total_test
    return test_accuracy


# ### Step 7: Displaying Results in a Table
# 
# The `display_results_as_table` function creates a markdown table that compares PyTorch and PyPradie across several key metrics:
# - **Train Accuracy**
# - **Test Accuracy**
# - **Total Training Time**
# - **Memory Usage**
# 

# In[7]:


# Function to create and display the result as a markdown table
def display_results_as_table(pytorch_data, pypradie_data, pytorch_test_acc, pypradie_test_acc):
    # Calculate average memory usage
    pytorch_memory = sum(pytorch_data['memory_usage']) / len(pytorch_data['memory_usage'])
    pypradie_memory = sum(pypradie_data['memory_usage']) / len(pypradie_data['memory_usage'])
    
    # Create a markdown table string with results
    table = f"""
| Metric            | PyTorch Value   | PyPradie Value   |
|-------------------|-----------------|------------------|
| Train Accuracy (%) | {pytorch_data['train_accuracy'][-1]:.2f} | {pypradie_data['train_accuracy'][-1]:.2f} |
| Test Accuracy (%)  | {pytorch_test_acc:.2f} | {pypradie_test_acc:.2f} |
| Total Training Time (s)  | {sum(pytorch_data['epoch_time']):.2f} | {sum(pypradie_data['epoch_time']):.2f} |
| Memory Usage (MB)  | {pytorch_memory:.2f} | {pypradie_memory:.2f} |
"""
    # Display the markdown table in a Jupyter Notebook
    display(Markdown(table))


# In[8]:


# Main function to run the experiment and compare PyTorch and PyPradie
def compare_libraries():
    libraries = ['PyTorch', 'PyPradie']
    comparison_data = {}

    for library in libraries:
        dl_framework = get_framework(library)
        
        # Load data and build the model
        train_loader, test_loader = load_data(dl_framework)
        model, criterion, optimizer = build_model(dl_framework)
        
        # Train the model
        print(f"\nTraining {library}...")
        train_data = train_model(dl_framework, model, criterion, optimizer, train_loader)
        
        # Test the model
        print(f"\nTesting {library}...")
        test_accuracy = test_model(dl_framework, model, test_loader)
        print(f"Test Accuracy for {library}: {test_accuracy:.2f}%")
        
        # Store data for comparison
        comparison_data[library] = {'train_data': train_data, 'test_accuracy': test_accuracy}
    
    # Extract results for both libraries
    pytorch_data = comparison_data['PyTorch']['train_data']
    pypradie_data = comparison_data['PyPradie']['train_data']
    pytorch_test_acc = comparison_data['PyTorch']['test_accuracy']
    pypradie_test_acc = comparison_data['PyPradie']['test_accuracy']
    
    # Display results as a markdown table
    display_results_as_table(pytorch_data, pypradie_data, pytorch_test_acc, pypradie_test_acc)

# Run the comparison
compare_libraries()


# ## Final Results and Interpretation
# 
# The results for both frameworks are presented below:
# 
# | Metric            | PyTorch Value   | PyPradie Value   |
# |-------------------|-----------------|------------------|
# | Train Accuracy (%) | 90.86           | 92.97            |
# | Test Accuracy (%)  | 91.49           | 93.36            |
# | Total Training Time (s)  | 11.44           | 6.43             |
# | Memory Usage (MB)  | 1.41            | 0.86             |
# 
# ### Interpretation:
# 
# - **Accuracy**: PyPradie slightly outperforms PyTorch in both training accuracy (92.97% vs. 90.86%) and test accuracy (93.36% vs. 91.49%). This demonstrates that PyPradie is capable of generalizing well on the MNIST task, despite being a custom framework for learning purposes.
# - **Training Time**: PyPradie completes training in nearly half the time of PyTorch (6.43s vs. 11.44s), showing a significant performance advantage for this specific task.
# - **Memory Usage**: PyPradie uses less memory (0.86 MB) than PyTorch (1.41 MB), highlighting its potential for efficiency.
# 
# While PyPradie is not designed to replace PyTorch, comparing the two helps us assess whether PyPradie’s performance is in the same ballpark. The results show that PyPradie is quite efficient for this task, and replicating the PyTorch API allows easy switching between the two libraries.
# 
