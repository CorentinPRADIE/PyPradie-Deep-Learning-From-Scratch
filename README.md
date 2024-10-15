# PyPradie: A Deep Learning Library from Scratch

**PyPradie** is a deep learning library built from scratch using only **NumPy** as the base. The aim of this project is to showcase my understanding of core deep learning concepts, such as automatic differentiation, optimization algorithms, and neural network layers, by implementing them step by step without relying on frameworks like PyTorch or TensorFlow.

## **Why Build PyPradie?**
The deep learning landscape is dominated by powerful libraries such as PyTorch and TensorFlow. These frameworks abstract away much of the complexity of machine learning, which is essential for production environments but makes it harder to grasp the underlying principles. **PyPradie** is a "from scratch" implementation designed to:
- **Reinforce** core concepts like autograd, backpropagation, and optimization.
- **Demonstrate** my ability to build and understand key components of deep learning libraries.
- **Provide a foundation** for more complex models, such as RNNs, LSTMs, and Transformers, which are widely used in **large language models (LLMs)**.

## **What's Already Implemented?**

The current version of PyPradie includes the following functionality:

- **Core Tensor Class**:
  - Implements fundamental operations (addition, multiplication, division, etc.) with support for **automatic differentiation**.
  - Supports gradient tracking and accumulation, enabling backpropagation through the computational graph.

- **Neural Network Layers**:
  - **Linear Layer**: Fully connected layer with forward and backward pass support.
  - **ReLU Activation Function**: Implements the ReLU non-linearity with autograd functionality.
  - **Sequential Layer**: Allows stacking of layers for feed-forward neural networks.
  
- **Loss Functions**:
  - **Cross-Entropy Loss**: For classification tasks, with support for softmax and gradient calculation.

- **Optimizers**:
  - **SGD (Stochastic Gradient Descent)**: Basic optimizer for updating model parameters based on gradients.

- **Utilities**:
  - **DataLoader**: Supports batching and shuffling of data, enabling batch gradient descent.
  - **No Gradient Context Manager**: Disables gradient tracking for inference.

## **Immediate Goals** (Roadmap)

The next steps in the development of PyPradie focus on enhancing performance and expanding model capabilities:

1. **GPU Support**:
   - Adding GPU support will significantly accelerate training, particularly for larger models. This will be achieved using **CuPy** to seamlessly switch between CPU and GPU execution.

2. **Adam Optimizer**:
   - Implementing the **Adam optimizer** will improve training convergence by combining momentum and adaptive learning rates, making it more efficient than SGD.

3. **Batch Gradient Descent**:
   - Extending batch gradient descent to support larger batch sizes and more complex training loops.

4. **Batch Normalization**:
   - Implementing **batch normalization** to stabilize training and speed up convergence, particularly for deeper networks.

## **Roadmap to RNNs, LSTMs, and Transformers**

Once the immediate goals are achieved, the project will focus on building more advanced models:

- **Recurrent Neural Networks (RNNs)**: Implementing basic RNNs for handling sequential data.
- **Long Short-Term Memory (LSTM)**: Building LSTMs to capture long-range dependencies in sequences, essential for tasks like language modeling and time series prediction.
- **Transformers**: Finally, building Transformer architectures to power models like **BERT** and **GPT**, which are widely used in large language models (LLMs).

## **How to Run the Example Notebooks**

To compare the performance of **PyPradie** and **PyTorch**, there is an example notebook located in `examples/Mnist_Pytorch_and_Pypradie_comparison.ipynb`.

### 1. Install Dependencies

First, ensure you have all the necessary dependencies by running:

000bash
pip install -r requirements.txt
000

The `requirements.txt` file contains the following packages:

000txt
torch
pandas
numpy
000

### 2. Install PyPradie in Editable Mode

To allow modifications to PyPradie without needing to reinstall it, install the package in **editable mode**:

000bash
pip install -e .
000

This will let you import the `pypradie` module directly into the notebooks.

### 3. Open the Example Notebook

To open the example notebook, use:

000bash
jupyter notebook
000

Navigate to the `examples/` folder and open `Mnist_Pytorch_and_Pypradie_comparison.ipynb`. You will be able to compare the performance of PyPradie with PyTorch in training on the MNIST dataset.

## **Why These Technologies Are Important**
The technologies implemented in PyPradie represent the **foundational concepts** behind modern machine learning libraries. By building them from scratch, I gain a deeper understanding of how these components work together to solve complex tasks like natural language processing and image recognition. This project highlights my ability to:
- **Implement complex models** from the ground up.
- **Understand and optimize** deep learning systems for practical tasks.
- **Adapt and extend** these implementations for state-of-the-art models like LSTMs and Transformers.

---

**PyPradie** is an ongoing project designed to not only build models but also to demonstrate my technical proficiency in deep learning and my readiness for machine learning engineering roles, particularly those focused on **large language models**.
