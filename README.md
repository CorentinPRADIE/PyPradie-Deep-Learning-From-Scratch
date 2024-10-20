# PyPradie :A Deep Learning Library from Scratch

**PyPradie** is a deep learning library built entirely from scratch using only **NumPy**. This project is a demonstration of my deep understanding of core machine learning concepts. By re-implementing the fundamental components of deep learning frameworks without relying on PyTorch or TensorFlow, I've reinforced my grasp on the mechanics that power modern AI models.

## **Why Build PyPradie?**

The world of deep learning is dominated by powerful libraries that abstract away the complexities of neural networks. While these tools are invaluable for rapid development and deployment, they often obscure the underlying principles that are crucial for a deep comprehension of machine learning. **PyPradie** was created to:

- **Reinforce** my knowledge of essential concepts like automatic differentiation, backpropagation, and optimization algorithms.
- **Demonstrate** my ability to build and understand the key components of deep learning frameworks from the ground up.
- **Provide a foundation** for constructing more advanced models such as RNNs, LSTMs, and Transformers, which are integral to **large language models (LLMs)**.

---

## **Features and Functionality**

PyPradie aims to be a comprehensive deep learning library. Below is a list of features along with brief descriptions:

### **Core Components**

- **Tensor Operations**  
  Fundamental operations (addition, multiplication, division) with support for automatic differentiation.

- **Automatic Differentiation (Autograd)**  
  Enables gradient tracking and accumulation for backpropagation through computational graphs.

- **GPU Support**  
  Utilize GPUs for faster computation using CuPy for seamless CPU/GPU operations.  
  *(WIP)*

### **Neural Network Layers**

- **Linear Layer**  
  Implements fully connected layers for neural networks.

- **Activation Functions**
  - **ReLU (Rectified Linear Unit)**  
    Activation function that introduces non-linearity.
  - **Sigmoid**  
    Activation function for outputting probabilities.
  - **Tanh (Hyperbolic Tangent)**  
    Activation function for mapping inputs to values between -1 and 1.

- **Batch Normalization**  
  Normalizes the inputs of each layer to improve training speed and stability.  
  *(WIP)*

### **Recurrent Neural Networks**

- **RNN Cell**  
  Basic recurrent neural network cell for processing sequential data.

- **LSTM Cell (Long Short-Term Memory)**  
  Advanced RNN cell that can capture long-range dependencies in sequences.  
  *(WIP)*

### **Embedding Layers**

- **Embedding Layer**  
  Maps discrete input tokens to continuous vector representations, essential for NLP tasks.

### **Transformer Architecture**

- **Multi-Head Attention Mechanism**  
  Allows the model to focus on different parts of the input sequence.

- **Positional Encoding**  
  Incorporates information about the position of elements in the sequence.

- **Transformer Encoder and Decoder Blocks**  
  Foundation for state-of-the-art models in NLP like BERT and GPT.

### **Optimization Algorithms**

- **Stochastic Gradient Descent (SGD)**  
  Basic optimizer for updating model parameters based on gradients.

- **Adam Optimizer**  
  Adaptive learning rate method combining momentum and RMSProp.  
  *(WIP)*

### **Utilities and Tools**

- **DataLoader**  
  Handles batching and shuffling of data for efficient training.

- **No-Gradient Context Manager**  
  Disables gradient tracking during inference to improve performance.

---

## **How to Get Started**

### **1. Install Dependencies**

Ensure you have the necessary packages installed:

999bash
pip install -r requirements.txt
999

*Note: There is a known typo in the `requirements.txt` file; make sure it includes `torch` instead of `toch`.*

### **2. Install PyPradie**

Install the package in editable mode to allow for modifications:

999bash
pip install -e .
999

### **3. Run an Example**

To see PyPradie in action, you can run the MNIST comparison example:

999bash
python examples/Mnist_Pytorch_and_Pypradie_comparison.py
999

---

**PyPradie** is more than just a project; it's a reflection of my dedication to mastering the intricacies of deep learning.

---

Feel free to explore the code, run the examples, and reach out with any questions or feedback.

---

# Contact

Corentin Pradie

[LinkedIn](https://www.linkedin.com/in/corentin-pradie-861185271/)  
[E-Mail](corentin.prad@gmail.com)